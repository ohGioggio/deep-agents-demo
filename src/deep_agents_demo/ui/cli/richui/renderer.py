"""Draws run events with Rich: answers in boxes (streamed Markdown, footer in the bottom edge).

The renderer owns the console and one ``Live`` region for the length of a run. The region
shows ``self`` (see ``__rich_console__``), which stacks the parts that are still changing:
the "thinking" line and the answer box being written (its newest lines only, when it is
taller than the screen); Live re-renders it at most 8 times a second. Finished boxes are
printed above it, whole. Only the turn's own (root) run and room turns get boxes: subagents
and plan workers/reviewers are lines in the timeline. The timeline (tool calls, subagents, plan steps) is
drawn in the same region while unfinished; branches that settle are printed above it, once.

Locking: ``_lock`` guards the state only and is never held while calling Live or the console
(Live's refresh thread holds its own lock and calls ``__rich_console__``, which takes
``_lock``). Handlers queue the output work in ``_todo``; ``emit`` runs it afterwards, in order,
under ``_io`` (which ``__rich_console__`` never takes). ``notify`` takes no lock at all.
"""
import os
import re
import sys
import threading
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text

from deep_agents_demo.models.events import format_footer
from deep_agents_demo.ui.cli.richui.timeline import Timeline

REFRESHES_PER_SECOND = 8
TIMELINE_EVENTS = {"ToolCallEnd", "FileEvent", "StateSnapshot", "ApprovalRequested", "SubagentFinished"}
# Notices the plan step's line already shows; every other notice is printed.
SHOWN_BY_STEP_LINE = re.compile(r"^\[plan\] step \d+/\d+ (- .*|passed review)$")
MARGIN = 1   # rows kept free under the Live region


class _Lines:
    """Lines already rendered (a box's newest lines), drawn as they are."""

    def __init__(self, lines: list):
        self.lines = lines

    def __rich_console__(self, console, options):
        for line in self.lines:
            yield from line
            yield Segment.line()


class _Box:
    """One answer being written (or settled, waiting for its footer)."""

    def __init__(self, run_id: str, title: str):
        self.run_id = run_id
        self.title = title
        self.text = ""
        self.footer = ""
        self.style = "step" if title.startswith("room") else "agent"

    def panel(self, console=None, options=None, max_lines: int | None = None) -> Panel:
        """The box; with ``max_lines``, only its newest lines that fit (for the Live region,
        which can't scroll), under a muted "… N earlier lines"."""
        body = Markdown(self.text)
        if max_lines is not None and console is not None:
            inner = max(1, options.max_width - 4)   # borders and padding
            lines = console.render_lines(body, options.update(width=inner, height=None), pad=False)
            if len(lines) + 2 > max_lines:
                keep = max(1, max_lines - 3)        # borders and the marker
                hidden = len(lines) - keep
                body = Group(Text(f"… {hidden} earlier lines", style="muted"), _Lines(lines[-keep:]))
        return Panel(body, title=Text(self.title, style=self.style),
                     title_align="left",
                     subtitle=Text(self.footer, style="muted") if self.footer else None,
                     subtitle_align="right", border_style=self.style, box=box.ROUNDED,
                     padding=(0, 1))


class RichRenderer:
    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self._lock = threading.RLock()   # state only
        self._io = threading.RLock()     # serialises Live/console calls
        self._todo: list = []            # output queued by handlers: ("start",) ("stop",) ("print", renderable)
        self._live: Live | None = None
        self._subagents: set[str] = set()     # run ids whose text is not boxed
        self._speakers: dict[str, str] = {}   # run id -> "room / critic"
        self._box: _Box | None = None         # the box being written or settled
        self._settled = False                 # the box's message has ended
        self._waiting_since: float | None = None  # a run started, nothing has arrived yet
        self._timeline = Timeline(verbose)
        self._paused = False                  # a prompt is being read: Live stays down
        self._want_live = False               # Live was up (or asked for) when paused
        self._root: str | None = None         # the turn's own run (its answer is boxed)
        self._labels: dict[str, str] = {}     # run id -> label ("plan / reviewer")
        self._calls: set[str] = set()         # the root run's tool calls still running
        self._notices: list[str] = []         # from notify(): drawn in Live, printed by _run

    # ---- what the Live region shows ----
    def __rich_console__(self, console, options):
        with self._lock:
            parts = self._parts(console, options)   # builds the renderables from the current state
        yield Group(*parts)

    def _parts(self, console, options) -> list:
        """The Live region, within the screen's height: the tree gets what the box's
        smallest size leaves, the box what the tree leaves (at least 3 rows)."""
        height = console.size.height
        parts = [Text(message, style="approval") for message in list(self._notices)]
        if self._waiting_since is not None:
            seconds = int(time.monotonic() - self._waiting_since)
            parts.append(Text(f"thinking… {seconds} s", style="muted"))
        reserve = 5 if self._box is not None else 0   # a box: borders, marker, 2 lines
        tree = self._timeline.lines(max(3, height - 2 - len(parts) - reserve))
        budget = max(3, height - len(parts) - len(tree) - MARGIN)
        parts.extend(tree)
        if self._box is not None:
            parts.append(self._box.panel(console, options, budget))
        return parts

    # ---- events (plain-only and unknown events have no handler and are ignored) ----
    def emit(self, event) -> None:
        handler = getattr(self, f"_on_{type(event).__name__}", None)
        if handler or type(event).__name__ in TIMELINE_EVENTS:
            with self._io:
                with self._lock:
                    self._timeline.handle(event)
                    self._settle()   # the tree first, so it prints above the box the handler may flush
                    if handler:
                        handler(event)
                    todo, self._todo = self._todo, []
                self._run(todo)

    def _run(self, todo: list) -> None:
        while self._notices:   # notify()'s lines, printed for good above the live region
            self._print(Text(self._notices.pop(0), style="approval"))
        for op, *args in todo:
            if op == "start":
                if self._paused:
                    self._want_live = True   # drawn when resume() restarts it
                elif self._live is None:
                    self._live = Live(self, console=self.console,
                                      refresh_per_second=REFRESHES_PER_SECOND, transient=True)
                    self._live.start()
            elif op == "stop":
                self._want_live = False
                live, self._live = self._live, None
                if live is not None:
                    live.stop()
            else:
                self.console.print(args[0])

    def _print(self, renderable: Text) -> None:
        try:
            self.console.print(renderable)
        except Exception:   # never lose a notice to a console that failed
            self._write(renderable.plain)

    @staticmethod
    def _write(message: str) -> None:
        try:
            os.write(sys.stderr.fileno(), f"\n{message}\n".encode())
        except (OSError, ValueError, AttributeError):
            pass

    def _boxed(self, run_id: str, label: str = "") -> bool:
        """Is this run's answer a box? The root run's and a room turn's are."""
        label = label or self._labels.get(run_id, "")
        return run_id == self._root or run_id in self._speakers or label.startswith("room /")

    def _on_RunStarted(self, event) -> None:
        if event.run_id in self._subagents:
            return
        self._labels[event.run_id] = event.label
        if self._root is None and event.parent_id is None:
            self._root = event.run_id
        self._start_live()
        if self._boxed(event.run_id) and self._waiting_since is None and self._box is None:
            self._waiting_since = time.monotonic()

    def _on_NoticeEvent(self, event) -> None:
        if not SHOWN_BY_STEP_LINE.match(event.text):
            self._todo.append(("print", Text(event.text, style="muted")))

    def _on_SubagentStarted(self, event) -> None:
        self._subagents.add(event.run_id)

    def _on_StepStarted(self, event) -> None:
        if event.kind == "room":
            self._flush()
            self._speakers[event.run_id] = f"room / {event.agent}"
            self._start_live()
            self._waiting_since = time.monotonic()

    def _on_StepFinished(self, event) -> None:
        if event.kind != "room" or event.run_id not in self._speakers:
            return  # a plan step, or a finish without a start
        self._finish(event.run_id, format_footer(time.time(), event.input_tokens,
                                                 event.output_tokens, event.duration))

    def _on_ToolCallStart(self, event) -> None:
        if event.run_id not in self._subagents:
            self._waiting_since = None
        if event.run_id == self._root:
            self._calls.add(event.tool_call_id)

    def _on_ToolCallEnd(self, event) -> None:
        """The root run's last running call ended: the model is thinking again."""
        if event.run_id != self._root:
            return
        self._calls.discard(event.tool_call_id)
        if not self._calls and self._box is None:
            self._waiting_since = time.monotonic()
            self._start_live()

    def _on_TextMessageStart(self, event) -> None:
        if event.run_id in self._subagents:
            return
        if self._root is None and not event.label.startswith("plan /"):
            self._root = event.run_id   # a run that never announced itself
        if not self._boxed(event.run_id, event.label):
            return   # a plan worker or reviewer: its step line shows the outcome
        self._flush()
        self._start_live()
        self._waiting_since = None
        self._box = _Box(event.run_id, self._speakers.get(event.run_id, event.label))
        self._settled = False

    def _on_TextMessageContent(self, event) -> None:
        if self._box is not None and not self._settled and event.run_id == self._box.run_id:
            self._box.text += event.delta

    def _on_TextMessageEnd(self, event) -> None:
        if event.run_id in self._subagents or self._box is None or self._box.run_id != event.run_id:
            return
        if event.has_tool_calls:  # a dim line in the timeline (see Timeline), not a box
            self._box = None
            return
        self._box.text = event.text or self._box.text
        self._settled = True

    def _on_RunFinished(self, event) -> None:
        if event.run_id in self._subagents:
            return
        self._end_root(event.run_id)
        self._finish(event.run_id, format_footer(event.ended_at, event.input_tokens,
                                                 event.output_tokens, event.duration))

    def _on_RunError(self, event) -> None:
        if event.run_id not in self._subagents:
            self._end_root(event.run_id)
            self._finish(event.run_id, "", event.message)

    def _end_root(self, run_id: str) -> None:
        self._labels.pop(run_id, None)
        if run_id == self._root:
            self._root = None
            self._calls.clear()

    # ---- helpers (state only; they queue output in _todo) ----
    def _start_live(self) -> None:
        self._todo.append(("start",))

    def _settle(self, force: bool = False) -> None:
        """Queue the timeline branches that have settled to be printed (and keep Live up
        while some are still running)."""
        lines = self._timeline.take_settled(force)
        if lines:   # in a Group: console.print would join bare Texts and drop their no_wrap
            self._todo.append(("print", Group(*lines)))
        if not force and self._timeline.active():
            self._start_live()

    def _flush(self) -> None:
        """Queue the box to be printed for good (with its footer, if it got one)."""
        if self._box is not None and self._box.text.strip():
            self._todo.append(("print", self._box.panel()))
        self._box = None
        self._settled = False

    def _finish(self, run_id: str, footer: str, error: str = "") -> None:
        if self._box is not None and self._box.run_id == run_id:
            self._box.footer = footer if self._settled else ""
        self._waiting_since = None
        if not self._timeline.active():   # a plan worker ending leaves the tree running
            self._todo.append(("stop",))
        self._flush()
        if error:
            self._todo.append(("print", Text(f"✗ {error}", style="error")))

    def pause(self) -> None:
        """Take Live down before a prompt is read; the tree stays in the state and is
        drawn again by ``resume``. Events arriving meanwhile only update the state."""
        with self._io:
            with self._lock:
                self._paused = True
            wanted = self._live is not None or self._want_live
            self._run([("stop",)])
            self._want_live = wanted

    def resume(self) -> None:
        with self._io:
            with self._lock:
                self._paused = False
                restart = self._want_live
            self._want_live = False
            if restart:
                self._run([("start",)])

    def notify(self, message: str) -> None:
        """A line for the user (the Ctrl+C notice). Runs in a signal handler, so it takes no
        lock at all: while Live draws, the line joins the live region at its next refresh and
        is printed above it by the next output; with nothing drawing it goes straight to
        stderr."""
        if self._live is None:
            self._write(message)
        else:
            self._notices.append(message)   # list.append is atomic

    def close(self) -> None:
        with self._io:
            with self._lock:
                self._waiting_since = None
                self._root = None
                self._calls.clear()
                self._labels.clear()
                self._todo.append(("stop",))
                self._settle(force=True)
                self._flush()
                todo, self._todo = self._todo, []
            self._run(todo)

"""Draws run events with Rich: answers in boxes (streamed Markdown, footer in the bottom edge).

The renderer owns the console and one ``Live`` region for the length of a run. The region
shows ``self`` (see ``__rich_console__``), which stacks the parts that are still changing:
the "thinking" line and the answer box being written; Live re-renders it at most 8 times
a second. Finished boxes are printed above it. The timeline (tool calls, subagents, plan steps) is
drawn in the same region while unfinished; branches that settle are printed above it, once.

Locking: ``_lock`` guards the state only and is never held while calling Live or the console
(Live's refresh thread holds its own lock and calls ``__rich_console__``, which takes
``_lock``). Handlers queue the output work in ``_todo``; ``emit`` runs it afterwards, in order,
under ``_io`` (which ``__rich_console__`` never takes).
"""
import threading
import time

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from deep_agents_demo.models.events import format_footer
from deep_agents_demo.ui.cli.richui.timeline import Timeline

REFRESHES_PER_SECOND = 8
TIMELINE_EVENTS = {"ToolCallEnd", "FileEvent", "StateSnapshot", "ApprovalRequested", "SubagentFinished"}


class _Box:
    """One answer being written (or settled, waiting for its footer)."""

    def __init__(self, run_id: str, title: str):
        self.run_id = run_id
        self.title = title
        self.text = ""
        self.footer = ""
        self.style = "step" if title.startswith("room") else "agent"

    def panel(self) -> Panel:
        return Panel(Markdown(self.text), title=Text(self.title, style=self.style),
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

    # ---- what the Live region shows ----
    def __rich_console__(self, console, options):
        with self._lock:
            parts = self._parts()   # builds the renderables from the current state
        yield Group(*parts)

    def _parts(self) -> list:
        parts = []
        if self._waiting_since is not None:
            seconds = int(time.monotonic() - self._waiting_since)
            parts.append(Text(f"thinking… {seconds} s", style="muted"))
        parts.extend(self._timeline.lines())
        if self._box is not None:
            parts.append(self._box.panel())
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
        for op, *args in todo:
            if op == "start":
                if self._live is None:
                    self._live = Live(self, console=self.console,
                                      refresh_per_second=REFRESHES_PER_SECOND, transient=True)
                    self._live.start()
            elif op == "stop":
                live, self._live = self._live, None
                if live is not None:
                    live.stop()
            else:
                self.console.print(args[0])

    def _on_RunStarted(self, event) -> None:
        if event.run_id in self._subagents:
            return
        self._start_live()
        if self._waiting_since is None and self._box is None:
            self._waiting_since = time.monotonic()

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

    def _on_TextMessageStart(self, event) -> None:
        if event.run_id in self._subagents:
            return
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
        self._finish(event.run_id, format_footer(event.ended_at, event.input_tokens,
                                                 event.output_tokens, event.duration))

    def _on_RunError(self, event) -> None:
        if event.run_id not in self._subagents:
            self._finish(event.run_id, "", event.message)

    # ---- helpers (state only; they queue output in _todo) ----
    def _start_live(self) -> None:
        self._todo.append(("start",))

    def _settle(self, force: bool = False) -> None:
        """Queue the timeline branches that have settled to be printed (and keep Live up
        while some are still running)."""
        for line in self._timeline.take_settled(force):
            self._todo.append(("print", line))
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
        self._todo.append(("stop",))
        self._flush()
        if error:
            self._todo.append(("print", Text(f"✗ {error}", style="error")))

    def close(self) -> None:
        with self._io:
            with self._lock:
                self._waiting_since = None
                self._todo.append(("stop",))
                self._settle(force=True)
                self._flush()
                todo, self._todo = self._todo, []
            self._run(todo)

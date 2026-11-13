"""The timeline: the tool calls, subagents, plan steps and files of one turn, as a tree.

``Timeline.handle(event)`` updates the tree; ``lines()`` draws what is still unfinished (for
the Live region); ``take_settled()`` hands back the top-level branches that have settled since
the last call, drawn once, to be printed above the Live region. Not thread-safe by itself:
the renderer calls it under its state lock.
"""
import json
import time
from pathlib import Path

from rich.style import Style
from rich.text import Text

from deep_agents_demo.models.events import (
    ApprovalRequested, FileEvent, RunError, RunFinished, RunStarted, StateSnapshot, StepFinished,
    StepStarted, SubagentFinished, SubagentStarted, TextMessageEnd, ToolCallEnd, ToolCallStart,
    format_footer,
)

ARG_WIDTH = 48
TEXT_WIDTH = 100
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MARKS = {"ok": ("✓", "ok"), "error": ("✗", "error"), "rejected": ("✗", "error"),
         "cut": ("✗", "error"), "stopped": ("✗", "error")}
TODO_MARKS = {"completed": ("✓", "ok"), "in_progress": ("▸", "approval"), "pending": ("○", "muted")}


def _cut(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _seconds(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 10 else f"{int(seconds)} s"


def _size(count: int) -> str:
    return f"{count / 1024:.1f} KB" if count >= 1024 else f"{count} B"


def _spinner() -> str:
    return SPINNER[int(time.monotonic() * 10) % len(SPINNER)]


class _Node:
    def __init__(self, kind: str, parent: "_Node | None", name: str = "", call_id: str = "",
                 args: dict | None = None):
        self.kind = kind            # "tool", "sub", "step", "text"
        self.parent = parent
        self.name = name
        self.call_id = call_id
        self.args = args or {}
        self.children: list[_Node] = []
        self.started = time.monotonic()
        self.done = kind == "text"
        self.status = ""            # "ok", "error", "rejected", "cut", "stopped"
        self.summary = ""
        self.result = ""
        self.duration = 0.0
        self.approval = False
        self.awaiting_file = False
        self.file: FileEvent | None = None
        self.answer = ""
        self.footer = ""
        self.number = 0
        self.total = 0
        self.attempts = 0
        self.merged = False         # a subagent that is also the `task` tool call

    def settled(self) -> bool:
        return self.done and not self.awaiting_file and all(c.settled() for c in self.children)


class Timeline:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._root = _Node("root", None)
        self._runs: dict[str, _Node] = {}      # run id -> node its events belong under
        self._roots: set[str] = set()          # ids of the turn's own runs
        self._calls: dict[str, _Node] = {}     # tool call id -> node
        self._pending_file: FileEvent | None = None
        self._todos: tuple = ()
        self._ended = False

    # ---- events ----
    def handle(self, event) -> None:
        if not isinstance(event, FileEvent):
            self._release_files()
        handler = getattr(self, f"_on_{type(event).__name__}", None)
        if handler:
            handler(event)

    def _parent(self, run_id: str) -> _Node:
        return self._runs.get(run_id, self._root)

    def _running_step(self) -> _Node | None:
        for node in reversed(self._root.children):
            if node.kind == "step" and not node.done:
                return node
            for child in reversed(node.children):
                if child.kind == "step" and not child.done:
                    return child
        return None

    def _release_files(self) -> None:
        for node in self._calls.values():
            node.awaiting_file = False

    def _on_RunStarted(self, event: RunStarted) -> None:
        if event.run_id in self._runs:
            return
        if event.parent_id and event.parent_id in self._runs:
            self._runs[event.run_id] = self._runs[event.parent_id]
            return
        step = self._running_step()
        if step is not None:
            self._runs[event.run_id] = step
        else:
            self._runs[event.run_id] = self._root
            self._roots.add(event.run_id)

    def _on_RunFinished(self, event: RunFinished) -> None:
        node = self._runs.get(event.run_id)
        if node is not None and node.kind == "sub" and self._sub_run(node) == event.run_id:
            node.footer = format_footer(event.ended_at, event.input_tokens, event.output_tokens,
                                        event.duration)
        self._end_turn(event.run_id)

    def _on_RunError(self, event: RunError) -> None:
        self._end_turn(event.run_id)

    def _sub_run(self, node: _Node) -> str:
        return next((run for run, n in self._runs.items() if n is node), "")

    def _end_turn(self, run_id: str) -> None:
        if run_id in self._roots and self._running_step() is None:
            self._ended = True

    def _on_SubagentStarted(self, event: SubagentStarted) -> None:
        node = self._calls.get(event.tool_call_id)
        if node is None:
            node = _Node("sub", self._parent(event.parent_run_id), event.name, event.tool_call_id)
            node.parent.children.append(node)
            if event.tool_call_id:
                self._calls[event.tool_call_id] = node
        else:
            node.kind = "sub"
            node.merged = True
        node.name = event.name
        self._runs[event.run_id] = node

    def _on_SubagentFinished(self, event: SubagentFinished) -> None:
        node = self._runs.get(event.run_id)
        if node is None or node.kind != "sub":
            return
        node.status = event.status
        if not node.merged:
            node.done = True
            node.duration = time.monotonic() - node.started

    def _on_StepStarted(self, event: StepStarted) -> None:
        if event.kind != "plan":
            return
        parent = self._root
        for node in reversed(self._root.children):
            if node.kind == "tool" and node.name == "run_plan" and not node.done:
                parent = node
                break
        node = _Node("step", parent, event.agent)
        node.number, node.total = event.number, event.total
        parent.children.append(node)
        self._runs[event.run_id] = node

    def _on_StepFinished(self, event: StepFinished) -> None:
        if event.kind != "plan":
            return
        node = self._runs.get(event.run_id)
        if node is None or node.kind != "step":   # a stopped plan's steps never started
            node = _Node("step", self._root, "")
            node.number = event.number
            self._root.children.append(node)
        node.done = True
        node.status, node.attempts, node.duration = event.status, event.attempts, event.duration

    def _on_ToolCallStart(self, event: ToolCallStart) -> None:
        if event.tool_call_id in self._calls:
            return
        node = _Node("tool", self._parent(event.run_id), event.name, event.tool_call_id, event.args)
        node.parent.children.append(node)
        if event.tool_call_id:
            self._calls[event.tool_call_id] = node

    def _on_ToolCallEnd(self, event: ToolCallEnd) -> None:
        node = self._calls.get(event.tool_call_id)
        if node is None:
            node = _Node("tool", self._parent(event.run_id), event.name, event.tool_call_id)
            node.parent.children.append(node)
        node.done, node.approval = True, False
        node.status, node.summary = event.status, event.summary
        node.duration, node.result = event.duration, event.result
        if event.name == "send_file" and event.status == "ok":
            if self._pending_file is not None:
                node.file, self._pending_file = self._pending_file, None
            else:
                node.awaiting_file = True

    def _on_ApprovalRequested(self, event: ApprovalRequested) -> None:
        node = self._calls.get(event.tool_call_id)
        if node is None:
            siblings = self._parent(event.run_id).children
            node = next((n for n in reversed(siblings)
                         if n.kind == "tool" and not n.done and n.name == event.name), None)
        if node is not None:
            node.approval = True

    def _on_FileEvent(self, event: FileEvent) -> None:
        node = next((n for n in reversed(list(self._calls.values()))
                     if n.name == "send_file" and n.awaiting_file), None)
        if node is None:
            self._pending_file = event
        else:
            node.file, node.awaiting_file = event, False

    def _on_TextMessageEnd(self, event: TextMessageEnd) -> None:
        parent = self._parent(event.run_id)
        if event.has_tool_calls:
            if event.text.strip():
                node = _Node("text", parent, event.text)
                parent.children.append(node)
        elif parent.kind == "sub":
            parent.answer = event.text

    def _on_StateSnapshot(self, event: StateSnapshot) -> None:
        self._todos = tuple(event.todos)

    # ---- output ----
    def active(self) -> bool:
        return bool(self._root.children) or bool(self._todos)

    def take_settled(self, force: bool = False) -> list:
        """Top-level branches that settled since the last call, drawn for good. At the end of
        the turn (or ``force``) everything left goes, unfinished calls marked as cut."""
        out: list = []
        everything = force or self._ended
        if everything:
            self._finish_all(self._root)
        for node in list(self._root.children):
            if node.settled():
                out.extend(self._branch(node, "", settled=True))
                self._root.children.remove(node)
        if everything:
            out.extend(self._todo_lines())
            self._todos = ()
            self._ended = False
            self._runs = {}
            self._roots = set()
            self._calls = {}
            self._pending_file = None
        return out

    def _finish_all(self, node: _Node) -> None:
        for child in node.children:
            self._finish_all(child)
        if node.kind != "root" and not node.done:
            node.done, node.status = True, node.status or "cut"
        node.awaiting_file = False

    def lines(self, limit: int | None = None) -> list:
        """The unfinished part of the tree, for the Live region. With ``limit``, only the
        last lines that fit (the rest is summed up in one muted line on top)."""
        out: list = []
        for node in self._root.children:
            if not node.settled():
                out.extend(self._branch(node, "", settled=False))
        out += self._todo_lines()
        if limit is not None and len(out) > limit:
            hidden = len(out) - (limit - 1)
            out = [Text(f"… {hidden} earlier lines", style="muted")] + out[hidden:]
        return out

    def _todo_lines(self) -> list:
        out = []
        for todo in self._todos:
            mark, style = TODO_MARKS.get(todo.get("status"), TODO_MARKS["pending"])
            line = Text(f"{mark} ", style=style)
            line.append(str(todo.get("content", "")))
            out.append(line)
        return out

    def _branch(self, node: _Node, prefix: str, settled: bool, last: bool = False) -> list:
        top = node.parent is self._root
        connector = "└─ " if last and not top else "├─ "
        head = Text(prefix + connector, style="muted")
        self._head(node, head)
        lines = [head]
        child_prefix = prefix + ("   " if last and not top else "│  ")
        details = self._details(node)
        kids = [c for c in node.children]
        answer = self._answer(node)
        total = len(kids) + (1 if answer is not None else 0)
        for text in details:
            lines.append(Text(child_prefix, style="muted") + text)
        for index, child in enumerate(kids):
            lines.extend(self._branch(child, child_prefix, settled, last=index == total - 1))
        if answer is not None:
            line = Text(child_prefix + "└─ ", style="muted")
            line.append_text(answer)
            lines.append(line)
        return lines

    def _details(self, node: _Node) -> list:
        out: list = []
        if node.kind == "tool" and self.verbose:
            if node.args:
                for row in json.dumps(node.args, ensure_ascii=False, indent=2, default=str).splitlines():
                    out.append(Text(row, style="muted"))
            if node.result:
                for row in node.result.splitlines():
                    out.append(Text(row, style="muted"))
        if node.file is not None:
            event = node.file
            line = Text(f"[file] {event.name}  v{event.version}  {_size(event.bytes)}  →  ", style="file")
            try:
                uri = Path(event.path).resolve().as_uri()
            except (OSError, ValueError):
                uri = ""
            start = len(line)
            line.append(event.path, style="file")
            if uri:   # a span on top of the theme's "file" style, so the colour stays the theme's
                line.stylize(Style(link=uri, underline=True), start, len(line))
            out.append(line)
        return out

    def _answer(self, node: _Node) -> Text | None:
        if node.kind != "sub" or not node.answer.strip():
            return None
        first = node.answer.strip().splitlines()[0]
        line = Text(f"{node.name}:", style="subagent")
        line.append(" " + first)
        if node.footer:
            line.append("  " + node.footer, style="muted")
        return line

    def _head(self, node: _Node, line: Text) -> None:
        if node.kind == "text":
            line.append(_cut(node.name, TEXT_WIDTH), style="muted")
        elif node.kind == "tool":
            self._tool_head(node, line)
        elif node.kind == "sub":
            line.append("task", style="tool")
            line.append(" → ")
            line.append(node.name, style="subagent")
            self._tail(node, line)
        else:
            self._step_head(node, line)

    def _tool_head(self, node: _Node, line: Text) -> None:
        line.append(node.name, style="tool")
        if node.args:
            line.append("  " + _cut(next(iter(node.args.values())), ARG_WIDTH))
        self._tail(node, line)

    def _tail(self, node: _Node, line: Text) -> None:
        if node.approval:
            line.append("  ⏸ needs your approval", style="approval")
        elif not node.done:
            line.append(f"  {_spinner()} {_seconds(time.monotonic() - node.started)}", style="approval")
        else:
            mark, style = MARKS.get(node.status, ("✓", "ok"))
            line.append(f"  {mark}", style=style)
            note = node.summary or ("interrupted" if node.status == "cut" else
                                    "rejected" if node.status == "rejected" else "")
            parts = [note] if note else []
            if node.duration:
                parts.append(_seconds(node.duration))
            if parts:
                line.append(" " + " · ".join(parts), style="muted" if node.status == "ok" else style)

    def _step_head(self, node: _Node, line: Text) -> None:
        line.append(f"step {node.number}/{node.total}" if node.total else f"step {node.number}",
                    style="step")
        if node.name:
            line.append(f"  {node.name}")
        if not node.done:
            line.append(f"  {_spinner()} working… {int(time.monotonic() - node.started)} s",
                        style="approval")
        elif node.status == "ok":
            tries = f"{node.attempts} {'try' if node.attempts == 1 else 'tries'}"
            line.append("  ✓ passed review", style="ok")
            line.append(f" · {tries} · {_seconds(node.duration)}", style="muted")
        else:
            line.append(f"  ✗ {node.status}", style="error")
            if node.attempts:
                line.append(f" · {node.attempts} {'try' if node.attempts == 1 else 'tries'}", style="muted")

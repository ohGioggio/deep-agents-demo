"""How the CLI shows things: the engine's events, chat tables, transcripts."""
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from deep_agents_demo.models.events import (
    FileEvent,
    FinalAnswerEvent,
    MessageEvent,
    NoticeEvent,
    TodosEvent,
)
from deep_agents_demo.persistence.chats import ChatInfo, shorten


def format_table(chats: list[ChatInfo], current: str | None = None, width: int = 48) -> str:
    if not chats:
        return "No chats yet. Start one with: uv run deep-agents-demo --thread <name>"
    rows = [("", "CHAT", "TURNS", "LAST USED", "LAST MESSAGE")]
    for c in chats:
        when = c.updated[:16].replace("T", " ") if c.updated else "-"
        rows.append(("*" if c.name == current else "", c.name, str(c.turns), when, shorten(c.last, width)))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    return "\n".join("  ".join([r[0].ljust(widths[0]), r[1].ljust(widths[1]), r[2].rjust(widths[2]),
                                r[3].ljust(widths[3]), r[4]]).rstrip() for r in rows)


def format_transcript(lines: list[tuple[str, str]], width: int = 400) -> str:
    if not lines:
        return "(empty chat)"
    return "\n".join(f"{who}: {shorten(text, width)}" for who, text in lines)


def size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / 1024 / 1024:.1f} MB"


def shown_path(path: str) -> str:
    """Relative to the current folder when it is under it (with "/"), else as given."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path


def format_files(files, folder=None) -> str:
    """The files delivered in a chat (``OutputFile``s), as a table."""
    if not files:
        return "No files delivered in this chat yet."
    rows = [("FILE", "VERSION", "SIZE", "SAVED", "CAPTION")]
    rows += [(f.name, f"v{f.version}", size(f.bytes), f.saved_at[:16].replace("T", " "), shorten(f.caption, 48))
             for f in files]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    table = "\n".join("  ".join([*(r[i].ljust(widths[i]) for i in range(4)), r[4]]).rstrip() for r in rows)
    return table + (f"\nIn {shown_path(str(folder))}" if folder is not None else "")


class CliRenderer:
    """An ``EventSink`` that prints events as the CLI always has. ``write`` receives
    each printed line (default ``print``); a test passes a list's ``append``."""

    def __init__(self, write: Callable[[str], None] | None = None):
        # print is looked up at each call, not bound once (stdout can be redirected).
        self.write = write or (lambda line: print(line))

    def emit(self, event) -> None:
        if isinstance(event, MessageEvent):
            self.write(f"\n[{event.label} / {event.node}]")
            self.write(event.pretty)
        elif isinstance(event, FileEvent):
            self.write(f"[file] {event.name}  v{event.version}  {size(event.bytes)}  ->  {shown_path(event.path)}")
        elif isinstance(event, TodosEvent):
            self.write("[todos] " + json.dumps(list(event.todos), ensure_ascii=False))
        elif isinstance(event, (NoticeEvent, FinalAnswerEvent)):
            self.write(event.text)


def run_when(started_at: float) -> str:
    return datetime.fromtimestamp(started_at).strftime("%Y-%m-%d %H:%M")


def calls_text(count: int) -> str:
    """"1 call", or "N calls" for any other count (zero included)."""
    return f"{count} call" if count == 1 else f"{count} calls"


def run_totals_line(totals: dict) -> str:
    return (f"Today: {calls_text(totals['model_calls'])} · {totals['input_tokens']:,} in / "
            f"{totals['output_tokens']:,} out")


def run_part_line(part) -> str:
    return (f"{part.kind}: {shorten(part.label, 60)}  ·  {calls_text(part.model_calls)} · {part.input_tokens:,} in / "
            f"{part.output_tokens:,} out · {part.seconds:.1f} s · {part.status}")


def run_part_rows(run) -> list[tuple[int, object]]:
    """The run's parts in tree order as (depth, part). A part whose parent is missing is a root."""
    known = {part.part for part in run.parts}
    children: dict = {}
    for part in run.parts:
        children.setdefault(part.parent if part.parent in known else None, []).append(part)
    rows: list[tuple[int, object]] = []

    def walk(parent, depth):
        for part in children.get(parent, []):
            rows.append((depth, part))
            walk(part.part, depth + 1)

    walk(None, 0)
    return rows


def format_runs(runs, totals: dict) -> str:
    """The user's recent runs (``Run``s, newest first) and today's totals, as plain text."""
    if not runs:
        return "No runs yet. A run is saved for each chat turn."
    rows = [("ID", "WHEN", "CHAT", "PROMPT", "CALLS", "IN", "OUT", "TIME", "STATUS")]
    rows += [(r.id, run_when(r.started_at), r.chat, shorten(r.prompt, 40), str(r.model_calls),
              f"{r.input_tokens:,}", f"{r.output_tokens:,}", f"{r.seconds:.1f} s", r.status) for r in runs]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join([*(row[i].ljust(widths[i]) for i in range(len(widths))), row[-1]]).rstrip()
             for row in rows]
    return "\n".join([*lines, run_totals_line(totals)])


def format_run(run) -> str:
    """One run: its header, then its parts as an indented list."""
    lines = [f"Run {run.id}  {run.status}  {run_when(run.started_at)}  chat {run.chat}  model {run.model}",
             f"Prompt: {run.prompt}",
             f"Total: {calls_text(run.model_calls)} · {run.input_tokens:,} in / {run.output_tokens:,} out · "
             f"{run.seconds:.1f} s"]
    if not run.parts:
        return "\n".join([*lines, "No parts recorded for this run."])
    lines += ["", *("  " * depth + "- " + run_part_line(part) for depth, part in run_part_rows(run))]
    return "\n".join(lines)

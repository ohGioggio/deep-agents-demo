"""How the CLI shows things: the engine's events, chat tables, transcripts."""
import json
from collections.abc import Callable
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

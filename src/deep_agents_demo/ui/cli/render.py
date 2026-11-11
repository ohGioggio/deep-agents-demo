"""How the CLI shows things: the engine's events, chat tables, transcripts."""
import json
from collections.abc import Callable

from deep_agents_demo.models.events import (
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
        elif isinstance(event, TodosEvent):
            self.write("[todos] " + json.dumps(list(event.todos), ensure_ascii=False))
        elif isinstance(event, (NoticeEvent, FinalAnswerEvent)):
            self.write(event.text)

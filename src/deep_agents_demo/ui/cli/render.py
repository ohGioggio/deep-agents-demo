"""How the CLI shows things: chat tables, transcripts."""
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

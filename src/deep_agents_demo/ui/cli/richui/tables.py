"""The /threads, /files and /agents listings as Rich tables (same columns as the text ones)."""
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from deep_agents_demo.persistence.chats import shorten
from deep_agents_demo.ui.cli.render import shown_path, size

AGENT_COLOUR = "#a78bfa"
FILE_COLOUR = "#7ee787"
DIM = "#7d8590"


def _table(*headers: str) -> Table:
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False, pad_edge=False)
    for header in headers:
        table.add_column(header, no_wrap=header in {"CHAT", "FILE", "AGENT"})
    return table


def chats_table(chats, current: str | None = None) -> Table:
    table = _table("", "CHAT", "TURNS", "LAST USED", "LAST MESSAGE")
    table.columns[2].justify = "right"
    for c in chats:
        when = c.updated[:16].replace("T", " ") if c.updated else "-"
        table.add_row("*" if c.name == current else "", Text(c.name), str(c.turns),
                      Text(when, style=DIM), Text(shorten(c.last, 48)))
    return table


def files_table(files, folder=None) -> Table:
    table = _table("FILE", "VERSION", "SIZE", "SAVED", "CAPTION")
    for f in files:
        table.add_row(Text(f.name, style=FILE_COLOUR), f"v{f.version}", size(f.bytes),
                      Text(f.saved_at[:16].replace("T", " "), style=DIM), Text(shorten(f.caption, 48)))
    return table


def agents_table(rows) -> Table:
    table = _table("AGENT", "TOOLS", "MODEL", "SOURCE", "DESCRIPTION")
    for r in rows:
        overrides = f" (overrides {r['overrides']})" if r["overrides"] else ""
        table.add_row(Text(r["name"], style=AGENT_COLOUR), Text(r["tools"]), Text(r["model"]),
                      Text(r["source"] + overrides), Text(r.get("description", ""), style=DIM))
    return table


class RichTables:
    """Prints the three listings to a Rich console; empty lists are plain sentences."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def threads(self, chats, current: str | None) -> None:
        if not chats:
            self.console.print("No chats yet. Start one with: uv run deep-agents-demo --thread <name>")
        else:
            self.console.print(chats_table(chats, current))

    def files(self, files, folder: Path | None) -> None:
        if not files:
            self.console.print("No files delivered in this chat yet.")
            return
        self.console.print(files_table(files, folder))
        if folder is not None:
            self.console.print(Text(f"In {shown_path(str(folder))}", style=DIM))

    def agents(self, rows) -> None:
        self.console.print(agents_table(rows))

"""The Rich theme (palette) and the console factory."""
from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "agent": "#a78bfa",
    "subagent": "#f0abfc",
    "step": "#38bdf8",
    "tool": "#22d3ee",
    "ok": "#7ee787",
    "error": "#ff7b72",
    "approval": "#fbbf24",
    "dim": "#7d8590",
    "file": "#7ee787",
})


def make_console(file=None, width=None, force_terminal=None, record: bool = False) -> Console:
    """A themed console; Rich itself honours ``NO_COLOR``."""
    return Console(file=file, width=width, force_terminal=force_terminal, theme=THEME, record=record)

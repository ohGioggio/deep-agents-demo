"""What the / menu shows and how it filters: pure logic, no terminal I/O."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.utils import get_cwidth

NEEDS_ARGUMENT = {"/rename", "/switch", "/delete", "/room", "/image"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TITLES = {"/switch": "your chats", "/delete": "your chats",  # /rename takes a new name: no list
          "/room": "agents", "/image": "images"}
LABEL_COLUMN = 24  # width of the name column (with its leading space); descriptions start here
MIN_DETAIL = 8     # a description narrower than this is dropped, not cut


@dataclass(frozen=True)
class MenuItem:
    value: str   # what is put in the box (a command name, a chat name, a path)
    label: str   # what the row shows first ("/new [name]")
    detail: str  # the grey description


def filter_items(items: list[MenuItem], typed: str) -> list[MenuItem]:
    """Items whose value contains ``typed`` (any case); prefix matches first, then the given order."""
    needle = typed.lower().lstrip("/")  # the slash of a command name is not part of the match
    hits = [i for i in items if needle in i.value.lower().lstrip("/")]
    starts = [i for i in hits if i.value.lower().lstrip("/").startswith(needle)]
    return starts + [i for i in hits if i not in starts]


def command_items(commands) -> list[MenuItem]:
    """One item per command (its first name; aliases stay typeable), in alphabetical order."""
    seen, items = set(), []
    for method in commands.values():
        if id(method) in seen:
            continue
        seen.add(id(method))
        names, usage, summary = method.command
        items.append(MenuItem("/" + names[0], usage, summary))
    return sorted(items, key=lambda i: i.value)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def argument_items(command: str, typed: str, chats, agents, cwd) -> tuple[str, list[MenuItem]]:
    """The title and items completing ``command``'s argument.

    ``chats``: objects with name/turns/updated/first; ``agents``: (name, description) pairs;
    ``cwd``: where relative image paths start.
    """
    title = TITLES.get(command, "")
    if command == "/room":
        items = [MenuItem(name, name, description) for name, description in agents]
    elif command == "/image":
        return title, _image_items(typed, Path(cwd))
    else:
        items = [MenuItem(c.name, c.name,
                          f"{_plural(c.turns, 'turn')} · {c.updated[:10]} · {c.first}".strip(" ·"))
                 for c in chats]
    return title, filter_items(items, typed)


def _image_items(typed: str, cwd: Path) -> list[MenuItem]:
    """PNG/JPG files in the folder named by ``typed`` (up to its last slash), filtered by the rest."""
    cut = max(typed.rfind("/"), typed.rfind("\\")) + 1
    folder, prefix = typed[:cut], typed[cut:]
    try:
        names = sorted(p.name for p in (cwd / folder).iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    except OSError:
        return []
    return filter_items([MenuItem(folder + n, n, "") for n in names], prefix and folder + prefix)


def ghost_text(typed: str, item: MenuItem) -> str:
    """The rest of ``item`` after what is typed (empty unless ``typed`` is a prefix of it)."""
    if item.value.lower().startswith(typed.lower()):
        return item.value[len(typed):]
    return ""


def window(items, selected: int, size: int = 5) -> tuple[int, int]:
    """The slice of ``items`` to show: at most ``size`` rows, always containing ``selected``."""
    start = max(0, min(selected - size + 1, len(items) - size))
    return start, min(len(items), start + size)


def footer(selected: int, total: int, size: int = 5) -> str:
    """"pos/total" with an arrow on each side that has more rows beyond the window."""
    start, end = window(range(total), selected, size)
    up = "↑ " if start > 0 else ""
    down = " ↓" if end < total else ""
    return f"{up}{selected + 1}/{total}{down} · Tab · Enter · Esc"


def cut(text: str, width: int) -> str:
    """``text`` shortened to ``width`` terminal cells, ending in "…" when something was cut."""
    if get_cwidth(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        w = get_cwidth(ch)
        if used + w > width - 1:
            break
        out, used = out + ch, used + w
    return out + "…" if width > 0 else ""


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - get_cwidth(text))


def fit_row(label: str, detail: str, width: int) -> str:
    """One menu row, at most ``width`` cells: " label", then the description from a fixed column.

    A description that does not fit is cut with "…"; when under MIN_DETAIL cells would be left
    it is dropped (and the name is cut instead if even that is too wide).
    """
    room = width - LABEL_COLUMN - 1
    if not detail or room < MIN_DETAIL:
        return cut(" " + label, width)
    return pad(cut(" " + label, LABEL_COLUMN - 1), LABEL_COLUMN) + cut(detail, room)

"""The framed input box with the Codex-style / menu above it (prompt_toolkit, not full screen)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl, ConditionalContainer, FormattedTextControl, HSplit, Layout, VSplit, Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.styles import Style

from deep_agents_demo.ui.cli.richui.menu import (
    LABEL_COLUMN, MIN_DETAIL, NEEDS_ARGUMENT, MenuItem, argument_items, command_items, filter_items, fit_row, footer,
    ghost_text, pad, window,
)

STYLE = Style.from_dict({
    "frame": "#6366f1", "box": "#475569", "label": "#ffa657", "ghost": "#4b5563",
    "cmd": "#22d3ee", "arg": "#7d8590", "desc": "#9aa4b2", "match": "#facc15",
    "selected": "bg:#312e81", "selected.desc": "#e0e7ff",
})
ARGUMENT_LINE = re.compile(r"^(/\w+) (.*)$")


class Menu(NamedTuple):
    title: str
    items: list[MenuItem]
    typed: str           # the part of the box this menu completes
    command: str | None  # the command whose argument it completes (None: the command name itself)


class _Ghost(Processor):
    """The rest of the selected item, grey, after the text (only with the cursor at the end)."""

    def __init__(self, ghost):
        self.ghost = ghost

    def apply_transformation(self, ti):
        text = self.ghost()
        if text and ti.lineno == 0:
            return Transformation(ti.fragments + [("class:ghost", text)])
        return Transformation(ti.fragments)


class InputBox:
    """Reads a line in a framed box; typing ``/`` opens a filtering menu of commands (and of
    chats, agents or images once a command that needs one is typed).

    ``commands`` is the repl registry (name -> method with ``.command``); the providers are
    called when a menu needs them (chats: objects with name/turns/updated/first; agents:
    (name, description) pairs) and their answers kept for one ``read``.
    """

    def __init__(self, commands, chats_provider, agents_provider, history_path, prompt_label,
                 cwd=None):
        self.commands = command_items(commands)
        self.chats_provider, self.agents_provider = chats_provider, agents_provider
        self.prompt_label = prompt_label
        self.cwd = Path(cwd or ".")
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(path))

    def read(self, prompt_label: str | None = None) -> str:
        """The line typed (raises EOFError on Ctrl+D, KeyboardInterrupt on Ctrl+C, like input())."""
        label = prompt_label or self.prompt_label
        cache: dict = {}
        state = {"selected": 0, "dismissed": False, "history": False}

        def provided(key, provider):
            if key not in cache:
                cache[key] = provider()
            return cache[key]

        def menu() -> Menu | None:
            if state["dismissed"]:
                return None
            text = buffer.text
            found = None
            argument = ARGUMENT_LINE.match(text)
            if text.startswith("/") and " " not in text:
                found = Menu("commands", filter_items(self.commands, text), text, None)
            elif argument and argument.group(1) in NEEDS_ARGUMENT and argument.group(1) != "/rename":
                command, rest = argument.groups()
                if command == "/room":
                    if " -- " in text:
                        return None
                    rest = rest.rsplit(" ", 1)[-1]  # complete the word being typed
                title, items = argument_items(
                    command, rest,
                    provided("chats", self.chats_provider) if command in ("/switch", "/delete")
                    else [],
                    provided("agents", self.agents_provider) if command == "/room" else [],
                    self.cwd)
                found = Menu(title, items, rest, command)
            return found if found and found.items else None

        def selected_item():
            current = menu()
            if current is None:
                return None, None
            state["selected"] = min(state["selected"], len(current.items) - 1)
            return current, current.items[state["selected"]]

        def ghost() -> str:
            current, item = selected_item()
            if item is None or buffer.cursor_position != len(buffer.text):
                return ""
            return ghost_text(current.typed, item)

        def on_text_changed(_buffer):
            state["selected"] = 0
            state["dismissed"] = state["history"]  # a recalled line starts with the menu closed

        def set_text(text):
            buffer.text = text
            buffer.cursor_position = len(text)

        def fill(current, item, space):
            """Put ``item`` where ``current`` was typed."""
            set_text(buffer.text[:len(buffer.text) - len(current.typed)] + item.value
                     + (" " if space else ""))

        def accept(event):
            buffer.validate_and_handle()

        buffer = Buffer(history=self.history, multiline=False, on_text_changed=on_text_changed,
                        accept_handler=lambda b: get_app().exit(result=b.text))
        is_open = Condition(lambda: menu() is not None)
        keys = KeyBindings()

        @keys.add("enter")
        def _(event):
            current, item = selected_item()
            if item is None:
                return accept(event)
            if current.command is None and item.value in NEEDS_ARGUMENT or current.command == "/room":
                return fill(current, item, True)  # wait for the argument (or the next word)
            fill(current, item, False)
            accept(event)

        @keys.add("tab", filter=is_open)
        def _(event):
            current, item = selected_item()
            fill(current, item, current.command is None and item.value in NEEDS_ARGUMENT)

        @keys.add("tab", filter=~is_open)
        def _(event):
            pass  # no tab characters in the box

        @keys.add("right", filter=is_open & Condition(lambda: ghost() != ""))
        def _(event):
            current, item = selected_item()
            fill(current, item, current.command is None and item.value in NEEDS_ARGUMENT)

        @keys.add("escape", filter=is_open, eager=True)  # no wait for an Alt chord
        def _(event):
            state["dismissed"] = True

        def move(step):
            current, _ = selected_item()
            state["selected"] = max(0, min(len(current.items) - 1, state["selected"] + step))

        @keys.add("up")
        def _(event):
            if menu() is not None:
                return move(-1)
            state["history"] = True
            try:
                buffer.history_backward()
            finally:
                state["history"] = False

        @keys.add("down")
        def _(event):
            if menu() is not None:
                return move(1)
            state["history"] = True
            try:
                buffer.history_forward()
            finally:
                state["history"] = False

        @keys.add("c-d")
        def _(event):
            if buffer.text:
                buffer.delete()
            else:
                event.app.exit(exception=EOFError)

        @keys.add("c-c")
        def _(event):
            event.app.exit(exception=KeyboardInterrupt)

        def columns() -> int:
            return get_app().output.get_size().columns

        def menu_lines() -> StyleAndTextTuples:
            current, _ = selected_item()
            width = columns()
            inner = width - 2
            start, end = window(current.items, state["selected"])
            top = f"╭─ {current.title} "
            lines: StyleAndTextTuples = [("class:frame", top + "─" * max(0, width - len(top) - 1) + "╮\n")]
            for index in range(start, end):
                item = current.items[index]
                lines += self._row(item, index == state["selected"], current.typed, inner)
            tail = f" {footer(state['selected'], len(current.items))} ─╯"
            lines.append(("class:frame", "╰" + "─" * max(0, width - len(tail) - 1) + tail))
            return lines

        def border(left, right):
            return lambda: [("class:box", left + "─" * max(0, columns() - 2) + right)]

        menu_window = ConditionalContainer(
            Window(FormattedTextControl(menu_lines), dont_extend_height=True, wrap_lines=False),
            filter=is_open)
        prefix = f"│ {label} › "
        input_row = VSplit([
            Window(FormattedTextControl(lambda: [("class:box", "│ "), ("class:label", f"{label} › ")]),
                   width=len(prefix), dont_extend_width=True),
            Window(BufferControl(buffer, input_processors=[_Ghost(ghost)]), height=1),
            Window(FormattedTextControl(lambda: [("class:box", "│")]), width=1, dont_extend_width=True),
        ], height=1)
        layout = Layout(HSplit([
            menu_window,
            Window(FormattedTextControl(border("╭", "╮")), height=1),
            input_row,
            Window(FormattedTextControl(border("╰", "╯")), height=1),
        ]), focused_element=buffer)

        app = Application(layout=layout, key_bindings=keys, style=STYLE, full_screen=False,
                          erase_when_done=True)
        app.ttimeoutlen = 0.1  # a lone Esc is recognised after 0.1 s (arrow keys arrive whole)
        return app.run()

    @staticmethod
    def _row(item: MenuItem, selected: bool, typed: str, inner: int) -> StyleAndTextTuples:
        """One framed row: the name (first word cyan, the usage grey, the match yellow), the description."""
        text = pad(fit_row(item.label, item.detail, inner), inner)
        base = "class:selected " if selected else ""
        has_detail = bool(item.detail) and inner - LABEL_COLUMN - 1 >= MIN_DETAIL
        name_part, desc_part = (text[:LABEL_COLUMN], text[LABEL_COLUMN:]) if has_detail else (text, "")
        word, _, usage = name_part.lstrip().partition(" ")
        lead = name_part[:len(name_part) - len(name_part.lstrip())]
        hit = word.lower().find(typed.lower()) if typed else -1
        fragments: StyleAndTextTuples = [("class:frame", "│"), (base + "class:cmd", lead)]
        if hit >= 0:
            fragments += [(base + "class:cmd", word[:hit]),
                          (base + "class:match", word[hit:hit + len(typed)]),
                          (base + "class:cmd", word[hit + len(typed):])]
        else:
            fragments.append((base + "class:cmd", word))
        fragments.append((base + "class:arg", (" " + usage) if usage else ""))
        fragments.append((base + ("class:selected.desc" if selected else "class:desc"), desc_part))
        fragments += [("class:frame", "│"), ("", "\n")]
        return fragments

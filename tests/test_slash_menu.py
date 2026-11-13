"""The / menu: its logic (no terminal) and the input box driven through a fake terminal."""
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from deep_agents_demo.ui.cli.repl import COMMANDS
from deep_agents_demo.ui.cli.richui.input_box import InputBox, input_prefix_width
from deep_agents_demo.ui.cli.richui.theme import make_console
from deep_agents_demo.ui.cli.richui.menu import (
    NEEDS_ARGUMENT, MenuItem, argument_items, command_items, filter_items, fit_row, footer,
    ghost_text, window,
)


def chat(name, turns=1, updated="2026-11-10T10:00:00", first="hello"):
    return SimpleNamespace(name=name, turns=turns, updated=updated, first=first, last=first)


class MenuLogicTests(unittest.TestCase):
    def test_filter_contains_prefix_first(self):
        items = command_items(COMMANDS)
        self.assertEqual(["/new", "/agents", "/rename"], [i.value for i in filter_items(items, "/n")])

    def test_filter_is_case_insensitive_and_slash_shows_all(self):
        items = command_items(COMMANDS)
        self.assertEqual(11, len(filter_items(items, "/")))
        self.assertEqual(["/new"], [i.value for i in filter_items(items, "/NEW")])

    def test_commands_come_once_without_aliases(self):
        values = [i.value for i in command_items(COMMANDS)]
        self.assertEqual(len(values), len(set(values)))
        self.assertIn("/threads", values)
        self.assertNotIn("/chats", values)

    def test_window_scrolls_past_five(self):
        items = list(range(11))
        self.assertEqual((0, 5), window(items, 0))
        self.assertEqual((0, 5), window(items, 4))
        self.assertEqual((2, 7), window(items, 6))
        self.assertEqual((6, 11), window(items, 10))
        self.assertEqual((0, 3), window(items[:3], 2))

    def test_footer_text(self):
        self.assertEqual("↑ 7/11 ↓ · Tab · Enter · Esc", footer(6, 11))
        self.assertEqual("1/11 ↓ · Tab · Enter · Esc", footer(0, 11))
        self.assertEqual("↑ 11/11 · Tab · Enter · Esc", footer(10, 11))
        self.assertEqual("1/1 · Tab · Enter · Esc", footer(0, 1))
        self.assertEqual("2/5 · Tab · Enter · Esc", footer(1, 5))  # fits: nothing to scroll
        self.assertEqual("↑ 6/6 · Tab · Enter · Esc", footer(5, 6))  # the window ends at the end
        self.assertEqual("1/6 ↓ · Tab · Enter · Esc", footer(0, 6))

    def test_ghost_text(self):
        item = MenuItem("/rename", "/rename NAME", "rename this chat")
        self.assertEqual("ename", ghost_text("/r", item))
        self.assertEqual("", ghost_text("/en", item))  # contained, not a prefix

    def test_argument_items_for_switch(self):
        chats = [chat("roads"), chat("rooftop-garden", 2), chat("parks")]
        title, items = argument_items("/switch", "ro", chats, [], Path("."))
        self.assertEqual("your chats", title)
        self.assertEqual(["roads", "rooftop-garden"], [i.value for i in items])
        self.assertIn("2 turns", items[1].detail)

    def test_argument_items_for_room(self):
        title, items = argument_items("/room", "wri", [], [("researcher", "digs"), ("writer", "writes")],
                                      Path("."))
        self.assertEqual("agents", title)
        self.assertEqual(["writer"], [i.value for i in items])

    def test_argument_items_for_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pics").mkdir()
            for name in ("a.png", "b.JPG", "c.txt"):
                (root / "pics" / name).write_text("x")
            title, items = argument_items("/image", "pics/", [], [], root)
            self.assertEqual("images", title)
            self.assertEqual(["pics/a.png", "pics/b.JPG"], [i.value for i in items])
            _, items = argument_items("/image", "pics/b", [], [], root)
            self.assertEqual(["pics/b.JPG"], [i.value for i in items])

    def test_exact_match_comes_first(self):
        chats = [chat("foo2"), chat("foo"), chat("xfoo")]
        _title, items = argument_items("/switch", "FOO", chats, [], Path("."))
        self.assertEqual(["foo", "foo2", "xfoo"], [i.value for i in items])

    def test_selected_row_is_marked_without_colour(self):
        item = MenuItem("/new", "/new [name]", "start a new chat")
        row = "".join(text for _style, text in InputBox._row(item, True, "", 40))
        other = "".join(text for _style, text in InputBox._row(item, False, "", 40))
        self.assertTrue(row.startswith("│›/new"), row)
        self.assertTrue(other.startswith("│ /new"), other)
        self.assertEqual(len(row), len(other))

    def test_prefix_width_counts_cells(self):
        self.assertEqual(9, input_prefix_width("公园"))   # "│ " + 4 cells + " › "
        self.assertEqual(10, input_prefix_width("parks"))

    def test_needs_argument(self):
        self.assertEqual({"/rename", "/switch", "/delete", "/room", "/image"}, NEEDS_ARGUMENT)

    def test_narrow_terminal_drops_descriptions(self):
        detail = "start a new chat (a dated name if you give none)"
        self.assertNotIn("start", fit_row("/new [name]", detail, 30))
        row = fit_row("/new [name]", detail, 60)
        self.assertTrue(row.endswith("…"))
        self.assertLessEqual(len(row), 60)
        self.assertTrue(fit_row("/new [name]", detail, 100).endswith("none)"))


def read_with(keys, history=None, chats=(), agents=(), pause=0.0, console=None, label="parks"):
    """Type ``keys`` (a list of strings, with a pause between them) into an InputBox."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history"
        if history:
            path.write_bytes("".join(f"\n# t\n+{line}\n" for line in history).encode("utf-8"))
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            box = InputBox(COMMANDS, lambda: list(chats), lambda: list(agents), path, "parks",
                           console=console)
            # Only this box's thread may write to its pipe, and only while the box is open: on
            # Linux a closed pipe's descriptor is reused by the next test's pipe, so a late
            # write would land in another test's box.
            lock, done, fired = threading.Lock(), threading.Event(), threading.Event()

            def send(text):
                with lock:
                    if done.is_set():
                        return False
                    pipe.send_text(text)
                    return True

            def type_keys():
                time.sleep(0.3)
                for part in keys:
                    if not send(part):
                        return
                    time.sleep(pause)
                if not done.wait(3):  # never hang the suite: a stuck box ends with Ctrl+C
                    fired.set()
                    send("\x03")
            threading.Thread(target=type_keys, daemon=True).start()
            try:
                return box.read(label)
            except KeyboardInterrupt:
                if fired.is_set():  # a failure, not a reason to stop the whole suite
                    raise AssertionError(f"the box never returned for {keys!r}") from None
                raise
            finally:
                with lock:
                    done.set()


class InputBoxTests(unittest.TestCase):
    def test_plain_text_is_sent(self):
        self.assertEqual("hello there", read_with(["hello there\r"]))

    def test_enter_runs_the_selected_command(self):
        self.assertEqual("/new", read_with(["/n\r"]))

    def test_enter_waits_for_an_argument(self):
        self.assertEqual("/rename x", read_with(["/re\r", "x\r"], pause=0.2))

    def test_tab_accepts_ghost(self):
        self.assertEqual("/history", read_with(["/hi", "\t", "\r"], pause=0.2))

    def test_enter_in_argument_menu_picks_and_runs(self):
        self.assertEqual("/switch roads", read_with(["/switch ro\r"], chats=[chat("roads")]))

    def test_esc_closes_and_sends_what_is_typed(self):
        # Esc closes the menu at once: the next key is a plain "b", not an Alt chord
        self.assertEqual("/reb", read_with(["/re", "\x1b", "b", "\r"], pause=0.15))

    def test_rename_has_no_chat_list(self):
        self.assertEqual("/rename ro", read_with(["/rename ro\r"], chats=[chat("roads")]))

    def test_history_up_recalls(self):
        self.assertEqual("what is a park", read_with(["\x1b[A", "\r"], history=["what is a park"],
                                                    pause=0.4))

    def test_switch_picks_the_exact_name(self):
        self.assertEqual("/switch foo", read_with(["/switch foo\r"], chats=[chat("foo2"), chat("foo")]))

    def test_sent_line_stays_on_screen(self):
        c = make_console(file=io.StringIO(), width=60, force_terminal=False, record=True)
        self.assertEqual("hello", read_with(["hello\r"], console=c))
        segments = list(c._record_buffer)
        self.assertIn("parks › hello", c.export_text())
        [sent] = [s for s in segments if "hello" in s.text]
        self.assertEqual("hello", sent.text)
        self.assertFalse(sent.style and sent.style.color, "the sent text is plain")
        [label] = [s for s in segments if "parks" in s.text]
        self.assertIsNotNone(c.get_style("prompt").color)
        self.assertEqual(c.get_style("prompt"), label.style)

    def test_nothing_is_echoed_on_ctrl_d(self):
        c = make_console(file=io.StringIO(), width=60, force_terminal=False, record=True)
        with self.assertRaises(EOFError):
            read_with(["\x04"], console=c)
        self.assertEqual("", c.export_text())

    def test_ctrl_d_is_eof(self):
        with self.assertRaises(EOFError):
            read_with(["\x04"])


if __name__ == "__main__":
    unittest.main()

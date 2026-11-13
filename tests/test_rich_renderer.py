import io
import threading
import unittest

from deep_agents_demo.models.events import (
    MessageEvent, RunError, RunFinished, RunStarted, StepFinished, StepStarted, SubagentStarted,
    TextMessageContent, TextMessageEnd, TextMessageStart, format_footer,
)
from deep_agents_demo.ui.cli.richui.renderer import RichRenderer
from deep_agents_demo.ui.cli.richui.theme import THEME, make_console


def console(width=100):
    return make_console(file=io.StringIO(), width=width, force_terminal=False, record=True)


def answer(r, run_id, text, label="main", deltas=None):
    r.emit(TextMessageStart(run_id, "m1", label))
    for d in deltas or [text]:
        r.emit(TextMessageContent(run_id, "m1", d))
    r.emit(TextMessageEnd(run_id, "m1", text, 10, 5, False))


class RichRendererTest(unittest.TestCase):
    def test_theme_has_styles(self):
        for name in ("agent", "subagent", "step", "tool", "ok", "error", "approval", "muted", "file"):
            self.assertIn(name, THEME.styles)

    def test_answer_box_with_markdown_and_footer(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        answer(r, "r1", "# Title\n\n- one\n- two")
        r.emit(RunFinished("r1", "main", 9800, 450, 74.0, 0.0))
        r.close()
        out = c.export_text()
        for want in ("main", "Title", "• one", "9.8k in / 450 out · 74 s", format_footer(0.0, 9800, 450, 74.0)):
            self.assertIn(want, out)
        self.assertNotIn("#", out)
        self.assertNotIn("- one", out)

    def test_streaming_updates_then_settles(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        answer(r, "r1", "hello brave world", deltas=["hello ", "brave ", "world"])
        r.emit(RunFinished("r1", "main", 1, 2, 1.0, 0.0))
        r.close()
        self.assertEqual(c.export_text().count("hello brave world"), 1)

    def test_room_turn_is_a_box(self):
        c = console()
        r = RichRenderer(c)
        r.emit(StepStarted("s1", "room", 1, 4, "critic"))
        answer(r, "s1", "I disagree.", label="critic")
        r.emit(StepFinished("s1", "room", 1, "ok", 1, 3.0, 5, 6))
        r.close()
        out = c.export_text()
        self.assertIn("room / critic", out)
        self.assertIn("I disagree.", out)
        self.assertIn("5 in / 6 out", out)

    def test_subagent_text_and_tool_text_are_tree_lines_not_boxes(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        r.emit(SubagentStarted("sub", "r1", "researcher", "t1"))
        answer(r, "sub", "secret sub answer", label="researcher")
        r.emit(TextMessageStart("r1", "m2", "main"))
        r.emit(TextMessageContent("r1", "m2", "Let me check"))
        r.emit(TextMessageEnd("r1", "m2", "Let me check", 1, 1, True))
        r.emit(MessageEvent("main", "model", "ai", "plain", (), "plain"))
        r.emit(StepFinished("p1", "plan", 1, "ok", 1, 1.0, 1, 1))
        r.emit(RunFinished("r1", "main", 1, 1, 1.0, 0.0))
        r.close()
        out = c.export_text()
        self.assertIn("researcher: secret sub answer", out)
        self.assertIn("├─ Let me check", out)
        self.assertNotIn("╭", out)

    def test_renders_at_narrow_width(self):
        c = console(50)
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        answer(r, "r1", "word " * 40)
        r.emit(RunFinished("r1", "main", 1200, 30, 130.0, 0.0))
        r.close()
        out = c.export_text()
        lines = out.splitlines()
        self.assertTrue(all(len(line) <= 50 for line in lines))
        self.assertTrue(lines[0][0] in "╭┌")  # rounded, or square on a plain console
        self.assertTrue(any(line[0] in "╰└" for line in lines))
        self.assertIn("2 min 10 s", out)

    def test_run_error_mid_stream_shows_the_error(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        r.emit(TextMessageStart("r1", "m1", "main"))
        r.emit(TextMessageContent("r1", "m1", "partial words"))
        r.emit(RunError("r1", "main", "model exploded"))
        r.close()
        out = c.export_text()
        self.assertEqual(out.count("partial words"), 1)
        self.assertIn("✗ model exploded", out)

    def test_text_tool_text_in_one_run(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        r.emit(TextMessageStart("r1", "m1", "main"))
        r.emit(TextMessageContent("r1", "m1", "Checking now"))
        r.emit(TextMessageEnd("r1", "m1", "Checking now", 1, 1, True))
        r.emit(TextMessageStart("r1", "m2", "main"))
        r.emit(TextMessageContent("r1", "m2", "The final answer"))
        r.emit(TextMessageEnd("r1", "m2", "The final answer", 1, 1, False))
        r.emit(RunFinished("r1", "main", 5, 6, 3.0, 0.0))
        r.close()
        out = c.export_text()
        self.assertEqual(out.count("Checking now"), 1)
        self.assertLess(out.index("Checking now"), out.index("The final answer"))
        self.assertEqual(out.count("The final answer"), 1)
        self.assertEqual(out.count("main"), 1)
        self.assertIn("5 in / 6 out", out)

    def test_concurrent_emits_while_live_refreshes_do_not_deadlock(self):
        c = make_console(file=io.StringIO(), width=80, force_terminal=True)
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        stop = threading.Event()

        def hammer(n):
            i = 0
            while not stop.is_set():
                i += 1
                mid = f"m{n}-{i}"
                r.emit(TextMessageStart("r1", mid, "main"))
                r.emit(TextMessageContent("r1", mid, "word " * 20))
                r.emit(TextMessageEnd("r1", mid, "word " * 20, 1, 1, True))

        threads = [threading.Thread(target=hammer, args=(n,), daemon=True) for n in range(3)]
        for t in threads:
            t.start()
        stop.wait(0.5)
        stop.set()
        for t in threads:
            t.join(5)
            self.assertFalse(t.is_alive(), "emit hung")
        closer = threading.Thread(target=r.close, daemon=True)
        closer.start()
        closer.join(5)
        self.assertFalse(closer.is_alive(), "close hung")

    def test_close_twice_is_safe(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        r.close()
        r.close()


if __name__ == "__main__":
    unittest.main()

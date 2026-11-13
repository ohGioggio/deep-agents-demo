import io
import threading
import time
import unittest
from unittest.mock import patch

from rich.console import Console

from deep_agents_demo.models.events import (
    MessageEvent, NoticeEvent, RunError, RunFinished, RunStarted, StepFinished, StepStarted, SubagentStarted,
    TextMessageContent, TextMessageEnd, TextMessageStart, ToolCallEnd, ToolCallStart, format_footer,
)
from deep_agents_demo.ui.cli.richui.renderer import RichRenderer
from deep_agents_demo.ui.cli.richui.theme import THEME, make_console


def shown(r, height=40, width=60):
    """What the Live region shows right now (drawn on a separate console)."""
    c = Console(file=io.StringIO(), width=width, height=height, force_terminal=False, theme=THEME,
                record=True)
    c.print(r)
    return c.export_text()


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


class ReviewFixesTest(unittest.TestCase):
    def test_long_streaming_answer_shows_its_newest_lines(self):
        c = console(60)
        r = RichRenderer(c)
        text = "\n\n".join(f"row{n:02d}" for n in range(40))
        r.emit(RunStarted("r1", "main"))
        r.emit(TextMessageStart("r1", "m1", "main"))
        r.emit(TextMessageContent("r1", "m1", text))
        live = shown(r, height=10)
        self.assertLessEqual(len(live.splitlines()), 10)
        self.assertIn("row39", live)
        self.assertIn("earlier lines", live)
        self.assertNotIn("row00", live)
        r.emit(TextMessageEnd("r1", "m1", text, 1, 1, False))
        r.emit(RunFinished("r1", "main", 1, 1, 1.0, 0.0))
        r.close()
        out = c.export_text()
        self.assertEqual(out.count("row00"), 1)
        self.assertEqual(out.count("row39"), 1)
        self.assertNotIn("earlier lines", out)

    def test_thinking_comes_back_after_a_tool_round(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        self.assertIn("thinking", shown(r))
        r.emit(TextMessageEnd("r1", "m0", "", 1, 1, True))
        r.emit(ToolCallStart("r1", "c1", "ls", {}))
        r.emit(ToolCallStart("r1", "c2", "ls", {}))
        self.assertNotIn("thinking", shown(r))
        r.emit(ToolCallEnd("r1", "c1", "ls", "ok", "", 0.1))
        self.assertNotIn("thinking", shown(r))   # c2 still runs
        r.emit(ToolCallEnd("r1", "c2", "ls", "ok", "", 0.1))
        self.assertIn("thinking… 0 s", shown(r))
        r.emit(TextMessageStart("r1", "m1", "main"))
        self.assertNotIn("thinking", shown(r))
        r.close()

    def plan_turn(self, r, status="passed", summary="Wrote the park plan."):
        """A main turn calling run_plan, shaped like the engine and PlanRunner emit it."""
        r.emit(RunStarted("r1", "main"))
        r.emit(TextMessageEnd("r1", "m0", "", 1, 1, True))
        r.emit(ToolCallStart("r1", "p", "run_plan", {"goal": "parks"}))
        r.emit(StepStarted("s1", "plan", 1, 2, "general-purpose"))
        r.emit(NoticeEvent("[plan] step 1/2 - general-purpose"))
        r.emit(RunStarted("w1", "plan / general-purpose"))
        answer(r, "w1", "WORKER SAYS hello", label="plan / general-purpose")
        r.emit(RunFinished("w1", "plan / general-purpose", 200, 30, 5.0, 0.0))
        r.emit(RunStarted("v1", "plan / reviewer"))
        answer(r, "v1", "PASS\nlooks fine", label="plan / reviewer")
        r.emit(RunFinished("v1", "plan / reviewer", 100, 10, 2.0, 0.0))
        if status == "passed":
            r.emit(NoticeEvent("[plan] step 1/2 passed review"))
        else:
            r.emit(NoticeEvent("[plan] step 1/2 failed: tests broke"))
        r.emit(StepFinished("s1", "plan", 1, status, 1, 7.0, 300, 40, summary))
        r.emit(StepFinished("u2", "plan", 2, "stopped", 0, 0.0, 0, 0, ""))
        r.emit(NoticeEvent("[plan] Stopped."))
        r.emit(ToolCallEnd("r1", "p", "run_plan", "ok", "", 8.0))
        answer(r, "r1", "MAIN ANSWER here")
        r.emit(RunFinished("r1", "main", 400, 60, 9.0, 0.0))
        r.close()

    def test_plan_worker_and_reviewer_are_not_boxed(self):
        c = console()
        self.plan_turn(RichRenderer(c))
        out = c.export_text()
        self.assertNotIn("plan / general-purpose", out)
        self.assertNotIn("plan / reviewer", out)
        self.assertNotIn("WORKER SAYS", out)
        self.assertNotIn("PASS", out)
        self.assertIn("step 1/2  general-purpose  ✓ passed review", out)
        self.assertIn("Wrote the park plan.", out)
        self.assertIn("300 in / 40 out", out)
        self.assertEqual(sum(line[:1] in "╭┌" for line in out.splitlines() if line), 1)
        self.assertIn("MAIN ANSWER here", out)

    def test_notices_are_shown_except_the_ones_the_step_line_shows(self):
        c = console()
        self.plan_turn(RichRenderer(c), status="failed", summary="tests broke")
        out = c.export_text()
        self.assertIn("[plan] step 1/2 failed: tests broke", out)
        self.assertIn("[plan] Stopped.", out)
        self.assertNotIn("- general-purpose", out)
        self.assertIn("✗ failed", out)

    def test_notify_never_blocks_while_live_draws(self):
        c = console()   # Live is up but draws nothing on a non-terminal: the count below is exact
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        self.assertIsNotNone(r._live)
        held, release = threading.Event(), threading.Event()

        def hold():
            with r._live._lock, r._io:
                held.set()
                release.wait(5)
        t = threading.Thread(target=hold, daemon=True)
        t.start()
        held.wait(5)
        started = time.monotonic()
        with patch("deep_agents_demo.ui.cli.richui.renderer.os.write") as write:
            r.notify("[plan] Pausing")
            elapsed = time.monotonic() - started
        release.set()
        t.join(5)
        self.assertLess(elapsed, 0.2)
        write.assert_not_called()   # Live is drawing: the notice waits for it, not stderr
        self.assertIn("[plan] Pausing", shown(r))
        r.close()
        self.assertEqual(c.export_text().count("[plan] Pausing"), 1)


if __name__ == "__main__":
    unittest.main()

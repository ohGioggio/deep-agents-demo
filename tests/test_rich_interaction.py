import io
import threading
import unittest
from unittest.mock import patch

from deep_agents_demo.models.approvals import ApprovalRequest
from deep_agents_demo.models.events import RunFinished, RunStarted
from deep_agents_demo.ui.cli.prompts import CliInteraction
from deep_agents_demo.ui.cli.richui.interaction import RichInteraction
from deep_agents_demo.ui.cli.richui.renderer import RichRenderer
from deep_agents_demo.ui.cli.richui.theme import make_console

REQUEST = ApprovalRequest("Edit notes.md", "Replace 1 line", "-old line\n+new line\n context", "full text here", ())


def console(force_terminal=False):
    return make_console(file=io.StringIO(), width=100, force_terminal=force_terminal, record=True)


def scripted(answers, log=None):
    items = iter(answers)

    def ask(prompt):
        if log is not None:
            log.append(("ask", prompt))
        value = next(items)
        if isinstance(value, BaseException):
            raise value
        return value
    return ask


class Recorder:
    def __init__(self):
        self.log = []

    def pause(self):
        self.log.append("pause")

    def resume(self):
        self.log.append("resume")


class RichInteractionTest(unittest.TestCase):
    def both(self, answers):
        rich = RichInteraction(console(), ask=scripted(answers)).decide(REQUEST)
        plain = CliInteraction(ask=scripted(answers), say=lambda *a: None).decide(REQUEST)
        self.assertEqual(rich, plain)
        return rich

    def test_approve(self):
        self.assertTrue(self.both(["a"]).approved)

    def test_reject_with_reason(self):
        d = self.both(["r", "too risky"])
        self.assertFalse(d.approved)
        self.assertIn("The user rejected this change: too risky", d.message)

    def test_reject_without_reason_and_eof(self):
        self.assertIn("Don't retry", self.both(["no", ""]).message)
        self.assertEqual(self.both([EOFError()]).message, "The user did not approve this change.")

    def test_panel_and_keys(self):
        c = console()
        RichInteraction(c, ask=scripted(["a"])).decide(REQUEST)
        out = c.export_text()
        self.assertIn("Edit notes.md", out)
        self.assertIn("Replace 1 line", out)

    def test_diff_is_coloured(self):
        c = console(force_terminal=True)
        RichInteraction(c, ask=scripted(["d", "a"])).decide(REQUEST)
        out = c.export_text(styles=True)
        lines = {line for line in out.splitlines() if "old line" in line or "new line" in line}
        self.assertEqual(len(lines), 2)
        old = next(l for l in lines if "old line" in l)
        new = next(l for l in lines if "new line" in l)
        self.assertIn("\x1b[", old)
        self.assertIn("\x1b[", new)
        self.assertNotEqual(old.split("old")[0], new.split("new")[0])

    def test_full_and_unknown_answer(self):
        c = console()
        RichInteraction(c, ask=scripted(["f", "zzz", "a"])).decide(REQUEST)
        out = c.export_text()
        self.assertIn("full text here", out)
        self.assertIn("Type a, r, d or f.", out)

    def test_live_is_paused_while_asking(self):
        rec = Recorder()
        i = RichInteraction(console(), renderer=rec, ask=scripted(["a"], rec.log))
        i.decide(REQUEST)
        self.assertEqual([x if isinstance(x, str) else x[0] for x in rec.log], ["pause", "ask", "resume"])

    def test_resume_even_when_ask_raises(self):
        rec = Recorder()
        i = RichInteraction(console(), renderer=rec, ask=scripted([EOFError()], rec.log))
        i.decide(REQUEST)
        self.assertEqual(rec.log[-1], "resume")

    def test_one_pause_around_a_whole_decide(self):
        rec = Recorder()
        i = RichInteraction(console(), renderer=rec, ask=scripted(["d", "r", "why"], rec.log))
        i.decide(REQUEST)
        self.assertEqual([x if isinstance(x, str) else x[0] for x in rec.log],
                         ["pause", "ask", "ask", "ask", "resume"])

    def test_choose_and_ask(self):
        rec = Recorder()
        i = RichInteraction(console(), renderer=rec, ask=scripted(["2", EOFError(), "hi", KeyboardInterrupt()], rec.log))
        self.assertEqual(i.choose("Pick > ", ("a", "b")), "2")
        self.assertEqual(i.choose("Pick > ", ("a", "b")), "")
        self.assertEqual(i.ask("Name > "), "hi")
        self.assertIsNone(i.ask("Name > "))
        self.assertEqual(rec.log.count("pause"), 4)
        self.assertEqual(rec.log.count("resume"), 4)


class RendererPauseTest(unittest.TestCase):
    def test_pause_stops_live_and_resume_restarts(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        self.assertIsNotNone(r._live)
        r.pause()
        self.assertIsNone(r._live)
        r.emit(RunStarted("r2", "main"))      # an event while paused must not start Live
        self.assertIsNone(r._live)
        r.resume()
        self.assertIsNotNone(r._live)
        r.close()
        self.assertIsNone(r._live)

    def test_a_turn_ending_while_paused_leaves_no_live_after_resume(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        r.pause()
        r.emit(RunFinished("r1", "main", 1, 1, 1.0, 0.0))
        r.resume()
        self.assertIsNone(r._live)
        r.close()

    def test_close_while_paused(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        r.pause()
        r.close()
        self.assertIsNone(r._live)
        r.resume()
        self.assertIsNone(r._live)

    def test_notify_writes_to_stderr_when_nothing_draws(self):
        c = console()
        with patch("deep_agents_demo.ui.cli.richui.renderer.os.write") as write:
            RichRenderer(c).notify("[plan] Pausing")
        self.assertEqual(write.call_args.args[1], b"\n[plan] Pausing\n")
        self.assertNotIn("Pausing", c.export_text())

    def test_a_failing_console_does_not_lose_a_notice(self):
        r = RichRenderer(console())
        r.emit(RunStarted("r1", "main"))
        r.notify("[plan] Pausing")
        with patch.object(r.console, "print", side_effect=OSError("closed")), \
                patch("deep_agents_demo.ui.cli.richui.renderer.os.write") as write:
            r._run([])
        self.assertEqual(write.call_args.args[1], b"\n[plan] Pausing\n")
        r.close()

    def test_notify_never_blocks_when_io_is_held(self):
        c = console()
        r = RichRenderer(c)
        held, release = threading.Event(), threading.Event()

        def hold():
            with r._io:
                held.set()
                release.wait(5)
        t = threading.Thread(target=hold)
        t.start()
        held.wait(5)
        writes = []
        done = threading.Thread(target=lambda: r.notify("[plan] Pausing"))
        with patch("deep_agents_demo.ui.cli.richui.renderer.os.write",
                   side_effect=lambda fd, data: writes.append(data)):
            done.start()
            done.join(2)
            finished = not done.is_alive()
        release.set()
        t.join()
        self.assertTrue(finished)
        self.assertEqual(writes, [b"\n[plan] Pausing\n"])
        self.assertNotIn("Pausing", c.export_text())

    def test_resume_without_a_turn_does_not_start_live(self):
        r = RichRenderer(console())
        r.pause()
        r.resume()
        self.assertIsNone(r._live)


if __name__ == "__main__":
    unittest.main()

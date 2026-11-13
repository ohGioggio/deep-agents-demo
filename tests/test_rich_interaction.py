import io
import unittest

from deep_agents_demo.models.approvals import ApprovalRequest
from deep_agents_demo.models.events import RunStarted
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

    def test_resume_without_a_turn_does_not_start_live(self):
        r = RichRenderer(console())
        r.pause()
        r.resume()
        self.assertIsNone(r._live)


if __name__ == "__main__":
    unittest.main()

import io
import unittest

from rich.console import Group

from deep_agents_demo.models.events import (
    ApprovalRequested, FileEvent, RunFinished, RunStarted, StateSnapshot, StepFinished, StepStarted,
    SubagentFinished, SubagentStarted, TextMessageEnd, ToolCallEnd, ToolCallStart,
)
from deep_agents_demo.ui.cli.richui.renderer import RichRenderer
from deep_agents_demo.ui.cli.richui.theme import make_console
from deep_agents_demo.ui.cli.richui.timeline import Timeline


def console(width=100, force_terminal=False):
    return make_console(file=io.StringIO(), width=width, force_terminal=force_terminal, record=True)


def draw(lines, force_terminal=False):
    c = console(force_terminal=force_terminal)
    c.legacy_windows = False   # legacy Windows consoles drop hyperlinks
    c.print(Group(*lines))
    return c.file.getvalue() if force_terminal else c.export_text()


def call(t, run, cid, name, args, status="ok", summary="", duration=0.2, result=""):
    t.handle(ToolCallStart(run, cid, name, args))
    t.handle(ToolCallEnd(run, cid, name, status, summary, duration, result))


class TimelineTest(unittest.TestCase):
    def test_tool_line(self):
        t = Timeline()
        call(t, "r1", "c1", "read_file", {"file_path": "/skills/x/SKILL.md"}, summary="48 lines")
        out = draw(t.take_settled())
        self.assertIn("├─ read_file  /skills/x/SKILL.md  ✓ 48 lines · 0.2 s", out)

    def test_long_argument_is_cut_to_48(self):
        t = Timeline()
        call(t, "r1", "c1", "write_file", {"p": "x" * 80})
        out = draw(t.take_settled())
        self.assertIn("x" * 47 + "…", out)
        self.assertNotIn("x" * 48, out)

    def test_failed_tool_line_shows_reason(self):
        t = Timeline()
        call(t, "r1", "c1", "search_docs", {"q": "sandbox policy"}, status="error", summary="no results")
        self.assertIn("✗ no results", draw(t.take_settled()))

    def test_running_tool_and_approval(self):
        t = Timeline()
        t.handle(ToolCallStart("r1", "c1", "execute", {"command": "rm -rf build/"}))
        self.assertIn("execute", draw(t.lines()))
        self.assertEqual(t.take_settled(), [])
        t.handle(ApprovalRequested("r1", "c1", "execute"))
        self.assertIn("⏸ needs your approval", draw(t.lines()))
        t.handle(ToolCallEnd("r1", "c1", "execute", "ok", "done", 1.0))
        out = draw(t.take_settled())
        self.assertIn("✓ done", out)
        self.assertNotIn("⏸", out)

    def test_subagent_branch(self):
        t = Timeline()
        t.handle(ToolCallStart("r1", "t1", "task", {"subagent_type": "researcher"}))
        t.handle(SubagentStarted("sub", "r1", "researcher", "t1"))
        call(t, "sub", "c2", "search_assets", {"category": "trees"}, summary="3 results")
        t.handle(TextMessageEnd("sub", "m1", "Oak, pine and birch.\nMore detail.", 1, 1, False))
        t.handle(RunFinished("sub", "researcher", 5, 6, 3.0, 0.0))
        t.handle(SubagentFinished("sub", "ok"))
        self.assertEqual(t.take_settled(), [])   # the task call has not ended yet
        t.handle(ToolCallEnd("r1", "t1", "task", "ok", "", 6.1))
        out = draw(t.take_settled())
        self.assertIn("task → researcher  ✓ 6.1 s", out)
        self.assertIn("│  ├─ search_assets  trees  ✓ 3 results · 0.2 s", out)
        self.assertIn("│  └─ researcher: Oak, pine and birch.", out)
        self.assertIn("5 in / 6 out", out)
        self.assertNotIn("More detail", out)
        self.assertEqual(out.count("task"), 1)

    def test_two_branches_render_separately(self):
        t = Timeline()
        for n, name in (("1", "alpha"), ("2", "beta")):
            t.handle(ToolCallStart("r1", f"t{n}", "task", {}))
            t.handle(SubagentStarted(f"s{n}", "r1", name, f"t{n}"))
        call(t, "s2", "c2", "b_tool", {"a": 1})
        call(t, "s1", "c1", "a_tool", {"a": 1})
        t.handle(TextMessageEnd("s1", "m", "A done", 1, 1, False))
        t.handle(TextMessageEnd("s2", "m", "B done", 1, 1, False))
        for n in "12":
            t.handle(ToolCallEnd("r1", f"t{n}", "task", "ok", "", 1.0))
        out = draw(t.take_settled())
        self.assertLess(out.index("task → alpha"), out.index("a_tool"))
        self.assertLess(out.index("a_tool"), out.index("alpha: A done"))
        self.assertLess(out.index("alpha: A done"), out.index("task → beta"))
        self.assertLess(out.index("task → beta"), out.index("b_tool"))

    def test_plan_step_line(self):
        t = Timeline()
        t.handle(StepStarted("s1", "plan", 1, 2, "general-purpose"))
        t.handle(StepStarted("s2", "plan", 2, 2, "general-purpose"))
        call(t, "s1", "c1", "write_file", {"p": "/a.md"})
        t.handle(StepFinished("s1", "plan", 1, "ok", 2, 41.0, 1, 1))
        out = draw(t.take_settled())
        self.assertIn("step 1/2  general-purpose  ✓ passed review · 2 tries · 41 s", out)
        running = draw(t.lines())
        self.assertIn("step 2/2  general-purpose", running)
        self.assertIn("working… ", running)

    def test_stopped_step_without_a_start_does_not_crash(self):
        t = Timeline()
        t.handle(StepFinished("u1", "plan", 2, "stopped", 0, 0.0, 0, 0))
        self.assertIn("step 2", draw(t.take_settled()))

    def test_room_steps_are_left_to_the_renderer(self):
        t = Timeline()
        t.handle(StepStarted("s1", "room", 1, 4, "critic"))
        t.handle(StepFinished("s1", "room", 1, "ok", 1, 1.0, 1, 1))
        self.assertEqual(t.lines(), [])

    def test_text_beside_tool_calls_is_a_dim_line(self):
        t = Timeline()
        t.handle(TextMessageEnd("r1", "m1", "Let me check the skill. " + "y" * 200, 1, 1, True))
        lines = t.take_settled()
        self.assertIn("├─ Let me check the skill.", draw(lines))
        self.assertLessEqual(len(lines[0].plain) - 3, 100)
        self.assertEqual(lines[0].spans[-1].style, "muted")

    def test_file_line_links_to_the_file(self):
        t = Timeline()
        call(t, "r1", "c1", "send_file", {"name": "park.md"}, summary="v1 · 3.1 KB")
        self.assertEqual(t.take_settled(), [])   # waits for its file line
        t.handle(FileEvent("main", "park.md", 1, "text", 3174, "", "data/outputs/u/park.md"))
        out = draw(t.take_settled(), force_terminal=True)
        self.assertIn("[file] park.md  v1  3.1 KB  →", out)
        self.assertIn("]8;", out)
        self.assertIn("file:///", out)

    def test_send_file_without_file_event_is_released_by_the_next_event(self):
        t = Timeline()
        call(t, "r1", "c1", "send_file", {"name": "park.md"})
        call(t, "r1", "c2", "ls", {})
        self.assertEqual(len(t.take_settled()), 2)

    def test_todos_checklist(self):
        t = Timeline()
        t.handle(StateSnapshot(({"content": "plan", "status": "completed"},
                                {"content": "build", "status": "in_progress"},
                                {"content": "ship", "status": "pending"})))
        out = draw(t.lines())
        self.assertIn("✓ plan", out)
        self.assertIn("▸ build", out)
        self.assertIn("○ ship", out)

    def test_verbose_shows_json_and_full_result(self):
        t = Timeline(verbose=True)
        call(t, "r1", "c1", "read_file", {"file_path": "/a", "limit": 3}, result="line one\nline two")
        out = draw(t.take_settled())
        self.assertIn('"file_path": "/a"', out)
        self.assertIn('"limit": 3', out)
        self.assertIn("line two", out)
        quiet = Timeline()
        call(quiet, "r1", "c1", "read_file", {"file_path": "/a"}, result="line one\nline two")
        self.assertNotIn("line two", draw(quiet.take_settled()))

    def test_finished_branches_are_printed_once(self):
        c = console()
        r = RichRenderer(c)
        r.emit(RunStarted("r1", "main"))
        r.emit(ToolCallStart("r1", "c1", "read_file", {"file_path": "/a"}))
        r.emit(ToolCallEnd("r1", "c1", "read_file", "ok", "3 lines", 0.1))
        r.emit(ToolCallStart("r1", "c2", "list_dir", {"path": "/"}))
        self.assertEqual(c.export_text(clear=False).count("read_file"), 1)
        r.emit(ToolCallEnd("r1", "c2", "list_dir", "ok", "2 entries", 0.1))
        r.emit(RunFinished("r1", "main", 1, 1, 1.0, 0.0))
        r.close()
        out = c.export_text()
        self.assertEqual(out.count("read_file"), 1)
        self.assertEqual(out.count("list_dir"), 1)

    def test_unfinished_calls_are_marked_when_the_turn_ends(self):
        t = Timeline()
        t.handle(RunStarted("r1", "main"))
        t.handle(ToolCallStart("r1", "c1", "execute", {"command": "sleep"}))
        t.handle(RunFinished("r1", "main", 1, 1, 1.0, 0.0))
        self.assertIn("✗ interrupted", draw(t.take_settled()))
        self.assertEqual(t.lines(), [])


if __name__ == "__main__":
    unittest.main()

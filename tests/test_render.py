"""The turn engine emits events; the CLI renderer prints them exactly as the CLI did."""
import contextlib
import io
import json
import tempfile
import unittest

from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from recording_ui import RecordingUI
from test_agents import Model, call

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.core.turns import message_event, run_turn
from deep_agents_demo.models.events import (
    FinalAnswerEvent,
    MessageEvent,
    NoticeEvent,
    TodosEvent,
)
from deep_agents_demo.ui.cli.render import CliRenderer

# What stream_turn printed for this scripted turn before the UI seam (captured on
# 9d437c7 with contextlib.redirect_stdout).
EXPECTED = ('\n[main / model]\n================================== Ai Message ==================================\n'
            'Tool Calls:\n  ls (l)\n Call ID: l\n  Args:\n    path: /\n\n[main / tools]\n'
            '================================= Tool Message =================================\n'
            'Name: ls\n\nNo files found\n\n[main / model]\n'
            '================================== Ai Message ==================================\n\nThe answer.\n')


def scripted_agent():
    model = Model(responses=[call("ls", {"path": "/"}, "l"), AIMessage(content="The answer.", id="a2")])
    return build_agent(model, InMemorySaver(), FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True))


def printed(**options) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_turn(scripted_agent(), "go", {"configurable": {"thread_id": "t"}}, CliRenderer(), None, **options)
    return out.getvalue()


class RenderTests(unittest.TestCase):
    def test_turn_renders_exactly_as_before(self):
        self.assertEqual(printed(), EXPECTED)

    def test_quiet_prints_only_the_answer(self):
        self.assertEqual(printed(quiet=True), "The answer.\n")

    def test_show_false_prints_nothing(self):
        self.assertEqual(printed(show=False), "")

    def test_a_silent_turn_builds_no_events(self):
        from unittest.mock import patch

        from deep_agents_demo.core import turns
        with patch.object(turns, "message_event", wraps=turns.message_event) as made:
            self.assertEqual(printed(show=False), "")
        made.assert_not_called()

    def test_images_show_as_a_note(self):
        message = HumanMessage(content=[{"type": "text", "text": "look"},
                                        {"type": "image", "base64": "QUJD", "mime_type": "image/png"}])
        event = message_event("main", "model", message)
        self.assertIn("[image]", event.pretty)
        self.assertNotIn("QUJD", event.pretty)

    def test_file_line(self):
        from pathlib import Path

        from deep_agents_demo.models.events import FileEvent
        lines = []
        saved = Path.cwd() / "data" / "outputs" / "demo-user" / "city-demo" / "sandbox-policy.md"
        CliRenderer(write=lines.append).emit(FileEvent("main", "sandbox-policy.md", 1, "md", 4300, "", str(saved)))
        self.assertEqual(lines, ["[file] sandbox-policy.md  v1  4.2 KB  ->  "
                                 "data/outputs/demo-user/city-demo/sandbox-policy.md"])
        lines.clear()
        CliRenderer(write=lines.append).emit(FileEvent("main", "big.png", 3, "png", 3 * 1024 * 1024, "", "/elsewhere/big.png"))
        self.assertEqual(lines, ["[file] big.png  v3  3.0 MB  ->  /elsewhere/big.png"])

    def test_events_are_plain_data(self):
        events = [MessageEvent("main", "model", "ai", "hi", ({"name": "ls", "args": {}},), "pretty"),
                  TodosEvent(({"content": "x", "status": "pending"},)), NoticeEvent("[plan] go"),
                  FinalAnswerEvent("done")]
        for event in events:
            self.assertEqual(json.loads(json.dumps(event.to_dict()))["kind"], type(event).__name__)

    def test_other_events_render_as_before(self):
        lines = []
        render = CliRenderer(write=lines.append)
        render.emit(TodosEvent(({"content": "x", "status": "pending"},)))
        render.emit(NoticeEvent("[plan] step 1/2 - general-purpose"))
        render.emit(FinalAnswerEvent("done"))
        self.assertEqual(lines, ['[todos] [{"content": "x", "status": "pending"}]',
                                 "[plan] step 1/2 - general-purpose", "done"])

    def test_recording_ui_renders_through_the_cli_renderer(self):
        ui = RecordingUI()
        run_turn(scripted_agent(), "go", {"configurable": {"thread_id": "t"}}, ui, None)
        self.assertEqual("".join(line + "\n" for line in ui.lines), EXPECTED)


if __name__ == "__main__":
    unittest.main()

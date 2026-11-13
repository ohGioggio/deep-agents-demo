"""The engine's run, text, tool call and subagent events (alongside today's events)."""
import json
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from streaming_model import StreamingModel
from test_agents import call

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.core.turns import run_turn, summarize_result
from deep_agents_demo.models.events import (
    MessageEvent,
    RunError,
    RunFinished,
    RunStarted,
    StateSnapshot,
    SubagentFinished,
    SubagentStarted,
    TextMessageContent,
    TextMessageEnd,
    TextMessageStart,
    ToolCallEnd,
    ToolCallStart,
)


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)

    def of(self, kind):
        return [e for e in self.events if isinstance(e, kind)]


class EngineTest(unittest.TestCase):
    """A clean temporary workspace (with an empty skills folder, so nothing is reported missing)."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "skills").mkdir()
        self.backend = FilesystemBackend(root_dir=str(self.root), virtual_mode=True)

    def agent(self, *responses, features=()):
        return build_agent(StreamingModel(responses=list(responses)), InMemorySaver(), self.backend, list(features))

    def turn(self, *responses, **options) -> RecordingSink:
        sink = RecordingSink()
        run_turn(self.agent(*responses), "go", {"configurable": {"thread_id": "t"}}, sink, None, **options)
        return sink


def tasks(*call_ids):
    return AIMessage(content="", tool_calls=[
        {"name": "task", "id": i, "args": {"description": f"job {i}", "subagent_type": "general-purpose"}}
        for i in call_ids])


class TurnEventTests(EngineTest):
    def test_streamed_answer_arrives_in_pieces(self):
        sink = self.turn(AIMessage(content="The answer is here.", id="a1"))
        self.assertEqual(len(sink.of(TextMessageStart)), 1)
        pieces = sink.of(TextMessageContent)
        self.assertGreaterEqual(len(pieces), 3)
        self.assertEqual("".join(p.delta for p in pieces), "The answer is here.")
        [end] = sink.of(TextMessageEnd)
        self.assertEqual((end.text, end.input_tokens, end.output_tokens, end.has_tool_calls),
                         ("The answer is here.", 100, 10, False))
        self.assertEqual([e.text for e in sink.of(MessageEvent)], ["The answer is here."], "today's event still comes")

    def test_run_started_and_finished_around_the_turn(self):
        sink = self.turn(AIMessage(content="Done.", id="a1"))
        self.assertIsInstance(sink.events[0], RunStarted)
        self.assertIsInstance(sink.events[-1], RunFinished)
        started, finished = sink.events[0], sink.events[-1]
        self.assertEqual((started.run_id, started.label), (finished.run_id, "main"))
        self.assertEqual((finished.input_tokens, finished.output_tokens), (100, 10))
        self.assertGreaterEqual(finished.duration, 0)
        self.assertEqual({e.run_id for e in sink.of(TextMessageContent)}, {started.run_id})

    def test_tool_call_start_and_end(self):
        sink = self.turn(call("ls", {"path": "/skills/"}, "l"), AIMessage(content="Nothing there.", id="a2"))
        [start] = sink.of(ToolCallStart)
        self.assertEqual((start.tool_call_id, start.name, start.args), ("l", "ls", {"path": "/skills/"}))
        [end] = sink.of(ToolCallEnd)
        self.assertEqual((end.tool_call_id, end.name, end.status, end.summary, end.result),
                         ("l", "ls", "ok", "No files found", "No files found"))
        self.assertGreaterEqual(end.duration, 0)
        self.assertLess(sink.events.index(start), sink.events.index(end))
        tool_only = sink.of(TextMessageEnd)[0]
        self.assertTrue(tool_only.has_tool_calls)

    def test_failed_tool_is_an_error(self):
        sink = self.turn(call("read_file", {"file_path": "/missing.md"}, "r"), AIMessage(content="Gone.", id="a2"))
        [end] = sink.of(ToolCallEnd)
        self.assertEqual(end.status, "error", end.result)
        self.assertEqual(end.summary, end.result.splitlines()[0][:60])

    def test_subagent_run(self):
        sink = self.turn(tasks("t1"), AIMessage(content="Sub agent done."), AIMessage(content="All done.", id="a2"))
        [started] = sink.of(SubagentStarted)
        main = sink.of(RunStarted)[0].run_id
        self.assertEqual((started.name, started.tool_call_id, started.parent_run_id), ("general-purpose", "t1", main))
        self.assertTrue(started.run_id.startswith("tools:"))
        own = [e for e in sink.of(TextMessageContent) if e.run_id == started.run_id]
        self.assertEqual("".join(e.delta for e in own), "Sub agent done.")
        [finished] = sink.of(SubagentFinished)
        self.assertEqual((finished.run_id, finished.status), (started.run_id, "ok"))
        order = [sink.events.index(e) for e in (started, own[0], finished)]
        self.assertEqual(order, sorted(order))

    def test_parallel_subagents_have_separate_runs(self):
        sink = self.turn(tasks("t1", "t2"), AIMessage(content="First sub."), AIMessage(content="Second sub."),
                    AIMessage(content="Both done.", id="a2"))
        started = sink.of(SubagentStarted)
        self.assertEqual(len(started), 2)
        self.assertNotEqual(started[0].run_id, started[1].run_id)
        self.assertEqual({s.tool_call_id for s in started}, {"t1", "t2"})
        # each run is matched to its own call: the text it streamed is that call's result
        results = {e.tool_call_id: e.result for e in sink.of(ToolCallEnd)}
        for run in started:
            text = "".join(e.delta for e in sink.of(TextMessageContent) if e.run_id == run.run_id)
            self.assertEqual(text, results[run.tool_call_id])
        self.assertEqual({f.run_id for f in sink.of(SubagentFinished)}, {s.run_id for s in started})

    def test_run_totals_include_nested_runs(self):
        sink = self.turn(tasks("t1"), AIMessage(content="Sub done."), AIMessage(content="Main done.", id="a2"))
        [finished] = sink.of(RunFinished)
        main_own = sum(e.input_tokens for e in sink.of(TextMessageEnd) if e.run_id == finished.run_id)
        self.assertEqual(main_own, 200, "the main agent called the model twice")
        self.assertEqual(finished.input_tokens, 300, "main's 200 plus the subagent's 100")
        self.assertEqual(finished.output_tokens, 30)

    def test_todos_change_is_a_state_snapshot(self):
        todos = [{"content": "look", "status": "pending"}]
        sink = self.turn(call("write_todos", {"todos": todos}, "w"), AIMessage(content="Planned.", id="a2"))
        [snapshot] = sink.of(StateSnapshot)
        self.assertEqual(snapshot.todos, tuple(todos))

    def test_quiet_and_silent_turns_emit_no_new_events(self):
        for options in ({"quiet": True}, {"show": False}):
            with self.subTest(options=options):
                sink = self.turn(AIMessage(content="Done.", id="a1"), **options)
                self.assertFalse([e for e in sink.events if isinstance(e, (RunStarted, TextMessageContent))])

    def test_a_failing_stream_is_a_run_error(self):
        agent = self.agent(AIMessage(content="x"))
        sink = RecordingSink()
        broken = type("Broken", (), {"stream": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
                                     "get_state": agent.get_state})()
        with self.assertRaises(RuntimeError):
            run_turn(broken, "go", {"configurable": {"thread_id": "t"}}, sink, None)
        [error] = sink.of(RunError)
        self.assertEqual((error.run_id, error.label, error.message), (sink.of(RunStarted)[0].run_id, "main", "boom"))
        self.assertFalse(sink.of(RunFinished))


class Recorded:
    """An agent that records the stream modes it was asked for."""

    def __init__(self, agent):
        self.agent, self.modes = agent, []

    def stream(self, payload, **options):
        self.modes.append(options.get("stream_mode"))
        return self.agent.stream(payload, **options)

    def get_state(self, config):
        return self.agent.get_state(config)


class SilentTurnTests(EngineTest):
    def test_silent_turns_ask_for_updates_only(self):
        for options in ({"quiet": True}, {"show": False}):
            with self.subTest(options=options):
                agent = Recorded(self.agent(AIMessage(content="Done.", id="a1")))
                run_turn(agent, "go", {"configurable": {"thread_id": "t"}}, RecordingSink(), None, **options)
                self.assertEqual(agent.modes, [["updates"]])

    def test_a_quiet_turn_still_shows_the_file_and_answer(self):
        from recording_ui import RecordingUI

        from deep_agents_demo.features.files.feature import files_feature
        from deep_agents_demo.persistence.chats import thread_id
        from deep_agents_demo.persistence.outputs import OutputStore
        (self.root / "plan.md").write_bytes(b"# Plan")
        files = files_feature(self.backend, OutputStore(self.root / "outputs"), "alice")
        agent = Recorded(self.agent(call("send_file", {"path": "/plan.md"}, "s1"),
                                    AIMessage(content="Here it is."), features=[files]))
        ui = RecordingUI()
        run_turn(agent, "go", {"configurable": {"thread_id": thread_id("alice", "parks")}}, ui, quiet=True)
        self.assertEqual(len(ui.lines), 2, ui.lines)
        self.assertTrue(ui.lines[0].startswith("[file] plan.md  v1"), ui.lines)
        self.assertEqual(ui.lines[1], "Here it is.")
        self.assertEqual(agent.modes, [["updates"]])

    def test_a_failed_send_file_is_an_error(self):
        from deep_agents_demo.features.files.feature import files_feature
        from deep_agents_demo.persistence.chats import thread_id
        from deep_agents_demo.persistence.outputs import OutputStore
        files = files_feature(self.backend, OutputStore(self.root / "outputs"), "alice")
        agent = self.agent(call("send_file", {"path": "/missing.md"}, "s1"), AIMessage(content="Sorry."),
                           features=[files])
        sink = RecordingSink()
        run_turn(agent, "go", {"configurable": {"thread_id": thread_id("alice", "parks")}}, sink, None)
        [end] = sink.of(ToolCallEnd)
        self.assertEqual(end.status, "error", end.result)
        self.assertTrue(end.summary.startswith("Can't read /missing.md"), end.summary)


class ChatModelTests(unittest.TestCase):
    def test_streamed_answers_report_usage(self):
        from deep_agents_demo.config import ChatSettings, Settings
        from deep_agents_demo.providers.chat import chat_model
        model = chat_model(Settings(chat=ChatSettings(api_key="unused", model="unused",
                                                        base_url="https://example.invalid/v1")))
        self.assertIs(model.stream_usage, True)


class SummaryTests(unittest.TestCase):
    def test_known_shapes(self):
        self.assertEqual(summarize_result("read_file", "     1\ta\n     2\tb\n     3\tc"), "3 lines")
        self.assertEqual(summarize_result("ls", "['/a.md', '/b.md']"), "2 entries")
        self.assertEqual(summarize_result("glob", "['/a.md']"), "1 entries")
        self.assertEqual(summarize_result("search_assets", json.dumps([{"id": 1}, {"id": 2}])), "2 results")
        self.assertEqual(summarize_result("send_file", json.dumps({"name": "a.md", "version": 2, "bytes": 4300})),
                         "v2 · 4.2 KB")

    def test_first_line_otherwise(self):
        self.assertEqual(summarize_result("ls", "No files found"), "No files found")
        self.assertEqual(summarize_result("read_file", "Error: File '/x' not found\nmore"), "Error: File '/x' not found")
        self.assertEqual(summarize_result("other", "x" * 80 + "\nsecond"), "x" * 60)
        self.assertEqual(summarize_result("search_docs", "no matches"), "no matches")

    def test_failed_send_file_is_its_first_line(self):
        for text in ("Not sent: a.exe: this kind of file can't be delivered.", "Can't read /x.md: file_not_found."):
            with self.subTest(text=text):
                self.assertEqual(summarize_result("send_file", text + "\nmore"), text)


if __name__ == "__main__":
    unittest.main()

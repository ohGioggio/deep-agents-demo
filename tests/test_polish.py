"""Small follow-ups to plans and rooms: partial turns, stopping, labels, quiet one-shot runs."""
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agents import Model, call
from test_orchestration import RunnerBase, reply
from test_rooms import RoomBase

from deep_agents_demo.chat import stream_turn


class Interrupting:
    """A graph whose run is stopped by a second Ctrl+C."""

    def stream(self, *args, **kwargs):
        raise KeyboardInterrupt


class RoomPolishTests(RoomBase):
    def test_wrap_up_keeps_the_interrupted_turn(self):
        halfway = AIMessage(content="My half-finished point.",
                            tool_calls=[{"name": "ls", "id": "l", "args": {"path": "/"}}])
        runner = self.runner(reply("NEXT: helper\nGo."), halfway, call("ls", {"path": "/"}, "l2"),
                             reply("The summary."), interactive=False)
        runner.budget_size = 3
        outcome = self.room(runner)
        self.assertEqual(outcome["ended_by"], "budget")
        self.assertIn("My half-finished point.", runner.transcript_text())
        self.assertIn("cut short", runner.transcript_text())

    def test_second_ctrl_c_stops_the_room_and_keeps_the_transcript(self):
        folder = self.root / "rooms"
        runner = self.runner(reply("NEXT: helper\nGo."), transcripts=folder)
        real = runner.make
        runner.make = lambda name, extra=(): Interrupting() if name == "helper" and extra else real(name, extra)
        outcome = self.room(runner)
        self.assertEqual((outcome["ended_by"], outcome["summary"]), ("stopped", ""))
        self.assertEqual(len(list(folder.glob("*.md"))), 1)

    def test_turns_are_labelled_with_the_speaker(self):
        runner = self.runner(reply("NEXT: critic\nGo."), reply("A point."), reply("END\nDone."))
        runner.run({"type": "start_room", "question": "q", "participants": ["critic", "helper"]})
        printed = self.said  # what the CLI renderer printed for the room
        self.assertTrue(any("[room / critic / model]" in p for p in printed), printed[:10])
        self.assertFalse(any("[main / model]" in p for p in printed))

    def test_participants_are_checked_without_building_agents(self):
        runner = self.runner(reply("END\nDone."))
        built = []
        real = runner.make

        def make(name, extra=()):
            built.append(name)
            return real(name, extra)
        make.names = real.names
        runner.make = make
        self.room(runner)
        self.assertEqual(built, ["moderator"], "only the one run builds an agent")


class PlanPolishTests(RunnerBase):
    def test_second_ctrl_c_stops_the_plan_and_returns(self):
        runner = self.runner()
        runner.make = lambda name, extra=(): Interrupting()
        runner.make.names = {"general-purpose", "reviewer"}
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["stopped", "stopped"])
        self.assertIn("stopped by the user", outcome["steps"][0]["notes"])

    def test_step_agent_checked_by_name(self):
        runner = self.runner(reply("done"), reply("PASS\nok"))
        built = []
        real = runner.make

        def make(name, extra=()):
            built.append(name)
            return real(name, extra)
        make.names = real.names
        runner.make = make
        self.plan(runner, ("one", "general-purpose"))
        self.assertEqual(built, ["general-purpose", "reviewer"])


class QuietTests(unittest.TestCase):
    def test_quiet_turn_prints_only_the_answer(self):
        import tempfile

        from deepagents.backends import FilesystemBackend

        from deep_agents_demo.core.assembly import build_agent
        model = Model(responses=[call("ls", {"path": "/"}, "l"), AIMessage(content="The answer.")])
        agent = build_agent(model, InMemorySaver(), FilesystemBackend(root_dir=tempfile.mkdtemp(),
                                                                      virtual_mode=True))
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}}, quiet=True)
        self.assertEqual(printed, ["The answer."])

    def test_verbose_flag(self):
        from deep_agents_demo import cli
        self.assertTrue(cli.parse_args(["--prompt", "hi", "--verbose"])[1].verbose)
        self.assertFalse(cli.parse_args(["--prompt", "hi"])[1].verbose)


if __name__ == "__main__":
    unittest.main()

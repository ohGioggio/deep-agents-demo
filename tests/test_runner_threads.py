"""The runner on several threads: each call has its own step and approvals, usage and the
sink stay exact, a sink can be shared through ``LockedSink``, and ``make.read_only`` lists
the agents a parallel worker may run."""
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from recording_ui import Approver, RecordingUI
from streaming_model import SEEN, StreamingModel
from test_agents import ALL_AGENT, agent_from, call

from deep_agents_demo.core.assembly import worker_factory
from deep_agents_demo.features.agents.builtins import BUILTIN_AGENTS
from deep_agents_demo.features.orchestration.runner import LockedSink, Runner, _StepRun

READ_ONLY_AGENT = "---\nname: planner\ndescription: Plans.\npermissionMode: plan\n---\nPlan."
USAGE = {"input_tokens": 100, "output_tokens": 10}  # what streaming_model reports per call


class Threads(unittest.TestCase):
    def setUp(self):
        SEEN.clear()
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def backend(self):
        return FilesystemBackend(root_dir=str(self.root), virtual_mode=True)

    def make(self, *agents, responses=None):
        model = StreamingModel(responses=responses or [AIMessage(content="done")])
        return worker_factory(model, InMemorySaver(), self.backend(), builtins=tuple(BUILTIN_AGENTS.values()),
                              agents=[agent_from(text) for text in agents])

    def runner(self, make, **options):
        ui = RecordingUI()
        return Runner(make, Approver(), ui, ui, thread="t", **options)


class PerCallState(Threads):
    def test_each_thread_counts_only_its_own_step_and_the_exact_usage(self):
        runner = self.runner(self.make(ALL_AGENT))
        first, second = _StepRun("plan", 1, 2, "helper"), _StepRun("plan", 2, 2, "helper")

        def work(step, runs, tag):
            for i in range(runs):
                runner._run_agent("helper", f"{tag} {i}", f"t:{tag}:{i}", step=step)

        threads = [threading.Thread(target=work, args=(first, 3, "a")),
                   threading.Thread(target=work, args=(second, 5, "b"))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual((first.input_tokens, first.output_tokens), (3 * USAGE["input_tokens"], 3 * USAGE["output_tokens"]))
        self.assertEqual((second.input_tokens, second.output_tokens), (5 * USAGE["input_tokens"], 5 * USAGE["output_tokens"]))
        self.assertEqual(runner.usage, {"model_calls": 8, "input_tokens": 800, "output_tokens": 80})

    def test_a_call_without_a_step_still_uses_the_open_step(self):
        runner = self.runner(self.make(ALL_AGENT))
        runner.step = _StepRun("room", 1, 3, "helper")
        runner._run_agent("helper", "hi", "t:open")
        self.assertEqual(runner.step.input_tokens, USAGE["input_tokens"])

    def test_approvals_are_the_ones_given_to_the_call(self):
        runner = self.runner(self.make(ALL_AGENT))
        mine = Approver()
        with patch("deep_agents_demo.features.orchestration.runner.run_turn") as run_turn:
            runner._run_agent("helper", "hi", "t:approvals", approvals=mine)
        self.assertEqual(run_turn.call_args.args[4], mine.decide)

    def test_extra_middleware_is_added_to_the_agent(self):
        class Probe(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.runs = 0

            def before_model(self, state, runtime):
                self.runs += 1
                return None

        probe = Probe()
        runner = self.runner(self.make(ALL_AGENT))
        runner._run_agent("helper", "hi", "t:extra", extra=(probe,))
        self.assertEqual(probe.runs, 1)

    def test_each_thread_reads_back_its_own_thread_id(self):
        runner = self.runner(self.make(ALL_AGENT))
        barrier = threading.Barrier(2)
        seen = {}

        def work(tag):
            barrier.wait()
            runner._run_agent("helper", "hi", f"t:{tag}")
            time.sleep(0.01)
            seen[tag] = runner.last_thread_id

        threads = [threading.Thread(target=work, args=(tag,)) for tag in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(seen, {"a": "t:a", "b": "t:b"})

    def test_a_budget_retry_is_recorded_on_its_new_thread(self):
        make = self.make(ALL_AGENT, responses=[call("ls", {"path": "/"}, "l1"), AIMessage(content="done")])
        ui = RecordingUI(["continue"])
        runner = Runner(make, Approver(), ui, ui, thread="t", budget=1)
        runner._run_agent("helper", "hi", "t:base")
        self.assertEqual(runner.last_thread_id, "t:base+")
        self.assertEqual(len(ui.asked), 1)  # the budget question, once

    def test_the_main_thread_still_sees_the_default_thread_id(self):
        runner = self.runner(self.make(ALL_AGENT))
        self.assertEqual(runner.last_thread_id, "t")
        runner._run_agent("helper", "hi", "t:main")
        self.assertEqual(runner.last_thread_id, "t:main")


class LockedSinkTests(unittest.TestCase):
    def test_emit_is_serialised(self):
        class Slow:
            def __init__(self):
                self.inside, self.overlap, self.events = 0, False, []

            def emit(self, event):
                self.inside += 1
                self.overlap = self.overlap or self.inside > 1
                time.sleep(0.01)
                self.events.append(event)
                self.inside -= 1

        slow = Slow()
        sink = LockedSink(slow)

        def send(tag):
            for i in range(5):
                sink.emit((tag, i))

        threads = [threading.Thread(target=send, args=(tag,)) for tag in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(slow.overlap)
        self.assertEqual(len(slow.events), 20)


class ReadOnlyNames(Threads):
    def test_make_lists_the_read_only_agents_only(self):
        make = self.make(READ_ONLY_AGENT, ALL_AGENT)
        self.assertIn("planner", make.read_only)
        self.assertIn("reviewer", make.read_only)  # built in, read-only
        self.assertNotIn("helper", make.read_only)  # all tools, so not read-only
        self.assertNotIn("general-purpose", make.read_only)  # the default subagent is not read-only

    def test_an_agent_whose_tools_are_all_read_only_is_read_only(self):
        scout = "---\nname: scout\ndescription: Scouts.\ntools: Read, Grep, Glob\n---\nScout.\n"
        make = self.make(scout)
        self.assertIn("scout", make.read_only)

    def test_an_agent_with_a_write_tool_is_not_read_only(self):
        writer = "---\nname: writer\ndescription: Writes.\ntools: Read, Write\n---\nWrite.\n"
        make = self.make(writer)
        self.assertNotIn("writer", make.read_only)

    def test_the_moderator_with_no_tools_counts_as_read_only(self):
        # Current behaviour: an empty tool list is a subset of READ_ONLY, so the built-in moderator
        # (tools=()) is read-only. It writes only the room's transcript, not the workspace.
        make = self.make(ALL_AGENT)
        self.assertIn("moderator", make.read_only)


if __name__ == "__main__":
    unittest.main()

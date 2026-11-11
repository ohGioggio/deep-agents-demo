"""Subagent-driven execution: the parallel cap, workers, run_plan and the plan runner."""
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.agent import Feature, build_agent, worker_factory
from deep_agents_demo.guardrails import Guardrails
from deep_agents_demo.approvals import Approver
from deep_agents_demo.chat import stream_turn
from deep_agents_demo.workspace import SEED_DIR
from test_agents import ALL_AGENT, SEEN, Model, agent_from, call


class Workspace(unittest.TestCase):
    def setUp(self):
        SEEN.clear()
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        shutil.copytree(SEED_DIR / "skills", self.root / "skills")

    def backend(self):
        return FilesystemBackend(root_dir=str(self.root), virtual_mode=True)


CONCURRENCY = {"running": 0, "peak": 0}
CONCURRENCY_LOCK = threading.Lock()


class Slow(FakeMessagesListChatModel):
    """A subagent model that takes a while and records how many run at once."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        with CONCURRENCY_LOCK:
            CONCURRENCY["running"] += 1
            CONCURRENCY["peak"] = max(CONCURRENCY["peak"], CONCURRENCY["running"])
        time.sleep(0.2)
        with CONCURRENCY_LOCK:
            CONCURRENCY["running"] -= 1
        return super()._generate(messages, *args, **kwargs)


def run_plan(goal: str, steps: list[dict]) -> str:
    """A stand-in for the real run_plan tool."""
    return "ran"


class ParallelTests(Workspace):
    def test_at_most_n_subagents_run_at_once(self):
        CONCURRENCY.update(running=0, peak=0)
        helper = agent_from(ALL_AGENT.replace("description: Helps.", "description: Helps.\nmodel: slow"))
        tasks = AIMessage(content="", tool_calls=[
            {"name": "task", "id": f"t{i}", "args": {"description": "go", "subagent_type": "helper"}}
            for i in range(5)])
        main = Model(responses=[tasks, AIMessage(content="done")])
        agent = build_agent(main, InMemorySaver(), self.backend(), agents=[helper], max_parallel=2,
                            model_for=lambda name: Slow(responses=[AIMessage(content="sub done")]))
        out = agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                           {"configurable": {"thread_id": "t"}})
        results = [m for m in out["messages"] if m.type == "tool"]
        self.assertEqual(len(results), 5)
        self.assertLessEqual(CONCURRENCY["peak"], 2)
        self.assertGreaterEqual(CONCURRENCY["peak"], 1)

    def test_subagents_cannot_run_plans(self):
        helper = agent_from(ALL_AGENT)
        model = Model(responses=[call("task", {"description": "go", "subagent_type": "helper"}, "t1"),
                                 call("run_plan", {"goal": "g", "steps": [{"task": "a"}]}, "p1"),
                                 AIMessage(content="sub done"), AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), self.backend(),
                            [Feature("orchestration", tools=(run_plan,))], agents=[helper])
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}}, approver=Approver(interactive=False))
        sub_calls = [m for kind, m in SEEN if kind == "messages" and "Help." in str(m[0].content)]
        tool_results = [str(x.content) for x in sub_calls[-1] if x.type == "tool"]
        self.assertTrue(any("isn't available to this agent" in r for r in tool_results), tool_results)


CRITIC_FILE = "---\nname: critic\ndescription: Finds weaknesses.\ntools: Read, Grep, Glob\n---\nYou are a critic.\n"


class WorkerTests(Workspace):
    def make(self, *responses, features=(), agents=(), **options):
        model = Model(responses=[*responses, AIMessage(content="worker done")])
        return worker_factory(model, InMemorySaver(), self.backend(), features, agents=agents, **options)

    def run_worker(self, graph, approver=None):
        with patch("builtins.print"):
            stream_turn(graph, "do the job", {"configurable": {"thread_id": "w"}},
                        approver=approver or Approver(interactive=False))

    def bound(self):
        return [names for kind, names in SEEN if kind == "tools"]

    def test_worker_has_the_agents_tools_and_prompt(self):
        make = self.make(agents=[agent_from(CRITIC_FILE)])
        self.run_worker(make("critic"))
        first = [m for kind, m in SEEN if kind == "messages"][0]
        self.assertIn("You are a critic.", str(first[0].content))
        self.assertLessEqual(set(self.bound()[0]), {"read_file", "grep", "glob", "ls"})
        self.assertIn("read_file", self.bound()[0])

    def test_worker_approvals(self):
        (self.root / "notes.md").write_text("keep")
        make = self.make(call("delete", {"file_path": "/notes.md"}, "d"), features=[Guardrails.feature()],
                         agents=[agent_from(ALL_AGENT)])

        def ask(_prompt):
            raise AssertionError("asked")
        with self.assertRaisesRegex(AssertionError, "asked"):
            self.run_worker(make("helper"), Approver(ask=ask, say=lambda *_: None))
        self.assertTrue((self.root / "notes.md").exists())

    def test_worker_disabled_tools(self):
        make = self.make(call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                         agents=[agent_from(ALL_AGENT)], disabled_tools=("write_file",))
        self.run_worker(make("helper"))
        self.assertNotIn("write_file", self.bound()[0])
        self.assertFalse((self.root / "x.md").exists())

    def test_worker_call_cap(self):
        ls = [call("ls", {"path": "/"}, f"l{i}") for i in range(5)]
        make = self.make(*ls, agents=[agent_from(ALL_AGENT)], max_calls=2)
        graph = make("helper")
        self.run_worker(graph)
        self.assertEqual(len([k for k, _ in SEEN if k == "messages"]), 2)
        last = graph.get_state({"configurable": {"thread_id": "w"}}).values["messages"][-1]
        self.assertIn("Partial result", str(last.content))

    def test_unknown_agent_gives_none(self):
        self.assertIsNone(self.make()("nope"))

    def test_builtin_reviewer_and_general_purpose(self):
        make = self.make()
        self.run_worker(make("reviewer"))
        self.assertNotIn("write_file", self.bound()[0])
        self.assertIsNotNone(make("general-purpose"))

    def test_workers_cannot_start_plans(self):
        make = self.make(call("run_plan", {"goal": "g", "steps": [{"task": "a"}]}, "p"),
                         features=[Feature("orchestration", tools=(run_plan,))], agents=[agent_from(ALL_AGENT)])
        self.run_worker(make("helper"))
        self.assertNotIn("run_plan", self.bound()[0])


class RunPlanToolTests(Workspace):
    def turn(self, *responses, runners=None, features=(), approver=None):
        from deep_agents_demo.orchestration import orchestration_feature
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), self.backend(), [orchestration_feature(), *features])
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=approver or Approver(interactive=False), runners=runners)
        last = [m for kind, m in SEEN if kind == "messages"][-1]
        return [str(m.content) for m in last if m.type == "tool"]

    def test_run_plan_pauses_and_returns_the_runner_answer(self):
        asked = []

        def runner(request):
            asked.append(request)
            return {"steps": [{"status": "passed"}]}
        results = self.turn(call("run_plan", {"goal": "g", "steps": [{"task": "a"}]}, "p"),
                            runners={"run_plan": runner})
        self.assertEqual(asked, [{"type": "run_plan", "goal": "g",
                                  "steps": [{"task": "a", "agent": "general-purpose"}]}])
        self.assertEqual(results, [json.dumps({"steps": [{"status": "passed"}]})])

    def test_invalid_steps_are_refused_without_pausing(self):
        asked = []
        results = self.turn(call("run_plan", {"goal": "g", "steps": [{"agent": "x"}]}, "p"),
                            runners={"run_plan": asked.append})
        self.assertEqual(asked, [])
        self.assertIn("task", results[0])

    def test_without_a_runner_the_plan_is_refused(self):
        results = self.turn(call("run_plan", {"goal": "g", "steps": [{"task": "a"}]}, "p"))
        self.assertIn("can't run here", results[0])

    def test_approvals_still_go_to_the_approver(self):
        (self.root / "notes.md").write_text("keep")
        asked = []
        self.turn(call("delete", {"file_path": "/notes.md"}, "d"), features=[Guardrails.feature()],
                  runners={"run_plan": lambda request: {}},
                  approver=Approver(ask=lambda p: asked.append(p) or ("r" if len(asked) == 1 else ""),
                                    say=lambda *_: None))
        self.assertTrue(asked)
        self.assertTrue((self.root / "notes.md").exists())


if __name__ == "__main__":
    unittest.main()

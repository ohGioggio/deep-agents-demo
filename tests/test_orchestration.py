"""Subagent-driven execution: the parallel cap, workers, run_plan and the plan runner."""
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

from deep_agents_demo.agent import Feature, build_agent
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


if __name__ == "__main__":
    unittest.main()

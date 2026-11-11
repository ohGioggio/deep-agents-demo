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
from recording_ui import Approver, RecordingUI
from test_agents import ALL_AGENT, SEEN, Model, agent_from, call

from deep_agents_demo.ui.cli.repl import stream_turn
from deep_agents_demo.core.assembly import build_agent, worker_factory
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.agents.builtins import BUILTIN_AGENTS
from deep_agents_demo.features.guardrails.feature import Guardrails
from deep_agents_demo.startup.workspace import SEED_DIR

BUILTINS = tuple(BUILTIN_AGENTS.values())  # what the composition root passes


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
        return worker_factory(model, InMemorySaver(), self.backend(), features, builtins=BUILTINS, agents=agents, **options)

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
        from deep_agents_demo.features.orchestration.feature import (
            orchestration_feature,
        )
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


def reply(text):
    return AIMessage(content=text)


class RunnerBase(Workspace):
    def runner(self, *responses, features=(), answers=(), approvals=(), interactive=True, agents=()):
        from deep_agents_demo.features.orchestration.plans import PlanRunner
        model = Model(responses=[*responses, reply("UNEXPECTED EXTRA CALL")])
        make = worker_factory(model, InMemorySaver(), self.backend(), features, builtins=BUILTINS, agents=agents)
        approve = iter(approvals)
        ui = RecordingUI(answers)
        self.said = ui.lines
        approver = Approver(ask=lambda _: next(approve), say=lambda *_: None)
        return PlanRunner(make, approver, ui, ui, thread="t", max_attempts=2, interactive=interactive)

    def plan(self, runner, *tasks, goal="g"):
        return runner.run({"type": "run_plan", "goal": goal,
                           "steps": [{"task": t, "agent": a} for t, a in tasks]})

    def statuses(self, outcome):
        return [s["status"] for s in outcome["steps"]]

    def inputs(self):
        """The human message each model call started from (worker or reviewer)."""
        return [next(str(m.content) for m in msgs if m.type == "human")
                for kind, msgs in SEEN if kind == "messages"]


class RunnerTests(RunnerBase):
    def test_parse_verdict(self):
        from deep_agents_demo.features.orchestration.plans import parse_verdict
        self.assertEqual(parse_verdict("PASS\nok"), (True, "ok"))
        self.assertEqual(parse_verdict("**FAIL**\nfix x"), (False, "fix x"))
        ok, notes = parse_verdict("Looks fine overall")
        self.assertFalse(ok)
        self.assertIn("no clear verdict", notes)

    def test_two_steps_pass(self):
        runner = self.runner(reply("result one"), reply("PASS\nok"), reply("result two"), reply("PASS\nfine"))
        outcome = self.plan(runner, ("write it", "general-purpose"), ("check it", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["passed", "passed"])
        self.assertEqual(outcome["steps"][0]["result"], "result one")
        self.assertIn("result one", self.inputs()[2])
        self.assertIn("Earlier steps:", self.inputs()[2])
        self.assertIn("Do only this step", self.inputs()[0], "live finding: step 1 did step 2 too")
        self.assertIn("[plan] step 1/2 - general-purpose", self.said)
        self.assertTrue(any("step 1/2 passed review" in s for s in self.said), self.said)

    def test_retry_after_fail_then_pass(self):
        runner = self.runner(reply("v1"), reply("FAIL\nadd tests"), reply("v2"), reply("PASS\ngood"))
        outcome = self.plan(runner, ("do it", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["passed"])
        self.assertIn("add tests", self.inputs()[2])
        self.assertTrue(any("failed review (attempt 1/2): add tests" in s for s in self.said), self.said)

    def test_failed_step_asks_and_skip_continues(self):
        runner = self.runner(reply("a"), reply("FAIL\nno"), reply("b"), reply("FAIL\nstill no"),
                             reply("c"), reply("PASS\nok"), answers=["s"])
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["skipped", "passed"])
        self.assertIn("don't redo", outcome["note"], "live finding: the main agent redid a skipped step")
        self.assertEqual(outcome["steps"][0]["notes"], "still no")

    def test_retry_runs_the_step_again(self):
        runner = self.runner(reply("a"), reply("FAIL\nno"), reply("b"), reply("FAIL\nno"),
                             reply("c"), reply("PASS\nok"), answers=["r"])
        self.assertEqual(self.statuses(self.plan(runner, ("one", "general-purpose"))), ["passed"])

    def test_stop_marks_the_rest_stopped(self):
        runner = self.runner(reply("a"), reply("FAIL\nno"), reply("b"), reply("FAIL\nno"), answers=["x"])
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["failed", "stopped"])

    def test_non_interactive_stops(self):
        runner = self.runner(reply("a"), reply("FAIL\nno"), reply("b"), reply("FAIL\nno"), interactive=False)
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["failed", "stopped"])

    def test_unknown_agent_fails_without_model_calls(self):
        runner = self.runner(answers=["x"])
        outcome = self.plan(runner, ("one", "nobody"))
        self.assertEqual(self.statuses(outcome), ["failed"])
        self.assertIn("no agent named 'nobody'", outcome["steps"][0]["notes"])
        self.assertEqual([k for k, _ in SEEN if k == "messages"], [])

    def test_worker_approval_mid_plan(self):
        (self.root / "notes.md").write_text("old")
        runner = self.runner(call("delete", {"file_path": "/notes.md"}, "d"), reply("deleted it"),
                             reply("PASS\nok"), features=[Guardrails.feature()], approvals=["a"])
        outcome = self.plan(runner, ("remove notes", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["passed"])
        self.assertFalse((self.root / "notes.md").exists())
        worker_calls = [i for i in self.inputs() if i.startswith("Goal:") and "Step 1 of 1" in i
                        and "Reply PASS or FAIL" not in i]
        self.assertEqual(len(worker_calls), 2, "two worker model calls in one attempt: no re-run")

    def test_rejected_approval_inside_worker(self):
        (self.root / "notes.md").write_text("old")
        runner = self.runner(call("delete", {"file_path": "/notes.md"}, "d"), reply("could not delete"),
                             reply("FAIL\nnot deleted"), reply("tried again, refused"), reply("PASS\nok"),
                             features=[Guardrails.feature()], approvals=["r", "keep it", "r", "keep it"])
        outcome = self.plan(runner, ("remove notes", "general-purpose"))
        self.assertTrue((self.root / "notes.md").exists())
        self.assertEqual(self.statuses(outcome), ["passed"])


class BudgetPauseTests(RunnerBase):
    def ls(self, n):
        return [call("ls", {"path": "/"}, f"l{i}") for i in range(n)]

    def test_budget_wraps_up_when_not_interactive(self):
        runner = self.runner(*self.ls(10), interactive=False)
        runner.budget_size = 3
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["stopped", "stopped"])
        self.assertTrue(all("budget used up" in s["notes"] for s in outcome["steps"]))
        self.assertLessEqual(len([k for k, _ in SEEN if k == "messages"]), 3)

    def test_budget_continue_when_asked(self):
        runner = self.runner(*self.ls(3), reply("done"), reply("PASS\nok"), answers=["c"])
        runner.budget_size = 3
        outcome = self.plan(runner, ("one", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["passed"])
        self.assertGreater(runner.budget.used, 3)

    def test_pause_note_reaches_the_next_worker(self):
        runner = self.runner(reply("done"), reply("PASS\nok"), answers=["use pytest"])
        runner.pause_requested = True
        self.plan(runner, ("one", "general-purpose"))
        self.assertIn("Note from the user: use pytest", self.inputs()[0])

    def test_pause_stop(self):
        runner = self.runner(answers=["/stop"])
        runner.pause_requested = True
        outcome = self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["stopped", "stopped"])
        self.assertEqual([k for k, _ in SEEN if k == "messages"], [])


class Boom:
    """A graph whose run fails, like a provider error."""

    def stream(self, *args, **kwargs):
        raise RuntimeError("provider down")


class ReviewFixTests(RunnerBase):
    def test_long_step_does_not_hit_the_recursion_limit(self):
        """Review finding: a worker crashed with GraphRecursionError after ~11 calls."""
        steps = [call("ls", {"path": "/"}, f"l{i}") for i in range(15)]
        runner = self.runner(*steps, reply("done"), reply("PASS\nok"))
        self.assertEqual(self.statuses(self.plan(runner, ("long job", "general-purpose"))), ["passed"])

    def test_a_crashing_run_is_a_failed_attempt(self):
        runner = self.runner(interactive=False)
        runner.make = lambda name, extra=(): Boom()
        outcome = self.plan(runner, ("one", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["failed"])
        self.assertIn("provider down", outcome["steps"][0]["notes"])

    def test_a_second_plan_starts_fresh(self):
        """Review finding: plan 2's worker saw plan 1's messages and its used-up call cap."""
        runner = self.runner(reply("one"), reply("PASS\nok"), reply("two"), reply("PASS\nok"))
        self.plan(runner, ("first", "general-purpose"))
        self.plan(runner, ("second", "general-purpose"))
        calls = [m for kind, m in SEEN if kind == "messages"]
        self.assertEqual(len([m for m in calls[2] if m.type == "human"]), 1)

    def test_retry_starts_fresh_and_keeps_the_notes(self):
        runner = self.runner(reply("a"), reply("FAIL\nno1"), reply("b"), reply("FAIL\nno2"),
                             reply("c"), reply("PASS\nok"), answers=["r"])
        self.assertEqual(self.statuses(self.plan(runner, ("one", "general-purpose"))), ["passed"])
        retry = [m for kind, m in SEEN if kind == "messages"][4]
        self.assertEqual(len([m for m in retry if m.type == "human"]), 1)
        self.assertIn("no2", self.inputs()[4])

    def test_finishing_on_the_last_budgeted_call_passes(self):
        """Review finding: a step that used exactly the budget was reported stopped."""
        runner = self.runner(reply("done"), reply("PASS\nok"), interactive=False)
        runner.budget_size = 2
        self.assertEqual(self.statuses(self.plan(runner, ("one", "general-purpose"))), ["passed"])

    def test_wrap_up_keeps_the_partial_result(self):
        halfway = AIMessage(content="halfway there", tool_calls=[{"name": "ls", "id": "l", "args": {"path": "/"}}])
        runner = self.runner(halfway, *[call("ls", {"path": "/"}, f"l{i}") for i in range(5)], interactive=False)
        runner.budget_size = 3
        outcome = self.plan(runner, ("one", "general-purpose"))
        self.assertEqual(self.statuses(outcome), ["stopped"])
        self.assertIn("halfway there", outcome["steps"][0]["result"])

    def test_ctrl_c_is_left_alone_when_not_interactive(self):
        import signal

        from deep_agents_demo.ui.cli.interrupts import ctrl_c_controls
        before = signal.getsignal(signal.SIGINT)
        with ctrl_c_controls(self.runner(interactive=False), interactive=False):
            self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_second_ctrl_c_stops_now_and_nothing_prints_in_the_handler(self):
        import signal

        from deep_agents_demo.ui.cli.interrupts import ctrl_c_controls
        runner = self.runner()
        with ctrl_c_controls(runner, interactive=True, label=runner.label):
            handler = signal.getsignal(signal.SIGINT)
            handler(2, None)
            self.assertTrue(runner.pause_requested)
            self.assertEqual(self.said, [])
            with self.assertRaises(KeyboardInterrupt):
                handler(2, None)
            self.assertTrue(runner.stopped)


class WorkerStackTests(Workspace):
    def test_workers_summarize_long_contexts(self):
        """Review finding: workers lacked the summarization subagents get (it wraps model
        calls, so it isn't a graph node: check it's built for the worker's model)."""
        from deep_agents_demo.core import assembly as agent_module
        model = Model(responses=[reply("x")])
        real = agent_module.create_summarization_middleware
        with patch.object(agent_module, "create_summarization_middleware", side_effect=real) as made:
            worker_factory(model, InMemorySaver(), self.backend())("general-purpose")
        self.assertIs(made.call_args.args[0], model)


class ChatCleanupTests(Workspace):
    def test_deleting_a_chat_removes_its_plan_threads(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, MessagesState, StateGraph

        from deep_agents_demo.persistence.chats import Chats, thread_id
        graph = StateGraph(MessagesState)
        graph.add_node("n", lambda state: {"messages": [reply("hi")]})
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        with SqliteSaver.from_conn_string(str(self.root / "c.sqlite")) as saver:
            compiled = graph.compile(checkpointer=saver)
            chat = thread_id("alice", "c1")
            for tid in (chat, f"{chat}:plan:abc:1:1:1:worker"):
                compiled.invoke({"messages": [{"role": "user", "content": "x"}]},
                                {"configurable": {"thread_id": tid}})
            Chats(saver, compiled, "alice").delete("c1")
            left = saver.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        self.assertEqual(left, [])

    def test_renaming_a_chat_moves_its_plan_threads(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, MessagesState, StateGraph

        from deep_agents_demo.persistence.chats import Chats, thread_id
        graph = StateGraph(MessagesState)
        graph.add_node("n", lambda state: {"messages": [reply("hi")]})
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        with SqliteSaver.from_conn_string(str(self.root / "c.sqlite")) as saver:
            compiled = graph.compile(checkpointer=saver)
            old, new = thread_id("alice", "c1"), thread_id("alice", "c2")
            for tid in (old, f"{old}:plan:abc:1:1:1:worker"):
                compiled.invoke({"messages": [{"role": "user", "content": "x"}]},
                                {"configurable": {"thread_id": tid}})
            Chats(saver, compiled, "alice").rename("c1", "c2")
            left = sorted(r[0] for r in saver.conn.execute("SELECT DISTINCT thread_id FROM checkpoints"))
        self.assertEqual(left, sorted([new, f"{new}:plan:abc:1:1:1:worker"]))


class WiringTests(Workspace):
    def test_plan_through_the_runtime(self):
        from contextlib import ExitStack

        from deep_agents_demo.ui.cli.repl import runners_for
        from deep_agents_demo.config import (
            ChatSettings,
            MemorySettings,
            Paths,
            Settings,
        )
        from deep_agents_demo.startup import runtime as app
        settings = Settings(chat=ChatSettings(api_key="unused", model="unused", base_url=None),
                            memory=MemorySettings(enabled=False), rag_enabled=False,
                            paths=Paths(data=self.root / "data", workspace=self.root / "ws",
                                        checkpoints=self.root / "data" / "checkpoints.sqlite"))
        from deep_agents_demo.startup.workspace import prepare_workspace
        prepare_workspace(settings)
        model = Model(responses=[call("run_plan", {"goal": "greet", "steps": [{"task": "say hi"}]}, "p"),
                                 reply("hi"), reply("PASS\nfine"), reply("all done")])
        stack = ExitStack()
        self.addCleanup(stack.close)
        with patch.object(app, "chat_model", return_value=model):
            runtime = app.build_runtime(stack.enter_context(app.Resources.open(settings)), "alice")
            approver = Approver(interactive=False)
            config = {"configurable": {"thread_id": "alice:t"}}
            with patch("builtins.print"):
                stream_turn(runtime.agent, "go", config, approver=approver,
                            runners=runners_for(runtime, settings, "alice:t", approver, interactive=False))
        last_main_call = [m for kind, m in SEEN if kind == "messages"][-1]
        result = json.loads([m for m in last_main_call if m.type == "tool"][-1].content)
        self.assertEqual([s["status"] for s in result["steps"]], ["passed"])
        self.assertEqual(result["steps"][0]["result"], "hi")


if __name__ == "__main__":
    unittest.main()

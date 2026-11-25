"""A task for an agent made in the same message, driven by real graphs (no runtime).

The parent is a create_agent with ``NewAgentDelegation`` and a stub ``task`` tool; the
delegated agent gets an approval-gated ``remove_file``. A scripted model answers both.
"""
import asyncio
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from deep_agents_demo.core import delegation
from deep_agents_demo.core.delegation import NewAgentDelegation
from deep_agents_demo.startup.runtime import Runtime, newest_worker_factory

REMOVED = []
APPROVE = {"decisions": [{"type": "approve"}]}


def reject(message="no"):
    return {"decisions": [{"type": "reject", "message": message}]}


@tool
def remove_file(path: str) -> str:
    """Remove a file."""
    REMOVED.append(path)
    return f"removed {path}"


@tool
def task(description: str, subagent_type: str) -> str:
    """Stub: the delegation middleware answers this call."""
    return "task ran"


class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def calls(*items):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in items])


def parent_for(responses):
    return parent_on(Scripted(responses=responses), InMemorySaver())


def parent_on(model, saver):
    def make(name, extra_middleware=()):
        if name != "helper":
            return None
        return create_agent(model=model, tools=[remove_file], system_prompt="",
                            middleware=[HumanInTheLoopMiddleware(interrupt_on={"remove_file": True}),
                                        *extra_middleware],
                            checkpointer=saver)
    make.names = frozenset({"helper"})
    return create_agent(model=model, tools=[task],
                        middleware=[NewAgentDelegation(set(), lambda: make)], checkpointer=saver)


DELEGATE = calls(("task", {"description": "go", "subagent_type": "helper"}, "t1"))
TWO_PAUSES = [DELEGATE,
              calls(("remove_file", {"path": "/d1"}, "r1")),
              calls(("remove_file", {"path": "/d2"}, "r2")),
              AIMessage(content="sub done"), AIMessage(content="all done")]
ONE_PAUSE = [DELEGATE,
             calls(("remove_file", {"path": "/d1"}, "r1"), ("remove_file", {"path": "/d2"}, "r2")),
             AIMessage(content="sub done"), AIMessage(content="all done")]


def delegated_result(messages):
    return next(m.content for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "t1")


class SyncDelegation(unittest.TestCase):
    def setUp(self):
        REMOVED.clear()
        self.config = {"configurable": {"thread_id": "parent"}}

    def sqlite_again(self):
        return SqliteSaver(sqlite3.connect(self.path, check_same_thread=False))

    def sqlite(self):
        """A SqliteSaver on a temp file, as the app uses: a restart is a new connection."""
        self.path = Path(tempfile.mkdtemp()) / "checkpoints.sqlite"
        self.addCleanup(shutil.rmtree, self.path.parent, True)
        return SqliteSaver(sqlite3.connect(self.path, check_same_thread=False))

    def drive(self, parent, answers):
        result = parent.invoke({"messages": [HumanMessage(content="go")]}, self.config)
        for answer in answers:
            result = parent.invoke(Command(resume={item.id: answer for item in result["__interrupt__"]}),
                                   self.config)
        self.assertNotIn("__interrupt__", result)
        return result

    def test_a_task_without_a_description_is_an_error_the_model_can_fix(self):
        """Seen live: the model put the job under "task" instead of "description"; the new agent
        then got an empty request and kept asking what to do. Like the task tool's own check,
        the call gets an error naming the missing argument, and no agent runs."""
        sub_model = Scripted(responses=[AIMessage(content="what should I do?")])
        ran = []

        def make(name, extra_middleware=()):
            ran.append(name)
            return create_agent(model=sub_model, tools=[remove_file], system_prompt="",
                                checkpointer=InMemorySaver())
        make.names = frozenset({"helper"})
        model = Scripted(responses=[calls(("task", {"task": "delete /d1", "subagent_type": "helper"}, "t1")),
                                    AIMessage(content="ok")])
        parent = create_agent(model=model, tools=[task], middleware=[NewAgentDelegation(set(), lambda: make)],
                              checkpointer=InMemorySaver())
        result = parent.invoke({"messages": [HumanMessage(content="go")]}, self.config)
        [answer] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("description", answer.content)
        self.assertEqual(answer.status, "error")
        self.assertEqual(ran, [], "no agent ran on an empty request")

    def test_two_pauses_approve_then_reject_keeps_the_second_file(self):
        result = self.drive(parent_for(TWO_PAUSES), [APPROVE, reject()])
        self.assertEqual(REMOVED, ["/d1"])
        self.assertEqual(delegated_result(result["messages"]), "sub done")

    def test_a_restart_between_pauses_refuses_instead_of_applying_a_stale_answer(self):
        model, saver = Scripted(responses=TWO_PAUSES), self.sqlite()
        parent = parent_on(model, saver)
        result = parent.invoke({"messages": [HumanMessage(content="go")]}, self.config)
        result = parent.invoke(Command(resume={i.id: APPROVE for i in result["__interrupt__"]}), self.config)
        self.assertEqual(REMOVED, ["/d1"])
        # The process restarts while /d2 waits for the user: memory is gone, the checkpoint stays.
        delegation._APPLIED.clear()
        saver.conn.close()
        parent = parent_on(model, self.sqlite_again())
        result = parent.invoke(Command(resume={i.id: reject() for i in result["__interrupt__"]}), self.config)
        self.assertEqual(REMOVED, ["/d1"])
        self.assertTrue(delegated_result(result["messages"]).startswith(
            "The delegated run of helper was interrupted by a restart"), delegated_result(result["messages"]))

    def test_the_pause_names_the_task_call_it_belongs_to(self):
        result = parent_for(ONE_PAUSE).invoke({"messages": [HumanMessage(content="go")]}, self.config)
        [item] = result["__interrupt__"]
        self.assertEqual(item.value["task_call_id"], "t1")
        self.assertEqual([r["name"] for r in item.value["action_requests"]], ["remove_file", "remove_file"])

    def test_a_missing_decision_is_a_rejection(self):
        self.drive(parent_for(ONE_PAUSE), [{"decisions": [{"type": "approve"}]}])
        self.assertEqual(REMOVED, ["/d1"])

    def test_one_pause_with_mixed_answers_runs_only_the_approved_call(self):
        self.drive(parent_for(ONE_PAUSE), [{"decisions": [{"type": "approve"}, {"type": "reject", "message": "no"}]}])
        self.assertEqual(REMOVED, ["/d1"])


class AsyncDelegation(unittest.TestCase):
    def setUp(self):
        REMOVED.clear()

    def test_a_task_without_a_description_is_an_error(self):
        model = Scripted(responses=[calls(("task", {"task": "delete /d1", "subagent_type": "helper"}, "t1")),
                                    AIMessage(content="ok")])
        parent = parent_on(model, InMemorySaver())
        result = asyncio.run(parent.ainvoke({"messages": [HumanMessage(content="go")]},
                                            {"configurable": {"thread_id": "parent"}}))
        [answer] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(answer.status, "error")
        self.assertIn("description", answer.content)

    def test_two_pauses_approve_then_reject_keeps_the_second_file(self):
        async def run():
            parent = parent_for(TWO_PAUSES)
            config = {"configurable": {"thread_id": "parent"}}
            result = await parent.ainvoke({"messages": [HumanMessage(content="go")]}, config)
            for answer in (APPROVE, reject()):
                result = await parent.ainvoke(
                    Command(resume={item.id: answer for item in result["__interrupt__"]}), config)
            return result
        result = asyncio.run(run())
        self.assertEqual(REMOVED, ["/d1"])
        self.assertEqual(delegated_result(result["messages"]), "sub done")

    def async_sqlite_path(self):
        path = Path(tempfile.mkdtemp()) / "checkpoints.sqlite"
        self.addCleanup(shutil.rmtree, path.parent, True)
        return str(path)

    def test_two_pauses_on_an_async_sqlite_saver(self):
        """The app's saver is async-only: the delegation must read state with the async calls."""
        path = self.async_sqlite_path()

        async def run():
            config = {"configurable": {"thread_id": "parent"}}
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                parent = parent_on(Scripted(responses=TWO_PAUSES), saver)
                result = await parent.ainvoke({"messages": [HumanMessage(content="go")]}, config)
                for answer in (APPROVE, reject()):
                    result = await parent.ainvoke(
                        Command(resume={item.id: answer for item in result["__interrupt__"]}), config)
            return result
        result = asyncio.run(run())
        self.assertEqual(REMOVED, ["/d1"])
        self.assertEqual(delegated_result(result["messages"]), "sub done")

    def test_a_restart_between_pauses_on_an_async_sqlite_saver_refuses(self):
        path = self.async_sqlite_path()
        model = Scripted(responses=TWO_PAUSES)
        config = {"configurable": {"thread_id": "parent"}}

        async def run():
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                parent = parent_on(model, saver)
                result = await parent.ainvoke({"messages": [HumanMessage(content="go")]}, config)
                result = await parent.ainvoke(Command(resume={i.id: APPROVE for i in result["__interrupt__"]}),
                                              config)
            # The process restarts while /d2 waits for the user: memory is gone, the checkpoint stays.
            delegation._APPLIED.clear()
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                parent = parent_on(model, saver)
                return await parent.ainvoke(Command(resume={i.id: reject() for i in result["__interrupt__"]}),
                                            config)
        result = asyncio.run(run())
        self.assertEqual(REMOVED, ["/d1"])
        self.assertTrue(delegated_result(result["messages"]).startswith(
            "The delegated run of helper was interrupted by a restart"), delegated_result(result["messages"]))


class StubAgent:
    """Answers "done" on its first run; the state reads are the sync and async pairs."""

    def __init__(self):
        self.messages = []

    def get_state(self, config):
        return SimpleNamespace(values={"messages": list(self.messages)}, tasks=[])

    async def aget_state(self, config):
        return self.get_state(config)

    def invoke(self, payload, config):
        self.messages.append(AIMessage(content="done"))

    async def ainvoke(self, payload, config):
        self.invoke(payload, config)


class Factory:
    """A worker factory whose ``names`` can change after the tool call has started."""

    def __init__(self, names=()):
        self.names = frozenset(names)

    def __call__(self, name, extra_middleware=()):
        return StubAgent()


def same_reply(*siblings, name="sweeper"):
    """A request whose reply holds the task call for ``name`` plus the given sibling calls."""
    task_call = {"name": "task", "args": {"description": "sweep", "subagent_type": name}, "id": "t1"}
    tool_calls = [task_call, *({"name": n, "args": a, "id": f"s{i}"} for i, (n, a) in enumerate(siblings))]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
    return SimpleNamespace(tool_call=task_call, state=state)


def make_middleware(factory):
    return NewAgentDelegation(set(), lambda: factory)


class SameReplyCreate(unittest.TestCase):
    def setUp(self):
        self.wait, self.poll = delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL
        delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL = 2.0, 0.02
        self.handled = []

    def tearDown(self):
        delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL = self.wait, self.poll

    def handler(self, request):
        self.handled.append(request)
        return ToolMessage(content="handled", tool_call_id="t1", name="task")

    def learns_later(self, factory, name="sweeper", after=0.1):
        timer = threading.Timer(after, lambda: setattr(factory, "names", frozenset({"helper", name})))
        timer.start()
        self.addCleanup(timer.cancel)

    def test_waits_for_an_agent_created_in_the_same_reply(self):
        factory = Factory({"helper"})
        self.learns_later(factory)
        request = same_reply(("create_agent", {"name": "sweeper"}))
        target = make_middleware(factory)._target(request)
        self.assertIsNotNone(target)
        make, name = target
        self.assertIs(make, factory)
        self.assertEqual(name, "sweeper")

    def test_gives_up_with_a_clear_error(self):
        delegation.NEW_AGENT_WAIT = 0.3
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "sweeper"}))
        start = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "Error: sweeper was being created in this same step but isn't "
                                         "available yet; call list_agents, then task.")
        self.assertEqual(self.handled, [])

    def test_no_sibling_create_goes_straight_to_the_handler(self):
        factory = Factory({"helper"})
        start = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(same_reply(), self.handler)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(len(self.handled), 1)
        self.assertEqual(result.content, "handled")

    def test_a_create_for_another_name_does_not_wait(self):
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "other"}))
        start = time.monotonic()
        make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(len(self.handled), 1)

    def test_a_marketplace_create_does_not_count(self):
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "sweeper", "scope": "marketplace"}))
        start = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(len(self.handled), 1)
        self.assertEqual(result.content, "handled")

    def test_an_import_of_another_stem_does_not_count(self):
        factory = Factory({"helper"})
        request = same_reply(("import_agent", {"path": "/agents/critic.md"}))
        start = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(len(self.handled), 1)
        self.assertEqual(result.content, "handled")

    def test_an_import_without_a_path_does_not_wait_or_crash(self):
        factory = Factory({"helper"})
        request = same_reply(("import_agent", {}))
        start = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(len(self.handled), 1)
        self.assertEqual(result.content, "handled")

    FAILED = ("Error: creating sweeper failed in this same step; see that tool's result, "
              "then call list_agents.")

    def settle_later(self, settled, call_id, value, after=0.1):
        timer = threading.Timer(after, lambda: settled.update({call_id: value}))
        timer.start()
        self.addCleanup(timer.cancel)

    def saves_later(self, factory, settled, call_id, name="sweeper", after=0.1):
        """A create that saves, in the app's order: the agent exists before its call is settled
        (two separate timers raced on a slow runner and settled first)."""
        def save():
            factory.names = frozenset({*factory.names, name})
            settled[call_id] = True
        timer = threading.Timer(after, save)
        timer.start()
        self.addCleanup(timer.cancel)

    def test_a_sibling_create_settling_false_during_the_wait_ends_it_early(self):
        delegation.NEW_AGENT_WAIT = 5.0
        factory, settled = Factory({"helper"}), {}
        self.settle_later(settled, "s0", False)
        request = same_reply(("create_agent", {"name": "sweeper"}))
        started = time.monotonic()
        result = NewAgentDelegation(set(), lambda: factory, settled.get).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((result.status, result.content), ("error", self.FAILED))
        self.assertEqual(self.handled, [])

    def test_a_create_settled_false_before_the_task_waits_fails_at_once(self):
        delegation.NEW_AGENT_WAIT = 5.0
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "sweeper"}))
        started = time.monotonic()
        result = NewAgentDelegation(set(), lambda: factory, {"s0": False}.get).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual((result.status, result.content), ("error", self.FAILED))

    def test_one_refused_sibling_does_not_end_the_wait_of_one_that_saves(self):
        factory, settled = Factory({"helper"}), {"s0": False}
        self.saves_later(factory, settled, "s1")
        request = same_reply(("create_agent", {"name": "sweeper"}), ("install_agent", {"name": "sweeper"}))
        result = NewAgentDelegation(set(), lambda: factory, settled.get).wrap_tool_call(request, self.handler)
        self.assertEqual(result.content, "done")
        self.assertEqual(self.handled, [])

    def test_the_wait_goes_on_while_one_creating_call_is_still_running(self):
        delegation.NEW_AGENT_WAIT = 0.3
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "sweeper"}), ("install_agent", {"name": "sweeper"}))
        result = NewAgentDelegation(set(), lambda: factory, {"s0": False}.get).wrap_tool_call(request, self.handler)
        self.assertEqual(result.content, "Error: sweeper was being created in this same step but isn't "
                                         "available yet; call list_agents, then task.")

    SAVED_NOT_RELOADED = ("Error: sweeper was saved, but the agents couldn't be reloaded in this step; "
                          "it can be used from the user's next message.")
    NOT_APPROVED = ("Error: creating sweeper was not approved, so it doesn't exist; don't call task for it "
                    "unless the user asks again.")

    def test_saved_but_not_in_the_factory_says_it_was_saved_but_not_reloaded(self):
        delegation.NEW_AGENT_WAIT = 5.0
        factory = Factory({"helper"})
        request = same_reply(("create_agent", {"name": "sweeper"}))
        started = time.monotonic()
        result = NewAgentDelegation(set(), lambda: factory, {"s0": True}.get).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual((result.status, result.content), ("error", self.SAVED_NOT_RELOADED))

    def test_a_sibling_create_answered_by_a_tool_message_is_not_waited_on(self):
        delegation.NEW_AGENT_WAIT = 5.0
        factory, settled = Factory({"helper"}), {}
        self.saves_later(factory, settled, "s1")
        request = same_reply(("create_agent", {"name": "sweeper"}), ("create_agent", {"name": "sweeper"}))
        request.state["messages"].append(ToolMessage(content="rejected", tool_call_id="s0", name="create_agent"))
        result = NewAgentDelegation(set(), lambda: factory, settled.get).wrap_tool_call(request, self.handler)
        self.assertEqual(result.content, "done")

    def test_every_creating_call_answered_ends_the_wait_at_once(self):
        delegation.NEW_AGENT_WAIT = 5.0
        request = same_reply(("create_agent", {"name": "sweeper"}))
        request.state["messages"].append(ToolMessage(content="rejected", tool_call_id="s0", name="create_agent"))
        started = time.monotonic()
        result = make_middleware(Factory({"helper"})).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual((result.status, result.content), ("error", self.NOT_APPROVED))
        self.assertEqual(self.handled, [])

    def test_the_last_lookup_finds_an_agent_that_appeared_as_the_call_settled(self):
        factory = Factory({"helper"})

        def settled(call_id):
            factory.names = frozenset({"helper", "sweeper"})  # appears right as the call settles
            return True

        request = same_reply(("create_agent", {"name": "sweeper"}))
        result = NewAgentDelegation(set(), lambda: factory, settled).wrap_tool_call(request, self.handler)
        self.assertEqual(result.content, "done")

    def test_async_path_ends_early_when_the_call_settles_false(self):
        delegation.NEW_AGENT_WAIT = 5.0
        factory, settled = Factory({"helper"}), {}
        self.settle_later(settled, "s0", False)
        request = same_reply(("create_agent", {"name": "sweeper"}))

        async def handler(request):
            raise AssertionError("must not be handled")

        started = time.monotonic()
        result = asyncio.run(NewAgentDelegation(set(), lambda: factory, settled.get)
                             .awrap_tool_call(request, handler))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((result.status, result.content), ("error", self.FAILED))

    def test_a_non_string_import_path_does_not_wait_or_crash(self):
        factory = Factory({"helper"})
        request = same_reply(("import_agent", {"path": ["/agents/sweeper.md"]}))
        started = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(result.content, "handled")

    def test_a_marketplace_import_does_not_wait(self):
        factory = Factory({"helper"})
        request = same_reply(("import_agent", {"path": "/agents/sweeper.md", "scope": "marketplace"}))
        started = time.monotonic()
        result = make_middleware(factory).wrap_tool_call(request, self.handler)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(result.content, "handled")

    def test_a_sibling_import_agent_counts(self):
        factory = Factory({"helper"})
        self.learns_later(factory)
        request = same_reply(("import_agent", {"path": "/agents/sweeper.md"}))
        target = make_middleware(factory)._target(request)
        self.assertIsNotNone(target)
        self.assertEqual(target[1], "sweeper")
        self.assertIs(target[0], factory)

    def test_two_tasks_for_the_same_new_agent_both_find_it(self):
        factory = Factory({"helper"})
        self.learns_later(factory)
        middleware = make_middleware(factory)
        request = same_reply(("create_agent", {"name": "sweeper"}))
        found = []
        threads = [threading.Thread(target=lambda: found.append(middleware._target(request))) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(found), 2)
        for target in found:
            self.assertIsNotNone(target)
            self.assertEqual(target[1], "sweeper")
            self.assertIs(target[0], factory)

    def test_async_path_waits_too(self):
        factory = Factory({"helper"})
        self.learns_later(factory)
        request = same_reply(("create_agent", {"name": "sweeper"}))
        middleware = make_middleware(factory)
        ticks = []

        async def handler(request):
            return ToolMessage(content="handled", tool_call_id="t1", name="task")

        async def ticker():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        async def run():
            task = asyncio.create_task(ticker())
            try:
                return await middleware.awrap_tool_call(request, handler)
            finally:
                task.cancel()

        result = asyncio.run(run())
        self.assertEqual(result.content, "done")
        self.assertGreater(len(ticks), 1, "the event loop kept running while the agent was awaited")


class RejectedCreate(unittest.TestCase):
    """Real graph: the user rejects the create_agent sibling; task must not wait for it."""

    def setUp(self):
        self.wait = delegation.NEW_AGENT_WAIT
        delegation.NEW_AGENT_WAIT = 5.0

    def tearDown(self):
        delegation.NEW_AGENT_WAIT = self.wait

    def test_a_rejected_create_makes_task_answer_at_once(self):
        @tool
        def create_agent_stub(name: str) -> str:
            """Create an agent."""
            return "created"
        create_agent_stub.name = "create_agent"

        factory = Factory({"helper"})  # never learns "sweeper"
        model = Scripted(responses=[
            calls(("create_agent", {"name": "sweeper"}, "c1"),
                  ("task", {"subagent_type": "sweeper", "description": "x"}, "t1")),
            AIMessage(content="ok")])
        parent = create_agent(
            model=model, tools=[create_agent_stub, task],
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"create_agent": True}),
                        NewAgentDelegation(set(), lambda: factory)],
            checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "parent"}}
        result = parent.invoke({"messages": [HumanMessage(content="go")]}, config)
        started = time.monotonic()
        result = parent.invoke(Command(resume={i.id: reject() for i in result["__interrupt__"]}), config)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(delegated_result(result["messages"]),
                         "Error: creating sweeper was not approved, so it doesn't exist; don't call task "
                         "for it unless the user asks again.")


class FailedRebuildDuringWait(unittest.TestCase):
    """The app's wiring: the wait polls the runtime, which rebuilds a waiting change."""

    def setUp(self):
        self.wait, self.poll = delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL
        delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL = 0.5, 0.02
        self.attempts, self.said, self.handled = [], [], []
        self.manager = SimpleNamespace(changed=True)

        def rebuild():
            self.attempts.append(1)
            raise RuntimeError("boom")

        self.runtime = Runtime(agent=None, chats=SimpleNamespace(agent=None), agent_manager=self.manager,
                               rebuild=rebuild, say=self.said.append, make_worker=Factory({"helper"}))
        self.middleware = NewAgentDelegation(set(), lambda: newest_worker_factory(self.runtime))

    def tearDown(self):
        delegation.NEW_AGENT_WAIT, delegation.NEW_AGENT_POLL = self.wait, self.poll

    def handler(self, request):
        self.handled.append(request)
        return ToolMessage(content="handled", tool_call_id="t1", name="task")

    def test_a_wait_attempts_one_rebuild_and_says_it_once(self):
        result = self.middleware.wrap_tool_call(same_reply(("create_agent", {"name": "sweeper"})), self.handler)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "Error: sweeper was being created in this same step but isn't "
                                         "available yet; call list_agents, then task.")
        self.assertEqual(self.handled, [])
        self.assertEqual(len(self.attempts), 1)
        self.assertEqual(len(self.said), 1)
        self.assertIn("It will be tried again after your next message.", self.said[0])

    def test_the_next_message_retries_a_failed_rebuild(self):
        self.middleware.wrap_tool_call(same_reply(("create_agent", {"name": "sweeper"})), self.handler)
        self.runtime.refresh(retry=True)  # what the REPL does after each message
        self.assertEqual(len(self.attempts), 2)

    def test_a_new_approved_change_retries_within_the_same_reply(self):
        self.middleware.wrap_tool_call(same_reply(("create_agent", {"name": "sweeper"})), self.handler)
        self.manager.changed = True  # another approved change arrives
        self.runtime.refresh()
        self.assertEqual(len(self.attempts), 2)


class RebuildWarnings(unittest.TestCase):
    """refresh() says a failed rebuild once per distinct error, and again after a good rebuild."""

    def setUp(self):
        self.said = []
        self.manager = SimpleNamespace(changed=False)
        self.outcomes = []

        def rebuild():
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return "agent", SimpleNamespace(), Factory()

        self.runtime = Runtime(agent=None, chats=SimpleNamespace(agent=None), agent_manager=self.manager,
                               rebuild=rebuild, say=self.said.append)

    def approve_and_refresh(self, outcome):
        self.outcomes.append(outcome)
        self.manager.changed = True
        self.runtime.refresh()

    def test_the_same_error_warns_once(self):
        self.approve_and_refresh(RuntimeError("boom"))
        self.approve_and_refresh(RuntimeError("boom"))
        self.assertEqual(len(self.said), 1)

    def test_a_different_error_warns_again(self):
        self.approve_and_refresh(RuntimeError("boom"))
        self.approve_and_refresh(RuntimeError("bang"))
        self.assertEqual(len(self.said), 2)
        self.assertIn("bang", self.said[1])

    def test_a_good_rebuild_resets_the_warning(self):
        self.approve_and_refresh(RuntimeError("boom"))
        self.approve_and_refresh("ok")
        self.approve_and_refresh(RuntimeError("boom"))
        self.assertEqual(len(self.said), 2)


if __name__ == "__main__":
    unittest.main()

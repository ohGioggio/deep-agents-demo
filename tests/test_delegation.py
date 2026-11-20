"""A task for an agent made in the same message, driven by real graphs (no runtime).

The parent is a create_agent with ``NewAgentDelegation`` and a stub ``task`` tool; the
delegated agent gets an approval-gated ``remove_file``. A scripted model answers both.
"""
import asyncio
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

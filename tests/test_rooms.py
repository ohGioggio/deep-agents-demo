"""Rooms: a moderated discussion between agents, with the user able to step in."""
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.agent import worker_factory
from deep_agents_demo.approvals import Approver
from deep_agents_demo.chat import stream_turn
from test_agents import ALL_AGENT, SEEN, Model, agent_from, call
from test_orchestration import Boom, Workspace, reply


def chat_threads(root, chat_suffix):
    """A SQLite store with a chat thread and one extra thread, and the Chats over it."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    graph = StateGraph(MessagesState)
    graph.add_node("n", lambda state: {"messages": [reply("hi")]})
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return graph, SqliteSaver.from_conn_string(str(root / "c.sqlite"))


class CleanupTests(Workspace):
    def test_deleting_a_chat_removes_its_room_threads(self):
        from deep_agents_demo.threads import Chats, thread_id
        graph, saver_cm = chat_threads(self.root, ":room:")
        with saver_cm as saver:
            compiled = graph.compile(checkpointer=saver)
            chat = thread_id("alice", "c1")
            for tid in (chat, f"{chat}:room:abc:1:critic"):
                compiled.invoke({"messages": [{"role": "user", "content": "x"}]},
                                {"configurable": {"thread_id": tid}})
            Chats(saver, compiled, "alice").delete("c1")
            left = saver.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()

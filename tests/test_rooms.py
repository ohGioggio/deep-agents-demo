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


class StartRoomToolTests(Workspace):
    def turn(self, *responses, runners=None, agents=()):
        from deep_agents_demo.agent import build_agent
        from deep_agents_demo.orchestration import orchestration_feature
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), self.backend(), [orchestration_feature()], agents=agents)
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=Approver(interactive=False), runners=runners)
        return [m for kind, m in SEEN if kind == "messages"]

    def results(self, calls):
        return [str(m.content) for m in calls[-1] if m.type == "tool"]

    def test_start_room_pauses_and_returns_the_runner_answer(self):
        asked = []

        def runner(request):
            asked.append(request)
            return {"summary": "agreed"}
        calls = self.turn(call("start_room", {"question": "q", "participants": ["critic", "helper"]}, "r"),
                          runners={"start_room": runner})
        self.assertEqual(asked, [{"type": "start_room", "question": "q", "participants": ["critic", "helper"]}])
        self.assertEqual(self.results(calls), [json.dumps({"summary": "agreed"})])

    def test_room_needs_two_participants_and_no_moderator(self):
        asked = []
        for participants in (["critic"], ["critic", "moderator"], ["critic", "critic"]):
            with self.subTest(participants=participants):
                SEEN.clear()
                calls = self.turn(call("start_room", {"question": "q", "participants": participants}, "r"),
                                  runners={"start_room": asked.append})
                self.assertIn("participants", self.results(calls)[0])
        self.assertEqual(asked, [])

    def test_without_a_runner_the_room_is_refused(self):
        calls = self.turn(call("start_room", {"question": "q", "participants": ["a", "b"]}, "r"))
        self.assertIn("can't run here", self.results(calls)[0])

    def test_subagents_cannot_start_rooms(self):
        calls = self.turn(call("task", {"description": "go", "subagent_type": "helper"}, "t"),
                          call("start_room", {"question": "q", "participants": ["a", "b"]}, "r"),
                          AIMessage(content="sub done"), agents=[agent_from(ALL_AGENT)])
        sub = [m for m in calls if "Help." in str(m[0].content)]
        self.assertTrue(any("isn't available to this agent" in str(x.content) for x in sub[-1] if x.type == "tool"))


if __name__ == "__main__":
    unittest.main()

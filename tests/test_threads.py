"""Chat management: real SQLite checkpoints, the real agent graph, a fake model."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.chat import ChatSession
from deep_agents_demo.core.turns import turn_input
from deep_agents_demo.persistence.chats import Chats, new_name, thread_id, valid_name
from deep_agents_demo.ui.cli.render import format_table


class ToolFreeFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ChatsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "workspace/skills").mkdir(parents=True)
        self.saver_cm = SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite"))
        self.saver = self.saver_cm.__enter__()
        model = ToolFreeFakeModel(responses=[AIMessage(content=f"answer {i}") for i in range(20)])
        self.agent = build_agent(model, self.saver,
                                 FilesystemBackend(root_dir=str(root / "workspace"), virtual_mode=True))
        self.chats = Chats(self.saver, self.agent, "alice")

    def tearDown(self):
        self.saver_cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def say(self, user, name, *prompts):
        for prompt in prompts:
            self.agent.invoke(turn_input(prompt), {"configurable": {"thread_id": thread_id(user, name)}})


class ChatsTests(ChatsTestCase):
    def test_lists_only_this_users_chats_newest_first(self):
        self.say("alice", "parks", "plan a park", "add pine trees")
        self.say("bob", "secret-project", "hello")
        self.say("alice", "roads", "plan a road")
        with self.saver.cursor() as cur:  # a thread not created by this CLI
            cur.execute("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) "
                        "VALUES ('not-json', '', 'x')")
        chats = self.chats.list()
        self.assertEqual([c.name for c in chats], ["roads", "parks"])
        parks = chats[1]
        self.assertEqual((parks.turns, parks.first, parks.last), (2, "plan a park", "add pine trees"))

    def test_transcript_has_user_messages_and_final_replies(self):
        self.say("alice", "parks", "plan a park")
        self.assertEqual(self.chats.transcript("parks"), [("You", "plan a park"), ("Agent", "answer 0")])

    def test_rename_moves_history_and_validates(self):
        self.say("alice", "parks", "plan a park")
        self.say("alice", "roads", "plan a road")
        self.chats.rename("parks", "green-spaces")
        self.assertFalse(self.chats.exists("parks"))
        self.assertEqual(self.chats.transcript("green-spaces")[0], ("You", "plan a park"))
        for old, new in (("roads", "green-spaces"), ("missing", "x"), ("roads", "bad name")):
            with self.assertRaises(ValueError):
                self.chats.rename(old, new)

    def test_delete_removes_only_that_chat(self):
        self.say("alice", "parks", "plan a park")
        self.say("alice", "roads", "plan a road")
        self.chats.delete("parks")
        self.assertEqual(self.chats.names(), ["roads"])

    def test_names_and_table(self):
        self.assertTrue(valid_name("chat-2026.v2_a"))
        for bad in ("", "-leading", "has space", "x" * 65, "../up"):
            self.assertFalse(valid_name(bad), bad)
        self.assertEqual(new_name(datetime(2026, 11, 10, 9, 5, 7)), "chat-20261110-090507")
        self.assertIn("No chats yet", format_table([]))


class ChatSessionTests(ChatsTestCase):
    def session(self, answers=()):
        self.out = []
        answers = iter(answers)
        return ChatSession(self.chats, "parks", ask=lambda _: next(answers), say=self.out.append)

    def test_plain_text_is_not_a_command(self):
        self.assertFalse(self.session().handle("plan a park"))

    def test_new_and_switch(self):
        self.say("alice", "roads", "plan a road")
        s = self.session()
        s.reload_skills = False
        s.handle("/new")
        self.assertTrue(s.current.startswith("chat-"))
        self.assertTrue(s.reload_skills, "a new chat must reload skills on its first turn")
        s.handle("/threads")
        self.assertIn(f"*  {s.current} (new: saved after your first message)", self.out[-1])
        s.handle("/new roads")
        self.assertIn("already exists", self.out[-1])
        s.handle("/switch nowhere")
        self.assertIn("No chat named 'nowhere'", self.out[-1])
        s.handle("/switch roads")
        self.assertEqual(s.current, "roads")
        self.assertIn("1 turns", self.out[-1])
        self.assertEqual(s.config["configurable"]["thread_id"], thread_id("alice", "roads"))

    def test_delete_asks_first(self):
        self.say("alice", "roads", "plan a road")
        self.session(answers=["n"]).handle("/delete roads")
        self.assertTrue(self.chats.exists("roads"))
        self.session(answers=["y"]).handle("/delete roads")
        self.assertFalse(self.chats.exists("roads"))

    def test_rename_history_threads_and_unknown(self):
        self.say("alice", "parks", "plan a park")
        s = self.session()
        s.handle("/rename green")
        self.assertEqual(s.current, "green")
        s.handle("/history")
        self.assertIn("You: plan a park", self.out[-1])
        s.handle("/threads")
        self.assertIn("*  green", self.out[-1])
        s.handle("/frobnicate")
        self.assertIn("Unknown command", self.out[-1])


if __name__ == "__main__":
    unittest.main()

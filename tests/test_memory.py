"""Memory lifecycle tests: tools, write policy, user scoping and automatic recall."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.agent import build_agent, system_prompt
from deep_agents_demo.memory import AutoRecallMiddleware, memory_tools
from deep_agents_demo.memory_policy import find_secrets, redact

KEY = "sk-proj-abcDEF1234567890ghijKLMNOP"
HEADER = "## Recalled memories about this user"


class FakeMemory:
    """Just enough of mem0.Memory: per-user storage and word-overlap search."""

    def __init__(self, rewrite=None):
        self.items, self.ids, self.rewrite = {}, itertools.count(1), rewrite
        self.add_calls = self.search_calls = 0

    def seed(self, user_id, text):
        memory_id = f"m{next(self.ids)}"
        self.items[memory_id] = {"id": memory_id, "memory": text, "user_id": user_id}
        return memory_id

    def add(self, messages, user_id):
        self.add_calls += 1
        text = messages[-1]["content"]
        text = self.rewrite(text) if self.rewrite else text
        memory_id = self.seed(user_id, text)
        return {"results": [{"id": memory_id, "memory": text, "event": "ADD"}]}

    def search(self, query, filters, top_k):
        self.search_calls += 1
        words = set(query.lower().split())
        hits = [dict(i, score=0.9) for i in self.items.values()
                if i["user_id"] == filters["user_id"] and words & set(i["memory"].lower().split())]
        return {"results": hits[:top_k]}

    def get_all(self, filters, top_k):
        return {"results": [i for i in self.items.values() if i["user_id"] == filters["user_id"]][:top_k]}

    def get(self, memory_id):
        return self.items.get(memory_id)

    def delete(self, memory_id):
        del self.items[memory_id]


def tools_for(memory, user="alice", recall_tool=True):
    return {t.__name__: t for t in memory_tools(memory, user, recall_tool=recall_tool)}


class WritePolicyTests(unittest.TestCase):
    def test_secrets_are_refused_before_reaching_mem0(self):
        mem = FakeMemory()
        result = json.loads(tools_for(mem)["remember"](f"My OpenAI key is {KEY}"))
        self.assertFalse(result["stored"])
        self.assertIn("OpenAI or Anthropic API key", result["detected"])
        self.assertNotIn(KEY, json.dumps(result))
        self.assertEqual(mem.add_calls, 0, "a secret must never be sent to Mem0")

    def test_secret_rewrites_from_mem0_are_deleted(self):
        mem = FakeMemory(rewrite=lambda text: f"User's password is hunter2 ({text})")
        result = json.loads(tools_for(mem)["remember"]("I like pine trees"))
        self.assertEqual(result["removed_as_secret"], 1)
        self.assertFalse(result["stored"])
        self.assertEqual(mem.items, {})

    def test_plain_facts_are_stored(self):
        mem = FakeMemory()
        result = json.loads(tools_for(mem)["remember"]("I prefer pine trees and use uv"))
        self.assertTrue(result["stored"])
        self.assertEqual(result["results"][0]["memory"], "I prefer pine trees and use uv")

    def test_detector_and_redaction(self):
        self.assertTrue(find_secrets("password: Tr0ub4dor&3"))
        self.assertTrue(find_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"))
        self.assertTrue(find_secrets("card 4111 1111 1111 1111"))
        for safe in ("I prefer pine trees", "The password manager I use is Bitwarden",
                     "my token budget is small", "Order 1234-5678 shipped"):
            self.assertEqual(find_secrets(safe), [], safe)
        self.assertEqual(redact(f"key {KEY} here"), "key [REDACTED] here")
        self.assertNotIn("MIIEow", redact("-----BEGIN RSA PRIVATE KEY-----\nMIIEow"))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.mem = FakeMemory()
        self.pine = self.mem.seed("alice", "Prefers pine trees")
        self.uv = self.mem.seed("alice", "Uses uv for Python")
        self.bobs = self.mem.seed("bob", "Prefers oak trees")
        self.legacy = self.mem.seed("alice", f"Old note with key {KEY}")
        self.tools = tools_for(self.mem)

    def test_list_is_scoped_and_redacted(self):
        listed = json.loads(self.tools["list_memories"]())
        self.assertEqual(listed["count"], 3)
        text = json.dumps(listed)
        self.assertNotIn("oak", text)
        self.assertNotIn(KEY, text)
        self.assertIn("[REDACTED]", text)

    def test_recall_is_scoped_and_redacted(self):
        hits = json.loads(self.tools["recall"]("trees key"))
        self.assertEqual({h["id"] for h in hits}, {self.pine, self.legacy})
        self.assertNotIn(KEY, json.dumps(hits))

    def test_forget_deletes_own_memory(self):
        result = json.loads(self.tools["forget"](self.pine))
        self.assertTrue(result["forgotten"])
        self.assertNotIn(self.pine, self.mem.items)

    def test_forget_refuses_other_users_and_unknown_ids_alike(self):
        other = json.loads(self.tools["forget"](self.bobs))
        missing = json.loads(self.tools["forget"]("m999"))
        self.assertFalse(other["forgotten"])
        self.assertEqual(other, missing, "must not reveal that another user's id exists")
        self.assertIn(self.bobs, self.mem.items)

    def test_auto_mode_has_no_recall_tool(self):
        self.assertEqual(sorted(tools_for(self.mem, recall_tool=False)),
                         ["forget", "list_memories", "remember"])

    def test_prompt_matches_recall_mode(self):
        self.assertIn("Use recall", system_prompt(use_mem0=True, recall_mode="tool"))
        self.assertIn("There is no recall tool", system_prompt(use_mem0=True, recall_mode="auto"))
        self.assertIn("Never store secrets", system_prompt(use_mem0=True, recall_mode="both"))


class RecordingModel(FakeMessagesListChatModel):
    """Answers without tools and records the system prompt of every call."""
    systems: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.systems.append(next((m for m in messages if m.type == "system"), None))
        return super()._generate(messages, stop, run_manager, **kwargs)


class AutoRecallTests(unittest.TestCase):
    def run_agent(self, mem, prompt):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "skills").mkdir()
            model = RecordingModel(responses=[AIMessage(content="ok")] * 4, systems=[])
            agent = build_agent(model, InMemorySaver(),
                                FilesystemBackend(root_dir=str(root), virtual_mode=True),
                                memory_tools(mem, "alice", recall_tool=False), use_mem0=True,
                                recall_mode="auto",
                                extra_middleware=[AutoRecallMiddleware(mem, "alice")])
            config = {"configurable": {"thread_id": "t"}}
            agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)
            history = agent.get_state(config).values["messages"]
            return model.systems, history

    def system_text(self, message):
        content = message.content
        return content if isinstance(content, str) else json.dumps(content)

    def test_relevant_memories_reach_the_model_but_not_the_history(self):
        mem = FakeMemory()
        mem.seed("alice", "Prefers pine trees")
        mem.seed("bob", "Prefers oak trees")
        systems, history = self.run_agent(mem, "Plan a park with trees")
        prompt = self.system_text(systems[0])
        self.assertIn(HEADER, prompt)
        self.assertIn("Prefers pine trees", prompt)
        self.assertNotIn("oak", prompt)
        self.assertFalse(any(HEADER in str(m.content) for m in history))
        self.assertNotIn("<agent_memory>", prompt, "Mem0 mode must not add file-based memory")

    def test_nothing_injected_without_matches_and_search_runs_once_per_message(self):
        mem = FakeMemory()
        mem.seed("alice", "Prefers pine trees")
        middleware = AutoRecallMiddleware(mem, "alice")
        self.assertEqual(middleware.recalled("budget question"), [])
        middleware.recalled("budget question")
        self.assertEqual(mem.search_calls, 1)
        systems, _ = self.run_agent(mem, "What is my budget?")
        self.assertNotIn(HEADER, self.system_text(systems[0]))


if __name__ == "__main__":
    unittest.main()

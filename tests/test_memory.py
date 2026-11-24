"""Memory lifecycle tests: tools, write policy, user scoping and automatic recall."""
import itertools
import json
import logging
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.core.secrets import find_secrets, redact
from deep_agents_demo.features.memory.feature import file_memory_feature, memory_feature
from deep_agents_demo.features.memory.recall import AutoRecallMiddleware
from deep_agents_demo.features.memory.service import MemoryService, WritePolicy
from deep_agents_demo.features.memory.tools import memory_tools
from deep_agents_demo.features.memory.updates import Operation

KEY = "sk-proj-abcDEF1234567890ghijKLMNOP"
HEADER = "## Recalled memories about this user"


class FakeMemory:
    """Just enough of mem0.Memory: per-user storage and word-overlap search."""

    def __init__(self, rewrite=None):
        self.items, self.ids, self.rewrite = {}, itertools.count(1), rewrite
        self.add_calls = self.search_calls = 0
        self.infers = []

    def seed(self, user_id, text):
        memory_id = f"m{next(self.ids)}"
        self.items[memory_id] = {"id": memory_id, "memory": text, "user_id": user_id}
        return memory_id

    def add(self, messages, user_id, infer=True):
        self.add_calls += 1
        self.infers.append(infer)
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

    def update(self, memory_id, data):
        self.items[memory_id]["memory"] = data
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        del self.items[memory_id]


def tools_for(memory, user="alice", recall_tool=True):
    return {t.__name__: t for t in memory_tools(MemoryService(memory, user), recall_tool=recall_tool)}


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
        service = MemoryService(self.mem, "alice")
        prompt = {mode: memory_feature(service, mode).prompt for mode in ("tool", "auto", "both")}
        self.assertIn("unless you call recall", prompt["tool"])
        self.assertIn("There is no recall tool", prompt["auto"])
        self.assertIn("forget with an id from list_memories when", prompt["auto"])
        self.assertIn("Never store secrets", prompt["both"])

    def test_feature_matches_recall_mode(self):
        service = MemoryService(self.mem, "alice")
        auto = memory_feature(service, "auto")
        self.assertEqual([t.__name__ for t in auto.tools], ["remember", "list_memories", "forget"])
        self.assertIsInstance(auto.middleware[0], AutoRecallMiddleware)
        tool = memory_feature(service, "tool")
        self.assertIn("recall", [t.__name__ for t in tool.tools])
        self.assertEqual(tool.middleware, ())
        self.assertEqual(len(memory_feature(service, "both").middleware), 1)

    def test_file_memory_feature(self):
        feature = file_memory_feature("/instructions")
        self.assertEqual(feature.memory_sources, ("/instructions/memories/AGENTS.md",))
        self.assertIn("/instructions/memories/AGENTS.md", feature.prompt)


class MemoryServiceTests(unittest.TestCase):
    def test_policy_is_injectable(self):
        mem = FakeMemory()
        strict = WritePolicy(find_secrets=lambda text: ["phone number"] if "333" in text else [],
                             redact=lambda text: text.replace("333", "***"))
        service = MemoryService(mem, "alice", strict)
        refused = service.remember("Call me on 3331234567")
        self.assertEqual((refused["stored"], refused["detected"]), (False, ["phone number"]))
        self.assertEqual(mem.add_calls, 0)
        mem.seed("alice", "Old note: 333 is my desk")
        self.assertEqual(service.list()[0]["memory"], "Old note: *** is my desk")

    def test_service_returns_plain_data_scoped_to_its_user(self):
        mem = FakeMemory()
        mem.seed("alice", "Prefers pine trees")
        mem.seed("bob", "Prefers oak trees")
        alice = MemoryService(mem, "alice")
        self.assertEqual([m["memory"] for m in alice.list()], ["Prefers pine trees"])
        self.assertEqual([m["memory"] for m in alice.search("trees")], ["Prefers pine trees"])
        self.assertEqual(alice.search("trees")[0]["score"], 0.9)
        stored = alice.remember("Uses uv")
        self.assertTrue(stored["stored"])
        self.assertTrue(alice.forget(stored["results"][0]["id"])["forgotten"])

    def test_auto_recall_goes_through_the_service(self):
        mem = FakeMemory()
        mem.seed("alice", f"Trees note with key {KEY}")
        mem.seed("bob", "Trees are oak for bob")
        recalled = AutoRecallMiddleware(MemoryService(mem, "alice")).recalled("trees")
        self.assertEqual(recalled, ["Trees note with key [REDACTED]"])


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
                                [memory_feature(MemoryService(mem, "alice"), "auto")])
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
        middleware = AutoRecallMiddleware(MemoryService(mem, "alice"))
        self.assertEqual(middleware.recalled("budget question"), [])
        middleware.recalled("budget question")
        self.assertEqual(mem.search_calls, 1)
        systems, _ = self.run_agent(mem, "What is my budget?")
        self.assertNotIn(HEADER, self.system_text(systems[0]))


class FakeDecider:
    """Answers with a set Operation, or raises; records what it was shown."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def decide(self, fact, recent, candidates):
        self.calls.append((fact, recent, candidates))
        if self.error:
            raise self.error
        return self.answer


class FakeRuntime:
    def __init__(self, state):
        self.state = state


class RememberDecisionTests(unittest.TestCase):
    def setUp(self):
        self.mem = FakeMemory()
        self.pine = self.mem.seed("alice", "Lives in Rome")
        self.bobs = self.mem.seed("bob", "Lives in Rome")

    def service(self, decider=None, mem=None):
        return MemoryService(mem or self.mem, "alice", decider=decider)

    def test_without_a_decider_the_text_is_added_without_extraction(self):
        result = self.service().remember("Likes green tea")
        self.assertEqual((result["action"], result["stored"]), ("added", True))
        self.assertEqual(self.mem.infers, [False])

    def test_add_stores_the_text_without_mem0_extraction(self):
        decider = FakeDecider(Operation(kind="ADD", reason="new topic"))
        result = self.service(decider).remember("Has a dog named Rex")
        self.assertEqual((result["action"], result["stored"]), ("added", True))
        self.assertEqual(result["results"][0]["memory"], "Has a dog named Rex")
        self.assertEqual(self.mem.infers, [False])
        self.assertEqual(result["reason"], "new topic")

    def test_update_changes_the_candidate_and_reports_before_and_after(self):
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=self.pine, text="Lives in Milan",
                                        reason="moved"))
        result = self.service(decider).remember("Now lives in Milan")
        self.assertEqual(result, {"stored": True, "action": "updated", "id": self.pine,
                                  "before": "Lives in Rome", "after": "Lives in Milan",
                                  "reason": "moved"})
        self.assertEqual(self.mem.items[self.pine]["memory"], "Lives in Milan")
        self.assertEqual(self.mem.add_calls, 0)

    def test_none_stores_nothing_and_names_the_memory(self):
        decider = FakeDecider(Operation(kind="NONE", memory_id=self.pine, reason="same"))
        result = self.service(decider).remember("Resides in Rome")
        self.assertEqual(result, {"stored": False, "action": "already known", "id": self.pine,
                                  "memory": "Lives in Rome", "reason": "same"})
        self.assertEqual(self.mem.add_calls, 0)
        self.assertEqual(self.mem.items[self.pine]["memory"], "Lives in Rome")

    def test_none_without_an_id_still_stores_nothing(self):
        """Seen live: the model answered NONE for a repeat but named no memory. NONE changes
        nothing, so there is nothing to protect: no duplicate is added."""
        for memory_id in (None, "not-a-candidate"):
            with self.subTest(memory_id=memory_id):
                decider = FakeDecider(Operation(kind="NONE", memory_id=memory_id, reason="same"))
                result = self.service(decider).remember("Yes, I live in Rome")
                self.assertEqual((result["stored"], result["action"]), (False, "already known"))
                self.assertNotIn("id", result)
                self.assertEqual(self.mem.add_calls, 0)

    def test_delete_deletes_nothing_and_points_to_forget(self):
        decider = FakeDecider(Operation(kind="DELETE", memory_id=self.pine, reason="retracted"))
        result = self.service(decider).remember("I never lived in Rome")
        self.assertEqual(result["action"], "to forget")
        self.assertEqual((result["id"], result["memory"], result["stored"]),
                         (self.pine, "Lives in Rome", False))
        self.assertIn("forget", result["advice"])
        self.assertIn(self.pine, self.mem.items, "DELETE must not delete anything itself")

    def test_an_id_not_among_the_candidates_adds_the_fact(self):
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=self.bobs, text="Lives in Milan"))
        result = self.service(decider).remember("Now lives in Milan")
        self.assertEqual(result["action"], "added")
        self.assertIn("note", result)
        self.assertEqual(self.mem.items[self.bobs]["memory"], "Lives in Rome")
        self.assertEqual(self.mem.infers, [False])

    def test_empty_text_adds_the_fact_itself(self):
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=self.pine, text="  "))
        result = self.service(decider).remember("Moved to Milan")
        self.assertEqual(result["action"], "added")
        self.assertIn("note", result)
        self.assertEqual(result["results"][0]["memory"], "Moved to Milan")
        self.assertEqual(self.mem.items[self.pine]["memory"], "Lives in Rome")

    def test_a_failed_decision_adds_the_fact_and_says_so(self):
        decider = FakeDecider(error=RuntimeError("model timeout"))
        result = self.service(decider).remember("Likes green tea")
        self.assertEqual((result["action"], result["stored"]), ("added", True))
        self.assertIn("model timeout", result["note"])
        self.assertEqual(result["results"][0]["memory"], "Likes green tea")

    def test_no_memories_yet_adds_without_asking_the_decider(self):
        decider = FakeDecider(Operation(kind="NONE", memory_id="x"))
        result = self.service(decider, FakeMemory()).remember("Likes green tea")
        self.assertEqual(result["action"], "added")
        self.assertEqual(decider.calls, [])

    def test_a_secret_in_the_new_text_is_refused_and_nothing_written(self):
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=self.pine, text=f"Key is {KEY}"))
        result = self.service(decider).remember("Update my key")
        self.assertEqual(result, {"stored": False, "reason": "refused: the new text looks like a secret"})
        self.assertEqual(self.mem.items[self.pine]["memory"], "Lives in Rome")
        self.assertEqual(self.mem.add_calls, 0)

    def test_recent_messages_reach_the_decider_redacted(self):
        decider = FakeDecider(Operation(kind="ADD"))
        self.service(decider).remember("Has a dog", recent=[f"my key is {KEY}", "hello"])
        shown = decider.calls[0][1]
        self.assertEqual(shown, ["my key is [REDACTED]", "hello"])
        self.assertNotIn(KEY, json.dumps(decider.calls))

    def test_the_tool_passes_the_last_six_messages_and_works_without_any(self):
        decider = FakeDecider(Operation(kind="ADD"))
        tools = {t.__name__: t for t in memory_tools(self.service(decider))}
        messages = [HumanMessage("m0"), AIMessage("m1"), ToolMessage(content="out", tool_call_id="c")]
        messages += [HumanMessage(f"m{i}") for i in range(3, 10)]
        tools["remember"]("Has a dog", FakeRuntime({"messages": messages}))
        self.assertEqual(decider.calls[-1][1], ["m4", "m5", "m6", "m7", "m8", "m9"])
        tools["remember"]("Has a cat", FakeRuntime({}))
        tools["remember"]("Has a bird", None)
        self.assertEqual([call[1] for call in decider.calls[-2:]], [[], []])

    def test_the_runtime_is_not_part_of_the_tool_schema(self):
        tools = {t.__name__: t for t in memory_tools(self.service(FakeDecider()))}
        self.assertEqual(sorted(StructuredTool.from_function(tools["remember"]).args), ["fact"])

    def test_only_this_users_memories_are_candidates(self):
        decider = FakeDecider(Operation(kind="NONE", memory_id=self.pine))
        self.service(decider).remember("Resides in Rome")
        candidate_ids = [c["id"] for c in decider.calls[0][2]]
        self.assertEqual(candidate_ids, [self.pine])

    def test_add_stores_the_decided_text_and_a_blank_one_stores_the_fact(self):
        decider = FakeDecider(Operation(kind="ADD", text="Has a dog named Rex"))
        result = self.service(decider).remember("I have a dog called Rex")
        self.assertEqual(result["results"][0]["memory"], "Has a dog named Rex")
        decider = FakeDecider(Operation(kind="ADD", text="  "))
        result = self.service(decider).remember("Has a cat named Tom")
        self.assertEqual(result["results"][0]["memory"], "Has a cat named Tom")

    def test_a_secret_in_the_added_text_is_refused_and_nothing_written(self):
        decider = FakeDecider(Operation(kind="ADD", text=f"Key is {KEY}"))
        result = self.service(decider).remember("Store my key")
        self.assertEqual(result, {"stored": False, "reason": "refused: the new text looks like a secret"})
        self.assertEqual(self.mem.add_calls, 0)

    def test_the_target_past_the_hundredth_memory_is_still_a_candidate(self):
        for i in range(150):
            self.mem.seed("alice", f"Fact number {i} about hobbies")
        target = self.mem.seed("alice", "Lives in Rome")
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=target, text="Lives in Milan"))
        result = self.service(decider).remember("Now lives in Milan")
        self.assertIn(target, [c["id"] for c in decider.calls[0][2]])
        self.assertEqual((result["action"], self.mem.items[target]["memory"]),
                         ("updated", "Lives in Milan"))

    def test_a_failed_update_adds_the_fact_and_says_so(self):
        class FailingUpdate(FakeMemory):
            def update(self, memory_id, data):
                raise RuntimeError("database locked")

        mem = FailingUpdate()
        pine = mem.seed("alice", "Lives in Rome")
        decider = FakeDecider(Operation(kind="UPDATE", memory_id=pine, text="Lives in Milan"))
        result = self.service(decider, mem).remember("Now lives in Milan")
        self.assertEqual((result["action"], result["stored"]), ("added", True))
        self.assertIn(f"couldn't update {pine}", result["note"])
        self.assertIn("database locked", result["note"])
        self.assertIn("Tell the user", result["note"], "a fallback must reach the user, not only the agent")
        self.assertEqual(mem.infers, [False])
        self.assertEqual(mem.items[pine]["memory"], "Lives in Rome")

    def test_the_reason_is_redacted(self):
        decider = FakeDecider(Operation(kind="NONE", memory_id=self.pine, reason=f"same, key {KEY}"))
        result = self.service(decider).remember("Resides in Rome")
        self.assertNotIn(KEY, json.dumps(result))
        self.assertIn("[REDACTED]", result["reason"])


class StartupImportTests(unittest.TestCase):
    def test_startup_does_not_import_langchain_community(self):
        import subprocess
        import sys

        code = ("import sys, deep_agents_demo.startup.runtime; "
                "print('langchain_community' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


class OptionalExtrasNoticeTests(unittest.TestCase):
    def test_only_mem0_optional_extras_notices_are_silenced(self):
        import deep_agents_demo.features.memory.service  # noqa: F401  (installs the filters)
        for name, notice in (("mem0.utils.spacy_models", "Failed to load spaCy lemma model: x"),
                             ("mem0.vector_stores.qdrant", "fastembed not installed - BM25 keyword search disabled.")):
            with self.assertLogs(name, "WARNING") as logs:
                logging.getLogger(name).warning(notice)
                logging.getLogger(name).warning("Collection is corrupt")
            self.assertEqual([r.getMessage() for r in logs.records], ["Collection is corrupt"])


if __name__ == "__main__":
    unittest.main()

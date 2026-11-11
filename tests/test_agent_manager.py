"""Agents written through the chat: versioned store, planning and approval, marketplace."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo import app
from deep_agents_demo.agent import build_agent
from deep_agents_demo.agent_registry import AgentRegistry
from deep_agents_demo.app import prepare_workspace
from deep_agents_demo.approvals import Approver
from deep_agents_demo.chat import stream_turn
from deep_agents_demo.config import ChatSettings, MemorySettings, Paths, Settings

from deep_agents_demo.agent_files import parse_agent_md
from deep_agents_demo.agent_manager import AGENT_FILE, AgentManager
from deep_agents_demo.skill_store import AGENT_TABLES, SkillStore


class Workspace(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


class StoreTests(Workspace):
    def test_agent_tables_are_separate(self):
        path = self.root / "data.sqlite"
        skills, agents = SkillStore(path), SkillStore(path, AGENT_TABLES)
        self.addCleanup(skills.close)
        self.addCleanup(agents.close)
        skills.save("user", "alice", "x", {"SKILL.md": "skill"}, "create")
        agents.save("user", "alice", "x", {"agent.md": "agent v1"}, "create")
        agents.save("user", "alice", "x", {"agent.md": "agent v2"}, "update")
        self.assertEqual(skills.get("user", "alice", "x").files, {"SKILL.md": "skill"})
        self.assertEqual(agents.get("user", "alice", "x").files, {"agent.md": "agent v2"})
        self.assertEqual(len(skills.history("user", "alice", "x")), 1)
        self.assertEqual(len(agents.history("user", "alice", "x")), 2)


KNOWN = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
                   "write_todos", "task", "search_assets"})


class ManagerTests(Workspace):
    def setUp(self):
        super().setUp()
        self.store = SkillStore(self.root / "agents.sqlite", AGENT_TABLES)
        self.addCleanup(self.store.close)
        self.manager = AgentManager(self.store, "alice", KNOWN)

    def create(self, name="critic", **kwargs):
        change = self.manager.plan_create(name, "Finds weaknesses.", "You are a critic.", **kwargs)
        self.assertEqual(change.errors, [])
        self.manager.apply(change)
        return change

    def test_create_renders_a_claude_code_file(self):
        change = self.manager.plan_create("critic", "Finds weaknesses.", "You are a critic.",
                                          tools=["Read", "grep"])
        self.assertEqual(change.errors, [])
        agent, _ = parse_agent_md(change.after[AGENT_FILE], "t", KNOWN)
        self.assertEqual(agent.tools, ("read_file", "grep"))
        self.assertTrue(change.summary().startswith("Agent change requested: CREATE 'critic'"))
        self.assertIn("Tools it uses: read_file, grep", change.summary())

    def test_unknown_tool_is_an_error(self):
        change = self.manager.plan_create("critic", "d", "p", tools=["make_coffee"])
        self.assertTrue(any("make_coffee" in e for e in change.errors), change.errors)

    def test_credentials_are_errors_and_wording_a_warning(self):
        leaked = self.manager.plan_create("critic", "d", "Use key sk-" + "a" * 40)
        self.assertTrue(leaked.errors)
        worded = self.manager.plan_create("critic", "d", "The token is rotated every 90 days.")
        self.assertEqual(worded.errors, [])
        self.assertTrue(worded.warnings)

    def test_update_changes_only_given_fields(self):
        self.create(tools=["read_file"])
        change = self.manager.plan_update("critic", model="nvidia/other")
        changed = [l for l in change.diff().splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        self.assertEqual(sorted(changed), ["+model: nvidia/other"])
        self.manager.apply(change)
        cleared = self.manager.plan_update("critic", tools=[])
        self.assertNotIn("tools:", cleared.after[AGENT_FILE])

    def test_remove_and_restore_add_versions(self):
        self.create()
        v1 = self.store.get("user", "alice", "critic").files[AGENT_FILE]
        self.manager.apply(self.manager.plan_update("critic", description="Sharper."))
        self.manager.apply(self.manager.plan_remove("critic"))
        self.manager.apply(self.manager.plan_restore("critic", 1))
        self.assertEqual([v.action for v in self.store.history("user", "alice", "critic")],
                         ["create", "update", "remove", "restore"])
        self.assertEqual(self.store.get("user", "alice", "critic").files[AGENT_FILE], v1)

    def test_import_one_file_and_refuse_a_folder(self):
        folder = self.root / "claude-agents"
        folder.mkdir()
        (folder / "reviewer2.md").write_text(
            "---\nname: reviewer2\ndescription: Reviews.\nmodel: sonnet\n---\nReview.", encoding="utf-8")
        change = self.manager.plan_import(str(folder / "reviewer2.md"))
        self.assertEqual((change.errors, change.name), ([], "reviewer2"))
        self.assertTrue(any("sonnet" in w for w in change.warnings), change.warnings)
        refused = self.manager.plan_import(str(folder))
        self.assertTrue(any("reviewer2.md" in e and "one at a time" in e for e in refused.errors), refused.errors)

    def test_apply_marks_changed_and_rejected_does_not(self):
        bad = self.manager.plan_create("critic", "d", "p", tools=["make_coffee"])
        self.assertFalse(self.manager.apply(bad)["saved"])
        self.assertFalse(self.manager.changed)
        self.create()
        self.assertTrue(self.manager.changed)

    def test_definitions_give_the_users_agents(self):
        self.create()
        installed, own, notes = self.manager.definitions()
        self.assertEqual((installed, notes), ([], []))
        self.assertEqual((own[0].name, own[0].source), ("critic", "yours"))


class MarketplaceTests(Workspace):
    def setUp(self):
        super().setUp()
        self.store = SkillStore(self.root / "agents.sqlite", AGENT_TABLES)
        self.addCleanup(self.store.close)
        self.alice = AgentManager(self.store, "alice", KNOWN)
        self.bob = AgentManager(self.store, "bob", KNOWN)
        self.alice.apply(self.alice.plan_create("critic", "Finds weaknesses.", "You are a critic."))
        self.alice.apply(self.alice.plan_publish("critic"))

    def test_publish_install_follow_latest(self):
        self.bob.apply(self.bob.plan_install("critic"))
        installed, own, _ = self.bob.definitions()
        self.assertEqual(([a.name for a in installed], own), (["critic"], []))
        self.assertEqual(installed[0].source, "marketplace")
        self.alice.apply(self.alice.plan_update("critic", description="Sharper critic."))
        self.alice.apply(self.alice.plan_publish("critic"))
        self.assertEqual(self.bob.definitions()[0][0].description, "Sharper critic.")

    def test_install_is_refused_twice_and_for_unknown_names(self):
        self.bob.apply(self.bob.plan_install("critic"))
        self.assertIn("already installed", self.bob.plan_install("critic").errors[0])
        unknown = self.bob.plan_install("nope").errors[0]
        self.assertIn("no agent named 'nope'", unknown)
        self.assertIn("browse_agents", unknown)

    def test_removed_from_marketplace(self):
        self.bob.apply(self.bob.plan_install("critic"))
        self.alice.apply(self.alice.plan_remove("critic", scope="marketplace"))
        installed, _own, notes = self.bob.definitions()
        self.assertEqual(installed, [])
        self.assertTrue(any("removed from the marketplace" in n for n in notes), notes)
        row = next(r for r in self.bob.overview() if r["name"] == "critic")
        self.assertIn("removed from the marketplace", row["note"])

    def test_browse_filters_and_marks_installed(self):
        self.bob.apply(self.bob.plan_install("critic"))
        self.assertEqual([(r["name"], r["installed"]) for r in self.bob.browse("weak")], [("critic", True)])
        self.assertEqual(self.bob.browse("nothing-like-this"), [])
        self.assertEqual(self.bob.browse()[0]["description"], "Finds weaknesses.")


SEEN = []  # the messages of every model call


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        SEEN.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


CREATE_HELPER = call("create_agent", {"name": "helper", "description": "Helps with files.",
                                      "prompt": "You help with files."}, "c1")


class ToolTests(Workspace):
    def setUp(self):
        super().setUp()
        self.store = SkillStore(self.root / "agents.sqlite", AGENT_TABLES)
        self.addCleanup(self.store.close)
        self.manager = AgentManager(self.store, "alice", KNOWN)
        (self.root / "skills").mkdir()

    def run_turn(self, *responses, answers=(), interactive=True):
        model = Model(responses=[*responses, AIMessage(content="done")])
        backend = FilesystemBackend(root_dir=str(self.root), virtual_mode=True)
        agent = build_agent(model, InMemorySaver(), backend, [self.manager.feature()])
        replies = iter(answers)
        approver = Approver(self.manager.plan, ask=lambda _: next(replies), say=lambda *_: None,
                            interactive=interactive)
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}}, approver=approver)

    def test_agent_tools_need_approval(self):
        self.run_turn(CREATE_HELPER, answers=["r", ""])
        self.assertIsNone(self.store.get("user", "alice", "helper"))
        self.run_turn(CREATE_HELPER, interactive=False)
        self.assertIsNone(self.store.get("user", "alice", "helper"))
        self.run_turn(CREATE_HELPER, answers=["a"])
        self.assertIsNotNone(self.store.get("user", "alice", "helper"))

    def test_registry_precedence_with_database(self):
        folder = self.root / "agents"
        folder.mkdir()
        (folder / "critic.md").write_text("---\nname: critic\ndescription: From a file.\n---\nx", encoding="utf-8")
        self.manager.apply(self.manager.plan_create("critic", "Mine.", "You are a critic."))
        installed, own, _ = self.manager.definitions()
        registry = AgentRegistry.load([folder], KNOWN, installed=installed, own=own)
        self.assertEqual((registry.get("critic").source, registry.get("critic").description), ("yours", "Mine."))
        row = next(r for r in registry.overview() if r["name"] == "critic")
        self.assertEqual(row["overrides"], str(folder / "critic.md"))


class RuntimeTests(Workspace):
    def settings(self):
        return Settings(chat=ChatSettings(api_key="unused", model="unused", base_url=None),
                        memory=MemorySettings(enabled=False), rag_enabled=False,
                        paths=Paths(data=self.root / "data", workspace=self.root / "ws",
                                    checkpoints=self.root / "data" / "checkpoints.sqlite"))

    def runtime(self, model):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        settings = self.settings()
        prepare_workspace(settings)
        with patch.object(app, "chat_model", return_value=model):
            return app.build_runtime(settings, "alice", stack, say=lambda *_: None), settings

    def turn(self, runtime, answer="a"):
        with patch("builtins.print"), patch.object(app, "chat_model", return_value=self.model):
            stream_turn(runtime.agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=Approver(runtime.preview, ask=lambda _: answer, say=lambda *_: None))
            return runtime.refresh()

    def test_new_agent_is_used_from_the_next_message(self):
        SEEN.clear()
        self.model = Model(responses=[CREATE_HELPER, AIMessage(content="created"),
                                      call("task", {"description": "go", "subagent_type": "helper"}, "t1"),
                                      AIMessage(content="sub done"), AIMessage(content="done")])
        runtime, _ = self.runtime(self.model)
        self.assertTrue(self.turn(runtime))
        self.assertIsNotNone(runtime.agents.get("helper"))
        self.turn(runtime)
        self.assertTrue(any(m and m[0].type == "system" and "You help with files." in str(m[0].content)
                            for m in SEEN), "the new agent ran")

    def test_rejected_change_does_not_rebuild(self):
        self.model = Model(responses=[CREATE_HELPER, AIMessage(content="ok")])
        runtime, _ = self.runtime(self.model)
        before = runtime.agent
        self.assertFalse(self.turn(runtime, answer="r"))
        self.assertIs(runtime.agent, before)


if __name__ == "__main__":
    unittest.main()

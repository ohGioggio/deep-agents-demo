"""Agents written through the chat: versioned store, planning and approval, marketplace."""
import shutil
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

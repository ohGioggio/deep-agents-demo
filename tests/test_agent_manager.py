"""Agents written through the chat: versioned store, planning and approval, marketplace."""
import shutil
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

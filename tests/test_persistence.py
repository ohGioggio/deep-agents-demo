"""Storage shared by features: the versioned store, chats, safe folder names."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.persistence.files import safe_folder_name
from deep_agents_demo.persistence.versioned_store import AGENT_TABLES, VersionedStore

# The schema as skill_store.py created it before the rename (commit 10edf2d).
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL, name TEXT NOT NULL,
    current_version INTEGER NOT NULL, removed INTEGER NOT NULL DEFAULT 0,
    UNIQUE (scope, owner, name));
CREATE TABLE IF NOT EXISTS skill_versions (
    skill_id INTEGER NOT NULL REFERENCES skills(id), version INTEGER NOT NULL,
    action TEXT NOT NULL, files TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (skill_id, version));
CREATE TABLE IF NOT EXISTS installs (
    owner TEXT NOT NULL, name TEXT NOT NULL, installed_at TEXT NOT NULL,
    PRIMARY KEY (owner, name));
"""


class VersionedStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_store_from_before_the_rename_still_opens(self):
        path = self.root / "skills.sqlite"
        db = sqlite3.connect(path)
        db.executescript(OLD_SCHEMA)
        files = {"SKILL.md": "---\nname: x\n---\nbody"}
        db.execute("INSERT INTO skills VALUES (1, 'user', 'alice', 'x', 1, 0)")
        db.execute("INSERT INTO skill_versions VALUES (1, 1, 'create', ?, '2026-01-01')",
                   (json.dumps(files),))
        db.execute("INSERT INTO installs VALUES ('alice', 'shared', '2026-01-01')")
        db.commit()
        db.close()
        store = VersionedStore(path)
        self.assertEqual(store.get("user", "alice", "x").files, files)
        self.assertEqual(store.installed("alice"), ["shared"])
        self.assertEqual(store.save("user", "alice", "x", files, "update"), 2)
        store.close()

    def test_agent_tables_are_separate(self):
        path = self.root / "both.sqlite"
        skills, agents = VersionedStore(path), VersionedStore(path, AGENT_TABLES)
        skills.save("user", "a", "n", {"SKILL.md": "s"}, "create")
        self.assertIsNone(agents.get("user", "a", "n"))
        skills.close()
        agents.close()


class SafeFolderNameTests(unittest.TestCase):
    def test_safe_folder_name(self):
        self.assertEqual(safe_folder_name("a b/c"), "a_b_c")
        self.assertEqual(safe_folder_name(""), "_")
        self.assertEqual(safe_folder_name("tools-check.2"), "tools-check.2")


if __name__ == "__main__":
    unittest.main()

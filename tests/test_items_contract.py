"""One contract, two kinds of versioned item: skills and agents both behave as an
``ItemManager`` wherever the approval flow uses them (plan → approve → apply)."""
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.core.items import MARKETPLACE, USER, ItemChange, ItemManager
from deep_agents_demo.features.agents.manager import AgentManager
from deep_agents_demo.features.skills.manager import SkillManager
from deep_agents_demo.persistence.versioned_store import AGENT_TABLES, VersionedStore

KNOWN = frozenset({"ls", "read_file", "write_file", "grep", "glob", "write_todos", "task"})


class ItemManagerContract:
    """Mixed into a TestCase that defines ``make_manager(store, owner)``, ``tables`` and
    ``create_args(name)`` (keyword arguments for ``create_<noun>``)."""

    tables: tuple

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = VersionedStore(self.root / "items.sqlite", self.tables)
        self.manager = self.make_manager(self.store, "alice")

    def tearDown(self):
        self.store.close()

    def plan(self, verb, **args):
        return self.manager.plan(f"{verb}_{self.manager.noun}", args)

    def create(self, name="helper"):
        change = self.plan("create", **self.create_args(name))
        self.assertEqual(change.errors, [])
        return self.manager.apply(change)

    def test_is_an_item_manager(self):
        self.assertIsInstance(self.manager, ItemManager)
        self.assertIsInstance(self.plan("create", **self.create_args("x")), ItemChange)

    def test_create_then_update_adds_versions(self):
        self.assertEqual(self.create()["version"], 1)
        change = self.plan("update", name="helper", description="Now something else.")
        self.assertEqual((change.version_from, change.version_to, change.errors), (1, 2, []))
        self.assertEqual(self.manager.apply(change)["version"], 2)
        self.assertEqual([v.action for v in self.store.history(USER, "alice", "helper")],
                         ["create", "update"])

    def test_remove_and_restore(self):
        self.create()
        self.manager.apply(self.plan("remove", name="helper"))
        self.assertIsNone(self.manager._live("helper", USER))
        restored = self.manager.apply(self.plan("restore", name="helper", version=1))
        self.assertEqual(restored["version"], 3)
        self.assertIsNotNone(self.manager._live("helper", USER))

    def test_publish_install_follow_latest(self):
        self.create()
        self.manager.apply(self.plan("publish", name="helper"))
        bob = self.make_manager(self.store, "bob")
        bob.apply(bob.plan(f"install_{bob.noun}", {"name": "helper"}))
        self.assertEqual(self.store.installed("bob"), ["helper"])
        self.manager.apply(self.plan("update", name="helper", description="Better."))
        self.manager.apply(self.plan("publish", name="helper"))
        listed = self.store.get(MARKETPLACE, "", "helper")
        self.assertEqual(listed.version, 2)
        self.assertEqual([r["version"] for r in bob.overview() if r["name"] == "helper"], [2])

    def test_invalid_change_saves_nothing(self):
        change = self.plan("create", **self.create_args("Not A Valid Name!"))
        self.assertTrue(change.errors)
        self.assertEqual(self.manager.apply(change)["saved"], False)
        self.assertIsNone(self.store.get(USER, "alice", "Not A Valid Name!"))

    def test_plan_dispatch_by_tool_name(self):
        self.assertIsNone(self.manager.plan("ls", {}))
        bad = self.manager.plan(f"create_{self.manager.noun}", {"nonsense": 1})
        self.assertTrue(bad.errors[0].startswith("bad arguments"))


class SkillManagerContract(ItemManagerContract, unittest.TestCase):
    tables = ("skills", "skill_versions", "installs")

    def make_manager(self, store, owner):
        return SkillManager(store, owner, self.root / "workspace")

    def create_args(self, name):
        return {"name": name, "description": "Greets people.", "instructions": "Say hello."}


class AgentManagerContract(ItemManagerContract, unittest.TestCase):
    tables = AGENT_TABLES

    def make_manager(self, store, owner):
        return AgentManager(store, owner, KNOWN)

    def create_args(self, name):
        return {"name": name, "description": "Helps.", "prompt": "You help."}


if __name__ == "__main__":
    unittest.main()

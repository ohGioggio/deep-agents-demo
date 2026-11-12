"""The output store: files the agent delivered, per user and chat, with every version."""
import json
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.persistence.outputs import OutputStore


class OutputStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "outputs"
        self.store = OutputStore(self.root)

    def test_save_then_list_and_read(self):
        saved = self.store.save("alice", "parks", "plan.md", b"# Plan", "first")
        self.assertEqual((saved.name, saved.version, saved.bytes, saved.kind, saved.caption),
                         ("plan.md", 1, 6, "md", "first"))
        self.assertEqual(self.store.list("alice", "parks"), [saved])
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"# Plan")
        index = json.loads((self.store.folder("alice", "parks") / ".index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index), 1)
        self.assertEqual(saved.to_dict()["path"], str(saved.path))

    def test_same_name_twice_versions(self):
        self.store.save("alice", "parks", "plan.md", b"one")
        second = self.store.save("alice", "parks", "plan.md", b"two")
        self.assertEqual(second.version, 2)
        self.assertEqual(self.store.read("alice", "parks", "plan.md", version=1), b"one")
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"two")
        self.assertEqual([(f.name, f.version) for f in self.store.list("alice", "parks")], [("plan.md", 2)])
        with self.assertRaises(KeyError):
            self.store.read("alice", "parks", "other.md")

    def test_folders_are_safe(self):
        self.store.save("a b/c", "x/../y", "n.md", b"x")
        self.assertEqual(self.store.folder("a b/c", "x/../y"), self.root / "a_b_c" / "x_.._y")
        written = [p for p in self.root.parent.rglob("*") if p.is_file()]
        self.assertTrue(written and all(self.root in p.parents for p in written))

    def test_delete_and_rename_thread(self):
        self.store.save("alice", "old", "a.md", b"a")
        self.store.save("alice", "taken", "b.md", b"b")
        self.store.rename_thread("alice", "old", "new")
        self.assertFalse(self.store.folder("alice", "old").exists())
        self.assertEqual(self.store.read("alice", "new", "a.md"), b"a")
        with self.assertRaises(ValueError):
            self.store.rename_thread("alice", "new", "taken")
        self.assertEqual(self.store.read("alice", "new", "a.md"), b"a")
        self.assertEqual(self.store.read("alice", "taken", "b.md"), b"b")
        self.store.delete_thread("alice", "new")
        self.assertFalse(self.store.folder("alice", "new").exists())
        self.store.delete_thread("alice", "never")
        self.store.rename_thread("alice", "never", "nowhere")
        self.assertFalse(self.store.folder("alice", "nowhere").exists())


    def test_dot_names_stay_inside_the_store(self):
        for user, chat in (("u", ".."), ("..", "outputs"), (".", "."), ("u", "...")):
            with self.subTest(user=user, chat=chat):
                folder = self.store.folder(user, chat).resolve()
                self.assertIn(self.root.resolve(), folder.parents)
        (self.root.parent / "keep.txt").write_text("other data", encoding="utf-8")
        self.store.save("u", "x", "a.md", b"a")
        self.store.delete_thread("u", "..")
        self.store.delete_thread("..", "..")
        self.assertTrue((self.root.parent / "keep.txt").exists())
        self.assertEqual(self.store.read("u", "x", "a.md"), b"a")

    def test_a_file_named_index_json_keeps_its_content(self):
        saved = self.store.save("alice", "parks", "index.json", b'{"rows": 3}')
        self.assertEqual(saved.path.read_bytes(), b'{"rows": 3}')
        self.assertEqual(self.store.list("alice", "parks")[0].name, "index.json")

    def test_names_differing_only_in_case_are_one_file(self):
        self.store.save("alice", "parks", "Report.md", b"first")
        second = self.store.save("alice", "parks", "report.md", b"second")
        self.assertEqual((second.name, second.version), ("Report.md", 2))
        self.assertEqual(self.store.read("alice", "parks", "Report.md", version=1), b"first")
        self.assertEqual(self.store.read("alice", "parks", "report.md"), b"second")
        self.assertEqual(len(self.store.list("alice", "parks")), 1)

    def test_a_failed_save_leaves_nothing_behind(self):
        from unittest.mock import patch

        from deep_agents_demo.persistence import outputs
        self.store.save("alice", "parks", "d.csv", b"a,b\n1,2\n")
        real, calls = outputs._write, []

        def write(target, data):
            calls.append(target)
            if len(calls) == 2:  # the latest copy is locked (open in Excel)
                raise PermissionError("[WinError 5] Access is denied")
            real(target, data)
        with patch.object(outputs, "_write", side_effect=write), self.assertRaises(PermissionError):
            self.store.save("alice", "parks", "d.csv", b"a,b,c\n1,2,3\n")
        folder = self.store.folder("alice", "parks")
        self.assertEqual(sorted(p.name for p in (folder / ".versions").iterdir()), ["d.csv.v1"])
        self.assertEqual([f.version for f in self.store.list("alice", "parks")], [1])


if __name__ == "__main__":
    unittest.main()

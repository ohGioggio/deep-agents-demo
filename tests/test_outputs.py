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

    def test_plain_lowercase_names_keep_their_folder(self):
        self.assertEqual(self.store.folder("demo-user", "parks"), self.root / "demo-user" / "parks")

    def test_chats_differing_in_case_or_symbols_have_their_own_files(self):
        names = ("parks", "Parks", "PARKS", "a b", "a_b", "città", "citt_")
        for chat in names:
            self.store.save("alice", chat, "plan.md", chat.encode("utf-8"))
        for chat in names:
            with self.subTest(chat=chat):
                self.assertEqual(self.store.read("alice", chat, "plan.md"), chat.encode("utf-8"))
                self.assertEqual([f.version for f in self.store.list("alice", chat)], [1])

    def test_a_chat_keeps_its_files_when_the_look_alike_is_deleted(self):
        self.store.save("alice", "Parks", "plan.md", b"upper")
        self.store.save("alice", "parks", "plan.md", b"lower")
        self.store.delete_thread("alice", "Parks")
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"lower")
        self.store.delete_thread("alice", "parks")
        self.assertEqual([p.name for p in (self.root / "alice").iterdir() if p.is_dir()], [])

    def test_users_differing_in_case_have_their_own_files(self):
        self.store.save("alice", "parks", "plan.md", b"lower")
        self.store.save("Alice", "parks", "plan.md", b"upper")
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"lower")
        self.assertEqual(self.store.read("Alice", "parks", "plan.md"), b"upper")

    def test_files_saved_before_owners_stay_with_their_chat(self):
        old = self.root / "alice" / "My_Chat" / ".versions"  # "My Chat", saved by an older version
        old.mkdir(parents=True)
        (old / "x.md.v1").write_bytes(b"x")
        (old.parent / ".index.json").write_text(json.dumps([{
            "name": "x.md", "version": 1, "bytes": 1, "kind": "md", "caption": "",
            "saved_at": "2026-11-12T10:00:00+00:00"}]), encoding="utf-8")
        self.assertEqual(self.store.folder("alice", "My Chat"), old.parent)
        self.assertEqual(self.store.read("alice", "My Chat", "x.md"), b"x")
        self.store.save("alice", "My Chat", "y.md", b"y")  # now it is My Chat's for good
        self.assertNotEqual(self.store.folder("alice", "My_Chat"), old.parent)

    def test_a_renamed_chat_takes_its_folder_name_with_it(self):
        self.store.save("alice", "Old", "a.md", b"a")
        self.store.rename_thread("alice", "Old", "new")
        self.assertEqual(self.store.read("alice", "new", "a.md"), b"a")
        self.store.save("alice", "Old", "b.md", b"b")  # a new chat with the old name starts empty
        self.assertEqual([f.name for f in self.store.list("alice", "Old")], ["b.md"])

    def test_a_broken_index_is_rebuilt_from_the_versions(self):
        self.store.save("alice", "parks", "plan.md", b"one")
        self.store.save("alice", "parks", "plan.md", b"two")
        self.store.save("alice", "parks", "data.csv", b"a,b\n1,2\n")
        folder = self.store.folder("alice", "parks")
        (folder / ".index.json").write_text("{not json", encoding="utf-8")
        self.assertCountEqual([(f.name, f.version) for f in self.store.list("alice", "parks")],
                              [("plan.md", 2), ("data.csv", 1)])
        self.assertEqual(self.store.read("alice", "parks", "plan.md", version=1), b"one")
        third = self.store.save("alice", "parks", "plan.md", b"three")
        self.assertEqual(third.version, 3)
        self.assertEqual(len(json.loads((folder / ".index.json").read_text(encoding="utf-8"))), 4)
        self.assertEqual([p.read_text(encoding="utf-8") for p in folder.glob(".index.json.broken*")],
                         ["{not json"], "the broken index is kept aside, not lost")

    def test_an_index_of_the_wrong_shape_is_rebuilt(self):
        self.store.save("alice", "parks", "plan.md", b"one")
        folder = self.store.folder("alice", "parks")
        for broken in ('{"a": 1}', '[{"name": "plan.md"}]', "[1, 2]"):
            with self.subTest(broken=broken):
                (folder / ".index.json").write_text(broken, encoding="utf-8")
                self.assertEqual([(f.name, f.version) for f in self.store.list("alice", "parks")],
                                 [("plan.md", 1)])

    def test_captions_are_one_line(self):
        saved = self.store.save("alice", "parks", "plan.md", b"x", "the plan\n\nfor  the\tpark")
        self.assertEqual(saved.caption, "the plan for the park")

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

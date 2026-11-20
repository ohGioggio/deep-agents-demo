"""The output store: files the agent delivered, per user and chat, with every version."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_agents_demo.persistence.outputs import DIFF_MAX_LINES, OutputStore, change_summary


def numbered(count: int, changed: int | None = None) -> bytes:
    """``count`` lines "line 0" ...; line ``changed`` reads "changed N" instead."""
    return "".join(f"changed {i}\n" if i == changed else f"line {i}\n" for i in range(count)).encode()


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

    def test_a_delete_stopped_after_its_files_still_finds_a_colliding_folder(self):
        # "My Chat" and "My_Chat" share a safe folder: the second name gets its hashed one.
        self.store.save("alice", "My Chat", "a.md", b"a")
        self.store.save("alice", "My_Chat", "b.md", b"b")
        hashed = self.store.folder("alice", "My_Chat")
        self.assertNotEqual(hashed, self.store.folder("alice", "My Chat"))
        self.store.delete_thread("alice", "My Chat")  # the plain folder is free now
        # Fail the last step, after the files are gone: the folder must still be the
        # hashed one, so a retry removes it instead of looking at the free plain name.
        with patch.object(Path, "rmdir", side_effect=OSError("in use")):
            with self.assertRaises(OSError):
                self.store.delete_thread("alice", "My_Chat")
        self.assertTrue(hashed.exists())
        self.assertEqual(self.store.folder("alice", "My_Chat"), hashed)
        self.store.delete_thread("alice", "My_Chat")  # the retry
        self.assertFalse(hashed.exists())

    def test_a_rename_whose_owner_write_fails_leaves_the_files_where_they_were(self):
        self.store.save("alice", "parks", "plan.md", b"# Plan")
        with patch("deep_agents_demo.persistence.outputs._write", side_effect=OSError("in use")):
            with self.assertRaises(OSError):
                self.store.rename_thread("alice", "parks", "gardens")
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"# Plan")  # visible under the old name
        self.assertFalse((self.root / "alice" / "gardens").exists())

    def test_a_failed_owner_restore_does_not_hide_why_the_delete_stopped(self):
        self.store.save("alice", "parks", "plan.md", b"x")
        with patch.object(Path, "rmdir", side_effect=OSError("in use")),                 patch("deep_agents_demo.persistence.outputs._write", side_effect=OSError("restore denied")):
            with self.assertRaises(OSError) as raised:
                self.store.delete_thread("alice", "parks")
        self.assertEqual(str(raised.exception), "in use")

    def test_a_failed_move_back_does_not_hide_why_the_rename_stopped(self):
        self.store.save("alice", "parks", "plan.md", b"# Plan")
        real_rename = Path.rename

        def rename(path, target):
            if Path(path).name == "gardens":  # the move back
                raise OSError("move back denied")
            return real_rename(path, target)
        with patch("deep_agents_demo.persistence.outputs._write", side_effect=OSError("owner denied")),                 patch.object(Path, "rename", rename):
            with self.assertRaises(OSError) as raised:
                self.store.rename_thread("alice", "parks", "gardens")
        self.assertEqual(str(raised.exception), "owner denied")

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


class DiffTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "outputs"
        self.store = OutputStore(self.root)

    def test_default_diff_is_the_last_two_versions(self):
        self.store.save("alice", "parks", "plan.md", b"one\ntwo\nthree\n")
        self.store.save("alice", "parks", "plan.md", b"one\n2\nthree\n")
        self.store.save("alice", "parks", "plan.md", b"one\n2\nthree\nfour\n")
        text = self.store.diff("alice", "parks", "plan.md")
        self.assertIn("--- plan.md v2", text)
        self.assertIn("+++ plan.md v3", text)
        self.assertIn("+four", text)
        self.assertNotIn("-two", text)  # v1 is not compared by default

    def test_explicit_versions(self):
        self.store.save("alice", "parks", "plan.md", b"a\n")
        self.store.save("alice", "parks", "plan.md", b"b\n")
        self.store.save("alice", "parks", "plan.md", b"c\n")
        text = self.store.diff("alice", "parks", "plan.md", 1, 3)
        self.assertIn("-a", text)
        self.assertIn("+c", text)

    def test_identical_versions_say_so(self):
        self.store.save("alice", "parks", "plan.md", b"same\n")
        self.store.save("alice", "parks", "plan.md", b"same\n")
        self.assertEqual(self.store.diff("alice", "parks", "plan.md"), "(no changes between v1 and v2)")

    def test_an_image_is_not_diffed(self):
        self.store.save("alice", "parks", "map.png", b"\x89PNG")
        self.store.save("alice", "parks", "map.png", b"\x89PNG2")
        with self.assertRaisesRegex(ValueError, "can't diff an image"):
            self.store.diff("alice", "parks", "map.png")

    def test_a_missing_version_says_which_exist(self):
        for text in (b"1", b"2", b"3"):
            self.store.save("alice", "parks", "park-plan.md", text)
        with self.assertRaisesRegex(ValueError, r"park-plan\.md has no v4 \(versions: 1–3\)"):
            self.store.diff("alice", "parks", "park-plan.md", 2, 4)

    def test_one_version_has_nothing_to_compare(self):
        self.store.save("alice", "parks", "plan.md", b"only")
        with self.assertRaisesRegex(ValueError, "plan.md has only v1"):
            self.store.diff("alice", "parks", "plan.md")

    def test_an_unknown_file(self):
        with self.assertRaisesRegex(ValueError, "No file named 'nope.md'"):
            self.store.diff("alice", "parks", "nope.md")

    def test_change_summary_counts_changed_lines(self):
        self.assertEqual(change_summary(b"a\nb\nc\n", b"a\nB\nc\nd\n", 1), "changed from v1: +2 −1 lines")
        self.assertEqual(change_summary(b"x\n", b"x\n", 4), "changed from v4: +0 −0 lines")

    def test_only_v2_given_compares_it_with_the_version_before(self):
        for text in (b"one\n", b"two\n", b"three\n"):
            self.store.save("alice", "parks", "plan.md", text)
        text = self.store.diff("alice", "parks", "plan.md", None, 2)
        self.assertIn("--- plan.md v1", text)
        self.assertIn("+++ plan.md v2", text)
        self.assertNotIn("v3", text)

    def test_only_v2_given_as_the_first_version_is_refused(self):
        self.store.save("alice", "parks", "plan.md", b"one")
        self.store.save("alice", "parks", "plan.md", b"two")
        with self.assertRaisesRegex(ValueError, "no version before v1"):
            self.store.diff("alice", "parks", "plan.md", None, 1)


    def test_just_over_the_size_cap_gives_a_note_not_a_diff(self):
        self.store.save("alice", "parks", "big.md", numbered(DIFF_MAX_LINES + 1))
        self.store.save("alice", "parks", "big.md", numbered(DIFF_MAX_LINES + 1, changed=5))
        with patch("deep_agents_demo.persistence.outputs.difflib.SequenceMatcher") as matcher:
            text = self.store.diff("alice", "parks", "big.md")
        matcher.assert_not_called()  # no line-by-line comparison at all
        self.assertEqual(text, f"big.md v1 and v2 are too large to compare line by line (over {DIFF_MAX_LINES} lines).")
        self.assertNotIn("\n", text)

    def test_at_the_size_cap_gives_a_normal_diff(self):
        self.store.save("alice", "parks", "big.md", numbered(DIFF_MAX_LINES))
        self.store.save("alice", "parks", "big.md", numbered(DIFF_MAX_LINES, changed=5))
        text = self.store.diff("alice", "parks", "big.md")
        self.assertIn("-line 5", text.splitlines())
        self.assertIn("+changed 5", text)

    def test_change_summary_over_the_cap_says_so(self):
        self.assertEqual(change_summary(numbered(DIFF_MAX_LINES + 1), numbered(3), 2),
                         "changed from v2 (too large to compare line by line)")
        self.assertEqual(change_summary(numbered(3), numbered(DIFF_MAX_LINES + 1), 1),
                         "changed from v1 (too large to compare line by line)")

    def test_change_summary_at_the_cap_counts_lines(self):
        self.assertEqual(change_summary(numbered(DIFF_MAX_LINES), numbered(DIFF_MAX_LINES, changed=5), 1),
                         "changed from v1: +1 −1 lines")


if __name__ == "__main__":
    unittest.main()

"""Data checks for the orchestration eval tasks: every answer and defect must be grounded in the files."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import orchestration_tasks as ot  # noqa: E402


def _joined(task):
    return "\n".join(task.files.values())


class TaskDataTests(unittest.TestCase):
    def test_thirteen_tasks_in_four_families(self):
        self.assertEqual(len(ot.TASKS), 13)
        names = [t.name for t in ot.TASKS]
        self.assertEqual(len(set(names)), len(names))
        counts = Counter(t.family for t in ot.TASKS)
        self.assertEqual(dict(counts), {"lookup": 4, "decision": 4, "review": 4, "safety": 1})

    def test_lookup_answers_are_in_the_files(self):
        for task in ot.TASKS:
            if task.family != "lookup":
                continue
            text = _joined(task).replace(",", "")
            for value in task.expected:
                with self.subTest(task=task.name, value=value):
                    self.assertIn(value, text)

    def test_decision_options_and_constraint_are_in_the_files(self):
        for task in ot.TASKS:
            if task.family != "decision":
                continue
            text = _joined(task).lower()
            with self.subTest(task=task.name):
                for option in task.options:
                    self.assertIn(option.lower(), text)
                self.assertTrue(
                    any(keyword.lower() in text for keyword in task.constraint),
                    f"no constraint keyword of {task.constraint} in files",
                )
                self.assertTrue(set(task.allowed) <= set(task.options))
                self.assertTrue(set(task.ruled_out) <= set(task.options))
                self.assertFalse(set(task.allowed) & set(task.ruled_out))

    def test_review_defect_lines_exist(self):
        for task in ot.TASKS:
            if task.family != "review":
                continue
            lines = task.files[task.review_file].splitlines()
            with self.subTest(task=task.name):
                self.assertTrue(task.defects)
                for line_no in task.defects:
                    self.assertLessEqual(line_no, len(lines))
                    self.assertTrue(lines[line_no - 1].strip(), f"line {line_no} is empty")

    def test_decision_format_asks_for_the_deciding_fact(self):
        self.assertEqual(
            ot.FORMATS["decision"],
            "Say which fact decides it, then finish with a line FINAL ANSWER: <option name>, "
            "naming exactly one option.")

    def test_safety_marker_is_in_the_files(self):
        for task in ot.TASKS:
            if task.family != "safety":
                continue
            self.assertIn(task.marker, _joined(task))

    def test_seed_and_prompt(self):
        for task in ot.TASKS:
            with self.subTest(task=task.name), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                ot.seed(workspace, task)
                for path, content in task.files.items():
                    written = workspace / "eval" / task.name / path
                    data = written.read_bytes()
                    self.assertNotIn(b"\r\n", data)
                    self.assertEqual(data, content.encode("utf-8"))
                prompt = ot.full_prompt(task)
                self.assertTrue(prompt.startswith(f"The files for this task are in /eval/{task.name}/."))
                self.assertTrue(prompt.endswith(ot.FORMATS[task.family]))


if __name__ == "__main__":
    unittest.main()

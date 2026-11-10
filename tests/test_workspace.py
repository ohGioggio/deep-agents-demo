"""Workspace seeding and skill tests."""
import re
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.agent import system_prompt
from deep_agents_demo.workspace import SEED_DIR, seed_files

SKILLS = SEED_DIR / "skills"


class SeedTests(unittest.TestCase):
    def test_seeds_memory_and_every_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            seed_files(workspace)
            self.assertTrue((workspace / "memories/AGENTS.md").is_file())
            for name in ("scene-planning", "sandbox"):
                self.assertTrue((workspace / "skills" / name / "SKILL.md").is_file(), name)

    def test_existing_files_are_preserved_and_new_skills_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            edited = workspace / "skills/scene-planning/SKILL.md"
            edited.parent.mkdir(parents=True)
            edited.write_text("my edits", encoding="utf-8")
            seed_files(workspace)
            self.assertEqual(edited.read_text(encoding="utf-8"), "my edits")
            self.assertTrue((workspace / "skills/sandbox/SKILL.md").is_file())


class SkillFormatTests(unittest.TestCase):
    def test_skills_have_frontmatter_matching_their_folder(self):
        for skill in SKILLS.glob("*/SKILL.md"):
            # Windows checkouts may use CRLF line endings.
            text = skill.read_text(encoding="utf-8").replace("\r\n", "\n")
            match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
            self.assertIsNotNone(match, skill)
            self.assertEqual(match.group(1), skill.parent.name)

    def test_sandbox_skill_teaches_uv_instead_of_pip(self):
        text = (SKILLS / "sandbox/SKILL.md").read_text(encoding="utf-8")
        for expected in ("uv venv", "uv pip install", "uv run --with", 'matplotlib.use("Agg")',
                         "openshell policy set"):
            self.assertIn(expected, text)


class PromptTests(unittest.TestCase):
    def test_sandbox_skill_hint_only_for_openshell(self):
        self.assertIn("sandbox skill", system_prompt("/instructions", True, "/sandbox", True))
        self.assertNotIn("sandbox skill", system_prompt("/instructions", True, "/home/daytona"))
        self.assertNotIn("sandbox skill", system_prompt("", True, None, True))


if __name__ == "__main__":
    unittest.main()

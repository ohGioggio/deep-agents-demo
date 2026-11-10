"""Workspace seeding, skills and system prompt tests."""
import re
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.agent import system_prompt
from deep_agents_demo.workspace import SEED_DIR, seed_files

OLD_MEMORY = b"# User preferences\nUse metres. Prefer realistic assets.\n"


class SeedTests(unittest.TestCase):
    def test_seeds_memory_and_every_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            seed_files(workspace)
            self.assertTrue((workspace / "memories/AGENTS.md").is_file())
            self.assertTrue((workspace / "skills/scene-planning/SKILL.md").is_file())
            self.assertTrue((workspace / "sandbox-skills/sandbox/SKILL.md").is_file())
            self.assertFalse((workspace / "skills/sandbox").exists())

    def test_existing_files_are_preserved_and_new_skills_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            edited = workspace / "skills/scene-planning/SKILL.md"
            edited.parent.mkdir(parents=True)
            edited.write_text("my edits", encoding="utf-8")
            seed_files(workspace)
            self.assertEqual(edited.read_text(encoding="utf-8"), "my edits")
            self.assertTrue((workspace / "sandbox-skills/sandbox/SKILL.md").is_file())

    def test_moved_sandbox_skill_old_copy_removed_only_if_unedited(self):
        source = (SEED_DIR / "sandbox-skills/sandbox/SKILL.md").read_bytes()
        for content, kept in ((source, False), (b"my edits", True)):
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                old = workspace / "skills/sandbox/SKILL.md"
                old.parent.mkdir(parents=True)
                old.write_bytes(content)
                seed_files(workspace)
                self.assertEqual(old.exists(), kept)


class SupersededSeedTests(unittest.TestCase):
    def test_unedited_old_seeds_are_upgraded_even_with_crlf(self):
        for content in (OLD_MEMORY, OLD_MEMORY.replace(b"\n", b"\r\n")):
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                target = workspace / "memories/AGENTS.md"
                target.parent.mkdir(parents=True)
                target.write_bytes(content)
                seed_files(workspace)
                self.assertEqual(target.read_bytes(), (SEED_DIR / "memories/AGENTS.md").read_bytes())

    def test_edited_seeds_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "memories/AGENTS.md"
            target.parent.mkdir(parents=True)
            target.write_bytes(OLD_MEMORY + b"I also like oak.\n")
            seed_files(workspace)
            self.assertIn(b"I also like oak.", target.read_bytes())


class SkillFormatTests(unittest.TestCase):
    def test_skills_have_frontmatter_matching_their_folder(self):
        for skill in SEED_DIR.glob("*/*/SKILL.md"):
            # Windows checkouts may use CRLF line endings.
            text = skill.read_text(encoding="utf-8").replace("\r\n", "\n")
            match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
            self.assertIsNotNone(match, skill)
            self.assertEqual(match.group(1), skill.parent.name)

    def test_sandbox_skill_teaches_uv_instead_of_pip(self):
        text = (SEED_DIR / "sandbox-skills/sandbox/SKILL.md").read_text(encoding="utf-8")
        for expected in ("uv venv", "uv pip install", "uv run --with", 'matplotlib.use("Agg")',
                         "openshell policy set"):
            self.assertIn(expected, text)


class SkillSourceTests(unittest.TestCase):
    def test_sandbox_skills_offered_only_with_openshell(self):
        from unittest.mock import patch
        import deep_agents_demo.agent as agent_module
        for flag, expected in ((False, ["/skills/"]), (True, ["/skills/", "/sandbox-skills/"])):
            with patch.object(agent_module, "create_deep_agent") as create:
                agent_module.build_agent(None, None, backend=object(), sandbox_skill=flag)
                self.assertEqual(create.call_args.kwargs["skills"], expected)


class PromptTests(unittest.TestCase):
    def test_indexed_documents_are_named_and_searched_first(self):
        prompt = system_prompt(doc_sources=["street-works-brief.md"])
        self.assertIn("indexed documents are: street-works-brief.md", prompt)
        self.assertIn("call search_docs first", prompt)
        self.assertIn("Use search_docs for questions about the user's documents", system_prompt())

    def test_prompt_is_general_purpose(self):
        prompt = system_prompt()
        self.assertIn("general-purpose assistant", prompt)
        for scene_specific in ("3D", "scene", "catalog research", "asset-researcher"):
            self.assertNotIn(scene_specific, prompt)

    def test_researcher_gets_read_only_lookups(self):
        from unittest.mock import patch
        import deep_agents_demo.agent as agent_module

        def search_docs(query: str) -> str:
            """Search documents."""
            return ""

        def remember(fact: str) -> str:
            """Save a fact."""
            return ""

        with patch.object(agent_module, "create_deep_agent") as create:
            agent_module.build_agent(None, None, backend=object(), extra_tools=[search_docs, remember])
        kwargs = create.call_args.kwargs
        [researcher] = kwargs["subagents"]
        self.assertEqual(researcher["name"], "researcher")
        self.assertEqual([t.__name__ for t in researcher["tools"]], ["search_assets", "search_docs"])
        self.assertEqual([t.__name__ for t in kwargs["tools"]],
                         ["search_assets", "triangle_budget", "search_docs", "remember"])

    def test_sandbox_skill_hint_only_for_openshell(self):
        self.assertIn("sandbox skill", system_prompt("/instructions", True, "/sandbox", True))
        self.assertNotIn("sandbox skill", system_prompt("/instructions", True, "/home/daytona"))
        self.assertNotIn("sandbox skill", system_prompt("", True, None, True))


if __name__ == "__main__":
    unittest.main()

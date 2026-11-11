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
            match = re.match(r"---\nname: (.+)\ndescription: (.+)\n(?:allowed-tools: .+\n)?---\n", text)
            self.assertIsNotNone(match, skill)
            self.assertEqual(match.group(1), skill.parent.name)

    def test_sandbox_skill_teaches_uv_instead_of_pip(self):
        text = (SEED_DIR / "sandbox-skills/sandbox/SKILL.md").read_text(encoding="utf-8")
        for expected in ("uv venv", "uv pip install", "uv run --with", 'matplotlib.use("Agg")',
                         "openshell policy set"):
            self.assertIn(expected, text)


def _tool(name):
    def tool(query: str) -> str:
        return ""
    tool.__name__ = tool.__doc__ = name
    return tool


class FeatureAssemblyTests(unittest.TestCase):
    """build_agent only assembles what features contribute."""

    def build(self, features, **kwargs):
        from unittest.mock import patch

        import deep_agents_demo.agent as agent_module
        with patch.object(agent_module, "create_deep_agent") as create:
            agent_module.build_agent(None, None, object(), features, **kwargs)
        return create.call_args.kwargs

    def test_sandbox_skills_offered_only_with_openshell(self):
        from deep_agents_demo.sandbox import sandbox_feature
        for openshell, expected in ((False, ["/i/skills/"]), (True, ["/i/skills/", "/i/sandbox-skills/"])):
            kwargs = self.build([sandbox_feature("/sandbox", openshell)], skills_prefix="/i")
            self.assertEqual(kwargs["skills"], expected)

    def test_tools_middleware_and_researcher_come_from_features(self):
        from deep_agents_demo.catalog import catalog_feature
        from deep_agents_demo.core.feature import Feature
        search_docs, remember, marker = _tool("search_docs"), _tool("remember"), object()
        kwargs = self.build([catalog_feature(),
                             Feature("memory", tools=(remember,), middleware=(marker,)),
                             Feature("documents", tools=(search_docs,), lookups=(search_docs,))])
        self.assertEqual([t.__name__ for t in kwargs["tools"]],
                         ["search_assets", "triangle_budget", "remember", "search_docs"])
        general, researcher = kwargs["subagents"]
        self.assertEqual(general["name"], "general-purpose")
        self.assertNotIn("tools", general, "it inherits the main agent's tools")
        self.assertEqual([t.__name__ for t in researcher["tools"]], ["search_assets", "search_docs"])
        self.assertIn(marker, kwargs["middleware"])
        self.assertIsNone(kwargs["memory"], "no file memory unless a feature asks for it")

    def test_no_lookups_means_no_researcher(self):
        kwargs = self.build([])
        self.assertEqual([s["name"] for s in kwargs["subagents"]], ["general-purpose"])
        self.assertNotIn("researcher", kwargs["system_prompt"])

    def test_file_memory_feature_turns_on_file_memory(self):
        from deep_agents_demo.memory import file_memory_feature
        kwargs = self.build([file_memory_feature("/i")], skills_prefix="/i")
        self.assertEqual(kwargs["memory"], ["/i/memories/AGENTS.md"])


class PromptTests(unittest.TestCase):
    def test_sections_come_in_feature_order_between_identity_and_closing(self):
        from deep_agents_demo.agent import CLOSING, IDENTITY
        from deep_agents_demo.core.feature import Feature
        prompt = system_prompt([Feature("a", prompt="FIRST. "), Feature("b", prompt="SECOND. ")])
        self.assertTrue(prompt.startswith(IDENTITY))
        self.assertTrue(prompt.endswith(CLOSING))
        self.assertLess(prompt.index("FIRST."), prompt.index("SECOND."))

    def test_indexed_documents_are_named_and_searched_first(self):
        from deep_agents_demo.rag import documents_prompt
        prompt = documents_prompt(["street-works-brief.md"])
        self.assertIn("indexed documents are: street-works-brief.md", prompt)
        self.assertIn("call search_docs first", prompt)
        self.assertIn("Use search_docs for questions about the user's documents", documents_prompt([]))

    def test_prompt_is_general_purpose(self):
        from deep_agents_demo.catalog import catalog_feature
        prompt = system_prompt([catalog_feature()])
        self.assertIn("general-purpose assistant", prompt)
        for scene_specific in ("3D", "scene", "catalog research", "asset-researcher"):
            self.assertNotIn(scene_specific, prompt)

    def test_search_docs_is_only_mentioned_when_documents_are_on(self):
        self.assertNotIn("search_docs", system_prompt([]))

    def test_sandbox_skill_hint_only_for_openshell(self):
        from deep_agents_demo.sandbox import sandbox_feature
        self.assertIn("sandbox skill", sandbox_feature("/sandbox", openshell_skill=True).prompt)
        self.assertNotIn("sandbox skill", sandbox_feature("/home/daytona").prompt)
        self.assertIn("/home/daytona", sandbox_feature("/home/daytona").prompt)


if __name__ == "__main__":
    unittest.main()

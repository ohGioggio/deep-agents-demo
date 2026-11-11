"""Subagents: the Claude Code agent file format, the registry, and agents as subagents."""
import shutil
import tempfile
import unittest
from pathlib import Path

from deep_agents_demo.agent_files import parse_agent_md, render_agent_md, resolve_model, translate_tools
from deep_agents_demo.agent_registry import AgentRegistry
from deep_agents_demo.workspace import SEED_DIR

KNOWN = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
                   "write_todos", "task", "search_assets"})
CRITIC = ("---\nname: critic\ndescription: Finds weaknesses.\ntools: Read, Grep, Glob\nmodel: sonnet\n"
          "color: red\nmemory: project\n---\nYou are a critic.\n")


class FormatTests(unittest.TestCase):
    def test_parses_a_claude_code_file(self):
        agent, notes = parse_agent_md(CRITIC, "folder", KNOWN)
        self.assertEqual((agent.name, agent.description, agent.prompt),
                         ("critic", "Finds weaknesses.", "You are a critic."))
        self.assertEqual(agent.tools, ("read_file", "grep", "glob", "ls"))
        self.assertIsNone(agent.model)
        self.assertTrue(any("sonnet" in n for n in notes), notes)
        self.assertTrue(any("memory" in n for n in notes), notes)
        self.assertEqual(agent.color, "red")
        self.assertEqual(agent.source, "folder")

    def test_tools_as_list_crlf_and_bom(self):
        text = ("﻿---\r\nname: lister\r\ndescription: Lists.\r\ntools:\r\n  - LS\r\n"
                "  - read_file\r\n---\r\nList things.\r\n")
        agent, _ = parse_agent_md(text, "f", KNOWN)
        self.assertEqual(agent.tools, ("ls", "read_file"))
        self.assertEqual(agent.prompt, "List things.")

    def test_omitted_tools_means_all(self):
        agent, _ = parse_agent_md("---\nname: a\ndescription: d\n---\nx", "f", KNOWN)
        self.assertIsNone(agent.tools)

    def test_unknown_and_spawn_tools_are_warnings(self):
        names, notes = translate_tools(["Read", "NotebookEdit", "Agent", "mcp__github__x"], KNOWN)
        self.assertEqual(names, ("read_file",))
        self.assertEqual(len(notes), 3)

    def test_nothing_resolves_means_not_loaded(self):
        agent, notes = parse_agent_md("---\nname: a\ndescription: d\ntools: NotebookEdit\n---\nx", "f", KNOWN)
        self.assertIsNone(agent)
        self.assertTrue(notes)

    def test_invalid_files_are_skipped_with_a_reason(self):
        for text in ("no frontmatter", "---\nname: [oops\n---\nx",
                     "---\nname: bad:name\ndescription: d\n---\nx", "---\ndescription: d\n---\nx"):
            with self.subTest(text=text):
                agent, notes = parse_agent_md(text, "f.md", KNOWN)
                self.assertIsNone(agent)
                self.assertIn("f.md", notes[0])

    def test_models(self):
        self.assertEqual(resolve_model("inherit"), (None, None))
        self.assertEqual(resolve_model(None), (None, None))
        self.assertIsNone(resolve_model("claude-opus-5-5")[0])
        self.assertIsNotNone(resolve_model("claude-opus-5-5")[1])
        self.assertEqual(resolve_model("nvidia/other-model"), ("nvidia/other-model", None))

    def test_permission_modes(self):
        plan, _ = parse_agent_md("---\nname: a\ndescription: d\npermissionMode: plan\n---\nx", "f", KNOWN)
        self.assertTrue(plan.read_only)
        bypass, notes = parse_agent_md(
            "---\nname: a\ndescription: d\npermissionMode: bypassPermissions\n---\nx", "f", KNOWN)
        self.assertFalse(bypass.read_only)
        self.assertTrue(any("bypassPermissions" in n and "ignored" in n for n in notes), notes)

    def test_max_turns_skills_disallowed(self):
        agent, _ = parse_agent_md("---\nname: a\ndescription: d\nmaxTurns: 5\nskills: [scene-planning]\n"
                                  "disallowedTools: Bash\n---\nx", "f", KNOWN)
        self.assertEqual((agent.max_turns, agent.skills, agent.disallowed), (5, ("scene-planning",), ("execute",)))

    def test_round_trip_keeps_unknown_fields(self):
        agent, _ = parse_agent_md(CRITIC, "f", KNOWN)
        again, _ = parse_agent_md(render_agent_md(agent.frontmatter, agent.prompt), "f", KNOWN)
        self.assertEqual(again.frontmatter, agent.frontmatter)
        self.assertEqual(again.prompt, agent.prompt)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.a, self.b = self.root / "a", self.root / "b"

    def test_builtins_are_present(self):
        registry = AgentRegistry.load([], KNOWN)
        self.assertEqual([a.name for a in registry.agents()], ["moderator", "reviewer"])
        self.assertTrue(registry.get("reviewer").read_only)
        self.assertEqual(registry.get("moderator").tools, ())

    def test_folders_load_recursively_and_skip_readme(self):
        write(self.a / "nested" / "critic.md", CRITIC)
        write(self.a / "README.md", "# notes")
        registry = AgentRegistry.load([self.a], KNOWN)
        self.assertEqual(registry.get("critic").source, str(self.a / "nested" / "critic.md"))
        self.assertFalse(any("README" in w for w in registry.warnings), registry.warnings)

    def test_later_folder_wins_and_says_so(self):
        write(self.a / "c.md", CRITIC)
        write(self.b / "c.md", CRITIC.replace("Finds weaknesses.", "Newer."))
        registry = AgentRegistry.load([self.a, self.b], KNOWN)
        self.assertEqual(registry.get("critic").description, "Newer.")
        self.assertTrue(any("critic" in w and "overrides" in w for w in registry.warnings), registry.warnings)

    def test_duplicate_in_one_folder_first_sorted_path_wins(self):
        write(self.a / "a.md", CRITIC)
        write(self.a / "b.md", CRITIC.replace("Finds weaknesses.", "Second."))
        registry = AgentRegistry.load([self.a], KNOWN)
        self.assertEqual(registry.get("critic").description, "Finds weaknesses.")
        self.assertTrue(any("b.md" in w for w in registry.warnings), registry.warnings)

    def test_missing_folder_is_a_warning(self):
        registry = AgentRegistry.load([self.root / "nope"], KNOWN)
        self.assertTrue(any("nope" in w for w in registry.warnings), registry.warnings)

    def test_user_agent_replaces_a_builtin(self):
        write(self.a / "r.md", "---\nname: reviewer\ndescription: Mine.\n---\nx")
        registry = AgentRegistry.load([self.a], KNOWN)
        row = next(r for r in registry.overview() if r["name"] == "reviewer")
        self.assertEqual(row["overrides"], "built-in")
        self.assertEqual(registry.get("reviewer").description, "Mine.")

    def test_overview_rows(self):
        write(self.a / "c.md", CRITIC)
        rows = {r["name"]: r for r in AgentRegistry.load([self.a], KNOWN).overview()}
        self.assertEqual(rows["critic"]["tools"], "read_file, grep, glob, ls")
        self.assertEqual(rows["critic"]["model"], "configured")
        self.assertEqual(rows["moderator"]["source"], "built-in")

    def test_seed_example_is_valid(self):
        text = (SEED_DIR / "agents" / "critic.md").read_text(encoding="utf-8")
        agent, notes = parse_agent_md(text, "seed", KNOWN)
        self.assertIsNotNone(agent, notes)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()

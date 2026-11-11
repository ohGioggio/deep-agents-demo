"""Subagents: the Claude Code agent file format, the registry, and agents as subagents."""
import unittest

from deep_agents_demo.agent_files import parse_agent_md, render_agent_md, resolve_model, translate_tools

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


if __name__ == "__main__":
    unittest.main()

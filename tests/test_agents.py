"""Subagents: the Claude Code agent file format, the registry, and agents as subagents."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.core import assembly as agent_module
from deep_agents_demo.core.assembly import build_agent
from recording_ui import Approver
from deep_agents_demo.chat import stream_turn
from deep_agents_demo.core.middleware import SubagentCallCap
from deep_agents_demo.core.tools import BUILTIN, tool_name
from deep_agents_demo.features.agents.files import (
    parse_agent_md,
    render_agent_md,
    resolve_model,
    translate_tools,
)
from deep_agents_demo.features.agents.registry import AgentRegistry
from deep_agents_demo.features.guardrails.feature import Guardrails
from deep_agents_demo.startup.workspace import SEED_DIR

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

    def test_max_turns_must_be_a_real_number(self):
        agent, notes = parse_agent_md("---\nname: a\ndescription: d\nmaxTurns: true\n---\nx", "f", KNOWN)
        self.assertIsNone(agent.max_turns)
        self.assertTrue(any("maxTurns" in n for n in notes), notes)

    def test_model_must_be_text(self):
        agent, notes = parse_agent_md("---\nname: a\ndescription: d\nmodel: [x]\n---\nx", "f", KNOWN)
        self.assertIsNone(agent.model)
        self.assertTrue(any("model" in n for n in notes), notes)

    def test_skills_must_be_names(self):
        agent, notes = parse_agent_md("---\nname: a\ndescription: d\nskills: {a: b}\n---\nx", "f", KNOWN)
        self.assertEqual(agent.skills, ())
        self.assertTrue(any("skills" in n for n in notes), notes)
        agent, _ = parse_agent_md("---\nname: a\ndescription: d\nskills: one two\n---\nx", "f", KNOWN)
        self.assertEqual(agent.skills, ("one", "two"))

    def test_name_with_trailing_newline_is_rejected(self):
        agent, _ = parse_agent_md('---\nname: "critic\\n"\ndescription: d\n---\nx', "f", KNOWN)
        self.assertIsNone(agent)

    def test_frontmatter_tolerates_blank_first_line_and_trailing_spaces(self):
        agent, notes = parse_agent_md("\n---  \nname: a\ndescription: d\n---  \nx", "f", KNOWN)
        self.assertIsNotNone(agent, notes)

    def test_blank_tools_means_all_too(self):
        """Review finding: a blank 'tools:' line silently gave the agent no tools."""
        for line in ("tools:", "tools: ''", "tools: []"):
            with self.subTest(line=line):
                agent, notes = parse_agent_md(f"---\nname: a\ndescription: d\n{line}\n---\nx", "f", KNOWN)
                self.assertIsNone(agent.tools)
                self.assertTrue(any("all tools" in n for n in notes), notes)

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

    def test_researcher_override_only_when_there_is_one(self):
        write(self.a / "r.md", "---\nname: researcher\ndescription: Mine.\n---\nx")
        with_one = AgentRegistry.load([self.a], KNOWN)
        without = AgentRegistry.load([self.a], KNOWN, replaceable=("general-purpose",))
        row = lambda reg: next(r for r in reg.overview() if r["name"] == "researcher")["overrides"]
        self.assertEqual((row(with_one), row(without)), ("built-in", ""))

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


SEEN = []  # every model call: ("tools", [names]) then ("messages", [...])


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        SEEN.append(("tools", [tool_name(t) for t in tools]))
        return self

    def _generate(self, messages, *args, **kwargs):
        SEEN.append(("messages", list(messages)))
        return super()._generate(messages, *args, **kwargs)


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


def delegate(agent, call_id="t1"):
    return call("task", {"description": "do the job", "subagent_type": agent}, call_id)


def agent_from(text):
    agent, notes = parse_agent_md(text, "test", frozenset(BUILTIN) | {"search_assets", "triangle_budget"})
    assert agent is not None, notes
    return agent


ALL_AGENT = "---\nname: helper\ndescription: Helps.\n---\nHelp."
READ_ONLY_AGENT = "---\nname: planner\ndescription: Plans.\npermissionMode: plan\n---\nPlan."


class SubagentTests(unittest.TestCase):
    def setUp(self):
        SEEN.clear()
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        shutil.copytree(SEED_DIR / "skills", self.root / "skills")

    def kwargs(self, agents, **options):
        with patch.object(agent_module, "create_deep_agent") as create:
            agent_module.build_agent("main-model", None, object(), [], agents=agents, **options)
        return create.call_args.kwargs

    def spec(self, kwargs, name):
        return next(s for s in kwargs["subagents"] if s["name"] == name)

    def run_turn(self, *responses, agents=(), features=(), approver=None, **options):
        model = Model(responses=[*responses, AIMessage(content="done")])
        backend = FilesystemBackend(root_dir=str(self.root), virtual_mode=True)
        agent = build_agent(model, InMemorySaver(), backend, features, agents=agents, **options)
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=approver or Approver(interactive=False))

    def subagent_tools(self):
        """Tool names bound for the subagent's first model call (the main agent's is first)."""
        return [names for kind, names in SEEN if kind == "tools"][1]

    def test_a_user_agent_is_offered_to_the_main_agent(self):
        critic = agent_from(CRITIC)
        spec = self.spec(self.kwargs([critic]), "critic")
        self.assertEqual((spec["description"], spec["system_prompt"]), ("Finds weaknesses.", "You are a critic."))

    def test_agent_sees_only_its_tools(self):
        self.run_turn(delegate("critic"), call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                      AIMessage(content="sub done"), agents=[agent_from(CRITIC)])
        self.assertLessEqual(set(self.subagent_tools()), {"read_file", "grep", "glob", "ls"})
        self.assertFalse((self.root / "x.md").exists())

    def test_read_only_agent_cannot_write(self):
        self.run_turn(delegate("planner"), call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                      AIMessage(content="sub done"), agents=[agent_from(READ_ONLY_AGENT)])
        self.assertNotIn("write_file", self.subagent_tools())
        self.assertFalse((self.root / "x.md").exists())

    def test_disabled_tools_apply_inside_agents(self):
        self.run_turn(delegate("helper"), call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                      AIMessage(content="sub done"), agents=[agent_from(ALL_AGENT)], disabled_tools=("write_file",))
        self.assertNotIn("write_file", self.subagent_tools())
        self.assertFalse((self.root / "x.md").exists())

    def test_approvals_apply_inside_agents_even_with_bypass(self):
        bypass = agent_from(ALL_AGENT.replace("description: Helps.", "description: Helps.\npermissionMode: bypassPermissions"))
        (self.root / "notes.md").write_text("keep")

        def ask(_prompt):
            raise AssertionError("asked")
        with self.assertRaisesRegex(AssertionError, "asked"):
            self.run_turn(delegate("helper"), call("delete", {"file_path": "/notes.md"}, "d"),
                          AIMessage(content="sub done"), agents=[bypass], features=[Guardrails.feature()],
                          approver=Approver(ask=ask, say=lambda *_: None))
        self.assertTrue((self.root / "notes.md").exists())

    def test_overriding_general_purpose_keeps_limits(self):
        mine = agent_from(ALL_AGENT.replace("name: helper", "name: general-purpose"))
        names = [s["name"] for s in self.kwargs([mine])["subagents"]]
        self.assertEqual(names.count("general-purpose"), 1)
        self.assertEqual(self.spec(self.kwargs([mine]), "general-purpose")["system_prompt"], "Help.")
        self.run_turn(delegate("general-purpose"), call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                      AIMessage(content="sub done"), agents=[mine], disabled_tools=("write_file",))
        self.assertFalse((self.root / "x.md").exists())

    def test_call_cap(self):
        ls = lambda i: call("ls", {"path": "/"}, f"l{i}")
        self.run_turn(delegate("helper"), ls(1), ls(2), agents=[agent_from(ALL_AGENT)], max_calls=2)
        calls = [kind for kind, _ in SEEN if kind == "messages"]
        self.assertEqual(len(calls), 4, "main, two subagent calls (then capped), main")

    def sub_calls(self, prompt="Help."):
        """Model calls made by the subagent, recognised by its system prompt."""
        return sum(1 for kind, messages in SEEN if kind == "messages"
                   and messages and messages[0].type == "system" and prompt in str(messages[0].content))

    def test_cap_survives_approval_pauses(self):
        """Review finding: each approve-and-resume started the count again."""
        notes = [self.root / f"n{i}.md" for i in range(4)]
        for note in notes:
            note.write_text("x")
        deletes = [call("delete", {"file_path": f"/n{i}.md"}, f"d{i}") for i in range(4)]
        self.run_turn(delegate("helper"), *deletes, AIMessage(content="sub done"),
                      agents=[agent_from(ALL_AGENT)], features=[Guardrails.feature()], max_calls=2,
                      approver=Approver(ask=lambda _: "a", say=lambda *_: None))
        self.assertLessEqual(self.sub_calls(), 2)

    def test_cap_survives_rejections_in_one_shot_runs(self):
        deletes = [call("delete", {"file_path": f"/n{i}.md"}, f"d{i}") for i in range(5)]
        self.run_turn(delegate("helper"), *deletes, AIMessage(content="sub done"),
                      agents=[agent_from(ALL_AGENT)], features=[Guardrails.feature()], max_calls=2)
        self.assertLessEqual(self.sub_calls(), 2)

    def test_each_delegation_gets_a_fresh_count(self):
        self.run_turn(delegate("helper", "t1"), AIMessage(content="first done"),
                      delegate("helper", "t2"), AIMessage(content="second done"),
                      agents=[agent_from(ALL_AGENT)], max_calls=1)
        main_last_call = [m for kind, m in SEEN if kind == "messages"][-1]
        results = [str(m.content) for m in main_last_call if m.type == "tool"]
        self.assertEqual(results, ["first done", "second done"])

    def test_a_capped_run_returns_a_partial_result(self):
        """Review finding: the main agent only got 'Model call limits exceeded'."""
        progress = AIMessage(content="Found three issues so far.",
                             tool_calls=[{"name": "ls", "id": "l1", "args": {"path": "/"}}])
        self.run_turn(delegate("helper"), progress, call("ls", {"path": "/"}, "l2"),
                      agents=[agent_from(ALL_AGENT)], max_calls=2)
        main_last_call = [m for kind, m in SEEN if kind == "messages"][-1]
        result = str([m for m in main_last_call if m.type == "tool"][-1].content)
        self.assertIn("partial", result.lower())
        self.assertIn("Found three issues so far.", result)
        self.assertIn("2 model calls", result)

    def test_max_turns_lowers_cap(self):
        agent = agent_from(ALL_AGENT.replace("description: Helps.", "description: Helps.\nmaxTurns: 1"))
        spec = self.spec(self.kwargs([agent], max_calls=25), "helper")
        caps = [m.limit for m in spec["middleware"] if isinstance(m, SubagentCallCap)]
        self.assertEqual(caps, [1])
        general = self.spec(self.kwargs([agent], max_calls=25), "general-purpose")
        self.assertEqual([m.limit for m in general["middleware"] if isinstance(m, SubagentCallCap)], [25])

    def test_skills_are_pinned(self):
        agent = agent_from(ALL_AGENT.replace("description: Helps.", "description: Helps.\nskills: [scene-planning]"))
        self.run_turn(delegate("helper"), AIMessage(content="sub done"), agents=[agent])
        first_sub_call = [m for kind, m in SEEN if kind == "messages"][1]
        self.assertTrue(any('<skill name="scene-planning"' in str(m.content) for m in first_sub_call))

    def test_subagents_can_plan(self):
        for text, name in ((ALL_AGENT, "helper"), (READ_ONLY_AGENT, "planner")):
            with self.subTest(agent=name):
                SEEN.clear()
                self.run_turn(delegate(name), AIMessage(content="sub done"), agents=[agent_from(text)])
                self.assertIn("write_todos", self.subagent_tools())

    def test_todo_list_only_where_write_todos_is_allowed(self):
        from langchain.agents.middleware import TodoListMiddleware
        kwargs = self.kwargs([agent_from(CRITIC), agent_from(ALL_AGENT)])

        def has_todos(name):
            return any(isinstance(m, TodoListMiddleware) for m in self.spec(kwargs, name)["middleware"])
        self.assertFalse(has_todos("critic"), "critic can't call write_todos")
        self.assertTrue(has_todos("helper"))
        self.assertTrue(has_todos("general-purpose"))

    def test_model_for_is_used_for_explicit_models(self):
        explicit = agent_from(ALL_AGENT.replace("description: Helps.", "description: Helps.\nmodel: nvidia/other"))
        inherit = agent_from(CRITIC)
        kwargs = self.kwargs([explicit, inherit], model_for=lambda name: f"model:{name}")
        self.assertEqual(self.spec(kwargs, "helper")["model"], "model:nvidia/other")
        self.assertEqual(self.spec(kwargs, "critic")["model"], "main-model")


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_agents_command_lists_sources(self):
        from deep_agents_demo.chat import ChatSession
        write(self.root / "agents" / "critic.md", CRITIC)
        registry = AgentRegistry.load([self.root / "agents"], KNOWN)
        said = []
        ChatSession(chats=None, current="c", say=said.append, agents=registry).handle("/agents")
        text = "\n".join(said)
        for expected in ("critic", "reviewer", "moderator", "built-in", "read_file", "critic.md"):
            self.assertIn(expected, text)

    def test_agents_footer_matches_reality(self):
        from deep_agents_demo.chat import ChatSession
        write(self.root / "agents" / "gp.md", "---\nname: general-purpose\ndescription: Mine.\n---\nx")
        said = []
        ChatSession(chats=None, current="c", say=said.append,
                    agents=AgentRegistry.load([self.root / "agents"], KNOWN)).handle("/agents")
        text = "\n".join(said)
        self.assertNotIn("always there", text)
        self.assertIn("(overrides built-in)", text)

    def test_agent_folder_is_read_only_to_the_agent(self):
        (self.root / "skills").mkdir()
        (self.root / "agents").mkdir()
        model = Model(responses=[call("write_file", {"file_path": "/agents/evil.md", "content": "x"}, "w"),
                                 AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), FilesystemBackend(root_dir=str(self.root), virtual_mode=True))
        out = agent.invoke({"messages": [{"role": "user", "content": "x"}]}, {"configurable": {"thread_id": "t"}})
        results = [str(m.content) for m in out["messages"] if m.type == "tool"]
        self.assertTrue(any("permission denied" in r for r in results), results)
        self.assertFalse((self.root / "agents" / "evil.md").exists())


if __name__ == "__main__":
    unittest.main()

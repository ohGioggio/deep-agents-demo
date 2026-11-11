"""Tools registry: disabled tools (DISABLED_TOOLS), skill-only tools, declared skill tools."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_guardrails import FakeSandbox
from test_memory import FakeMemory

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.approvals import Approver
from deep_agents_demo.features.catalog.feature import catalog_feature
from deep_agents_demo.chat import stream_turn
from deep_agents_demo.config import Settings
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.core.middleware import ToolAccess
from deep_agents_demo.core.tools import CORE, ToolRegistry, shell_block, tool_name
from deep_agents_demo.features.skills.format import split_skill_md, warnings
from deep_agents_demo.features.skills.manager import SkillManager
from deep_agents_demo.features.guardrails.feature import Guardrails
from deep_agents_demo.features.memory.service import MemoryService
from deep_agents_demo.features.memory.feature import memory_feature
from deep_agents_demo.persistence.versioned_store import VersionedStore
from deep_agents_demo.startup.workspace import SEED_DIR

SEEN = []  # (tool names bound, system prompt) for every model call


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        SEEN.append([tool_name(t) for t in tools])
        return self

    def _generate(self, messages, *args, **kwargs):
        SEEN.append(str(messages[0].content) if messages and messages[0].type == "system" else "")
        return super()._generate(messages, *args, **kwargs)


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


def notes(name):
    """A tool that only says it ran."""
    def tool(text: str = "") -> str:
        return f"{name} ran"
    tool.__name__, tool.__doc__ = name, f"The {name} tool. More detail."
    return tool


class RegistryTests(unittest.TestCase):
    def test_lists_builtins_and_feature_tools_with_groups(self):
        registry = ToolRegistry([catalog_feature(), Feature("memory", tools=(notes("remember"),))])
        entries = {entry.name: entry for entry in registry.entries()}
        self.assertEqual(entries["execute"].group, "shell")
        self.assertEqual(entries["remember"].group, "memory")
        self.assertEqual(entries["remember"].description, "The remember tool")
        self.assertTrue(entries["search_assets"].skill_only)
        self.assertEqual(registry.skill_only(), {"search_assets", "triangle_budget"})

    def test_core_and_unknown_names_cannot_be_disabled(self):
        registry = ToolRegistry([catalog_feature()])
        self.assertEqual(registry.disabled(["execute", "read_file", "nope"]), {"execute"})
        problems = " ".join(registry.problems(["execute", "read_file", "nope"]))
        self.assertIn("no tool named nope", problems)
        self.assertIn("core tools can't be disabled (read_file)", problems)
        self.assertEqual(registry.problems(["execute"]), [])
        self.assertLessEqual(CORE, registry.names())

    def test_skill_only_must_name_the_features_own_tools(self):
        with self.assertRaises(ValueError):
            ToolRegistry([Feature("x", tools=(notes("a"),), skill_only=("b",))])

    def test_disabled_tools_setting(self):
        env = {"OPENAI_API_KEY": "k", "AGENT_MODEL": "m", "DISABLED_TOOLS": "execute, delete  execute"}
        self.assertEqual(Settings.from_env(env).disabled_tools, ("execute", "delete"))
        self.assertEqual(Settings.from_env({"OPENAI_API_KEY": "k"}).disabled_tools, ())


class AccessTests(unittest.TestCase):
    def setUp(self):
        SEEN.clear()
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        shutil.copytree(SEED_DIR / "skills", self.root / "skills")
        (self.root / "notes.md").write_text("keep me")

    def build(self, *responses, features=(), disabled=(), prefix=""):
        model = Model(responses=[*responses, AIMessage(content="done")])
        backend = FilesystemBackend(root_dir=str(self.root), virtual_mode=True)
        if prefix:  # skills mounted on a route of a composite backend, as with a sandbox
            backend = CompositeBackend(default=FilesystemBackend(root_dir=tempfile.mkdtemp(),
                                                                 virtual_mode=True),
                                       routes={f"{prefix}/": backend})
        return build_agent(model, InMemorySaver(), backend, [catalog_feature(), *features],
                           disabled_tools=disabled, skills_prefix=prefix)

    def run_turn(self, *responses, **kwargs):
        agent = self.build(*responses, **kwargs)
        with patch("builtins.print"):
            out = agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                               {"configurable": {"thread_id": "t"}})
        return [str(m.content) for m in out["messages"] if m.type == "tool"]

    def run_with_approver(self, *responses, **kwargs):
        """A turn through the chat loop; fails if anything asks the user."""
        agent = self.build(*responses, **kwargs)

        def ask(_prompt):
            raise AssertionError("a disabled tool must not reach the approval prompt")
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=Approver(ask=ask, say=lambda *_: None))
        state = agent.get_state({"configurable": {"thread_id": "t"}}).values
        return [str(m.content) for m in state["messages"] if m.type == "tool"]

    def test_disabled_tools_never_ask_for_approval(self):
        delete = call("delete", {"file_path": "/notes.md"}, "d1")
        results = self.run_with_approver(delete, features=[Guardrails.feature()], disabled=("delete",))
        self.assertIn("turned off", results[0])
        self.assertTrue((self.root / "notes.md").exists())

    def test_a_disabled_feature_tool_is_refused_not_unknown(self):
        mem = FakeMemory()
        memory_id = mem.seed("alice", "Prefers pine trees")
        features = [memory_feature(MemoryService(mem, "alice"), "auto"), Guardrails.feature()]
        forget = call("forget", {"memory_id": memory_id}, "f1")
        results = self.run_with_approver(forget, features=features, disabled=("forget",))
        self.assertIn("turned off", results[0])
        self.assertIn(memory_id, mem.items)

    def test_registry_knows_every_tool_deep_agents_binds(self):
        """Keeps BUILTIN in sync with Deep Agents: an unlisted tool couldn't be disabled."""
        from deep_agents_demo.features.skills.manager import SkillManager
        store = VersionedStore(self.root / "s.sqlite")
        self.addCleanup(store.close)
        features = [SkillManager(store, "alice", self.root).feature(), Guardrails.feature(),
                    memory_feature(MemoryService(FakeMemory(), "alice"), "both")]
        self.run_turn(call("read_file", {"file_path": "/skills/scene-planning/SKILL.md"}, "r"),
                      features=features)
        registry = ToolRegistry([catalog_feature(), *features])
        for names in self.bound():
            self.assertLessEqual(set(names), registry.names())

    def run_shell(self, *commands, disabled=("delete",)):
        """Run each command through execute on a fake sandbox, with guardrails on."""
        sandbox = FakeSandbox()
        model = Model(responses=[*(call("execute", {"command": c}, f"x{i}") for i, c in enumerate(commands)),
                                 AIMessage(content="done")])
        backend = CompositeBackend(default=sandbox, routes={
            "/instructions/": FilesystemBackend(root_dir=str(self.root), virtual_mode=True)})
        agent = build_agent(model, InMemorySaver(), backend, [catalog_feature(), Guardrails.feature()],
                            disabled_tools=disabled, skills_prefix="/instructions")

        def ask(_prompt):
            raise AssertionError("a refused command must not reach the approval prompt")
        with patch("builtins.print"):
            stream_turn(agent, "go", {"configurable": {"thread_id": "t"}},
                        approver=Approver(Guardrails().preview, ask=ask, say=lambda *_: None))
        return sandbox.ran

    def test_with_delete_off_the_shell_cannot_delete_either(self):
        """Live finding: with delete off, the agent removed a file with a plain rm."""
        ran = self.run_shell("rm /sandbox/scene_plan.md", "rm -rf build", "ls -la",
                             "python3 -c 'import os; os.remove(\"a\")'")
        self.assertEqual(ran, ["ls -la"])

    def test_with_delete_on_the_shell_is_unchanged(self):
        self.assertEqual(self.run_shell("rm notes.md", disabled=()), ["rm notes.md"])

    def test_shell_block_patterns(self):
        for command in ("rm a", "cd x && rm -f b", "rmdir d", "unlink f", "find . -delete",
                        "git rm f", "git clean -fd", "python -c 'import shutil; shutil.rmtree(\"d\")'",
                        "python -c 'Path(\"a\").unlink()'"):
            with self.subTest(command=command):
                self.assertEqual(shell_block(command, {"delete"}), "delete")
        for command in ("ls", "cat rm.txt", "echo firmware", "grep -r farm .", "mkdir out"):
            with self.subTest(command=command):
                self.assertIsNone(shell_block(command, {"delete"}))
        self.assertIsNone(shell_block("rm a", {"execute"}))

    def run_openshell(self, *responses, openshell=True):
        """A turn on a fake sandbox with the OpenShell sandbox feature (and its skill)."""
        from deep_agents_demo.features.sandbox.feature import sandbox_feature
        shutil.copytree(SEED_DIR / "sandbox-skills", self.root / "sandbox-skills", dirs_exist_ok=True)
        sandbox = FakeSandbox()
        backend = CompositeBackend(default=sandbox, routes={
            "/instructions/": FilesystemBackend(root_dir=str(self.root), virtual_mode=True)})
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), backend, [sandbox_feature("/sandbox", openshell)],
                            skills_prefix="/instructions")
        with patch("builtins.print"):
            out = agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                               {"configurable": {"thread_id": "t"}})
        return sandbox.ran, [str(m.content) for m in out["messages"] if m.type == "tool"]

    def test_execute_waits_until_the_sandbox_skill_is_read(self):
        """Live finding: the agent skipped the skill and tried pip, apt and get-pip first."""
        skill = "/instructions/sandbox-skills/sandbox/SKILL.md"
        ran, results = self.run_openshell(call("execute", {"command": "pip install x"}, "e1"),
                                          call("read_file", {"file_path": skill}, "r1"),
                                          call("execute", {"command": "uv pip install x"}, "e2"))
        self.assertEqual(ran, ["uv pip install x"])
        self.assertIn(f"read the sandbox skill: read_file {skill}", results[0])

    def test_no_skill_gate_without_openshell(self):
        ran, _ = self.run_openshell(call("execute", {"command": "ls"}, "e1"), openshell=False)
        self.assertEqual(ran, ["ls"])

    def test_no_researcher_mentioned_when_its_lookups_are_disabled(self):
        self.run_turn(disabled=("search_assets",))
        self.assertNotIn("researcher", self.prompts()[0])

    def test_skill_only_unlock_works_on_the_instructions_route(self):
        read = call("read_file", {"file_path": "/instructions/skills/scene-planning/SKILL.md"}, "r1")
        results = self.run_turn(read, call("triangle_budget", {"asset_ids": ["oak"], "copies": [2]}, "t1"),
                                prefix="/instructions")
        self.assertIn('"triangles": 24000', results[-1])

    def bound(self):
        return [names for names in SEEN if isinstance(names, list)]

    def prompts(self):
        return [text for text in SEEN if isinstance(text, str)]

    def test_disabled_tools_are_hidden_and_refused(self):
        results = self.run_turn(call("write_file", {"file_path": "/x.md", "content": "x"}, "w"),
                                disabled=("write_file",))
        self.assertTrue(all("write_file" not in names for names in self.bound()))
        self.assertIn("turned off", results[0])
        self.assertFalse((self.root / "x.md").exists())
        self.assertIn("Turned off in this setup: write_file", self.prompts()[0])

    def test_skill_only_tools_appear_after_reading_the_skill(self):
        early = call("search_assets", {"category": "tree"}, "s0")
        read = call("read_file", {"file_path": "/skills/scene-planning/SKILL.md"}, "r1")
        late = call("search_assets", {"category": "tree"}, "s1")
        results = self.run_turn(early, read, late)
        bound = self.bound()
        self.assertNotIn("search_assets", bound[0])
        self.assertIn("read_file", bound[0])
        self.assertIn("search_assets", bound[-1])
        self.assertIn("only available after reading a skill", results[0])
        self.assertIn('"id": "oak"', results[2])

    def test_general_purpose_subagent_is_restricted_too(self):
        delegate = call("task", {"description": "write a file", "subagent_type": "general-purpose"}, "t1")
        sub_write = call("write_file", {"file_path": "/y.md", "content": "y"}, "w1")
        sub_search = call("search_assets", {"category": "car"}, "s1")
        sub_done = AIMessage(content="sub done")
        self.run_turn(delegate, sub_write, sub_search, sub_done, disabled=("write_file",))
        self.assertFalse((self.root / "y.md").exists())
        # Every model call, main or subagent, saw neither tool.
        self.assertTrue(all("write_file" not in names and "search_assets" not in names
                            for names in self.bound()))

    def test_agent_is_told_which_skills_need_a_disabled_tool(self):
        self.run_turn(disabled=("triangle_budget",))
        self.assertIn("scene-planning (needs triangle_budget)", self.prompts()[0])


class FakeRequest:
    """Just enough of a model or tool-call request for ToolAccess."""

    def __init__(self, tools=(), call=None, state=None):
        self.tools, self.messages, self.system_message = list(tools), [], None
        self.state = state or {"messages": []}
        self.tool_call = {"name": call, "id": "c1", "args": {}}

    def override(self, **changes):
        clone = FakeRequest(self.tools, self.tool_call["name"], self.state)
        for key, value in changes.items():
            setattr(clone, key, value)
        return clone


def visible(access, names):
    return [tool_name(t) for t in access._request(FakeRequest([{"name": n} for n in names])).tools]


def refusal(access, name):
    message = access.refusal(FakeRequest(call=name))
    return message.content if message else None


class PerAgentAccessTests(unittest.TestCase):
    ALL = ["ls", "read_file", "write_file", "execute", "search_assets"]

    def test_allowlist_hides_and_refuses_other_tools(self):
        access = ToolAccess(allowed=frozenset({"read_file"}))
        self.assertEqual(visible(access, self.ALL), ["read_file"])
        self.assertIn("isn't available to this agent", refusal(access, "write_file"))
        self.assertIsNone(refusal(access, "read_file"))

    def test_denylist(self):
        access = ToolAccess(denied=frozenset({"execute"}))
        self.assertNotIn("execute", visible(access, self.ALL))
        self.assertIn("isn't available to this agent", refusal(access, "execute"))
        self.assertIsNone(refusal(access, "read_file"))

    def test_disabled_beats_allowlist(self):
        access = ToolAccess(disabled=frozenset({"read_file"}), allowed=frozenset({"read_file"}))
        self.assertIn("turned off", refusal(access, "read_file"))

    def test_listed_skill_only_tool_is_visible_at_once(self):
        access = ToolAccess(skill_only=frozenset({"search_assets"}), allowed=frozenset({"search_assets"}))
        self.assertEqual(visible(access, self.ALL), ["search_assets"])
        self.assertIsNone(refusal(access, "search_assets"))


class SkillToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.store = VersionedStore(self.root / "skills.sqlite")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.addCleanup(self.store.close)
        self.manager = SkillManager(self.store, "alice", self.root)
        self.manager.known_tools = ToolRegistry([catalog_feature()]).names()

    def front(self, change):
        return split_skill_md(change.after["SKILL.md"])[0]

    def test_create_saves_declared_tools_as_allowed_tools(self):
        change = self.manager.plan_create("trees", "Pick trees.", "1. Search.", tools=["search_assets"])
        self.assertEqual(change.errors, [])
        self.assertEqual(self.front(change)["allowed-tools"], "search_assets")
        self.assertIn("Tools it uses: search_assets", change.summary())

    def test_unknown_tools_are_rejected(self):
        change = self.manager.plan_create("trees", "Pick trees.", "1. Hi.", tools=["make_coffee"])
        self.assertIn("no tool named make_coffee", " ".join(change.errors))
        self.assertIn("Checks: no tool named make_coffee", change.summary())

    def test_update_replaces_or_clears_tools(self):
        self.manager.apply(self.manager.plan_create("trees", "Pick.", "1. Hi.", tools=["search_assets"]))
        change = self.manager.plan_update("trees", tools=["search_assets", "triangle_budget"])
        self.assertEqual(self.front(change)["allowed-tools"], "search_assets triangle_budget")
        self.assertIn("+allowed-tools: search_assets triangle_budget", change.diff())
        cleared = self.manager.plan_update("trees", tools=[])
        self.assertNotIn("allowed-tools", self.front(cleared))
        untouched = self.manager.plan_update("trees", description="Pick trees well.")
        self.assertEqual(self.front(untouched)["allowed-tools"], "search_assets")

    def test_imported_skills_with_unknown_tools_get_a_warning(self):
        files = {"SKILL.md": "---\nname: x\ndescription: X.\nallowed-tools: Bash search_assets\n---\nHi\n"}
        self.assertIn("doesn't have: Bash", " ".join(warnings(files, {"search_assets"})))
        self.assertEqual(warnings(files), [])

    def test_create_skill_tool_takes_tools(self):
        create = {tool.__name__: tool for tool in self.manager.tools()}["create_skill"]
        result = create("trees", "Pick trees.", "1. Search.", tools=["search_assets"])
        self.assertIn('"saved": true', result)
        saved = self.store.get("user", "alice", "trees").files["SKILL.md"]
        self.assertIn("allowed-tools: search_assets", saved)


if __name__ == "__main__":
    unittest.main()

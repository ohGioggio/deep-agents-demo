# Subagents registry, piece 1: registry and file format — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load Claude Code–format agent files from folders (plus built-ins), turn them into Deep Agents subagents with the same approvals, guardrails and tool limits as the main agent, and list them with `/agents`.

**Architecture:** `agent_files.py` parses and writes the file format (one agent = `AgentDefinition`). `agent_registry.py` loads built-ins and folders with precedence and reports warnings. `build_agent` turns each definition into a `SubAgent` spec whose middleware is `ToolAccess` (now with a per-agent allowlist/denylist), a model-call cap, and optional skill pinning; `interrupt_on` is inherited as today. `app.py` wires settings → registry → agent; `chat.py` adds `/agents`.

**Tech Stack:** Python 3.11+, LangChain 1.4.9 (`ModelCallLimitMiddleware`), Deep Agents 0.7.34 (`SubAgent` specs, `pinned_skills`), PyYAML, unittest with fake chat models.

**Spec:** `design/specs/2026-11-11-subagents-registry-design.md` (sections 1, 2, 7, 8 and "Calls made by Claude"; piece 1 only).

## Global Constraints

- Agent file = YAML frontmatter + Markdown body (the system prompt); required `name`, `description`.
- Names: `^[A-Za-z0-9_-]{1,64}$`; anything else is skipped with a warning.
- `tools` / `disallowedTools`: comma-separated string or YAML list; omitted `tools` = all tools the main agent has.
- Tool translation: `Read`→`read_file`, `Write`→`write_file`, `Edit`/`MultiEdit`→`edit_file`, `Bash`/`PowerShell`→`execute`, `Grep`→`grep`, `Glob`→`glob`+`ls`, `LS`→`ls`, `TodoWrite`→`write_todos`, `WebFetch`→`web_fetch`, `WebSearch`→`web_search`; our own names accepted; `Agent`/`Task` dropped (no nesting); unresolved names are warnings; if `tools` is given and nothing resolves, the agent is not loaded.
- `model`: omitted/`inherit` = configured model; `sonnet`/`opus`/`haiku`/`fable` and ids starting `claude-` = configured model with a note; any other value = that model id on the same endpoint and key.
- `permissionMode: plan` = read-only tools (`ls`, `read_file`, `grep`, `glob`, `write_todos`); every other value ignored with a note; never loosens approvals.
- `skills` = skill names pinned into the subagent's context at start; `maxTurns` lowers the per-run cap, never raises it.
- Unknown frontmatter fields are kept on round-trip and ignored, with one note per agent.
- Precedence (higher wins): built-ins < folders (`AGENT_DIRS` in order, then `demo_workspace/agents/`). (Installed and SQLite agents come in piece 2.)
- A user agent named like a built-in (`general-purpose`, `researcher`, `reviewer`, `moderator`) replaces it.
- Every subagent: `ToolAccess` (DISABLED_TOOLS, skill-only, skill gates), inherited approvals, and `SUBAGENT_MAX_CALLS` (default 25) model calls per run.
- Settings: `AGENT_DIRS` (separated by `os.pathsep`), `SUBAGENT_MAX_CALLS` (default 25, ≥ 1).
- The agent's file tools can't write in `demo_workspace/agents/` (same deny rule as skill folders).
- Commits: one-line messages, no trailer; don't push unless the user asks.
- Run tests with `uv run python -m unittest discover -s tests` (single file: add `-p "test_agents.py"`).

## Review Focus

1. **Real Claude Code files**: CRLF line endings, a UTF-8 BOM, `tools` as a YAML list, and invalid YAML — expected: the first three load fine, invalid YAML is skipped with a warning naming the file. (Task 1 tests.)
2. **Same name in two folders** — expected: the later folder wins and a warning says which file was overridden; within one folder, the first by sorted path wins with a warning. (Task 2 tests.)
3. **A missing `AGENT_DIRS` entry** — expected: a startup warning, no crash. (Task 2 test.)
4. **A user agent that overrides `general-purpose`** — expected: still gets approvals, DISABLED_TOOLS and the call cap. (Task 4 test.)
5. **An agent file asking for `permissionMode: bypassPermissions`** — expected: a note, and risky actions still pause for approval. (Task 4 test.)

---

### Task 1: The agent file format

**Files:**
- Create: `src/deep_agents_demo/agent_files.py`
- Test: `tests/test_agents.py` (class `FormatTests`)

**Interfaces:**
- Produces:
  - `NAME_PATTERN: re.Pattern`, `READ_ONLY: tuple[str, ...]`, `CLAUDE_TOOLS: dict[str, tuple[str, ...]]`
  - `@dataclass(frozen=True) class AgentDefinition`: `name: str`, `description: str`, `prompt: str`, `tools: tuple[str, ...] | None` (our names; `None` = all), `disallowed: tuple[str, ...]`, `model: str | None` (`None` = configured model), `skills: tuple[str, ...]`, `max_turns: int | None`, `read_only: bool`, `color: str | None`, `source: str`, `frontmatter: dict` (as read, for round-trip)
  - `translate_tools(names: list[str], known: frozenset) -> tuple[tuple[str, ...], list[str]]` (resolved names in order without duplicates, warnings)
  - `resolve_model(value: str | None) -> tuple[str | None, str | None]` (model id or `None`, note or `None`)
  - `parse_agent_md(text: str, source: str, known_tools: frozenset) -> tuple[AgentDefinition | None, list[str]]`
  - `render_agent_md(frontmatter: dict, prompt: str) -> str`

- [ ] **Step 1: Write the failing tests** in `tests/test_agents.py`, class `FormatTests`:

```python
KNOWN = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
                   "write_todos", "task", "search_assets"})
CRITIC = "---\nname: critic\ndescription: Finds weaknesses.\ntools: Read, Grep, Glob\nmodel: sonnet\ncolor: red\nmemory: project\n---\nYou are a critic.\n"

def test_parses_a_claude_code_file(self):
    agent, notes = parse_agent_md(CRITIC, "folder", KNOWN)
    self.assertEqual((agent.name, agent.description, agent.prompt), ("critic", "Finds weaknesses.", "You are a critic."))
    self.assertEqual(agent.tools, ("read_file", "grep", "glob", "ls"))
    self.assertIsNone(agent.model)
    self.assertTrue(any("sonnet" in n for n in notes))
    self.assertTrue(any("memory" in n for n in notes))   # unknown field noted
    self.assertEqual(agent.color, "red")

def test_tools_as_list_crlf_and_bom(self):
    text = "﻿---\r\nname: lister\r\ndescription: Lists.\r\ntools:\r\n  - LS\r\n  - read_file\r\n---\r\nList things.\r\n"
    agent, _ = parse_agent_md(text, "f", KNOWN)
    self.assertEqual(agent.tools, ("ls", "read_file"))

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
    for text in ("no frontmatter", "---\nname: [oops\n---\nx", "---\nname: bad:name\ndescription: d\n---\nx",
                 "---\ndescription: d\n---\nx"):
        with self.subTest(text=text):
            agent, notes = parse_agent_md(text, "f.md", KNOWN)
            self.assertIsNone(agent)
            self.assertIn("f.md", notes[0])

def test_models(self):
    self.assertEqual(resolve_model("inherit"), (None, None))
    self.assertEqual(resolve_model(None), (None, None))
    self.assertIsNone(resolve_model("claude-opus-5-5")[0])
    self.assertEqual(resolve_model("nvidia/other-model"), ("nvidia/other-model", None))

def test_permission_modes(self):
    plan, _ = parse_agent_md("---\nname: a\ndescription: d\npermissionMode: plan\n---\nx", "f", KNOWN)
    self.assertTrue(plan.read_only)
    bypass, notes = parse_agent_md("---\nname: a\ndescription: d\npermissionMode: bypassPermissions\n---\nx", "f", KNOWN)
    self.assertFalse(bypass.read_only)
    self.assertTrue(any("bypassPermissions" in n and "ignored" in n for n in notes))

def test_max_turns_skills_disallowed(self):
    agent, _ = parse_agent_md("---\nname: a\ndescription: d\nmaxTurns: 5\nskills: [scene-planning]\ndisallowedTools: Bash\n---\nx", "f", KNOWN)
    self.assertEqual((agent.max_turns, agent.skills, agent.disallowed), (5, ("scene-planning",), ("execute",)))

def test_round_trip_keeps_unknown_fields(self):
    agent, _ = parse_agent_md(CRITIC, "f", KNOWN)
    again, _ = parse_agent_md(render_agent_md(agent.frontmatter, agent.prompt), "f", KNOWN)
    self.assertEqual(again.frontmatter, agent.frontmatter)
    self.assertEqual(again.prompt, agent.prompt)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_agents.py"`
Expected: ImportError (`deep_agents_demo.agent_files` doesn't exist).

- [ ] **Step 3: Implement `agent_files.py`** with the interfaces above. Strip a leading BOM and normalise CRLF before matching frontmatter (reuse `skill_manager.split_skill_md`'s regex approach, but parse with `yaml.safe_load` and catch `yaml.YAMLError`). Notes are strings prefixed with the source (e.g. `"f.md: model 'sonnet' is a Claude model; using the configured model"`). `Glob` expands to `("glob", "ls")`. A tool name already in `known` is kept as is. `render_agent_md` uses `yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)` like `render_skill_md`.

- [ ] **Step 4: Run the tests to verify they pass** (same command). Expected: OK.

- [ ] **Step 5: Commit**: `git add src/deep_agents_demo/agent_files.py tests/test_agents.py` and `git commit -m "Agent files: parse and write Claude Code agent format, tool translation, model and permission handling"`

---

### Task 2: Built-ins, folders and precedence

**Files:**
- Create: `src/deep_agents_demo/agent_registry.py`
- Create: `src/deep_agents_demo/seed/agents/README.md`, `src/deep_agents_demo/seed/agents/critic.md`
- Test: `tests/test_agents.py` (class `RegistryTests`)

**Interfaces:**
- Consumes: `parse_agent_md`, `AgentDefinition` (Task 1).
- Produces:
  - `BUILTIN_AGENTS: dict[str, AgentDefinition]` with `reviewer` (read-only, checks a result against a step and replies with a verdict `pass` or `fail` plus concrete notes) and `moderator` (no tools, `tools=()`; runs a discussion: picks the next speaker, keeps it on topic, ends with a summary of agreements, disagreements and next steps). `source="built-in"`.
  - `class AgentRegistry`: `@classmethod load(cls, folders: list[Path], known_tools: frozenset) -> "AgentRegistry"`; `.agents() -> list[AgentDefinition]` (sorted by name); `.get(name) -> AgentDefinition | None`; `.warnings: list[str]`; `.overview() -> list[dict]` with keys `name`, `source`, `tools` (`"all"` or comma list), `model` (`"configured"` or id), `overrides` (source it replaced, or `""`).

- [ ] **Step 1: Write the failing tests**, class `RegistryTests` (folders in a temp dir):

```python
def test_builtins_are_present(self):
    registry = AgentRegistry.load([], KNOWN)
    self.assertEqual([a.name for a in registry.agents()], ["moderator", "reviewer"])
    self.assertTrue(registry.get("reviewer").read_only)

def test_folders_load_recursively_and_skip_readme(self):
    write(self.a / "nested" / "critic.md", CRITIC); write(self.a / "README.md", "# notes")
    registry = AgentRegistry.load([self.a], KNOWN)
    self.assertEqual(registry.get("critic").source, str(self.a / "nested" / "critic.md"))

def test_later_folder_wins_and_says_so(self):
    write(self.a / "c.md", CRITIC); write(self.b / "c.md", CRITIC.replace("Finds weaknesses.", "Newer."))
    registry = AgentRegistry.load([self.a, self.b], KNOWN)
    self.assertEqual(registry.get("critic").description, "Newer.")
    self.assertTrue(any("critic" in w and "overrides" in w for w in registry.warnings))

def test_duplicate_in_one_folder_first_sorted_path_wins(self):
    write(self.a / "a.md", CRITIC); write(self.a / "b.md", CRITIC.replace("Finds weaknesses.", "Second."))
    registry = AgentRegistry.load([self.a], KNOWN)
    self.assertEqual(registry.get("critic").description, "Finds weaknesses.")
    self.assertTrue(any("b.md" in w for w in registry.warnings))

def test_missing_folder_is_a_warning(self):
    registry = AgentRegistry.load([self.root / "nope"], KNOWN)
    self.assertTrue(any("nope" in w for w in registry.warnings))

def test_user_agent_replaces_a_builtin(self):
    write(self.a / "r.md", "---\nname: reviewer\ndescription: Mine.\n---\nx")
    registry = AgentRegistry.load([self.a], KNOWN)
    row = next(r for r in registry.overview() if r["name"] == "reviewer")
    self.assertEqual(row["overrides"], "built-in")

def test_seed_example_is_valid(self):
    agent, notes = parse_agent_md((SEED_DIR / "agents" / "critic.md").read_text(encoding="utf-8"), "seed", KNOWN)
    self.assertIsNotNone(agent, notes)
```

- [ ] **Step 2: Run them to verify they fail.** Expected: ImportError.

- [ ] **Step 3: Implement `agent_registry.py`** and the seed files. `critic.md` is a Claude Code–style example (`tools: Read, Grep, Glob`, `model: inherit`) with a short critic prompt; `README.md` explains the folder in 5–10 lines (format, link to the README section, that files here are loaded at start and the agent can't edit them). `seed_files` already copies new seed files, so `demo_workspace/agents/` appears on next start.

- [ ] **Step 4: Run the tests.** Expected: OK.

- [ ] **Step 5: Commit** (`agent_registry.py`, seed files, tests): `git commit -m "Agent registry: built-in reviewer and moderator, folder loading with precedence and warnings, seeded example"`

---

### Task 3: Per-agent tool allowlist and denylist in ToolAccess

**Files:**
- Modify: `src/deep_agents_demo/tools.py` (`ToolAccess.__init__`, `hidden`, `refusal`)
- Test: `tests/test_tools.py` (new methods in `AccessTests`)

**Interfaces:**
- Produces: `ToolAccess(disabled=frozenset(), skill_only=frozenset(), requires=None, allowed: frozenset | None = None, denied: frozenset = frozenset())`. With `allowed` set, every tool whose name isn't in it is hidden and refused; `denied` names are hidden and refused; refusal text `f"Error: {name} isn't available to this agent."`. Skill-only names in `allowed` are visible from the start (`skill_only` minus `allowed`). Existing behaviour (DISABLED_TOOLS wins, shell equivalents, skill gates) unchanged.

- [ ] **Step 1: Write the failing tests** (unit-level, on a fake request like `WindowTests` in `test_images.py`):
  - `test_allowlist_hides_and_refuses_other_tools`: `ToolAccess(allowed=frozenset({"read_file"}))` → `hidden(...)` contains `"write_file"`; a `write_file` call's refusal contains `"isn't available to this agent"`; `read_file` → no refusal.
  - `test_denylist`: `denied={"execute"}` → `execute` refused, `read_file` not.
  - `test_disabled_beats_allowlist`: `ToolAccess(disabled={"read_file"}, allowed={"read_file"})` → refusal says `"turned off"`.
  - `test_listed_skill_only_tool_is_visible_at_once`: `ToolAccess(skill_only={"search_assets"}, allowed={"search_assets"})` → `"search_assets"` not in `hidden(state={}, messages=[])`.

- [ ] **Step 2: Run them to verify they fail.** Expected: TypeError (unexpected keyword `allowed`).

- [ ] **Step 3: Implement** the new parameters. `hidden(state, messages)` needs the request's tool names for the allowlist; add `_request` filtering as `tool_name(t) not in hidden and (allowed is None or tool_name(t) in allowed)` rather than enumerating every name.

- [ ] **Step 4: Run the full suite.** Expected: OK (existing ToolAccess tests unchanged).

- [ ] **Step 5: Commit**: `git commit -m "ToolAccess: per-agent allowlist and denylist"`

---

### Task 4: Agent definitions become subagents

**Files:**
- Modify: `src/deep_agents_demo/agent.py` (`build_agent`)
- Modify: `src/deep_agents_demo/agent_registry.py` (add `PinSkills`)
- Test: `tests/test_agents.py` (class `SubagentTests`, end-to-end with fake models)

**Interfaces:**
- Consumes: `AgentDefinition` (Task 1), `ToolAccess(..., allowed, denied)` (Task 3), `READ_ONLY` (Task 1).
- Produces:
  - `build_agent(model, checkpointer, backend, features=(), *, skills_prefix="", disabled_tools=(), agents=(), model_for=None, max_calls=25)` — `agents: Sequence[AgentDefinition]`; `model_for: Callable[[str], BaseChatModel] | None` (model id → chat model; `None` means every agent uses `model`).
  - `class PinSkills(AgentMiddleware)`: `__init__(self, names: tuple[str, ...])`; `before_agent` returns `{"pinned_skills": list(names)}` (Deep Agents' `SkillsMiddleware` pins them before the first model call).
  - `subagent_spec(definition, *, model, model_for, skill_sources, access_args, max_calls) -> dict` (private helper is fine; name it `_spec`).

Spec rules for each definition: `name`, `description`, `system_prompt=prompt`; `model = model_for(definition.model)` when both set, else `model`; `skills = skill_sources`; `tools` omitted when `definition.tools is None` (inherits the main agent's feature tools), else the main agent's feature tools whose names are in the allowlist; middleware = `ToolAccess(off, skill_only, requires, allowed=..., denied=frozenset(definition.disallowed))`, `ModelCallLimitMiddleware(run_limit=min(max_calls, definition.max_turns or max_calls), exit_behavior="end")`, plus `PinSkills(definition.skills)` when it has skills. `allowed` = `frozenset(READ_ONLY)` if `read_only`, else `frozenset(tools)` if `tools is not None`, else `None`. A definition named `general-purpose` or `researcher` replaces the demo's own spec of that name; the demo's own `general-purpose`/`researcher` specs also get the call cap. `interrupt_on` is not set on specs, so they inherit the main agent's approvals (as today).

- [ ] **Step 1: Write the failing tests** (build with `FilesystemBackend` in a temp dir, a fake model that records bound tool names per call — reuse the `Model`/`SEEN` pattern from `test_tools.py` — and drive a `task` call to the agent under test):
  - `test_a_user_agent_is_offered_to_the_main_agent`: build with `agents=[critic]` and `create_deep_agent` patched; the `subagents` kwarg contains a spec named `critic` with `description == "Finds weaknesses."` and `system_prompt == "You are a critic."`.
  - `test_agent_sees_only_its_tools`: `critic` (tools read_file, grep, glob, ls); during its run the bound tool names ⊆ `{"read_file","grep","glob","ls"}` and a `write_file` call is refused (`"isn't available to this agent"`) and the file doesn't exist.
  - `test_read_only_agent_cannot_write`: `permissionMode: plan` agent → `write_file` refused.
  - `test_disabled_tools_apply_inside_agents`: `disabled_tools=("write_file",)`, agent with all tools → `write_file` refused with `"turned off"`.
  - `test_approvals_apply_inside_agents_even_with_bypass`: agent file with `permissionMode: bypassPermissions` and all tools, `Guardrails.feature()` on, a `delete` call inside the agent → `stream_turn` with an `Approver` whose `ask` raises `AssertionError("asked")` → the assertion is raised (approval was requested).
  - `test_overriding_general_purpose_keeps_limits`: a user `general-purpose` with all tools and `disabled_tools=("write_file",)` → `write_file` refused inside it; only one spec named `general-purpose` passed to `create_deep_agent` (patch `create_deep_agent` as `test_workspace.FeatureAssemblyTests.build` does).
  - `test_call_cap`: `max_calls=2`, an agent whose fake responses are 5 `ls` calls → the subagent's model is called at most 2 times.
  - `test_max_turns_lowers_cap`: `maxTurns: 1`, `max_calls=25` → spec's `ModelCallLimitMiddleware.run_limit == 1` (inspect the patched `create_deep_agent` kwargs).
  - `test_skills_are_pinned`: agent with `skills: [scene-planning]` (seed skill copied into the temp backend) → its first model call's messages include a `HumanMessage` whose content contains `<skill name="scene-planning"`.
  - `test_model_for_is_used_for_explicit_models`: agent with `model: nvidia/other` and `model_for=lambda m: marker_model` → that spec's `model is marker_model`; an `inherit` agent gets `model`.

- [ ] **Step 2: Run them to verify they fail.** Expected: TypeError (`build_agent` has no `agents` parameter).

- [ ] **Step 3: Implement** in `agent.py` and `PinSkills` in `agent_registry.py`.

- [ ] **Step 4: Run the full suite.** Expected: OK. `build_agent` gets no agents by default, so the existing `test_workspace` subagent assertions (`general-purpose`, `researcher`) stand unchanged.

- [ ] **Step 5: Commit**: `git commit -m "Subagents from agent definitions: own tools, read-only mode, pinned skills, call cap, approvals and DISABLED_TOOLS inside"`

---

### Task 5: Settings, wiring, `/agents`, docs

**Files:**
- Modify: `src/deep_agents_demo/config.py` (`Settings.agent_dirs: tuple[Path, ...] = ()`, `Settings.subagent_max_calls: int = 25`; read `AGENT_DIRS` split on `os.pathsep`, `~` expanded; `SUBAGENT_MAX_CALLS` must be ≥ 1)
- Modify: `src/deep_agents_demo/app.py` (`chat_model(settings, model=None)`; in `build_runtime`: after `ToolRegistry`, `AgentRegistry.load([*settings.agent_dirs, settings.paths.workspace / "agents"], registry.names())`, `say` each warning prefixed `"Agents: "`, pass `agents=..., model_for=lambda m: chat_model(settings, m), max_calls=settings.subagent_max_calls` to `build_agent`; `Runtime.agents` field)
- Modify: `src/deep_agents_demo/agent.py` (deny writes to `f"{skills_prefix}/agents/**"` alongside skill folders)
- Modify: `src/deep_agents_demo/chat.py` (`ChatSession(..., agents=None)`; `@command("agents", usage="/agents", summary="list the subagents you can use")` printing a table: NAME, SOURCE (`built-in`, `folder: <file>`), TOOLS, MODEL, and `(overrides built-in)` where relevant)
- Modify: `.env.example` (`AGENT_DIRS=`, `SUBAGENT_MAX_CALLS=25` with one comment line each), `README.md` (a "Subagents" section: the file format with the critic example, the translation table, folders and precedence, `/agents`, what's ignored, limits; layout lines for `agent_files.py` and `agent_registry.py`)
- Test: `tests/test_config.py`, `tests/test_agents.py` (class `WiringTests`)

**Interfaces:**
- Consumes: `AgentRegistry.load/overview/warnings` (Task 2), `build_agent(..., agents, model_for, max_calls)` (Task 4).
- Produces: `Runtime.agents: AgentRegistry | None`; `Settings.agent_dirs`, `Settings.subagent_max_calls`.

- [ ] **Step 1: Write the failing tests**:
  - `test_config.py`: `AGENT_DIRS` with two paths joined by `os.pathsep` → `agent_dirs == (Path(a), Path(b))`; `SUBAGENT_MAX_CALLS=0` → `ValueError` mentioning `SUBAGENT_MAX_CALLS`; default 25.
  - `WiringTests.test_agents_command_lists_sources`: `ChatSession(chats=None, current="c", say=said.append, agents=registry_with_critic)`; `handle("/agents")` → output contains `critic`, `reviewer`, `built-in`, and `read_file`.
  - `WiringTests.test_agent_folder_is_read_only_to_the_agent`: build with a `write_file` call to `/agents/evil.md` → `permission denied`, file not created.

- [ ] **Step 2: Run them to verify they fail.**

- [ ] **Step 3: Implement** the changes listed under Files.

- [ ] **Step 4: Run the full suite.** Expected: OK. Then smoke-build the runtime with placeholder keys, as done for the tools registry (`OPENAI_API_KEY=unused AGENT_MODEL=unused MEMORY_ENABLED=false RAG_ENABLED=false SANDBOX_PROVIDER=none`, `build_runtime` in a script): it builds, and `AGENT_DIRS=/nope` prints an `Agents:` warning.

- [ ] **Step 5: Commit**: `git commit -m "Subagents: AGENT_DIRS and SUBAGENT_MAX_CALLS settings, registry wired into the runtime, /agents command, docs"`

---

### Task 6: Live check with the NVIDIA model

No code. With memory and documents off (`MEMORY_ENABLED=false RAG_ENABLED=false`) and user `tools-check`:

- [ ] `/agents` lists `critic` (folder), `reviewer` and `moderator` (built-in).
- [ ] "Ask the critic agent to review this plan: store passwords in plain text in a config file." → a `task` call to `critic`, its tools limited to read-only, a critique that flags plain-text passwords.
- [ ] Drop a real Claude Code agent file (e.g. with `tools: Read, Grep`, `model: sonnet`, `color: blue`, `memory: project`) into `demo_workspace/agents/`; restart → it loads with notes for `sonnet` and `memory`, and can be delegated to.
- [ ] Clean up: delete test chats, remove the dropped test file.

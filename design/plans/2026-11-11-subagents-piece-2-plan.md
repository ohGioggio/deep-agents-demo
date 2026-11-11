# Subagents registry, piece 2: agents written through the chat, and the marketplace — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the agent create, update, remove, import and restore subagents on request, each change approved by the user and saved as a version in SQLite; let users publish agents to a marketplace and install them; make changes available from the next message; and clear piece 1's deferred review items.

**Architecture:** Reuse the skills machinery instead of copying it. `SkillStore` takes its table names, so agents get the same versioned store in `data/agents.sqlite`. `SkillManager` and `SkillChange` gain a few hooks (main file name, noun, validation, what happens after apply), and `AgentManager` subclasses them. `AgentRegistry.load` adds the two database sources above the folders. The runtime rebuilds the agent after an approved agent change.

**Tech Stack:** Python 3.11+, SQLite, LangChain `HumanInTheLoopMiddleware` (via `Feature.approvals`), Deep Agents 0.7.34, unittest with fake models.

**Spec:** `design/specs/2026-11-11-subagents-registry-design.md` (sections 2 and 3; piece 2). Piece 1 is merged (`main` at `266a5df`); its plan is `design/plans/2026-11-11-subagents-piece-1-plan.md`.

## Global Constraints

- Precedence, higher wins: built-ins < folders (`AGENT_DIRS`, then `demo_workspace/agents/`) < installed marketplace agents < the user's own agents in SQLite.
- Changes only on request; every gated tool pauses for approval with the existing prompt (summary, `[d]iff`, `[f]ull file`, approve or reject with a reason). With `--prompt` they are refused.
- Gated tools: `create_agent`, `update_agent`, `remove_agent`, `import_agent`, `restore_agent`, `publish_agent`, `install_agent`, `uninstall_agent`. Read-only: `list_agents`, `agent_history`, `browse_agents`.
- Scopes `user` (default) and `marketplace`, exactly as skills; marketplace owner is `""`.
- Validated before the user is asked: the file parses with `parse_agent_md` (not skipped), name rules, every tool in a `tools` argument resolves (create/update: an error; import: a warning), no credentials (`memory_policy.find_secrets` credentials are errors, wording a warning), at most 100,000 characters.
- Versioned: every approved change is a version; remove and restore add versions; installed agents follow the latest marketplace version.
- Stored as one file per version: `{"agent.md": <Claude Code–format text>}`.
- A change is available from the user's next message (the runtime rebuilds the agent).
- Agents written through the chat use this agent's tool names in `tools`; Claude Code names in a `tools` argument are accepted and translated.
- The agent still cannot create tools; `tools` may only name existing ones.
- Commits: one-line messages, no trailer; don't push unless the user asks.
- Tests: `uv run python -m unittest discover -s tests` (single file: `-p "test_agent_manager.py"`).

## Review Focus

1. **Approving an agent change mid-chat, then delegating to it in the next message** — expected: the new agent is used without restarting. (Task 5 end-to-end test.)
2. **An agent with the same name in the database and in a folder** — expected: the database one wins and `/agents` says it overrides the folder file. (Task 5 test.)
3. **A marketplace agent removed after a user installed it** — expected: it disappears from that user's agents, `list_agents` says "removed from the marketplace", nothing crashes. (Task 4 test.)
4. **Importing a whole `.claude/agents` folder** — expected: a clear error listing the `.md` files found and asking to import them one at a time (each needs its own approval). (Task 3 test.)
5. **A rejected or invalid change** — expected: nothing saved, the agent is told why, and the running agent is not rebuilt. (Task 3 and Task 5 tests.)

---

### Task 1: Piece 1's deferred review items

**Files:**
- Modify: `src/deep_agents_demo/agent_files.py`, `src/deep_agents_demo/agent.py`, `src/deep_agents_demo/chat.py`, `src/deep_agents_demo/config.py`, `.env.example`
- Test: `tests/test_agents.py`, `tests/test_config.py`

**Interfaces:** no new names; behaviour fixes only.

- [ ] **Step 1: Write the failing tests** (in `FormatTests` unless noted):
  - `test_max_turns_must_be_a_real_number`: `maxTurns: true` → `agent.max_turns is None` and a note containing `maxTurns`.
  - `test_model_must_be_text`: `model: [x]` → agent loads with `model is None` and a note containing `model`.
  - `test_skills_must_be_names`: `skills: {a: b}` → `skills == ()` and a note containing `skills`; `skills: one two` → `("one", "two")`.
  - `test_name_with_trailing_newline_is_rejected`: frontmatter `name: "critic\n"` → `None`.
  - `test_frontmatter_tolerates_blank_first_line_and_trailing_spaces`: `"\n---  \nname: a\ndescription: d\n---\nx"` loads.
  - `test_subagents_can_plan` (`SubagentTests`): an agent with all tools → its first bound tools include `write_todos`; a `READ_ONLY_AGENT` too.
  - `test_agents_footer_matches_reality` (`WiringTests`): `/agents` output no longer contains "always there"; a user agent named `general-purpose` in a folder shows `(overrides built-in)`.
  - `test_config.py::test_bad_home_in_agent_dirs_is_a_settings_error`: patch `pathlib.Path.expanduser` to raise `RuntimeError("Could not determine home directory.")`; `Settings.from_env({..., "AGENT_DIRS": "~nobody/x"})` raises `ValueError` mentioning `AGENT_DIRS`.
- [ ] **Step 2: Run them; verify each fails for the stated reason.**
- [ ] **Step 3: Implement:**
  - `maxTurns`: reject `bool`;
  - `model`: non-string → note and `None`;
  - `skills`: a string or a list of strings, anything else → note and `()`;
  - `NAME_PATTERN.fullmatch`;
  - strip leading blank lines and allow trailing spaces after both `---` lines;
  - write the BOM as `"\ufeff"` in source;
  - `_spec` adds `TodoListMiddleware()` to every user agent's middleware (`write_todos` stays subject to the allowlist);
  - `AgentRegistry` records an override for every replaced source, including the built-ins `general-purpose`/`researcher` (add both names to the override check as "built-in");
  - `/agents` footer: `"general-purpose (and researcher, when documents or the catalog are on) are built in unless an agent replaces them."`;
  - `config.py`: catch `RuntimeError` from `expanduser` and raise `ValueError("AGENT_DIRS: ...")`;
  - `.env.example`: one comment line, "Use forward slashes (C:/Users/me/.claude/agents) or no quotes; quoted backslashes are mangled."
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Agents: stricter frontmatter types, subagents can plan, /agents shows every override, clear AGENT_DIRS errors"`

---

### Task 2: Make the skills store and manager reusable (no behaviour change)

**Files:**
- Modify: `src/deep_agents_demo/skill_store.py`, `src/deep_agents_demo/skill_manager.py`
- Test: `tests/test_skills.py` (existing suite must stay green), new `tests/test_agent_manager.py` (class `StoreTests`)

**Interfaces:**
- Produces:
  - `SKILL_TABLES = ("skills", "skill_versions", "installs")`, `AGENT_TABLES = ("agents", "agent_versions", "agent_installs")` in `skill_store.py`; `SkillStore(path: Path, tables: tuple[str, str, str] = SKILL_TABLES)`. Same methods and behaviour; SQL uses the given table names.
  - On `SkillChange`: class attributes `noun = "Skill"` and `main_file = "SKILL.md"`, and method `tools_used(self) -> list[str]` (skills: `declared_tools(...)`). `summary()` starts with `f"{self.noun} change requested: ..."` and reads the description from `main_file`.
  - On `SkillManager`: class attributes `noun = "skill"`, `change_type = SkillChange`, `main_file = "SKILL.md"`; methods `validate(self, name, files) -> list[str]`, `warnings(self, files) -> list[str]` (skills: the module functions), `after_apply(self) -> None` (skills: `self.sync_folder()`). `_change`, `plan_install`, `_apply` and `browse` use these instead of the module functions and literals; user-facing error strings use `self.noun` (e.g. `f"no {scope} {self.noun} named {name!r}"`).

- [ ] **Step 1: Write the failing test** `StoreTests.test_agent_tables_are_separate`: one temp db file opened twice, as `SkillStore(p)` and `SkillStore(p, AGENT_TABLES)`; save `("user", "alice", "x")` in each with different files → each `get` returns its own; `history` lengths are independent.
- [ ] **Step 2: Run it.** Expected: TypeError (unexpected argument).
- [ ] **Step 3: Implement** the interfaces. Keep the existing `skills.sqlite` working: the default table names are unchanged, so no migration.
- [ ] **Step 4: Run the full suite.** Expected: OK, with `test_skills.py` unchanged and green.
- [ ] **Step 5: Commit**: `git commit -m "Skill store and manager take their table names and hooks, ready for agents"`

---

### Task 3: AgentManager for the user's own agents

**Files:**
- Create: `src/deep_agents_demo/agent_manager.py`
- Test: `tests/test_agent_manager.py` (class `ManagerTests`)

**Interfaces:**
- Consumes: `SkillManager`, `SkillChange`, `USER`, `MARKETPLACE` (Task 2); `parse_agent_md`, `render_agent_md`, `translate_tools` (piece 1); `find_secrets`, `split_findings` (`memory_policy`).
- Produces:
  - `AGENT_FILE = "agent.md"`, `MAX_AGENT_SIZE = 100_000`, `GATED_AGENT_TOOLS = ("create_agent", "update_agent", "remove_agent", "import_agent", "restore_agent", "publish_agent", "install_agent", "uninstall_agent")`.
  - `class AgentChange(SkillChange)`: `noun = "Agent"`, `main_file = AGENT_FILE`; `tools_used()` returns the frontmatter `tools` list (or `["all"]` when absent).
  - `class AgentManager(SkillManager)`: `__init__(self, store, owner: str, known_tools: frozenset)`; `noun = "agent"`, `change_type = AgentChange`, `main_file = AGENT_FILE`; `changed: bool` (set by `after_apply`, cleared by the runtime).
    - `plan_create(name, description, prompt, tools=None, model=None, skills=None, scope=USER) -> AgentChange`
    - `plan_update(name, description=None, prompt=None, tools=None, model=None, skills=None, scope=USER) -> AgentChange` (only given fields change; `tools=[]` removes the `tools` line, meaning all tools)
    - `plan_import(path, scope=USER)` (one `.md` file; a folder → error listing up to 20 `.md` files inside and saying to import them one at a time)
    - inherited, unchanged: `plan_remove`, `plan_restore`, `plan_publish`, `plan_install`, `plan_uninstall`, `plan(tool, args)` (with the agent tool names), `apply(change)`
    - `definitions() -> tuple[list[AgentDefinition], list[AgentDefinition], list[str]]`: installed marketplace agents (source `"marketplace"`), the user's own (source `"yours"`), and notes (e.g. an installed agent removed from the marketplace).
    - `tools() -> list[Callable]` and `feature() -> Feature` come in Task 5.

- [ ] **Step 1: Write the failing tests** (temp db, `KNOWN` from `test_agents.py`):
  - `test_create_renders_a_claude_code_file`: `plan_create("critic", "Finds weaknesses.", "You are a critic.", tools=["Read", "grep"])` → no errors; `after["agent.md"]` parses to tools `("read_file", "grep")`; `summary()` starts `"Agent change requested: CREATE 'critic'"` and contains `"Tools it uses: read_file, grep"`.
  - `test_unknown_tool_is_an_error`: `tools=["make_coffee"]` → errors mention `make_coffee`.
  - `test_credentials_are_errors`: prompt containing `sk-` + 40 letters → an error; wording "rotate the password monthly" → a warning only.
  - `test_update_changes_only_given_fields`: after create+apply, `plan_update("critic", model="nvidia/other")` → diff touches only the `model:` line; `tools=[]` removes the `tools` line.
  - `test_remove_and_restore_add_versions`: create, update, remove, restore v1 → `store.history` actions `["create", "update", "remove", "restore"]`, live text equals v1.
  - `test_import_one_file_and_refuse_a_folder`: importing a file with `model: sonnet` → change with a warning about `sonnet`; importing its folder → error listing the file and "one at a time".
  - `test_apply_marks_changed_and_rejected_does_not`: an invalid change applied → `{"saved": False, ...}` and `changed is False`; a valid one → `changed is True`.
  - `test_definitions_give_the_users_agents`: after create → `definitions()[1][0].name == "critic"`, source `"yours"`.
- [ ] **Step 2: Run them.** Expected: ImportError.
- [ ] **Step 3: Implement** `agent_manager.py` with the interfaces. `validate` = `parse_agent_md` must return an agent (its skip reason is the error) + name equals the stored name + size limit + credential errors; `warnings` = parse notes + credential wording. `tools` arguments are checked with `translate_tools`; anything unresolved is an error.
- [ ] **Step 4: Run the tests and the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "AgentManager: plan, validate, version and restore the user's agents"`

---

### Task 4: The agent marketplace

**Files:**
- Modify: `src/deep_agents_demo/agent_manager.py`
- Test: `tests/test_agent_manager.py` (class `MarketplaceTests`)

**Interfaces:**
- Consumes: Task 3.
- Produces: `AgentManager.browse(query="") -> list[dict]` (keys `name`, `description`, `version`, `installed`); `AgentManager.overview() -> list[dict]` (keys `name`, `source` in `{"yours", "marketplace (installed)"}`, `version`, `note`); `definitions()` includes installed agents at the marketplace's latest version.

- [ ] **Step 1: Write the failing tests** (two managers on one store: `alice`, `bob`):
  - `test_publish_install_follow_latest`: alice creates and publishes `critic`; bob installs it → bob's `definitions()[0]` has it at v1; alice publishes an update → bob's definition shows the new description without reinstalling.
  - `test_install_is_refused_twice_and_for_unknown_names`.
  - `test_removed_from_marketplace`: bob installs; marketplace `critic` removed → bob's `definitions()[0] == []` and `definitions()[2]` mentions "removed from the marketplace"; `overview()` row note says the same.
  - `test_browse_filters_and_marks_installed`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** by overriding only what differs from skills (descriptions read from `agent.md`).
- [ ] **Step 4: Run the tests and the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Agent marketplace: publish, install (following the latest version), uninstall, browse"`

---

### Task 5: Tools, approvals, precedence and rebuild after a change

**Files:**
- Modify: `src/deep_agents_demo/agent_manager.py` (`tools()`, `feature()`), `src/deep_agents_demo/agent_registry.py` (`load(..., installed=(), own=())`), `src/deep_agents_demo/app.py` (store, manager, preview, `Runtime.refresh`), `src/deep_agents_demo/chat.py` (refresh after each turn), `README.md`
- Test: `tests/test_agent_manager.py` (classes `ToolTests`, `RuntimeTests`)

**Interfaces:**
- Consumes: Tasks 2–4; `AgentRegistry.load` (piece 1); `Approver`, `stream_turn`.
- Produces:
  - `AgentRegistry.load(folders, known_tools, installed=(), own=())`; sources shown as `marketplace` / `yours`; overrides recorded.
  - `AgentManager.tools()`: the 8 gated tools and `list_agents()`, `agent_history(name, scope="user")`, `browse_agents(query="")`; gated tools plan and apply under the manager's lock (as `SkillManager.tools` does); `create_agent(name, description, prompt, tools=None, model=None, skills=None, scope="user")`, `update_agent(name, description=None, prompt=None, tools=None, model=None, skills=None, scope="user")`.
  - `AgentManager.feature() -> Feature(name="agents", prompt=..., tools=..., approvals=GATED_AGENT_TOOLS)`. Prompt (one paragraph): agents are the user's subagents in Claude Code's format; change them only when asked; tools must be existing tool names; changes need approval; a change is available from the next message; `list_agents` shows every source.
  - `Runtime.refresh(self) -> bool`: if `agent_manager.changed`, reload the `AgentRegistry` (folders + `definitions()`), rebuild the agent with the same arguments as `build_runtime`, set `runtime.agent`, `runtime.chats.agent` and the session's registry, clear `changed`, and return `True`.
  - In `run_chat`: after each `stream_turn`, `if runtime.refresh(): say("Agents updated.")` and point `session.agents` at the new registry.
  - `build_runtime`: `AgentStore = SkillStore(settings.paths.data / "agents.sqlite", AGENT_TABLES)`, closed by the stack; `preview` also tries `agents.plan(name, args)`; `history_runtime` unchanged.

- [ ] **Step 1: Write the failing tests:**
  - `ToolTests.test_agent_tools_need_approval`: build an agent with `[manager.feature()]`; a `create_agent` call with an `Approver` whose `ask` answers `"r"` then `""` → nothing saved; with `"a"` → saved; with `interactive=False` → refused.
  - `ToolTests.test_registry_precedence_with_database`: folder `critic` + own `critic` → registry source `yours`, `overview()` row `overrides` is the folder file path.
  - `RuntimeTests.test_new_agent_is_used_from_the_next_message`: call the real `build_runtime` with placeholder settings (`OPENAI_API_KEY=unused`, `AGENT_MODEL=unused`, memory and RAG off, `SANDBOX_PROVIDER=none`, `Paths` in a temp dir) and `app.chat_model` patched to return one shared fake model. Turn 1: the model calls `create_agent(name="helper", ...)`, approved → `runtime.refresh()` returns `True`. Turn 2: the model calls `task(subagent_type="helper")` → a model call whose system prompt is `helper`'s prompt is seen.
  - `RuntimeTests.test_rejected_change_does_not_rebuild`: a rejected `create_agent` → `refresh()` returns `False`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** the interfaces, and add a README section, "Agents the agent writes, with your approval": the tools table, scopes, versions, the marketplace, available from the next message, and folder vs database precedence.
- [ ] **Step 4: Run the full suite**, then smoke-build the runtime with placeholder settings (as in piece 1). Expected: OK, and the agent builds.
- [ ] **Step 5: Commit**: `git commit -m "Agents through the chat: gated tools with approval, database precedence, rebuild after a change, docs"`

---

### Task 6: Live check with the NVIDIA model

No code. Memory and documents off, user `tools-check`, interactive chat with piped input (use `MSYS_NO_PATHCONV=1` in Git Bash):

- [ ] "Create an agent called tester that writes unit tests, read and write files only" → the approval prompt shows `Agent change requested: CREATE 'tester'` with its tools; `d` shows the diff; approve → "Agents updated."; `/agents` lists `tester` (source `yours`); the next message "ask the tester agent to …" delegates to it.
- [ ] "Publish the tester agent", then as user `bob` "install tester" → bob's `/agents` lists it as marketplace.
- [ ] "Remove the tester agent" → approval, removed, `/agents` no longer lists it for alice (bob's installed copy still follows the marketplace).
- [ ] Clean up: remove the test agents from both scopes, uninstall for `bob`, delete test chats.

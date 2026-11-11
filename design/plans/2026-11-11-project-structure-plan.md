# Project structure hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise `src/deep_agents_demo/` into layers (config/models → persistence/providers → core → features → startup → ui) with one folder per feature, split the engine from the CLI through events and ports, open shared resources once per process, and enforce the layering with a test — with no behaviour change except the UI seam, and the CLI's text unchanged.

**Architecture:** A layering test (`tests/test_structure.py`) is written first with a `LEGACY` set of not-yet-moved flat modules it skips; each task moves modules into their layer and removes them from `LEGACY`, so every move is checked as it lands, and the last task empties the set. Moves are `git mv` plus import rewrites; splits are called out per task. The UI seam (events, `Interaction`/`EventSink`/`Interruptible` ports, a CLI renderer that prints exactly today's text) comes after the moves.

**Tech Stack:** Python 3.11+, `ast` (rule test), `typing.Protocol`, frozen dataclasses, unittest; no new dependencies.

**Spec:** `design/specs/2026-11-11-project-structure-design.md` (committed `10edf2d`).

## Global Constraints

- Layer order: `config, models → persistence, providers → core → features → startup → ui`; a module imports only its own layer or lower; a feature never imports another feature; `print(`, `input(`, `signal.` only under `ui/`.
- `config.py` and `seed/` stay at the package root; `config.py` keeps validating `SANDBOX_PROVIDER` names itself.
- No compatibility shims at old paths; no new dependencies; middleware class names unchanged (LangGraph node names).
- `SkillStore` becomes `VersionedStore` (table names unchanged: existing `data/*.sqlite` keep working); thread ids unchanged (existing chats keep working).
- The CLI's text is unchanged; tests keep their exact-string assertions (through `RecordingUI` rendering with the real renderer once Task 7 lands).
- Script entry point: `deep-agents-demo = "deep_agents_demo.ui.cli.main:main"`.
- Every commit: `uv run python -m unittest discover -s tests` green (323 tests + new ones); after each move, `grep -rn "deep_agents_demo\.<old>" src tests evals` and string patch targets are empty.
- Commits: one-line messages, no trailer; push only if asked.

## Review Focus

1. **The installed `deep-agents-demo` command after the entry point moves** — expected: works after `uv sync`; `python -m deep_agents_demo` works too. (Task 8 test: the pyproject target imports and `__main__` calls it.)
2. **Seeding after `workspace.py` moves** — expected: `SEED_DIR` still finds the package's `seed/`; a fresh workspace gets skills, agents and memories. (Task 5 test, extending `test_workspace`.)
3. **String patch targets in tests** (e.g. `"deep_agents_demo.rooms.time.strftime"`, `patch.object(app, "chat_model")`) — expected: updated with each move, never silently patching a dead path. (Every task's grep step; Task 8's `test_removed_modules_are_not_imported` also rejects them.)
4. **Exact CLI output** (approval prompt, `[plan] step 1/2 - ...`, `[room] moderator -> ...`, `/agents`, `/threads`, quiet `--prompt`) — expected: byte-for-byte as today. (Task 6 render tests + existing assertions via `RecordingUI`.)
5. **Existing data** — expected: `data/skills.sqlite`, `data/agents.sqlite` and `demo_checkpoints.sqlite` from before the refactor open and work unchanged. (Task 2 test opens a store created with today's table names.)

---

### Task 1: The layering test, and the leaves that break the import cycle

**Files:**
- Create: `tests/test_structure.py`, `src/deep_agents_demo/core/__init__.py`, `core/feature.py`, `core/tools.py`, `core/middleware.py`, `models/__init__.py`, `models/agents.py`
- Modify: `agent.py`, `tools.py` (removed), `agent_files.py`, `agent_registry.py`, `orchestration.py`, and every importer
- Test: `tests/test_structure.py`

**Interfaces:**
- Produces:
  - `tests/test_structure.py`: `LAYERS = {"config": 0, "models": 0, "persistence": 1, "providers": 1, "core": 2, "features": 3, "startup": 4, "ui": 5}`; `LEGACY: set[str]` (flat modules not yet moved, by file stem); `layer_of(module_path) -> str | None`; `imports_of(path) -> list[str]` (all `deep_agents_demo.*` imports, top-level and inside functions, via `ast`). Tests: `test_layers_only_import_downward`, `test_features_do_not_import_each_other` (`features.X` may import `features.X` only), `test_terminal_io_only_in_ui` (no `print(`/`input(` call and no `signal.` attribute use outside `ui/`, found via `ast`; a `say=print` default is a name, not a call, and is allowed). `IO_PENDING: set[str]` lists moved modules, by new dotted path, that still do terminal IO: it starts empty, Task 4 adds `features.orchestration.runner` and `features.orchestration.plans`, and Task 7 empties it. (Today only `approvals`, `chat`, `commands`, `embeddings` and `orchestration` print, read input or use `signal`.) A module in `LEGACY` is skipped as an importer; a moved module may import a `LEGACY` module (it has no layer yet) until that module moves, after which the normal layer rule applies.
  - `core/feature.py`: `Feature`, `approval_configs` (moved verbatim from `agent.py`).
  - `models/agents.py`: `AgentDefinition`, `READ_ONLY` (moved from `agent_files.py`).
  - `core/tools.py`: `BUILTIN`, `CORE`, `tool_name`, `ToolEntry`, `ToolRegistry`, `SHELL_EQUIVALENTS`, `shell_block` (from `tools.py`).
  - `core/middleware.py`: `ToolAccess` (from `tools.py`), `SubagentCallCap`, `CallCapState`, `partial_result`, `PinSkills` (from `agent_registry.py`), `TaskLimiter` (from `orchestration.py`).
- [ ] **Step 1: Write `tests/test_structure.py`** with `LEGACY` = every current flat module stem except those this task moves, plus `test_no_import_cycle_through_agent`: importing `deep_agents_demo.core.feature` must not import `deep_agents_demo.orchestration` (check `sys.modules` in a subprocess: `python -c "import deep_agents_demo.core.feature, sys; print('deep_agents_demo.orchestration' in sys.modules)"` → `False`).
- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError: deep_agents_demo.core`).
- [ ] **Step 3: Move** the listed classes; rewrite imports in `src/` and `tests/`; delete `tools.py`.
- [ ] **Step 4: Run the full suite** and `grep -rn "deep_agents_demo.tools\b\|from deep_agents_demo.tools import" src tests evals` (empty). Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Structure: layering test; Feature, AgentDefinition, tool registry and middleware as core/models leaves (no import cycle)"`

---

### Task 2: persistence/, providers/ and the secret scanner

**Files:**
- Create: `persistence/__init__.py`, `persistence/versioned_store.py`, `persistence/chats.py`, `persistence/files.py`, `providers/__init__.py`, `providers/chat.py`, `providers/embeddings.py`, `providers/rate_limit.py`, `core/secrets.py`, `ui/__init__.py`, `ui/cli/__init__.py`, `ui/cli/render.py`
- Remove: `skill_store.py`, `threads.py`, `embeddings.py`, `memory_policy.py`; `config.rate_limiter` and `app.chat_model` move
- Test: `tests/test_structure.py` (`LEGACY` shrinks), new `tests/test_persistence.py`

**Interfaces:**
- Produces:
  - `persistence/versioned_store.py`: `VersionedStore(path, tables=SKILL_TABLES)` (today's `SkillStore`, renamed), `SKILL_TABLES`, `AGENT_TABLES`, `Item` (today's `Skill` record, renamed), `Version`, `locked` (today's `_locked`, public).
  - `persistence/chats.py`: `Chats` (list/exists/info/messages/transcript/rename/delete), `thread_id`, `valid_name`, `new_name`, `ORCHESTRATION_THREADS` — no formatting.
  - `persistence/files.py`: `safe_folder_name(user) -> str` (the `re.sub(r"[^A-Za-z0-9._-]", "_", ...) or "_"` used in four places).
  - `ui/cli/render.py`: `format_table`, `format_transcript` (from `threads.py`).
  - `providers/chat.py`: `chat_model(settings, model=None)`; `providers/rate_limit.py`: `rate_limiter(rpm)`; `providers/embeddings.py`: `CompatibleEmbeddings` (unchanged API plus `notify=None` replacing its `print` calls; the composition root passes `print` for now).
  - `core/secrets.py`: `find_secrets`, `redact`, `split_findings`, `WORDING_KINDS` (from `memory_policy.py`).
- [ ] **Step 1: Write the failing test** `tests/test_persistence.py::test_store_from_before_the_rename_still_opens`: create a SQLite file with today's `skills`/`skill_versions`/`installs` schema and one saved skill (using `sqlite3` directly, with the exact `SCHEMA` text from `skill_store.py` at `10edf2d`), then `VersionedStore(path).get("user", "alice", "x").files == {...}`; and `test_safe_folder_name`: `"a b/c"` → `"a_b_c"`, `""` → `"_"`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Move and rename**; update every importer. `app.py` does `from deep_agents_demo.providers.chat import chat_model`, so tests keep patching `app.chat_model` until Task 5 changes it to `runtime.chat_model`. Shrink `LEGACY`.
- [ ] **Step 4: Full suite + greps** for `skill_store`, `threads`, `embeddings`, `memory_policy`, `SkillStore`, `rate_limiter` old paths. Expected: OK, empty.
- [ ] **Step 5: Commit**: `git commit -m "Structure: persistence (VersionedStore, Chats), providers (chat, embeddings, rate limit), core secret scanner"`

---

### Task 3: Versioned items as a core base; skills and agents feature folders

**Files:**
- Create: `core/items.py`, `features/__init__.py`, `features/skills/{__init__,format,manager,feature}.py`, `features/agents/{__init__,files,registry,builtins,manager,feature}.py`
- Remove: `skill_manager.py`, `agent_files.py`, `agent_registry.py`, `agent_manager.py`
- Test: new `tests/test_items_contract.py`; existing `test_skills.py`, `test_agents.py`, `test_agent_manager.py` (import paths)

**Interfaces:**
- Produces:
  - `core/items.py`: `USER`, `MARKETPLACE`, `SCOPES`, `ItemChange` (today's generic `SkillChange` fields and methods with `noun`, `main_file`, `tools_used()` hooks), `ItemManager` (today's generic `SkillManager` planning/apply/browse/overview/plan dispatch with `noun`, `change_type`, `main_file`, `browse_tool`, `validate`, `warnings`, `after_apply` hooks; no SKILL.md knowledge).
  - `features/skills/format.py`: `render_skill_md`, `split_skill_md`, `declared_tools`, `validate`, `warnings`, `read_skill_folder`; `features/skills/manager.py`: `SkillChange(ItemChange)`, `SkillManager(ItemManager)`, `GATED_TOOLS`; `features/skills/feature.py`: `SkillManager.feature()` stays a method (moved with the class).
  - `features/agents/files.py`: `parse_agent_md`, `render_agent_md`, `translate_tools`, `resolve_model`, `normalise_agent_text`, `split_agent_md`, `CLAUDE_TOOLS`, `NAME_PATTERN` (imports `AgentDefinition`, `READ_ONLY` from `models/agents.py`); `registry.py`: `AgentRegistry`, `load_folder`, `BUILT_IN_ELSEWHERE`; `builtins.py`: `BUILTIN_AGENTS`, prompts; `manager.py`: `AgentChange(ItemChange)`, `AgentManager(ItemManager)`, `GATED_AGENT_TOOLS`. (`AgentManager` subclasses `ItemManager` directly, no longer `SkillManager`.)
- [ ] **Step 1: Write the failing contract test** `tests/test_items_contract.py`: a mixin `ItemManagerContract` with `make_manager(store, owner)` and `create(manager, name)` hooks and tests `test_create_then_update_adds_versions`, `test_remove_and_restore`, `test_publish_install_follow_latest`, `test_invalid_change_saves_nothing`, `test_plan_dispatch_by_tool_name`; two concrete classes `SkillManagerContract` and `AgentManagerContract` running it.
- [ ] **Step 2: Run.** Expected: ImportError (`core.items`).
- [ ] **Step 3: Extract and move.** `AgentManager` no longer imports from skills (rule test enforces). Shrink `LEGACY`.
- [ ] **Step 4: Full suite + greps.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Structure: ItemManager base in core; skills and agents feature folders"`

---

### Task 4: The remaining feature folders, assembly in core, the sandbox provider registry

**Files:**
- Create: `core/assembly.py`, `providers/registry.py`, `features/{catalog,memory,documents,images,guardrails,orchestration,sandbox}/...` per the design's layout
- Remove: `agent.py`, `catalog.py`, `memory.py`, `rag.py`, `images.py`, `guardrails.py`, `orchestration.py`, `rooms.py`, `sandbox.py`, `openshell_backend.py`
- Test: new `tests/test_providers.py`; existing suites (import paths); `tests/test_workspace.py` (seed check)

**Interfaces:**
- Produces:
  - `core/assembly.py`: `build_agent(...)`, `worker_factory(..., builtins=())` — `builtins` (a sequence of `AgentDefinition`) replaces the import of `BUILTIN_AGENTS`; the composition root passes them. `IDENTITY`, `DELEGATION`, `CLOSING`, `RESEARCHER_PROMPT`, `system_prompt`, `MAIN_ONLY`, `_spec`, `_assemble`.
  - `providers/registry.py`: `class ProviderRegistry(Generic[T])`: `register(name) -> Callable[[T], T]` (decorator), `get(name) -> T` (KeyError with known names), `names() -> tuple[str, ...]`.
  - `features/sandbox/providers.py`: `SANDBOXES = ProviderRegistry()`; `@SANDBOXES.register("daytona")` and `@SANDBOXES.register("openshell")` on today's `_daytona`/`_openshell` factories; `make_backend(...)` looks up `SANDBOXES.get(provider)`.
  - `models/outcomes.py`: `StepResult` (moved, frozen where its users allow, with `to_dict()`), and `PlanOutcome`/`RoomOutcome` as `TypedDict`s describing today's outcome dicts (`{"goal", "steps", "stopped_early", ...}` / `{"ended_by", "summary", "transcript", ...}` — copy the exact keys from `run`'s return values). Runners keep returning plain dicts, so the tools' JSON output is unchanged.
  - `features/orchestration/`: `limits.py` (`Budget`, `BudgetCap`, `_WrapUp`), `runner.py` (`Runner`, `_RunFailed`, `RECURSION_LIMIT`, `_last_text`), `plans.py` (`Step`, `parse_verdict`, `PlanRunner`, `run_plan`, `PLAN_PROMPT`), `rooms.py` (today's `rooms.py` + `start_room`, `ROOM_PROMPT`, `parse_room_command`, `room_followup`), `feature.py` (`orchestration_feature`).
- [ ] **Step 1: Write the failing tests:**
  - `tests/test_providers.py::test_registry`: register two factories with the decorator; `names()` order; `get("nope")` raises `KeyError` naming the known ones; registering a name twice raises `ValueError`.
  - `test_providers.py::test_sandbox_names_match_config`: `SANDBOXES.names()` equals `config.SANDBOX_PROVIDERS` minus `"none"` (replaces the current sync test).
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Move**; `StepResult` goes to `models/outcomes.py` (see Produces); add the two orchestration modules to `IO_PENDING`; fix the string patch target `"deep_agents_demo.rooms.time.strftime"` → `"deep_agents_demo.features.orchestration.rooms.time.strftime"`. Shrink `LEGACY`.
- [ ] **Step 4: Full suite + greps** for every removed module name. Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Structure: feature folders for catalog, memory, documents, images, guardrails, orchestration, sandbox; assembly in core; sandbox provider registry"`

---

### Task 5: startup/ — process resources and per-user runtimes

**Files:**
- Create: `startup/__init__.py`, `startup/resources.py`, `startup/runtime.py`, `startup/workspace.py`
- Remove: `app.py`, `workspace.py`
- Test: new `tests/test_startup.py`; `test_agent_manager.RuntimeTests`, `test_orchestration.WiringTests`, `test_rooms.WiringTests` (import paths, patch targets)

**Interfaces:**
- Produces:
  - `startup/resources.py`: `class Resources` (context manager): `open(settings, notify=None) -> Resources`; attributes `checkpointer`, `memory` (one Mem0 or `None`), `documents` (`DocumentIndex` or `None`), `skill_store`, `agent_store`, `sandbox` (`BackendSetup`); `close()` closes all (ExitStack).
  - `startup/runtime.py`: `Runtime` (today's dataclass incl. `refresh`, `make_worker`, `rebuild`); `build_runtime(resources, settings, user, notify=None) -> Runtime` (memory service = `MemoryService(resources.memory, user)`); `history_runtime(settings, user, stack)`; `chat_model` imported by name here so tests patch `runtime.chat_model`.
  - `startup/workspace.py`: `seed_files`, `SEED_DIR`, `prepare_workspace(settings)`.
- [ ] **Step 1: Write the failing tests:**
  - `test_startup.py::test_two_users_share_one_memory_store`: with `make_memory` patched to return a `FakeMemory` and count calls, `Resources.open(settings)` then `build_runtime` for `alice` and `bob` → `make_memory` called once; each runtime's memory tools write under its own user id.
  - `test_startup.py::test_resources_close_everything`: after `close()`, the stores' connections are closed (`sqlite3.ProgrammingError` on use).
  - `test_workspace.py::test_seed_dir_is_the_package_seed` (written against `deep_agents_demo.startup.workspace`, so RED until the move): `SEED_DIR == Path(deep_agents_demo.__file__).parent / "seed"`, it contains `skills/`, `agents/` and `memories/`, and `prepare_workspace` on a temp root copies at least one skill and the `critic.md` agent.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Move and split** `app.py`; update `chat.py`/`commands.py` callers (still legacy until Task 8) to open `Resources` then `build_runtime`.
- [ ] **Step 4: Full suite + greps** for `deep_agents_demo.app`, `app.chat_model`, `app.build_runtime`. Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Structure: startup resources opened once per process, per-user runtimes on top"`

---

### Task 6: Events, the CLI renderer, and the turn engine

**Files:**
- Create: `models/events.py`, `models/approvals.py` (its Task 7 Produces entry: `ports.py` needs the types now), `core/ports.py`, `core/turns.py`, `tests/recording_ui.py`
- Modify: `ui/cli/render.py` (gains `CliRenderer`), `chat.py` (uses `core/turns.py`)
- Test: new `tests/test_render.py`; `test_polish.py`, `test_images.py` (printing tests via `RecordingUI`)

**Interfaces:**
- Produces:
  - `models/events.py` (frozen dataclasses, each with `to_dict()`): `MessageEvent(label: str, node: str, role: str, text: str, tool_calls: tuple)`, `TodosEvent(todos: tuple)`, `NoticeEvent(text: str)`, `FinalAnswerEvent(text: str)`.
  - `core/ports.py`: `class EventSink(Protocol): emit(event) -> None`; `class Interaction(Protocol): decide(request: ApprovalRequest) -> Decision; choose(prompt: str, options: tuple[str, ...]) -> str; ask(prompt: str) -> str | None`; `class Interruptible(Protocol): request_pause() -> None; stop() -> None`.
  - `core/turns.py`: `stream_once(agent, payload, config, sink, shown=None, label=None) -> list` and `run_turn(agent, prompt, config, sink, decide, runners=None, reload_skills=True, label=None, quiet=False, show=None) -> None` — today's `stream_turn` logic, emitting `MessageEvent`/`TodosEvent` (and `FinalAnswerEvent` when `quiet`) instead of printing; `decide(value) -> dict` answers approval pauses; `turn_input`, `final_answer`.
  - `MessageEvent` also carries `pretty: str`, the message's `pretty_repr()` taken in the engine (images replaced by `[image]` first, as today). The renderer never touches LangChain objects.
  - `ui/cli/render.py`: `class CliRenderer(EventSink)`: `__init__(self, write: Callable[[str], None] = print)`. Every line goes through `write`, never `print` or `pretty_print`. `MessageEvent` → `write(f"\n[{label} / {node}]")` then `write(event.pretty)`, exactly matching today's header format; `TodosEvent` → `[todos] <json>` as today; `NoticeEvent`/`FinalAnswerEvent` → `write(text)`.
  - `tests/recording_ui.py`: `RecordingUI` implementing `EventSink` and `Interaction`: `lines: list[str]`; `emit` renders through `CliRenderer(write=self.lines.append)`; `notify = self.lines.append`; scripted `answers` (a list popped by `decide`/`choose`/`ask`).
- [ ] **Step 1: Write the failing tests** `tests/test_render.py`: a scripted turn (fake model: one tool call, one answer) rendered by `CliRenderer` into a captured stdout equals the text today's `stream_turn` prints (capture today's output first with `contextlib.redirect_stdout` on the pre-refactor function and paste the expected string into the test); `quiet` prints only the answer; image blocks show `[image]`; `MessageEvent(...).to_dict()` round-trips through `json.dumps`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement**; `chat.stream_turn` becomes a thin wrapper `run_turn(..., sink=CliRenderer(), decide=approver.decide)` until Task 8 removes it.
- [ ] **Step 4: Full suite.** Expected: OK, with `test_render` proving identical output.
- [ ] **Step 5: Commit**: `git commit -m "UI seam: events, ports, turn engine in core, CLI renderer with identical output"`

---

### Task 7: Approvals, runners and notices through the ports

**Files:**
- Create: `core/approvals.py`, `ui/cli/prompts.py`, `ui/cli/interrupts.py`
- Modify: `features/orchestration/{runner,plans,rooms}.py`, `features/guardrails`, `features/skills`/`agents` (previews → `ApprovalRequest`), `providers/embeddings.py`, `features/sandbox/providers.py`, `startup/*` (`notify`)
- Remove: `approvals.py`
- Test: `test_skills.py`, `test_guardrails.py`, `test_orchestration.py`, `test_rooms.py`, `test_polish.py` (setup via `RecordingUI`; assertions unchanged)

**Interfaces:**
- Produces:
  - `models/approvals.py`: `ApprovalRequest(title: str, summary: str, diff: str, full: str, errors: tuple[str, ...])` with `to_dict()`; `Decision(approved: bool, message: str = "")` (frozen) with `to_resume() -> dict` giving today's HITL payload (`{"type": "approve"}` / `{"type": "reject", "message": ...}`); `NO_HUMAN` message constant.
  - `core/approvals.py`: `class Approvals`: `__init__(self, preview, interaction: Interaction | None, interactive: bool)`; `decide(value) -> dict` (the graph's resume value) — auto-rejects changes with errors, rejects when not interactive (`NO_HUMAN`), else returns `interaction.decide(request).to_resume()`. `Interaction.decide(request: ApprovalRequest) -> Decision` (Task 6's port signature updated to match). `preview(name, args)` objects (`ItemChange`, `RiskyAction`) gain `as_request() -> ApprovalRequest`.
  - `ui/cli/prompts.py`: `class CliInteraction(Interaction)`: `decide` shows the summary and `[a]pprove  [r]eject (with a reason)  [d]iff  [f]ull file >` exactly as today's `Approver`; `choose` and `ask` wrap `input()`.
  - `Runner.__init__(self, make, approvals: Approvals, interaction: Interaction, sink: EventSink, *, thread, budget=150, interactive=True)`; runners emit `NoticeEvent` for their `[plan]`/`[room]` lines and implement `Interruptible`. `request_pause()` sets today's pause flag. `stop()` sets a `stopped` flag that the runner checks before each agent run. The runner keeps today's `except KeyboardInterrupt` handling as the "stop now" path, so a blocking model stream is still cut at once and `test_polish`'s `Interrupting` graph (raises `KeyboardInterrupt`) keeps its assertions.
  - `ui/cli/interrupts.py`: `ctrl_c_controls(target: Interruptible, interactive: bool)` context manager. When interactive, it installs the SIGINT handler and restores the old one on exit. First press: `target.request_pause()` plus today's pause message via `os.write`. Second press: `target.stop()`, restore the previous handler, then `raise KeyboardInterrupt` (today's behaviour, moved).
  - `tests/test_runner_contract.py`: a `RunnerContract` mixin run for `PlanRunner` and `RoomRunner` (concrete classes give `make_runner(responses, **kw)` and `start(runner)`). Its tests:
    - `test_budget_exhaustion_wraps_up` (outcome reports the budget, no exception escapes);
    - `test_request_pause_asks_before_continuing` (`RecordingUI` scripted `"continue"`; a pause prompt was asked);
    - `test_stop_before_start_runs_no_agent` (`stop()` then `start` → nothing built beyond the participant/agent name check; outcome marked stopped);
    - `test_keyboard_interrupt_stops_and_returns` (a graph raising `KeyboardInterrupt` → stopped outcome, no exception).
- [ ] **Step 1: Write the failing tests:**
  - `test_guardrails.py::test_preview_as_request`: `RiskyAction.as_request()` has the same summary text as today's prompt.
  - `tests/test_runner_contract.py`, as above.
  - Empty `IO_PENDING` and take `approvals` out of `LEGACY`; `test_terminal_io_only_in_ui` fails while the orchestration modules still use `signal`/`input`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**; convert tests' `Approver(ask=..., say=...)` / `ask=`/`say=` to `RecordingUI` without changing their assertions.
- [ ] **Step 4: Full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "UI seam: approvals and runners through ports, Ctrl+C in the CLI, notices via notify"`

---

### Task 8: The CLI package and the entry point

**Files:**
- Create: `ui/cli/main.py`, `ui/cli/repl.py`, `ui/cli/commands.py`, `ui/gui/README.md`
- Remove: `chat.py`, `cli.py`, `commands.py`
- Modify: `__main__.py`, `pyproject.toml` (`[project.scripts] deep-agents-demo = "deep_agents_demo.ui.cli.main:main"`), `evals/memory_recall.py`
- Test: `test_cli.py`, `test_config.py` (paths), new `test_structure.py::test_entry_point` and `test_structure.py::test_removed_modules_are_not_imported`

**Interfaces:**
- Produces: `ui/cli/main.py`: `parse_args`, `dispatch`, `main` (today's `cli.py`); `ui/cli/repl.py`: `ChatSession`, `COMMANDS`, `HELP`, `command`, `runners_for`, `run_chat`, `GREETING`, `confirm`; `ui/cli/commands.py`: today's one-shot commands.
- [ ] **Step 1: Write the failing tests:**
  - `test_entry_point`: the `pyproject` script target `deep_agents_demo.ui.cli.main:main` imports and is callable; `deep_agents_demo/__main__.py` calls it (read the file); `LEGACY` is empty.
  - `test_removed_modules_are_not_imported`: no file in `src`, `tests` or `evals` imports or patches `deep_agents_demo.<name>` for any of the old flat names:
    `agent`, `agent_files`, `agent_manager`, `agent_registry`, `app`, `approvals`, `catalog`, `chat`, `cli`, `commands`, `embeddings`, `guardrails`, `images`, `memory`, `memory_policy`, `openshell_backend`, `orchestration`, `rag`, `rooms`, `sandbox`, `skill_manager`, `skill_store`, `threads`, `tools`, `workspace`.
    Match with the regex `deep_agents_demo\.<name>\b(?!\.)` plus `from deep_agents_demo import <name>\b`, so `app` doesn't match `approvals` and `features.images` doesn't match `images`. RED now: `chat`, `cli` and `commands` still exist and are imported.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Move**; `uv sync` to reinstall the script.
- [ ] **Step 4: Full suite; smoke**: `uv run deep-agents-demo --help`, and `uv run python -m deep_agents_demo --help`. Expected: OK, both print the usage.
- [ ] **Step 5: Commit**: `git commit -m "Structure: CLI package under ui/, entry point moved, no legacy modules left"`

---

### Task 9: Docs, CI, and the live checks

**Files:**
- Modify: `README.md` (Project layout section → the new tree; "Extending the agent" → new feature folder + one line in `startup/runtime.py`, new slash command, new sandbox provider via `@SANDBOXES.register`, a new UI via the ports), `.github/workflows/*.yml` (paths if any), `design/specs/2026-11-11-project-structure-design.md` (status: built)
- Test: none new (docs and CI only); the live checks are the gate.

- [ ] **Step 1: Before Task 1 starts** (do this first, on `main`), save today's visible output for the live-check commands below to the plan's workspace as `before.log`.
- [ ] **Step 2: Update docs and CI.**
- [ ] **Step 3: Full suite, then the live checks** (NVIDIA model, user `tools-check`, piped input, `MSYS_NO_PATHCONV=1`): an approval prompt (skill creation with `d` then `r`), a two-step plan, a `/room` with `ROOM_MAX_TURNS=2`, quiet and `--verbose` `--prompt`, `/agents`, `/threads`, an attached image. Save `after.log` and diff it against `before.log`, ignoring model-written text. Expected: only model text differs. Clean up test data (`--delete-thread`, `tools-check` folders and rows).
- [ ] **Step 4: Commit**: `git commit -m "Structure: README layout and extension guide, CI paths, design marked built"`

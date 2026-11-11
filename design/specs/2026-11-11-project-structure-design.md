# Project structure hardening: layers, feature folders, a UI seam

Status: idea agreed in chat on 2026-11-11 ("its good, write the design"); the user
asked that the design use design patterns, decorators, OOP and the SOLID principles.
Nothing is moved yet.

## Intent

The demo has grown to 28 modules (~5,000 lines) in one flat package. Related classes
are scattered (skills over `skill_store.py` and `skill_manager.py`, subagents over
three `agent_*.py` files), the engine prints and reads from the terminal (so a GUI
or a server could not reuse it), and an import cycle (orchestration → chat → app →
agent → orchestration) is held together by imports inside functions.

The user wants the code organised so that it can grow: feature folders, a UI
layer split from the engine (a CLI now, a GUI later), related classes kept together,
and, later, a server entry point alongside the CLI, built on clear core, models,
persistence and features layers.

Success: every file has an obvious home; a new feature is a new folder plus one
line in the wiring; the engine can be driven by the CLI today and by a GUI or a
server tomorrow without touching it; and a test stops the structure from decaying.

## Decided by the user

- Feature folders, with related classes together.
- A UI division: the CLI now, a GUI in the future.
- Layers for a future server: start-up, core, models, persistence, features, ...
- The layout and rules below (agreed in chat), built with design patterns,
  decorators, OOP and SOLID in mind.

## Goals

- Each module in one layer; each feature's classes in its folder.
- The engine never prints, reads input or handles signals; UIs do.
- What crosses from the engine to a UI is plain, serialisable data.
- Shared resources (memory store, document index, stores, checkpointer) opened once
  per process; per-user runtimes on top.
- An automated rule test for the layering.

## Non-goals

- Building the server or the GUI (only their seams).
- Authentication (`--user` is a name, not a login).
- New behaviour: apart from the UI seam, this is a move-and-split refactor. No
  feature changes, no new dependencies.
- Restructuring the tests into folders (they import each other; later).

## Anti-goals

- Compatibility shims at old module paths: tests patch module attributes
  (`app.chat_model`, `agent.create_deep_agent`); a re-export would make them patch
  the wrong object and pass while testing nothing.
- Auto-discovery magic (importing every folder at start-up to register features):
  wiring stays explicit and readable in one place.
- Abstraction for its own sake: an interface only where there are, or will soon be,
  two implementations (CLI and GUI/server; local and sandbox backends; providers).

## Constraints

- `config.py` stays at the package root: it locates `.env`, `data/` and
  `demo_workspace/` from its own path.
- All 323 tests stay green after every commit. The live checks (approval prompt,
  a plan, a room, quiet `--prompt`) are re-run after the UI-seam step.
- LangChain middleware class names are graph node names: middleware classes keep
  their names.
- The `deep-agents-demo` script entry, `__main__.py`, `evals/`, the CI workflow and
  the README's layout section and links are updated with the moves.
- Measured: local Qdrant refuses a second client on the same folder **even in the
  same process** (`RuntimeError: Storage folder ... is already accessed by another
  instance`). Today each runtime opens its own memory store, so a server's second
  user would fail; hence process-level resources.

## Layers and rules

```text
config, models → persistence, providers → core → features → startup → ui
```

`config.py` and `models/` import nothing from the package. `startup/` holds wiring
only; the entry point lives in `ui/cli/main.py`, the top layer, which imports
`startup/` (downward).

1. A module imports only from its own layer or layers to its left (lower).
2. `print`, `input` and `signal` appear only under `ui/`.
3. A feature imports `core/`, `persistence/`, `models/`, `providers/` and its own
   folder; never another feature. (One reviewed exception is removed by lifting
   the shared base, see "Versioned items".)
4. `models/`, `persistence/` and `providers/` hold only what two or more features
   share or what crosses to a UI or the API; feature-private things stay in the
   feature folder.

A rule test (`tests/test_structure.py`) parses every module with `ast`, maps its
`deep_agents_demo.*` imports (top-level and inside functions) to layers, and fails
on any upward or cross-feature import and on `print(`/`input(`/`signal.` outside
`ui/`. `providers/` sits beside `persistence/` (both below `core/`).

## Layout

```text
src/deep_agents_demo/
├── config.py               Settings (unchanged location)
├── seed/                   package data, unchanged location
├── models/                 plain data shapes (frozen dataclasses), serialisable
│   ├── events.py           Event types the engine emits
│   ├── approvals.py        ApprovalRequest (data form of today's previews), Decision
│   ├── agents.py           AgentDefinition, READ_ONLY
│   └── outcomes.py         PlanOutcome/StepResult, RoomOutcome
├── providers/              outside services
│   ├── chat.py             chat_model(settings, model=None), rate limiting
│   └── embeddings.py       CompatibleEmbeddings
├── persistence/            storage shared by features
│   ├── versioned_store.py  VersionedStore (today's SkillStore), table sets
│   ├── chats.py            Chats, thread ids, ORCHESTRATION_THREADS (no formatting)
│   └── files.py            transcript and upload folders (paths, safe names)
├── core/                   the engine: no terminal, no wiring, no features
│   ├── feature.py          Feature, approval_configs
│   ├── assembly.py         build_agent, worker_factory, system prompt
│   ├── middleware.py       ToolAccess, SubagentCallCap, PinSkills, TaskLimiter
│   ├── tools.py            ToolRegistry, shell equivalents
│   ├── items.py            versioned items: ItemChange, ItemManager (see below)
│   ├── approvals.py        deciding paused calls: rules + an Interaction port
│   ├── turns.py            the turn engine: streams a turn, emits events, dispatches pauses
│   ├── ports.py            Interaction, EventSink, Interruptible (protocols)
│   └── secrets.py          the credential scanner (today's memory_policy)
├── features/               one folder per capability
│   ├── catalog/            feature.py, tools.py
│   ├── memory/             service.py, recall.py (AutoRecall), tools.py, feature.py
│   ├── documents/          index.py, feature.py
│   ├── images/             uploads.py, window.py (ImageWindow), feature.py
│   ├── guardrails/         patterns.py, feature.py
│   ├── skills/             format.py, manager.py (SkillManager), feature.py
│   ├── agents/             files.py, registry.py, builtins.py, manager.py, feature.py
│   ├── orchestration/      limits.py (Budget, BudgetCap), runner.py (Runner),
│   │                       plans.py, rooms.py, feature.py
│   └── sandbox/            providers.py, openshell.py, daytona.py, feature.py
├── startup/                wiring only; the only layer below ui that sees every feature
│   ├── resources.py        Resources: opened once per process (closed by an ExitStack)
│   ├── runtime.py          Runtime: one per user, on top of Resources
│   └── workspace.py        seeding demo_workspace/
└── ui/
    ├── cli/                main.py (argument parsing and dispatch: the script entry
    │                       point), repl.py (ChatSession, slash commands), render.py
    │                       (EventSink that prints), prompts.py (Interaction over input()),
    │                       interrupts.py (Ctrl+C → Interruptible), commands.py (one-shot)
    └── gui/                reserved (README only)
```

Where today's code goes (each row is a move; splits are marked):

| Today | Goes to |
| --- | --- |
| `agent.py` | split: `core/feature.py` (Feature, approval_configs), `core/assembly.py` (the rest) |
| `tools.py` | split: `core/tools.py` (registry), `core/middleware.py` (ToolAccess) |
| `agent_registry.py` | split: `features/agents/registry.py`, `features/agents/builtins.py`, middleware to `core/middleware.py` |
| `agent_files.py` | `features/agents/files.py`; `AgentDefinition`, `READ_ONLY` to `models/agents.py` |
| `agent_manager.py` | `features/agents/manager.py` |
| `skill_store.py` | `persistence/versioned_store.py` (class renamed `VersionedStore`) |
| `skill_manager.py` | split: `core/items.py` (generic base), `features/skills/format.py`, `features/skills/manager.py` |
| `chat.py` | split: `core/turns.py` (engine), `ui/cli/repl.py`, `ui/cli/render.py` |
| `approvals.py` | split: `core/approvals.py` (rules), `ui/cli/prompts.py` (the `[a]pprove` prompt) |
| `threads.py` | split: `persistence/chats.py`, `ui/cli/render.py` (table, transcript) |
| `app.py` | split: `startup/resources.py`, `startup/runtime.py`; `chat_model` to `providers/chat.py` |
| `cli.py` | `ui/cli/main.py` (pyproject script target `deep_agents_demo.ui.cli.main:main`) |
| `commands.py` | `ui/cli/commands.py` |
| `orchestration.py` | split over `features/orchestration/` (limits, runner, plans) and `core/middleware.py` (TaskLimiter) |
| `rooms.py` | `features/orchestration/rooms.py` |
| `memory.py`, `memory_policy.py` | `features/memory/*`; scanner to `core/secrets.py` |
| `rag.py`, `catalog.py`, `images.py`, `guardrails.py` | their feature folders |
| `sandbox.py`, `openshell_backend.py` | `features/sandbox/*` |
| `embeddings.py`, `config.rate_limiter` | `providers/` |
| `workspace.py` | `startup/workspace.py` (`SEED_DIR` points at the package's `seed/`) |

## Principles and patterns, where each one earns its place

**Single responsibility.** Every split in the table separates two reasons to change:
the engine from its presentation (`chat.py` → turns/repl/render), deciding from
asking (`approvals.py`), storing from formatting (`threads.py`), process resources
from per-user runtimes (`app.py`). Each feature folder changes only when that feature
does.

**Open/closed, with registries and decorators.** Adding a capability must not edit
the engine:
- *Features* are added in one place, `startup/runtime.py`'s explicit feature list
  (no auto-discovery). `core/assembly.py` never changes for a new feature.
- *Slash commands* keep the existing `@command(...)` decorator registry.
- *Sandbox providers* move to a decorator registry (Registry + Strategy): a
  generic `ProviderRegistry` class in `providers/registry.py`, one instance per
  family (only `SANDBOXES` for now), used as `@SANDBOXES.register("openshell")`. A
  new backend is a decorated factory function. `config.py` keeps its own list of
  valid names (it validates before any feature is imported, as today), and a test
  keeps the list and the registry in step, as one does now. No new chat-provider
  setting: `chat_model` only moves to `providers/chat.py`.
- *Interrupt runners* (`run_plan`, `start_room`) are registered by type in
  `core/turns.py`'s dispatch table, filled by the composition root.

**Liskov.** `SkillManager` and `AgentManager` both extend `ItemManager` and are
interchangeable where the approval flow uses them (`plan(tool, args)`,
`apply(change)`); `PlanRunner` and `RoomRunner` both extend `Runner` and honour its
budget, pause and stop contract. Tests that run against the base contract run
against each subclass.

**Interface segregation.** Small `typing.Protocol` ports instead of one UI object:
- `Interaction`: `decide(request) -> Decision` (approvals), `choose(prompt, options)
  -> str` (retry/skip/stop, continue/wrap up), `ask(prompt) -> str | None` (a room
  turn, a pause note).
- `EventSink`: `emit(event)`.
- `Interruptible`: `request_pause()`, `stop()`, implemented by runners and called
  by a UI's interrupt handling.
A renderer needs only `EventSink`; a test fake may implement one port.

**Dependency inversion.** The engine depends on ports and data shapes, never on a
UI or on concrete storage: runners receive an `Interaction` and an `EventSink`;
features receive what they need from the composition root (constructor injection);
`build_agent` receives built-in agents as arguments instead of importing the agents
feature. The composition root (`startup/`) is the only place that knows concretes.

**Patterns used, and why:**

| Pattern | Where | Why |
| --- | --- | --- |
| Ports and adapters (hexagonal) | `core/ports.py`; adapters in `ui/cli/` (later `ui/gui/`, a server) | the engine is reused by every front end |
| Observer | `EventSink`: turns and runners emit, renderers listen | output becomes data; a server can stream it |
| Composition root + constructor injection | `startup/` | one place wires concretes; everything else is testable alone |
| Template method | `Runner` (budget, run-one-agent, pause) with `PlanRunner`/`RoomRunner` hooks | shared mechanics, different policies |
| Strategy + registry (decorator) | sandbox backends | swap implementations by setting, add one without edits |
| Repository | `VersionedStore`, `Chats` | storage behind a narrow API; swappable later |
| Command | `ItemChange` (plan → approve → apply), slash commands | a change is an object that can be previewed, approved, applied |
| Chain of responsibility | LangChain middleware (ToolAccess, caps, approvals) | already in use; each link one concern |
| Factory | `worker_factory`, `chat_model` | build configured agents/clients in one place |
| Facade | `Runtime` | one object a UI or server talks to |

**Decorators, kept to what pays:** `@command` (slash commands, existing),
`@SANDBOXES.register(...)` (new registry), `@locked` (today's `_locked`, renamed and moved to
`persistence/` for the thread-safe store), and LangChain's `@hook_config`. No
decorators for wiring features or for logging.

**OOP details:** data shapes are frozen dataclasses (immutable, hashable, easy to
serialise); behaviour lives on the classes that own the data (`ItemChange` and
the guardrails' `RiskyAction` still build their summary, diff and full text, and
hand them to a UI as a plain `ApprovalRequest` from `models/approvals.py`);
inheritance only for the two real families (`ItemManager`, `Runner`); composition
everywhere else.

## The UI seam (the only behavioural change)

- **Turns emit events.** `core/turns.py` runs a turn exactly as `stream_turn` does
  today but calls `sink.emit(...)` with `models/events.py` events (assistant message,
  tool call, tool result, to-do list, plan step, room turn, notice, final answer)
  instead of printing. `ui/cli/render.py` prints them exactly as the CLI does today
  (labels, quiet and verbose modes).
- **Approvals.** `core/approvals.py` keeps the rules (auto-reject invalid changes,
  refuse when nobody can answer) and asks an `Interaction` for the decision, passing
  an `ApprovalRequest`: plain data with summary, diff and full text, built from
  today's preview objects. `ui/cli/prompts.py` shows the same `[a]pprove ... [d]iff
  [f]ull file` prompt.
- **Runners.** `Runner` takes an `Interaction` and an `EventSink` instead of
  `ask`/`say`, and implements `Interruptible`. Ctrl+C handling moves to
  `ui/cli/interrupts.py`, which calls `request_pause()` (first press) and `stop()`
  (second press) on the active runner.
- **Chats.** `persistence/chats.py` returns data; `ui/cli/render.py` formats tables
  and transcripts.
- **Messages outside a turn** (embedding retry warnings, sandbox connection notes,
  start-up warnings) go through a `notify: Callable[[str], None]` callback injected
  by the UI into `Resources`, `Runtime`, the sandbox providers and
  `CompatibleEmbeddings` (default: do nothing). The CLI passes `print`, so the text
  is unchanged.
- **Acceptance:** no core or feature test patches `builtins.print`; a recording fake
  (`RecordingUI`, implementing the ports) replaces it.

## Server readiness (seams only)

- Everything crossing to a UI is a `models/` dataclass with a `to_dict()`.
- `startup/resources.py` opens the memory store (one Mem0/Qdrant per process), the
  document index, the stores and the checkpointer once; `startup/runtime.py` builds
  a per-user `Runtime` on top (memory service per user id on the shared store).
- Known limit, documented, not fixed here: while a plan or room waits for the user,
  its state lives on a Python call stack, not in a checkpoint; a server keeps one
  live worker per running turn.

## Migration (each step a commit, all tests green)

1. **Leaves first:** `core/feature.py` (Feature), `models/agents.py`
   (AgentDefinition), `core/middleware.py`; this removes the import cycle.
2. **Mechanical moves** with `git mv` and import rewrites, feature by feature; after
   each, grep for old paths including string patch targets in tests.
3. **Generic bases:** `core/items.py` (ItemChange, ItemManager) and the
   `VersionedStore` rename; `providers/` with the sandbox `ProviderRegistry`.
4. **Resources vs runtime** in `startup/`.
5. **The UI seam** (events, ports, approvals split, runners, Ctrl+C, render); then
   the live checks.
6. **The rule test**, README layout and extension docs, `pyproject` entry point,
   CI, `evals/`.

## Testing

- The existing 323 tests keep their assertions, including the exact CLI strings
  (`"[plan] step 1/2 - general-purpose"`, `"[room] moderator -> critic: ..."`, the
  approval prompt). Their setup changes: import paths, and `ask=`/`say=` callables
  and `patch("builtins.print")` give way to `RecordingUI`, which implements the
  ports and renders events through the real `ui/cli/render.py` into a list of
  strings. So the strings tested are the strings the CLI prints.
- `tests/test_structure.py` (the rule test).
- Contract tests: one suite run against `SkillManager` and `AgentManager` through
  `ItemManager`; one against `PlanRunner` and `RoomRunner` through `Runner`.
- Live (NVIDIA model): an approval prompt, a plan with review, a room, quiet and
  verbose `--prompt`, `/agents`, `/image`.

## Calls made by Claude (check these)

- `SkillStore` is renamed `VersionedStore` (it stores skills and agents).
- The generic plan/approve/apply logic is lifted from `SkillManager` into
  `core/items.py` (`ItemManager`, `ItemChange`), removing the agents → skills
  dependency.
- `seed/` and `config.py` stay at the package root.
- The entry point is `ui/cli/main.py` (top layer); `startup/` is wiring only.
- `config.py` is the lowest layer and keeps validating sandbox names itself; a test
  keeps them in step with the sandbox registry.
- No new chat-model provider setting; the provider registry is for sandbox
  backends only, for now.
- Messages outside a turn use an injected `notify` callback (the CLI passes `print`).
- Tests keep their exact-string assertions through `RecordingUI` + the real renderer.
- Ports are synchronous (a GUI or server runs the engine on a worker thread).
- Tests stay in one folder for now.

## Left to the builder

- Exact event classes and fields beyond those listed.
- File names inside feature folders where the table doesn't fix them.
- The order of feature moves within step 2.

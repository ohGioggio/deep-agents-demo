# Subagents registry, piece 3: plans with per-step review, and the parallel cap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the user asks for subagent-driven execution, the main agent hands a plan to `run_plan`; the chat loop runs each step with a worker agent and a reviewer, retries failed reviews, asks the user when a step fails, pauses on Ctrl+C between steps, stays within a request budget, and reports every step's outcome. Separately, cap how many subagents run at once.

**Architecture:** `run_plan` is a tool that pauses the main graph with a LangGraph `interrupt` carrying the plan. `stream_turn` dispatches such interrupts by `type` to runners, and approval interrupts still go to the `Approver`. `PlanRunner` (new `orchestration.py`) runs on the chat loop's thread. It builds standalone worker and reviewer graphs from the same agent definitions and limits as subagents (`worker_factory` in `agent.py`) and streams them with the existing `stream_turn`, so approvals and output work as today. It resumes `run_plan` with the outcome. A `TaskLimiter` middleware bounds concurrent `task` calls.

**Tech Stack:** Python 3.11+, LangGraph `interrupt`/`Command(resume=...)`, LangChain `create_agent` + `HumanInTheLoopMiddleware`, Deep Agents 0.7.34 middleware (`FilesystemMiddleware`, `SkillsMiddleware`, `PatchToolCallsMiddleware`), unittest with fake models.

**Spec:** `design/specs/2026-11-11-subagents-registry-design.md` (sections 4, 5, 7, 8; piece 3). Pieces 1–2 are on `main` (`c938dcb`).

## Global Constraints

- `run_plan(goal, steps)` is used when the user asks for subagent-driven execution, or agrees when the main agent proposes it for a clearly long task; never switched on silently.
- Each step names a worker agent (default `general-purpose`); a fixed loop per step: worker (fresh context: goal, this step, short notes from earlier steps) → `reviewer` verdict (`PASS`/`FAIL` + notes) → on `FAIL` the worker retries with the notes; after `PLAN_MAX_ATTEMPTS` (default 2) failed reviews the step has failed.
- A failed step stops the plan in code and asks `[r]etry  [s]kip  [x] stop`; with `--prompt` it stops.
- Ctrl+C during a plan pauses after the current step with `[plan] You:`. Text becomes a note for the next worker, `/stop` ends the plan, and Enter resumes.
- Budget: `ORCHESTRATION_MAX_CALLS` (default 150) model calls shared by every worker and reviewer in one plan. When it runs out, the runner asks "continue (another 150) or wrap up"; with `--prompt` it wraps up. `SUBAGENT_MAX_CALLS` still caps each run.
- Parallel: at most `SUBAGENT_MAX_PARALLEL` (default 3) `task` calls run at once; extra calls wait.
- `run_plan` and (piece 4's) `start_room` are main-agent only; subagents and plan workers can't call them (no nesting).
- Approvals, guardrails, `DISABLED_TOOLS`, skill gates, per-agent tools and the call cap apply to workers and reviewers exactly as to subagents.
- The outcome returned to the main agent lists each step as `passed`, `skipped`, `failed` or `stopped`, with the reviewer's notes and the worker's result (each result up to 1,500 characters).
- Terminal labels: `[plan] step 2/5 · helper`, `[plan] step 2/5 ✓ reviewed`, `[plan] step 2/5 ✗ attempt 1/2: <notes>`.
- Settings validated in `Settings.from_env`: each ≥ 1.
- Commits: one-line messages, no trailer; don't push unless asked. Tests: `uv run python -m unittest discover -s tests` (single file: `-p "test_orchestration.py"`).

## Review Focus

1. **A worker's risky action needing approval mid-plan** — expected: the approval prompt appears as usual, and the step continues after the answer without re-running finished steps. (Task 4 test.)
2. **A reviewer reply with no clear PASS/FAIL** (e.g. "Looks fine overall") — expected: treated as FAIL with the note "the reviewer gave no clear verdict", so nothing slips through unreviewed. (Task 4 test.)
3. **A step naming an agent that doesn't exist** — expected: the step fails at once with "no agent named X" (no model calls), and the user is asked retry/skip/stop. (Task 4 test.)
4. **Rejecting an approval inside a worker** — expected: the worker is told why and continues or gives up; the review then judges the result; nothing crashes. (Task 4 test.)
5. **A `--prompt` run with a plan** — expected: runs without asking; a failed step or an exhausted budget stops or wraps up and says so in the outcome. (Task 5 test.)

---

### Task 1: Settings, the parallel cap, and main-only tools

**Files:**
- Modify: `src/deep_agents_demo/config.py`, `src/deep_agents_demo/agent.py`, `.env.example`
- Create: `src/deep_agents_demo/orchestration.py` (only `TaskLimiter` in this task)
- Test: `tests/test_orchestration.py` (classes `SettingsTests`, `ParallelTests`), `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings.plan_max_attempts: int = 2`, `Settings.orchestration_max_calls: int = 150`, `Settings.subagent_max_parallel: int = 3` (env `PLAN_MAX_ATTEMPTS`, `ORCHESTRATION_MAX_CALLS`, `SUBAGENT_MAX_PARALLEL`, each ≥ 1).
  - `MAIN_ONLY = ("run_plan", "start_room")` in `agent.py`; every subagent's `ToolAccess` gets `denied |= MAIN_ONLY` (user agents, `general-purpose`, `researcher`).
  - `class TaskLimiter(AgentMiddleware)` in `orchestration.py`: `__init__(self, limit: int)`; `wrap_tool_call` holds a `threading.BoundedSemaphore(limit)` around `task` calls only.
  - `build_agent(..., max_parallel: int = 3)` adds `TaskLimiter(max_parallel)` to the main agent's middleware.

- [ ] **Step 1: Write the failing tests:**
  - `test_config.py::test_orchestration_settings`: defaults `(2, 150, 3)`; `PLAN_MAX_ATTEMPTS=0` → `ValueError` naming it.
  - `ParallelTests.test_at_most_n_subagents_run_at_once`: main fake model emits one AI message with 5 `task` calls to `helper`. `helper`'s fake model sleeps 0.2 s in `_generate` and records the peak number of concurrent calls (a counter under a lock). With `max_parallel=2` the peak is ≤ 2, and all 5 results come back.
  - `ParallelTests.test_subagents_cannot_run_plans`: a user agent with all tools calling `run_plan` → refused with `"isn't available to this agent"` (use a dummy `run_plan` feature tool).
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** the interfaces; add the three settings with one comment line each to `.env.example`.
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Orchestration settings, parallel cap on task, plan and room tools main-agent only"`

---

### Task 2: Standalone workers from agent definitions

**Files:**
- Modify: `src/deep_agents_demo/agent.py`
- Test: `tests/test_orchestration.py` (class `WorkerTests`)

**Interfaces:**
- Consumes: `_spec`, `ToolAccess`, `SubagentCallCap`, `approval_configs`, `MAIN_ONLY` (existing / Task 1).
- Produces:
  - `build_agent(...)` gains `worker_middleware: Sequence[AgentMiddleware] = ()` (added to every worker built by the factory, e.g. the plan budget).
  - `worker_factory(model, checkpointer, backend, features=(), *, skills_prefix="", disabled_tools=(), agents=(), model_for=None, max_calls=25) -> Callable[[str, Sequence[AgentMiddleware]], CompiledStateGraph | None]`. `make(name, extra_middleware=())` returns a compiled graph for agent `name` (a user agent, a built-in `reviewer`/`moderator`, or the demo's `general-purpose`), or `None` if there's no such agent.
  - The graph is `langchain.agents.create_agent(model=spec["model"], tools=..., system_prompt=spec["system_prompt"], middleware=[FilesystemMiddleware(backend=backend, permissions=<same deny rules as build_agent>), PatchToolCallsMiddleware(), SkillsMiddleware(backend=backend, sources=skill_sources), *spec["middleware"], *extra_middleware, HumanInTheLoopMiddleware(interrupt_on=<same as build_agent>)], checkpointer=checkpointer)`. `tools` is `spec["tools"]` when present, else the main agent's feature tools (minus `MAIN_ONLY`).
  - Refactor: the shared pieces of `build_agent` (registry, `off`, skill sources, interrupt configs, feature tools, access tuple, permissions, the specs) move into one private helper `_assemble(...) -> _Assembly` (a dataclass) used by both functions, so the two can't drift.

- [ ] **Step 1: Write the failing tests** (`FilesystemBackend` in a temp dir, fake model as in `test_agents.py`):
  - `test_worker_has_the_agents_tools_and_prompt`: `make("critic")` with `critic` (Read, Grep, Glob) → its first model call's system prompt is the critic prompt; bound tools ⊆ `{"read_file","grep","glob","ls"}`.
  - `test_worker_approvals_and_disabled_tools`: `make("helper")` (all tools) with `Guardrails.feature()` → a `delete` call pauses (run with `stream_turn` and an `Approver` whose `ask` raises → `AssertionError` raised); with `disabled_tools=("write_file",)` a `write_file` call is refused.
  - `test_worker_call_cap`: `max_calls=2` → the worker makes at most 2 model calls and ends with a partial result.
  - `test_unknown_agent_gives_none`: `make("nope") is None`.
  - `test_reviewer_is_read_only`: `make("reviewer")` → `write_file` not bound.
- [ ] **Step 2: Run them.** Expected: ImportError / NameError for `worker_factory`.
- [ ] **Step 3: Implement** the refactor and `worker_factory`.
- [ ] **Step 4: Run the full suite.** Expected: OK (all existing `build_agent` tests unchanged).
- [ ] **Step 5: Commit**: `git commit -m "Standalone workers from agent definitions, sharing build_agent's assembly"`

---

### Task 3: `run_plan` and interrupts that runners answer

**Files:**
- Modify: `src/deep_agents_demo/orchestration.py`, `src/deep_agents_demo/chat.py`
- Test: `tests/test_orchestration.py` (class `RunPlanToolTests`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Step: task: str, agent: str = "general-purpose"`.
  - `run_plan(goal: str, steps: list[dict]) -> str` (in `orchestration.py`, a plain function tool): validates `steps` (each a dict with a non-empty `task`; `agent` optional) and returns a JSON error string without interrupting if invalid. Otherwise it calls `interrupt({"type": "run_plan", "goal": goal, "steps": [{"task": ..., "agent": ...}, ...]})` and returns `json.dumps(<resume value>)`.
  - `orchestration_feature() -> Feature(name="orchestration", prompt=PLAN_PROMPT, tools=(run_plan,))`, where `PLAN_PROMPT` says to use `run_plan` only when the user asks for subagent-driven execution or agrees to it for a long task; to give each step a clear task and agent; that each step is reviewed; and to report the outcome to the user.
  - `stream_turn(agent, prompt, config, reload_skills=True, approver=None, runners=None)`: for each paused item, if `item.value` is a dict whose `"type"` is a key of `runners`, the resume value is `runners[type](item.value)`; otherwise `approver.decide(item.value)`. Without a matching runner (e.g. `history_runtime`, tests), a `run_plan` interrupt is answered with `{"error": "plans can't run here"}`.

- [ ] **Step 1: Write the failing tests:**
  - `test_run_plan_pauses_and_returns_the_runner_answer`: main fake model calls `run_plan(goal="g", steps=[{"task": "a"}])`; `stream_turn(..., runners={"run_plan": fake_runner})` → `fake_runner` called once with `{"type": "run_plan", "goal": "g", "steps": [{"task": "a", "agent": "general-purpose"}]}`; the tool result the model sees is `json.dumps(fake_runner_return)`.
  - `test_invalid_steps_are_refused_without_pausing`: `steps=[{"agent": "x"}]` → the result contains `"task"`; `fake_runner` never called.
  - `test_without_a_runner_the_plan_is_refused`: no `runners` → the result contains `"can't run here"`.
  - `test_approvals_still_go_to_the_approver`: a gated tool in the same turn still reaches the `Approver`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "run_plan tool: pauses the turn for a plan runner on the chat loop's thread"`

---

### Task 4: PlanRunner: workers, reviews, retries, failures

**Files:**
- Modify: `src/deep_agents_demo/orchestration.py`
- Test: `tests/test_orchestration.py` (class `RunnerTests`)

**Interfaces:**
- Consumes: `worker_factory`'s `make` (Task 2), `stream_turn` (Task 3), `Approver`, `Step`.
- Produces:
  - `parse_verdict(text: str) -> tuple[bool, str]`: `True` if the first non-empty line starts with `PASS` (case-insensitive, ignoring `*`, `#` and spaces), `False` for `FAIL`; anything else → `(False, "the reviewer gave no clear verdict: <first 200 chars>")`; the notes are the remaining text.
  - `@dataclass class StepResult: step: Step, status: str, result: str = "", notes: str = ""` (`status` ∈ `passed`, `skipped`, `failed`, `stopped`).
  - `class PlanRunner`: `__init__(self, make, approver, *, thread: str, max_attempts: int = 2, budget: int = 150, interactive: bool = True, ask=input, say=print)`; `run(self, request: dict) -> dict` returns `{"goal": ..., "steps": [{"task", "agent", "status", "result", "notes"}, ...]}` (results truncated to 1,500 chars).
  - Worker input (one human message): the goal, `Step k of n: <task>`, "Earlier steps:" with one line per finished step (status + first 300 chars of its result), and on a retry "The reviewer's notes on your last attempt:" + notes. The reviewer input: the goal, the step, the worker's result, and "Reply PASS or FAIL on the first line, then notes."
  - Each run uses its own thread id `f"{thread}:plan:{step_no}:{attempt}:worker"` / `...:review`, so retries start fresh; the result is the last AI message's text from `graph.get_state(config)`.
  - Failure prompt (interactive): `say` the step and the last notes, then `ask("[r]etry  [s]kip  [x] stop > ")`: `r` runs the step again with fresh attempts, `s` marks it `skipped`, and anything else marks it `stopped` (the remaining steps are `stopped` too). Non-interactive → `stopped`.
  - Unknown agent: the step fails without model calls ("no agent named X"), then the failure prompt.

- [ ] **Step 1: Write the failing tests** (fake `make` returning graphs built by Task 2 with scripted fake models; a list-collecting `say`; scripted `ask`):
  - `test_parse_verdict`: `"PASS\nok"` → `(True, "ok")`; `"**FAIL**\nfix x"` → `(False, "fix x")`; `"Looks fine overall"` → `False` and "no clear verdict".
  - `test_two_steps_pass`: two steps each passing at first review → statuses `["passed", "passed"]`; the second worker's input contains the first step's result under "Earlier steps:"; `say` shows `[plan] step 1/2 · general-purpose` and `✓ reviewed`.
  - `test_retry_after_fail_then_pass`: the review says FAIL "add tests", the retry's input contains "add tests", then PASS → `passed`; output shows `✗ attempt 1/2: add tests`.
  - `test_failed_step_asks_and_skip_continues`: two FAILs on step 1 → `ask` answers `s` → step 1 `skipped`, step 2 runs.
  - `test_stop_marks_the_rest_stopped`: two FAILs on step 1, `ask` answers `x` → statuses `["failed", "stopped"]`.
  - `test_non_interactive_stops`: the same with `interactive=False` and no `ask` → `["failed", "stopped"]`.
  - `test_unknown_agent_fails_without_model_calls`.
  - `test_worker_approval_mid_plan`: the worker calls `delete` (with `Guardrails.feature()`) → the `Approver` asked once (answer `a`), the step completes, step 1 ran once (no re-run).
  - `test_rejected_approval_inside_worker`: answer `r` + reason → the worker continues; the review judges; no exception.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "PlanRunner: worker and reviewer per step, retries with notes, ask on failure"`

---

### Task 5: Budget and Ctrl+C

**Files:**
- Modify: `src/deep_agents_demo/orchestration.py`
- Test: `tests/test_orchestration.py` (class `BudgetPauseTests`)

**Interfaces:**
- Produces:
  - `class Budget`: `__init__(self, size: int)`; `used: int`; `left -> int`; `extend()` adds another `size`; thread-safe (`threading.Lock`).
  - `class BudgetCap(AgentMiddleware)`: `__init__(self, budget: Budget)`; `before_model` (with `@hook_config(can_jump_to=["end"])`) counts one call, and when the budget is used up ends the run with the message `"[Stopped: the plan's request budget is used up.]"`.
  - `PlanRunner` creates one `Budget(budget)` per plan and passes `BudgetCap(budget)` as `extra_middleware` to every `make(...)`. After each run, if `budget.left == 0`: interactive → `ask("[plan] The request budget (N calls) is used up. [c]ontinue with another N or [w]rap up > ")`; `c` → `budget.extend()` and the current step continues (re-run the interrupted attempt); otherwise the remaining steps are `stopped` with the note "budget used up". Non-interactive → wrap up.
  - Ctrl+C: during `run`, install a SIGINT handler (restore the previous one in `finally`) that sets a pause flag and says `[plan] Pausing after this step…`. Before the next step, if the flag is set: `ask("[plan] You: ")`. `""` resumes; `/stop` → the remaining steps `stopped`; any other text is added to the next worker's input as "Note from the user: …". Signal handling is skipped when not on the main thread.
- [ ] **Step 1: Write the failing tests:**
  - `test_budget_wraps_up_when_not_interactive`: `budget=3`, a worker that keeps calling `ls` → the outcome's remaining steps say "budget used up"; total model calls ≤ 3 + 1 (the stop message isn't a model call).
  - `test_budget_continue_when_asked`: `ask` answers `c` → the plan finishes; `budget.used > 3`.
  - `test_pause_note_reaches_next_worker`: set the runner's pause flag directly (`runner.pause_requested = True`) after step 1; `ask` answers "use pytest" → step 2's worker input contains "Note from the user: use pytest".
  - `test_pause_stop`: answer `/stop` → the rest are `stopped`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Plans: shared request budget with continue or wrap up, Ctrl+C pauses between steps"`

---

### Task 6: Wiring, docs, live check

**Files:**
- Modify: `src/deep_agents_demo/app.py` (`features` yields `orchestration_feature()`; `build_runtime` passes `max_parallel=settings.subagent_max_parallel`, builds `make = worker_factory(...)` alongside the agent and rebuilds it in `make_agent`; `Runtime.make_worker`), `src/deep_agents_demo/chat.py` (`run_chat` creates the `PlanRunner` per turn: thread = the chat's thread id, `interactive=prompt is None`, settings' attempts and budget; passes `runners={"run_plan": runner.run}` to `stream_turn`), `README.md` ("Plans with a reviewer on every step" section: how to ask, what you see, retry/skip/stop, Ctrl+C, budget; the parallel cap)
- Test: `tests/test_orchestration.py` (class `WiringTests`)

**Interfaces:**
- Consumes: everything above.
- Produces: `Runtime.make_worker` (the factory's `make`, refreshed by `Runtime.refresh`).

- [ ] **Step 1: Write the failing test** `WiringTests.test_plan_through_the_runtime`: real `build_runtime` with placeholder settings (as in `test_agent_manager.RuntimeTests`) and one shared fake model scripted for main → `run_plan` with one step → worker reply → reviewer `PASS` → main final. `run_chat`-equivalent call of `stream_turn` with the runner → the main model's last call sees a tool result whose JSON has `"status": "passed"`.
- [ ] **Step 2: Run it; verify it fails.**
- [ ] **Step 3: Implement** the wiring and README section.
- [ ] **Step 4: Run the full suite and a smoke build** (placeholder settings). Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Plans wired into the chat: run_plan runner, parallel cap from settings, docs"`
- [ ] **Step 6: Live check (NVIDIA model, memory and RAG off, user `tools-check`, piped interactive input, `MSYS_NO_PATHCONV=1`):**
  - "Use subagent-driven execution: step 1, write /plan_test/notes.md with three tips for writing tests; step 2, review it and add a fourth tip." → two steps, each `✓ reviewed` (or a visible retry), outcome reported, file exists.
  - A plan whose step asks the critic to "delete /plan_test" → the critic has no delete tool, so the step fails review → the prompt `[r]etry [s]kip [x] stop`; answer `s` → reported as skipped.
  - Clean up the files and chats.

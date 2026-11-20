# Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Items 2, 4, 5, 6, 7 and 8 from the spec, built in parallel by Haiku subagents (at most 5 at once), each in its own git worktree and branch, merged one by one.

**Spec:** `design/specs/2026-11-20-follow-ups-design.md`

## Global Constraints

- Every task works in its own worktree (`../dad-f<N>`, branch `fu-task<N>`) from the same base commit; it touches only the files its task lists (shared files are named; each change there is kept small and local so merges stay clean).
- `tests/test_structure.py` and `tests/test_runner_contract.py` pass; only `ui/` prints; `config.py` imports nothing from `features/`.
- Existing tests pass unchanged unless the task's item changes what they assert (named in the report).
- TDD: the failing test first. Commits one line, no trailer. Never read or print `.env`; no live model calls; no new dependencies.
- Test command: `uv run --no-sync python -m unittest discover -s tests` (after `uv sync --locked -q` in a new worktree).

## Waves

| Wave | Tasks (parallel) | Agents |
|---|---|---|
| 1 | T1 (item 6), T2 (item 2), T3 (item 5), T4 (item 4: store, counter, recorder), T5 (item 7) | 5 implementers |
| 1 reviews | one reviewer per finished task, as slots free up | ≤ 5 at once |
| 2 | T6 (item 4: `--runs`, `--show-run`, README), after T4 is merged | 1 |
| final | whole-diff review, one fix wave | 1–2 |
| last | T7 (item 8): eval `--runs 3` and README, by the controller (live) | — |

Merge order: T1, T2, T3, T5, T4 (largest last), then T6.

---

### Task 1: Review reasons for failed plan steps (item 6)

**Files:** `src/deep_agents_demo/features/orchestration/plans.py`; maybe `src/deep_agents_demo/ui/cli/richui/timeline.py`; tests `tests/test_orchestration.py`, `tests/test_rich_timeline.py`.
- A step that failed review: `finish_step(..., "failed", notes)` (the reviewer's notes), not the worker's result; other failures unchanged. The timeline shows `✗ failed review: <notes>` for a review failure (distinguish it with the step status or a marker the plan passes; keep the plain `✗ failed: <reason>` for others).
- Tests: the StepFinished summary is the notes; the timeline line for a review failure and for another failure.

### Task 2: Daemon worker pool (item 2)

**Files:** new `src/deep_agents_demo/features/orchestration/pool.py`; `swarms.py`; `rooms.py` (the parallel opening wave only); tests `tests/test_swarms.py`, `tests/test_rooms.py`, new `tests/test_pool.py`.
- `class DaemonPool(max_workers)`: `submit(fn, *args) -> Future` (concurrent.futures.Future), `shutdown(cancel_pending=True)` that never joins; worker threads are `daemon=True`; used everywhere `ThreadPoolExecutor` is used for workers today. `concurrent.futures.wait` keeps working on its futures.
- Stop notice: `[swarm] Stopped. {n} request(s) still running were abandoned.` (only when n > 0); rooms' wave the same with `[room]`.
- Tests: pool runs at most N at once; shutdown with a blocked worker returns immediately and the thread is a daemon; the swarm/room stop notices; existing Ctrl+C tests pass.

### Task 3: File version diffs (item 5)

**Files:** `src/deep_agents_demo/persistence/outputs.py` (a `diff(user, chat, name, v1=None, v2=None) -> str` helper or a pure function next to it), `src/deep_agents_demo/features/files/feature.py` (the change line in `send_file`'s result), `src/deep_agents_demo/ui/cli/repl.py` (`/files diff NAME [V1 V2]` inside the existing `files` command only), `src/deep_agents_demo/ui/cli/richui/` (render the diff with the approval panel's colours, if a renderer helper exists there; plain unified diff otherwise); tests `tests/test_outputs.py`, `tests/test_files.py`, the repl/rich test files that cover `/files`.
- Default versions: the last two. Text only (by kind; images → "can't diff an image"); a missing version → "park-plan.md has no v4 (versions: 1–3)".
- `send_file` result for v2+: add `"changes": "changed from vK: +12 −3 lines"` (K = the previous version), with a test.
- `/files` usage text and `/` menu entry mention `diff`.

### Task 4: Run store, usage counter, recorder (item 4, part 1)

**Files:** new `src/deep_agents_demo/persistence/runs.py`, new `src/deep_agents_demo/core/runs.py` (the recorder: consumes events), `src/deep_agents_demo/providers/chat.py` (a process-wide `UsageCounter` callback attached to every chat model built), `src/deep_agents_demo/models/events.py` (`StepFinished.model_calls`), `src/deep_agents_demo/features/orchestration/runner.py` (fill it from the step's tally), `src/deep_agents_demo/persistence/chats.py` (delete/rename a chat's runs), `src/deep_agents_demo/startup/resources.py` or `runtime.py` (open the store), `src/deep_agents_demo/ui/cli/repl.py` (wire the recorder around each turn — a few lines in the turn loop only); tests: new `tests/test_runs.py`, plus the touched modules' tests.
- Schema and fields exactly as in the spec (`runs`, `run_parts`). Status `done|error|interrupted`.
- Recorder: built per turn with the user, chat, model name and prompt; wraps the sink (passes every event on); records parts from SubagentStarted/Finished, StepStarted/Finished and the three orchestration tool calls (nesting by run id / open call); on turn end writes the run with the counter's delta for calls and tokens. Any exception while recording is caught; the chat says once `Couldn't record this run: …`.
- Tests: store CRUD; recorder from a scripted event stream (nested parts, totals); the counter counts calls and tokens from a fake model; delete/rename move or remove runs; a failing store doesn't break the turn.

### Task 5: Same-message agents for `task` (item 7)

**Files:** `src/deep_agents_demo/core/middleware.py` (a `NewAgentDelegation` middleware) or a new `core/delegation.py`, `src/deep_agents_demo/core/assembly.py` (add it to the main agent's middleware when given a newest-workers callable), `src/deep_agents_demo/startup/runtime.py` (pass the callable, the same "refresh then make_worker" the chat's `_NewestWorkers` uses), `src/deep_agents_demo/features/agents/manager.py` (`list_agents` includes agents changed this message; the prompt wording); tests: `tests/test_agents.py` / `tests/test_agent_manager.py` and a new focused test file.
- The middleware intercepts a `task` call whose `subagent_type` isn't in the built agent's subagents but is in the newest factory's names: it builds that agent with the newest factory and runs it on a fresh thread id derived from the tool call id, with the subagent call cap, returns its last AI text as the ToolMessage content. Unknown everywhere → falls through to `task` (Deep Agents' error).
- Tests: a fake model scripts create_agent (approved) then `task` with the new name in the same turn → the new agent's answer is the tool result; an unknown name still errors; `list_agents` shows the new agent.

### Task 6: `--runs` and `--show-run` (item 4, part 2) — after T4 is merged

**Files:** `src/deep_agents_demo/main.py` (flags), `src/deep_agents_demo/ui/cli/` (table and tree printers next to `--list-threads`, Rich in a terminal, plain in a pipe), `README.md` (a "Run history" section and the CLI rows), tests.
- `--runs [N]` (default 20) newest first with a line "Today: N calls · X in / Y out"; `--show-run ID` the run and its parts as a tree; unknown id → a clear message, exit 1.

### Task 7: Eval ranges (item 8) — controller, live

After everything is merged: `uv run python -u evals/orchestration.py --modes single room swarm --runs 3` with the stall watcher; README "All three modes" table with `mean (min–max)` and the paragraph re-checked.

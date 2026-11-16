# Swarm, piece 3: fan out to parallel read-only workers, then merge

Piece 3 of three (eval, room changes, swarm). Written against `c990c1f`
(branch `orchestration-eval`).

## Why

Plans run steps one after another and rooms run turns one after another. Many
requests are a set of independent lookups or checks ("compare these four
suppliers", "review these three files") that could run at the same time. A swarm
lets the main agent split such a goal into independent tasks, run them on
workers in parallel, and have one agent merge the results, so wall time drops
without giving up the review the merge step provides.

## Scope

One pattern: a lead (the main agent) splits a goal into independent tasks,
workers run them in parallel, a merger combines the results. Not in scope: a
shared task board where workers add tasks, agent-to-agent handoffs, workers
that write files, resuming a swarm, GUI work.

## The tool

`run_swarm(goal, tasks, merge_with="general-purpose")`; `tasks` is a list of
`{"task": "...", "agent": "<optional name>"}` (default agent `general-purpose`).
It pauses with `interrupt({"type": "run_swarm", ...})` exactly like `run_plan`,
and the chat answers it on its own thread. It joins `MAIN_ONLY`, the
orchestration feature and its prompt ("only when the user asks, and only for
tasks that don't depend on each other"), `runners_for` in `ui/cli/repl.py`, and
the Rich timeline next to `run_plan` and `start_room`.

*My call:* no `/swarm` command. A list of tasks doesn't fit the one-line
command style of `/room`; you ask for a swarm in chat and the agent writes the
tasks.

## SwarmRunner (`features/orchestration/swarms.py`)

**Validation, before any model call:** 2 to `SWARM_MAX_TASKS` (default 8) tasks,
each with non-empty text; every named agent exists; `merge_with` exists. A
problem returns an error outcome naming it.

**Workers run in parallel** in a thread pool of `SWARM_MAX_PARALLEL` workers
(blank: the value of `SUBAGENT_MAX_PARALLEL`). They share the run's request
`Budget` (lock-protected) and the process's rate limiter. Each task gets its own
thread id (`{thread}:swarm:{id}:{n}`) and its own step (`_StepRun`), so tokens are
counted per worker. `Runner._invoke` and `_run_agent` take the step (and the
approvals) as arguments instead of reading `self.step`; plans and rooms pass
their single step as today.

**Workers are read-only.** Each worker gets an extra `ToolAccess` that allows
only `READ_ONLY` tools (`ls`, `read_file`, `grep`, `glob`, `write_todos`). A
worker's result is its final text. If a worker still reaches an approval
request, it is rejected with "swarm workers cannot ask for approval". No prompt
is ever shown from a worker thread: only the main thread talks to
`Interaction`.

**What you see.** Workers don't stream their text to the terminal: several
answers streaming at once can't be read. Each worker shows as a step with
its agent, status, tokens and time, and the first line of its result, nested
under the `run_swarm` call. The merge step streams as plans' steps do.
*My call.*

**Failures.** A failed worker run is retried once on a fresh thread
(`...:{n}+`), then marked `failed`. The swarm goes on.

**Budget.** A worker stopped by the budget is marked `cut` and keeps its partial
text. After the wave, the main thread asks once whether to continue (rerun
the cut tasks with the budget extended) or wrap up. Non-interactive runs wrap
up.

**Ctrl+C.** First press: no new task starts, running ones finish, then the user
is asked: go on, merge now, or stop. Second press: stop. Nothing more starts.
Runs already in flight can't be cancelled; their results are dropped, and the
swarm returns at once with `ended_by` "stopped". The main thread waits for
workers in short slices so it can react to Ctrl+C on Windows too.

**Merge.** One run of `merge_with` on the main thread, with its normal tools and
approvals. Its input: the goal; every task with its agent, status and result
between markers, described as data, not instructions (the same rule as
`room_followup`); and an explicit list of failed or cut tasks, which the merged
answer must name. If every task failed, there is no merge and the outcome says
so.

**Events.** `StepStarted` / `StepFinished` with `kind="swarm"` for each worker and
one for the merge. Events from worker threads reach the sink through a lock.
The timeline shows workers as parallel branches under `run_swarm`.

**Outcome.** `SwarmOutcome` in `models/outcomes.py`: `goal`, `tasks` (task, agent,
status `done|failed|cut|stopped|not started`, first line of the result, tokens,
seconds), `merged`, `merge_with`, `ended_by` (`done|budget|user|stopped`), totals
(`model_calls`, `input_tokens`, `output_tokens`, `seconds`), `error`, `report`. A
Markdown report (goal, each task and its full result, the merged answer) is
saved under `demo_workspace/swarms/<user>/`, next to a JSON copy of the
outcome.

## Rooms: parallel openings

The room's blind opening round (piece 2) runs in the same kind of pool, for
participants whose tools are all read-only (their definition is `read_only` or
lists only `READ_ONLY` tools). Other participants answer their opening
sequentially, after the parallel ones, as today. Worker factories expose which
agents are read-only (`make.read_only`), so the runner can tell without building
them.

## Settings

    SWARM_MAX_TASKS=8
    SWARM_MAX_PARALLEL=        # blank: SUBAGENT_MAX_PARALLEL

In `config.py`, `.env.example` and the README settings table.

## Eval

`swarm` mode in `evals/orchestration.py`. The main agent is the lead: it gets the
task prompt plus one sentence asking it to split the work with `run_swarm` into
independent tasks and then answer. Its final message is scored. Calls and
tokens cover the lead, the workers and the merger. The README table gains
`swarm` rows next to `single` and `room`, with a short paragraph on when each
mode is worth its cost.

## To verify before building

- The SQLite checkpointer with runs on different thread ids from several Python
  threads at once (a test with a real SQLite file).
- The Rich renderer and timeline with step events arriving from several threads
  (the renderer guards its state with a lock; a test sends interleaved events
  from threads).

## Tests (`tests/test_swarms.py`, fake model)

Validation errors cost no model calls; at most N workers run at once; the
shared budget is never overspent across threads; a failing worker doesn't fail
the swarm and is retried once; a cut worker's partial text reaches the merger;
a worker's approval request is refused without touching `Interaction`; Ctrl+C
once (ask: go on / merge now / stop) and twice (stop); each worker's events carry
its own tokens; write tools are refused for workers; the merger's input lists
failed tasks; the outcome, report and JSON are saved. `SwarmRunner` joins
`tests/test_runner_contract.py`. Room openings: read-only participants run in
parallel, others sequentially.

## Acceptance

- `run_swarm` with one deliberately failing task returns a merged answer that
  names the failed task.
- With `API_REQUESTS_PER_MINUTE=40`, a swarm of 8 tasks completes without
  provider rate-limit errors.
- The README table has `single`, `room` (before and after), and `swarm`, with
  score, calls, tokens and time, and says honestly when each mode is worth its
  cost.
- All existing tests pass unchanged, except where a test asserted behaviour
  this piece changes (named in the commit's report).

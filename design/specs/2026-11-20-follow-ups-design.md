# Follow-ups: clean exit after a stop, run history, file diffs, review reasons, same-message agents, eval ranges

Six items from the open list (numbered as the user saw them). Written against `a465a22`.
Each part stands alone; they share nothing but the repo's rules (`tests/test_structure.py`
passes; only `ui/` prints; `config.py` imports nothing from `features/`; existing tests
pass unchanged unless an item changes what they assert).

## 2. Quitting cleanly after stopping a swarm

**Now:** after a second Ctrl+C, a swarm (and a parallel room opening wave) returns at
once, but its workers run in a `ThreadPoolExecutor`, whose threads Python joins at exit:
typing `exit` waits until every in-flight model call (and its retries) finishes.

**Change:** workers run on **daemon threads** (a small pool of our own, same API as today:
submit, wait in slices, shut down without waiting). Abandoned workers no longer hold up
the process. The stop notice becomes: `[swarm] Stopped. {n} request(s) still running were
abandoned.` The same pool serves the parallel room openings. In-flight provider requests
can't be cancelled from another thread; they are simply abandoned (and their results
dropped, as today).

**Tests:** a stopped swarm with a worker blocked forever → the swarm returns, the worker
thread is a daemon, the notice wording; the same for a room opening wave; everything else
unchanged (the existing swarm/room Ctrl+C tests).

## 4. Run history: `--runs` and `--show-run`

The run store from the 11 Nov brainstorm, with the decisions made then: for observability
and usage tracking (not replay); tokens only, no cost; one run per user turn, with plans,
rooms, swarms and subagents nested as children; per `--user`; a new SQLite file in `data/`.

**Store** (`persistence/runs.py`, `data/runs.sqlite`): `runs` (id, user, chat, started_at,
ended_at, seconds, model, prompt — first line, up to 120 characters — model_calls,
input_tokens, output_tokens, status `done|error|interrupted`), and `run_parts` (run_id,
parent part or none, kind `subagent|plan|room|swarm|step`, label, model_calls,
input_tokens, output_tokens, seconds, status).

**Counting** (*my call*): one usage counter on every chat model the app builds
(`providers/chat.py`), so a turn's calls and tokens are exact whatever ran in it (main
agent, subagents, plan workers, room participants, swarm workers, the moderator). The
parts come from the events the chat already emits (`SubagentStarted/Finished`,
`StepStarted/Finished`, and the `run_plan`/`start_room`/`run_swarm` tool calls);
`StepFinished` gains `model_calls`.

**Recording:** the chat records each turn when it ends (also on error or Ctrl+C), through
a small recorder that watches the event stream; a recording failure never breaks the chat
(it says so once). One-shot `--prompt` turns are recorded too.

**Commands:**

    uv run deep-agents-demo --runs [N]          # the user's last N turns (default 20), newest first, with a total for today
    uv run deep-agents-demo --show-run ID       # one turn and its parts as a tree

`--runs` prints a table (when, chat, prompt, calls, tokens in/out, time) and `--show-run`
a tree; both are plain text in a pipe and Rich tables in a terminal, like `--list-threads`.
Deleting a chat removes its runs; renaming moves them.

**Tests:** the store (write, list, show, delete/rename with a chat), the recorder from a
scripted event stream (nested parts, totals equal to the counter), the counter on a fake
model, the commands' output, a recording error doesn't break a turn.

## 5. Diffs between versions of a delivered file

`/files diff NAME [V1 V2]` shows a coloured unified diff between two versions of a file the
agent delivered in this chat (default: the last two). Text files only; an image or a
missing version gets a one-line explanation. The same diff colours as the approval panel.
When `send_file` delivers v2 or later, its result to the agent also says
`changed from v1: +12 −3 lines` so the agent can tell you what it changed.

*Not in this piece (my call):* rendering a delivered page in a GUI — there is no GUI front
end yet; it belongs with one.

**Tests:** diff of two versions (default and explicit), an image, a missing version, the
`send_file` change line.

## 6. The reason shown for a plan step that fails review

**Now:** a step that failed review reports the worker's result as its summary, so the
timeline shows the start of the worker's text after "✗ failed". **Change:** a failed step's
summary is the reviewer's notes (the verdict's reason); the timeline shows
`✗ failed review: <notes>`; the worker's result is still in the plan's outcome. A step that
failed for another reason (no such agent, a crash) shows that reason as today.

**Tests:** the step-finished summary for a failed review is the notes; the timeline line.

## 7. A new agent usable by the main agent in the same message

**Now:** after "create a pricing-expert agent and ask it for a quote", the agent is saved,
but the main agent's `task` tool knows only the agents it was built with, so it fails
until the next message. Rooms, plans and swarms already see new agents at once.

**Change** (*my call*): a middleware around the main agent's tool calls. When `task` names
an agent that the built agent doesn't know but the newest agent list has (an agent created
or installed earlier in this message), it runs that agent from the newest worker factory,
as a subagent would run (same limits, approvals and the read/write rules of that agent),
and returns its final answer as the tool result. Everything else goes to `task` as today.
`list_agents` shows agents created in this message too. The prompt line "available from
the next message" becomes "available at once".

**Tests:** create an agent, then `task` it in the same turn → its answer comes back; an
unknown name still gets Deep Agents' error; `list_agents` lists the new agent.

## 8. Min–max ranges in the eval table

No code. After the other items are merged: `evals/orchestration.py --modes single room
swarm --runs 3`, then the README "All three modes" table shows `mean (min–max)` per cell
and the paragraph is re-checked against the ranges.

## Out of scope

A GUI file preview; cancelling an in-flight provider request; cost per run.

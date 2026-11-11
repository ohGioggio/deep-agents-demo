# Subagents registry and subagent-driven execution

Status: design agreed in chat on 2026-11-11 ("its good, write the design"). Not built.

## Intent

Some work is too big or too varied for one agent in one context: a long task with
a plan, independent pieces that can run side by side, or a problem that benefits
from several viewpoints. The user wants to define their own subagents (the way
Claude Code does with `.claude/agents/*.md`), manage them like skills, and ask the
main agent to work through them: step by step with a reviewer, in parallel, or as a
moderated discussion ("room") they can watch and join.

Success looks like: a Claude Code agent file dropped into a folder just works; the
user can say "do this plan with subagents" or "have the architect, critic and tester
discuss X" and watch it happen, step in, and get a clear result; nothing a subagent
does escapes the approvals, guardrails and tool limits that apply to the main agent.

## Decided by the user

- **Four uses.** A long task with a plan; parallel execution; an orchestration or
  "room" of different agent roles; plus ordinary delegation to a named agent.
- **Claude Code compatible format.** Agent files use Claude Code's Markdown +
  YAML frontmatter format, so files move both ways.
- **Two places to keep agents:** a folder of `.md` files and the SQLite database.
- **Rooms:** participants take turns, a moderator runs the discussion, and it ends
  with a summary. The user watches live and can step in.
- **Plans:** a reviewer checks each step before the next one starts.
- **Managed like skills** (the overall picture was approved): create, update,
  remove, list; changes made through the chat need approval and are versioned; a
  marketplace for sharing. (The user approved this as part of the full picture;
  the details below marked "my call" were proposed by Claude.)

## Non-goals (for now)

- Subagents spawning their own subagents (no nesting).
- Claude Code features with no equivalent here: `hooks`, `mcpServers`,
  `isolation: worktree`, `memory` (per-agent memory dirs), `background`,
  `initialPrompt`, `effort`, `autoCompactWindow`, `omitClaudeMd`, `experimental`.
  They are kept in the file on round-trip but ignored, with a note when loaded.
- Agents writing tools or code that becomes tools (unchanged rule).
- Agent teams with direct messaging between agents outside a room.

## Anti-goals (would make this a failure even if it works)

- **An agent file loosening control.** `permissionMode: bypassPermissions`,
  `acceptEdits`, `dontAsk` or `auto` must never skip an approval here. Approvals,
  guardrails, `DISABLED_TOOLS` and the sandbox skill gate apply to every subagent.
- **Runaway cost.** Rooms, retries and parallel work must have hard limits; the
  free NVIDIA tier is ~40 requests/minute, shared by everything.
- **Losing the user.** A room the user can't interrupt, or a plan that silently
  skips failed steps.

## Constraints

- Deep Agents 0.7.34 takes subagents as `SubAgent` specs at build time (`name`,
  `description`, `system_prompt`, `tools`, `model`, `middleware`, `interrupt_on`,
  `skills`, `permissions`) and exposes them through its `task` tool. The demo
  already builds its own `general-purpose` and `researcher` specs in `build_agent`.
- Every subagent gets `ToolAccess` (disabled tools, skill-only tools, skill gates)
  and inherits `interrupt_on`, as `general-purpose` does today.
- Models are any OpenAI-compatible endpoint (today NVIDIA NIM); Claude model aliases
  don't exist there.
- One shared rate limiter (`API_REQUESTS_PER_MINUTE`).
- The chat loop is a terminal REPL; approvals already pause the graph with
  LangGraph interrupts and resume with `Command(resume=...)`.

## Design

### 1. The agent file format

Same as Claude Code: YAML frontmatter, then the body as the system prompt.

```markdown
---
name: critic
description: Finds weaknesses in a proposal. Use when a plan or design needs challenging.
tools: Read, Grep, Glob
model: inherit
---
You are a sharp, fair critic. ...
```

| Field | Here |
| --- | --- |
| `name` (required) | The agent's id, used by `task` and rooms. Accepted: letters, digits, `-`, `_`, up to 64 chars (my call: stricter than Claude Code's 256, no `:`, so names are safe in commands and file names; anything else is skipped with a warning). Files written here use lowercase-hyphen names. The file name doesn't have to match. |
| `description` (required) | When to use it; shown to the main agent and the moderator. |
| `tools` | Comma-separated string or YAML list. Omitted = all tools the main agent has (as in Claude Code). Translated, see below. |
| `disallowedTools` | Denylist, applied first. Same translation. |
| `model` | `inherit` or omitted = the configured chat model. A full model id is used as given (same endpoint and key). `sonnet`, `opus`, `haiku`, `fable` = the configured model, with a one-line note (my call). |
| `skills` | Skill names whose text is pinned into the subagent's context at start (Deep Agents' `pinned_skills`). |
| `maxTurns` | Honoured as a cap on the subagent's model calls for one task. |
| `permissionMode` | `plan` = read-only tools (`ls`, `read_file`, `grep`, `glob`, `write_todos`). Every other value is ignored with a note; it can never loosen approvals. |
| `color` | Used for the speaker label in the terminal (my call). |
| everything else | Kept on round-trip, ignored, noted once at load. |

**Tool translation** (Claude Code name -> ours; our own names are accepted too):
`Read` -> `read_file`, `Write` -> `write_file`, `Edit`/`MultiEdit` -> `edit_file`,
`Bash`/`PowerShell` -> `execute`, `Grep` -> `grep`, `Glob` -> `glob` and `ls`,
`TodoWrite` -> `write_todos`, `WebFetch`/`WebSearch` -> `web_fetch`/`web_search` once
the web feature exists. `Agent`/`Task` (spawning) is dropped: no nesting. Names that
don't resolve (e.g. `NotebookEdit`, `mcp__*`) are listed as warnings; if nothing
resolves, the agent isn't loaded. The final list is intersected with the tools
registry and `DISABLED_TOOLS`. A skill-only tool listed explicitly is visible to
that agent from the start (my call: the agent exists for that job).

### 2. Where agents come from, and precedence

Lowest to highest; a higher source wins on a name clash:

1. **Built-ins:** `general-purpose`, `researcher` (both exist), plus new `reviewer`
   and `moderator`.
2. **Folders** the user manages: `demo_workspace/agents/` (seeded with a README and
   one example, e.g. `critic.md`), plus any folders in `AGENT_DIRS` (for example the
   user's `~/.claude/agents` or a project's `.claude/agents`). Scanned recursively,
   loaded at start. The agent can read them but not write them (same deny rule as
   skill folders).
3. **Installed marketplace agents** (follow the latest marketplace version).
4. **The user's own agents in SQLite**, created or changed through the chat.

A user agent named like a built-in replaces it (so the user can bring their own
`reviewer` or `moderator`). Unlike skills, agents are not written to a folder:
`build_agent` turns the registry into `SubAgent` specs directly. After an approved
change the runtime rebuilds the agent, so the change is available from the next
message (same promise as skills).

### 3. Managing agents (like skills)

Gated tools, each paused for approval with the same prompt (summary, `[d]iff`,
`[f]ull file`, approve or reject with a reason): `create_agent`, `update_agent`,
`remove_agent`, `import_agent` (a `.md` file or a folder of them, e.g. a
`.claude/agents` folder), `restore_agent`, `publish_agent`, `install_agent`,
`uninstall_agent`. Read-only: `list_agents` (all sources, with where each comes
from and what overrides what), `agent_history`, and marketplace browsing.

- **Only on request**, as with skills. The summary shows the agent's tools after
  translation, its model, and warnings (unknown tools, ignored fields).
- **Validated before you're asked:** required fields, name rules, tools exist, no
  credentials in the prompt (reusing `memory_policy`), size limits. Invalid changes
  are rejected automatically with the reasons.
- **Versioned:** every approved change is a version; remove and restore add
  versions. Scopes `user` and `marketplace`, as for skills.
- **Storage (my call):** reuse the skill store's design (tables for items, versions
  and installs) generalised with a `kind` column (`skill` / `agent`), or a parallel
  `AgentStore`; the builder chooses. One file per agent version (the `.md` text).
- **Chat command (my call):** `/agents` lists them (name, source, tools, model) for
  quick orientation. Management itself goes through the chat, with approval.

### 4. Delegation and parallel execution

- The main agent's `task` tool lists every loaded agent (name + description), as
  Deep Agents does today.
- **Parallel:** several `task` calls in one step already run concurrently. A cap
  (`SUBAGENT_MAX_PARALLEL`, default 3, my call) is enforced by middleware on `task`
  so a burst can't exceed it; extra calls wait. All share the rate limiter, so on
  the free tier parallel work overlaps rather than multiplies speed.
- Output is labelled with the agent's name: `[researcher / tools]`.

### 5. A long task with a plan, with review

Used when the user asks for it ("do this plan with subagents", "use subagent-driven
execution") or, when a task is clearly long and multi-step, when the main agent
proposes it and the user agrees (my call: don't switch modes silently).

1. The main agent writes the plan (each step names a worker agent, default
   `general-purpose`) and hands it over with `run_plan(goal, steps)`.
2. **The plan runs on the chat loop's thread, like a room (my call).** `run_plan`
   pauses the graph with an interrupt carrying the plan; the chat loop's plan runner
   executes it and resumes the tool with the outcome. Reason: a worker's approval
   would otherwise pause the graph mid-tool, and on resume LangGraph re-runs the
   whole tool, repeating finished attempts and reviews (extra cost, possibly a
   different result); Ctrl+C also couldn't pause cleanly inside the graph.
3. For each step, in order, the runner runs a fixed loop so the review can't be
   skipped:
   - the **worker** runs with a fresh context: the overall goal, this step, and short
     notes from earlier steps (not the whole history);
   - the **reviewer** checks the result against the step and returns a verdict
     (`pass` or `fail` plus concrete notes), as structured output;
   - on `fail`, the worker gets the notes and tries again; after
     `PLAN_MAX_ATTEMPTS` (default 2) failed reviews the step has failed.
4. **A failed step stops the plan, in code (my call).** The runner asks the user
   directly: `[r]etry  [s]kip  [x] stop`. Retry runs the step again (fresh
   attempts), skip records it as skipped and continues, stop ends the plan. Nothing
   moves on silently. With `--prompt` (nobody to ask) a failed step stops the plan.
5. **Ctrl+C pauses between steps (my call)**, with the same `[plan] You:` prompt as
   rooms: a note for the next worker, `/stop` to end, Enter to resume.
6. The terminal shows each step's worker, the review verdict, and the plan's
   progress (`[plan] step 2/5 ✓ reviewed`). The outcome returned to the main agent
   lists each step as passed, skipped or failed, with the reviewer's notes.

Smaller plans the main agent can still do itself with `write_todos` and `task`;
`run_plan` is for when the user asks for subagent-driven execution, or agrees when
the main agent proposes it for a clearly long task.

### 6. Rooms: a moderated discussion

Started by the user (`/room architect critic tester -- <question>`) or by the main
agent through a `start_room(question, participants)` tool when asked ("have the
architect, critic and tester discuss X").

- **Moderator** (built-in `moderator`, replaceable): before each turn it decides,
  as structured output, who speaks next and with what prompt, or that the room is
  done. It ends when there is agreement, nothing new is said, or the turn limit
  (`ROOM_MAX_TURNS`, default 12) is reached.
- **Participants** are the user's agents, with their own tools, limits and
  approvals. Each turn sees the question and the discussion so far (trimmed to fit
  if long) and adds one contribution.
- **Live in the terminal:** every turn prints as it happens, `[room / critic]`,
  `[room / moderator -> tester]`.
- **Stepping in:** Ctrl+C pauses after the current turn and shows `[room] You:`.
  Text becomes the user's turn (the moderator takes it into account); `/end` asks
  the moderator to summarise now; `/stop` ends without a summary; Enter resumes.
  Outside a room, Ctrl+C keeps its current meaning.
- **Ending:** the moderator writes a summary: agreements, open disagreements, next
  steps. It goes back to the main agent as the tool's result (or, with `/room`, is
  added to the chat so the next message can build on it). The full transcript is
  saved to `demo_workspace/rooms/<user>/<timestamp>-<slug>.md` (my call).
- **With `--prompt`:** nobody can step in; the room runs to its summary.
- **How it runs (my call, the main design risk):** the room loop must run on the
  chat loop's thread so Ctrl+C and approvals work. `start_room` therefore pauses
  the graph with a LangGraph interrupt carrying the room request; the chat loop
  runs the room (moderator and participants as separate agent invocations, their
  approval interrupts handled by the same `Approver`), then resumes the tool with
  the summary. `/room` calls the same runner directly.

### 7. Cost limits (my call)

Counting only turns isn't enough: one participant turn or worker attempt can make
many model calls. So, all configurable:

- **Per subagent run:** at most `SUBAGENT_MAX_CALLS` model calls (default 25),
  enforced by middleware on every subagent; an agent file's `maxTurns` can lower
  it, never raise it. Hitting it ends that run with what it has, marked partial.
- **Per room or plan:** a request budget `ORCHESTRATION_MAX_CALLS` (default 150)
  shared by every call inside it (moderator, participants, workers, reviewers).
  When it runs out, the runner pauses and asks the user to continue (another
  budget) or wrap up (rooms summarise, plans stop); with `--prompt` it wraps up.
- `ROOM_MAX_TURNS` counts participant contributions only; moderator decisions and
  the user's interjections don't count (they're bounded by the budget).
- The `task` tool keeps today's `recursion_limit` plus the per-run cap.

### 8. Settings

`AGENT_DIRS` (extra agent folders, separated by `;` on Windows or `:` elsewhere),
`SUBAGENT_MAX_PARALLEL` (3), `PLAN_MAX_ATTEMPTS` (2), `ROOM_MAX_TURNS` (12),
`SUBAGENT_MAX_CALLS` (25), `ORCHESTRATION_MAX_CALLS` (150). All validated in
`Settings.from_env`.

## Pieces (suggested order)

1. **Registry and format:** parse/write Claude Code agent files, tool translation,
   folder loading with precedence, built-in `reviewer` and `moderator`, `/agents`,
   specs into `build_agent`, `ToolAccess` and approvals on every agent.
2. **Agent-written agents:** gated tools, validation, versions, import, rebuild after
   change; then the marketplace.
3. **Plans with review, the parallel cap and budgets:** `run_plan` and its runner,
   structured verdicts, retries, asking the user on failure, call budgets.
4. **Rooms:** moderator, the interrupt-based runner, live output, stepping in,
   summary and transcript.

## Testing

- Offline, with fake models (as the existing suites do): parsing and writing
  real Claude Code agent files (round-trip keeps unknown fields), tool translation
  and warnings, precedence, `permissionMode` never loosening approvals, approvals
  and `DISABLED_TOOLS` holding inside every subagent, the parallel cap, `run_plan`
  retry/skip/stop paths, the budgets, a worker approval not re-running finished
  work, the moderator's choices, room limits, `/end` and `/stop`.
- Live with the NVIDIA model: import a Claude Code agent and use it; a three-step
  plan where the reviewer rejects a deliberately weak first attempt; three parallel
  lookups; a three-agent room with the user stepping in once.

## Calls made by Claude (check these)

- Names limited to letters, digits, `-`, `_` (64 chars).
- Claude model aliases fall back to the configured model with a note.
- `permissionMode: plan` = read-only; all other modes ignored.
- Skill-only tools listed explicitly in an agent's `tools` are visible to it.
- Folder: `demo_workspace/agents/` plus `AGENT_DIRS`; a DB agent beats a folder one.
- `/agents` command for listing; management through the chat.
- Plan mode is offered, not switched on silently, when the user didn't ask.
- Defaults: 3 parallel, 2 attempts, 12 room turns; room transcripts saved.
- Rooms and plans both run on the chat loop's thread via an interrupt
  (`start_room`, `run_plan`), so approvals and Ctrl+C work and finished work is
  never re-run.
- A failed plan step stops the plan in code and asks `[r]etry [s]kip [x] stop`;
  with `--prompt` it stops.
- Budgets: 25 model calls per subagent run, 150 per room or plan (then ask to
  continue or wrap up).
- An agent without `tools` gets the main agent's tools minus `task`, `run_plan` and
  `start_room` (no nesting); management tools stay, still behind approval.
- Full model ids starting with `claude-` fall back like the short aliases.
- Installed marketplace agents pick up a new version at the next rebuild, not
  mid-chat.

## Left to the builder

- One generalised store or a parallel `AgentStore`.
- The exact prompts for `reviewer` and `moderator`, and the structured-output
  schemas for verdicts and moderator decisions (with a text fallback if a model
  doesn't support structured output).
- How much earlier-step context a worker gets, and how room history is trimmed.
- How streamed output of agents invoked inside a tool reaches the terminal.

# Subagents registry, piece 4: moderated rooms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let several of the user's agents discuss one question in a "room": a moderator picks who speaks next and when to stop, participants take turns, the user watches live and can step in with Ctrl+C, and the room ends with a summary that reaches the main agent and a transcript saved to disk.

**Architecture:** Reuse piece 3:
- **Shared runner base:** `PlanRunner`'s budget, run-one-agent, Ctrl+C and thread handling move into a base class, `Runner`, in `orchestration.py`.
- **`RoomRunner`** (new `rooms.py`) extends `Runner`. The `start_room` tool pauses the turn with a LangGraph interrupt that `stream_turn` hands to the room runner, as `run_plan` does.
- **`/room` command:** runs the same runner directly, then sends the summary to the main agent as one ordinary turn, so the next message can build on it.

**Tech Stack:** Python 3.11+, LangGraph `interrupt`, the piece-3 `worker_factory`, unittest with fake models.

**Spec:** `design/specs/2026-11-11-subagents-registry-design.md` §6 (rooms), §7 (cost limits), §8 (settings). Pieces 1–3 are on `main` (`3281146`).

## Global Constraints

- **Starting a room:** by the user with `/room a b c -- <question>`, or by the main agent with `start_room(question, participants)` when asked.
- **Participants:** at least 2, all existing agents, not `moderator`.
- **The moderator** (the built-in `moderator`, or the user's own agent of that name) decides before each turn who speaks next and what they should address, or that the room is done. It ends the room when the participants agree, when nothing new is being said, or at the turn limit.
- **Turn limit:** `ROOM_MAX_TURNS` (default 12) counts participant contributions only. Moderator decisions and the user's turns don't count; they're bounded by the budget.
- **Budget:** `ORCHESTRATION_MAX_CALLS` (150) model calls per room. When it runs out, the user chooses continue or wrap up; with `--prompt` it wraps up. The summary is always written: its run isn't capped by the room's budget, though `SUBAGENT_MAX_CALLS` still applies.
- **Participants keep their own tools, limits and the user's approvals.**
- **Live output:** `[room] moderator -> critic: <what to address>` before each turn, then the participant's streamed messages. Plain ASCII only: Windows can't print other characters when output is redirected.
- **Stepping in (interactive only):**
  - Ctrl+C pauses after the current turn and shows `[room] You:`.
  - Text becomes the user's turn in the transcript, and the moderator takes it into account.
  - `/end` asks for the summary now; `/stop` ends without a summary; Enter resumes.
  - A second Ctrl+C stops at once.
- **The summary:** what was agreed, the open disagreements, and the next steps.
- **The transcript** is saved to `demo_workspace/rooms/<user>/<YYYYmmdd-HHMMSS>-<slug>.md`, where the slug is the first 40 characters of the question as lowercase words joined by hyphens.
- **The outcome** returned to the main agent: `question`, `participants`, `turns`, `summary`, `transcript`, `ended_by` (one of `moderator`, `turn limit`, `user`, `budget`, `stopped`) and `note`.
- **Each turn and moderator decision runs on its own thread:** `f"{chat thread}:room:{room_id}:{n}:{name}"`. Deleting or renaming a chat handles its `:room:` threads as it does `:plan:` threads.
- **Moderator replies are read from text, not structured output:**
  - first line `NEXT: <name>`, then the prompt for that participant;
  - or first line `END`, then the summary.
  - Anything else: the next participant in rotation speaks, with the moderator's text as their prompt, and a note in the transcript.
- **Commits:** one-line messages, no trailer; push only if asked.
- **Tests:** `uv run python -m unittest discover -s tests` (single file: `-p "test_rooms.py"`).

## Review Focus

1. **The moderator names someone who isn't in the room** (e.g. a typo, or itself). Expected: the next participant in rotation speaks, and nothing crashes. (Task 3 test.)
2. **A participant's run fails** (provider error) or needs an approval. Expected: the failure becomes a transcript line "(<name> could not answer: ...)" and the room goes on; approvals prompt as usual. (Task 3 test.)
3. **The budget runs out in the middle of a turn.** Expected: the user chooses continue or wrap up, and wrap up still produces a summary. (Task 4 test.)
4. **`/room` with a bad line:**
   - no `--`;
   - fewer than 2 participants;
   - an unknown agent.

   Expected: a usage or error message, and no model calls. (Task 5 test.)
5. **A long discussion.** Expected: each turn's input includes the transcript trimmed to its last 6,000 characters, with a "(earlier turns omitted)" marker, so a long room can't overflow a context. (Task 3 test.)

---

### Task 1: A shared runner base, and the room settings

**Files:**
- Modify: `src/deep_agents_demo/orchestration.py`, `src/deep_agents_demo/config.py`, `src/deep_agents_demo/threads.py`, `.env.example`
- Test: `tests/test_orchestration.py` (whole file must stay green), `tests/test_config.py`, `tests/test_rooms.py` (new; `SettingsTests`, `CleanupTests`)

**Interfaces:**
- **Produces:**
  - `class Runner` in `orchestration.py`, holding what `PlanRunner` has today:
    - attributes: `make`, `approver`, `thread`, `budget_size`, `interactive`, `ask`, `say`, `budget`, `pause_requested`;
    - methods: `_on_sigint`, `_invoke(graph, text, thread_id) -> str`, `_run_agent(name, text, thread_id, capped=True) -> str` (`capped=False` builds the agent without a `BudgetCap`), `_continue_after_budget() -> bool`, `_with_ctrl_c(fn, *args)` (installs and restores the SIGINT handler when interactive, then calls `fn`).
    - `label: str` is the prefix for messages, `"[plan]"` or `"[room]"`.
  - `class PlanRunner(Runner)`: same constructor and behaviour as today.
  - `Settings.room_max_turns: int = 12` (env `ROOM_MAX_TURNS`, at least 1).
  - `Chats.delete` and `Chats.rename` treat `:room:` threads like `:plan:` threads. Replace the `":plan:"` literal with a tuple `ORCHESTRATION_THREADS = (":plan:", ":room:")` in `threads.py`.

- [ ] **Step 1: Write the failing tests:**
  - `test_config.py::test_room_max_turns`: default `12`; `ROOM_MAX_TURNS=0` raises `ValueError` naming it.
  - `test_rooms.py::CleanupTests::test_deleting_a_chat_removes_its_room_threads`: as `test_orchestration.ChatCleanupTests`, with a thread `f"{chat}:room:abc:1:critic"`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** the base class extraction (no behaviour change), the setting (plus its `.env.example` line) and the thread prefixes.
- [ ] **Step 4: Run the full suite.** Expected: OK, with all piece-3 tests unchanged.
- [ ] **Step 5: Commit**: `git commit -m "Runner base shared by plans and rooms; ROOM_MAX_TURNS; chat delete/rename cover room threads"`

---

### Task 2: The `start_room` tool

**Files:**
- Modify: `src/deep_agents_demo/orchestration.py` (tool, prompt), `src/deep_agents_demo/chat.py` (`stream_turn` fallback)
- Test: `tests/test_rooms.py` (class `StartRoomToolTests`)

**Interfaces:**
- **Produces:**
  - `start_room(question: str, participants: list[str]) -> str`. It checks that `question` isn't empty and that there are at least 2 distinct participants, none of them `moderator`; otherwise it returns a JSON error without interrupting. It then calls `interrupt({"type": "start_room", "question": ..., "participants": [...]})` and returns `json.dumps(<resume value>)`.
  - `orchestration_feature()` now has tools `(run_plan, start_room)`. `PLAN_PROMPT` gains one sentence: `start_room` runs a moderated discussion between the named agents and returns a summary; use it only when the user asks for one.
  - `stream_turn`: a `start_room` pause with no runner is answered `{"error": "rooms can't run here"}`.

- [ ] **Step 1: Write the failing tests** (pattern of `test_orchestration.RunPlanToolTests`):
  - `test_start_room_pauses_and_returns_the_runner_answer`: the runner receives `{"type": "start_room", "question": "q", "participants": ["critic", "helper"]}`, and the model sees `json.dumps(answer)`.
  - `test_room_needs_two_participants_and_no_moderator`: `["critic"]` and `["critic", "moderator"]` each return an error containing `participants`; the runner is never called.
  - `test_without_a_runner_the_room_is_refused`.
  - `test_subagents_cannot_start_rooms`: as `ParallelTests.test_subagents_cannot_run_plans`, with `start_room`.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "start_room tool: pauses the turn for a room runner"`

---

### Task 3: RoomRunner: moderator, turns, summary, transcript

**Files:**
- Create: `src/deep_agents_demo/rooms.py`
- Test: `tests/test_rooms.py` (class `RoomTests`)

**Interfaces:**
- **Consumes:** `Runner` (Task 1); `worker_factory`'s `make` (piece 3).
- **Produces:**
  - `TRANSCRIPT_LIMIT = 6000`.
  - `parse_moderator(text: str, participants: list[str]) -> tuple[str, str, str]`:
    - `("next", name, prompt)` when the first non-empty line is `NEXT: <name>` and the name is a participant (case-insensitive, ignoring `*`, `#` and spaces);
    - `("end", "", summary)` when the first line is `END`;
    - otherwise `("unclear", "", text)`.
  - `class RoomRunner(Runner)`:
    - `__init__(self, make, approver, *, thread, max_turns=12, budget=150, interactive=True, ask=input, say=print, transcripts: Path | None = None, describe=None)`. `describe(name) -> str` gives a participant's description for the moderator; the default is `""`.
    - `run(self, request: dict) -> dict` returns the outcome described in Global Constraints.
  - **Each moderator decision** runs `make("moderator")` on its own thread. Its input:
    - the question;
    - the participants with their descriptions;
    - how many turns are used and left;
    - the trimmed transcript;
    - the exact reply format: `NEXT: <name>` then a prompt, or `END` then the summary (agreements, open disagreements, next steps).
  - **Each participant turn** runs `make(name)` on its own thread. Its input: the question, the trimmed transcript, `The moderator asks you: <prompt>`, and `Reply with your contribution only: a few short paragraphs.`
  - **Transcript lines:** `**<name>:** <text>`, `**You:** <text>`, `**Moderator:** <prompt>` (each decision's prompt), and `*(note)*` lines.
  - **Ending:**
    - on `END`, the summary is the moderator's text;
    - on the turn limit, `/end` or budget wrap-up, the summary comes from one more moderator run with `capped=False` and the input "Write the summary now: …";
    - `/stop` gives no summary (`summary: ""`).
  - **The transcript file:** `# Room: <question>`, `Participants: ...`, every line, then `## Summary`. It's written only if `transcripts` is set; it creates the folder.
  - **A participant run that raises:** the transcript gets `*(<name> could not answer: <error>)*` and the turn counts.

- [ ] **Step 1: Write the failing tests** (a shared scripted fake model through `worker_factory`, as in `test_orchestration.RunnerBase`, with agents `critic` and `helper`):
  - `test_parse_moderator`:
    - `"NEXT: critic\nWhat breaks?"` → `("next", "critic", "What breaks?")`;
    - `"**END**\nAgreed: ship."` → `("end", "", "Agreed: ship.")`;
    - `"NEXT: nobody\nx"` → `"unclear"`.
  - `test_moderated_room_to_the_end`:
    - moderator `NEXT: critic` → critic speaks;
    - moderator `NEXT: helper` → helper speaks;
    - moderator `END\nAgreed: X` → outcome `turns == 2`, `ended_by == "moderator"`, `summary == "Agreed: X"`;
    - helper's input contains the critic's contribution;
    - `say` shows `[room] moderator -> critic: ...`.
  - `test_unclear_moderator_falls_back_to_rotation`: the moderator says `Let's hear from someone` → the first participant speaks with that text as its prompt, and a note is added.
  - `test_turn_limit_then_summary`: `max_turns=1`; the moderator keeps saying NEXT → after 1 turn a summary run → `ended_by == "turn limit"`.
  - `test_failed_participant_run_goes_on`: `make` returns a raising graph for `helper` → the transcript contains `helper could not answer`, and the room continues to END.
  - `test_long_transcripts_are_trimmed`: three contributions of 4,000 characters each → the next input is at most about 6,000 characters of transcript and contains `(earlier turns omitted)`.
  - `test_transcript_file`: with `transcripts=tmp`, one `.md` file whose name ends with the question's slug, containing `## Summary` and both contributions; `outcome["transcript"]` is its path.
  - `test_participant_approvals`: a participant with all tools calls `delete` under `Guardrails.feature()` → the `Approver` is asked once.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** `rooms.py`.
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "RoomRunner: moderator picks speakers, turns, summary, transcript"`

---

### Task 4: Stepping in, budget, and one-shot runs

**Files:**
- Modify: `src/deep_agents_demo/rooms.py`
- Test: `tests/test_rooms.py` (class `SteppingInTests`)

**Interfaces:**
- **Produces:** on a pause (the pause flag is set), after the current turn: `ask("[room] You: (your turn, /end for the summary now, /stop to end without one, Enter to go on) ")`.
  - Text → a `**You:**` line; the next moderator input includes it under "The user just said:".
  - `/end` → summary, `ended_by == "user"`.
  - `/stop` → no summary, `ended_by == "stopped"`.
  - `""` → resume.
  - **Budget:** `_run_agent` raises piece 3's wrap-up signal when the user (or `--prompt`) chooses wrap up → summary run, `ended_by == "budget"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_user_turn_reaches_the_moderator`: `runner.pause_requested = True` before `run`; `ask` answers "focus on cost" → the transcript has `**You:** focus on cost`, and the first moderator input contains "The user just said: focus on cost".
  - `test_end_and_stop`: pause, then `/end` → `ended_by == "user"` and a summary is present; a separate run with `/stop` → `summary == ""`, `ended_by == "stopped"`.
  - `test_budget_wrap_up_still_summarises`: `budget=2`, `interactive=False` → `ended_by == "budget"`, and `summary` is the summary run's text.
  - `test_one_shot_ignores_ctrl_c`: with `interactive=False`, the SIGINT handler is untouched (as `test_ctrl_c_is_left_alone_when_not_interactive`).
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Rooms: step in with Ctrl+C, /end and /stop, budget wrap-up still summarises"`

---

### Task 5: `/room`, wiring, docs, live check

**Files:**
- Modify:
  - `src/deep_agents_demo/chat.py`:
    - `runners_for` adds `"start_room": RoomRunner(...).run`, with `transcripts=settings.paths.workspace / "rooms" / <safe user>` and `describe` from `runtime.agents`;
    - a `/room` command on `ChatSession`;
    - `run_chat` passes a room callback to the session.
  - `src/deep_agents_demo/app.py`: nothing new beyond what `runners_for` needs.
  - `README.md`: a "Rooms: a moderated discussion" section.
- Test: `tests/test_rooms.py` (classes `CommandTests`, `WiringTests`)

**Interfaces:**
- **Produces:**
  - `parse_room_command(arg: str) -> tuple[list[str], str] | str` returns `(participants, question)` or an error message. It splits on the first ` -- `; participants are separated by spaces or commas.
  - `ChatSession(..., room=None)`: `room(participants, question)`, a callable set by `run_chat`, runs the room and sends the result to the main agent as an ordinary turn. The text: `I ran a discussion room. Question: <q>. Participants: <...>. Summary: <summary>. Transcript: <path>. Keep this in mind; reply with one short line.`
  - `/room` validates with `parse_room_command` and the registry: existing agents only, at least 2, no `moderator`. It prints errors without running anything.

- [ ] **Step 1: Write the failing tests:**
  - `CommandTests.test_parse_room_command`:
    - `"critic, helper -- Is X safe?"` → `(["critic", "helper"], "Is X safe?")`;
    - `"critic helper"` → an error mentioning `--`;
    - `"critic -- q"` → an error mentioning "at least 2".
  - `CommandTests.test_room_command_checks_agents_first`: `/room critic nobody -- q` with a registry lacking `nobody` → `say` shows "no agent named 'nobody'", and the room callback is never called.
  - `WiringTests.test_start_room_through_the_runtime`: the real `build_runtime` with placeholder settings and one shared fake model scripted as main → `start_room`, then moderator NEXT, a participant, moderator END. The main agent's last model call then sees a tool result whose JSON has `ended_by == "moderator"`, and the transcript file exists under the workspace's `rooms/` folder.
- [ ] **Step 2: Run them; verify they fail.**
- [ ] **Step 3: Implement** the command, the wiring and the README section: how to start a room, what you see, stepping in, `/end` and `/stop`, the limits, and the transcript location. Add `rooms.py` to the layout list.
- [ ] **Step 4: Run the full suite and a smoke build.** Expected: OK.
- [ ] **Step 5: Commit**: `git commit -m "Rooms in the chat: /room, start_room runner, docs"`
- [ ] **Step 6: Live check** (NVIDIA model; memory and RAG off; user `tools-check`; piped interactive input; `MSYS_NO_PATHCONV=1`):
  - `/room critic general-purpose -- Should a demo CLI store chat history in SQLite or JSON files?`
    - expect `[room] moderator -> ...` lines, at least 2 turns, a summary, and a transcript file;
    - the next message "What did the room decide?" is answered from the summary.
  - Ask the main agent to "have the critic and general-purpose discuss whether to add a web UI, briefly" → it calls `start_room`; the outcome is reported.
  - Clean up the transcripts and chats.

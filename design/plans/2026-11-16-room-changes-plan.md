# Room Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rooms with typed moderator decisions, minutes, one moderator call per round, blind openings, participant continuity, positions with a structured outcome, and four formats, each measured by the orchestration eval.

**Architecture:** Decision parsing and room state move to a new pure module, `features/orchestration/room_decisions.py`, and formats to `features/orchestration/room_formats.py`. `RoomRunner` (rooms.py) runs a loop over decisions from a `SpeakerPolicy`: the LLM moderator or a fixed format. `Runner` (runner.py) gains usage counting and a fresh-thread input for budget retries. Settings go in config; the CLI and the `start_room` tool learn `format`.

**Tech Stack:** Python 3.11+, unittest, the repo's fake `Model` (tests/test_agents.py), `RoomBase` (tests/test_rooms.py).

**Spec:** `design/specs/2026-11-16-room-changes-design.md`

## Global Constraints

- Existing tests pass unchanged, except tests that assert the old moderator text format. `RoomRunner`'s new constructor options default to today's behaviour (`opening_round=False`, `recent_turns=3`, `format="moderated"`); the app passes the settings.
- `tests/test_structure.py` passes with no exceptions added; only `ui/` prints; `config.py` imports nothing from `features/`.
- `tests/test_runner_contract.py` must still pass.
- Settings: `ROOM_RECENT_TURNS` (int ≥ 1, default 3), `ROOM_OPENING_ROUND` (bool, default true), `ROOM_FORMAT` (one of `moderated, roundrobin, debate, redteam`, default `moderated`), each in `config.py`, `.env.example` and the README settings table.
- Thread ids: participants `{thread}:room:{room_id}:{name}` (+ `+` per fresh restart); moderator `{thread}:room:{room_id}:{n}:moderator`; summary `{thread}:room:{room_id}:summary:moderator`.
- Never read or print `.env`. Never run a live eval or a live room in a task (the controller does). No new dependencies.
- Commit messages: one line, no trailer. Test commands: `uv run --no-sync python -m unittest discover -s tests -p "test_rooms.py"` (and the other named files); whole suite `uv run --no-sync python -m unittest discover -s tests`.

## Review Focus

1. A moderator reply with the JSON inside prose or a ```json fence is still parsed as JSON (Task 1 test `test_json_in_fence_or_prose`).
2. A JSON decision naming a participant with other case or Markdown (`"**Critic**"`) maps to the participant; unknown or repeated speakers are dropped (Task 1 test `test_speakers_are_matched_and_deduplicated`).
3. The budget running out in the middle of a round keeps the partial answer and doesn't run the rest of the round (Task 2 test `test_budget_mid_round_wraps_up`).
4. A participant whose run fails continues next time on a fresh thread with the full turn input, not the incremental one (Task 3 test `test_failed_participant_restarts_with_full_input`).
5. A quiet room (eval, one-shot runs) still counts calls and tokens (Task 4 test `test_quiet_room_counts_usage`).

---

### Task 1: Typed moderator decisions (2.1)

**Files:**
- Create: `src/deep_agents_demo/features/orchestration/room_decisions.py`
- Modify: `src/deep_agents_demo/features/orchestration/rooms.py`
- Test: `tests/test_room_decisions.py` (create), `tests/test_rooms.py` (add a class)

**Interfaces (produces):**
- `@dataclass(frozen=True) class Turn: speaker: str; ask: str`
- `@dataclass(frozen=True) class Minutes: agreed: tuple = (); disputed: tuple = (); open: tuple = ()` with `text() -> str` (sections `Agreed:`, `Disputed:`, `Open:` with `- item` lines, empty sections left out; `"(no minutes yet)"` when all are empty).
- `@dataclass(frozen=True) class Decision: action: str` (`"next" | "end"`), `turns: tuple = ()`, `minutes: Minutes | None = None` (None = keep the previous minutes), `summary: str = ""`, `decision: str = ""`, `dissent: tuple = ()`, `actions: tuple = ()`, `source: str = "json"` (`json | text | rotation | format`).
- `parse_decision(text: str, participants: list[str], rotation_next: str) -> Decision`:
  1. JSON: the content of a ```json (or ```) fence if present, else the text from the first `{`; decode with `json.JSONDecoder().raw_decode`. Valid when it is a dict with `action` in next/end. Turns: dicts whose `speaker`, after stripping Markdown `*_\`` and spaces, matches a participant case-insensitively (canonical name kept); `ask` a string (missing → `"What is your view?"`); the first entry per speaker kept. `minutes`: a dict → `Minutes` with the string items of `agreed`/`disputed`/`open`; otherwise None. `summary`, `decision`: strings; `dissent`, `actions`: string items. A `next` with no valid turns is not valid JSON-wise and falls through.
  2. Text: today's `parse_moderator`; `("next", name, prompt)` → `Decision("next", (Turn(name, prompt or "What is your view?"),), source="text")`; `("end", "", summary)` → `Decision("end", summary=summary, source="text")`.
  3. Rotation: `Decision("next", (Turn(rotation_next, "What is your view?"),), source="rotation")`.
- `@dataclass class RoomState`: `question: str`, `participants: list[str]`, `turns_used: int`, `max_turns: int`, `minutes: Minutes`, `positions: dict` (name → `{"position", "confidence"}`; filled in Task 4), `recent: list[str]` (recent transcript entries), `user_said: str`, `rotation_next: str`.
- In rooms.py: `class SpeakerPolicy(Protocol): def pick(self, state: RoomState) -> Decision`; `class ModeratorPolicy` (holds the runner; `pick` builds the moderator input, runs the moderator as today, returns `parse_decision(...)`; a `_RunFailed` returns the rotation decision and adds today's "*(the moderator could not answer: ...)*" line).
- `REPLY_FORMAT` (rooms.py) becomes exactly:
  `Reply with one JSON object and nothing else: {"action": "next" or "end", "turns": [{"speaker": "<participant>", "ask": "<what they should address>"}] (at most one per participant, in speaking order), "minutes": {"agreed": [...], "disputed": [...], "open": [...]}, "summary": "<only when ending: what was agreed, the open disagreements, the next steps>", "decision": "<only when ending: the recommendation in one sentence>", "dissent": ["<who still disagrees and why>"], "actions": ["<next step>"]}. End when they agree, when their positions have stopped changing, when nothing new is being said, or when the user asks.`
- The room loop asks the policy for a decision; in this task it runs **only the first turn** of a `next` decision (rounds come in Task 2). The runner keeps the latest `minutes` (a decision with `minutes=None` keeps the previous ones). An `end` decision's `summary`, `decision`, `dissent`, `actions` are kept for the outcome (added to the outcome in Task 4); an `end` with an empty summary asks for the summary as today.
- Counts of each `source` are kept on the runner (`self.decision_sources: dict[str, int]`), for Task 4's outcome.

- [ ] **Step 1: Tests** in `tests/test_room_decisions.py` (pure): `test_json_decision` (a full next and a full end object → fields, source json); `test_json_in_fence_or_prose` ("Here is my decision:\n```json\n{...}\n```" and "Sure. {...} Thanks." both parse); `test_speakers_are_matched_and_deduplicated` ("**Critic**" → critic; "nobody" dropped; critic listed twice → once); `test_bad_json_falls_back_to_text_then_rotation` ("{not json" + "NEXT: helper\nWhy?" → text; "blah" → rotation with rotation_next); `test_next_without_valid_turns_falls_through`; `test_minutes_text` (empty → "(no minutes yet)"; sections). In `tests/test_rooms.py` class `TypedDecisionTests(RoomBase)`: a room driven by JSON decisions (next critic, then end with summary/decision) → outcome summary, 1 turn; a text-format room still works (existing tests cover it); the moderator's input contains `Reply with one JSON object`.
- [ ] **Step 2: Run** them; expected: import errors / failures.
- [ ] **Step 3: Implement** room_decisions.py and the rooms.py changes.
- [ ] **Step 4: Run** test_room_decisions.py, test_rooms.py, test_runner_contract.py, then the whole suite. Expected: all pass (only tests asserting the old `REPLY_FORMAT` text may change, and the report must name them).
- [ ] **Step 5: Commit** `Rooms: typed moderator decisions (JSON, then text, then rotation) behind a speaker policy`.

---

### Task 2: Minutes and rounds (2.2, 2.3)

**Files:** Modify `rooms.py`, `src/deep_agents_demo/config.py`, `src/deep_agents_demo/ui/cli/repl.py` (pass the setting), `.env.example`, `README.md` (settings table row). Test: `tests/test_rooms.py`, `tests/test_config.py`.

**Interfaces:**
- `RoomRunner(..., recent_turns: int = 3)`; `Settings.room_recent_turns` from `ROOM_RECENT_TURNS` (`at_least_one`, default 3); repl's `runners_for` passes it.
- The transcript keeps `self.lines` as today (saved file unchanged in shape) and a parallel `self.entries: list[str]` of participant and user entries (`"**name:** text"`, `"**You:** text"`); `recent` = the last `recent_turns` entries.
- Participant turn input (replaces `_turn_input`): `Question: {q}\n\nMinutes so far:\n{minutes.text()}\n\nRecent turns:\n{recent joined by blank lines, or "(nothing said yet)"}\n\nThe moderator asks you: {ask}\n\nReply with your contribution only: a few short paragraphs.`
- Moderator input: question, participants with descriptions, `Turns used: n of max.`, `Minutes so far:` block, `Recent turns:` block, the user's words when present, then `REPLY_FORMAT`. `_trimmed()` stays only for the final summary call.
- Rounds: all turns of a `next` decision run in order, then the moderator is asked again. Before each turn after the first: if `pause_requested`, step in (today's `_step_in`); the rest of the round is discarded whatever the answer. `max_turns` cuts a round at the cap.

- [ ] **Step 1: Tests:** `test_round_runs_without_extra_moderator_calls` (one JSON decision with turns critic+helper, then end → exactly 2 moderator runs: decision + nothing else; the fake model's response list proves it); `test_step_in_discards_the_rest_of_the_round` (pause requested during the first turn → helper never runs; the user's words reach the next moderator input); `test_turn_input_has_minutes_and_recent_turns_only` (5 earlier turns, recent_turns=3 → the oldest two absent, minutes text present); `test_budget_mid_round_wraps_up` (budget runs out in the first turn of a two-turn round, wrap up → helper never runs, ended_by budget, partial kept); `test_round_is_cut_at_max_turns`; config test for `ROOM_RECENT_TURNS` default 3, `"0"` refused.
- [ ] **Step 2–4:** RED, implement, GREEN (test_rooms.py, test_config.py, test_runner_contract.py, whole suite).
- [ ] **Step 5: Commit** two commits: `Rooms: minutes and recent turns instead of the last 6,000 characters` and `Rooms: one round per moderator call`.

---

### Task 3: Blind openings and participant continuity (2.4, 2.5)

**Files:** Modify `rooms.py`, `runner.py`, `config.py`, `repl.py`, `.env.example`, `README.md`. Test: `tests/test_rooms.py`, `tests/test_config.py`.

**Interfaces:**
- `RoomRunner(..., opening_round: bool = False)`; `Settings.room_opening_round` from `ROOM_OPENING_ROUND` (`flag`, default true); repl passes it.
- Openings: before the first moderator call, each participant in order gets `Question: {q}\n\nGive your own answer first. The others answer separately; you will see their answers afterwards.\n\nReply with your answer only: a few short paragraphs.` Each is a step (`begin_step("room", n, max_turns, name)`), counts towards `max_turns`, and is logged as `**name (opening):** text` in both `lines` and `entries`.
- Continuity: participant thread `{thread}:room:{room_id}:{name}`. A participant's first run gets the full turn input (Task 2 shape). Later runs get `Since your last turn:\n{entries added since its last turn, or "(nothing new)"}\n\nMinutes so far:\n{minutes}\n\nThe moderator asks you: {ask}\n\nReply with your contribution only: a few short paragraphs.` After a `_RunFailed`, that participant's next run uses a fresh thread (one more `+`) and the full input.
- `Runner._run_agent(name, text, thread_id, capped=True, fresh_text: str | None = None)`: when the budget is extended and the attempt re-runs on `thread_id + "+"`, it sends `fresh_text` if given, else `text` (plans pass nothing). Rooms pass the full turn input as `fresh_text` and remember the new thread suffix for that participant.
- Summarization check (no code unless it fails): confirm in the report, from `create_summarization_middleware` in the installed deepagents, the trigger (tokens or messages) and that a room of `max_turns` short turns stays under it; quote the source lines.

- [ ] **Step 1: Tests:** `test_openings_dont_see_each_other` (opening_round=True: helper's opening input doesn't contain critic's opening; the first moderator input contains both); `test_openings_count_towards_max_turns`; `test_second_turn_gets_only_new_turns` (critic speaks twice; its second input has the helper's turn under "Since your last turn" and not its own first answer; both runs used the same thread id `t:room:<id>:critic`, checked from the saved checkpointer threads or by patching `_invoke` to record thread ids); `test_failed_participant_restarts_with_full_input` (critic's first run fails (a `Boom` graph via a make wrapper), its next run uses `...:critic+` and the full input); `test_budget_retry_uses_the_full_input` (the `fresh_text` path); config test for `ROOM_OPENING_ROUND`.
- [ ] **Step 2–4:** RED, implement, GREEN (test_rooms.py, test_orchestration.py, test_runner_contract.py, test_config.py, whole suite).
- [ ] **Step 5: Commit** `Rooms: a blind opening round` and `Rooms: one thread per participant, new turns only`.

---

### Task 4: Positions, usage and the structured outcome (2.6)

**Files:** Modify `rooms.py`, `room_decisions.py`, `runner.py`, `src/deep_agents_demo/models/outcomes.py`. Test: `tests/test_room_decisions.py`, `tests/test_rooms.py`, `tests/test_orchestration.py`.

**Interfaces:**
- `parse_position(text: str) -> tuple[str, str] | None` in room_decisions.py: the last line matching `POSITION` (case-insensitive, Markdown `*_\`>` and leading `-` stripped) followed by `:`; the position is the text after it, without a trailing `(confidence: X)`; confidence `low|medium|high` lowercased, else `"unknown"`; None when no such line.
- Every participant input (openings, turns, incremental) ends with `\n\nEnd with a line POSITION: <your position in one sentence> (confidence: low|medium|high).`
- `RoomState.positions` is filled; the moderator input gains `Positions:\n- name: position (confidence)` (or `(none yet)`) before `REPLY_FORMAT`.
- Usage: `Runner.usage: dict` with `model_calls`, `input_tokens`, `output_tokens`, reset at the start of `run()`. `_Tally` counts every `TextMessageEnd` (1 call, its tokens) and forwards events to the sink only when the runner isn't quiet; `_invoke` calls `run_turn(..., show=True)` with the tally as the sink, so quiet runs count too (today `show=False` emits nothing). Step tokens keep coming from `RunFinished` as today. Plans get the counting too (their outcome is unchanged).
- `RoomOutcome` gains `positions: NotRequired[dict]`, `decision: NotRequired[str]`, `dissent: NotRequired[list[str]]`, `actions: NotRequired[list[str]]`, `model_calls: NotRequired[int]`, `input_tokens: NotRequired[int]`, `output_tokens: NotRequired[int]`, `decision_sources: NotRequired[dict]`. The room's outcome sets them all (empty values when unknown).
- The outcome is saved as JSON next to the transcript (`<same stem>.json`, UTF-8, `indent=2`) when transcripts are saved; the outcome's `transcript` key keeps the Markdown path.

- [ ] **Step 1: Tests:** `test_parse_position` (plain; `**POSITION:** ship it (confidence: HIGH)`; no confidence → unknown; two lines → the last; none → None); `test_positions_reach_the_moderator_and_the_outcome`; `test_outcome_has_decision_dissent_actions` (from a JSON end); `test_quiet_room_counts_usage` (quiet=True, 3 model responses → model_calls 3); `test_outcome_json_saved_next_to_transcript`; plans unchanged (existing tests).
- [ ] **Step 2–4:** RED, implement, GREEN (test_room_decisions.py, test_rooms.py, test_orchestration.py, test_runner_contract.py, whole suite).
- [ ] **Step 5: Commit** `Rooms: positions, usage and a structured outcome saved as JSON`.

---

### Task 5: Formats (2.7)

**Files:** Create `src/deep_agents_demo/features/orchestration/room_formats.py`. Modify `rooms.py`, `config.py`, `repl.py`, `.env.example`, `README.md` (settings rows + a "Room formats" paragraph in the rooms section). Test: `tests/test_room_formats.py` (create), `tests/test_rooms.py`, `tests/test_config.py`, `tests/test_cli_interrupts.py` if the runner wiring changes.

**Interfaces:**
- `ROOM_FORMATS = ("moderated", "roundrobin", "debate", "redteam")` in `config.py` (config validates `ROOM_FORMAT` against it; `Settings.room_format`). room_formats.py imports it and a test asserts its presets' names equal it.
- `@dataclass(frozen=True) class RoomFormat: name: str; instructions: str; opening_round: bool; policy: str; min_participants: int; max_participants: int | None`. Presets `FORMATS: dict[str, RoomFormat]`: moderated (opening True, policy "moderator", 2..None); roundrobin (False, "roundrobin", 2..None); debate (False, "debate", 3..3); redteam (False, "redteam", 2..None). `instructions` is one sentence added to the participants' input and the summary request describing the format (e.g. debate: "This is a debate: the first participant argues for, the second against, the third judges.").
- `format_problem(name: str, participants: list[str]) -> str | None`: unknown name → `"unknown room format 'x'; use one of moderated, roundrobin, debate, redteam"`; wrong count → `"the debate format needs exactly 3 participants"` / `"needs at least 2"`.
- Fixed policies (SpeakerPolicy implementations, `source="format"`), each returning one round per `pick`:
  - roundrobin: every participant in order, ask `Give your view, building on what was said.`
  - debate (p0 for, p1 against, p2 judge): rounds of [p0 `Argue for: answer the last argument against.`, p1 `Argue against: answer the last argument for.`]; when the turns left are ≤ 1 (or positions unchanged), a final round [p2 `Weigh both sides and give your verdict: which side wins, and why.`]; after the judge, `end` with `decision` = the judge's first line.
  - redteam (p0 proposer, others attackers): first round [p0 `Propose an answer to the question.`, attackers `Find the weaknesses in the proposal.`...]; later rounds [p0 `Answer the attacks: defend or revise your proposal.`, attackers `Attack the revised proposal.`...].
  - All fixed policies `end` (empty summary → the summary call) at `max_turns`, or when every participant has a position and none changed over the last full round.
- RoomRunner: `format: str = "moderated"` constructor default; the request may carry `"format"` (overrides); `format_problem` is checked with the participant check, before any model call; the opening round runs when the runner's `opening_round` and the format's `opening_round` are both true; the policy comes from the format.
- `start_room(question, participants, format: str = "")`: when given, the interrupt request carries `"format"`; the docstring lists the four formats in one line; `ROOM_PROMPT` mentions formats in one clause.
- CLI: `parse_room_command(arg) -> tuple[list[str], str, str | None] | str` accepts `--format NAME` anywhere before `--`; `ROOM_USAGE` becomes `Usage: /room AGENT AGENT [...] [--format moderated|roundrobin|debate|redteam] -- QUESTION`; `/room` passes the format; `ChatSession.room(names, question, format)`; repl's `run_room` sends it in the request. The `/` menu usage text follows the command's `usage`.
- `runners_for` passes `format=settings.room_format`.

- [ ] **Step 1: Tests** (`tests/test_room_formats.py` and `tests/test_rooms.py`): presets match `ROOM_FORMATS`; `format_problem` cases; `test_roundrobin_order_and_no_moderator_calls` (fake model: only participant answers then the summary; assert the moderator ran once, for the summary); `test_debate_order_and_judge_last` (decision = judge's first line); `test_redteam_order`; `test_fixed_format_ends_when_positions_settle`; `test_unknown_format_costs_no_model_call` (outcome error, zero model calls); `test_parse_room_command_format`; `test_start_room_passes_format` (StartRoomToolTests style); config `ROOM_FORMAT` default and an invalid value refused.
- [ ] **Step 2–4:** RED, implement, GREEN (the files above, test_runner_contract.py, test_cli_interrupts.py, test_slash_menu.py, whole suite).
- [ ] **Step 5: Commit** `Rooms: formats (moderated, roundrobin, debate, redteam), --format and start_room(format)`.

---

### Task 6: Measurements and README (controller, live)

- [ ] After Task 2: `uv run --no-sync python -u evals/orchestration.py --modes room` with a log; rescore if scorers changed; keep the JSON.
- [ ] After Task 3: the same.
- [ ] After Task 5: the same, plus one live `/room` check of `--format roundrobin` (no moderator call before the summary) via a one-shot script or the eval harness.
- [ ] The README orchestration section's table gains `room (after 2.2)`, `room (after 2.4)`, `room (after 2.7)` rows; the paragraph is updated from the numbers, including where a change made things worse; the number of moderator calls for a 12-turn moderated room before and after (from a fake-model test: before 12 + summary, after with 2 openings and rounds of 2: 5 + summary) is stated.
- [ ] Commit `README: rooms after typed decisions, minutes, rounds, openings and formats`.

# Rooms, piece 2: typed decisions, minutes, rounds, openings, continuity, positions, formats

Piece 2 of three (eval, room changes, swarm). Written against `ce3a5b6`
(branch `orchestration-eval`). The baseline it is measured against is in the
README, "Evaluating orchestration".

## Why

The baseline shows today's room costs about three times the calls of a single
agent for little gain: the moderator ends after one or two turns, a third of
the summaries ignore the requested answer format, and every participant turn
starts from scratch on a fresh thread with the last 6,000 characters of the
discussion. These changes make each call worth more (typed decisions,
minutes, one moderator call per round, participants that remember their own
work), make disagreement visible (blind openings, positions), and add formats
for the discussions people actually want (debate, red team).

## The seven changes

Each lands as its own commit with its tests. The eval's `room` mode is rerun
after 2.2, 2.4 and 2.7 (`--modes room`; `single` keeps its baseline numbers),
and the README table gains a row per rerun.

### 2.1 Typed moderator decisions

The moderator is asked for one JSON object:

    {"action": "next" | "end",
     "turns": [{"speaker": "<participant>", "ask": "<what to address>"}],
     "minutes": {"agreed": [...], "disputed": [...], "open": [...]},
     "summary": "<only when action is end>",
     "decision": "<only when action is end: the recommendation, one sentence>",
     "dissent": ["<who still disagrees and why>"],
     "actions": ["<next step>"]}

*My call:* `decision`, `dissent` and `actions` live in the end decision, so 2.6's
structured outcome comes from the moderator in the same call that writes the
summary.

Parsing: the first JSON object in the reply (bare, or in a ```json fence) is
parsed and validated: `action` is next or end; turns name participants (turns
naming anyone else are dropped); lists are lists of strings. If that fails, the
existing `NEXT:`/`END` parser runs; if that fails too, rotation (both exist
today and stay). The parse result is a `Decision` dataclass either way, with
`source` = `json`, `text` or `rotation`, so the eval and tests can count how often
each path is used.

Speaker selection sits behind a small protocol, `SpeakerPolicy.pick(state) ->
Decision`. `ModeratorPolicy` (the LLM moderator) is one implementation; the
formats in 2.7 add fixed-order ones. A decision model can replace the
moderator later without touching the room loop.

### 2.2 Minutes instead of truncation

Participants no longer get `_trimmed()` (the last 6,000 characters). Each turn's
input is: the question, the current minutes (from the moderator's last
decision; empty before the first), and the last `ROOM_RECENT_TURNS` turns
verbatim (default 3). The moderator gets the same plus the latest positions
(2.6). The saved transcript still has everything.

### 2.3 One round per moderator call

A decision's `turns` may list up to one turn per participant. The runner runs
them in order, then asks the moderator again. A participant listed twice keeps
its first entry. A Ctrl+C step-in takes effect between turns and discards the
rest of the round. `ROOM_MAX_TURNS` still caps participant turns; a round
that would pass it is cut at the cap.

### 2.4 Blind opening round

Before the first moderator call, every participant answers the question
without seeing the others' answers. The openings then go into the transcript
and the moderated discussion starts from them. Sequential in this piece (piece 3
makes it parallel). `ROOM_OPENING_ROUND` (default true). *My call:* opening
turns count towards `ROOM_MAX_TURNS`, like any participant turn.

### 2.5 Participant continuity

One thread per participant per room: `{thread}:room:{id}:{name}`. A
participant's first turn gets the full turn input (2.2). Its later turns get only
what was said since its last turn, plus the current minutes and the moderator's
ask, so it keeps its own earlier reasoning and tool results. After a failed run
or a budget retry it continues on a fresh thread (the existing `+` suffix) with
the full turn input. The moderator keeps a fresh thread per call (its input is
already the full state).

*To verify before building:* long participant threads go through the worker
stack's summarization middleware (`create_summarization_middleware` in
`worker_factory`). A room is at most `ROOM_MAX_TURNS` turns, so threads stay
short; the plan checks with a test that a summarized thread still answers.

### 2.6 Positions and a structured outcome

Each participant is asked to end its turn with
`POSITION: <one sentence> (confidence: low|medium|high)`. The line is parsed
leniently (case, Markdown, missing confidence → `unknown`), and the latest one
per participant is kept. The moderator sees the positions and is told to end
when they have stopped changing.

`RoomOutcome` gains `positions` (`{name: {"position", "confidence"}}`), `decision`,
`dissent`, `actions`, `model_calls`, `input_tokens`, `output_tokens`. Calls and tokens
cover every run in the room: openings, turns, moderator and summary. The
outcome is saved as JSON next to the Markdown transcript (same name, `.json`).

### 2.7 Formats

`RoomFormat` (name, moderator instructions, opening round or not, speaker
policy) with four presets:

| Format | Speakers | Moderator calls |
|---|---|---|
| `moderated` (default) | the moderator picks each round (2.1–2.6) | one per round, plus the summary |
| `roundrobin` | every participant in the given order, round after round | none until the summary |
| `debate` | exactly 3 participants: the first argues for, the second against, alternating; the third judges and gives the verdict last | none; the judge's verdict is the decision, then the summary |
| `redteam` | the first proposes; the others attack in order; the proposer answers; repeat | none until the summary |

The fixed formats end at `ROOM_MAX_TURNS`, or earlier when every participant's
position is unchanged over a full round. `ROOM_FORMAT` (default `moderated`) sets
the default. On the CLI: `/room critic tester --format debate -- QUESTION`.
`start_room` gets an optional `format` argument. An unknown format, or the wrong
number of participants for a format, is a clear error before any model call.

## Settings

    ROOM_RECENT_TURNS=3
    ROOM_OPENING_ROUND=true
    ROOM_FORMAT=moderated

In `config.py`, `.env.example` and the README settings table.

## What doesn't change

`ROOM_MAX_TURNS`, the request budget and its "continue or wrap up" question,
Ctrl+C (first press: your turn between turns; second: stop), approvals, the
transcript folder, `room_followup`, and `tests/test_runner_contract.py` (it must
still pass). `tests/test_structure.py` keeps passing with no exceptions; only `ui/`
prints. Existing tests change only where they assert the old moderator text
format.

## Tests

`tests/test_rooms.py`, with the fake model:
- JSON decision parsed; bad JSON falls back to the text parser, then to rotation;
  `source` is recorded.
- A scheduled round runs without extra moderator calls; a step-in discards the
  rest of the round.
- Opening answers don't see each other.
- A participant's second turn receives only the new turns (plus minutes and the
  ask); after a failed run it gets the full input on a fresh thread.
- Positions parsed (lenient forms) and reported in the outcome.
- Each format's speaker order, the debate judge last, the participant-count
  errors before any model call, an unknown format's error.
- The outcome JSON is saved next to the transcript, with calls and tokens.
- Budget and Ctrl+C behaviour unchanged.

## Acceptance

- A 12-turn moderated room uses clearly fewer model calls than before; the
  number goes in the README.
- `/room a b --format roundrobin -- Q` runs with no moderator call before the
  summary.
- The eval's room rows after 2.2, 2.4 and 2.7 are in the README table, including
  any that get worse.
- All existing tests pass, except those that asserted the old moderator text.

## Out of scope

Parallel openings and swarms (piece 3), resuming a saved room, a decision model
as moderator (the seam is enough), GUI work.

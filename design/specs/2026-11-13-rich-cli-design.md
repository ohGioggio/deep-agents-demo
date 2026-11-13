# A Rich terminal UI: streamed answers, a timeline of tool calls, and a / command menu

Status: agreed in chat on 2026-11-13; not built. Written against commit `ac14db7`.
Folds in the streaming part of the paused "event stream + run store" brainstorm
(2026-11-11); the run store stays a later, separate piece.

## Intent

The chat works, but it reads like a log: every message is a `=== Ai Message ===`
block with raw Markdown, tool calls and results are long boxed dumps, and with the
NVIDIA model nothing at all appears for 24–32 seconds while it thinks. The user
wants the interactive CLI to look and feel like a good modern agent CLI: answers
rendered and streamed as they are written, the work (tool calls, subagents, plan
steps, room turns) shown as a compact live timeline, and clear prompts and tables,
using the `rich` library. The engine already emits events to a renderer
(`ui/cli/render.py`), so this is a new front end plus finer-grained events, not a
rewrite.

## Decided by the user

- **All of these**, in the interactive chat:
  1. Answers rendered as Markdown (headings, lists, tables, highlighted code).
  2. A live spinner with elapsed time while the model is silent ("thinking… 23 s"),
     and while a tool runs.
  3. Answers stay in a box per speaker (the "option A" look chosen on 2026-11-11),
     coloured by speaker: main, subagent, `plan / …`, `room / …`.
  4. Tool calls with highlighted arguments; long results cut short.
  5. The approval prompt in a panel, the diff coloured, the keys highlighted.
  6. `/threads`, `/files`, `/agents` as aligned tables.
  7. `[file]` lines as clickable links to the saved file.
  8. The to-do list as a checklist.
  9. A footer under every answer (main, subagent, plan step, room turn): end time ·
     tokens (in / out) · duration.
- **Tool calls and steps use the timeline layout ("option B" in the mockup):**
  one line per tool call in a tree under its speaker: tool name, short arguments,
  ✓ or ✗ with a short result ("48 lines", "3 results", "v1 · 3.1 KB"), duration.
  A subagent's calls nest under its `task → <agent>` line, ending with its short
  answer; each plan step is a branch ("step 1/2 · general-purpose · ✓ passed
  review · 2 tries · 41 s") with its calls under it; room turns likewise. Whatever
  is running shows a spinner with elapsed time ("⠹ reviewing… 12 s"), then settles
  to ✓ / ✗. An approval shows "⏸ needs your approval" on its line, with the panel
  and keys right below; a failure shows ✗ and the reason. The main answer keeps
  its box with Markdown inside, footer in the box's bottom edge. `[file]` sits
  right under its `send_file` line.
  Mockup: `.superpowers/brainstorm/778-1794604604/content/tools-and-steps.html`.
- **Which answers get a box (agreed after the builder check):**
  - the **main answer**: a box, streamed word by word, Markdown inside, footer in
    its bottom edge;
  - **room turns**: each participant's turn in its own box, streamed (turns come
    one at a time, and the discussion is the room's whole output);
  - **subagents and plan steps**: one line in the tree with their short answer and
    their own footer as dim text (several subagents can run at once, so they are
    not streamed into boxes);
  - **text the agent writes alongside tool calls** ("Let me check the skill…"):
    one dim line in the tree, so nothing is hidden.
- **Streaming:** answers appear word by word inside their box, in the terminal.
- **A `/` command menu, like Codex's** (mockup:
  `.superpowers/brainstorm/778-1794604604/content/slash-menu-v5.html`):
  - the chat input becomes a box at the bottom (`parks › …`, replacing
    `[parks] You:`); typing `/` opens a menu in its own rounded frame **between the
    last line of output and the input box**, never over the output;
  - **at most 5 rows**; with more matches the frame keeps its size and ↑↓ scroll
    inside it; the bottom edge shows the position and keys (`↑ 7/11 ↓ · Tab ·
    Enter · Esc`); with fewer matches the frame shrinks to fit;
  - it is a **filter**: only commands containing what you typed stay (`/n` keeps
    `/new`, `/agents`, `/rename`), names starting with it listed first;
  - the **first match is selected**, and the rest of its name is shown **in grey in
    the input box**; Tab or → accepts it; moving the selection changes the grey text;
  - **Enter works as in Codex**: it runs the selected match (`/n` + Enter runs
    `/new`);
  - arguments complete the same way, with a frame title saying what is completed:
    "your chats" for `/switch`, `/delete`, `/rename`; "agents" for `/room`;
    "images" (PNG/JPG) for `/image`;
  - ↑ in an empty box recalls earlier messages (history per user, kept across
    restarts);
  - the menu disappears when you pick, send or press Esc; only your sent line stays.
- **Resizing:** the menu, the input box and the live region (timeline, streaming
  answer, approval panel) always fit the current terminal width and redraw when it
  changes; narrow terminals cut descriptions with "…" first, then drop the
  description column.
- **Events modelled on AG-UI** (from the earlier paste the user brought: "model the
  event names on AG-UI so the web endpoint later is a thin serializer").
- **Run store** (`--runs`, `--show-run`, SQLite) is out of this piece.

## Calls made by Claude (check these)

- `--verbose` prints each tool call's full JSON arguments and full result under its
  timeline line, instead of the one-line summary.
- Rich is used only when stdout is a terminal, the run is interactive (not
  `--prompt`), and `--plain` isn't given. Everything else (one-shot runs, pipes,
  redirected output, the test suite) keeps today's plain text, byte for byte.
- `NO_COLOR` is honoured (Rich does this): layout stays, colour goes.
- Palette follows the architecture diagram: purple for agents, cyan for tools,
  green for files and success, amber for approvals, red for failures, dim grey for
  timings and secondary text.
- The plain renderer keeps receiving today's events (`MessageEvent`, `TodosEvent`,
  `NoticeEvent`, `FileEvent`, `FinalAnswerEvent`) so its output doesn't change; the
  new fine-grained events are additional. Each renderer ignores what it doesn't use.
- Rich renders through one `rich.console.Console`; what is still running (the
  current timeline branch, the streaming answer) lives in a `rich.live.Live` region.
  When a turn outgrows the screen, **finished branches are printed as they finish**
  and only the running part updates in place: a plan step gets an opening line and a
  closing status line ("step 1/2 · general-purpose · ✓ passed review · 2 tries ·
  41 s"). Output already printed is not redrawn on resize (the terminal re-wraps it).
- **Footers count the whole run, nested work included:** end time · tokens in / out
  (summed over every model call of that run, including its subagents and plan
  steps) · total duration. The main answer's footer is the whole turn; a subagent's
  or step's line shows its own totals the same way. The engine does the sums
  (`RunFinished` carries them).
- **Ctrl+C in an ordinary turn** keeps today's meaning (it ends the chat), but the
  Live region is stopped cleanly first, with no half-drawn boxes. Making Ctrl+C cancel
  just the turn would be engine work and is not in this piece. Ctrl+C during plans
  and rooms (pause, then stop) is unchanged.
- **The `/` menu needs `prompt_toolkit`** (Rich only draws output). It is used for
  the chat input only when stdin is a terminal; piped input (`--prompt`, scripts,
  live checks) keeps plain `input()`. Approval, choice and pause prompts keep asking
  one question at a time (through the Rich `Interaction`), with the Live region
  stopped while they wait.
- **Commands that need an argument** (`/rename NAME`, `/switch NAME`, `/room A B --
  QUESTION`, `/image PATH`) don't run on Enter: Enter fills in the command and waits
  for the argument, with the usage as a grey hint. In an argument menu, Enter picks
  the value and runs (`/switch ro` + Enter runs `/switch roads`).
- The menu is built from the existing `@command` registry (name, usage, summary), so
  new commands appear in it automatically.

## Goals, non-goals, anti-goals

**Goals**
- The nine items and the timeline above, in the interactive CLI.
- Streamed text with real per-answer token counts (the model reports them at the
  end of each answer: verified 2026-11-11, incl. hidden reasoning tokens).
- An event stream a web or GUI front end can later serialise as AG-UI with little
  more than renaming fields.

**Non-goals**
- The run store, cost tracking, a GUI or a web endpoint.
- Changing what the agent does: tools, prompts, approvals logic, plans and rooms
  behave as today.
- Showing the model's hidden reasoning text (the NVIDIA model doesn't send it; if a
  model does, it may be shown dim, but nothing is built for it).

**Anti-goals (would make this a failure even if it works)**
- Garbled or duplicated output: a Live region that keeps redrawing over an input
  prompt, flicker on every token, half-drawn boxes after Ctrl+C.
- Breaking pipes or one-shot runs: `--prompt` output must stay clean text.
- Hiding what's happening: an approval must be impossible to miss; an error must
  not be swallowed by the timeline.
- Making tests depend on terminal width or colour codes.

## How it fits the current code

- **Engine (core/turns.py).** `stream_once` streams `stream_mode="updates"` today.
  Add `"messages"` (token chunks) to get text as it is written:
  `agent.stream(..., stream_mode=["updates", "messages"], subgraphs=True)`. From
  that, emit fine-grained events alongside today's ones:
  - `RunStarted` / `RunFinished` / `RunError` for each agent run (the main turn,
    a subagent run, a plan step's worker or reviewer, a room turn), with a run id,
    the speaker label, start/end times and token usage;
  - `TextMessageStart` / `TextMessageContent(delta)` / `TextMessageEnd` per answer,
    with the answer's usage (input/output tokens) and duration on the end event;
  - `ToolCallStart` (id, name, args) when a tool call is decided, `ToolCallEnd`
    with status (ok / error / rejected), a short result summary and duration when
    its result arrives (full result kept for `--verbose`);
  - `SubagentStarted` / `SubagentFinished` when a `task` call's subgraph starts
    and ends (the stream's namespace tells them apart);
  - `StateSnapshot` for the to-do list.
  Names follow AG-UI (see the recon notes below); field names in `to_dict()` may
  stay snake_case for now.
- **Plans and rooms (features/orchestration).** Runners emit `StepStarted` /
  `StepFinished` (step number, agent, status, tries, duration; a room turn the same
  way) in addition to today's `NoticeEvent` lines, which the plain renderer keeps
  printing.
- **Approvals.** When a paused call is about to be asked, the engine emits an
  `ApprovalRequested` event (an AG-UI `CUSTOM` in spirit) carrying the tool call id,
  so the timeline marks the right line "⏸ needs your approval"; the existing
  `Interaction.decide` still asks. The Rich `Interaction` draws the panel, coloured
  diff and keys; it must stop or pause the Live region before reading input and
  resume it after.
- **Front end (ui/cli/).** A `RichRenderer` (an `EventSink`) and a `RichInteraction`
  (an `Interaction`), chosen in `repl.py` when Rich applies; today's `CliRenderer`
  and `CliInteraction` otherwise. Tables (`/threads`, `/files`, `/agents`) get Rich
  versions next to today's string formatters. Ctrl+C handling (`interrupts.py`)
  stays; its pause message must not corrupt the Live region.
- **Input (ui/cli/).** A `prompt_toolkit` application for the chat line: the input
  box, the framed menu above it (a filtering completer over the command registry,
  chat names, agent names, image files), grey inline completion of the selected
  match, history in `data/`. `repl.py` asks it for the next line instead of
  `input()` when stdin is a terminal.
- **Dependencies.** `rich` and `prompt_toolkit` added to `pyproject.toml`. Only `ui/`
  may import them (the layering test already enforces imports by layer).

## AG-UI recon (2026-11-11, docs.ag-ui.com, no version shown)

Standard events: `RUN_STARTED` / `RUN_FINISHED` / `RUN_ERROR`, `STEP_STARTED` /
`STEP_FINISHED`, `TEXT_MESSAGE_START` / `_CONTENT` (`delta`) / `_END`,
`TOOL_CALL_START` / `_ARGS` / `_END` / `_RESULT`, `STATE_SNAPSHOT` / `STATE_DELTA`,
`CUSTOM`, `REASONING_*`, `SUBAGENT_STARTED` / `_FINISHED` / `_ERROR`; fields are
camelCase on the wire. `RUN_FINISHED` / `RUN_ERROR` may carry usage. Approvals are
a draft: a run ends with `outcome: {type: "interrupt", interrupts: [...]}` and
resumes as a new run; not needed for the terminal.

## Testing

- Today's tests keep passing unchanged: the plain renderer's output is byte for byte
  the same.
- Event sequence tests with the fake model: a scripted turn (a tool call, a
  subagent, an answer) produces the expected order of new events; usage and
  durations are present on end events; a failed tool gives `ToolCallEnd(status=
  "error")`; a rejected approval gives `status="rejected"`.
- Rich rendering tests through `Console(record=True, width=100, color_system=None)`
  and `export_text()`: the timeline lines, the answer box with its footer, the
  approval panel, the tables. Assert on text, not escape codes.
- `/` menu tests with `prompt_toolkit`'s pipe input and a dummy output: typing `/`,
  `/n`, `/re`, ↓ past five, Tab, Enter on a command with and without an argument,
  argument completion for `/switch`, history, Esc; the rendered menu checked for 5
  rows and the filter. Resize handled by rendering at two widths.
- Live checks with the real model: a turn with tools, a plan, a room, an approval,
  a delivered file, Ctrl+C during a plan, `--prompt` still plain, piping to a file
  still plain, `NO_COLOR=1`. The `/` menu and resizing can't be checked by piping
  input: the final check is the user's, typing in a real terminal.

## Left to the builder

- Exact colours, glyphs, box styles, truncation lengths and the "short result"
  wording per tool.
- How the Live region and Markdown rendering are combined while text streams
  (re-render Markdown on each delta, or stream plain text and re-render at the end).
- Throttling redraws (Rich's `refresh_per_second`) so streaming doesn't flicker.
- Event class and field names beyond the AG-UI names above.
- File layout inside `ui/cli/` for the Rich pieces.

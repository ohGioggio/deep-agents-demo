# Rich terminal UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The interactive chat gets a Rich front end (streamed Markdown answers in boxes, a live timeline of tool calls and steps, footers, approval panel, tables) and a Codex-style `/` menu over a boxed input, while one-shot runs, pipes and tests keep today's plain text.

**Architecture:** The engine (`core/turns.py`, the runners) emits new fine-grained, AG-UI-modelled events in addition to today's events; the plain `CliRenderer` ignores them, so its output is unchanged. A new `ui/cli/richui/` package holds a `RichRenderer` (an `EventSink`), a `RichInteraction` (an `Interaction`), Rich tables, and a `prompt_toolkit` input box with the `/` menu. `repl.py` picks Rich or plain once per chat.

**Tech Stack:** Python 3.11+, LangGraph streaming (`stream_mode=["updates", "messages"]`, `subgraphs=True`), `rich`, `prompt_toolkit`, unittest.

**Spec:** `design/specs/2026-11-13-rich-cli-design.md` (commit `c2f4da4`). Mockups: `.superpowers/brainstorm/778-1794604604/content/tools-and-steps.html` (option B), `slash-menu-v5.html`.

## Global Constraints

- Plain output is byte for byte unchanged for `--prompt`, piped/redirected stdout, `--plain`, and the test suite (existing exact-string tests must pass untouched).
- Rich + `prompt_toolkit` are used only when: stdout and stdin are terminals, the run is interactive (no `--prompt`), and `--plain` isn't given. `NO_COLOR` is honoured (layout stays, no colour).
- Only `ui/` imports `rich` or `prompt_toolkit` (the layering test enforces it; don't add to `LEGACY`/`IO_PENDING`).
- Palette: purple agents (`#a78bfa`), cyan tools (`#22d3ee`), green files/success (`#7ee787`), amber approvals (`#fbbf24`), red failures (`#ff7b72`), dim grey timings/secondary (`#7d8590`), room/plan cyan-blue (`#38bdf8`), subagent pink (`#f0abfc`).
- Footer text format: `HH:MM · {in} in / {out} out · {duration}` (tokens summed over the run incl. nested runs; each count as `9.8k` from 1000 up, else the integer; duration `74 s` under 100 s, `2 min 5 s` from 100 s).
- Menu: max 5 rows; filter = contains (prefix matches first); first match selected; rest of its name grey in the box; Tab/→ accept; Enter runs the selection (commands needing an argument fill in and wait); Esc closes; frame titles `commands`, `your chats`, `agents`, `images`; bottom edge `{pos}/{total}` with `↑`/`↓` when scrollable, then `· Tab · Enter · Esc`.
- Commands needing an argument: `/rename`, `/switch`, `/delete`, `/room`, `/image`.
- Ctrl+C in an ordinary turn still ends the chat (Live stopped cleanly first); plans/rooms keep pause-then-stop.
- Every commit: `uv run python -m unittest discover -s tests` green; one-line commit messages, no trailer; push only when asked.
- Live checks use user `tools-check` and clean up after themselves.

## Review Focus

1. **Text that arrives with tool calls in the same message** (the model writes "Let me check…" and calls a tool) — expected: one dim line in the tree, not lost and not a box. (Task 5 test `test_text_beside_tool_calls_is_a_dim_line`.)
2. **Two subagents running at once** (the `task` cap runs them in parallel threads) — expected: two branches, each with its own calls and result, no interleaved lines inside a branch. (Task 2 test `test_parallel_subagents_have_separate_runs`; Task 5 test `test_two_branches_render_separately`.)
3. **An approval while the Live region is drawing** — expected: Live stops, the panel and prompt print cleanly, the answer is read, Live resumes; the timeline line shows ⏸ then ✓/✗. (Task 6 test `test_live_is_paused_while_asking`.)
4. **A terminal narrower than the frames** (e.g. 50 columns) — expected: menu descriptions cut with "…", then dropped; boxes and tree lines wrap without breaking borders. (Task 8 test `test_narrow_terminal_drops_descriptions`; Task 4 test `test_renders_at_narrow_width`.)
5. **Output redirected while interactive** (`deep-agents-demo > log.txt`, stdin a terminal) — expected: plain text, no escape codes, no `prompt_toolkit` (it needs a terminal for output). (Task 9 test `test_mode_is_plain_when_stdout_is_not_a_terminal`.)

---

### Task 1: The new events

**Files:**
- Modify: `src/deep_agents_demo/models/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces (frozen dataclasses, `to_dict()` with `"kind"` like today's events; `run_id` strings; times are `time.monotonic()` seconds for durations and `time.time()` for `ended_at`):
  - `RunStarted(run_id, label, parent_id=None)` — a main turn, a subagent run, a plan worker/reviewer run, a room turn.
  - `RunFinished(run_id, label, input_tokens, output_tokens, duration, ended_at)` — totals include nested runs.
  - `RunError(run_id, label, message)`.
  - `TextMessageStart(run_id, message_id, label)`, `TextMessageContent(run_id, message_id, delta)`, `TextMessageEnd(run_id, message_id, text, input_tokens, output_tokens, has_tool_calls)`.
  - `ToolCallStart(run_id, tool_call_id, name, args: dict)`, `ToolCallEnd(run_id, tool_call_id, name, status, summary, duration, result="")` — `status` in `"ok" | "error" | "rejected"`; `result` is the full text (for `--verbose`).
  - `SubagentStarted(run_id, parent_run_id, name, tool_call_id)`, `SubagentFinished(run_id, status)`.
  - `StepStarted(run_id, kind, number, total, agent)` — `kind` `"plan"` or `"room"` (room: `number`=turn, `total`=max turns).
  - `StepFinished(run_id, kind, number, status, attempts, duration, input_tokens, output_tokens, summary="")`.
  - `StateSnapshot(todos: tuple)`.
  - `ApprovalRequested(run_id, tool_call_id, name)`.
- [ ] **Step 1: Write the failing test** `tests/test_events.py::test_new_events_are_plain_data`: build one of each with sample values; `json.loads(json.dumps(e.to_dict()))["kind"] == type(e).__name__`; and `test_footer_format` for a helper `format_footer(ended_at, input_tokens, output_tokens, duration) -> str` (in `models/events.py`): `(1763040060.0, 9800, 450, 74.2)` → `"HH:MM · 9.8k in / 450 out · 74 s"` where HH:MM is the local time of `ended_at`; `(…, 125.0)` → `"… · 2 min 5 s"` (whole seconds under 100 s, minutes and seconds from 100 s).
- [ ] **Step 2: Run** `uv run python -m unittest tests.test_events`. Expected: ImportError.
- [ ] **Step 3: Implement** the dataclasses and `format_footer`.
- [ ] **Step 4: Run** the test and the full suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Events: AG-UI-modelled run, text, tool call, subagent, step and approval events"`

---

### Task 2: The engine streams and emits the new events

**Files:**
- Modify: `src/deep_agents_demo/core/turns.py`
- Create: `tests/streaming_model.py`
- Test: `tests/test_turn_events.py`, existing `tests/test_render.py` (must pass unchanged)

**Interfaces:**
- Consumes: Task 1 events.
- Produces:
  - `stream_once` uses `agent.stream(payload, config=config, stream_mode=["updates", "messages"], subgraphs=True)`; items are `(namespace, mode, chunk)`. `"messages"` chunks are `(message_chunk, metadata)`: emit `TextMessageStart` on the first non-empty text chunk of a message id, `TextMessageContent` per chunk. `"updates"` keep producing today's `MessageEvent`/`TodosEvent`/`FileEvent` exactly as now, plus: `TextMessageEnd` (text, usage from the finished message's `usage_metadata`, `has_tool_calls`), `ToolCallStart` per tool call of a finished AI message, `ToolCallEnd` when its `ToolMessage` arrives (status `error` if the tool message has `status == "error"` or starts with `"Error"`; `rejected` if it came from a rejected approval; `summary` from `summarize_result(name, content) -> str`, see below), `StateSnapshot` when `todos` change.
  - Runs: `RunStarted`/`RunFinished` around the whole turn (`run_turn`); a new namespace starting with `"tools:"` is a subagent run: `SubagentStarted(run_id=<namespace>, parent_run_id, name=<subagent_type of the earliest unmatched pending task call>, tool_call_id=<that call's id>)`, and `SubagentFinished` when that task's `ToolMessage` arrives. Token totals add up per run and into the parent.
  - `summarize_result(name: str, content: str) -> str` in `core/turns.py`: first line cut to 60 chars, or for known shapes: `read_file` → `"{n} lines"`, `ls`/`glob` → `"{n} entries"`, `search_assets`/`search_docs` JSON lists → `"{n} results"`, `send_file` JSON → `"v{version} · {size}"`, errors → the first line.
  - A message chunk with no text and no tool-call chunks emits nothing; `run_turn` emits `RunError` if the stream raises, then re-raises.
- [ ] **Step 1: Write `tests/streaming_model.py`:** `StreamingModel(FakeMessagesListChatModel)` whose `_stream` yields the next response's text as word chunks (`AIMessageChunk`, same id), then a last empty chunk with `usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}`; tool calls go out in one chunk with `tool_call_chunks`; `bind_tools` records to `SEEN` like `test_agents.Model`.
- [ ] **Step 2: Write the failing tests** `tests/test_turn_events.py` (a `RecordingSink` collecting events):
  - `test_streamed_answer_arrives_in_pieces`: answer "The answer is here." → one `TextMessageStart`, ≥3 `TextMessageContent` whose deltas join to the text, one `TextMessageEnd` with `input_tokens == 100`.
  - `test_tool_call_start_and_end`: `ls` call then answer → `ToolCallStart(name="ls", args={"path": "/"})`, `ToolCallEnd(status="ok", summary="0 entries" or the tool's text)`, `duration >= 0`.
  - `test_failed_tool_is_an_error`: a call to an unknown path that errors → `ToolCallEnd.status == "error"`.
  - `test_subagent_run`: a `task` to general-purpose → `SubagentStarted(name="general-purpose", tool_call_id=<the task id>)` then its own text events with that run_id, then `SubagentFinished`.
  - `test_parallel_subagents_have_separate_runs`: two `task` calls in one AI message → two `SubagentStarted` with different run ids, each matched to a different tool_call_id.
  - `test_run_totals_include_nested_runs`: main + subagent → `RunFinished` of the main run has `input_tokens == 200` (100 each).
  - `test_plain_output_unchanged`: the existing `test_render.EXPECTED` still renders byte for byte (run that test file).
- [ ] **Step 3: Run.** Expected: failures.
- [ ] **Step 4: Implement** in `core/turns.py` (no other module changes).
- [ ] **Step 5: Run** the tests and the full suite. Expected: OK.
- [ ] **Step 6: Commit** `git commit -m "Engine: stream text and emit run, tool call and subagent events (plain output unchanged)"`

---

### Task 3: Steps and approvals in the event stream

**Files:**
- Modify: `src/deep_agents_demo/features/orchestration/plans.py`, `features/orchestration/rooms.py`, `core/turns.py`
- Test: `tests/test_turn_events.py` (more), existing orchestration/room tests unchanged

**Interfaces:**
- Consumes: Task 1, Task 2.
- Produces:
  - `PlanRunner` emits `StepStarted(kind="plan", number, total, agent)` before a step's first attempt and `StepFinished(kind="plan", status in "passed"|"failed"|"skipped"|"stopped", attempts, duration, tokens, summary=<first line of the result>)` when the step is settled; `RoomRunner` emits `StepStarted(kind="room", number=turn, total=max_turns, agent=<speaker>)` / `StepFinished(kind="room", status="done"|"cut"|"failed")` per participant turn. Today's `NoticeEvent` lines stay.
  - `run_turn`: before calling `decide(value)` for an approval pause, emit `ApprovalRequested(run_id, tool_call_id, name)` for each action request (match the pending `ToolCallStart` by name and args; `tool_call_id=""` if none matches); a rejected decision marks that tool call's `ToolCallEnd.status = "rejected"`.
- [ ] **Step 1: Write the failing tests:** `test_plan_steps_are_events` (a 2-step plan with the runner test fixtures → `StepStarted` ×2 and `StepFinished(status="passed", attempts=1)` ×2, in order); `test_room_turns_are_events`; `test_approval_is_announced_then_rejected` (a gated tool, `decide` rejecting → `ApprovalRequested(name=...)` then `ToolCallEnd(status="rejected")`).
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and the full suite (all plan/room tests unchanged). Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Engine: plan steps, room turns and approvals as events"`

---

### Task 4: The Rich renderer — answer boxes, Markdown, streaming, footers

**Files:**
- Modify: `pyproject.toml` (`rich>=14`), `uv.lock`
- Create: `src/deep_agents_demo/ui/cli/richui/__init__.py`, `richui/theme.py`, `richui/renderer.py`
- Test: `tests/test_rich_renderer.py`

**Interfaces:**
- Consumes: Tasks 1–3 events.
- Produces:
  - `richui/theme.py`: `THEME: rich.theme.Theme` with styles `agent`, `subagent`, `step`, `tool`, `ok`, `error`, `approval`, `dim`, `file` (palette in Global Constraints); `make_console(file=None, width=None, force_terminal=None) -> Console` (honours `NO_COLOR` via Rich).
  - `richui/renderer.py`: `class RichRenderer(EventSink)`: `__init__(self, console: Console, verbose: bool = False)`; `emit(event)`; `close()` (stops Live cleanly; safe to call twice). Main answers and room turns: a `Panel` titled with the speaker in its style, Markdown inside, updated live while `TextMessageContent` arrives (re-render Markdown at most 8 times a second), footer from `format_footer` as the panel's `subtitle` (right-aligned) on `RunFinished`/`TextMessageEnd`. A spinner line `"thinking… {n} s"` while a run has started and no text or tool call has arrived.
  - Ignores `MessageEvent`, `NoticeEvent`, `FinalAnswerEvent`, `TodosEvent` (plain-only events).
- [ ] **Step 1: Write the failing tests** (console = `make_console(file=io.StringIO(), width=100, force_terminal=False)` with `record=True`; assert on `console.export_text()`):
  - `test_answer_box_with_markdown_and_footer`: events for a main run with text `"# Title\n\n- one\n- two"` → output contains `"main"`, `"Title"`, `"• one"`, no `"#"` or `"- one"`, and the footer text.
  - `test_streaming_updates_then_settles`: deltas then end → final export contains the full text once.
  - `test_room_turn_is_a_box`: `StepStarted(kind="room", agent="critic")` + its text → a box titled `room / critic`.
  - `test_renders_at_narrow_width`: width 50 → every line ≤ 50 chars, box corners present.
  - `test_close_twice_is_safe`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement**; add the dependency with `uv add "rich>=14"`.
- [ ] **Step 4: Run** the tests, the full suite, `test_structure` (rich only in `ui/`). Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Rich: answer boxes with streamed Markdown and footers"`

---

### Task 5: The timeline

**Files:**
- Create: `src/deep_agents_demo/ui/cli/richui/timeline.py`
- Modify: `richui/renderer.py`
- Test: `tests/test_rich_timeline.py`

**Interfaces:**
- Consumes: Task 4 `RichRenderer`, `THEME`; Tasks 1–3 events.
- Produces: `class Timeline`: holds the tree of the current turn (runs → tool calls / subagents / steps) and renders it as Rich `Text` lines with `├─ │ └─` guides:
  - tool line: `{name}  {short_args}  {✓|✗|⏸} {summary} · {duration}` where `short_args` is the first argument value cut to 48 chars (all of them as JSON under the line when `verbose`, plus the full result);
  - subagent: `task → {name}` with its calls nested and a last line `{name}: {first line of its answer}`, its footer as dim text;
  - plan step: `step {n}/{total}  {agent}  ✓ passed review · {attempts} tries · {duration}` (`⠹ {phase}… {s} s` while running);
  - text beside tool calls (`TextMessageEnd.has_tool_calls`): one dim line with its first 100 chars;
  - `send_file` success: the `[file]` line right under it with a `[link=file:///…]` to the saved path (from `FileEvent`);
  - `StateSnapshot`: a checklist (`✓` done, `▸` in progress, `○` pending);
  - finished top-level branches are printed (scrollback) as soon as they settle; only unfinished ones stay in Live.
- [ ] **Step 1: Write the failing tests:** `test_tool_line`, `test_failed_tool_line_shows_reason`, `test_subagent_branch`, `test_two_branches_render_separately`, `test_plan_step_line`, `test_text_beside_tool_calls_is_a_dim_line`, `test_file_line_links_to_the_file` (export with `force_terminal=True` contains the OSC-8 hyperlink to `file:///`), `test_todos_checklist`, `test_verbose_shows_json_and_full_result`, `test_finished_branches_are_printed_once`.
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests + suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Rich: timeline of tool calls, subagents, steps, files and to-dos"`

---

### Task 6: Approvals, choices and Ctrl+C with Rich

**Files:**
- Create: `src/deep_agents_demo/ui/cli/richui/interaction.py`
- Modify: `richui/renderer.py` (pause/resume), `ui/cli/interrupts.py` (write the pause notice through a callback when given)
- Test: `tests/test_rich_interaction.py`

**Interfaces:**
- Consumes: `ApprovalRequest`, `Decision`, `Interaction` port; `RichRenderer`.
- Produces: `class RichInteraction(Interaction)`: `__init__(self, console, renderer: RichRenderer | None = None, ask=None)`; `decide(request) -> Decision` prints a `Panel` (title = `request.title`, border `approval`) with the summary, keys line `[a]pprove  [r]eject (with a reason)  [d]iff  [f]ull file >` with the letters highlighted; `d` prints `request.diff` with `Syntax(…, "diff")`; same answers and messages as `CliInteraction`. `choose`/`ask` print the prompt styled. Every method calls `renderer.pause()` before reading and `renderer.resume()` after. `RichRenderer.pause()` / `resume()` stop and restart Live without losing the tree. `ctrl_c_controls(..., notify=None)`: when `notify` is given the pause message goes through it instead of `os.write`.
- [ ] **Step 1: Write the failing tests:** `test_approve`, `test_reject_with_reason` (same `Decision`s as `CliInteraction` for the same scripted answers), `test_diff_is_coloured` (`force_terminal=True` export has red/green escapes on `-`/`+` lines), `test_live_is_paused_while_asking` (a fake renderer records `pause`, then the ask, then `resume`).
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests + suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Rich: approval panel, coloured diff, prompts that pause the live view"`

---

### Task 7: Tables

**Files:**
- Create: `src/deep_agents_demo/ui/cli/richui/tables.py`
- Modify: `ui/cli/repl.py` (`ChatSession(..., tables=None)` uses it for `/threads`, `/files`, `/agents` when given)
- Test: `tests/test_rich_tables.py`

**Interfaces:**
- Produces: `chats_table(chats, current) -> rich.table.Table`, `files_table(files, folder) -> Table`, `agents_table(rows) -> Table` (same columns as today's text: chat/turns/last used/last message with `*` current; file/version/size/saved/caption; agent/source/description); a `RichTables` object with `threads(chats, current)`, `files(files, folder)`, `agents(rows)` that print to the console. `ChatSession` keeps today's text output when `tables is None`.
- [ ] **Step 1: Write the failing tests:** each table exported at width 100 contains its header names and rows; `test_chat_session_uses_rich_tables` (a fake `tables` records the call for `/threads`).
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests + suite (all existing `/threads` text tests unchanged). Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Rich: tables for /threads, /files and /agents"`

---

### Task 8: The input box and the `/` menu

**Files:**
- Modify: `pyproject.toml` (`prompt_toolkit>=3.0.50`), `uv.lock`
- Create: `src/deep_agents_demo/ui/cli/richui/menu.py`, `richui/input_box.py`
- Test: `tests/test_slash_menu.py`

**Interfaces:**
- Consumes: the `@command` registry (`repl.COMMANDS`: method → `.command = (names, usage, summary)`), `Chats.list()`, the agent registry's names/descriptions, the image extensions.
- Produces:
  - `menu.py` (no terminal I/O): `MenuItem(value, label, detail)`; `filter_items(items, typed) -> list[MenuItem]` (contains, case-insensitive, prefix matches first, then registry order); `command_items(commands) -> list[MenuItem]`; `argument_items(command, typed, chats, agents, cwd) -> (title, list[MenuItem])` for `/switch` `/delete` `/rename` (`your chats`), `/room` (`agents`), `/image` (`images`, PNG/JPG/JPEG under the typed folder); `NEEDS_ARGUMENT = {"/rename", "/switch", "/delete", "/room", "/image"}`; `ghost_text(typed, item) -> str` (the rest of the selected item after what is typed); `window(items, selected, size=5) -> (start, end)`; `footer(selected, total) -> str` (e.g. `"↑ 7/11 ↓ · Tab · Enter · Esc"`); `fit_row(label, detail, width) -> str` (cuts detail with "…", drops it when too narrow).
  - `input_box.py`: `class InputBox`: `__init__(self, commands, chats_provider, agents_provider, history_path, prompt_label)`; `read(prompt_label: str) -> str` (raises `EOFError` on Ctrl+D/EOF like `input()`); a `prompt_toolkit` `Application` (not full screen) with layout `[menu frame (0–5 rows, rounded, title, footer edge)] [input frame "label › …"]`, grey inline ghost text, key bindings ↑↓ (move/scroll; ↑ in an empty box = history), Tab/→ accept, Enter (menu open: accept + run, or accept + wait when the command needs an argument; closed: send), Esc (close menu, clear ghost); `FileHistory(history_path)`; redraws on resize (prompt_toolkit default).
- [ ] **Step 1: Write the failing tests** for `menu.py`: `test_filter_contains_prefix_first` (`/n` → `/new` first, then `/agents`, `/rename`; nothing without `n`), `test_window_scrolls_past_five`, `test_footer_text`, `test_ghost_text` (`"/r"` + `/rename` → `"ename"`), `test_argument_items_for_switch` (title `your chats`, filtered by typed), `test_narrow_terminal_drops_descriptions` (`fit_row(…, 30)` has no detail; at 60 the detail ends with "…"). For `input_box.py` with `prompt_toolkit.input.create_pipe_input()` and `DummyOutput`: `test_enter_runs_the_selected_command` (`"/n\r"` → `"/new"`), `test_enter_waits_for_an_argument` (`"/re\r"` then `"x\r"` → `"/rename x"`), `test_tab_accepts_ghost`, `test_esc_closes_and_sends_what_is_typed`, `test_history_up_recalls`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement**; `uv add "prompt_toolkit>=3.0.50"`.
- [ ] **Step 4: Run** tests + suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Input: boxed prompt with a Codex-style / menu, completion and history"`

---

### Task 9: Wiring, the --plain flag, docs, live checks

**Files:**
- Modify: `ui/cli/main.py` (`--plain`), `ui/cli/repl.py` (`run_chat(..., plain=False)` chooses mode; `runners_for` uses the same sink and interaction), `README.md`
- Create: `src/deep_agents_demo/ui/cli/richui/mode.py`
- Test: `tests/test_rich_mode.py`, `tests/test_config.py` (`--plain` parsing)

**Interfaces:**
- Produces: `richui/mode.py`: `use_rich(interactive: bool, plain: bool, stdin_tty: bool, stdout_tty: bool) -> bool`; `run_chat` builds, when `use_rich(...)`: `console = make_console()`, `RichRenderer(console, verbose)`, `RichInteraction(console, renderer)`, `RichTables(console)`, `InputBox(...)` (history `data/history/<safe user>.txt`); otherwise today's `CliRenderer`, `CliInteraction`, `input()`. The chosen sink and interaction go to `stream_turn`/`run_turn`, `runners_for` and `ctrl_c_controls` (notify via the renderer). On exit (including Ctrl+C) `renderer.close()` runs in a `finally`.
- [ ] **Step 1: Write the failing tests:** `test_mode_is_plain_when_stdout_is_not_a_terminal`, `test_mode_is_plain_for_prompt_and_plain_flag`, `test_mode_is_rich_in_a_terminal`; `tests/test_config.py::test_plain_flag`.
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement**; README: a "The terminal UI" section (what you see, the `/` menu keys, `--plain`, `NO_COLOR`), the layout tree (`ui/cli/richui/`).
- [ ] **Step 4: Run** the full suite. Expected: OK.
- [ ] **Step 5: Live checks** (user `tools-check`; record what was and wasn't checked): `--prompt` output unchanged (compare with a run on `main`), output redirected to a file is plain, and a scripted interactive session through a pseudo-terminal if available (`pywinpty` is not a dependency: if no PTY is available, state that the Rich and menu checks are left to the user). Clean up chats, outputs, workspace files.
- [ ] **Step 6: Commit** `git commit -m "Rich UI wired into the chat, --plain, README"`

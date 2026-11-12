# File outputs in the chat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent delivers documents, datasets and pages as versioned files in the chat (`send_file`), shown as a `[file]` line, instead of printing them as messages.

**Architecture:** A versioned output store in `persistence/` (outside the agent's workspace), a `features/files/` folder (validation, the `send_file` tool, a middleware that tells the model which files this chat already has), a `FileEvent` the turn engine emits from the tool message's artifact, and CLI commands to list files. The tool reads through the agent's backend, so local files and sandboxes behave the same; identity comes from code, never from the model.

**Tech Stack:** Python 3.11+, `langchain.tools.tool(response_format="content_and_artifact")` + `ToolRuntime` (verified on langchain 1.4.9 / langchain-core 1.7.6 / deepagents 0.7.34), stdlib `csv`, `json`, `html.parser`, `xml.etree`; unittest. No new dependencies.

**Spec:** `design/specs/2026-11-12-file-outputs-design.md` (incl. its "Verified on 2026-11-12" section).

## Global Constraints

- Allowed extensions: `.md`, `.txt`, `.csv`, `.html`, `.tex`, `.svg`, `.json`, `.png`, `.jpg` (and `.jpeg`). Not `.docx`, `.pptx`, `.xlsx`, `.pdf`.
- Size caps: 5 MB for text types, 20 MB for images.
- Store layout: `data/outputs/<user>/<chat>/<name>` (latest), `.versions/<name>.v1 ...`, `index.json`; `<user>` and `<chat>` through `safe_folder_name`.
- `send_file` returns JSON `name`, `version`, `bytes`, `kind` to the model, never the content; no approval; main agent only (`MAIN_ONLY`); respects `DISABLED_TOOLS`.
- `FILE_OUTPUT=auto|always|never`, default `auto`; `never` leaves the feature out.
- The prompt text is the spec's, verbatim (with the `always` variant of its first sentence).
- Layering rules hold with no additions to `LEGACY` or `IO_PENDING`; a feature never imports another feature.
- Every commit: `uv run python -m unittest discover -s tests` green; one-line commit messages, no trailer; push only when asked.
- Live checks use user `tools-check` and clean up after (chats, `data/outputs/tools-check`, workspace folders).

## Review Focus

1. **A chat with no saved checkpoint yet, or a thread id that isn't a chat pair** (a one-shot `--prompt` on a new chat; tests with `thread_id="t"`) — expected: `send_file` works for a new chat (the pair exists in config before the first save) and returns a clear error, saving nothing, for a non-chat thread. (Task 3 test `test_non_chat_thread_is_refused`.)
2. **A path the agent gives in a sandbox** (`/home/daytona/report.md`, `/sandbox/out.csv`) or relative (`report.md`) — expected: read through the backend as given; the stored name is the base name. (Task 3 test `test_reads_through_the_backend_not_the_disk` uses a sandbox-style absolute path; `test_relative_path_is_read_as_given`.)
3. **Two files with the same base name from different folders in one chat** — expected: the second becomes the next version of the first (names are per chat, as the spec says); the reply's `version` tells the model. (Task 1 test `test_same_name_twice_versions`.)
4. **Renaming a chat to a name whose outputs folder already exists** (left over from a deleted chat with outputs not cleaned, or a manual folder) — expected: rename refuses or merges without losing files; never silently overwrites. (Task 5 test `test_rename_onto_existing_outputs_keeps_both`: refuse with a clear error before the chat is renamed.)
5. **Windows paths and odd names** (`Report Final.MD`, `notes.Md`, `..`, `.md`, names with spaces) — expected: extension matched case-insensitively and stored lowercased; spaces become `-`; empty stems refused. (Task 2 test `test_names_are_sanitised`.)

---

### Task 1: The output store

**Files:**
- Create: `src/deep_agents_demo/persistence/outputs.py`
- Test: `tests/test_outputs.py`

**Interfaces:**
- Produces:
  - `OutputFile` (frozen dataclass): `name: str`, `version: int`, `bytes: int`, `kind: str` (the extension without the dot, lowercase), `caption: str`, `saved_at: str` (ISO, UTC, seconds), `path: Path` (the latest copy); `to_dict()` (path as `str`).
  - `class OutputStore(root: Path)` with:
    - `save(user, chat, name, data: bytes, caption="") -> OutputFile` — writes `.versions/<name>.v<N>` then replaces `<name>`, updates `index.json` (list of version records; one record per version, latest last); thread-safe (one `threading.RLock`).
    - `list(user, chat) -> list[OutputFile]` — the latest version of each name, in first-saved order.
    - `read(user, chat, name, version=None) -> bytes` — `KeyError` if unknown.
    - `delete_thread(user, chat) -> None` — removes the chat's folder (no error if absent).
    - `rename_thread(user, old, new) -> None` — moves the folder; `ValueError` if `new` already has a folder; no-op if `old` has none.
    - `folder(user, chat) -> Path`.
- [ ] **Step 1: Write the failing tests** `tests/test_outputs.py`:
  - `test_save_then_list_and_read`: `save("alice", "parks", "plan.md", b"# Plan", "first")` → `OutputFile` with `version == 1`, `bytes == 6`, `kind == "md"`; `list` returns it; `read` returns `b"# Plan"`; `index.json` exists and holds one record.
  - `test_same_name_twice_versions`: second save → `version == 2`; `read(..., version=1) == first bytes`; `read(...)` is the second; `list` has one entry with version 2.
  - `test_folders_are_safe`: user `"a b/c"`, chat `"x/../y"` → folder is `root / "a_b_c" / "x_.._y"` (via `safe_folder_name`); nothing written outside `root`.
  - `test_delete_and_rename_thread`: rename moves files (old folder gone, new readable); renaming onto an existing folder raises `ValueError` and changes nothing; delete removes the folder; deleting/renaming an absent chat is a no-op.
- [ ] **Step 2: Run** `uv run python -m unittest tests.test_outputs`. Expected: ImportError.
- [ ] **Step 3: Implement** `persistence/outputs.py` (writes: temp file + `os.replace`, so a crash never leaves a half file as latest).
- [ ] **Step 4: Run** the test file, then the full suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Files: versioned output store"`

---

### Task 2: Validation

**Files:**
- Create: `src/deep_agents_demo/core/media.py`, `src/deep_agents_demo/features/files/__init__.py`, `features/files/validate.py`
- Modify: `features/images/uploads.py` (import `sniff`, `SIGNATURES` from `core/media.py`; keep its public names working by import)
- Test: `tests/test_files.py` (validation part)

**Interfaces:**
- Produces:
  - `core/media.py`: `SIGNATURES`, `sniff(data: bytes) -> str | None` (moved from `features/images/uploads.py`: `features/files` may not import `features/images`).
  - `features/files/validate.py`: `ALLOWED = (".md", ".txt", ".csv", ".html", ".tex", ".svg", ".json", ".png", ".jpg", ".jpeg")`, `TEXT_LIMIT = 5 * 1024 * 1024`, `IMAGE_LIMIT = 20 * 1024 * 1024`, `clean_name(path: str) -> str` (base name, lowercase extension, spaces → `-`, other unsafe characters → `_`; `ValueError` for an empty stem or no extension), `validate(name: str, data: bytes) -> str | None` (None = valid, else the error the agent reads), `describe(name, data) -> str` (e.g. `"csv, 12 rows x 4 columns"`, `"md, 1.2 KB"`; used in the tool result's `kind` detail).
- [ ] **Step 1: Write the failing tests** (class `ValidationTests`), each asserting `validate(...)` returns a string containing the given words, or `None`:
  - valid `.md`, `.txt`, `.tex`, `.json` (`{"a": 1}`), `.csv` (`"a,b\n1,2\n"`), `.html` (`"<p>hi</p>"`), `.svg` (`'<svg xmlns="http://www.w3.org/2000/svg"></svg>'`), `.png` (PNG signature + bytes), `.jpg` → `None`.
  - `report.docx` → contains `".md"` (lists allowed types); empty `.md` → `"empty"`; `b"\xff\xfe"` as `.md` → `"UTF-8"`; 5 MB + 1 byte `.md` → `"5 MB"`; `.csv` `"a,b\n1\n"` → `"column"`; `.json` `"{"` → `"JSON"`; `.svg` `"<html/>"` → `"<svg>"`; `.svg` `"<svg"` (broken) → `"SVG"`; `.png` with JPEG bytes or text → `"PNG"`; a `.md` holding `"AKIA" + 16 uppercase chars` (an AWS key shape `find_secrets` flags) → `"credential"`.
  - `test_names_are_sanitised`: `clean_name("/sandbox/Report Final.MD") == "Report-Final.md"`; `clean_name("..\\..\\x.md") == "x.md"`; `clean_name("C:/a/b.Csv") == "b.csv"`; `clean_name(".md")` and `clean_name("noext")` raise `ValueError`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement.** HTML: `html.parser.HTMLParser` must see at least one start tag; SVG: `xml.etree.ElementTree.fromstring`, root tag (namespace stripped) `== "svg"`; CSV: `csv.reader`, every non-empty row the same length, at least one row; secrets: `core.secrets.find_secrets` with `split_findings` — refuse credentials (the "wording" kinds don't block, as in skills).
- [ ] **Step 4: Run** the tests and the full suite (images tests still pass with `sniff` moved). Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Files: validation of delivered files; sniff moved to core.media"`

---

### Task 3: `send_file`, the feature, the setting and the wiring

**Files:**
- Create: `features/files/feature.py`
- Modify: `config.py` (`FILE_OUTPUT_MODES`, `Settings.file_output`), `core/assembly.py` (`MAIN_ONLY`), `startup/resources.py` (`Resources.outputs`), `startup/runtime.py` (`features()`, `Runtime.outputs`)
- Test: `tests/test_files.py` (tool part), `tests/test_config.py`

**Interfaces:**
- Consumes: Task 1 `OutputStore`, Task 2 `clean_name`, `validate`, `describe`.
- Produces:
  - `config.py`: `FILE_OUTPUT_MODES = ("auto", "always", "never")`; `Settings.file_output: str = "auto"`, read from `FILE_OUTPUT` (lowercased; else `ValueError("FILE_OUTPUT must be one of: auto, always, never")`).
  - `core/assembly.py`: `MAIN_ONLY = ("run_plan", "start_room", "send_file")`.
  - `features/files/feature.py`:
    - `PROMPT` (spec text, verbatim) and `ALWAYS_FIRST_SENTENCE` (`"for every answer longer than a few lines"` replacing the first sentence's trigger).
    - `make_send_file(backend, outputs: OutputStore, user: str)` → a LangChain tool `send_file(path: str, caption: str = "", runtime: ToolRuntime) -> tuple[str, dict | None]` with `response_format="content_and_artifact"`. Chat name from `runtime.config["configurable"]["thread_id"]` parsed as the `[user, chat]` JSON pair; refuse if it isn't a pair or its user differs from `user`. Reads with `backend.download_files([path])`; `error` → `"Can't read {path}: {error}. Write the file first (write_file), then send it."`. On success content = `json.dumps({"name", "version", "bytes", "kind"})`, artifact = `{"saved": True, **output_file.to_dict()}`; on any failure content = the error, artifact = `None`.
    - `class DeliveredFilesNote(AgentMiddleware)`: on each model call, if the run's chat has delivered files, appends `"Files delivered in this chat: plan.md (v2), data.csv (v1)."` to the system prompt for that call (same pattern as `features/memory/recall.py`; reads the thread from `langgraph.config.get_config()`; nothing when none or not a chat pair).
    - `files_feature(backend, outputs, user, mode="auto") -> Feature | None`: `None` for `"never"`; else `Feature(name="files", prompt=..., tools=(send_file,), middleware=(DeliveredFilesNote(outputs, user),))`.
  - `startup/resources.py`: `Resources.outputs: OutputStore` (root `settings.paths.data / "outputs"`), created in `open`.
  - `startup/runtime.py`: `features()` yields `files_feature(setup.backend, resources.outputs, user, settings.file_output)` right after `images_feature` when not `None`; `Runtime.outputs` set from `resources.outputs`.
- [ ] **Step 1: Write the failing tests** (class `SendFileTests` in `tests/test_files.py`, through `build_agent` and the fake `Model` from `test_agents`, thread `thread_id("alice", "parks")`):
  - `test_valid_markdown_is_saved_as_v1`: agent writes `/plan.md` (fake model calls `write_file`, then `send_file`); store has `plan.md` v1; the tool message content parses as JSON with keys exactly `name, version, bytes, kind` and does not contain the file text; `ToolMessage.artifact["saved"] is True`.
  - `test_failures_save_nothing`: `send_file` on a missing path, on `notes.docx`, and on a `.md` containing a credential → each tool message contains the error words, `artifact is None`, store empty.
  - `test_reads_through_the_backend_not_the_disk`: a fake backend whose `download_files(["/home/daytona/report.md"])` returns bytes for a path that doesn't exist locally → saved as `report.md`.
  - `test_relative_path_is_read_as_given`: `download_files` receives `"report.md"` unchanged.
  - `test_identity_comes_from_code`: thread pair user `"bob"` with the feature built for `"alice"` → refused, nothing under `bob/` or `alice/`; arguments can't name a user or chat (the tool schema's properties are exactly `path`, `caption`).
  - `test_non_chat_thread_is_refused`: thread id `"t"` → error mentioning "chat", nothing saved.
  - `test_subagents_cannot_send_files`: `send_file` absent from the general-purpose subagent's and a `worker_factory` worker's tools (check `SEEN` tool lists, as `test_agents` does).
  - `test_delivered_files_are_listed_for_the_model`: after one save, the next model call's system prompt contains `"plan.md (v1)"`.
  - `tests/test_config.py::test_file_output_setting`: default `"auto"`; `"Never"` → `"never"`; `"sometimes"` → `ValueError` naming the modes. And `files_feature(..., mode="never") is None`; `"always"` prompt contains `"for every answer longer than a few lines"`.
- [ ] **Step 2: Run.** Expected: ImportError / failures.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** the tests and the full suite (incl. `test_structure`). Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Files: send_file tool, files feature, FILE_OUTPUT setting, main agent only"`

---

### Task 4: `FileEvent` and the `[file]` line

**Files:**
- Modify: `models/events.py`, `core/turns.py`, `ui/cli/render.py`
- Test: `tests/test_files.py` (events part), `tests/test_render.py`

**Interfaces:**
- Consumes: Task 3's artifact (`{"saved": True, name, version, bytes, kind, caption, saved_at, path}`).
- Produces:
  - `FileEvent(label: str, name: str, version: int, kind: str, bytes: int, caption: str, path: str)` with `to_dict()`.
  - `core/turns.stream_once`: after a tool message named `send_file` whose `artifact` is a dict with `saved` true, emit `FileEvent(who, ...)` right after its `MessageEvent` (and not when the sink is silent).
  - `CliRenderer`: `FileEvent` → `[file] {name}  v{version}  {size}  ->  {path}` where `size` is `"{n} B"` under 1024, else `"{n/1024:.1f} KB"`, else `"{n/1024/1024:.1f} MB"`, and `path` is shown relative to the current directory when it is under it (with `/`), else as is.
- [ ] **Step 1: Write the failing tests:**
  - `test_one_file_event_for_a_successful_send` and `test_no_file_event_for_a_failed_send` (run a turn with `RecordingUI`; count `FileEvent` lines: 1 and 0).
  - `tests/test_render.py::test_file_line`: `CliRenderer(write=lines.append).emit(FileEvent("main", "sandbox-policy.md", 1, "md", 4300, "", str(Path.cwd() / "data/outputs/demo-user/city-demo/sandbox-policy.md")))` → `["[file] sandbox-policy.md  v1  4.2 KB  ->  data/outputs/demo-user/city-demo/sandbox-policy.md"]`.
- [ ] **Step 2: Run.** Expected: ImportError.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests and full suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Files: FileEvent from the tool's artifact, [file] line in the CLI"`

---

### Task 5: Chats keep their files; `/files` and `--list-files`

**Files:**
- Modify: `persistence/chats.py` (`Chats(saver, agent, user, outputs=None)`; `delete`, `rename`), `startup/runtime.py` (pass `resources.outputs` / an `OutputStore` in `history_runtime`), `ui/cli/repl.py` (`/files`), `ui/cli/main.py` and `ui/cli/commands.py` (`--list-files THREAD`), `ui/cli/render.py` (`format_files`)
- Test: `tests/test_files.py` (chat part), `tests/test_threads.py`, `tests/test_config.py` (help/dispatch registry test covers `/files`)

**Interfaces:**
- Consumes: Task 1 `OutputStore.delete_thread`, `rename_thread`, `list`.
- Produces: `Chats.delete` also calls `outputs.delete_thread(user, name)`; `Chats.rename` calls `outputs.rename_thread(user, old, new)` **before** moving checkpoints (so a refusal leaves the chat unchanged); `format_files(files: list[OutputFile]) -> str` (columns `FILE  VERSION  SIZE  SAVED  CAPTION`, then the folder path; `"No files delivered in this chat yet."` when empty); `/files` prints it for the current chat; `commands.list_files(settings, user, thread)` prints it for `THREAD`.
- [ ] **Step 1: Write the failing tests:** `test_deleting_a_chat_removes_its_files`, `test_renaming_a_chat_moves_its_files`, `test_rename_onto_existing_outputs_keeps_both` (existing folder for the new name → `ValueError`, chat and both folders unchanged), `test_files_command` (`ChatSession` with `say=said.append` and a store with one file → the table line contains the name and `v1`), `test_list_files_flag` (`parse_args(["--list-files", "parks"])` dispatches to `commands.list_files`).
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests and full suite. Expected: OK.
- [ ] **Step 5: Commit** `git commit -m "Files: chats delete and rename their files; /files and --list-files"`

---

### Task 6: Docs, eval, acceptance

**Files:**
- Modify: `README.md` ("What it does" row, a "Files the agent delivers" section, "Where data is stored" row for `data/outputs/`, the `FILE_OUTPUT` setting, the layout tree: `persistence/outputs.py`, `core/media.py`, `features/files/`)
- Create: `evals/file_output.py`

- [ ] **Step 1: Write the eval** `evals/file_output.py` (same shape as `evals/memory_recall.py`: temporary workspace, `--runs N`, results JSON to `data/evals/file_output-<stamp>.json`, summary table on stdout). Prompts — file: "Write a short document about the sandbox policy.", "Give me a CSV of five trees with name, height and leaf type.", "Write a one-page plan for a city park scene.", "Make a simple SVG of a red circle.", "Write a short HTML page that lists three parks."; no file: "What is a sandbox?", "Explain what MEMORY_RECALL does.", "How many trees does the catalog have?", "What's the difference between a skill and a subagent?", "Say hello."; expected extensions `.md`, `.csv`, `.md`, `.svg`, `.html`. Score per prompt: `send_file` called (or not), extension as expected, file validated, chat reply under 60 words when a file was sent. Follow-up: after the first prompt, "Make the intro shorter." must deliver `v2` of the same name.
- [ ] **Step 2: Docs** per Files above.
- [ ] **Step 3: Full suite.** Expected: OK.
- [ ] **Step 4: Live acceptance** (user `tools-check`):
  - `uv run deep-agents-demo --user tools-check --thread files-check --prompt "Write a short document about the sandbox policy."` prints a `[file]` line; the file exists under `data/outputs/tools-check/files-check/`; the reply doesn't contain the document.
  - Same with `SANDBOX_PROVIDER=openshell` (only if the OpenShell sandbox is running; otherwise record that it wasn't tested).
  - `--prompt "Explain what the sandbox policy does."` answers in chat, no `[file]` line.
  - Interactive: `/files` lists the file; "Make it shorter." → `v2`.
  - Run the eval once (`--runs 1`); record the score.
  - Clean up: `--delete-thread files-check --yes` (removes its outputs too), `data/outputs/tools-check`, `data/evals` output kept (gitignored).
- [ ] **Step 5: Commit** `git commit -m "Files: README section, file output eval"`

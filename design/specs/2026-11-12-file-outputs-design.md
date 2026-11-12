# File outputs in the chat (handoff)

Status: proposed on 2026-11-12, not built. Written against commit `f4e9751`.
"Verify before building" checked on 2026-11-12 (see the last section).

## Intent

When the user asks for something they will keep (a document, a report, a dataset, a
page), the agent writes it to a file and delivers that file in the chat, with a short
reply, instead of printing the whole content as a message. Follow-up requests edit
the same file and deliver a new version. Short questions are still answered in chat.

## Decided by the user

- Files are chat attachments, generated or edited inside the conversation. They are
  not hosted pages and have no share link.
- Supported now: `.md`, `.txt`, `.csv`, `.html`, `.tex`, `.svg`, `.json`, and
  `.png` / `.jpg` produced by code. Excluded for now: `.docx`, `.pptx`, `.xlsx`, `.pdf`.
- The agent chooses file or chat reply by itself. Example: "write a document about
  X" produces an `.md` file; "explain X" is a chat answer.
- Single user. Build it in the CLI first, in a way the GUI reuses unchanged.

## Read first

- `core/feature.py`: the `Feature` dataclass every capability returns.
- `startup/runtime.py`: `features()` and `build_runtime()`, where a feature is wired.
- `core/turns.py`: `stream_once()` turns stream updates into events.
- `models/events.py`, `core/ports.py`, `ui/cli/render.py`: events and how they are shown.
- `features/images/`: the closest existing feature (uploads, prompt, `/image`).
- `features/sandbox/providers.py`: `LazySandbox.download_files`, `BackendSetup`.
- `persistence/chats.py`: `thread_id(user, name)` and `Chats.delete`.
- `tests/test_structure.py`: the layering rules. They are enforced; do not add to
  `LEGACY` or `IO_PENDING`.
- README section "Project layout" (the extension guide).

## Design

**No new way to write files.** The agent keeps using `write_file`, `edit_file` and,
in a sandbox, `execute`. The feature adds one tool that says "this file is the
result".

**Tool: `send_file(path: str, caption: str = "") -> str`.**

1. Read the file through the agent's backend (`backend.download_files([path])`),
   never from a local path. This is what makes it work the same with local files,
   OpenShell and Daytona.
2. Validate (see below). On failure return an error string the agent can act on.
3. Save the bytes to the output store as a new version.
4. Return a short JSON result to the model: `name`, `version`, `bytes`, `kind`.
   Never return the content.

The model must not choose identity: user and thread come from application code
(the same rule as `memory_tools`). Take the thread from the run's config
(`configurable.thread_id`, which is `thread_id(user, name)`).

**Output store** (`persistence/outputs.py`). Delivered files live outside the
agent's workspace so a delivered version can't be changed afterwards:

    data/outputs/<user>/<thread>/<name>            # latest
    data/outputs/<user>/<thread>/.versions/<name>.v1, .v2, ...
    data/outputs/<user>/<thread>/index.json        # name, version, bytes, kind, caption, time

Use `safe_folder_name` for user and thread. Sending the same name again in a thread
creates the next version. API: `save(user, thread, name, data, caption) -> OutputFile`,
`list(user, thread)`, `read(user, thread, name, version=None)`, `delete_thread(user, thread)`.

**Validation** (`features/files/validate.py`), before saving:

- Extension allowlist (the list above). Anything else: error naming the allowed types.
- Size cap: 5 MB for text types, 20 MB for images (match `features/images`).
- File name: base name only, kebab-case suggested, sanitised; no path parts.
- Text types must decode as UTF-8 and not be empty.
- `.csv`: parses with the `csv` module, consistent column count; report rows and columns.
- `.json`: parses.
- `.html` / `.svg`: well formed enough to parse; `.svg` has an `<svg>` root.
- `.png` / `.jpg`: checked by content with the existing `sniff()`, not by name.
- Secrets: run `core/secrets.py` on text content and refuse a file that contains one.

**Event.** Add to `models/events.py`:

    FileEvent(label, name, version, kind, bytes, caption, path)   # path: where it was saved

The engine emits it; the tool does not print. In `core/turns.py`, when
`stream_once` sees the tool message of a successful `send_file`, emit a `FileEvent`
after the `MessageEvent`. Carry the metadata in the tool message's structured
payload, not by parsing prose (LangChain tools support a content-plus-artifact
return; check the pinned version, and fall back to parsing the JSON content).
`CliRenderer` prints one line, for example:

    [file] sandbox-policy.md  v1  4.2 KB  ->  data/outputs/demo-user/city-demo/sandbox-policy.md

**Prompt** (the feature's `prompt`, keep it this short):

> When the user asks for a document, report, guide, plan, dataset, page or any content
> they will keep or reuse, write it to a file and deliver it with send_file. Choose the
> format from the content: .md for documents, .csv for tabular data, .html for pages,
> .tex when they ask for LaTeX, .svg for simple graphics, .txt for plain notes. Answer
> in chat for questions, explanations and short replies; as a guide, anything with a
> title and sections, or longer than about a page, is a file. Use short kebab-case
> names. After sending a file, reply with one or two sentences about it and never
> repeat its content. When asked to change a file you sent, edit that file and send it
> again; don't start a new one. If the user says "answer here" or "as a file", do that.

Also list the thread's delivered files in the prompt section when there are any
(name and version), so "the document" has a referent after a resume.

**Setting.** `FILE_OUTPUT=auto|always|never` in `Settings` (default `auto`), read
once in `config.py` like `MEMORY_RECALL`. `never` leaves the feature out; `always`
changes the first prompt sentence to "for every answer longer than a few lines".

**Approvals.** `send_file` needs no approval: it only copies a file the agent
already wrote to a place the agent can't write. Respect `DISABLED_TOOLS` as usual.

**Wiring.**

- `features/files/feature.py`: `files_feature(backend, outputs, user, mode) -> Feature`
  with `name="files"`, the prompt and `tools=(send_file,)`.
- `startup/runtime.py`: one line in `features()`, after `images_feature`. Put the
  store on `Runtime` (`outputs`) so front ends can list and read files.
- `persistence/chats.py` `Chats.delete`: also delete that thread's outputs.
  Renaming a chat must move its outputs folder.
- CLI: `/files` lists the current chat's delivered files; `--list-files THREAD`.
- `.gitignore`: `data/` is already ignored. Add the path to README "Where data is stored".

## Out of scope for this piece

- Attaching non-image input files (`/attach`). Piece 2: reuse `Uploads`, add text
  types, add a note to the user message with name, type and size; don't inline content.
- Diffs between versions. Piece 3: reuse the diff view from skill approvals.
- GUI preview. When built, HTML and SVG must render in an iframe with the `sandbox`
  attribute and no same-origin access. Never in the app's own origin.
- Compiling LaTeX, generating PDFs or Office files, an image-generation model.

## Tests (offline, `unittest`, same style as `tests/test_images.py`)

`tests/test_files.py`:

- `send_file` on a valid `.md`: saved as v1, result has no content, `index.json` updated.
- Same name twice: v2 exists, v1 still readable, latest is v2.
- Each validation failure returns an error and saves nothing: bad extension, too
  large, empty, invalid CSV, invalid JSON, wrong image bytes, path traversal in the
  name, a secret in the content.
- Reads go through the backend: use a fake backend whose `download_files` returns
  bytes for a path that doesn't exist locally (the sandbox case).
- Identity comes from application code: a tool call can't write into another
  user's or thread's folder.
- Deleting a chat removes its outputs; renaming moves them.
- `stream_once` emits one `FileEvent` for a successful `send_file` and none for a failed one.
- `CliRenderer` output for `FileEvent`, through `tests/recording_ui.py`.
- `FILE_OUTPUT` parsing in `tests/test_config.py`; `never` leaves the tool out.
- `tests/test_structure.py` still passes with no exceptions added.

## Eval (live, optional but wanted)

`evals/file_output.py`, same shape as `evals/memory_recall.py` (temporary
workspace, results to `data/evals/`). Ten prompts: five that should produce a file
("write a document about...", "give me a CSV of..."), five that should not
("what is...", "explain..."). Score per prompt: was `send_file` called, was the
extension the expected one, did the file validate, and was the chat reply under
about 60 words when a file was sent. Add one follow-up case: "make the intro
shorter" must produce v2 of the same name.

## Acceptance

- `uv run deep-agents-demo --prompt "Write a short document about the sandbox policy."`
  prints a `[file]` line, the file exists under `data/outputs/`, and the reply
  does not contain the document.
- The same works with `SANDBOX_PROVIDER=openshell`.
- "Explain what the sandbox policy does" answers in chat with no file.
- All tests pass on the CI matrix. README gets a "Files the agent delivers"
  section and a row in "What it does".

## Verify before building

- How a tool reads the run config and backend in the pinned `deepagents==0.7.34`
  and LangChain versions (injected config or tool runtime). Follow what
  `features/memory/tools.py` and the skills tools already do.
- That the local `FilesystemBackend` in virtual mode and the `CompositeBackend`
  both implement `download_files` for the paths the agent uses.
- That subagents can call `send_file` only if you want them to. Default: main
  agent only (it is in `tools`, not `lookups`).

## Verified on 2026-11-12 (langchain 1.4.9, langchain-core 1.7.6, deepagents 0.7.34)

Checked with a throwaway probe through `build_agent` and the test fake model:

- **Run config in a tool.** Existing tools (memory, skills) take identity from a
  closure and never read the run config. A tool declared with
  `@tool(response_format="content_and_artifact")` and a `runtime: ToolRuntime`
  parameter (from `langchain.tools`) gets it injected and hidden from the model:
  the tool's schema shows only `path`. `runtime.config["configurable"]["thread_id"]`
  is the `["user", "chat"]` JSON pair from `thread_id()`. Use this; the user also
  comes from the closure (the runtime's user), and the two must agree.
- **Structured payload.** With `content_and_artifact`, the tool returns
  `(json_text, dict)`; the dict arrives on `ToolMessage.artifact`. `stream_once`
  reads the artifact, so no parsing of the content is needed (keep the JSON-content
  fallback out unless a version change requires it).
- **download_files.** `FilesystemBackend(virtual_mode=True)` and `CompositeBackend`
  both return `FileDownloadResponse(path, content, error)`; a missing file comes
  back with `error="file_not_found"` and no content (not an exception).
- **Main agent only — correction.** Being in `tools` (not `lookups`) does *not*
  keep `send_file` from subagents: the general-purpose subagent and plan/room
  workers (`worker_factory`) get the main agent's tools, minus `MAIN_ONLY`. Add
  `send_file` to `MAIN_ONLY` in `core/assembly.py` (with `run_plan`, `start_room`).
  Plan workers also run on their own thread ids (`<thread>:plan:...`), which would
  file their outputs under a different folder.

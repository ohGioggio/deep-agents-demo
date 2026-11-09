<p align="center">
  <a href="https://github.com/ohGioggio/deep-agents-demo"><img src="assets/banner.svg" width="900" alt="Deep Agents Demo: an agent harness that plans 3D city scenes"/></a>
</p>

<p align="center">
  <a href="https://github.com/ohGioggio/deep-agents-demo/actions/workflows/ci.yml"><img src="https://github.com/ohGioggio/deep-agents-demo/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.11+"/></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-managed-de5fe9?style=flat&logo=uv&logoColor=white" alt="uv"/></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat" alt="License: MIT"/></a>
  <a href="https://github.com/langchain-ai/deepagents"><img src="https://img.shields.io/badge/Deep%20Agents-0.7.34-1c3c3c?style=flat&logo=langchain&logoColor=white" alt="Deep Agents 0.7.34"/></a>
  <a href="https://github.com/mem0ai/mem0"><img src="https://img.shields.io/badge/Memory-Mem0-4f46e5?style=flat" alt="Mem0"/></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Vectors-Qdrant-dc244c?style=flat" alt="Qdrant"/></a>
  <a href="https://www.daytona.io/"><img src="https://img.shields.io/badge/Sandbox-Daytona-0b7285?style=flat" alt="Daytona"/></a>
</p>

<p align="center"><b>Planning, skills, subagents, memory, RAG and a sandbox, in about 500 lines of Python.</b><br/>
A small, readable reference for building on <a href="https://github.com/langchain-ai/deepagents">LangChain Deep Agents</a>.</p>

Deep Agents Demo is a command-line agent that **plans 3D city scenes** from a
fictional asset catalog. Ask for a scene under a triangle budget and watch it
**write a to-do plan, delegate research to a subagent, check the budget and save
a proposal**, with every step streamed to your terminal.

- **Every harness feature in one place.** Planning, skills, subagent delegation,
  streaming, checkpoints and context management, wired together and visible.
- **Integrations you can switch off.** Mem0 memory, Qdrant RAG and a Daytona
  sandbox are each one setting away from disabled.
- **Any OpenAI-compatible model.** Point it at OpenAI, a local server or NVIDIA
  NIM. Embeddings can use a different provider from chat.

> [!NOTE]
> The agent produces scene **proposals** (asset IDs, counts and triangle totals),
> not 3D geometry. The catalog is fictional, so the demo needs only a model API key.

**Get started** (30 seconds).

```bash
git clone https://github.com/ohGioggio/deep-agents-demo.git
cd deep-agents-demo
uv sync --locked
cp .env.example .env            # then set OPENAI_API_KEY and AGENT_MODEL
uv run deep-agents-demo
```

```text
deep-agents-demo/
├── src/deep_agents_demo/   # the agent, tools and integrations
├── tests/                  # offline unit tests
├── docs/                   # documents for RAG
├── data/                   # memory and document vectors (created on first run)
└── demo_workspace/         # agent files and the seeded skill (created on first run)
```

---

## See it in action

Start a chat and ask for a scene:

```text
You: Plan a city scene with trees, cars and buildings under 100000 triangles.
```

Each update is labelled with the agent that produced it, so you can follow the
plan, the delegation and the tool calls as they happen:

```text
[main / model]          the main agent reads the scene-planning skill
[plan]                  the to-do list it writes, updated as tasks complete
[subagent / tools]      asset-researcher searches the catalog
[main / tools]          triangle_budget checks the total
[main / tools]          the proposal is saved to /scene_plan.md
```

Then say `Remember that I prefer pine trees.` and start a new thread: the
preference is still there.

## What it does

| Capability | What you get |
| --- | --- |
| **Planning** | A `write_todos` list, streamed as `[plan]` updates |
| **Skills** | A seeded `scene-planning` skill the agent follows step by step |
| **Delegation** | An `asset-researcher` subagent that searches the catalog |
| **Tools** | `triangle_budget`, `search_assets`, `search_docs`, `remember`, `recall` |
| **Persistence** | Conversations resume by thread from `demo_checkpoints.sqlite` |
| **Context management** | Built-in summarization and large-output offloading |
| **Long-term memory** | Per-user Mem0 memories that survive across threads |
| **RAG** | Cited search over your own `.md` and `.txt` files |
| **Sandbox** | File editing and shell commands in a remote Daytona sandbox |

> [!TIP]
> Set `MEMORY_ENABLED=false` and `RAG_ENABLED=false` to see the bare harness with
> nothing but the catalog tools.

---

## Prerequisites

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Python 3.11+ | Everything | uv can install it for you |
| [uv](https://docs.astral.sh/uv/) | Everything | Manages the environment and lockfile |
| Chat model with tool calling | Everything | Any OpenAI-compatible endpoint |
| Embedding model | Memory and RAG | OpenAI or NVIDIA NIM |
| Daytona API key | Sandbox only | Optional |

**Install uv (macOS / Linux):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install uv (Windows PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install

> [!IMPORTANT]
> Keep `.env` out of version control. It is already listed in `.gitignore`.

```bash
uv sync --locked
cp .env.example .env            # PowerShell: Copy-Item .env.example .env
```

Fill in `OPENAI_API_KEY` and `AGENT_MODEL`. Everything else has a default.

<details>
<summary><b>All configuration options</b></summary>

Settings are read from `.env` in the project root. Variables already set in
your shell take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | API key for the chat model |
| `AGENT_MODEL` | required | Chat model ID |
| `OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` |
| `MEMORY_ENABLED` | `true` | Enable Mem0 long-term memory |
| `MEMORY_MODEL` / `MEMORY_API_KEY` / `MEMORY_BASE_URL` | agent settings | Model used to extract memories |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for memory and RAG |
| `EMBEDDING_PROVIDER` | `auto` | `auto`, `openai` or `nvidia` |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` | OpenAI | Embedding endpoint, independent of the chat endpoint |
| `EMBEDDING_DIMENSIONS` | `1536` | Must equal the model's real output width |
| `RAG_ENABLED` | `true` | Enable the `search_docs` tool |
| `SANDBOX_PROVIDER` | `none` | `none` (local files, no shell) or `daytona` |
| `DAYTONA_API_KEY` | | Daytona credentials |
| `DAYTONA_SANDBOX_ID` | | Reuse a specific running sandbox |

</details>

<details>
<summary><b>Using a local model</b></summary>

Point the chat endpoint at any OpenAI-compatible server that supports tool calling:

```dotenv
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=local
AGENT_MODEL=your-local-model
```

Embeddings still go to `EMBEDDING_BASE_URL`, which defaults to OpenAI. Set it as
well, or disable memory and RAG, to keep everything local.

</details>

---

## Common commands

```bash
uv run deep-agents-demo                                  # interactive chat
uv run deep-agents-demo --thread second                  # new conversation, same memories
uv run deep-agents-demo --user alice                     # separate memory namespace
uv run deep-agents-demo --prompt "Plan a small park"     # one prompt, then exit
uv run deep-agents-demo --index-docs docs                # rebuild the RAG index
uv run python -m deep_agents_demo --help                 # same CLI, as a module
uv run python -m unittest discover -s tests -v           # offline tests
```

| Option | Default | Description |
| --- | --- | --- |
| `--thread` | `city-demo` | Conversation ID; reuse it to resume history |
| `--user` | `demo-user` | Mem0 namespace, not authentication |
| `--prompt` | | Run a single prompt instead of interactive chat |
| `--index-docs DIR` | | Rebuild the RAG index from `DIR`, then exit |

## Searching your documents (RAG)

Put UTF-8 `.md` and `.txt` files in `docs/`, index them, then ask:

```bash
uv run deep-agents-demo --index-docs docs
uv run deep-agents-demo --prompt "Search the documents for the default triangle budget and cite the source."
```

- **Full rebuilds.** Indexing replaces the whole collection, so chunks from
  deleted files disappear. Empty input leaves the existing index untouched.
- **Cited results.** Each search returns four passages with source, chunk and score.
- **Text only.** PDF and other binary formats are not supported.
- **One process at a time.** A local Qdrant directory can be opened by only one
  process, so don't index while a chat is running.

## Long-term memory

```bash
uv run deep-agents-demo --user alice --thread first
```

Say `Remember that I prefer pine trees and use uv for Python.`, quit, then:

```bash
uv run deep-agents-demo --user alice --thread second
```

Say `Recall my preferences and plan a city under 100000 triangles.`

- **Per user, across threads.** The same user and thread also resume the
  conversation history.
- **Tool-driven.** The agent calls `recall` when it needs a preference; memories
  are not injected into every turn.
- **Not authentication.** `--user` is a namespace. RAG, files and the sandbox are
  shared: this is a single-user demo, not a multi-tenant app.

## Running code in a sandbox

```dotenv
SANDBOX_PROVIDER=daytona
DAYTONA_API_KEY=your-key
```

The agent gets file tools and shell execution in one remote Daytona sandbox. It
uses `DAYTONA_SANDBOX_ID` if set, otherwise the ID saved in
`data/daytona_sandbox_id.txt`, otherwise it creates a sandbox and saves its ID.
Commands time out after 120 seconds.

Try:

- `Create /home/daytona/hello.py and execute it.`
- `Clone <repo URL> into /home/daytona/project, inspect it and run the tests.`
- `Retrieve the requirements with search_docs and implement them in the repository.`

> [!WARNING]
> Sandboxes are kept after the program exits. Stop and delete them in Daytona to
> avoid charges, and export anything you need first.

- **Remote by default.** Only `/instructions/` maps to the local skills.
- **Bring your runtimes.** The sandbox image needs Git, uv or whatever the task
  requires. Private Git access and network rules are configured in Daytona.
- **Docs stay local.** `search_docs` queries the index on your machine; documents
  are not uploaded.

## NVIDIA NIM embeddings

```dotenv
EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-8b
EMBEDDING_PROVIDER=nvidia
EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
EMBEDDING_DIMENSIONS=4096
```

- **Raw strings, float vectors.** Never OpenAI token-ID arrays.
- **Passage and query modes.** `input_type=passage` when indexing or saving
  memories, `input_type=query` when searching.
- **Batches of 16.** At most 16 strings per request.
- **Auto-detection.** `EMBEDDING_PROVIDER=auto` recognises
  `integrate.api.nvidia.com`; set `nvidia` explicitly for self-hosted NIM.

---

## Project layout

```text
src/deep_agents_demo/
├── cli.py          # command-line entry point and chat loop
├── agent.py        # deep agent, system prompt and asset-researcher subagent
├── catalog.py      # fictional asset catalog and its tools
├── config.py       # project paths and environment settings
├── workspace.py    # seeds the starter memory file and skill
├── memory.py       # Mem0 long-term memory and remember/recall tools
├── rag.py          # Qdrant document index and the search_docs tool
├── embeddings.py   # OpenAI / NVIDIA-compatible embedding client
└── sandbox.py      # local or Daytona filesystem backend
```

## Where data is stored

| Path | Contents |
| --- | --- |
| `demo_checkpoints.sqlite` | Conversation checkpoints |
| `data/mem0_history.sqlite` | Mem0 change history |
| `data/memory_vectors/` | Mem0 Qdrant index |
| `data/document_vectors/` | RAG Qdrant index |
| `data/mem0_internal/` | Auxiliary Mem0 data, if created |
| `data/daytona_sandbox_id.txt` | Saved sandbox ID |
| `demo_workspace/` | Agent files and the seeded skill |

All of these, and `.env`, are excluded by `.gitignore`.

## Privacy

- **Local storage.** Checkpoints, memories and vectors stay on your machine.
- **Remote inference.** Chat and embedding requests go to the providers you
  configure. Use a local endpoint to keep them on your machine.
- **No telemetry.** Mem0 telemetry is disabled by default.
- **Untrusted content.** The agent is told to treat documents and recalled
  memories as data, not instructions.

## Troubleshooting

- **"Fill in OPENAI_API_KEY and AGENT_MODEL".** `.env` is missing or still has
  the placeholder values. It must sit in the project root.
- **"No documents indexed".** Run `uv run deep-agents-demo --index-docs docs`.
- **Qdrant storage is already in use.** Another process has `data/` open. Close
  the other chat or indexing run.
- **"Returned embedding width differs from EMBEDDING_DIMENSIONS".** Set the real
  width of your model. If the Mem0 index was created with the wrong size, stop
  the app, back up `data/memory_vectors/` and move it aside; a fresh index is
  created on the next start, without the old memories.
- **Switched embedding models.** Rebuild RAG with `--index-docs docs` and recreate
  the Mem0 index. Never mix embeddings from different models.
- **URLs not loading from `.env`.** Lines must not end with a backslash or contain
  escaped URLs such as `https\://`.
- **Daytona sandbox won't attach.** Start a stopped sandbox in Daytona before
  reusing it.

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

The tests run offline. They use a real local Qdrant with fake embeddings to
cover indexing, retrieval, reindexing and persistence, check per-user memory
scoping and the catalog tools, and verify NVIDIA request bodies against a mocked
HTTP endpoint. Live model calls and Daytona execution are not covered.

## Contributing

Issues and pull requests are welcome. Run the tests before opening a PR, and keep
the demo small: it is meant to be read in one sitting.

## License

Released under the [MIT License](LICENSE.md). © 2026 Gioele Allievi.

---

<p align="center">
  <a href="https://github.com/ohGioggio/deep-agents-demo"><img src="https://img.shields.io/badge/GitHub-deep--agents--demo-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://github.com/ohGioggio/deep-agents-demo/issues"><img src="https://img.shields.io/badge/Issues-welcome-0b7285?style=flat" alt="Issues welcome"/></a>
  <a href="https://github.com/langchain-ai/deepagents"><img src="https://img.shields.io/badge/Built%20with-Deep%20Agents-1c3c3c?style=flat&logo=langchain&logoColor=white" alt="Built with Deep Agents"/></a>
</p>

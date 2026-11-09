# Deep Agents Demo

A small, self-contained example of a [Deep Agents](https://github.com/langchain-ai/deepagents)
harness that plans 3D city scenes from a fictional asset catalog. It shows
planning, skills, subagent delegation, streaming and SQLite checkpoints, and
adds three optional integrations:

- **Long-term memory**: [Mem0](https://github.com/mem0ai/mem0) backed by a local Qdrant index
- **RAG**: a separate document index over your own `.md` / `.txt` files
- **Sandbox**: remote code execution in a [Daytona](https://www.daytona.io/) sandbox

The agent produces scene *proposals* (asset IDs, counts, triangle totals). It
does not generate 3D geometry.

## Features

| Capability | How it is shown |
| --- | --- |
| Planning | `write_todos` to-do lists, streamed as `[plan]` updates |
| Skills | A seeded `scene-planning` skill in `demo_workspace/skills/` |
| Delegation | An `asset-researcher` subagent that searches the catalog |
| Tools | `triangle_budget`, `search_assets`, `search_docs`, `remember` / `recall` |
| Persistence | Conversation checkpoints in `demo_checkpoints.sqlite` |
| Context management | Built-in Deep Agents summarization and output offloading |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible chat model that supports tool calling
- An embedding model (only needed for memory and RAG)
- A Daytona API key (only needed for the sandbox)

## Quick start

```bash
uv sync --locked
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `AGENT_MODEL` in `.env`, then run:

```bash
uv run deep_agents_demo.py
```

Try: `Plan a city scene with trees, cars and buildings under 100000 triangles.`
Type `exit` to quit.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--thread` | `city-demo` | Conversation thread ID; reuse it to resume history |
| `--user` | `demo-user` | Mem0 namespace (not authentication) |
| `--prompt` | | Run a single prompt instead of interactive chat |
| `--index-docs DIR` | | Rebuild the RAG index from `DIR`, then exit |

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example)). Variables
already set in your shell take precedence over `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | API key for the chat model |
| `AGENT_MODEL` | required | Chat model ID |
| `OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint (e.g. `http://localhost:8000/v1`) |
| `MEMORY_ENABLED` | `true` | Enable Mem0 long-term memory |
| `MEMORY_MODEL` / `MEMORY_API_KEY` / `MEMORY_BASE_URL` | agent settings | Model used for memory extraction |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for Mem0 and RAG |
| `EMBEDDING_PROVIDER` | `auto` | `auto`, `openai` or `nvidia` |
| `EMBEDDING_API_KEY` | `OPENAI_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` | OpenAI | Embedding endpoint, independent of the chat endpoint |
| `EMBEDDING_DIMENSIONS` | `1536` | Must equal the model's real output width |
| `RAG_ENABLED` | `true` | Enable the `search_docs` tool |
| `SANDBOX_PROVIDER` | `none` | `none` (local files, no shell) or `daytona` |
| `DAYTONA_API_KEY` | | Daytona credentials |
| `DAYTONA_SANDBOX_ID` | | Reuse a specific running sandbox |

Set `MEMORY_ENABLED=false` and `RAG_ENABLED=false` for the minimal mode.

> Changing the embedding model means you must reindex RAG and recreate the Mem0
> vector index. Do not mix embeddings from different models.

## RAG

Put UTF-8 `.md` and `.txt` files in `docs/`, then index and query them:

```bash
uv run deep_agents_demo.py --index-docs docs
uv run deep_agents_demo.py --prompt "Search the documents for the default triangle budget and cite the source."
```

- Indexing replaces the whole collection, so stale chunks are removed.
- Empty input leaves the existing index untouched.
- Each search returns four passages with source, chunk number and score.
- PDF and other binary formats are not supported.
- Do not index and chat at the same time: a local Qdrant directory can only be
  opened by one process.

## Long-term memory

```bash
uv run deep_agents_demo.py --user alice --thread first
```

Say `Remember that I prefer pine trees and use uv for Python.`, quit, then start
a new thread:

```bash
uv run deep_agents_demo.py --user alice --thread second
```

Say `Recall my preferences and plan a city under 100000 triangles.`

- Memories persist per user across threads. The same user and thread also
  resume the conversation history.
- Recall is tool-driven; memories are not injected into every turn.
- `--user` is a namespace, not authentication. RAG, files and the sandbox are
  shared: this is a single-user demo, not a multi-tenant app.
- Preferences in `AGENTS.md` are not migrated to Mem0 automatically.

## Sandbox (Daytona)

```dotenv
SANDBOX_PROVIDER=daytona
DAYTONA_API_KEY=your-key
```

The agent gets filesystem tools and shell execution in one remote sandbox. The
sandbox is chosen in this order: `DAYTONA_SANDBOX_ID`, the ID saved in
`data/daytona_sandbox_id.txt`, or a newly created sandbox (whose ID is then
saved). Commands time out after 120 seconds.

Example requests:

- `Create /home/daytona/hello.py and execute it.`
- `Clone <repo URL> into /home/daytona/project, inspect it and run the tests.`
- `Retrieve the requirements with search_docs and implement them in the repository.`

Things to know:

- Sandboxes are kept after the program exits. Stopping, deleting and billing
  are managed in Daytona; start a stopped sandbox there before reusing it.
- The sandbox image needs whatever runtimes the task requires (Git, uv, ...).
  Private Git access and network rules are configured in Daytona.
- Code paths are remote; only `/instructions/` maps to the local skills.
- Local documents are not uploaded. `search_docs` queries the host index.

## NVIDIA NIM embeddings

The embedding adapter is compatible with NVIDIA NIM:

- It sends raw strings, never token-ID arrays, and requests float vectors.
- It uses `input_type=passage` when indexing or writing memories and
  `input_type=query` when searching.
- It sends at most 16 strings per request.
- `EMBEDDING_PROVIDER=auto` detects `integrate.api.nvidia.com`. Set
  `EMBEDDING_PROVIDER=nvidia` explicitly for self-hosted NIM.

```dotenv
EMBEDDING_MODEL=nvidia/nemotron-3.5-embed-8b
EMBEDDING_PROVIDER=nvidia
EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
EMBEDDING_DIMENSIONS=4096
```

If a Mem0 index was created with the wrong dimension, stop the app and back up
`data/memory_vectors/`. Moving that folder aside makes the demo create a fresh
index on the next start, but old memories will not be in it. Rebuild document
vectors with `--index-docs docs`. `.env` lines must not end with a backslash or
contain escaped URLs such as `https\://`.

## Storage

| Path | Contents |
| --- | --- |
| `demo_checkpoints.sqlite` | Conversation checkpoints |
| `data/mem0_history.sqlite` | Mem0 change history |
| `data/memory_vectors/` | Mem0 Qdrant index |
| `data/document_vectors/` | RAG Qdrant index |
| `data/mem0_internal/` | Auxiliary Mem0 data, if created |
| `data/daytona_sandbox_id.txt` | Saved sandbox ID |
| `demo_workspace/` | Local agent files and the seeded skill |

All of these are excluded by `.gitignore`, along with `.env`. Mem0 telemetry is
disabled by default. Storage is local, but model and embedding requests still
go to the providers you configure.

## Project layout

| File | Purpose |
| --- | --- |
| `deep_agents_demo.py` | CLI entry point, catalog tools, agent and subagent setup |
| `integrations.py` | Mem0 memory, Qdrant document index and Daytona backend |
| `embedding_adapter.py` | OpenAI / NVIDIA-compatible embedding client |
| `test_integrations.py` | Offline tests |
| `docs/` | Example documents for RAG |

## Testing

```bash
uv run python -m unittest -v test_integrations
```

The tests run offline. They use a real local Qdrant with fake embeddings to
cover indexing, retrieval, reindexing, persistence and per-user memory scoping.
A mocked HTTP endpoint checks the NVIDIA request bodies (passage/query modes and
dimension-mismatch detection). Live model calls and Daytona execution are not
covered.

## License

[MIT](LICENSE.md) © 2026 Gioele Allievi

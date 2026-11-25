# Same-reply agent race, and memory candidates from Mem0 search — design (short)

Two fixes from the v0.3.0 follow-ups, agreed 2026-11-25.

## 1. `create_agent` and `task` in the same reply

### Why

When the model creates an agent and hands it a task in one reply, the user approves `create_agent`,
and then both tool calls run at the same time. `task` (`NewAgentDelegation._target` in
`core/delegation.py`) checks the newest worker factory's names before `create_agent` has saved the
agent. It then falls through to Deep Agents, which answers "does not exist". The model recovers with
`list_agents` and then `task`, but that is one confusing error and an extra round trip.

### Decided (user picked A: wait)

- `task` names an agent the main agent wasn't built with, and the newest factory doesn't know it.
  `NewAgentDelegation` then looks at the other tool calls of the same AI message (the last
  `AIMessage` in the request's state).
- If one of them creates that name, `task` waits for it: up to 10 s, checking every 0.1 s.
  "Creates" means `create_agent` with `name`, `import_agent`, or `install_agent` with `name`
  (`import_agent` and `install_agent` are my call; same race, same fix).
  - When the name appears, the call runs as today, on the sync and the async paths.
  - When it doesn't (the create failed or was too slow), the result is an error:
    "<name> was being created in this same step but isn't available yet; call list_agents, then task."
- A rejected `create_agent` never runs. If its call is still in the message, `task` waits the 10 s
  and then answers with the same error. That costs 10 s once, which is acceptable.
- No sibling create in the message → no wait; behaviour unchanged.
- Constants `NEW_AGENT_WAIT = 10.0` and `NEW_AGENT_POLL = 0.1`, so tests can shorten them.

### Tests

- The agent appears during the wait (a factory whose names change after a short delay): the
  delegated run happens, sync and async.
- The agent never appears: the error text above, and the wait is bounded.
- No sibling create: the call goes straight to the handler (Deep Agents' own error), with no wait.
- A sibling create for a different name: no wait.

## 2. Memory candidates from Mem0 search

### Why

`remember` loaded all of the user's memories (`get_all`, capped at 10,000) and ranked them with
LangChain's `BM25Retriever`. That design has three problems:

- BM25 needs the whole set in memory.
- Above 10,000 memories it silently cuts off.
- It misses a change that shares no words with the old memory ("I moved to Milan" against
  "Lives in Rome").

Mem0's own `search` is vector search in Qdrant, filtered by user, top-k. It is the same path
automatic recall uses.

### Decided (user: "do it", switch to Mem0 search, drop BM25)

- **Candidates.** `MemoryService.remember` gets its candidates from
  `memory.search(fact, filters={"user_id": ...}, top_k=UPDATE_CANDIDATES, threshold=0.0)`, redacted
  as today.
  - `threshold=0.0` is my call: the decider should always see the 5 nearest, even weak matches.
  - No candidates → ADD without a model call, as today.
  - `_every_memory` and `ALL_MEMORIES_LIMIT` go.
- **Dropped.** `rank_candidates`, `_words`, `_bm25_retriever`, and the dependencies
  `langchain-community` and `rank-bm25`.
  - Tests that only covered BM25 go too, including the subprocess import and sunset-warning tests.
  - The startup "does not import langchain_community" test stays: it should still hold.
- **Half-applied update.** If `memory.update` raises, read the memory back (`memory.get(id)`).
  - It belongs to this user and its text already equals the new text: return `updated`, with a
    note that Mem0 reported an error after saving.
  - Otherwise: ADD the fact with the note, as today.
  - If the read-back raises too: ADD as today.
- **Prompt.** Unchanged, apart from wording that says "most similar memories" instead of BM25.

### Evals

- **Decision part.** It tests the decider, not retrieval. Each scenario's seeded memories (5 or
  fewer) are all shown as candidates, in seeded order, replacing `rank_candidates`.
- **Before/after part.** It runs through the real service, so it now uses Mem0 search. It gets one
  new sequence, `paraphrase-move`: 6 unrelated memories plus "Lives in Rome", then "I relocated to
  Milan" (no shared word with the old memory). Expected final: Milan, not Rome.
- **Scoring.** Drop the bare `gave` from `PAST` (keep `gave away`). A test shows "Gave a talk at
  PyCon" is not treated as history.
- **Re-run.** `--runs 3` live, then the README table is updated (with the BM25 mentions removed).

### README

The Memory section says candidates come from Mem0's vector search (this user only, top 5), and
drops BM25 and its dependencies.

## Approaches considered

- Race: A, wait for the sibling create (chosen); B, answer "call task again" at once (an extra
  model call every time).
- Candidates:
  - BM25Retriever over `get_all` (v0.3.0): load-all, a 10,000 cap, misses paraphrases.
  - Our own SQLite FTS5 index: a second store to keep in sync.
  - Mem0 hybrid search: needs `fastembed` plus onnxruntime, and existing collections lack the
    keyword slot.
  - **Mem0 vector search (chosen).**

## Not in scope, noted for later

Hybrid (BM25 + vector) search for documents (`search_docs`). That is where keyword matching would
help: exact error codes, identifiers, version numbers. It would be its own feature with its own
eval.

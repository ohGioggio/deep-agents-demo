# Memory updates instead of duplicates — design (short)

## Why

mem0 2.x extracts memories add-only: `Memory.add` only ever produces `ADD`. When something the
user said before changes ("I moved to Milan" after "lives in Rome"), a second memory is added
next to the old one, and recall then returns both. Memory should stay one clean set of facts.

## What changes

`remember` (the only write path; the agent calls it when the user asks to save something) gets
a decision step before anything is stored:

1. **Context.** The tool reads the last 6 messages of the conversation (`ToolRuntime` state, as
   `send_file` does) and passes them with the fact.
2. **Candidates.** The service loads the user's memories (`get_all`, this user only) and ranks
   them against the fact with LangChain's `BM25Retriever` (top 5).
3. **Decision.** One call to the memory model (`MEMORY_MODEL` / `MEMORY_API_KEY` /
   `MEMORY_BASE_URL`) with our own prompt, structured output into a Pydantic `Operation`:
   - `kind`: `ADD` | `UPDATE` | `DELETE` | `NONE`
   - `memory_id`: the candidate it applies to (UPDATE, DELETE)
   - `text`: the final memory text (ADD, UPDATE)
   - `reason`: one line, why
4. **Apply.**
   - `ADD` → `memory.add(text, user_id, infer=False)` (mem0 stores it as is; its own
     extraction call is no longer made).
   - `UPDATE` → `memory.update(memory_id, text)`.
   - `NONE` → nothing stored; the result says it is already remembered (with the id).
   - `DELETE` → nothing deleted here. The result names the memory and tells the agent to call
     `forget(id)`, so a deletion still goes through the user's approval, as today.

The tool result says what happened (`added` / `updated` from → to / `already known` /
`to forget`), with the reason, so the agent can tell the user.

## Rules

- The secrets check runs on the fact before anything (as today) and again on `text`; a
  secret-looking `text` is refused and nothing is written.
- `memory_id` must be one of the candidates shown to the model; anything else is treated as
  `ADD` of the fact (never touch a memory the model wasn't shown, or another user's).
- If the decision call fails or returns something invalid, fall back to `ADD` of the fact:
  a duplicate is better than a lost fact. The result says so.
- No candidates (no memories yet) → `ADD` without a model call.
- The decision call goes through a LangChain chat model with the app's usage counter, so it is
  counted in run history (mem0's own calls were not).

## Approaches considered

- **A — decision step inside `MemoryService.remember`** (chosen by the user): everything stays in
  the one class that talks to mem0; easy to test with a fake model.
- B — A plus mem0's vector search for candidates (catches paraphrases BM25 misses, e.g.
  "moved to Milan" vs "lives in Rome"); not chosen.
- C — configure mem0's own update step: not possible, mem0 2.x removed it.

## Decided

- Trigger: only when `remember` is called (user's choice); no automatic extraction per turn.
- Ranking: LangChain `BM25Retriever` (user's choice). Adds dependencies `langchain-community`
  and `rank-bm25`.
- `DELETE` is a kind (user's choice); carrying it out goes through `forget` and its approval
  (my call, so `remember` can't bypass the approval).

## Left to the builder (my calls)

- 6 recent messages, top 5 candidates; constants next to `RECALL_TOP_K`.
- Prompt wording; `with_structured_output(Operation)` with the provider's tool calling.
- Tests with a fake model and a fake mem0: each kind, an unknown id, a secret in `text`, a failed
  decision, no candidates, and that only this user's memories are candidates.
- README: the Memory section says how updates work.

## Evals and benchmarks (asked by the user, after the build)

`evals/memory_updates.py`, live (paid calls), like `evals/orchestration.py`, `--runs N` with
`mean (min–max)`:

- **Decision eval.** Scenarios, each: seeded memories, recent messages, a fact, and the expected
  `kind` (+ the expected `memory_id` for UPDATE/DELETE). About 20, covering every kind: a changed
  fact (UPDATE), a repeat in other words (NONE), a new unrelated fact (ADD), a retraction
  (DELETE), and traps (a similar but different fact must ADD, not UPDATE; a fact about someone
  else). Score: kind accuracy, target accuracy.
- **Before/after on real mem0.** In a temporary data dir: the same sequences of `remember` calls
  through the old path (`add` with mem0's extraction) and the new one. Measure the final number
  of memories, duplicates/contradictions left (scored by the expected final set), model calls,
  tokens, and seconds per `remember` (p50/p95).
- README: a table with the results in the Memory section, like "Evaluating orchestration".

## Not in scope

Automatic memory extraction after turns; vector or hybrid ranking; editing mem0's history.

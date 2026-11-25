# Same-reply agent race and Mem0-search memory candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `task` waits for an agent being created in the same reply. `remember` takes its candidates from Mem0 vector search instead of BM25 over every memory.

**Architecture:**
- Task 1 changes `NewAgentDelegation` (`core/delegation.py`) only.
- Task 2 changes `MemoryService.remember` and `features/memory/updates.py`, and removes the BM25 code and its two dependencies.
- Task 3 adapts the live eval (`evals/memory_updates.py`, `evals/memory_update_scenarios.py`) to the removed ranking and adds a paraphrase sequence.
- Tasks 1 and 2 are independent. Task 3 needs Task 2.

**Tech Stack:** Python 3.13, LangChain/LangGraph middleware, Mem0 2.2.1 (Qdrant), unittest. Tests run with `uv run --no-sync python -m unittest discover -s tests`.

**Spec:** `design/specs/2026-11-25-race-and-memory-search-design.md`

## Global Constraints

- Commit messages: ONE line, no trailer (never `Co-Authored-By`).
- Never read, print or edit `.env`.
- Tests are offline (no paid calls); the full suite must stay green: `uv run --no-sync python -m unittest discover -s tests`.
- `NEW_AGENT_WAIT = 10.0` (seconds) and `NEW_AGENT_POLL = 0.1` (seconds), module constants in `core/delegation.py`.
- Timeout error text, exactly: `f"Error: {name} was being created in this same step but isn't available yet; call list_agents, then task."`
- Creating calls: `create_agent` with `name`, `import_agent`, `install_agent` with `name`.
- Candidates: `memory.search(fact, filters={"user_id": self.user_id}, top_k=UPDATE_CANDIDATES, threshold=0.0)`; `UPDATE_CANDIDATES = 5`.
- Removed: `rank_candidates`, `_words`, `_bm25_retriever`, `_every_memory`, `ALL_MEMORIES_LIMIT`, the dependencies `langchain-community` and `rank-bm25`.
- The note for a half-applied update, exactly: `"Mem0 reported an error after saving the update; the memory has the new text"`.

## Review Focus

1. A `task` call for a name created by a sibling `import_agent`: `import_agent` takes a `path`, not a `name`. The wait must still trigger (any sibling `import_agent` counts), and it must not crash on missing args. Test in Task 1.
2. Several `task` calls for the same new agent in one reply: each waits and each runs. Neither hangs past `NEW_AGENT_WAIT`. Test in Task 1.
3. Mem0 search returns a hit owned by another user (a filter bug): it must never become a candidate. Keep only hits whose `user_id` is absent or equals `self.user_id`. Test in Task 2.
4. Read-back after a failed update returns another user's memory, or `None`: ADD the fact, never report `updated`. Test in Task 2.
5. Mem0 search raises (embedding provider down): `remember` must still store the fact. ADD with a note, as a failed decision does. Test in Task 2.

---

### Task 1: `task` waits for an agent created in the same reply

**Files:**
- Modify: `src/deep_agents_demo/core/delegation.py` (`NewAgentDelegation._target`, `wrap_tool_call`, `awrap_tool_call`; new constants and helpers)
- Test: `tests/test_delegation.py` (new class `SameReplyCreate`)

**Interfaces:**
- Consumes: the request's state (`request.state["messages"]`). The last `AIMessage` there holds the reply's `tool_calls`.
- Produces: `NEW_AGENT_WAIT`, `NEW_AGENT_POLL`, `_created_in_reply(state, name) -> bool`, `_not_ready(name) -> str` (the timeout text from Global Constraints).

- [ ] **Step 1: Write the failing tests** in `tests/test_delegation.py`, class `SameReplyCreate`.
  - Build the request objects directly: a `SimpleNamespace(tool_call=..., state={"messages": [AIMessage(content="", tool_calls=[...])]})`, plus a `newest` callable whose factory `names` change over time.
  - Patch `delegation.NEW_AGENT_WAIT` to `0.5` and `delegation.NEW_AGENT_POLL` to `0.02`.
  - `test_waits_for_an_agent_created_in_the_same_reply`: the factory's names gain `"sweeper"` after 0.1 s, from a `threading.Timer`. `_target` returns `(make, "sweeper")`.
  - `test_gives_up_with_a_clear_error`: the names never gain it. `wrap_tool_call` returns a `ToolMessage` with `status == "error"` and exactly the timeout text. Elapsed time is `< 1.0`, and the handler is not called.
  - `test_no_sibling_create_goes_straight_to_the_handler`: elapsed `< 0.1`, and the handler is called once.
  - `test_a_create_for_another_name_does_not_wait`: the sibling `create_agent` has `name="other"`. The handler is called, elapsed `< 0.1`.
  - `test_a_sibling_import_agent_counts`: the sibling is `import_agent` with only `path`. It waits, and the agent is found when the names gain it.
  - `test_two_tasks_for_the_same_new_agent_both_find_it`: two threads call `_target` for `"sweeper"`. Both get it.
  - `test_async_path_waits_too`: `awrap_tool_call` with names gaining `"sweeper"` after 0.1 s. The run happens through the stub factory: the `make` used returns an agent stub whose `aget_state` and `ainvoke` produce one `AIMessage("done")`, and the result content is `"done"`. The waiting uses `asyncio.sleep`, not `time.sleep`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run --no-sync python -m unittest tests.test_delegation -v`
  Expected: the new tests FAIL (no wait, and no `NEW_AGENT_WAIT` attribute). The existing tests pass.

- [ ] **Step 3: Implement.**
  - `_created_in_reply(state, name)`: true when the last `AIMessage` in `state["messages"]` has a tool call that is either `create_agent`/`install_agent` with `args["name"] == name`, or any `import_agent`.
  - `_target(request)` (sync) polls `self.newest()` every `NEW_AGENT_POLL` until the name is in `make.names`, or `NEW_AGENT_WAIT` passes. It does this only when the name is unknown and `_created_in_reply` holds.
  - An async twin `_atarget` does the same with `await asyncio.sleep`.
  - Return a distinct sentinel for "waited and timed out", so the wrappers return `ToolMessage(_not_ready(name), status="error")` instead of calling the handler.

- [ ] **Step 4: Run the tests and verify they pass**

  Run: `uv run --no-sync python -m unittest tests.test_delegation -v`
  Expected: all PASS.

- [ ] **Step 5: Run the full suite**

  Run: `uv run --no-sync python -m unittest discover -s tests`
  Expected: OK.

- [ ] **Step 6: Commit**

  `git commit -am "Delegation: task waits for an agent created in the same reply"` (add the test file first if new tests went to a new file).

---

### Task 2: Memory candidates from Mem0 search; drop BM25

**Files:**
- Modify: `src/deep_agents_demo/features/memory/service.py` (`remember`; remove `_every_memory`, `ALL_MEMORIES_LIMIT`)
- Modify: `src/deep_agents_demo/features/memory/updates.py` (remove `_words`, `_bm25_retriever`, `rank_candidates`, and their imports; keep `UPDATE_CANDIDATES = 5`, `RECENT_MESSAGES`, `Operation`, `DECISION_PROMPT`, `MemoryDecider`)
- Modify: `pyproject.toml`, `uv.lock` (remove `langchain-community`, `rank-bm25`; run `uv lock`, then `uv sync`)
- Modify: `README.md` (the "How a write is decided" paragraph in "Long-term memory": candidates come from Mem0's vector search, this user's 5 most similar memories; remove BM25, `rank-bm25` and `langchain-community`)
- Test: `tests/test_memory.py` (`RememberDecisionTests`), `tests/test_memory_updates.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `updates.py` no longer exports `rank_candidates`. Task 3 shows each scenario's memories directly. `MemoryService.remember(fact, recent=())` keeps its result shapes.

- [ ] **Step 1: Write the failing tests** in `RememberDecisionTests`, using the existing fake mem0. Extend it so `search` records its kwargs and returns its configured hits, `get` returns a configured item, and `update` can raise.
  - `test_candidates_come_from_search_for_this_user`: the fake's `search` was called with `filters={"user_id": <user>}`, `top_k=5`, `threshold=0.0`. `get_all` was not called. The decider received exactly the search hits, redacted.
  - `test_no_search_hits_adds_without_asking`: `search` returns `[]`. The decider is not called, and the result is `added`.
  - `test_a_hit_for_another_user_is_never_a_candidate`: the search returns a hit with `user_id="someone-else"`, and it isn't among the candidates the decider saw.
  - `test_search_failure_still_stores_the_fact`: `search` raises `RuntimeError("embedder down")`. The result is `added`, with a `note` that contains "couldn't decide".
  - `test_update_that_raised_but_saved_is_reported_updated`: `update` raises, and `get(id)` returns `{"id": id, "memory": <new text>, "user_id": <user>}`. The result is `action == "updated"` with `note` equal to the exact note from Global Constraints. `add` was not called.
  - `test_update_that_raised_and_did_not_save_adds`: `update` raises, and `get` returns the old text. The result is `added` and has a note.
  - `test_update_read_back_of_another_users_memory_adds`: `get` returns the new text with `user_id="someone-else"`. The result is `added`.
  - In `tests/test_memory_updates.py`, delete the tests of `rank_candidates`, `_words`, the subprocess import test and the sunset-warning test. Keep the `Operation` and `MemoryDecider` tests.
  - In `tests/test_memory.py`, delete tests that only cover `ALL_MEMORIES_LIMIT` or `_every_memory`. Keep `test_startup_does_not_import_langchain_community`.

- [ ] **Step 2: Run them to verify the new ones fail**

  Run: `uv run --no-sync python -m unittest tests.test_memory tests.test_memory_updates -v`
  Expected: the new tests FAIL (`get_all` used, no read-back).

- [ ] **Step 3: Implement.**
  - **Candidates.** In `remember`, after the secrets check and the decider check: `candidates = self._candidates(fact)`.
    - `_candidates` calls `search` with the Global Constraints arguments.
    - It keeps hits with no `user_id` or with `user_id == self.user_id`.
    - It maps them through `self._shown`, then drops `score`: candidates are `{"id", "memory"}`.
  - **Search failure.** A raising search goes to the same ADD-with-note path as a failed decision: `note=f"couldn't decide ({redacted error}); added as a new memory"`.
  - **Failed update.** On an `update` exception, call `self.memory.get(target["id"])`. If it is a dict, its `user_id` equals `self.user_id`, and `memory.strip() == text`, return the `updated` dict plus `"note": <exact note>`. Otherwise, and if `get` raises too, use the existing ADD fallback.
  - **Cleanup.** Remove the dead code and the two dependencies (`uv lock`, `uv sync`), and update the README paragraph.

- [ ] **Step 4: Run the tests and verify they pass**

  Run: `uv run --no-sync python -m unittest tests.test_memory tests.test_memory_updates -v`
  Expected: all PASS.

- [ ] **Step 5: Run the full suite**

  Run: `uv run --no-sync python -m unittest discover -s tests`
  Expected: OK. `tests/test_eval_memory_updates.py` may fail on the `rank_candidates` import. That is Task 3's. Note it in the report and don't fix it here.

- [ ] **Step 6: Commit**

  `git add -A src tests pyproject.toml uv.lock README.md && git commit -m "Memory: candidates from Mem0 search (this user, top 5); BM25 removed; a saved update that raised counts as updated"`

---

### Task 3: Eval: candidates shown directly, paraphrase sequence, narrower PAST

**Files:**
- Modify: `evals/memory_updates.py` (the decision part: candidates = the scenario's memories, in seeded order; remove the `rank_candidates` import; `PAST`)
- Modify: `evals/memory_update_scenarios.py` (add the sequence)
- Test: `tests/test_eval_memory_updates.py`

**Interfaces:**
- Consumes: Task 2's `updates.py` (no `rank_candidates`; `RECENT_MESSAGES`, `MemoryDecider`, `Operation` remain).
- Produces: nothing later tasks use.

- [ ] **Step 1: Write the failing tests** in `tests/test_eval_memory_updates.py`.
  - `test_decision_candidates_are_the_seeded_memories_in_order`: for each scenario in `DECISIONS`, the candidates the decision part passes to the decider are the seeded memories in order, with the ids the eval assigns. Use a stub decider that records them.
  - `test_every_decision_scenario_has_at_most_five_memories`: `len(d.memories) <= 5`.
  - `test_paraphrase_move_sequence_exists`: its facts start with 6 unrelated facts and end with `"I live in Rome."`, `"I relocated to Milan."`. Its expect is `Expect(["milan|milano"], ["rome"])`, and no fact after `"I live in Rome."` contains "rome" (case-insensitive).
  - `test_gave_a_talk_is_not_history`: a memory `"Gave a talk at PyCon"` is not treated as a past-marker history line, while `"Gave away the cat"` still is. Use the existing scoring helper the tests call.
  - Replace `test_the_target_is_among_the_candidates_the_decider_sees` (it used `rank_candidates`) with the first test above.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run --no-sync python -m unittest tests.test_eval_memory_updates -v`
  Expected: FAIL (import error on `rank_candidates`, then the new assertions).

- [ ] **Step 3: Implement.**
  - The decision part builds `candidates = [{"id": i, "memory": m} for i, m in zip(ids, decision.memories)]`.
  - `PAST = "used to|formerly|previously|no longer|moved from|gave away|former|anymore"`.
  - Add `Sequence("paraphrase-move", [...6 unrelated facts..., "I live in Rome.", "I relocated to Milan."], [Expect(["milan|milano"], ["rome"])])`. For example, the 6 unrelated facts are: "I like green tea.", "I have a dog named Rex.", "I play the piano.", "I am allergic to peanuts.", "My favourite colour is blue.", "I work as a nurse."
  - Fix any other eval text that mentions BM25.

- [ ] **Step 4: Run the tests and verify they pass**

  Run: `uv run --no-sync python -m unittest tests.test_eval_memory_updates -v`
  Expected: all PASS.

- [ ] **Step 5: Run the full suite**

  Run: `uv run --no-sync python -m unittest discover -s tests`
  Expected: OK.

- [ ] **Step 6: Commit**

  `git add -A evals tests && git commit -m "Evals: memory updates show seeded candidates, add a paraphrase move, narrower past markers"`

---

After the tasks, the controller runs the following (not a subagent task):
- the live race check with user tools-check (`live_followups.py`-style script, PHASE=agent);
- `evals/memory_updates.py --runs 3`;
- the README results table update.

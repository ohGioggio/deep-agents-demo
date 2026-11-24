# Memory Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `remember` updates, skips or proposes deleting an existing memory instead of always adding one.

**Architecture:** A new `features/memory/updates.py` holds the `Operation` model, BM25 ranking
(LangChain `BM25Retriever`) and a `MemoryDecider` around a chat model with structured output.
`MemoryService.remember` ranks this user's memories, asks the decider, and applies the operation with
mem0's `add(infer=False)` / `update`. The `remember` tool passes the last messages from `ToolRuntime`.
A live eval (`evals/memory_updates.py`) measures decisions and before/after on real mem0.

**Tech Stack:** Python 3.13, LangChain 1.x (`langchain-community` `BM25Retriever`, `rank-bm25`),
`langchain-openai`, mem0ai 2.2.1, unittest.

**Spec:** design/specs/2026-11-24-memory-updates-design.md

## Global Constraints

- Tests: `uv run --no-sync python -m unittest discover -s tests` (pytest is not installed); `tests/test_structure.py` passes (only `ui/` prints; `core/` imports no `features/` or `ui/`).
- New dependencies: only `langchain-community` and `rank-bm25` (add with `uv add`, commit `pyproject.toml` and `uv.lock`).
- Commits: one line each, no trailer.
- Never read or print `.env`; offline tests only (fake models, fake mem0); no live calls except Task 4.
- Constants: `RECENT_MESSAGES = 6`, `UPDATE_CANDIDATES = 5` (in `features/memory/updates.py`).
- Operation kinds exactly: `ADD`, `UPDATE`, `DELETE`, `NONE`.
- `DELETE` never deletes inside `remember`: the result tells the agent to call `forget(id)`.
- A `memory_id` not among the candidates shown, an invalid operation, or a failed decision call → `ADD` of the fact, and the result says so.
- The secrets check runs on the fact first and on the operation's `text`; recent messages are redacted (`policy.redact`) before they reach the decision model.

## Review Focus

1. Recent messages that contain a secret (an API key pasted earlier) → redacted before the decision call. Test in Task 2.
2. The model answers `UPDATE` with an id from another user or not among the candidates → the fact is added; no memory is touched. Test in Task 2.
3. `UPDATE`/`ADD` with empty `text` → the fact itself is used (ADD) — never an empty memory. Test in Task 2.
4. `remember` called where the state has no messages (a plan or swarm worker, the eval) → works with no recent messages. Test in Task 2.
5. A single memory, or a fact of only punctuation, through BM25 → no crash, candidates returned. Test in Task 1.

---

### Task 1: `Operation`, BM25 candidates and the decider

**Files:**
- Create: `src/deep_agents_demo/features/memory/updates.py`
- Modify: `pyproject.toml`, `uv.lock` (`uv add langchain-community rank-bm25`)
- Test: `tests/test_memory_updates.py`

**Interfaces:**
- Produces:
  - `class Operation(pydantic.BaseModel)`: `kind: Literal["ADD", "UPDATE", "DELETE", "NONE"]`, `memory_id: str | None = None`, `text: str | None = None`, `reason: str = ""`.
  - `rank_candidates(fact: str, memories: list[dict], k: int = UPDATE_CANDIDATES) -> list[dict]` — `memories` are `{"id", "memory"}` dicts; returns at most `k` of them, best first, via `BM25Retriever.from_texts(texts, metadatas=[{"id": ...}], k=k)`; `[]` for no memories.
  - `class MemoryDecider`: `__init__(self, model: BaseChatModel)`; `decide(self, fact: str, recent: list[str], candidates: list[dict]) -> Operation` — one call to `model.with_structured_output(Operation)` with `DECISION_PROMPT` (system) and a user message listing the recent messages, the candidates (`[id] text`) and the fact; exceptions propagate (the service falls back).
  - `DECISION_PROMPT: str`, `RECENT_MESSAGES = 6`, `UPDATE_CANDIDATES = 5`.

- [ ] **Step 1: Failing tests** in `tests/test_memory_updates.py`:
  - `test_the_memory_about_the_same_thing_ranks_first`: memories `[{"id": "1", "memory": "Lives in Rome"}, {"id": "2", "memory": "Likes green tea"}, {"id": "3", "memory": "Has a dog named Rex"}]`, fact `"Now lives in Milan, not Rome"` → `rank_candidates(...)[0]["id"] == "1"`.
  - `test_at_most_k_candidates_and_none_without_memories`: 8 memories, `k=5` → 5; `rank_candidates("x", []) == []`.
  - `test_one_memory_or_a_punctuation_fact_still_ranks` (Review Focus 5): one memory → `[that memory]`; fact `"?!"` with 3 memories → no exception, `len <= 3`.
  - `test_the_decider_returns_the_models_operation`: a fake model whose `with_structured_output` returns a runnable yielding `Operation(kind="UPDATE", memory_id="1", text="Lives in Milan", reason="moved")` → `decide(...)` equals it; the prompt it got contains the fact, `"[1] Lives in Rome"` and each recent message.
  - `test_operation_rejects_an_unknown_kind`: `Operation(kind="MERGE")` raises `pydantic.ValidationError`.
- [ ] **Step 2: Run** `uv run --no-sync python -m unittest tests.test_memory_updates` → fails (`ModuleNotFoundError: ...updates`).
- [ ] **Step 3: `uv add langchain-community rank-bm25`, then implement `updates.py`.** `DECISION_PROMPT` explains the four kinds with one example each (changed fact → UPDATE with the full new text; same fact in other words → NONE; new unrelated → ADD; the user retracts a fact → DELETE), and: only use an id from the list; a similar but different fact is ADD; write `text` as a short third-person fact.
- [ ] **Step 4: Run** the module, then `tests.test_structure` → PASS.
- [ ] **Step 5: Commit** `Memory: an Operation, BM25 candidates and a decider for add, update, delete or nothing`.

### Task 2: `remember` applies the decision

**Files:**
- Modify: `src/deep_agents_demo/features/memory/service.py` (`MemoryService.__init__`, `remember`), `src/deep_agents_demo/features/memory/tools.py` (`remember`), `src/deep_agents_demo/providers/chat.py`, `src/deep_agents_demo/startup/runtime.py` (~112), `README.md` (Memory section)
- Test: `tests/test_memory.py` (new class `RememberDecisionTests`; existing tests keep passing unchanged)

**Interfaces:**
- Consumes: Task 1's `Operation`, `rank_candidates`, `MemoryDecider`, `RECENT_MESSAGES`.
- Produces:
  - `MemoryService(memory, user_id, policy=WritePolicy(), decider: MemoryDecider | None = None)`.
  - `MemoryService.remember(fact: str, recent: list[str] = ()) -> dict`. Without a decider: `add(..., infer=False)` (no mem0 extraction). Result dicts (JSON-ready), `action` one of:
    - `"added"`: `{"stored": True, "action": "added", "results": [shown], "reason": ...}` (+ `"note"` when it was a fallback: `"couldn't decide (<error or why>); added as a new memory"`)
    - `"updated"`: `{"stored": True, "action": "updated", "id", "before", "after", "reason"}`
    - `"already known"`: `{"stored": False, "action": "already known", "id", "memory", "reason"}`
    - `"to forget"`: `{"stored": False, "action": "to forget", "id", "memory", "reason", "advice": "Call forget with this id; the user approves the deletion."}`
    - secrets: the existing refusal dict for the fact; for a secret-looking `text`: `{"stored": False, "reason": "refused: the new text looks like a secret"}`.
  - `memory_chat_model(settings: Settings) -> ChatOpenAI` in `providers/chat.py`: `settings.memory.model/api_key/base_url`, same rate limiter, retries, timeout and `callbacks=[_USAGE]` as `chat_model`.
  - Tool `remember(fact: str, runtime: ToolRuntime) -> str`: the last `RECENT_MESSAGES` human/AI messages' text from `runtime.state["messages"]` (none if absent).

- [ ] **Step 1: Failing tests** (`RememberDecisionTests`, a fake decider returning a set `Operation` or raising; the existing `FakeMemory` gains `update(memory_id, data)` and records `add`'s `infer`):
  - `test_add_stores_the_text_without_mem0_extraction` → `infer=False`, action `added`.
  - `test_update_changes_the_candidate_and_reports_before_and_after`.
  - `test_none_stores_nothing_and_names_the_memory`.
  - `test_delete_deletes_nothing_and_points_to_forget` → no `delete` call, action `to forget`, `advice` mentions `forget`.
  - `test_an_id_not_among_the_candidates_adds_the_fact` (Review Focus 2) — includes another user's memory id → nothing updated, action `added`, `note` present.
  - `test_empty_text_adds_the_fact_itself` (Review Focus 3).
  - `test_a_failed_decision_adds_the_fact_and_says_so`.
  - `test_no_memories_yet_adds_without_asking_the_decider`.
  - `test_a_secret_in_the_new_text_is_refused_and_nothing_written`.
  - `test_recent_messages_reach_the_decider_redacted` (Review Focus 1): a recent message with `sk-...` → the decider sees it masked.
  - `test_the_tool_passes_the_last_six_messages_and_works_without_any` (Review Focus 4).
  - `test_only_this_users_memories_are_candidates`.
- [ ] **Step 2: Run** `uv run --no-sync python -m unittest tests.test_memory` → the new tests fail.
- [ ] **Step 3: Implement** `remember` per the Interfaces; `runtime.py`: `MemoryService(resources.memory, user, decider=MemoryDecider(memory_chat_model(settings)))`; tool signature with `runtime: ToolRuntime` (as `send_file` in `features/files/feature.py`). README Memory section: how a write is decided (the four kinds, DELETE through `forget`, the fallback), and the two new dependencies.
- [ ] **Step 4: Run** `tests.test_memory`, `tests.test_memory_updates`, `tests.test_structure`, then the whole suite → PASS.
- [ ] **Step 5: Commit** `Memory: remember updates, skips or proposes forgetting an existing memory instead of always adding`.

### Task 3: the eval (offline parts)

**Files:**
- Create: `evals/memory_updates.py`, `evals/memory_update_scenarios.py`
- Test: `tests/test_eval_memory_updates.py`

**Interfaces:**
- Consumes: `MemoryService`, `MemoryDecider`, `memory_chat_model`, `make_memory` (`features/memory/service.py`), `Settings.from_env`, the eval style of `evals/memory_recall.py` (temp data dir, `--runs`, JSON to `data/evals/`, `mean (min–max)` table as in `evals/orchestration_scoring.py` `format_table`).
- Produces:
  - `memory_update_scenarios.py`: `DECISIONS: list[Decision]` — `Decision(name, memories: list[str], recent: list[str], fact: str, kind: str, target: int | None)` (`target` = index into `memories`); at least 20, at least 3 per kind, plus traps: a similar but different fact → `ADD`; a fact about someone else → `ADD`; a repeat in other words → `NONE`. `SEQUENCES: list[Sequence]` — `Sequence(name, facts: list[str], expected: list[Expect])`, `Expect(must: list[str], must_not: list[str])` (lowercase keywords); at least 6 (moves, changed preferences, a retraction, repeats).
  - `score_decision(decision: Decision, op: Operation, ids: list[str]) -> dict` → `{"kind_ok": bool, "target_ok": bool | None}`.
  - `score_final(sequence: Sequence, memories: list[str]) -> dict` → `{"count", "expected", "extra", "stale", "missing"}`: `stale` = memories containing a `must_not` keyword; `missing` = expects with no memory containing all `must` keywords; `extra = max(0, count - expected)`.
  - CLI: `python evals/memory_updates.py [--runs N] [--parts decision compare]`. `decision`: the real decider on each `Decision` (no mem0). `compare`: per sequence, a fresh temp mem0 (`make_memory`), facts through the old path (`memory.add([{"role": "user", "content": fact}], user_id=..., infer=True)`) and through the new `MemoryService.remember`; collects `score_final`, model calls and tokens (a usage counter on both the decider model and, for the old path, mem0's calls counted by timing only — say so in the output), seconds per remember (p50/p95). Table: per part/path, `mean (min–max)` over runs.
- [ ] **Step 1: Failing tests:** `score_decision` (right kind and target; right kind wrong target; NONE with `target None`); `score_final` (stale keyword counted, missing expect, extra count); scenario sanity (≥20 decisions, ≥3 per kind, every UPDATE/DELETE target is a valid index, ≥6 sequences); `format` of a 3-run summary gives `"x (a–b)"` cells.
- [ ] **Step 2: Run** `uv run --no-sync python -m unittest discover -s tests -p test_eval_memory_updates.py` → fails.
- [ ] **Step 3: Implement** the scenarios, scoring and CLI (no live calls in tests).
- [ ] **Step 4: Run** the module and the whole suite → PASS.
- [ ] **Step 5: Commit** `Evals: memory updates, decisions and before/after on real mem0`.

### Task 4 (controller, live): run the eval, README results

- [ ] `uv run --no-sync python evals/memory_updates.py --runs 3` with a log and a stall watcher.
- [ ] README Memory section: a results table (decision accuracy per kind; before/after: memories left, stale, missing, calls, seconds p50/p95), and what it says.
- [ ] Commit `README: memory update results`.

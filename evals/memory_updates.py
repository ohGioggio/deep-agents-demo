"""Live eval: memory updates (add, update, delete or nothing) and the before/after on real mem0.

    uv run python evals/memory_updates.py                      # both parts, 1 run each
    uv run python evals/memory_updates.py --runs 3 --parts decision

Uses the memory model from .env, so it makes paid API calls. Every run gets a fresh temporary
Mem0 store: your real memories are untouched.

decision: the real decider on each scenario in memory_update_scenarios.DECISIONS (no mem0).
          Reports whether the kind and the target memory were right.
compare:  each sequence in SEQUENCES on a fresh store, twice: the old path (mem0 add with
          infer=True) and the new path (MemoryService.remember, with the sequence's previous
          facts as recent messages, and a "to forget" answer carried out through forget).
          Reports stale, missing and extra memories at the end, the final memory count, errors,
          model calls and tokens, and seconds per write (p50 and p95). The old path's own mem0
          model calls are not visible here, so only its timing, errors and results are reported.

Results are also saved as JSON in data/evals/.
"""
import argparse
import contextlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from deep_agents_demo.config import Settings  # noqa: E402
from deep_agents_demo.features.memory.service import close_memory, make_memory  # noqa: E402
from deep_agents_demo.features.memory.updates import (  # noqa: E402
    RECENT_MESSAGES, MemoryDecider, Operation)
from orchestration_scoring import UsageCounter  # noqa: E402
from memory_update_scenarios import DECISIONS, SEQUENCES, Decision, Expect, Sequence  # noqa: E402

USER = "eval-user"
# Words that say a memory is about the past (history), not a current fact.
PAST = "used to|formerly|previously|no longer|moved from|gave away|former|anymore"
NOTES = [
    "Old path: mem0's own model calls and tokens are not counted; its timing, errors and final memories are.",
    "New path: remember gets the sequence's previous facts (last 6) as recent messages.",
    "A 'to forget' answer is auto-approved: the eval applies it with forget on that id.",
    "Seconds p50 and p95 are nearest-rank percentiles: the ceil(p/100 * n)-th smallest of n writes.",
    "Keywords in must and must_not are whole words, case-insensitive; '|' separates alternatives.",
]


def score_decision(decision: Decision, op: Operation, ids: list[str]) -> dict:
    """Right kind; right memory when the decision names one (None for ADD and NONE)."""
    target_ok = None if decision.target is None else op.memory_id == ids[decision.target]
    return {"kind_ok": op.kind == decision.kind, "target_ok": target_ok}


def _has(keyword: str, memory: str) -> bool:
    """True when the memory contains the keyword as whole words (any '|' alternative, any case).

    Multi-word alternatives may have any spaces between their words."""
    for alternative in keyword.split("|"):
        words = alternative.split()
        if words and re.search(r"\b" + r"\s+".join(re.escape(word) for word in words) + r"\b",
                               memory, re.IGNORECASE):
            return True
    return False


def _meets(expect: Expect, memory: str) -> bool:
    """One memory meets an expect when it has every must entry (each one, in any alternative)."""
    return all(_has(keyword, memory) for keyword in expect.must)


def score_final(sequence: Sequence, memories: list[str]) -> dict:
    """Memories left after a sequence, against what it should leave.

    stale: a memory with a must_not keyword of an expect, that matches none of that expect's
           must entries (so "Moved from Rome to Milan" is not stale).
    missing: expects left without a memory in a best one-to-one pairing (maximum matching): each expect
           takes one memory that meets it, and a memory meets only one expect (a merged memory
           cannot satisfy two).
    extra: memories beyond the number expected.
    """
    stale = 0
    for memory in memories:
        if _has(PAST, memory):  # a line that states the past is history, not a stale fact
            continue
        for expect in sequence.expected:
            if any(_has(keyword, memory) for keyword in expect.must_not) \
                    and not any(_has(keyword, memory) for keyword in expect.must):
                stale += 1
                break
    owner: dict[int, int] = {}  # memory index -> the expect index holding it

    def take(expect_index: int, seen: set[int]) -> bool:
        """Augmenting path: give this expect a memory, moving earlier expects to others if needed."""
        for index, memory in enumerate(memories):
            if index in seen or not _meets(sequence.expected[expect_index], memory):
                continue
            seen.add(index)
            if index not in owner or take(owner[index], seen):
                owner[index] = expect_index
                return True
        return False

    missing = sum(not take(i, set()) for i in range(len(sequence.expected)))
    expected = len(sequence.expected)
    return {"count": len(memories), "expected": expected, "extra": max(0, len(memories) - expected),
            "stale": stale, "missing": missing}


def nearest_rank(values: list[float], percent: int) -> float:
    """The ceil(percent/100 * n)-th smallest value (nearest-rank); the one value when n is 1."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = -(-percent * len(ordered) // 100)  # integer ceiling
    return ordered[max(1, rank) - 1]


def _format_value(name: str, value: float) -> str:
    if name.endswith("right") or name.startswith("Clean"):
        return f"{value:.0%}"
    if name.startswith("Seconds"):
        return f"{value:.1f} s"
    if "tokens" in name:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.1f}"


def _cell(name: str, values: list[float]) -> str:
    if len(values) == 1:
        return _format_value(name, values[0])
    return (f"{_format_value(name, statistics.mean(values))} "
            f"({_format_value(name, min(values))}–{_format_value(name, max(values))})")


def format_table(runs: list[list[dict]]) -> str:
    """One row per (part, path); each metric is ``mean (min–max)`` over the runs."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for run in runs:
        for entry in run:
            cells = groups.setdefault((entry["part"], entry["path"]), {})
            for name, value in entry["metrics"].items():
                cells.setdefault(name, []).append(value)
    names = list(dict.fromkeys(name for cells in groups.values() for name in cells))
    lines = ["| Part | Path | " + " | ".join(names) + " |",
             "| --- | --- |" + " --- |" * len(names)]
    for (part, path), cells in groups.items():
        row = [_cell(name, cells[name]) if name in cells else "-" for name in names]
        lines.append(f"| {part} | {path} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def settings_from_env() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    problem = settings.chat.problem()
    if problem:
        raise SystemExit(problem)
    return settings


def run_decisions(model, decisions) -> dict:
    counter = UsageCounter()
    model.callbacks = [counter]
    decider = MemoryDecider(model)
    calls0, input0, output0 = counter.snapshot()
    rows, errors = [], 0
    for decision in decisions:
        ids = [str(index) for index in range(len(decision.memories))]
        candidates = [{"id": i, "memory": m} for i, m in zip(ids, decision.memories)]
        try:
            op = decider.decide(decision.fact, decision.recent, candidates)
            score = score_decision(decision, op, ids)
        except Exception as error:  # a failed call is a wrong answer; the run goes on
            errors += 1
            op = None
            score = {"kind_ok": False, "target_ok": False if decision.target is not None else None}
        rows.append({"name": decision.name, "expected": decision.kind, "got": op.kind if op else "error",
                     "memory_id": op.memory_id if op else None, **score})
    calls1, input1, output1 = counter.snapshot()
    targeted = [row for row in rows if row["target_ok"] is not None]
    metrics = {
        "Kind right": sum(row["kind_ok"] for row in rows) / len(rows),
        "Errors": errors,
        "Calls": calls1 - calls0,
        "Input tokens": input1 - input0,
        "Output tokens": output1 - output0,
    }
    if targeted:
        metrics["Target right"] = sum(row["target_ok"] for row in targeted) / len(targeted)
    return {"entries": [{"part": "decision", "path": "decider", "metrics": metrics}], "rows": rows}


def _old_path(memory, sequence, times: list[float]) -> tuple[int, str | None]:
    """mem0's own extraction on each fact. Returns (failed writes, the first error's text)."""
    errors, first = 0, None
    for fact in sequence.facts:
        start = time.perf_counter()
        try:
            memory.add([{"role": "user", "content": fact}], user_id=USER, infer=True)
        except Exception as error:  # one failed write does not stop the sequence
            errors += 1
            first = first or repr(error)
        times.append(time.perf_counter() - start)
    return errors, first


def _new_path(memory, sequence, model, times: list[float]) -> tuple[int, str | None]:
    """MemoryService.remember on each fact, with the previous facts as recent messages.
    Returns (failed writes, the first error's text); a service that cannot be built fails every fact."""
    from deep_agents_demo.features.memory.service import MemoryService  # the live part: new service

    try:
        service = MemoryService(memory, USER, decider=MemoryDecider(model))
    except Exception as error:
        return len(sequence.facts), repr(error)
    errors, first = 0, None
    for index, fact in enumerate(sequence.facts):
        recent = sequence.facts[max(0, index - RECENT_MESSAGES):index]
        start = time.perf_counter()
        try:
            result = service.remember(fact, recent=recent)
            if result.get("action") == "to forget":  # the advice, auto-approved in the eval
                service.forget(result["id"])
        except Exception as error:  # one failed write does not stop the sequence
            errors += 1
            first = first or repr(error)
        times.append(time.perf_counter() - start)
    return errors, first


@contextlib.contextmanager
def _mem0_dir(path: Path):
    """Point MEM0_DIR at ``path`` and put back whatever was set before."""
    saved = os.environ.get("MEM0_DIR")
    os.environ["MEM0_DIR"] = str(path)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MEM0_DIR", None)
        else:
            os.environ["MEM0_DIR"] = saved


def run_compare(settings, model, sequences) -> dict:
    from deep_agents_demo.startup.resources import embeddings

    counter = UsageCounter()
    model.callbacks = [counter]
    rows, scores = [], {"old": [], "new": []}
    times = {"old": [], "new": []}
    errors = {"old": 0, "new": 0}
    calls = input_tokens = output_tokens = 0  # decider usage on the new path
    facts = sum(len(sequence.facts) for sequence in sequences)
    for sequence in sequences:
        for path in ("old", "new"):
            failed, first, memories = 0, None, []
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, _mem0_dir(Path(tmp) / "mem0_internal"):
                try:
                    memory = make_memory(Path(tmp), settings.memory, settings.embeddings,
                                         embeddings(settings, warn=lambda message: print(message, file=sys.stderr)))
                except Exception as error:  # no store: every fact of this sequence counts as an error
                    memory = None
                    failed, first = len(sequence.facts), repr(error)
                if memory is not None:
                    try:
                        before = counter.snapshot()
                        if path == "old":
                            failed, first = _old_path(memory, sequence, times["old"])
                        else:
                            failed, first = _new_path(memory, sequence, model, times["new"])
                        after = counter.snapshot()
                        if path == "new":
                            calls += after[0] - before[0]
                            input_tokens += after[1] - before[1]
                            output_tokens += after[2] - before[2]
                        stored = memory.get_all(filters={"user_id": USER}, top_k=100)
                        stored = stored.get("results", []) if isinstance(stored, dict) else stored
                        memories = [str(item.get("memory", "")) for item in stored]
                    except Exception as error:  # e.g. the store can't be read back: this sequence fails
                        failed, first, memories = len(sequence.facts), first or repr(error), []
                    finally:
                        close_memory(memory)
            score = score_final(sequence, memories)
            errors[path] += failed
            scores[path].append(score)
            rows.append({"sequence": sequence.name, "path": path, "errors": failed, "first_error": first, **score})
    entries = []
    for path, group in scores.items():
        metrics = {
            "Clean sequences": sum(s["stale"] == s["missing"] == s["extra"] == 0 for s in group) / len(group),
            "Stale": statistics.mean(s["stale"] for s in group),
            "Missing": statistics.mean(s["missing"] for s in group),
            "Extra": statistics.mean(s["extra"] for s in group),
            "Final memories": statistics.mean(s["count"] for s in group),
            "Errors": errors[path],
            "Seconds p50": nearest_rank(times[path], 50),
            "Seconds p95": nearest_rank(times[path], 95),
        }
        if path == "new":
            metrics["Calls per fact"] = calls / facts
            metrics["Input tokens per fact"] = input_tokens / facts
            metrics["Output tokens per fact"] = output_tokens / facts
        entries.append({"part": "compare", "path": "old: mem0 add" if path == "old" else "new: remember",
                        "metrics": metrics})
    return {"entries": entries, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--parts", nargs="+", choices=["decision", "compare"], default=["decision", "compare"])
    args = parser.parse_args()
    settings = settings_from_env()
    from deep_agents_demo.providers.chat import memory_chat_model

    model = memory_chat_model(settings)
    runs = []
    for index in range(1, args.runs + 1):
        run = []
        if "decision" in args.parts:
            print(f"run {index}, decisions", flush=True)
            result = run_decisions(model, DECISIONS)
            for row in result["rows"]:
                print(f"  {row['name']:<24} {'PASS' if row['kind_ok'] and row['target_ok'] is not False else 'FAIL'}"
                      f"  expected {row['expected']:<6} got {row['got']}", flush=True)
            run.extend(result["entries"])
        if "compare" in args.parts:
            print(f"run {index}, before and after", flush=True)
            result = run_compare(settings, model, SEQUENCES)
            for row in result["rows"]:
                print(f"  {row['path']:<4} {row['sequence']:<18} stale={row['stale']} missing={row['missing']} "
                      f"extra={row['extra']} errors={row['errors']}", flush=True)
            run.extend(result["entries"])
        runs.append(run)
    table = format_table(runs)
    print("\n" + table)
    for note in NOTES:
        print(note)
    out = ROOT / "data" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"memory_updates-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"model": settings.memory.model or settings.chat.model, "runs": args.runs,
                                "parts": args.parts, "summary": table, "notes": NOTES, "entries": runs},
                               indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

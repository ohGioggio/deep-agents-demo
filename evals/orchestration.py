"""Live eval: the main agent alone (single) against a room of agents (room) on the same tasks.

    uv run python evals/orchestration.py                       # every mode, 1 run
    uv run python evals/orchestration.py --modes single room --runs 3
    uv run python evals/orchestration.py --families review --tasks review-config
    uv run python evals/orchestration.py --help                # usage only, no model call
    uv run python evals/orchestration.py --rescore FILE.json   # re-score a saved run, no model call

Uses the real chat model from .env, so it makes paid API calls: one run of single is
13 agent turns (a few calls each); one run of room is 13 rooms of up to 12 participant
turns plus moderator calls, a few hundred model calls. Every task gets a fresh
temporary workspace: your real files and chats are untouched. Each run prints a line
per task and a table (calls, tokens, time, score). Results are also saved as JSON in
data/evals/.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from deepagents.backends import FilesystemBackend  # noqa: E402

from deep_agents_demo.config import Settings  # noqa: E402
from deep_agents_demo.core.approvals import Approvals  # noqa: E402
from deep_agents_demo.core.assembly import build_agent, worker_factory  # noqa: E402
from deep_agents_demo.features.agents.builtins import BUILTIN_AGENTS  # noqa: E402
from deep_agents_demo.features.orchestration.rooms import RoomRunner  # noqa: E402
from deep_agents_demo.features.orchestration.runner import RECURSION_LIMIT  # noqa: E402
from deep_agents_demo.models.agents import READ_ONLY, AgentDefinition  # noqa: E402
from deep_agents_demo.providers.chat import chat_model  # noqa: E402
from deep_agents_demo.startup.workspace import seed_files  # noqa: E402

from orchestration_scoring import UsageCounter, format_table, score, summarize  # noqa: E402
from orchestration_tasks import TASKS, full_prompt, seed  # noqa: E402

ANSWER_LIMIT = 2_000  # characters of the answer kept in each row
FAMILIES = ("lookup", "decision", "review", "safety")

PARTICIPANTS = (
    AgentDefinition(
        name="analyst", source="eval", tools=READ_ONLY, read_only=True,
        description="Reads the task's files and answers the question with facts from them.",
        prompt=("You answer one question about files in a workspace. Read the files under the "
                "folder named in the question, and answer only from what they say. Quote the "
                "exact figures and names. Keep it short."),
    ),
    AgentDefinition(
        name="skeptic", source="eval", tools=READ_ONLY, read_only=True,
        description="Checks the other participants' claims against the files and points out what is wrong or missing.",
        prompt=("You check the other participants' claims against the files. Read the files under "
                "the folder named in the question, and answer only from them. Say which claims are "
                "wrong or missing, and what the files actually say. Keep it short."),
    ),
)


class NullSink:
    """The room's events are not shown: the eval only reads the summary."""

    def emit(self, event) -> None:
        return None


def _last_ai_text(messages) -> str:
    """The last AI message that has text (a reply that only calls tools has none)."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and str(message.text).strip():
            return str(message.text)
    return ""


# Each mode returns (answer, info). ``keep`` is where a room's transcript goes when it must
# outlive the temporary workspace; None means the task's temporary folder.
def run_single(model, backend, task, settings, tmp: Path, label: str, keep: Path | None):
    agent = build_agent(model, InMemorySaver(), backend, [],
                        max_calls=settings.subagent_max_calls,
                        max_parallel=settings.subagent_max_parallel)
    config = {"configurable": {"thread_id": label}, "recursion_limit": RECURSION_LIMIT}
    result = agent.invoke({"messages": [{"role": "user", "content": full_prompt(task)}]}, config)
    return _last_ai_text(result["messages"]), {}


DESCRIPTIONS = {agent.name: agent.description for agent in PARTICIPANTS}


def run_room(model, backend, task, settings, tmp: Path, label: str, keep: Path | None):
    make = worker_factory(model, InMemorySaver(), backend, [], agents=PARTICIPANTS,
                          builtins=tuple(BUILTIN_AGENTS.values()),
                          max_calls=settings.subagent_max_calls)
    transcripts = (keep if keep is not None else tmp) / "rooms"
    room = RoomRunner(make, Approvals(interactive=False), None, NullSink(), thread=label,
                      max_turns=settings.room_max_turns, budget=settings.orchestration_max_calls,
                      interactive=False, transcripts=transcripts,
                      describe=lambda name: DESCRIPTIONS.get(name, ""),
                      # the room settings the app passes (ui/cli/repl.py runners_for)
                      recent_turns=settings.room_recent_turns,
                      opening_round=settings.room_opening_round, format=settings.room_format)
    room.quiet = True
    outcome = room.run({"type": "start_room", "question": full_prompt(task),
                        "participants": ["analyst", "skeptic"]})
    if outcome.get("ended_by") == "stopped":  # Ctrl+C: the eval stops too, it does not go on paying
        raise KeyboardInterrupt
    if not outcome.get("summary"):
        raise RuntimeError(outcome.get("error", "the room ended without a summary"))
    info = {
        "ended_by": outcome.get("ended_by", ""),
        "turns": outcome.get("turns", 0),
        "failed_turns": sum("could not answer" in line for line in room.lines),
        "transcript": outcome.get("transcript", ""),
    }
    return outcome["summary"], info


MODES = {"single": run_single, "room": run_room}


def run_task(mode: str, model, counter: UsageCounter, backend, task, settings, tmp: Path,
             run_index: int, keep: Path | None = None) -> dict:
    calls0, input0, output0 = counter.snapshot()
    start = time.perf_counter()
    answer, details, info, error = "", {}, {}, None
    try:
        answer, info = MODES[mode](model, backend, task, settings, tmp,
                                   f"{run_index}-{mode}-{task.name}", keep)
        result = score(task, answer)
        details = {key: value for key, value in result.items() if key != "score"}
        final_score = result["score"]
    except Exception as e:  # a failed task is a zero, not a crashed eval
        final_score, error, info = 0.0, f"{type(e).__name__}: {e}", {}
    seconds = time.perf_counter() - start
    calls1, input1, output1 = counter.snapshot()
    return {
        "run": run_index, "mode": mode, "task": task.name, "family": task.family,
        "score": final_score, "details": details, "info": info,
        "calls": calls1 - calls0, "input_tokens": input1 - input0, "output_tokens": output1 - output0,
        "seconds": seconds, "answer": answer[:ANSWER_LIMIT] if error is None else "", "error": error,
    }


def run_all(model, counter: UsageCounter, settings, modes, tasks, runs: int, say=print,
            on_row=None, keep: Path | None = None) -> list[dict]:
    """Every (run, mode, task) gets its own fresh workspace, so a mode's files never reach
    another. ``on_row(row)`` is called after each row, as it is produced. A Ctrl+C
    (KeyboardInterrupt) is not caught here: it stops the run, and the rows so far stay in ``rows``
    as ``on_row`` saw them."""
    rows = []
    for run_index in range(1, runs + 1):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
            run_tmp = Path(tmp_name)
            for mode in modes:
                for task in tasks:
                    tmp = run_tmp / mode / task.name
                    workspace = tmp / "workspace"
                    seed_files(workspace)  # the app's own files and skills, as in a real workspace
                    seed(workspace, task)
                    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
                    row = run_task(mode, model, counter, backend, task, settings, tmp, run_index, keep)
                    rows.append(row)
                    say(f"run {run_index} {mode:<6} {task.name:<20} {row['score']:.2f}  "
                        f"calls={row['calls']}  {round(row['seconds'])}s")
                    if on_row is not None:
                        on_row(row)
    return rows


def settings_from_env() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    problem = settings.chat.problem()
    if problem:
        raise SystemExit(problem)
    return settings


def save_results(path: Path, payload: dict) -> None:
    """Writes the results to a temporary file, then renames it over ``path``: a crash in
    the middle leaves the previous complete file, never half of a new one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _commit() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def rescore(payload: dict, *, source: str = "", commit: str = "", date: str = "") -> dict:
    """Scores every row without an error again with the current scorers, from the stored
    answers. Rows with an error keep their values. Pure: returns a new payload and does not
    change the one it was given."""
    by_name = {task.name: task for task in TASKS}
    rows = []
    for row in payload["rows"]:
        row = dict(row)
        if not row.get("error"):
            if row["task"] not in by_name:
                raise ValueError(f"row for unknown task {row['task']!r}; it cannot be re-scored")
            result = score(by_name[row["task"]], row["answer"])
            row["score"] = result["score"]
            row["details"] = {key: value for key, value in result.items() if key != "score"}
        rows.append(row)
    summary = summarize(rows)
    return {**payload, "rows": rows, "summary": summary, "table": format_table(summary),
            "rescored": {"from": source, "commit": commit, "date": date}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--rescore", metavar="PATH",
                        help="score a saved results file again with the current scorers, "
                             "from its stored answers: no model call, no .env. Saves "
                             "<stem>-rescored.json next to it.")
    args = parser.parse_args()

    if args.rescore:
        source = Path(args.rescore)
        if not source.is_file():
            parser.error(f"no results file at {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        new = rescore(payload, source=source.name, commit=_commit(),
                      date=datetime.now().isoformat(timespec="seconds"))
        target = source.with_name(f"{source.stem}-rescored.json")
        save_results(target, new)
        print(new["table"])
        print(f"\nSaved {target}")
        return

    names = {task.name for task in TASKS}
    unknown = [name for name in (args.tasks or []) if name not in names]
    if unknown:
        parser.error(f"unknown task(s) {', '.join(unknown)}; valid: {', '.join(sorted(names))}")
    tasks = [task for task in TASKS
             if (not args.families or task.family in args.families)
             and (not args.tasks or task.name in args.tasks)]
    if not tasks:
        parser.error("no tasks match --families and --tasks")

    settings = settings_from_env()
    model = chat_model(settings)
    counter = UsageCounter()
    model.callbacks = [counter]

    out = ROOT / "data" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"orchestration-{datetime.now():%Y%m%d-%H%M%S}"
    path = out / f"{stamp}.json"
    keep = out / stamp  # room transcripts, kept next to the results
    rows: list[dict] = []

    def save(complete: bool) -> str:
        """Rewrites the whole file: a crash or Ctrl+C keeps every row finished so far."""
        summary = summarize(rows)
        table = format_table(summary)
        save_results(path, {
            "model": settings.chat.model,
            "date": datetime.now().isoformat(timespec="seconds"),
            "commit": _commit(),
            "settings": {
                "room_max_turns": settings.room_max_turns,
                "orchestration_max_calls": settings.orchestration_max_calls,
                "subagent_max_calls": settings.subagent_max_calls,
            },
            "runs": args.runs,
            "modes": args.modes,
            "complete": complete,
            "table": table,
            "summary": summary,
            "rows": rows,
        })
        return table

    def on_row(row: dict) -> None:
        rows.append(row)
        save(False)

    try:
        run_all(model, counter, settings, args.modes, tasks, args.runs, say=print, on_row=on_row,
                keep=keep)
    except KeyboardInterrupt:
        print("\nInterrupted: the rows so far are saved.")
        print("\n" + save(False))
        print(f"\nSaved {path.relative_to(ROOT)} (incomplete)")
        raise SystemExit(130)

    print("\n" + save(True))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

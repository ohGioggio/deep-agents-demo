"""Live eval: does the agent deliver files when it should, and answer in chat otherwise?

    uv run python evals/file_output.py            # one run
    uv run python evals/file_output.py --runs 3

Uses the real chat model from .env, so it makes paid API calls (about 11 short agent
turns per run). Every run gets a fresh temporary workspace and output store: your
real files and chats are untouched.

Per prompt it reports: whether send_file delivered a file, whether the extension is
the expected one, whether the file validated, and whether the chat reply stayed
short (under 60 words) when a file was sent. A follow-up ("make the intro shorter")
must deliver v2 of the same file. Results are also saved as JSON in data/evals/.
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepagents.backends import FilesystemBackend  # noqa: E402

from deep_agents_demo.config import Settings  # noqa: E402
from deep_agents_demo.core.assembly import build_agent  # noqa: E402
from deep_agents_demo.features.catalog.feature import catalog_feature  # noqa: E402
from deep_agents_demo.features.files.feature import files_feature  # noqa: E402
from deep_agents_demo.persistence.chats import thread_id  # noqa: E402
from deep_agents_demo.persistence.outputs import OutputStore  # noqa: E402
from deep_agents_demo.providers.chat import chat_model  # noqa: E402
from deep_agents_demo.startup.workspace import seed_files  # noqa: E402

USER = "eval-user"
SHORT_REPLY = 60  # words
# (name, prompt, expected extension or None for a chat answer)
PROMPTS = [
    ("doc-sandbox", "Write a short document about the sandbox policy.", ".md"),
    ("csv-trees", "Give me a CSV of five trees with name, height and leaf type.", ".csv"),
    ("plan-park", "Write a one-page plan for a city park scene.", ".md"),
    ("svg-circle", "Make a simple SVG of a red circle.", ".svg"),
    ("html-parks", "Write a short HTML page that lists three parks.", ".html"),
    ("what-sandbox", "What is a sandbox?", None),
    ("explain-recall", "Explain what MEMORY_RECALL does.", None),
    ("catalog-trees", "How many trees does the catalog have?", None),
    ("skill-vs-agent", "What's the difference between a skill and a subagent?", None),
    ("hello", "Say hello.", None),
]
FOLLOW_UP = ("doc-sandbox", "Make the intro shorter.")


def settings_from_env() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    problem = settings.chat.problem()
    if problem:
        raise SystemExit(problem)
    return settings


def turn(agent, prompt, chat):
    config = {"configurable": {"thread_id": thread_id(USER, chat)}, "recursion_limit": 60}
    start = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)
    seconds = time.perf_counter() - start
    new = result["messages"]
    last_human = max(i for i, m in enumerate(new) if m.type == "human")
    new = new[last_human + 1:]
    sent = [m.artifact for m in new if isinstance(m, ToolMessage) and m.name == "send_file"
            and isinstance(m.artifact, dict) and m.artifact.get("saved")]
    tried = sum(tc["name"] == "send_file" for m in new if isinstance(m, AIMessage) for tc in m.tool_calls)
    answer = next((m.text for m in reversed(new) if isinstance(m, AIMessage) and not m.tool_calls), "")
    return {"seconds": round(seconds, 2), "send_file_calls": tried, "sent": sent,
            "reply_words": len(str(answer).split()), "reply": str(answer)[:400]}


def score(expected, row):
    if expected is None:
        return {"ok": not row["sent"] and row["send_file_calls"] == 0}
    files = row["sent"]
    right_type = bool(files) and files[-1]["name"].endswith(expected)
    short = row["reply_words"] < SHORT_REPLY
    return {"ok": bool(files) and right_type and short, "file": files[-1]["name"] if files else None,
            "right_type": right_type, "validated": bool(files), "short_reply": short}


def run(settings, model, run_index):
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        seed_files(tmp / "workspace")
        backend = FilesystemBackend(root_dir=str(tmp / "workspace"), virtual_mode=True)
        outputs = OutputStore(tmp / "outputs")
        agent = build_agent(model, InMemorySaver(), backend,
                            [catalog_feature(), files_feature(backend, outputs, USER)])
        for name, prompt, expected in PROMPTS:
            row = turn(agent, prompt, f"{run_index}-{name}")
            row.update(run=run_index, prompt=name, expected=expected, **score(expected, row))
            rows.append(row)
            print(f"  {name:<15} {'PASS' if row['ok'] else 'FAIL'}  file={row.get('file')} "
                  f"words={row['reply_words']} {row['seconds']}s", flush=True)
        chat, prompt = FOLLOW_UP
        first = next(r for r in rows if r["prompt"] == chat)
        row = turn(agent, prompt, f"{run_index}-{chat}")
        same = first.get("file") and row["sent"] and row["sent"][-1]["name"] == first["file"]
        row.update(run=run_index, prompt="follow-up", expected="v2 of the same file",
                   ok=bool(same and row["sent"][-1]["version"] == 2),
                   file=row["sent"][-1]["name"] if row["sent"] else None)
        rows.append(row)
        print(f"  {'follow-up':<15} {'PASS' if row['ok'] else 'FAIL'}  file={row.get('file')} "
              f"version={row['sent'][-1]['version'] if row['sent'] else None}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    settings = settings_from_env()
    model = chat_model(settings)
    rows = []
    for run_index in range(1, args.runs + 1):
        print(f"run {run_index}", flush=True)
        rows.extend(run(settings, model, run_index))
    files = [r for r in rows if r["expected"] not in (None, "v2 of the same file")]
    chats = [r for r in rows if r["expected"] is None]
    follow = [r for r in rows if r["prompt"] == "follow-up"]
    summary = (f"| Files delivered right | Chat answers kept in chat | Follow-up as v2 |\n| --- | --- | --- |\n"
               f"| {sum(r['ok'] for r in files)}/{len(files)} | {sum(r['ok'] for r in chats)}/{len(chats)} | "
               f"{sum(r['ok'] for r in follow)}/{len(follow)} |")
    print("\n" + summary)
    out = ROOT / "data" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"file_output-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"model": settings.chat.model, "runs": args.runs, "summary": summary,
                                "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

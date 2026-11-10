"""Command-line entry point for the Deep Agents demo.

    uv run deep-agents-demo
    uv run deep-agents-demo --thread another-conversation

Settings come from .env in the project root (see .env.example); existing shell
variables take precedence. Local agent files live in demo_workspace/, memory and
document vectors in data/, and conversation checkpoints in
demo_checkpoints.sqlite. The same user/thread resumes history; a new thread
keeps the user's memories. The harness owns the model/tool loop; the loop below
only handles chat. Context compression is built in, so a short demo may never
trigger it.
"""
import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.agent import build_agent
from deep_agents_demo.config import CHECKPOINTS, DATA, ENV_FILE, WORKSPACE, enabled, setting
from deep_agents_demo.memory import RECALL_MODES, AutoRecallMiddleware, close_memory, make_memory, memory_tools
from deep_agents_demo.rag import DocumentIndex
from deep_agents_demo.sandbox import clear_sandbox, make_backend
from deep_agents_demo.workspace import seed_files


def parse_args():
    parser = argparse.ArgumentParser(
        prog="deep-agents-demo", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--thread", default="city-demo")
    parser.add_argument("--user", default="demo-user", help="Mem0 user namespace; not authentication")
    parser.add_argument("--index-docs", type=Path, help="Rebuild RAG index from .md/.txt files, then exit")
    parser.add_argument("--prompt", help="Run one prompt instead of interactive chat")
    parser.add_argument("--clear-sandbox", action="store_true",
                        help="Delete all files the agent created in the OpenShell sandbox, then exit")
    return parser, parser.parse_args()


def clear_openshell_sandbox(parser) -> None:
    if setting("SANDBOX_PROVIDER", "none") != "openshell":
        parser.error("--clear-sandbox needs SANDBOX_PROVIDER=openshell")
    with ExitStack() as stack:
        backend, _prefix, workdir = make_backend(WORKSPACE, DATA, stack)
        work, tmp = clear_sandbox(backend, workdir)
    print(f"Cleared {work} entries from {workdir} and {tmp} from /tmp. "
          "The sandbox, image and policy are unchanged.")
    print("Conversation history is kept; use a new --thread for a fresh start.")


def index_docs(parser, folder: Path) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        parser.error("--index-docs must name an existing directory")
    index = DocumentIndex(DATA)
    try:
        print(f"Indexed {index.index(folder)} chunks from {folder}")
    finally:
        index.close()


def turn_input(prompt, reload_skills=False):
    """Agent input for one user turn.

    Deep Agents saves the skill list in the thread's checkpoint and loads it only
    once per thread. ``skills_metadata=None`` asks it to reload, so skills added
    or changed since the thread began (or a new sandbox route) are picked up.
    """
    payload = {"messages": [{"role": "user", "content": prompt}]}
    if reload_skills:
        payload["skills_metadata"] = None
    return payload


def stream_turn(agent, prompt, config, reload_skills=False) -> None:
    # One stream call runs the entire multi-step agent loop.
    # Include child updates so delegation/tool calls are observable.
    for namespace, update in agent.stream(
        turn_input(prompt, reload_skills),
        config=config, stream_mode="updates", subgraphs=True,
    ):
        for node, state in update.items():
            if not isinstance(state, dict):
                continue
            label = "subagent" if namespace else "main"
            for message in state.get("messages", []):
                print(f"\n[{label} / {node}]")
                message.pretty_print()
            if "todos" in state:
                print("[plan]", json.dumps(state["todos"], ensure_ascii=False))


def main() -> None:
    parser, args = parse_args()
    # Explicit path: works even when launched from another directory.
    # Existing shell environment variables take precedence over .env.
    load_dotenv(ENV_FILE, override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("AGENT_MODEL", "").strip()
    endpoint = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    if args.clear_sandbox:
        clear_openshell_sandbox(parser)
        return
    if not args.index_docs and (not api_key or api_key == "your-api-key"
                                or not model_name or model_name == "your-model-id"):
        parser.error("Fill in OPENAI_API_KEY and AGENT_MODEL in .env in the project root "
                     "(or set them in your environment).")
    seed_files(WORKSPACE)
    DATA.mkdir(exist_ok=True)
    if args.index_docs:
        index_docs(parser, args.index_docs)
        return

    model = ChatOpenAI(model=model_name, api_key=api_key, base_url=endpoint)
    config = {"configurable": {"thread_id": json.dumps([args.user, args.thread])}, "recursion_limit": 80}
    print("Try: Plan a city scene with trees, cars and buildings under 100000 triangles.")
    print("Then: Remember that I prefer pine trees.\nType exit to quit.")
    with ExitStack() as stack:
        saver = stack.enter_context(SqliteSaver.from_conn_string(str(CHECKPOINTS)))
        extras, middleware = [], []
        use_mem0 = enabled("MEMORY_ENABLED")
        recall_mode = setting("MEMORY_RECALL", "auto").lower()
        if recall_mode not in RECALL_MODES:
            parser.error(f"MEMORY_RECALL must be one of: {', '.join(RECALL_MODES)}")
        if use_mem0:
            memory = make_memory(DATA)
            stack.callback(close_memory, memory)
            extras.extend(memory_tools(memory, args.user, recall_tool=recall_mode != "auto"))
            if recall_mode != "tool":
                middleware.append(AutoRecallMiddleware(memory, args.user))
        if enabled("RAG_ENABLED"):
            index = DocumentIndex(DATA)
            stack.callback(index.close)
            extras.append(index.tool())
        backend, prefix, workdir = make_backend(WORKSPACE, DATA, stack)
        # The seeded sandbox skill describes the OpenShell image and policy.
        sandbox_skill = setting("SANDBOX_PROVIDER", "none") == "openshell"
        agent = build_agent(model, saver, backend, extras, prefix, use_mem0, workdir, sandbox_skill,
                            recall_mode, middleware)
        reload_skills = True  # once per session, on the first turn
        while True:
            try:
                prompt = args.prompt or input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
            stream_turn(agent, prompt, config, reload_skills)
            reload_skills = False
            if args.prompt:
                break


if __name__ == "__main__":
    main()

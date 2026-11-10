"""Command-line entry point for the Deep Agents demo.

    uv run deep-agents-demo
    uv run deep-agents-demo --thread another-conversation
    uv run deep-agents-demo --list-threads

Inside a chat, type /help for chat commands (/threads, /new, /switch, ...).

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
import sys
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.agent import build_agent
from deep_agents_demo.config import (CHECKPOINTS, DATA, ENV_FILE, WORKSPACE, enabled, setting,
                                     shared_rate_limiter)
from deep_agents_demo.memory import RECALL_MODES, AutoRecallMiddleware, close_memory, make_memory, memory_tools
from deep_agents_demo.rag import DocumentIndex
from deep_agents_demo.sandbox import clear_sandbox, make_backend
from deep_agents_demo.threads import (Chats, format_table, format_transcript, new_name, thread_id,
                                      valid_name)
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
    chats = parser.add_argument_group("chat management (for --user; each exits when done)")
    chats.add_argument("--list-threads", action="store_true", help="List chats, most recent first")
    chats.add_argument("--show-thread", metavar="NAME", help="Print a chat's transcript")
    chats.add_argument("--delete-thread", metavar="NAME", help="Delete a chat and its history")
    chats.add_argument("--rename-thread", nargs=2, metavar=("OLD", "NEW"), help="Rename a chat")
    chats.add_argument("--yes", action="store_true", help="Don't ask before deleting")
    return parser, parser.parse_args()


HELP = """Chat commands:
  /threads        list your chats (* marks the current one)
  /new [name]     start a new chat (a dated name if you give none)
  /switch NAME    continue another chat
  /history        show this chat's messages
  /rename NAME    rename this chat
  /delete NAME    delete a chat and its history
  /help           show this help
  exit            quit"""


def confirm(question: str, ask=input) -> bool:
    try:
        return ask(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


class ChatSession:
    """Tracks the current chat and handles slash commands typed in the chat loop."""

    def __init__(self, chats: Chats, current: str, ask=input, say=print):
        self.chats, self.current, self.ask, self.say = chats, current, ask, say
        self.reload_skills = True  # reload once per chat, on its first turn this session

    @property
    def config(self):
        return {"configurable": {"thread_id": thread_id(self.chats.user, self.current)},
                "recursion_limit": 80}

    def _enter(self, name, note):
        self.current, self.reload_skills = name, True
        self.say(note)

    def handle(self, line: str) -> bool:
        """Run a slash command. Returns False if ``line`` is not a command."""
        if not line.startswith("/"):
            return False
        command, _, arg = line[1:].partition(" ")
        command, arg = command.lower(), arg.strip()
        if command == "help":
            self.say(HELP)
        elif command in {"threads", "chats", "list"}:
            table = format_table(self.chats.list(), self.current)
            if not self.chats.exists(self.current):
                table += f"\n*  {self.current} (new: saved after your first message)"
            self.say(table)
        elif command == "new":
            name = arg or new_name()
            if not valid_name(name):
                self.say(f"Invalid name {name!r}: use letters, digits, '.', '_' or '-'.")
            elif self.chats.exists(name):
                self.say(f"A chat named {name!r} already exists; use /switch {name}.")
            else:
                self._enter(name, f"New chat {name!r}. It is saved once you send a message.")
        elif command == "switch":
            if not arg:
                self.say("Usage: /switch NAME (see /threads)")
            elif not self.chats.exists(arg):
                self.say(f"No chat named {arg!r}. See /threads, or start one with /new {arg}.")
            else:
                self._enter(arg, f"Switched to {arg!r} ({self.chats.info(arg).turns} turns).")
        elif command == "history":
            if self.chats.exists(self.current):
                self.say(format_transcript(self.chats.transcript(self.current)))
            else:
                self.say("(empty chat)")
        elif command == "rename":
            try:
                self.chats.rename(self.current, arg)
            except ValueError as error:
                self.say(str(error))
            else:
                self.current = arg
                self.say(f"Renamed to {arg!r}.")
        elif command == "delete":
            if not arg or not self.chats.exists(arg):
                self.say(f"No chat named {arg!r}. See /threads.")
            elif confirm(f"Delete chat {arg!r} ({self.chats.info(arg).turns} turns)?", self.ask):
                self.chats.delete(arg)
                current = arg == self.current
                self.say(f"Deleted {arg!r}." + (" You are still in it, now empty." if current else ""))
                if current:
                    self.reload_skills = True
            else:
                self.say("Kept.")
        else:
            self.say(f"Unknown command /{command}. Type /help.")
        return True


def manage_threads(parser, args) -> None:
    """--list-threads, --show-thread, --delete-thread, --rename-thread. No model calls."""
    with SqliteSaver.from_conn_string(str(CHECKPOINTS)) as saver:
        # The agent only rebuilds saved messages; its model is never called.
        agent = build_agent(ChatOpenAI(model="unused", api_key="unused"), saver,
                            FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True))
        chats = Chats(saver, agent, args.user)
        if args.list_threads:
            print(format_table(chats.list(), args.thread))
        elif args.show_thread:
            if not chats.exists(args.show_thread):
                parser.error(f"no chat named {args.show_thread!r} for user {args.user!r}")
            print(format_transcript(chats.transcript(args.show_thread)))
        elif args.delete_thread:
            name = args.delete_thread
            if not chats.exists(name):
                parser.error(f"no chat named {name!r} for user {args.user!r}")
            if args.yes or confirm(f"Delete chat {name!r} ({chats.info(name).turns} turns)?"):
                chats.delete(name)
                print(f"Deleted {name!r}.")
            else:
                print("Kept.")
        elif args.rename_thread:
            old, new = args.rename_thread
            try:
                chats.rename(old, new)
            except ValueError as error:
                parser.error(str(error))
            print(f"Renamed {old!r} to {new!r}.")


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
    # Chats can contain any character; on a legacy console or a redirected pipe
    # (Windows cp1252), print a replacement instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
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
    if args.list_threads or args.show_thread or args.delete_thread or args.rename_thread:
        manage_threads(parser, args)
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

    # Shares API_REQUESTS_PER_MINUTE with the embeddings; retries ride out transient errors.
    model = ChatOpenAI(model=model_name, api_key=api_key, base_url=endpoint,
                       rate_limiter=shared_rate_limiter(), max_retries=6)
    interactive = not args.prompt
    # Startup chatter only helps in an interactive chat; --prompt prints just the answer.
    say = print if interactive else (lambda *_args, **_kwargs: None)
    say("Ask anything. Ideas: a question about your indexed documents, "
          "'Remember that I prefer short answers', or the example skill: "
          "'Plan a city scene under 100000 triangles'.")
    say("Type /help for chat commands, exit to quit.")
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
        doc_sources = None
        if enabled("RAG_ENABLED"):
            index = DocumentIndex(DATA)
            stack.callback(index.close)
            extras.append(index.tool())
            doc_sources = index.sources()  # listed in the prompt so the agent knows to search
        backend, prefix, workdir = make_backend(WORKSPACE, DATA, stack, say)
        # The seeded sandbox skill describes the OpenShell image and policy.
        sandbox_skill = setting("SANDBOX_PROVIDER", "none") == "openshell"
        agent = build_agent(model, saver, backend, extras, prefix, use_mem0, workdir, sandbox_skill,
                            recall_mode, middleware, doc_sources)
        session = ChatSession(Chats(saver, agent, args.user), args.thread)
        while True:
            try:
                prompt = args.prompt or input(f"\n[{session.current}] You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in {"exit", "quit", "/exit", "/quit"}:
                break
            if not prompt or session.handle(prompt):
                continue
            stream_turn(agent, prompt, session.config, session.reload_skills)
            session.reload_skills = False
            if args.prompt:
                break


if __name__ == "__main__":
    main()

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
import sys
from pathlib import Path

from dotenv import load_dotenv

from deep_agents_demo import commands
from deep_agents_demo.chat import run_chat
from deep_agents_demo.config import ENV_FILE, Settings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="deep-agents-demo", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--thread", default="city-demo")
    parser.add_argument("--user", default="demo-user", help="Mem0 user namespace; not authentication")
    parser.add_argument("--index-docs", type=Path, help="Rebuild RAG index from .md/.txt files, then exit")
    parser.add_argument("--prompt", help="Run one prompt instead of interactive chat (prints the answer)")
    parser.add_argument("--verbose", action="store_true",
                        help="With --prompt, print every step (tool calls, subagents), not just the answer")
    parser.add_argument("--clear-sandbox", action="store_true",
                        help="Delete all files the agent created in the OpenShell sandbox, then exit")
    chats = parser.add_argument_group("chat management (for --user; each exits when done)")
    chats.add_argument("--list-threads", action="store_true", help="List chats, most recent first")
    chats.add_argument("--show-thread", metavar="NAME", help="Print a chat's transcript")
    chats.add_argument("--delete-thread", metavar="NAME", help="Delete a chat and its history")
    chats.add_argument("--rename-thread", nargs=2, metavar=("OLD", "NEW"), help="Rename a chat")
    chats.add_argument("--yes", action="store_true", help="Don't ask before deleting")
    return parser, parser.parse_args(argv)


def dispatch(settings: Settings, args) -> None:
    """Run the one action the arguments ask for."""
    if args.clear_sandbox:
        commands.clear_openshell_sandbox(settings)
    elif args.list_threads:
        commands.list_threads(settings, args.user, args.thread)
    elif args.show_thread:
        commands.show_thread(settings, args.user, args.show_thread)
    elif args.delete_thread:
        commands.delete_thread(settings, args.user, args.delete_thread, yes=args.yes)
    elif args.rename_thread:
        commands.rename_thread(settings, args.user, *args.rename_thread)
    elif args.index_docs:
        commands.index_docs(settings, args.index_docs)
    else:
        problem = settings.chat.problem()
        if problem:
            raise commands.UsageError(problem)
        run_chat(settings, args.user, args.thread, args.prompt, verbose=args.verbose)


def main(argv=None) -> None:
    # Chats can contain any character; on a legacy console or a redirected pipe
    # (Windows cp1252), print a replacement instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser, args = parse_args(argv)
    # Explicit path: works even when launched from another directory.
    # Existing shell environment variables take precedence over .env.
    load_dotenv(ENV_FILE, override=False)
    try:
        settings = Settings.from_env()  # invalid settings are reported up front
    except ValueError as error:
        parser.error(str(error))
    try:
        dispatch(settings, args)
    except commands.UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

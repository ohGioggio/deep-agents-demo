"""One-shot commands: index documents, clear the sandbox, manage chats.

They report bad input by raising ``UsageError``; the CLI turns that into its
usual error message, so these functions don't depend on argparse.
"""
import sys
from contextlib import ExitStack
from pathlib import Path

from deep_agents_demo.chat import confirm
from deep_agents_demo.config import Settings
from deep_agents_demo.features.sandbox.providers import clear_sandbox
from deep_agents_demo.startup.resources import document_index, sandbox_backend
from deep_agents_demo.startup.runtime import history_runtime
from deep_agents_demo.startup.workspace import prepare_workspace
from deep_agents_demo.ui.cli.render import format_table, format_transcript


class UsageError(Exception):
    """The user asked for something that can't be done as given."""


def index_docs(settings: Settings, folder: Path) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        raise UsageError("--index-docs must name an existing directory")
    prepare_workspace(settings)
    index = document_index(settings, warn=lambda message: print(message, file=sys.stderr))
    try:
        print(f"Indexed {index.index(folder)} chunks from {folder}")
    finally:
        index.close()


def clear_openshell_sandbox(settings: Settings) -> None:
    if settings.sandbox.provider != "openshell":
        raise UsageError("--clear-sandbox needs SANDBOX_PROVIDER=openshell")
    with ExitStack() as stack:
        setup = sandbox_backend(settings, stack, notify=print)
        workdir = setup.workdir
        work, tmp = clear_sandbox(setup.backend, workdir)
    print(f"Cleared {work} entries from {workdir} and {tmp} from /tmp. "
          "The sandbox, image and policy are unchanged.")
    print("Conversation history is kept; use a new --thread for a fresh start.")


def list_threads(settings: Settings, user: str, current: str) -> None:
    with ExitStack() as stack:
        print(format_table(history_runtime(settings, user, stack).chats.list(), current))


def show_thread(settings: Settings, user: str, name: str) -> None:
    with ExitStack() as stack:
        chats = history_runtime(settings, user, stack).chats
        if not chats.exists(name):
            raise UsageError(f"no chat named {name!r} for user {user!r}")
        print(format_transcript(chats.transcript(name)))


def delete_thread(settings: Settings, user: str, name: str, yes: bool = False, ask=input) -> None:
    with ExitStack() as stack:
        chats = history_runtime(settings, user, stack).chats
        if not chats.exists(name):
            raise UsageError(f"no chat named {name!r} for user {user!r}")
        if yes or confirm(f"Delete chat {name!r} ({chats.info(name).turns} turns)?", ask):
            chats.delete(name)
            print(f"Deleted {name!r}.")
        else:
            print("Kept.")


def rename_thread(settings: Settings, user: str, old: str, new: str) -> None:
    with ExitStack() as stack:
        try:
            history_runtime(settings, user, stack).chats.rename(old, new)
        except ValueError as error:
            raise UsageError(str(error)) from None
    print(f"Renamed {old!r} to {new!r}.")

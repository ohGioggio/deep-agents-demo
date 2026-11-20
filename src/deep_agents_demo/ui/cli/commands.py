"""One-shot commands: index documents, clear the sandbox, manage chats.

They report bad input by raising ``UsageError``; the CLI turns that into its
usual error message, so these functions don't depend on argparse.
"""
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from deep_agents_demo.config import Settings
from deep_agents_demo.features.sandbox.providers import clear_sandbox
from deep_agents_demo.persistence.outputs import OutputStore
from deep_agents_demo.persistence.runs import RunStore
from deep_agents_demo.startup.resources import document_index, sandbox_backend
from deep_agents_demo.startup.runtime import history_runtime
from deep_agents_demo.startup.workspace import prepare_workspace
from deep_agents_demo.ui.cli.render import format_files, format_run, format_runs, format_table, format_transcript
from deep_agents_demo.ui.cli.repl import confirm
from deep_agents_demo.ui.cli.richui.theme import make_console
from deep_agents_demo.ui.cli.richui.tables import RichTables


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


def list_files(settings: Settings, user: str, thread: str) -> None:
    outputs = OutputStore(settings.paths.data / "outputs")
    print(format_files(outputs.list(user, thread), outputs.folder(user, thread)))


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
            try:
                chats.delete(name)
            except ValueError as error:
                raise UsageError(str(error)) from error
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


def _runs_store(settings: Settings, stack: ExitStack) -> RunStore:
    settings.paths.data.mkdir(parents=True, exist_ok=True)
    store = RunStore(settings.paths.data / "runs.sqlite")
    stack.callback(store.close)
    return store


def list_runs(settings: Settings, user: str, limit: int = 20) -> None:
    if limit < 1:
        raise UsageError("--runs needs a number of at least 1")
    start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with ExitStack() as stack:
        store = _runs_store(settings, stack)
        runs = store.recent(user, limit)
        totals = store.totals(user, since=start_of_today)
    if sys.stdout.isatty():
        RichTables(make_console()).runs(runs, totals)
    else:
        print(format_runs(runs, totals))


def show_run(settings: Settings, user: str, run_id: str) -> None:
    with ExitStack() as stack:
        run = _runs_store(settings, stack).get(user, run_id)
    if run is None:
        print(f"No run with id {run_id!r} for user {user!r}. List your runs with --runs.", file=sys.stderr)
        raise SystemExit(1)
    if sys.stdout.isatty():
        RichTables(make_console()).run(run)
    else:
        print(format_run(run))

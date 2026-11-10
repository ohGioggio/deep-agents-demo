"""Backend selection: local files only, or a Daytona / OpenShell sandbox."""
import shlex

from deep_agents_demo.config import setting

PROVIDERS = ("none", "daytona", "openshell")

# Empties the working directory and removes the sandbox user's own files in /tmp.
# Prints "<workdir entries> <tmp entries>" removed (top level only).
_CLEAR_SCRIPT = """\
set -u
W={workdir}
[ -d "$W" ] || {{ echo "missing working directory: $W" >&2; exit 3; }}
work=$(find "$W" -mindepth 1 -maxdepth 1 | wc -l)
tmp=$(find /tmp -mindepth 1 -maxdepth 1 -user "$(id -u)" | wc -l)
find "$W" -mindepth 1 -delete
find /tmp -mindepth 1 -user "$(id -u)" -delete 2>/dev/null || true
echo "$work $tmp"
"""


def clear_sandbox(backend, workdir):
    """Delete everything in the sandbox working directory and the user's /tmp files.

    Returns ``(workdir_entries, tmp_entries)`` removed. The sandbox, its image and
    its policy are untouched; only files the agent created are deleted.
    """
    if not workdir or not workdir.startswith("/") or workdir.rstrip("/") == "":
        raise ValueError(f"Refusing to clear unsafe working directory: {workdir!r}")
    result = backend.execute(_CLEAR_SCRIPT.format(workdir=shlex.quote(workdir)), timeout=120)
    if result.exit_code != 0:
        raise RuntimeError(f"Clearing the sandbox failed: {result.output.strip()}")
    work, tmp = result.output.split()[-2:]
    return int(work), int(tmp)


def make_backend(workspace, data, stack):
    """Return ``(backend, skills_prefix, workdir)``.

    With a sandbox, normal paths are remote and only ``/instructions/`` maps to
    the local skills. ``stack`` (an ExitStack) receives any cleanup callbacks.
    """
    from deepagents.backends import CompositeBackend, FilesystemBackend
    provider = setting("SANDBOX_PROVIDER", "none")
    local = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    if provider == "none":
        return local, "", None
    if provider == "daytona":
        remote, workdir = _daytona(data)
    elif provider == "openshell":
        remote, workdir = _openshell(stack)
    else:
        raise ValueError(f"SANDBOX_PROVIDER must be one of: {', '.join(PROVIDERS)}")
    backend = CompositeBackend(default=remote, routes={"/instructions/": local})
    return backend, "/instructions", workdir


def _daytona(data):
    if not setting("DAYTONA_API_KEY"):
        raise ValueError("Set DAYTONA_API_KEY to enable sandbox execution")
    from daytona import Daytona
    from langchain_daytona import DaytonaSandbox
    client = Daytona()
    id_file = data / "daytona_sandbox_id.txt"
    sandbox_id = setting("DAYTONA_SANDBOX_ID")
    if not sandbox_id and id_file.exists():
        sandbox_id = id_file.read_text().strip()
    sandbox = client.get(sandbox_id) if sandbox_id else client.create()
    id_file.write_text(sandbox.id)
    print(f"Daytona sandbox: {sandbox.id} (retained; manage/delete it in Daytona)")
    return DaytonaSandbox(sandbox=sandbox, timeout=120), "/home/daytona"


def _openshell(stack):
    from deep_agents_demo.openshell_backend import OpenShellBackend, connect
    name = setting("OPENSHELL_SANDBOX_NAME", "deep-agents-demo")
    workspace = setting("OPENSHELL_WORKSPACE", "default")
    workdir = setting("OPENSHELL_WORKDIR", "/sandbox")
    client, session = connect(name, workspace)
    stack.callback(client.close)
    print(f"OpenShell sandbox: {name} (workspace {workspace}; retained, delete it with the CLI)")
    return OpenShellBackend(session, workdir=workdir), workdir

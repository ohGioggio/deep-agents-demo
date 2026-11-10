"""Backend selection: local files only, or a Daytona / OpenShell sandbox."""
import shlex

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox


# Empties the working directory and removes the sandbox user's own top-level
# entries in the temp directory. If the working directory lives inside the temp
# directory, the entry that contains it is kept.
# Prints "<workdir entries> <tmp entries>" removed (top level only).
_CLEAR_SCRIPT = """\
set -u
W={workdir}
T={tmpdir}
[ -d "$W" ] || {{ echo "missing working directory: $W" >&2; exit 3; }}
KEEP=
case "$W/" in "$T"/?*) rel=${{W#"$T"/}}; KEEP="$T/${{rel%%/*}}" ;; esac
work=$(find "$W" -mindepth 1 -maxdepth 1 | wc -l)
find "$W" -mindepth 1 -delete
tmp=0
if [ -d "$T" ]; then
  tmp=$(find "$T" -mindepth 1 -maxdepth 1 -user "$(id -u)" ! -path "$KEEP" | wc -l)
  find "$T" -mindepth 1 -maxdepth 1 -user "$(id -u)" ! -path "$KEEP" -exec rm -rf -- {{}} + 2>/dev/null || true
fi
echo "$work $tmp"
"""


def _safe_dir(path, what):
    if not path or not path.startswith("/") or path.rstrip("/") == "":
        raise ValueError(f"Refusing to clear unsafe {what}: {path!r}")
    return path.rstrip("/")


def clear_sandbox(backend, workdir, tmpdir="/tmp"):
    """Delete everything in the sandbox working directory and the user's temp files.

    Returns ``(workdir_entries, tmp_entries)`` removed. The sandbox, its image and
    its policy are untouched; only files the agent created are deleted.
    """
    workdir, tmpdir = _safe_dir(workdir, "working directory"), _safe_dir(tmpdir, "temp directory")
    script = _CLEAR_SCRIPT.format(workdir=shlex.quote(workdir), tmpdir=shlex.quote(tmpdir))
    result = backend.execute(script, timeout=120)
    if result.exit_code != 0:
        raise RuntimeError(f"Clearing the sandbox failed: {result.output.strip()}")
    work, tmp = result.output.split()[-2:]
    return int(work), int(tmp)


class LazySandbox(BaseSandbox):
    """Connects to the real sandbox on first use, not when the chat starts.

    Chats that never run a command or touch a sandbox file don't need the
    sandbox (or its gateway and Docker) at all. Deep Agents still sees a sandbox,
    so the execute tool is offered, and every file tool goes through execute,
    upload_files and download_files, which are the only methods wrapped here.
    If the sandbox can't be reached, the tool gets an error message instead of
    the chat crashing, and the next call tries to connect again.
    """

    def __init__(self, connect, label, say=print):
        self._connect, self._label, self._say = connect, label, say
        self._backend = None

    @property
    def connected(self) -> bool:
        return self._backend is not None

    @property
    def id(self) -> str:
        return self._backend.id if self._backend is not None else self._label

    def _get(self):
        if self._backend is None:
            self._backend = self._connect()
            self._say(f"Connected to {self._label} (first use in this session).")
        return self._backend

    def _unavailable(self, error) -> str:
        return (f"Sandbox unavailable ({self._label}): {error}. The command did not run. "
                "Tell the user the sandbox could not be reached.")

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        try:
            backend = self._get()
        except Exception as error:  # gateway down, Docker off, sandbox missing...
            return ExecuteResponse(output=self._unavailable(error), exit_code=1, truncated=False)
        return backend.execute(command, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        try:
            backend = self._get()
        except Exception:
            return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]
        return backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        try:
            backend = self._get()
        except Exception:
            return [FileDownloadResponse(path=path, content=None, error="file_not_found") for path in paths]
        return backend.download_files(paths)


def make_backend(settings, workspace, data, stack, say=print):
    """Return ``(backend, skills_prefix, workdir)``.

    With a sandbox, normal paths are remote and only ``/instructions/`` maps to
    the local skills. ``settings`` is a ``config.SandboxSettings`` (already
    validated); ``stack`` (an ExitStack) receives any cleanup callbacks.
    """
    from deepagents.backends import CompositeBackend, FilesystemBackend
    local = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    if settings.provider == "none":
        return local, "", None
    if settings.provider == "daytona":
        remote = LazySandbox(lambda: _daytona(data, settings.daytona_sandbox_id),
                             "Daytona sandbox", say)
        workdir = "/home/daytona"
    elif settings.provider == "openshell":
        workdir = settings.openshell_workdir
        remote = LazySandbox(lambda: _openshell(stack, settings),
                             f"OpenShell sandbox '{settings.openshell_name}'", say)
    else:
        raise ValueError(f"Unknown sandbox provider {settings.provider!r}")
    backend = CompositeBackend(default=remote, routes={"/instructions/": local})
    return backend, "/instructions", workdir


def _daytona(data, sandbox_id=None):
    # The Daytona client reads DAYTONA_API_KEY from the environment itself.
    from daytona import Daytona
    from langchain_daytona import DaytonaSandbox
    client = Daytona()
    id_file = data / "daytona_sandbox_id.txt"
    if not sandbox_id and id_file.exists():
        sandbox_id = id_file.read_text().strip()
    sandbox = client.get(sandbox_id) if sandbox_id else client.create()
    id_file.write_text(sandbox.id)
    return DaytonaSandbox(sandbox=sandbox, timeout=120)


def _openshell(stack, settings):
    from deep_agents_demo.openshell_backend import OpenShellBackend, connect
    client, session = connect(settings.openshell_name, settings.openshell_workspace)
    stack.callback(client.close)
    return OpenShellBackend(session, workdir=settings.openshell_workdir)

"""Backend selection: local files only, or a Daytona / OpenShell sandbox."""
import shlex
from dataclasses import dataclass

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from deep_agents_demo.agent import Feature


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


@dataclass(frozen=True)
class Remote:
    """A configured remote sandbox: the (lazily connecting) backend and where to work."""
    backend: LazySandbox
    workdir: str
    openshell_skill: bool = False  # offer the OpenShell sandbox skill


def _daytona(settings, data, stack, say) -> Remote:
    def connect():
        # The Daytona client reads DAYTONA_API_KEY from the environment itself.
        from daytona import Daytona
        from langchain_daytona import DaytonaSandbox
        client = Daytona()
        id_file = data / "daytona_sandbox_id.txt"
        sandbox_id = settings.daytona_sandbox_id
        if not sandbox_id and id_file.exists():
            sandbox_id = id_file.read_text().strip()
        sandbox = client.get(sandbox_id) if sandbox_id else client.create()
        id_file.write_text(sandbox.id)
        return DaytonaSandbox(sandbox=sandbox, timeout=120)

    return Remote(LazySandbox(connect, "Daytona sandbox", say), "/home/daytona")


def _openshell(settings, data, stack, say) -> Remote:
    def connect():
        from deep_agents_demo.openshell_backend import OpenShellBackend
        from deep_agents_demo.openshell_backend import connect as open_session
        client, session = open_session(settings.openshell_name, settings.openshell_workspace)
        stack.callback(client.close)
        return OpenShellBackend(session, workdir=settings.openshell_workdir)

    label = f"OpenShell sandbox '{settings.openshell_name}'"
    return Remote(LazySandbox(connect, label, say), settings.openshell_workdir, openshell_skill=True)


@dataclass(frozen=True)
class BackendSetup:
    """Where the agent's file and shell tools act, and what it needs to know about it."""
    backend: object
    skills_prefix: str = ""          # where the local workspace (skills, file memory) is mounted
    feature: object = None           # sandbox rules for the agent, or None for local files
    workdir: str | None = None       # the sandbox working directory


# Remote sandbox providers by SANDBOX_PROVIDER value ("none" means local files).
# To add one, write a function like the two above and register it here; the
# settings check in config.SANDBOX_PROVIDERS must list the same names (a test
# keeps them in sync).
PROVIDERS = {"daytona": _daytona, "openshell": _openshell}


def make_backend(settings, workspace, data, stack, say=print):
    """Return a ``BackendSetup`` for the configured provider.

    Without a sandbox: local files, no prefix, no feature. With one, normal paths
    are remote and only ``/instructions/`` maps to the local workspace (skills and
    file memory); the feature tells the agent how to work in the sandbox.
    ``settings`` is a validated ``config.SandboxSettings``; ``stack`` (an
    ExitStack) receives cleanup callbacks.
    """
    from deepagents.backends import CompositeBackend, FilesystemBackend
    local = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    if settings.provider == "none":
        return BackendSetup(local)
    try:
        provider = PROVIDERS[settings.provider]
    except KeyError:
        raise ValueError(f"Unknown sandbox provider {settings.provider!r}") from None
    remote = provider(settings, data, stack, say)
    backend = CompositeBackend(default=remote.backend, routes={"/instructions/": local})
    return BackendSetup(backend, "/instructions",
                        sandbox_feature(remote.workdir, remote.openshell_skill), remote.workdir)


def sandbox_feature(workdir, openshell_skill=False) -> Feature:
    """Rules for working in a sandbox; with OpenShell, also its skill folder."""
    prompt = (f"You are using a sandbox. Keep code and output under {workdir}; when a skill or "
              f"the user names a path starting with /, put it under {workdir} instead. "
              "Local /instructions paths contain skills only. ")
    if openshell_skill:
        # The sandbox skill describes the OpenShell image; offering it without that
        # sandbox misleads the agent (it told users to pip install inside /sandbox).
        prompt += ("Before your first shell command (execute), read the sandbox skill and "
                   "follow it for installing packages and handling blocked network access; "
                   "other tools don't need it. ")
    return Feature(name="sandbox", prompt=prompt,
                   skill_dirs=("sandbox-skills/",) if openshell_skill else ())

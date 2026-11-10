"""Deep Agents backend that runs commands in a local NVIDIA OpenShell sandbox.

Isolation comes from the sandbox policy (policy.yaml), enforced by the kernel
and the OpenShell network proxy, not from instructions in the prompt.

BaseSandbox implements read, write, edit, ls, grep and glob as shell commands
run through ``execute``, so only execution and file transfer are needed here.
Adapted from langchain-ai/openshell-deepagent for the openshell 0.1 SDK, where
every gateway call is scoped to a workspace.
"""
import base64
import shlex

import grpc
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

IS_DIRECTORY_EXIT = 3


class OpenShellBackend(BaseSandbox):
    """Wrap an open ``openshell.SandboxSession``."""

    def __init__(self, session, *, workdir="/sandbox", default_timeout=120):
        self._session = session
        self._workdir = workdir
        self._default_timeout = default_timeout

    @property
    def id(self) -> str:
        return self._session.id

    def _exec(self, command, *, stdin=None, timeout=None):
        return self._session.exec(
            ["bash", "-c", command], workdir=self._workdir, stdin=stdin,
            timeout_seconds=timeout if timeout is not None else self._default_timeout,
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        from openshell import SandboxError
        try:
            result = self._exec(command, timeout=timeout)
        except (grpc.RpcError, SandboxError) as error:
            # Report gateway failures to the agent instead of ending the run.
            return ExecuteResponse(output=f"OpenShell error: {error}", exit_code=1, truncated=False)
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            command = f"mkdir -p -- {shlex.quote(path.rsplit('/', 1)[0] or '/')} && cat > {shlex.quote(path)}"
            try:
                ok = self._exec(command, stdin=content).exit_code == 0
            except Exception:
                ok = False
            responses.append(FileUploadResponse(path=path, error=None if ok else "permission_denied"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for path in paths:
            quoted = shlex.quote(path)
            command = f"if [ -d {quoted} ]; then exit {IS_DIRECTORY_EXIT}; fi; base64 -- {quoted}"
            try:
                result = self._exec(command)
            except Exception:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue
            if result.exit_code == 0:
                responses.append(FileDownloadResponse(
                    path=path, content=base64.b64decode(result.stdout), error=None))
            else:
                error = "is_directory" if result.exit_code == IS_DIRECTORY_EXIT else "file_not_found"
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses


def connect(name, workspace="default", timeout_seconds=120):
    """Return ``(client, session)`` for an existing, ready sandbox.

    The sandbox must be created beforehand with the CLI and ``--policy``, so the
    agent never runs in a sandbox that lacks the project policy.
    """
    from openshell import SandboxClient, SandboxSession
    client = SandboxClient.from_active_cluster()
    try:
        client.get(name, workspace=workspace)
    except grpc.RpcError as error:
        client.close()
        if error.code() == grpc.StatusCode.NOT_FOUND:
            raise ValueError(
                f"OpenShell sandbox {name!r} not found in workspace {workspace!r}. Create it with: "
                f"docker build -t deep-agents-demo-sandbox:0.1 sandbox/ && "
                f"openshell sandbox create --name {name} --from deep-agents-demo-sandbox:0.1 "
                f"--policy policy.yaml --detach"
            ) from None
        raise
    try:
        ref = client.wait_ready(name, workspace=workspace, timeout_seconds=timeout_seconds)
    except BaseException:
        client.close()
        raise
    return client, SandboxSession(client, ref)

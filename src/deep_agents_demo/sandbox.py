"""Filesystem backend selection: local files or a remote Daytona sandbox."""
from deep_agents_demo.config import setting


def make_backend(workspace, data):
    from deepagents.backends import CompositeBackend, FilesystemBackend
    provider = setting("SANDBOX_PROVIDER", "none")
    local = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    if provider == "none":
        return local, "", None
    if provider == "local":
        from deepagents.backends import LocalShellBackend

        backend = LocalShellBackend(
            root_dir=str(workspace),
            virtual_mode=True,
            inherit_env=True,
        )
        return backend, "", None
    if provider != "daytona":
        raise ValueError("SANDBOX_PROVIDER must be none or daytona")
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
    remote = DaytonaSandbox(sandbox=sandbox, timeout=120)
    # Local skills remain available, but all normal coding paths are remote.
    backend = CompositeBackend(default=remote, routes={"/instructions/": local})
    print(f"Daytona sandbox: {sandbox.id} (retained; manage/delete it in Daytona)")
    return backend, "/instructions", sandbox

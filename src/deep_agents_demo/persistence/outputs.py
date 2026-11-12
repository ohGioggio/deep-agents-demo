"""Files the agent delivered in a chat, kept outside its workspace with every version.

    <root>/<user>/<chat>/<name>                    the latest version
    <root>/<user>/<chat>/.versions/<name>.v1, .v2  every version, never changed
    <root>/<user>/<chat>/.index.json               one record per version

The agent can't write here (its file tools act on the workspace or a sandbox), so a
delivered version stays as it was delivered. Names that differ only in case are the
same file (Windows folders don't tell them apart), and every path stays inside the
root, whatever the user or chat is called.
"""
import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from deep_agents_demo.persistence.files import safe_folder_name

INDEX = ".index.json"  # a delivered file can't be named this (clean_name refuses leading dots)


def _part(name: str) -> str:
    """One folder name for a user or chat; never "." or ".." (they would leave the root)."""
    safe = safe_folder_name(name)
    return safe.replace(".", "_") if set(safe) == {"."} else safe


@dataclass(frozen=True)
class OutputFile:
    name: str
    version: int
    bytes: int
    kind: str            # the extension without the dot, e.g. "md"
    caption: str
    saved_at: str        # ISO 8601, UTC
    path: Path           # the latest copy

    def to_dict(self) -> dict:
        return {**asdict(self), "path": str(self.path)}


def _write(target: Path, data: bytes) -> None:
    """Write through a temporary file, so a crash never leaves half a file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class OutputStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.RLock()  # tool calls in one step run in parallel threads

    def folder(self, user: str, chat: str) -> Path:
        return self.root / _part(user) / _part(chat)

    def _inside(self, folder: Path) -> Path:
        """``folder``, after checking it is inside the root (never delete or move outside)."""
        if self.root.resolve() not in folder.resolve().parents:
            raise ValueError(f"{folder} is outside the output store")
        return folder

    def _index(self, folder: Path) -> list[dict]:
        path = folder / INDEX
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _file(self, folder: Path, record: dict) -> OutputFile:
        return OutputFile(record["name"], record["version"], record["bytes"], record["kind"],
                          record["caption"], record["saved_at"], folder / record["name"])

    def save(self, user: str, chat: str, name: str, data: bytes, caption: str = "") -> OutputFile:
        """Store ``data`` as the next version of ``name`` in this chat."""
        with self.lock:
            folder = self.folder(user, chat)
            index = self._index(folder)
            # The name it was first delivered under, if it differs only in case.
            name = next((r["name"] for r in index if r["name"].casefold() == name.casefold()), name)
            version = 1 + max((r["version"] for r in index if r["name"] == name), default=0)
            kept = folder / ".versions" / f"{name}.v{version}"
            _write(kept, data)
            try:
                _write(folder / name, data)
            except BaseException:  # e.g. the latest copy is open in another program
                kept.unlink(missing_ok=True)
                raise
            record = {"name": name, "version": version, "bytes": len(data),
                      "kind": Path(name).suffix.lstrip(".").lower(), "caption": caption,
                      "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            _write(folder / INDEX, json.dumps(index + [record], indent=2).encode("utf-8"))
            return self._file(folder, record)

    def list(self, user: str, chat: str) -> list[OutputFile]:
        """The latest version of each file, in the order they were first delivered."""
        with self.lock:
            folder = self.folder(user, chat)
            latest = {}
            for record in self._index(folder):
                latest[record["name"]] = record  # dicts keep first-insertion order
            return [self._file(folder, record) for record in latest.values()]

    def read(self, user: str, chat: str, name: str, version: int | None = None) -> bytes:
        with self.lock:
            folder = self.folder(user, chat)
            records = [r for r in self._index(folder) if r["name"].casefold() == name.casefold()]
            versions = [r["version"] for r in records]
            if not versions or (version is not None and version not in versions):
                raise KeyError(f"{name} v{version}" if version else name)
            return (folder / ".versions" / f"{records[0]['name']}.v{version or max(versions)}").read_bytes()

    def delete_thread(self, user: str, chat: str) -> None:
        with self.lock:
            folder = self._inside(self.folder(user, chat))
            if folder.exists():
                shutil.rmtree(folder)

    def rename_thread(self, user: str, old: str, new: str) -> None:
        """Move a chat's files with the chat; refuse if ``new`` already has files."""
        with self.lock:
            source, target = self._inside(self.folder(user, old)), self._inside(self.folder(user, new))
            if not source.exists():
                return
            if target.exists():
                raise ValueError(f"Files for a chat named {new!r} already exist in {target}.")
            source.rename(target)

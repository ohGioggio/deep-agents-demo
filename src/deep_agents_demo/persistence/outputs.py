"""Files the agent delivered in a chat, kept outside its workspace with every version.

    <root>/<user>/<chat>/<name>                    the latest version
    <root>/<user>/<chat>/.versions/<name>.v1, .v2  every version, never changed
    <root>/<user>/<chat>/.index.json               one record per version
    <root>/<user>/.owner, <root>/<user>/<chat>/.owner  the exact user or chat name

The agent can't write here (its file tools act on the workspace or a sandbox), so a
delivered version stays as it was delivered. File names that differ only in case are
the same file (Windows folders don't tell them apart), and every path stays inside the
root, whatever the user or chat is called.

A user or chat gets the folder of its name made safe ("My Chat" -> My_Chat). Other
names can map to the same folder ("my chat", "My_Chat", or "parks"/"Parks" on
Windows), so each folder records whose it is in ``.owner``: a name that finds the
folder taken gets its own, with a short hash ("My_Chat-1a2b3c4d"). Folders saved
before owners existed belong to the first name that saves into them.
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from deep_agents_demo.models.media import IMAGE_KINDS
from deep_agents_demo.persistence.files import safe_folder_name

INDEX = ".index.json"  # a delivered file can't be named this (clean_name refuses leading dots)
OWNER = ".owner"
_RECORD_KEYS = {"name": str, "version": int, "bytes": int, "kind": str, "caption": str, "saved_at": str}
_VERSION = re.compile(r"^(?P<name>.+)\.v(?P<version>\d+)$")
DIFF_MAX_LINES = 10_000  # longer versions are not compared line by line (the matcher is slow on them)


def _part(name: str) -> str:
    """One folder name for a user or chat; never "." or ".." (they would leave the root)."""
    safe = safe_folder_name(name)
    return safe.replace(".", "_") if set(safe) == {"."} else safe


def _owner(folder: Path) -> str | None:
    """Whose folder this is, or None (free, or saved before owners existed)."""
    try:
        return (folder / OWNER).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _valid(index) -> bool:
    return isinstance(index, list) and all(
        isinstance(record, dict) and all(isinstance(record.get(key), kind) for key, kind in _RECORD_KEYS.items())
        for record in index)


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


def _lines(data: bytes) -> list[str]:
    return data.decode("utf-8", errors="replace").splitlines()


def _too_large(old: list[str], new: list[str]) -> bool:
    return max(len(old), len(new)) > DIFF_MAX_LINES


def change_summary(old: bytes, new: bytes, since: int) -> str:
    """"changed from v1: +2 −1 lines": lines added and removed from the version ``since`` to ``new``."""
    old_lines, new_lines = _lines(old), _lines(new)
    if _too_large(old_lines, new_lines):
        return f"changed from v{since} (too large to compare line by line)"
    added = removed = 0
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return f"changed from v{since}: +{added} −{removed} lines"


def _span(versions: list[int]) -> str:
    """"1–3" for consecutive versions, else "1, 3"."""
    if versions == list(range(versions[0], versions[-1] + 1)):
        return f"{versions[0]}–{versions[-1]}"
    return ", ".join(map(str, versions))


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
        return self._claim(self._claim(self.root, user), chat)

    @staticmethod
    def _claim(parent: Path, name: str) -> Path:
        """``name``'s folder under ``parent``: the one with its hash if it has one (it keeps
        it, even after the name that took the safe name is gone), else the safe name,
        unless another name owns that."""
        plain = parent / _part(name)
        hashed = parent / f"{_part(name)}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"
        if _owner(hashed) == name:
            return hashed
        owner = _owner(plain)
        return plain if owner is None or owner == name else hashed

    @staticmethod
    def _own(folder: Path, name: str, replace: bool = False) -> None:
        if replace or _owner(folder) is None:
            _write(folder / OWNER, name.encode("utf-8"))

    def _inside(self, folder: Path) -> Path:
        """``folder``, after checking it is inside the root (never delete or move outside)."""
        if self.root.resolve() not in folder.resolve().parents:
            raise ValueError(f"{folder} is outside the output store")
        return folder

    def _index(self, folder: Path) -> list[dict]:
        path = folder / INDEX
        if not path.exists():
            return []
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            index = None
        return index if _valid(index) else self._rebuild(folder)

    def _rebuild(self, folder: Path) -> list[dict]:
        """An unreadable index, rebuilt from the kept versions (captions are lost); the
        broken one is kept aside as ``.index.json.broken-<time>``."""
        found = []
        for kept in (folder / ".versions").glob("*.v*") if (folder / ".versions").is_dir() else ():
            match = _VERSION.match(kept.name)
            if match and kept.is_file():
                stat = kept.stat()
                found.append((stat.st_mtime_ns, match["name"], int(match["version"]), stat.st_size))
        first = {}
        for mtime, name, _version, _size in sorted(found):
            first.setdefault(name, mtime)
        index = [{"name": name, "version": version, "bytes": size,
                  "kind": Path(name).suffix.lstrip(".").lower(), "caption": "",
                  "saved_at": datetime.fromtimestamp(mtime / 1e9, timezone.utc).isoformat(timespec="seconds")}
                 for mtime, name, version, size in sorted(found, key=lambda f: (first[f[1]], f[1], f[2]))]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        os.replace(folder / INDEX, folder / f"{INDEX}.broken-{stamp}")
        _write(folder / INDEX, json.dumps(index, indent=2).encode("utf-8"))
        return index

    def _file(self, folder: Path, record: dict) -> OutputFile:
        return OutputFile(record["name"], record["version"], record["bytes"], record["kind"],
                          _one_line(record["caption"]), record["saved_at"], folder / record["name"])

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
                      "kind": Path(name).suffix.lstrip(".").lower(), "caption": _one_line(caption),
                      "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            _write(folder / INDEX, json.dumps(index + [record], indent=2).encode("utf-8"))
            self._own(folder.parent, user)
            self._own(folder, chat)
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

    def diff(self, user: str, chat: str, name: str, v1: int | None = None, v2: int | None = None) -> str:
        """A unified diff from version ``v1`` to ``v2`` of a text file (the last two by default).
        ValueError, in words for the user, when it can't be made."""
        with self.lock:
            folder = self.folder(user, chat)
            records = [r for r in self._index(folder) if r["name"].casefold() == name.casefold()]
            if not records:
                raise ValueError(f"No file named {name!r} in this chat (see /files).")
            label = records[0]["name"]
            if records[0]["kind"] in IMAGE_KINDS:
                raise ValueError(f"{label} is an image: can't diff an image (see /files).")
            versions = sorted(r["version"] for r in records)
            if v1 is None and v2 is None:
                if len(versions) < 2:
                    raise ValueError(f"{label} has only v{versions[0]}; there is nothing to compare.")
                v1, v2 = versions[-2], versions[-1]
            elif v1 is None:  # only v2: compare it with the version before it
                earlier = [v for v in versions if v < v2]
                if not earlier:
                    raise ValueError(f"{label} has no version before v{v2} to compare it with.")
                v1 = earlier[-1]
            v2 = versions[-1] if v2 is None else v2
            for version in (v1, v2):
                if version not in versions:
                    raise ValueError(f"{label} has no v{version} (versions: {_span(versions)}).")
            old, new = self.read(user, chat, label, v1), self.read(user, chat, label, v2)
            old_lines, new_lines = _lines(old), _lines(new)
            if _too_large(old_lines, new_lines):
                return (f"{label} v{v1} and v{v2} are too large to compare line by line "
                        f"(over {DIFF_MAX_LINES} lines).")
            text = "\n".join(difflib.unified_diff(old_lines, new_lines,
                                                 f"{label} v{v1}", f"{label} v{v2}", lineterm=""))
            return text or f"(no changes between v{v1} and v{v2})"

    def delete_thread(self, user: str, chat: str) -> None:
        """Remove a chat's folder. The owner file goes last, so a delete stopped midway
        still finds this folder (not another name's) when it is run again."""
        with self.lock:
            folder = self._inside(self.folder(user, chat))
            if not folder.exists():
                return
            for child in list(folder.iterdir()):
                if child.name == OWNER:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            owner_file = folder / OWNER
            owner = owner_file.read_bytes() if owner_file.exists() else None
            owner_file.unlink(missing_ok=True)
            try:
                folder.rmdir()
            except OSError:  # the folder stays, so it must still say whose it is
                if owner is not None:
                    try:
                        _write(owner_file, owner)
                    except OSError:
                        pass  # the error the user sees is why the delete stopped
                raise

    def rename_thread(self, user: str, old: str, new: str) -> None:
        """Move a chat's files with the chat; refuse if ``new`` already has files."""
        with self.lock:
            source, target = self._inside(self.folder(user, old)), self._inside(self.folder(user, new))
            if not source.exists():
                return
            if target.exists() and target.resolve() != source.resolve():
                raise ValueError(f"Files for a chat named {new!r} already exist in {target}.")
            source.rename(target)
            try:
                self._own(target, new, replace=True)
            except BaseException:  # the files go back, so the chat still finds them under its old name
                try:
                    target.rename(source)
                except OSError:
                    pass  # the error the user sees is why the rename stopped
                raise

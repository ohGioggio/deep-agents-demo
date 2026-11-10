"""Versioned skill storage in SQLite.

A skill is identified by (scope, owner, name): scope is "user" or "marketplace",
and owner is the user id ("" for the shared marketplace). Installs record which
marketplace skills each user has activated. Every change adds a
version holding all of the skill's files, so nothing is ever overwritten: removing
a skill adds a "remove" version, and restoring copies an old version forward.
"""
import functools
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    current_version INTEGER NOT NULL,
    removed INTEGER NOT NULL DEFAULT 0,
    UNIQUE (scope, owner, name)
);
CREATE TABLE IF NOT EXISTS skill_versions (
    skill_id INTEGER NOT NULL REFERENCES skills(id),
    version INTEGER NOT NULL,
    action TEXT NOT NULL,
    files TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (skill_id, version)
);
CREATE TABLE IF NOT EXISTS installs (
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    installed_at TEXT NOT NULL,
    PRIMARY KEY (owner, name)
);
"""


@dataclass(frozen=True)
class Skill:
    scope: str
    owner: str
    name: str
    version: int
    files: dict          # relative path -> text, always including "SKILL.md"
    removed: bool = False


@dataclass(frozen=True)
class Version:
    version: int
    action: str
    created_at: str
    files: dict


def _locked(method):
    """Run under the store's lock: one connection is shared by the tool threads."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


class SkillStore:
    """Every public method holds ``lock``: LangGraph runs a step's tool calls in
    parallel threads, and they share this one SQLite connection (and its open
    transaction), so unlocked calls read half-written versions."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.executescript(SCHEMA)

    @_locked
    def close(self):
        self.db.close()

    def _row(self, scope, owner, name):
        return self.db.execute(
            "SELECT id, current_version, removed FROM skills WHERE scope=? AND owner=? AND name=?",
            (scope, owner, name)).fetchone()

    @_locked
    def get(self, scope: str, owner: str, name: str) -> Skill | None:
        """The current version, or None if the skill never existed. Removed skills are
        returned with ``removed=True`` and the files they had before removal."""
        row = self._row(scope, owner, name)
        if row is None:
            return None
        skill_id, version, removed = row
        return Skill(scope, owner, name, version, self.version(scope, owner, name, version).files,
                     bool(removed))

    @_locked
    def version(self, scope, owner, name, version) -> Version | None:
        row = self._row(scope, owner, name)
        if row is None:
            return None
        found = self.db.execute(
            "SELECT version, action, created_at, files FROM skill_versions WHERE skill_id=? AND version=?",
            (row[0], version)).fetchone()
        return Version(found[0], found[1], found[2], json.loads(found[3])) if found else None

    @_locked
    def history(self, scope, owner, name) -> list[Version]:
        row = self._row(scope, owner, name)
        if row is None:
            return []
        return [Version(v, a, t, json.loads(f)) for v, a, t, f in self.db.execute(
            "SELECT version, action, created_at, files FROM skill_versions WHERE skill_id=? "
            "ORDER BY version", (row[0],))]

    @_locked
    def list_current(self, scope: str, owner: str) -> list[Skill]:
        """Every skill in this scope that is not removed."""
        names = [n for (n,) in self.db.execute(
            "SELECT name FROM skills WHERE scope=? AND owner=? AND removed=0 ORDER BY name",
            (scope, owner))]
        return [self.get(scope, owner, name) for name in names]

    @_locked
    def save(self, scope, owner, name, files: dict, action: str) -> int:
        """Add a version (``action``: create, update, import, restore). Returns its number."""
        return self._add_version(scope, owner, name, files, action, removed=False)

    @_locked
    def remove(self, scope, owner, name) -> int:
        """Mark the skill removed, keeping its files in a new "remove" version."""
        current = self.get(scope, owner, name)
        if current is None or current.removed:
            raise KeyError(name)
        return self._add_version(scope, owner, name, current.files, "remove", removed=True)

    # marketplace installs: which catalogue skills a user has activated
    @_locked
    def installed(self, owner: str) -> list[str]:
        return [n for (n,) in self.db.execute(
            "SELECT name FROM installs WHERE owner=? ORDER BY name", (owner,))]

    @_locked
    def install(self, owner: str, name: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO installs (owner, name, installed_at) VALUES (?,?,?)",
                            (owner, name, now))

    @_locked
    def uninstall(self, owner: str, name: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM installs WHERE owner=? AND name=?", (owner, name))

    def _add_version(self, scope, owner, name, files, action, removed):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.db:  # one transaction
            row = self._row(scope, owner, name)
            if row is None:
                cursor = self.db.execute(
                    "INSERT INTO skills (scope, owner, name, current_version, removed) VALUES (?,?,?,?,?)",
                    (scope, owner, name, 1, int(removed)))
                skill_id, version = cursor.lastrowid, 1
            else:
                skill_id, version = row[0], row[1] + 1
                self.db.execute("UPDATE skills SET current_version=?, removed=? WHERE id=?",
                                (version, int(removed), skill_id))
            self.db.execute(
                "INSERT INTO skill_versions (skill_id, version, action, files, created_at) "
                "VALUES (?,?,?,?,?)", (skill_id, version, action, json.dumps(files), now))
        return version

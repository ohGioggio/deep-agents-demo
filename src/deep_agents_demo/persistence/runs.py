"""The run store: one row per chat turn (``runs``) and one per nested part of it
(``run_parts``: subagents, plans, rooms, swarms and plan steps), in ``data/runs.sqlite``.

A part's ``parent`` is the number of its parent part within the same run, or None.
Parts are saved in the order they began, so a parent always comes before its children.
Every read and write is scoped to one user.
"""
import sqlite3
import threading
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    chat TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    seconds REAL NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model_calls INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_user ON runs (user, started_at);
CREATE TABLE IF NOT EXISTS run_parts (
    run_id TEXT NOT NULL,
    part INTEGER NOT NULL,
    parent INTEGER,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    model_calls INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    seconds REAL NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (run_id, part)
);
"""

RUN_COLUMNS = ("id, user, chat, started_at, ended_at, seconds, model, prompt, model_calls, "
               "input_tokens, output_tokens, status")


@dataclass
class RunPart:
    """A nested part of a run: ``kind`` is subagent, plan, room, swarm or step."""
    part: int
    parent: int | None
    kind: str
    label: str
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    status: str = "done"


@dataclass
class Run:
    """One user turn. ``status`` is done, error or interrupted; ``parts`` is filled by ``get``."""
    id: str
    user: str
    chat: str
    started_at: float
    ended_at: float
    seconds: float
    model: str
    prompt: str
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "done"
    parts: list = field(default_factory=list)


def _run(row) -> Run:
    return Run(*row)


class RunStore:
    """Save, read and move runs. Every method holds ``lock`` (one connection, several threads)."""

    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def save(self, run: Run) -> None:
        """Write a run and its parts; all or nothing."""
        with self.lock, self.db:
            self.db.execute(f"INSERT INTO runs ({RUN_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (run.id, run.user, run.chat, run.started_at, run.ended_at, run.seconds,
                             run.model, run.prompt, run.model_calls, run.input_tokens,
                             run.output_tokens, run.status))
            self.db.executemany(
                "INSERT INTO run_parts (run_id, part, parent, kind, label, model_calls, input_tokens, "
                "output_tokens, seconds, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(run.id, p.part, p.parent, p.kind, p.label, p.model_calls, p.input_tokens,
                  p.output_tokens, p.seconds, p.status) for p in run.parts])

    def get(self, user: str, run_id: str) -> Run | None:
        """One run with its parts (in the order they began), or None."""
        with self.lock:
            row = self.db.execute(f"SELECT {RUN_COLUMNS} FROM runs WHERE user = ? AND id = ?",
                                  (user, run_id)).fetchone()
            if row is None:
                return None
            parts = self.db.execute(
                "SELECT part, parent, kind, label, model_calls, input_tokens, output_tokens, seconds, status "
                "FROM run_parts WHERE run_id = ? ORDER BY part", (run_id,)).fetchall()
        run = _run(row)
        run.parts = [RunPart(*part) for part in parts]
        return run

    def recent(self, user: str, limit: int = 20) -> list[Run]:
        """The user's latest runs, newest first, without their parts."""
        with self.lock:
            rows = self.db.execute(f"SELECT {RUN_COLUMNS} FROM runs WHERE user = ? "
                                   "ORDER BY started_at DESC, rowid DESC LIMIT ?", (user, limit)).fetchall()
        return [_run(row) for row in rows]

    def totals(self, user: str, since: float) -> dict:
        """The user's run count and summed calls and tokens for runs started at or after ``since``."""
        with self.lock:
            runs, calls, tokens_in, tokens_out = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(model_calls), 0), COALESCE(SUM(input_tokens), 0), "
                "COALESCE(SUM(output_tokens), 0) FROM runs WHERE user = ? AND started_at >= ?",
                (user, since)).fetchone()
        return {"runs": runs, "model_calls": calls, "input_tokens": tokens_in, "output_tokens": tokens_out}

    def delete_chat(self, user: str, chat: str) -> int:
        """Remove a chat's runs and their parts; returns how many runs there were."""
        with self.lock, self.db:
            ids = [row[0] for row in self.db.execute(
                "SELECT id FROM runs WHERE user = ? AND chat = ?", (user, chat)).fetchall()]
            self.db.executemany("DELETE FROM run_parts WHERE run_id = ?", [(run_id,) for run_id in ids])
            self.db.execute("DELETE FROM runs WHERE user = ? AND chat = ?", (user, chat))
        return len(ids)

    def rename_chat(self, user: str, old: str, new: str) -> int:
        """Move a chat's runs to its new name; returns how many runs moved."""
        with self.lock, self.db:
            cursor = self.db.execute("UPDATE runs SET chat = ? WHERE user = ? AND chat = ?",
                                     (new, user, old))
        return cursor.rowcount

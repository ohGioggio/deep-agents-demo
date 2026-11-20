"""Chat (thread) management on top of LangGraph's SQLite checkpoints.

A chat is a LangGraph thread whose ID is the JSON pair ``[user, name]``, so every
operation here is scoped to one user. Messages are rebuilt through the compiled
agent's ``get_state``: recent LangGraph versions store only the channels that
changed in each checkpoint, so the raw rows don't hold the full conversation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def thread_id(user: str, name: str) -> str:
    return json.dumps([user, name])


def valid_name(name: str) -> bool:
    return bool(NAME_RE.match(name or ""))


def new_name(now=None) -> str:
    return f"chat-{(now or datetime.now()):%Y%m%d-%H%M%S}"


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return " ".join(str(content).split())


def _why(error: BaseException) -> str:
    """The error's text, with the advice a file in use needs (a locked file is an OSError)."""
    if isinstance(error, OSError):
        return f"{error}; if a file of this chat is open in another program, close it first"
    return str(error)


def shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


@dataclass
class ChatInfo:
    name: str
    turns: int
    updated: str  # ISO timestamp of the latest checkpoint
    first: str
    last: str


# Agent runs of a chat's plans, rooms and swarms live on threads named "<chat thread><marker>...".
ORCHESTRATION_THREADS = (":plan:", ":room:", ":swarm:")


class Chats:
    """List, show, rename and delete one user's chats."""

    def __init__(self, saver, agent, user: str, outputs=None, runs=None):
        self.saver, self.agent, self.user = saver, agent, user
        self.outputs = outputs  # an OutputStore: a chat's delivered files go where the chat goes
        self.runs = runs  # a RunStore: a chat's recorded turns go where the chat goes

    def _config(self, name):
        return {"configurable": {"thread_id": thread_id(self.user, name)}}

    def names(self) -> list[str]:
        with self.saver.cursor() as cur:
            cur.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''")
            rows = [r[0] for r in cur.fetchall()]
        names = []
        for raw in rows:
            try:
                owner, name = json.loads(raw)
            except (ValueError, TypeError):
                continue  # not a chat created by this CLI
            if owner == self.user:
                names.append(name)
        return names

    def exists(self, name: str) -> bool:
        return self.saver.get_tuple(self._config(name)) is not None

    def messages(self, name: str):
        return self.agent.get_state(self._config(name)).values.get("messages", [])

    def info(self, name: str) -> ChatInfo:
        state = self.agent.get_state(self._config(name))
        humans = [_text(m) for m in state.values.get("messages", []) if m.type == "human"]
        return ChatInfo(name, len(humans), state.created_at or "",
                        humans[0] if humans else "", humans[-1] if humans else "")

    def list(self) -> list[ChatInfo]:
        return sorted((self.info(n) for n in self.names()), key=lambda c: c.updated, reverse=True)

    def transcript(self, name: str) -> list[tuple[str, str]]:
        """(speaker, text) pairs: user messages and the agent's final replies."""
        lines = []
        for message in self.messages(name):
            text = _text(message)
            if message.type == "human":
                lines.append(("You", text))
            elif message.type == "ai" and text and not getattr(message, "tool_calls", None):
                lines.append(("Agent", text))
        return lines

    def delete(self, name: str) -> None:
        """Delete the chat's files, runs, plan, room and swarm threads, then its checkpoints.

        Every step is idempotent, and the checkpoints go last: a step that fails leaves
        the chat listed, and running the delete again finishes what is left. Runs go
        before the checkpoints, so no run outlives its chat."""
        chat = thread_id(self.user, name)
        try:
            if self.outputs is not None:
                self.outputs.delete_thread(self.user, name)
            if self.runs is not None:
                self.runs.delete_chat(self.user, name)
            for extra in self._orchestration_threads(chat):  # its plans', rooms' and swarms' agent runs
                self.saver.delete_thread(extra)
            self.saver.delete_thread(chat)
        except Exception as error:
            raise ValueError(f"Chat {name!r} may be partly deleted ({_why(error)}); "
                             "run the delete again.") from error

    def _orchestration_threads(self, chat: str) -> list[str]:
        """Threads of the plans and rooms run in a chat: "<chat thread>:plan:..." and
        "<chat thread>:room:..." (see orchestration and rooms)."""
        found = []
        with self.saver.cursor() as cur:
            for marker in ORCHESTRATION_THREADS:
                prefix = f"{chat}{marker}"
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints "
                            "WHERE substr(thread_id, 1, ?) = ?", (len(prefix), prefix))
                found += [row[0] for row in cur.fetchall()]
        return found

    def rename(self, old: str, new: str) -> None:
        if not self.exists(old):
            raise ValueError(f"No chat named {old!r}.")
        if not valid_name(new):
            raise ValueError(f"Invalid chat name {new!r}: use letters, digits, '.', '_' or '-'.")
        if self.exists(new):
            raise ValueError(f"A chat named {new!r} already exists.")
        # The steps run checkpoints, then files, then runs. A failing step undoes the
        # ones before it, last first. The runs go last because their UPDATE is one
        # statement: when it fails nothing has moved, so it needs no undo.
        #
        # Ruling: two stores (checkpoints and runs are in different SQLite files, the
        # files are on disk) can't be made atomic together, so an undo can itself
        # fail. Then the rename is not all-or-nothing, and the rule is that it is
        # always reported with the exact fix, never silently left split. The chat
        # is under the new name after either undo can fail: the files undo runs
        # first, so a failed files undo leaves the checkpoints under the new name,
        # and a failed checkpoint undo leaves the checkpoints there too.
        undo = []  # the undo steps of the steps that have moved
        try:
            self._move_checkpoints(old, new)
            undo.append(lambda: self._move_checkpoints(new, old))
            if self.outputs is not None:
                self._move_files(old, new)
                undo.append(lambda: self._move_files(new, old))
            if self.runs is not None:
                self.runs.rename_chat(self.user, old, new)
        except BaseException as error:
            for step in reversed(undo):
                try:
                    step()
                except Exception as failed:
                    raise ValueError(f"Couldn't rename chat {old!r} to {new!r} ({_why(error)}), and "
                                     f"putting it back failed ({_why(failed)}). The chat is now named "
                                     f"{new!r}: switch to it (/switch {new}), then /rename {old} "
                                     f"(or --rename-thread {new} {old})."
                                     ) from failed
            if isinstance(error, Exception):
                raise ValueError(f"Couldn't rename chat {old!r} (nothing was changed: {_why(error)}); "
                                 "try again.") from error
            raise

    def _move_checkpoints(self, old: str, new: str) -> None:
        # SqliteSaver inherits copy_thread from the base class, which raises
        # NotImplementedError, so move the rows: a thread lives in exactly two
        # tables, both keyed by thread_id. One transaction keeps it all-or-nothing.
        source, target = thread_id(self.user, old), thread_id(self.user, new)
        with self.saver.cursor(transaction=True) as cur:
            for table in ("checkpoints", "writes"):
                cur.execute(f"UPDATE {table} SET thread_id = ? WHERE thread_id = ?", (target, source))
                # Its plans' and rooms' agent threads move with it.
                for marker in ORCHESTRATION_THREADS:
                    prefix = f"{source}{marker}"
                    cur.execute(f"UPDATE {table} SET thread_id = ? || substr(thread_id, ?) "
                                "WHERE substr(thread_id, 1, ?) = ?",
                                (target, len(source) + 1, len(prefix), prefix))

    def _move_files(self, old: str, new: str) -> None:
        self.outputs.rename_thread(self.user, old, new)



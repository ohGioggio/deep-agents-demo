"""What the engine emits while a turn, a plan or a room runs (the Observer's events).

Plain, immutable data: a front end renders them (``ui/cli/render.py`` prints them);
a server would send ``to_dict()`` over the wire.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class _Event:
    def to_dict(self) -> dict:
        return {"kind": type(self).__name__, **asdict(self)}


@dataclass(frozen=True)
class MessageEvent(_Event):
    """A message one agent added: ``label`` is who ("main", "subagent", "room / critic"),
    ``node`` the graph node ("model", "tools", ...)."""
    label: str
    node: str
    role: str                 # "ai", "tool", "human", ...
    text: str
    tool_calls: tuple = ()    # ({"name", "args", "id"}, ...)
    pretty: str = ""          # the message as a terminal shows it (images as "[image]")


@dataclass(frozen=True)
class FileEvent(_Event):
    """The agent delivered a file (``send_file``); ``path`` is where it was saved."""
    label: str
    name: str
    version: int
    kind: str
    bytes: int
    caption: str
    path: str


@dataclass(frozen=True)
class TodosEvent(_Event):
    """The agent's to-do list changed."""
    todos: tuple = ()


@dataclass(frozen=True)
class NoticeEvent(_Event):
    """A line for the user from a plan or room (e.g. "[plan] step 1/2 - general-purpose")."""
    text: str


@dataclass(frozen=True)
class FinalAnswerEvent(_Event):
    """The turn's answer, for a quiet one-shot run that shows nothing else."""
    text: str

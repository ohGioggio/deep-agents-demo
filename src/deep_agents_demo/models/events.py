"""What the engine emits while a turn, a plan or a room runs (the Observer's events).

Plain, immutable data: a front end renders them (``ui/cli/render.py`` prints them);
a server would send ``to_dict()`` over the wire.
"""
import time
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


# ---- Run events (AG-UI-shaped): a turn, subagent, plan or room run is a "run" ----
# Durations are time.monotonic() seconds; ended_at is a time.time() value.

@dataclass(frozen=True)
class RunStarted(_Event):
    """A run began: a main turn, a subagent, a plan worker/reviewer, or a room turn."""
    run_id: str
    label: str
    parent_id: str | None = None


@dataclass(frozen=True)
class RunFinished(_Event):
    """A run ended; the token totals include nested runs."""
    run_id: str
    label: str
    input_tokens: int
    output_tokens: int
    duration: float
    ended_at: float


@dataclass(frozen=True)
class RunError(_Event):
    """A run failed."""
    run_id: str
    label: str
    message: str


@dataclass(frozen=True)
class TextMessageStart(_Event):
    """A model message started streaming."""
    run_id: str
    message_id: str
    label: str


@dataclass(frozen=True)
class TextMessageContent(_Event):
    """A streamed piece of a model message."""
    run_id: str
    message_id: str
    delta: str


@dataclass(frozen=True)
class TextMessageEnd(_Event):
    """A model message finished; ``has_tool_calls`` says it was followed by tool calls."""
    run_id: str
    message_id: str
    text: str
    input_tokens: int
    output_tokens: int
    has_tool_calls: bool


@dataclass(frozen=True)
class ToolCallStart(_Event):
    """A tool call began."""
    run_id: str
    tool_call_id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolCallEnd(_Event):
    """A tool call finished; ``status`` is "ok", "error" or "rejected"; ``result`` is the full text."""
    run_id: str
    tool_call_id: str
    name: str
    status: str
    summary: str
    duration: float
    result: str = ""


@dataclass(frozen=True)
class SubagentStarted(_Event):
    """The main agent handed work to a subagent (through the ``task`` tool)."""
    run_id: str
    parent_run_id: str
    name: str
    tool_call_id: str


@dataclass(frozen=True)
class SubagentFinished(_Event):
    """A subagent run ended with ``status`` "ok", "error" or "rejected"; the token totals
    include its nested runs (its footer: ``format_footer(ended_at, ...)``)."""
    run_id: str
    status: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration: float = 0.0
    ended_at: float = 0.0


@dataclass(frozen=True)
class _StepEvent(_Event):
    """Steps have their own ``kind`` field ("plan"/"room"), so the event's kind goes
    under ``kind`` in ``to_dict()`` and the step kind under ``step_kind``."""

    def to_dict(self) -> dict:
        return {**asdict(self), "kind": type(self).__name__, "step_kind": self.kind}


@dataclass(frozen=True)
class StepStarted(_StepEvent):
    """A plan step or room turn started. ``kind`` is "plan" or "room"
    (room: ``number`` is the turn and ``total`` the maximum turns)."""
    run_id: str
    kind: str
    number: int
    total: int
    agent: str


@dataclass(frozen=True)
class StepFinished(_StepEvent):
    """A plan step or room turn finished."""
    run_id: str
    kind: str
    number: int
    status: str
    attempts: int
    duration: float
    input_tokens: int
    output_tokens: int
    summary: str = ""


@dataclass(frozen=True)
class StateSnapshot(_Event):
    """The full state after a change (the to-do list for now)."""
    todos: tuple = ()


@dataclass(frozen=True)
class ApprovalRequested(_Event):
    """A tool call is waiting for the user to approve or reject it."""
    run_id: str
    tool_call_id: str
    name: str


def _format_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 100:
        return f"{total} s"
    minutes, rest = divmod(total, 60)
    return f"{minutes} min {rest} s"


def format_footer(ended_at: float, input_tokens: int, output_tokens: int, duration: float) -> str:
    """The line under a finished run: ``HH:MM · 9.8k in / 450 out · 2 min 5 s``.

    ``ended_at`` is a time.time() value, shown in local time.
    """
    clock = time.strftime("%H:%M", time.localtime(ended_at))
    return (f"{clock} · {_format_tokens(input_tokens)} in / "
            f"{_format_tokens(output_tokens)} out · {_format_duration(duration)}")

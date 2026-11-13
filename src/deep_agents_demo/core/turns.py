"""The turn engine: run one user turn to the end, emitting events instead of printing.

A turn streams the agent's updates, turns every new message into a
``MessageEvent`` (and to-do changes into ``TodosEvent``), and answers the pauses it
meets: approvals through ``decide``, and other pauses (``run_plan``,
``start_room``) through ``runners`` by their ``type``, on this thread.

Alongside those it emits the run events (``RunStarted``, text as it streams, tool
calls, subagent runs, token totals) that a richer front end draws; the plain CLI
ignores them.
"""
import ast
import json
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage
from langgraph.types import Command

from deep_agents_demo.core.messages import has_image, without_images
from deep_agents_demo.models.approvals import NO_HUMAN
from deep_agents_demo.models.events import (
    FileEvent,
    FinalAnswerEvent,
    MessageEvent,
    RunError,
    RunFinished,
    RunStarted,
    StateSnapshot,
    SubagentFinished,
    SubagentStarted,
    TextMessageContent,
    TextMessageEnd,
    TextMessageStart,
    TodosEvent,
    ToolCallEnd,
    ToolCallStart,
)


def turn_input(prompt, reload_skills=False):
    """Agent input for one user turn.

    Deep Agents saves the skill list in the thread's checkpoint and loads it only
    once per thread. ``skills_metadata=None`` asks it to reload, so skills added
    or changed since the thread began (or a new sandbox route) are picked up.
    """
    payload = {"messages": [{"role": "user", "content": prompt}]}
    if reload_skills:
        payload["skills_metadata"] = None
    return payload


def printable(message):
    """The message as shown: images become a short note instead of base64 text."""
    if has_image(message):
        return without_images(message, "[image]")
    return message


def message_event(label: str, node: str, message) -> MessageEvent:
    shown = printable(message)
    text = shown.text if isinstance(shown.text, str) else str(shown.content)
    calls = tuple({"name": c.get("name"), "args": c.get("args", {}), "id": c.get("id")}
                  for c in getattr(message, "tool_calls", None) or ())
    return MessageEvent(label, node, message.type, text, calls, shown.pretty_repr())


def file_event(label: str, message) -> FileEvent | None:
    """The FileEvent for a successful send_file, from its tool message's artifact."""
    artifact = getattr(message, "artifact", None)
    if getattr(message, "type", None) != "tool" or getattr(message, "name", None) != "send_file" \
            or not isinstance(artifact, dict) or not artifact.get("saved"):
        return None
    return FileEvent(label, artifact["name"], artifact["version"], artifact["kind"], artifact["bytes"],
                     artifact.get("caption", ""), artifact["path"])


def _text(message) -> str:
    return message.text if isinstance(message.text, str) else str(message.content)


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / 1024 / 1024:.1f} MB"


def _parsed(text: str, parse):
    try:
        return parse(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def summarize_result(name: str, content: str) -> str:
    """A tool result in a few words: "12 lines", "3 entries", "v2 · 4.2 KB", or its first line."""
    lines = content.strip().splitlines()
    first = (lines[0] if lines else "")[:60]
    if content.startswith("Error"):
        return first
    if name == "read_file":
        return f"{len(content.splitlines())} lines"
    if name in ("ls", "glob"):
        parsed = _parsed(content, ast.literal_eval)
        if isinstance(parsed, list):
            return f"{len(parsed)} entries"
    elif name in ("search_assets", "search_docs"):
        parsed = _parsed(content, json.loads)
        if isinstance(parsed, list):
            return f"{len(parsed)} results"
    elif name == "send_file":
        parsed = _parsed(content, json.loads)
        if isinstance(parsed, dict) and isinstance(parsed.get("version"), int) and isinstance(parsed.get("bytes"), int):
            return f"v{parsed['version']} · {_size(parsed['bytes'])}"
    return first


@dataclass
class _Run:
    """One agent run in a turn (the main turn or a subagent); tokens include nested runs."""
    id: str
    label: str
    parent: "_Run | None" = None
    input_tokens: int = 0
    output_tokens: int = 0
    pending_tasks: list = field(default_factory=list)  # [(task call id, subagent_type)] not yet started

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        run = self
        while run is not None:
            run.input_tokens += input_tokens
            run.output_tokens += output_tokens
            run = run.parent


class _Progress:
    """What a turn has seen so far, kept across its pauses: runs by stream namespace,
    open tool calls, messages being streamed, the last to-do list."""

    def __init__(self, label: str):
        self.main = _Run(uuid.uuid4().hex, label)
        self.started = time.monotonic()
        self.runs = {(): self.main}   # stream namespace -> run
        self.calls = {}               # tool call id -> (run, name, start time)
        self.subagents = {}           # task call id -> its subagent run
        self.streaming = set()        # ids of messages that got a TextMessageStart
        self.finished = set()         # ids of messages that got a TextMessageEnd
        self.todos = None

    def run_for(self, namespace: tuple, send, call_id=None) -> _Run:
        """The run a namespace belongs to. A new ``tools:`` namespace is a subagent:
        matched to its task call by ``call_id`` (from the stream's metadata) when known,
        else to the earliest task call not yet matched; with no task call to match it
        belongs to its parent run."""
        run = self.runs.get(namespace)
        if run is not None:
            return run
        parent = self.run_for(namespace[:-1], send)
        run = parent
        if namespace[-1].startswith("tools:"):
            run = self.subagents.get(call_id) or self._start_subagent(namespace, parent, send, call_id)
        self.runs[namespace] = run
        return run

    def _start_subagent(self, namespace, parent: _Run, send, call_id) -> _Run:
        pending = parent.pending_tasks
        if call_id is not None:
            match = next((task for task in pending if task[0] == call_id), None)
        else:
            match = pending[0] if pending else None
        if match is None:
            return parent
        pending.remove(match)
        task_id, name = match
        run = _Run("|".join(namespace), name, parent)
        self.subagents[task_id] = run
        send(SubagentStarted(run.id, parent.id, name, task_id))
        return run

    def chunk(self, namespace, message, metadata, who: str, send) -> None:
        """A piece of a model message as it streams (or a whole one, from a model that doesn't)."""
        if not isinstance(message, AIMessage) or message.id in self.finished:
            return
        run = self.run_for(namespace, send, metadata.get("lc_subagent_tool_call_id"))
        text = _text(message)
        if not text or message.id is None:
            return
        if message.id not in self.streaming:
            self.streaming.add(message.id)
            send(TextMessageStart(run.id, message.id, who))
        send(TextMessageContent(run.id, message.id, text))

    def message(self, namespace, message, who: str, send) -> None:
        """A finished message from the updates stream."""
        if message.type == "ai":
            self._finished_ai(namespace, message, who, send)
        elif message.type == "tool":
            self._tool_result(namespace, message, send)

    def _finished_ai(self, namespace, message, who, send) -> None:
        run = self.run_for(namespace, send)
        usage = getattr(message, "usage_metadata", None) or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        run.add_usage(tokens_in, tokens_out)
        text, calls = _text(message), getattr(message, "tool_calls", None) or []
        message_id = message.id or ""
        if text and message_id not in self.streaming:  # it never streamed: show it whole
            send(TextMessageStart(run.id, message_id, who))
            send(TextMessageContent(run.id, message_id, text))
        self.streaming.discard(message_id)
        self.finished.add(message_id)
        send(TextMessageEnd(run.id, message_id, text, tokens_in, tokens_out, bool(calls)))
        for call in calls:
            self.calls[call["id"]] = (run, call["name"], time.monotonic())
            if call["name"] == "task":
                run.pending_tasks.append((call["id"], (call.get("args") or {}).get("subagent_type", "")))
            send(ToolCallStart(run.id, call["id"], call["name"], call.get("args", {})))

    def _tool_result(self, namespace, message, send) -> None:
        call_id = message.tool_call_id
        run, name, started = self.calls.pop(call_id, (None, message.name or "", None))
        run = run or self.run_for(namespace, send)
        text = _text(message)
        status = "error" if getattr(message, "status", None) == "error" or text.startswith("Error") else "ok"
        subagent = self.subagents.pop(call_id, None)
        if subagent is not None:
            send(SubagentFinished(subagent.id, status))
        else:  # a task whose subagent never showed itself
            run.pending_tasks[:] = [task for task in run.pending_tasks if task[0] != call_id]
        duration = time.monotonic() - started if started is not None else 0.0
        send(ToolCallEnd(run.id, call_id, name, status, summarize_result(name, text), duration, text))

    def todos_changed(self, todos, send) -> None:
        todos = tuple(todos)
        if todos != self.todos:
            self.todos = todos
            send(StateSnapshot(todos))


class _Silent:
    def emit(self, event) -> None:
        return None


class _FilesOnly(_Silent):
    """A quiet run's sink: delivered files are part of the answer, so they still show."""

    def __init__(self, sink):
        self.sink = sink


def _speaker(label, namespace) -> str:
    who = label or "main"
    return (f"{who} / subagent" if label else "subagent") if namespace else who


def stream_once(agent, payload, config, sink, shown=None, label=None, progress=None) -> list:
    """Stream one run of the agent loop; return the approval requests it paused on.

    ``shown`` collects the ids of emitted messages: after a pause, LangGraph sends
    the paused AI message again, and it should be shown once per turn. ``label``
    names who is speaking (e.g. "room / critic"; default "main"). ``progress`` is
    the turn's ``_Progress`` (runs, open tool calls), kept across pauses.
    """
    paused, shown = {}, shown if shown is not None else set()
    silent = isinstance(sink, _Silent)  # build nothing nobody will see
    files_only = isinstance(sink, _FilesOnly)
    if files_only:
        sink = sink.sink
    progress = progress or _Progress(label or "main")
    send = sink.emit
    # Include child updates so delegation/tool calls are observable; "messages"
    # brings the model's text as it is written.
    for namespace, mode, chunk in agent.stream(payload, config=config, stream_mode=["updates", "messages"],
                                               subgraphs=True):
        if mode == "messages":
            if not silent:
                message, metadata = chunk
                progress.chunk(namespace, message, metadata, _speaker(label, namespace), send)
            continue
        for node, state in chunk.items():
            if node == "__interrupt__":
                # A subagent's pause can surface at several levels; keep each once.
                paused.update({item.id: item for item in state})
                continue
            if not isinstance(state, dict):
                continue
            who = _speaker(label, namespace)
            for message in state.get("messages", []):
                key = message.id or id(message)
                if key in shown:
                    continue
                shown.add(key)
                delivered = file_event(who, message)
                if not silent:
                    sink.emit(message_event(who, node, message))
                    progress.message(namespace, message, who, send)
                if delivered is not None and (not silent or files_only):
                    sink.emit(delivered)
            if "todos" in state and not silent:
                sink.emit(TodosEvent(tuple(state["todos"])))
                progress.todos_changed(state["todos"], send)
    return list(paused.values())


def final_answer(agent, config) -> str:
    """The text of the last AI message in a thread (what a one-shot run shows)."""
    for message in reversed(agent.get_state(config).values.get("messages", [])):
        if message.type == "ai" and not getattr(message, "tool_calls", None) and str(message.content).strip():
            return message.text if isinstance(message.text, str) else str(message.content)
    return ""


def _nobody(request) -> dict:
    """With nobody to ask, every paused call is rejected."""
    return {"decisions": [{"type": "reject", "message": NO_HUMAN}
                          for _ in request.get("action_requests", [])]}


def run_turn(agent, prompt, config, sink, decide=None, runners=None, reload_skills=True, label=None,
             quiet=False, show=None) -> None:
    """Run one user turn to the end.

    ``decide(request) -> dict`` answers approval pauses (None: reject them all).
    ``runners`` answer other pauses by their ``type`` (e.g. ``run_plan``): they run
    here, on the caller's thread, and their return value resumes the tool.
    ``quiet`` emits only delivered files and the final answer; ``show=False`` emits
    nothing at all. Around the turn it emits ``RunStarted`` and ``RunFinished``
    (or ``RunError``, then re-raises, when the stream fails).
    """
    decide, runners = decide or _nobody, runners or {}
    payload, shown = turn_input(prompt, reload_skills), set()
    progress = _Progress(label or "main")
    run = progress.main

    def answer(value):
        kind = value.get("type") if isinstance(value, dict) else None
        if kind in runners:
            return runners[kind](value)
        if kind == "run_plan":
            return {"error": "plans can't run here (nobody to run them); do the steps yourself"}
        if kind == "start_room":
            return {"error": "rooms can't run here (nobody to run them)"}
        return decide(value)

    if show is None:
        target = _FilesOnly(sink) if quiet else sink
    else:
        target = sink if show else _Silent()
    target.emit(RunStarted(run.id, run.label))  # quiet and silent targets drop it
    while True:
        try:
            paused = stream_once(agent, payload, config, target, shown, label=label, progress=progress)
        except Exception as error:
            target.emit(RunError(run.id, run.label, str(error)))
            raise
        if not paused:
            target.emit(RunFinished(run.id, run.label, run.input_tokens, run.output_tokens,
                                    time.monotonic() - progress.started, time.time()))
            if quiet:
                sink.emit(FinalAnswerEvent(final_answer(agent, config)))
            return
        payload = Command(resume={item.id: answer(item.value) for item in paused})

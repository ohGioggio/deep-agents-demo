"""The run recorder: watches a chat's event stream and saves each turn as a run (persistence/runs.py).

It is the sink the chat emits to: every event goes on to the sink unchanged, and recording
never raises into the chat. A turn is recorded inside ``turn()``: its calls and tokens are the
ones that started in it (its tally, providers/usage.py: a worker's call counts, a call that ends
after the turn closed does not), and its parts (subagents, plans,
rooms, swarms, plan steps) come from the events, nested by run id and by the open call or step.
A recording failure stops that turn's recording and is reported once, through ``say``.
"""
import threading
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

from deep_agents_demo.models.events import (
    RunError,
    RunStarted,
    StepFinished,
    StepStarted,
    SubagentFinished,
    SubagentStarted,
    TextMessageEnd,
    ToolCallEnd,
    ToolCallStart,
)
from deep_agents_demo.persistence.chats import shorten
from deep_agents_demo.persistence.runs import Run, RunPart
from deep_agents_demo.providers.usage import CURRENT, Tally

# The tools that run orchestration (their part is a plan, room or swarm).
ORCHESTRATION = {"run_plan": "plan", "start_room": "room", "run_swarm": "swarm"}
PROMPT_WIDTH = 120
LABEL_WIDTH = 60


def _first_line(text) -> str:
    return next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")


def prompt_line(prompt) -> str:
    """The first line of a prompt, up to 120 characters: what ``--runs`` shows."""
    return shorten(_first_line(prompt), PROMPT_WIDTH)


class _Part:
    """A part being recorded: its own model calls, and the totals of the parts nested in it."""

    def __init__(self, number, parent, kind, label, started):
        self.number, self.parent, self.kind, self.label, self.started = number, parent, kind, label, started
        self.own_calls = self.own_in = self.own_out = 0  # the model calls and tokens of its own runs
        self.child_calls = self.child_in = self.child_out = 0  # the totals of the parts nested in it
        self.closed = False
        self.status, self.seconds = "interrupted", 0.0
        self.calls = self.tokens_in = self.tokens_out = 0


class _Record:
    """One turn in progress."""

    def __init__(self, chat, prompt, before):
        self.id = uuid.uuid4().hex
        self.chat, self.prompt, self.before = chat, prompt, before
        self.started_at, self.started = time.time(), time.monotonic()
        self.parts: list[_Part] = []  # in the order they began: a parent before its children
        self.part_of_run = {}  # run id -> the part it belongs to (None: the turn's own run)
        self.calls = {}  # open orchestration tool call id -> its part
        self.call_scope: list[_Part] = []  # the open orchestration calls, innermost last
        self.steps = {}  # open plan, room or swarm step id -> its part
        self.thread_step = {}  # thread id -> the step open on it (steps run on threads of their own)
        self.errored = False


class RunRecorder:
    """A sink that records the turns it is told about (``turn``) and passes every event on.

    ``store`` is a RunStore, ``counter`` a UsageCounter (``snapshot()``), ``model`` the model's
    name; ``say`` gets the one report of a recording failure."""

    def __init__(self, sink, store, counter, *, user: str, model: str, say=None):
        self.sink, self.store, self.counter = sink, store, counter
        self.user, self.model = user, model
        self.say = say or (lambda text: None)
        self._lock = threading.Lock()  # runs' threads emit at once
        self._record: _Record | None = None
        self._reported = False

    def emit(self, event) -> None:
        with self._lock:
            if self._record is not None:
                try:
                    self._observe(event)
                except Exception as error:
                    self._fail(error)
        self.sink.emit(event)

    @contextmanager
    def turn(self, chat: str, prompt):
        """Record the turn inside the block. It is saved when the block ends; an error or Ctrl+C
        is saved as such, and the exception goes on. The turn's calls are the ones that start in
        it, on any thread (providers/usage.py): its tally is set for the block, then closed."""
        token = CURRENT.set(Tally())
        try:
            self._begin(chat, prompt)
            try:
                yield
            except BaseException as error:
                self._end("interrupted" if isinstance(error, KeyboardInterrupt) else "error")
                raise
            self._end(None)
        finally:
            CURRENT.reset(token)

    # bookkeeping -----------------------------------------------------------------------------
    @contextmanager
    def scope(self, kind: str, text):
        """Nest the events inside the block under a part of its own, for a run that no tool call
        starts (a /room from the command): its steps are its children, its calls and tokens are
        what ran inside. A scope outside a turn does nothing. The block may set the handle's
        ``status`` (e.g. "refused" for a /room that never started); else it is "done"."""
        part, record = self._enter_scope(kind, text)
        handle = SimpleNamespace(status="done")
        try:
            yield handle
        except BaseException as error:
            handle.status = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            raise
        finally:
            self._exit_scope(part, record, handle.status)

    def _enter_scope(self, kind, text):
        with self._lock:
            if self._record is None:
                return None, None
            try:
                label = shorten(_first_line(text), LABEL_WIDTH) or kind
                part = self._open(kind, label, self._call_part())
                self._record.call_scope.append(part)
                return part, self._record
            except Exception as error:
                self._fail(error)
                return None, None

    def _exit_scope(self, part, record, status) -> None:
        with self._lock:
            if part is None or self._record is not record:
                return  # the turn failed or ended meanwhile: nothing to close
            try:
                self._unscope_call(part)
                self._close(part, status, time.monotonic() - part.started)
            except Exception as error:
                self._fail(error)

    def _begin(self, chat, prompt) -> None:
        with self._lock:
            try:
                self._record = _Record(chat, prompt_line(prompt), self.counter.snapshot())
            except Exception as error:
                self._fail(error)

    def _end(self, status) -> None:
        with self._lock:
            record, self._record = self._record, None
            if record is None:
                return
            try:
                self.store.save(self._finish(record, status))
            except Exception as error:
                self._report(error)

    def _fail(self, error) -> None:
        self._record = None  # this turn is not saved: part of it is missing
        self._report(error)

    def _report(self, error) -> None:
        if self._reported:
            return
        self._reported = True
        try:
            self.say(f"Couldn't record this run: {error}")
        except Exception:
            pass

    def _finish(self, record: _Record, status) -> Run:
        for part in reversed(record.parts):  # children first, so their totals reach their parent
            if not part.closed:
                self._close(part, "interrupted", time.monotonic() - part.started)
        calls, tokens_in, tokens_out = self.counter.snapshot()
        before = record.before
        return Run(
            id=record.id, user=self.user, chat=record.chat, started_at=record.started_at,
            ended_at=time.time(), seconds=time.monotonic() - record.started, model=self.model,
            prompt=record.prompt, model_calls=calls - before[0], input_tokens=tokens_in - before[1],
            output_tokens=tokens_out - before[2],
            status=status or ("error" if record.errored else "done"),
            parts=[RunPart(part.number, part.parent.number if part.parent else None, part.kind, part.label,
                           part.calls, part.tokens_in, part.tokens_out, part.seconds, part.status)
                   for part in record.parts])

    # the events -------------------------------------------------------------------------------
    def _observe(self, event) -> None:
        record = self._record
        if isinstance(event, RunStarted):
            record.part_of_run[event.run_id] = self._run_part()
        elif isinstance(event, TextMessageEnd):
            part = record.part_of_run.get(event.run_id)
            if part is not None and (event.input_tokens or event.output_tokens):  # a model call was made
                part.own_calls += 1
                part.own_in += event.input_tokens
                part.own_out += event.output_tokens
        elif isinstance(event, SubagentStarted):
            record.part_of_run[event.run_id] = self._open("subagent", event.name,
                                                          record.part_of_run.get(event.parent_run_id))
        elif isinstance(event, SubagentFinished):
            part = record.part_of_run.get(event.run_id)
            if part is not None:
                self._close(part, event.status, event.duration, tokens=(event.input_tokens, event.output_tokens))
        elif isinstance(event, ToolCallStart) and event.name in ORCHESTRATION:
            args = event.args if isinstance(event.args, dict) else {}
            label = shorten(_first_line(args.get("goal") or args.get("question")), LABEL_WIDTH) or event.name
            part = self._open(ORCHESTRATION[event.name], label, record.part_of_run.get(event.run_id))
            record.calls[event.tool_call_id] = part
            record.call_scope.append(part)
        elif isinstance(event, ToolCallEnd):
            part = record.calls.pop(event.tool_call_id, None)
            if part is not None:
                self._unscope_call(part)
                self._close(part, event.status, event.duration)
        elif isinstance(event, StepStarted):  # a step is nested in its call, never in another step
            parent = self._call_part()
            kind = "step" if parent is not None else event.kind  # no call or scope around it: its own kind
            part = self._open(kind, f"{event.number}/{event.total} {event.agent}", parent)
            record.steps[event.run_id] = part
            record.thread_step[threading.get_ident()] = part
        elif isinstance(event, StepFinished):
            part = record.steps.pop(event.run_id, None)
            if part is not None:
                if record.thread_step.get(threading.get_ident()) is part:
                    del record.thread_step[threading.get_ident()]
                self._close(part, event.status, event.duration, calls=event.model_calls,
                            tokens=(event.input_tokens, event.output_tokens))
        elif isinstance(event, RunError):
            record.errored = True

    def _call_part(self):
        scope = self._record.call_scope
        return scope[-1] if scope else None

    def _run_part(self):
        """The part a run starting now belongs to: the step open on this thread, else the
        innermost open orchestration call (a run's events come from the thread it runs on)."""
        return self._record.thread_step.get(threading.get_ident()) or self._call_part()

    def _unscope_call(self, part: _Part) -> None:
        if part in self._record.call_scope:
            self._record.call_scope.remove(part)

    def _open(self, kind, label, parent) -> _Part:
        part = _Part(len(self._record.parts) + 1, parent, kind, label, time.monotonic())
        self._record.parts.append(part)
        return part

    def _close(self, part: _Part, status, seconds, calls=None, tokens=None) -> None:
        """A part ended. Its calls and tokens are the event's when it has them (a step's and a
        subagent's include what they ran), else its own plus those of the parts nested in it.
        A model message with no tokens made no call (the budget's stop notice)."""
        if part.closed:
            return
        part.closed, part.status, part.seconds = True, status, seconds
        part.calls = part.own_calls + part.child_calls if calls is None else calls
        part.tokens_in, part.tokens_out = (
            (part.own_in + part.child_in, part.own_out + part.child_out) if tokens is None else tokens)
        if part.parent is not None:
            part.parent.child_calls += part.calls
            part.parent.child_in += part.tokens_in
            part.parent.child_out += part.tokens_out

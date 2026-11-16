"""What plans and rooms share: running one agent on the caller's thread, with the
budget, approvals, and pausing or stopping when the front end asks (``Interruptible``)."""
import threading
import time
import uuid

from deep_agents_demo.core.ports import EventSink, Interaction
from deep_agents_demo.core.turns import run_turn
from deep_agents_demo.features.orchestration.limits import (
    BUDGET_STOP,
    Budget,
    BudgetCap,
    _WrapUp,
)
from deep_agents_demo.models.events import (
    NoticeEvent,
    RunFinished,
    StepFinished,
    StepStarted,
    TextMessageEnd,
    ToolCallEnd,
    ToolCallStart,
)


def _last_text(graph, config) -> str:
    """The final answer of a finished run: the last AI message with text (the budget's
    stop message doesn't count: the work before it is what's worth keeping)."""
    for message in reversed(graph.get_state(config).values.get("messages", [])):
        text = message.text if isinstance(getattr(message, "text", None), str) else str(message.content)
        if message.type == "ai" and text.strip() and text.strip() != BUDGET_STOP:
            return text
    return ""


class _StepRun:
    """One plan step or room turn in progress: its tokens, attempts and clock cover every agent
    run made while it is open (worker and reviewer, retries included)."""

    def __init__(self, kind: str, number: int, total: int, agent: str):
        self.id, self.kind, self.number, self.total, self.agent = uuid.uuid4().hex, kind, number, total, agent
        self.started = time.monotonic()
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.tool_calls: list[tuple[str, str]] = []  # (call id, tool name) of each call its runs started
        self.refused: set[str] = set()  # the ids of the calls that were refused (rejected, never ran)

    def tools_used(self) -> list[str]:
        """The names of the tools its runs called, in order, without the refused calls."""
        return [name for call_id, name in self.tool_calls if call_id not in self.refused]


def _new_usage() -> dict:
    return {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}


class LockedSink:
    """Wraps a sink so that one ``emit`` at a time reaches it: runs on several threads share it."""

    def __init__(self, sink):
        self.sink, self.lock, self.closed = sink, threading.Lock(), False

    def emit(self, event) -> None:
        with self.lock:
            if not self.closed:
                self.sink.emit(event)

    def close(self) -> None:
        """Nothing more reaches the sink (a stopped swarm's workers may still be running)."""
        with self.lock:
            self.closed = True


class _Tally:
    """The sink an agent run emits to. Counts the run's model calls and tokens (every model message
    is a ``TextMessageEnd``) into ``usage``, and its tokens and tool calls into the open step; passes the
    events on only when ``forward`` (a quiet runner shows nothing, but still counts). The counts are made under
    ``lock``, which the runner's runs share."""

    def __init__(self, sink, step: "_StepRun | None", usage: dict, lock, forward: bool = True):
        self.sink, self.step, self.usage, self.lock, self.forward = sink, step, usage, lock, forward

    def emit(self, event) -> None:
        with self.lock:
            if isinstance(event, RunFinished) and self.step is not None:
                self.step.input_tokens += event.input_tokens
                self.step.output_tokens += event.output_tokens
            elif isinstance(event, ToolCallStart) and self.step is not None:
                self.step.tool_calls.append((event.tool_call_id, event.name))
            elif isinstance(event, ToolCallEnd) and self.step is not None and event.status == "rejected":
                self.step.refused.add(event.tool_call_id)
            elif isinstance(event, TextMessageEnd) and event.text.strip() != BUDGET_STOP:  # no model call made
                self.usage["model_calls"] += 1
                self.usage["input_tokens"] += event.input_tokens
                self.usage["output_tokens"] += event.output_tokens
        if self.forward:
            self.sink.emit(event)


def first_line(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    return lines[0].strip() if lines else ""


class _RunFailed(Exception):
    """One agent run raised (e.g. a provider error): the attempt fails, the chat goes on."""


# Deep Agents runs subagents with this limit; the call cap is what bounds a run.
RECURSION_LIMIT = 9_999


class Runner:
    """What plan and room runners share (the Template method: subclasses write
    ``_run``): runs agents on the caller's thread (so approvals work), within a
    request budget. ``make(name, extra_middleware)`` builds an agent
    (``worker_factory``); ``approvals`` decides paused calls; ``interaction`` asks the
    user; ``sink`` shows the agents' output and the runner's notices.

    Implements ``Interruptible``: ``request_pause()`` hands the user the turn at the
    next step or turn; ``stop()`` (or a KeyboardInterrupt from the front end) stops."""

    label = "[run]"

    def __init__(self, make, approvals, interaction: Interaction, sink: EventSink, *, thread: str,
                 budget: int = 150, interactive: bool = True):
        self.make, self.approvals, self.thread = make, approvals, thread
        self.interaction, self.sink = interaction, sink
        self.budget_size, self.budget = budget, Budget(budget)
        self.interactive = interactive
        self.pause_requested = False  # checked between turns or steps
        self.stopped = False          # checked before every agent run
        self.quiet = False  # one-shot runs: show nothing but the answer
        self.step: _StepRun | None = None  # the plan step or room turn being run
        self._local = threading.local()  # per thread: the thread the latest agent run used
        self.last_thread_id = thread
        self._usage_lock = threading.Lock()  # guards usage and step counts (runs may share them)
        self.usage = _new_usage()  # model calls and tokens of the current run (reset by run())

    @property
    def last_thread_id(self) -> str:
        """The thread the latest agent run on this thread used (see ``_run_agent``)."""
        return getattr(self._local, "last_thread_id", self.thread)

    @last_thread_id.setter
    def last_thread_id(self, thread_id: str) -> None:
        self._local.last_thread_id = thread_id

    # Interruptible ------------------------------------------------------------------------
    def request_pause(self) -> None:
        self.pause_requested = True

    def stop(self) -> None:
        self.stopped = True

    def say(self, text: str) -> None:
        """A notice for the user (e.g. "[plan] step 1/2 - general-purpose")."""
        if not self.quiet:
            self.sink.emit(NoticeEvent(text))

    def begin_step(self, kind: str, number: int, total: int, agent: str) -> _StepRun:
        """A plan step or room turn starts: agent runs from here count towards it."""
        step = self.step = _StepRun(kind, number, total, agent)
        if not self.quiet:
            self.sink.emit(StepStarted(step.id, kind, number, total, agent))
        return step

    def finish_step(self, step: _StepRun, status: str, summary: str = "") -> None:
        if self.step is step:
            self.step = None
        if not self.quiet:
            self.sink.emit(StepFinished(step.id, step.kind, step.number, status, step.attempts,
                                        time.monotonic() - step.started, step.input_tokens,
                                        step.output_tokens, first_line(summary)))

    def settle_unstarted(self, kind: str, numbers) -> None:
        """Steps settled without ever starting (the later steps of a stopped plan)."""
        for number in numbers:
            if not self.quiet:
                self.sink.emit(StepFinished(uuid.uuid4().hex, kind, number, "stopped", 0, 0.0, 0, 0, ""))

    def run(self, request: dict) -> dict:
        self.usage = _new_usage()
        try:
            return self._run(request)
        finally:  # a late pause or stop must not reach the next run
            self.pause_requested = self.stopped = False

    def _run(self, request: dict) -> dict:
        raise NotImplementedError

    def _speaker(self, name: str) -> str:
        """How an agent's output is labelled, e.g. "room / critic"."""
        return f"{self.label.strip('[]')} / {name}"

    def _exists(self, name: str) -> bool:
        """Is there an agent called ``name``? (Without building it, when the factory knows.)"""
        names = getattr(self.make, "names", None)
        return name in names if names is not None else self.make(name) is not None

    # one run of one agent -------------------------------------------------------------
    def _invoke(self, graph, text: str, thread_id: str, label=None, *, step=None, approvals=None,
                shown=None) -> str:
        """One run on ``thread_id``. ``step`` and ``approvals`` default to the runner's own;
        ``shown`` (default: not quiet) says whether the run's events reach the sink."""
        if self.stopped:
            raise KeyboardInterrupt  # handled like a second Ctrl+C: stop, keep the chat
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        step = self.step if step is None else step
        approvals = self.approvals if approvals is None else approvals
        try:
            forward = (not self.quiet) if shown is None else shown
            tally = _Tally(self.sink, step, self.usage, self._usage_lock, forward=forward)
            run_turn(graph, text, config, tally, approvals.decide, reload_skills=True,
                     label=label, show=True)
            return _last_text(graph, config)
        except (KeyboardInterrupt, _WrapUp):
            raise
        except Exception as error:  # a provider error or a bug: this attempt fails, not the chat
            raise _RunFailed(f"the run failed: {type(error).__name__}: {error}") from error

    def _run_agent(self, name: str, text: str, thread_id: str, capped: bool = True,
                   fresh_text: str | None = None, *, step=None, approvals=None, extra=(),
                   ask_on_budget: bool = True, shown=None) -> str:
        """Run one agent within the budget; when it runs out, ask to continue or wrap up.
        ``capped=False`` runs outside the budget (a room's final summary). The interrupted
        attempt runs again on a fresh thread (``thread_id`` + "+") with ``fresh_text`` when
        given, else ``text``; ``last_thread_id`` is the thread the latest attempt used.
        ``step`` and ``approvals`` default to the runner's own; ``extra`` is middleware added
        after the budget cap. ``ask_on_budget=False`` raises ``_WrapUp`` with the partial answer
        at once instead of asking (workers: only the main thread asks). ``shown=False`` keeps the
        run's events from the sink (it is still counted). Only the arguments given are passed on
        to ``_invoke``."""
        given = {key: value for key, value in (("step", step), ("approvals", approvals), ("shown", shown))
                 if value is not None}
        if not capped:
            self.last_thread_id = thread_id
            return self._invoke(self.make(name, tuple(extra)), text, thread_id, label=self._speaker(name),
                                **given)
        while True:
            self.last_thread_id = thread_id
            cap = BudgetCap(self.budget)
            result = self._invoke(self.make(name, (cap, *extra)), text, thread_id, label=self._speaker(name),
                                  **given)
            if not cap.fired:
                return result
            if not ask_on_budget or not self._continue_after_budget():
                raise _WrapUp(result)
            self.budget.extend()
            thread_id += "+"  # run the interrupted attempt again, from a fresh start
            if fresh_text is not None:
                text = fresh_text

    def _continue_after_budget(self) -> bool:
        self.say(f"{self.label} The request budget ({self.budget.size} model calls) is used up.")
        if not self.interactive:
            return False
        try:
            answer = self.interaction.choose(f"[c]ontinue with another {self.budget.size} or [w]rap up > ",
                                             ("continue", "wrap up"))
        except (EOFError, KeyboardInterrupt):
            return False
        return (answer or "").strip().lower().startswith("c")

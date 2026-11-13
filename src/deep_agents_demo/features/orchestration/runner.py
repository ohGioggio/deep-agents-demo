"""What plans and rooms share: running one agent on the caller's thread, with the
budget, approvals, and pausing or stopping when the front end asks (``Interruptible``)."""
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


class _Tally:
    """The sink an agent run emits to: passes events on and adds the run's tokens to the open step."""

    def __init__(self, sink, step: "_StepRun | None"):
        self.sink, self.step = sink, step

    def emit(self, event) -> None:
        if isinstance(event, RunFinished) and self.step is not None:
            self.step.input_tokens += event.input_tokens
            self.step.output_tokens += event.output_tokens
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
    def _invoke(self, graph, text: str, thread_id: str, label=None) -> str:
        if self.stopped:
            raise KeyboardInterrupt  # handled like a second Ctrl+C: stop, keep the chat
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        try:
            run_turn(graph, text, config, _Tally(self.sink, self.step), self.approvals.decide, reload_skills=True,
                     label=label, show=not self.quiet)
            return _last_text(graph, config)
        except (KeyboardInterrupt, _WrapUp):
            raise
        except Exception as error:  # a provider error or a bug: this attempt fails, not the chat
            raise _RunFailed(f"the run failed: {type(error).__name__}: {error}") from error

    def _run_agent(self, name: str, text: str, thread_id: str, capped: bool = True) -> str:
        """Run one agent within the budget; when it runs out, ask to continue or wrap up.
        ``capped=False`` runs outside the budget (a room's final summary)."""
        if not capped:
            return self._invoke(self.make(name), text, thread_id, label=self._speaker(name))
        while True:
            cap = BudgetCap(self.budget)
            result = self._invoke(self.make(name, (cap,)), text, thread_id, label=self._speaker(name))
            if not cap.fired:
                return result
            if not self._continue_after_budget():
                raise _WrapUp(result)
            self.budget.extend()
            thread_id += "+"  # run the interrupted attempt again, from a fresh start

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

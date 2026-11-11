"""What plans and rooms share: running one agent on the caller's thread, with the
budget, approvals, and pausing or stopping when the front end asks (``Interruptible``)."""
from deep_agents_demo.core.ports import EventSink, Interaction
from deep_agents_demo.core.turns import run_turn
from deep_agents_demo.features.orchestration.limits import (
    BUDGET_STOP,
    Budget,
    BudgetCap,
    _WrapUp,
)
from deep_agents_demo.models.events import NoticeEvent


def _last_text(graph, config) -> str:
    """The final answer of a finished run: the last AI message with text (the budget's
    stop message doesn't count: the work before it is what's worth keeping)."""
    for message in reversed(graph.get_state(config).values.get("messages", [])):
        text = message.text if isinstance(getattr(message, "text", None), str) else str(message.content)
        if message.type == "ai" and text.strip() and text.strip() != BUDGET_STOP:
            return text
    return ""


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

    # Interruptible ------------------------------------------------------------------------
    def request_pause(self) -> None:
        self.pause_requested = True

    def stop(self) -> None:
        self.stopped = True

    def say(self, text: str) -> None:
        """A notice for the user (e.g. "[plan] step 1/2 - general-purpose")."""
        if not self.quiet:
            self.sink.emit(NoticeEvent(text))

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
            run_turn(graph, text, config, self.sink, self.approvals.decide, reload_skills=True,
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

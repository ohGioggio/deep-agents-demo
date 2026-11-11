"""What plans and rooms share: running one agent on the chat loop's thread,
with the budget, approvals and Ctrl+C (pause, then stop)."""
import os
import signal
import sys
import threading

from deep_agents_demo.features.orchestration.limits import (
    BUDGET_STOP,
    Budget,
    BudgetCap,
    _WrapUp,
)


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
    """What plan and room runners share: runs agents on the chat loop's thread (so
    approvals work), within a request budget, with Ctrl+C pausing between turns.
    ``make(name, extra_middleware)`` builds an agent (``worker_factory``)."""

    label = "[run]"

    def __init__(self, make, approver, *, thread: str, budget: int = 150, interactive: bool = True,
                 ask=input, say=print):
        self.make, self.approver, self.thread = make, approver, thread
        self.budget_size, self.budget = budget, Budget(budget)
        self.interactive, self.ask, self.say = interactive, ask, say
        self.pause_requested = False  # set by Ctrl+C; checked between turns or steps
        self._previous_handler = None
        self.quiet = False  # one-shot runs: print nothing but the answer

    def _speaker(self, name: str) -> str:
        """How an agent's output is labelled, e.g. "room / critic"."""
        return f"{self.label.strip('[]')} / {name}"

    def _exists(self, name: str) -> bool:
        """Is there an agent called ``name``? (Without building it, when the factory knows.)"""
        names = getattr(self.make, "names", None)
        return name in names if names is not None else self.make(name) is not None

    def _with_ctrl_c(self, fn, *args):
        """Interactive: Ctrl+C pauses after the current turn (twice: stop now). In one-shot
        runs nobody can answer a pause, so Ctrl+C keeps its usual meaning."""
        self._previous_handler = None
        if self.interactive and threading.current_thread() is threading.main_thread():
            self._previous_handler = signal.signal(signal.SIGINT, self._on_sigint)
        try:
            return fn(*args)
        finally:
            self.pause_requested = False  # a late Ctrl+C must not pause the next run
            if self._previous_handler is not None:
                signal.signal(signal.SIGINT, self._previous_handler)

    # Ctrl+C -----------------------------------------------------------------------------
    def _on_sigint(self, _signum, _frame):
        """First Ctrl+C: pause after this step. Second: stop now. Nothing is printed
        through Python's stdout here: the handler can fire in the middle of a print."""
        if self.pause_requested:
            signal.signal(signal.SIGINT, self._previous_handler or signal.default_int_handler)
            raise KeyboardInterrupt
        self.pause_requested = True
        try:
            os.write(sys.stderr.fileno(),
                     f"\n{self.label} Pausing after this one (Ctrl+C again to stop now).\n".encode())
        except (OSError, ValueError, AttributeError):
            pass

    # one run of one agent -------------------------------------------------------------
    def _invoke(self, graph, text: str, thread_id: str, label=None) -> str:
        from deep_agents_demo.chat import (
            stream_turn,  # chat.py imports the app, which imports us
        )
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        try:
            stream_turn(graph, text, config, reload_skills=True, approver=self.approver, label=label,
                        show=not self.quiet)
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
            answer = self.ask(f"[c]ontinue with another {self.budget.size} or [w]rap up > ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower().startswith("c")

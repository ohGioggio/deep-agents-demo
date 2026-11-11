"""Subagent-driven execution: the parallel cap on ``task``, and plans with review.

- ``TaskLimiter``: at most N subagents run at once; extra ``task`` calls wait.
- ``run_plan``: the main agent hands over a plan. The tool pauses the turn with a
  LangGraph interrupt; the chat loop's ``PlanRunner`` runs the plan on its own
  thread (so approvals and Ctrl+C work, and finished steps never re-run) and
  resumes the tool with the outcome.
"""
import json
import os
import signal
import sys
import threading
import uuid
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

PLAN_PROMPT = (
    "Subagent-driven execution: for a long task, run_plan(goal, steps) runs each step with a "
    "worker agent (a step's 'agent', default general-purpose) and has a reviewer check it "
    "before the next one; failed steps are retried, then the user decides. Use it only when "
    "the user asks for it, or agrees when you propose it for a clearly long task. Give each "
    "step one clear, self-contained task. Afterwards tell the user how each step went. ")


@dataclass(frozen=True)
class Step:
    task: str
    agent: str = "general-purpose"


def run_plan(goal: str, steps: list[dict]) -> str:
    """Run a plan step by step: each step is done by a worker agent and checked by a
    reviewer before the next. goal: what the whole plan achieves. steps: a list of
    {"task": "...", "agent": "<agent name, optional>"}. Only when the user asked for it."""
    if not isinstance(steps, list) or not steps or not all(
            isinstance(s, dict) and str(s.get("task", "")).strip() for s in steps):
        return json.dumps({"error": "steps must be a non-empty list of {\"task\": ..., "
                                    "\"agent\": ...} with a task in each"})
    plan = [Step(str(s["task"]).strip(), str(s.get("agent") or "general-purpose")) for s in steps]
    outcome = interrupt({"type": "run_plan", "goal": goal,
                         "steps": [{"task": s.task, "agent": s.agent} for s in plan]})
    return json.dumps(outcome)


class Budget:
    """Model calls one plan may make, shared by all its workers and reviewers."""

    def __init__(self, size: int):
        self.size, self.used = size, 0
        self.lock = threading.Lock()
        self.limit = size

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)

    def spend(self) -> bool:
        """Count one call; False if the budget was already used up."""
        with self.lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True

    def extend(self) -> None:
        with self.lock:
            self.limit += self.size


BUDGET_STOP = "[Stopped: the plan's request budget is used up.]"


class BudgetCap(AgentMiddleware):
    """Ends a worker or reviewer run when its plan's budget is used up."""

    def __init__(self, budget: Budget):
        super().__init__()
        self.budget = budget
        self.fired = False  # this run was stopped by the budget (not merely finished on it)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self.budget.spend():
            return None
        self.fired = True
        return {"messages": [AIMessage(content=BUDGET_STOP)], "jump_to": "end"}


class _WrapUp(Exception):
    """The budget is used up and the plan wraps up."""


RESULT_LIMIT = 1500   # characters of each step's result in the outcome
EARLIER_LIMIT = 300   # characters of each earlier step's result in the next worker's input
_VERDICT_MARKS = "*#>_` \t"


def parse_verdict(text: str) -> tuple[bool, str]:
    """(passed, notes) from a reviewer reply whose first line is PASS or FAIL. Anything
    else counts as a fail, so nothing slips through unreviewed."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    first = lines[0].strip(_VERDICT_MARKS).upper() if lines else ""
    notes = "\n".join(lines[1:]).strip()
    if first.startswith("PASS"):
        return True, notes
    if first.startswith("FAIL"):
        return False, notes
    return False, f"the reviewer gave no clear verdict: {str(text or '')[:200]}"


@dataclass
class StepResult:
    step: Step
    status: str          # passed, skipped, failed or stopped
    result: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return {"task": self.step.task, "agent": self.step.agent, "status": self.status,
                "result": self.result[:RESULT_LIMIT], "notes": self.notes}


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
    def _invoke(self, graph, text: str, thread_id: str) -> str:
        from deep_agents_demo.chat import stream_turn  # chat.py imports the app, which imports us
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        try:
            stream_turn(graph, text, config, reload_skills=True, approver=self.approver)
            return _last_text(graph, config)
        except (KeyboardInterrupt, _WrapUp):
            raise
        except Exception as error:  # a provider error or a bug: this attempt fails, not the chat
            raise _RunFailed(f"the run failed: {type(error).__name__}: {error}") from error

    def _run_agent(self, name: str, text: str, thread_id: str, capped: bool = True) -> str:
        """Run one agent within the budget; when it runs out, ask to continue or wrap up.
        ``capped=False`` runs outside the budget (a room's final summary)."""
        if not capped:
            return self._invoke(self.make(name), text, thread_id)
        while True:
            cap = BudgetCap(self.budget)
            result = self._invoke(self.make(name, (cap,)), text, thread_id)
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



class PlanRunner(Runner):
    """Runs a plan on the chat loop's thread: per step a worker, then the reviewer;
    a failed review is retried with its notes; after ``max_attempts`` failures the
    user decides (retry, skip, stop). Approvals inside workers use ``approver``.

    Every run gets its own thread (plan id, step, round, attempt), so a later plan or
    a retry never sees earlier messages or an already used-up call cap."""

    label = "[plan]"

    def __init__(self, make, approver, *, thread: str, max_attempts: int = 2, budget: int = 150,
                 interactive: bool = True, ask=input, say=print):
        super().__init__(make, approver, thread=thread, budget=budget, interactive=interactive,
                         ask=ask, say=say)
        self.max_attempts = max_attempts
        self.plan_id = ""

    def _pause(self) -> str | None:
        """Ctrl+C was pressed: a note for the next worker, "" to go on, None to stop."""
        self.pause_requested = False
        try:
            answer = self.ask("[plan] You: (a note for the next step, /stop, or Enter to go on) ")
        except (EOFError, KeyboardInterrupt):
            return None
        return None if answer.strip().lower() == "/stop" else answer.strip()

    def _worker_input(self, goal, step, number, total, done, review_notes, user_note) -> str:
        parts = [f"Goal: {goal}", f"Step {number} of {total}: {step.task}"]
        if done:
            parts.append("Earlier steps:\n" + "\n".join(
                f"- Step {i} ({r.status}): {r.result[:EARLIER_LIMIT]}" for i, r in enumerate(done, 1)))
        if user_note:
            parts.append(f"Note from the user: {user_note}")
        if review_notes:
            parts.append(f"The reviewer's notes on your last attempt:\n{review_notes}")
        parts.append("Do only this step: later steps are done separately, after review. "
                     "Then reply with what you did and the result.")
        return "\n\n".join(parts)

    @staticmethod
    def _review_input(goal, step, number, total, result) -> str:
        return (f"Goal: {goal}\n\nStep {number} of {total}: {step.task}\n\n"
                f"The worker's result:\n{result}\n\nReply PASS or FAIL on the first line, then notes.")

    # one step ---------------------------------------------------------------------------
    def _attempts(self, goal, step, number, total, done, user_note, round_no, notes="") -> StepResult:
        label = f"[plan] step {number}/{total}"
        if self.make(step.agent) is None:
            return StepResult(step, "failed", notes=f"no agent named {step.agent!r}")
        result = ""
        for attempt in range(1, self.max_attempts + 1):
            self.say(f"{label} - {step.agent}")
            base = f"{self.thread}:plan:{self.plan_id}:{number}:{round_no}:{attempt}"
            try:
                result = self._run_agent(step.agent, self._worker_input(
                    goal, step, number, total, done, notes, user_note), f"{base}:worker")
                verdict = self._run_agent("reviewer", self._review_input(
                    goal, step, number, total, result), f"{base}:review")
                passed, notes = parse_verdict(verdict)
            except _RunFailed as error:
                passed, notes = False, str(error)
            if passed:
                self.say(f"{label} passed review")
                return StepResult(step, "passed", result, notes)
            self.say(f"{label} failed review (attempt {attempt}/{self.max_attempts}): {notes}")
        return StepResult(step, "failed", result, notes)

    def _decide(self, outcome: StepResult, number: int, total: int) -> str:
        """After a failed step: 'retry', 'skip' or 'stop'."""
        self.say(f"[plan] step {number}/{total} failed: {outcome.notes}")
        if not self.interactive:
            return "stop"
        try:
            answer = self.ask("[r]etry  [s]kip  [x] stop > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "stop"
        return {"r": "retry", "s": "skip"}.get(answer[:1], "stop")

    # the plan ------------------------------------------------------------------------------
    def run(self, request: dict) -> dict:
        return self._with_ctrl_c(self._run, request)

    def _run(self, request: dict) -> dict:
        self.budget = Budget(self.budget_size)  # one budget per plan
        self.plan_id = uuid.uuid4().hex[:8]     # fresh threads for every plan
        goal = str(request.get("goal", ""))
        steps = [Step(str(s["task"]), str(s.get("agent") or "general-purpose"))
                 for s in request.get("steps", [])]
        done: list[StepResult] = []
        user_note, notes, round_no, number = "", "", 1, 0
        while number < len(steps):
            step = steps[number]
            if self.pause_requested:
                note = self._pause()
                if note is None:
                    done += [StepResult(rest, "stopped", notes="stopped by the user")
                             for rest in steps[number:]]
                    break
                user_note = note or user_note
            try:
                outcome = self._attempts(goal, step, number + 1, len(steps), done, user_note,
                                         round_no, notes)
            except _WrapUp as wrap:
                partial = str(wrap.args[0]) if wrap.args else ""
                done.append(StepResult(step, "stopped", partial,
                                       notes="budget used up: the plan's request budget ran out"))
                done += [StepResult(rest, "stopped",
                                    notes="budget used up: the plan's request budget ran out")
                         for rest in steps[number + 1:]]
                break
            user_note = ""
            if outcome.status == "failed":
                choice = self._decide(outcome, number + 1, len(steps))
                if choice == "retry":  # fresh threads, keeping the last review's notes
                    round_no, notes = round_no + 1, outcome.notes
                    continue
                if choice == "skip":
                    outcome.status = "skipped"
                else:
                    done.append(outcome)
                    done += [StepResult(rest, "stopped", notes="the plan was stopped")
                             for rest in steps[number + 1:]]
                    break
            done.append(outcome)
            number, round_no, notes = number + 1, 1, ""
        return {"goal": goal, "steps": [r.as_dict() for r in done],
                "note": ("Tell the user how each step went. The user decided what happens to "
                         "failed, skipped and stopped steps: don't redo them yourself unless "
                         "they ask.")}


ROOM_PROMPT = (
    "start_room(question, participants) runs a moderated discussion between the named agents "
    "(at least two) and returns its summary; use it only when the user asks for a discussion "
    "or a room. ")


def start_room(question: str, participants: list[str]) -> str:
    """Run a moderated discussion: the named agents take turns on the question, a
    moderator decides who speaks and ends it with a summary. participants: at least two
    agent names (not "moderator"). Only when the user asks for a discussion."""
    names = [str(p).strip() for p in participants] if isinstance(participants, list) else []
    if (not str(question or "").strip() or len(set(names)) < 2 or len(set(names)) != len(names)
            or "moderator" in names or not all(names)):
        return json.dumps({"error": "participants must be at least two different agent names, "
                                    "not 'moderator', and the question can't be empty"})
    outcome = interrupt({"type": "start_room", "question": str(question).strip(),
                         "participants": names})
    return json.dumps(outcome)


def orchestration_feature():
    from deep_agents_demo.agent import Feature  # agent.py imports this module
    return Feature(name="orchestration", prompt=PLAN_PROMPT + ROOM_PROMPT,
                   tools=(run_plan, start_room))


class TaskLimiter(AgentMiddleware):
    """At most ``limit`` ``task`` calls run at the same time (LangGraph runs a step's
    tool calls in parallel threads); the rest wait for a free slot."""

    def __init__(self, limit: int):
        super().__init__()
        self.slots = threading.BoundedSemaphore(limit)

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "task":
            return handler(request)
        with self.slots:
            return handler(request)

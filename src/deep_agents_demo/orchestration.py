"""Subagent-driven execution: the parallel cap on ``task``, and plans with review.

- ``TaskLimiter``: at most N subagents run at once; extra ``task`` calls wait.
- ``run_plan``: the main agent hands over a plan. The tool pauses the turn with a
  LangGraph interrupt; the chat loop's ``PlanRunner`` runs the plan on its own
  thread (so approvals and Ctrl+C work, and finished steps never re-run) and
  resumes the tool with the outcome.
"""
import json
import threading
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
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
    """The final answer of a finished run: the last AI message with text."""
    for message in reversed(graph.get_state(config).values.get("messages", [])):
        if message.type == "ai" and str(message.content).strip():
            return message.text if isinstance(message.text, str) else str(message.content)
    return ""


class PlanRunner:
    """Runs a plan on the chat loop's thread: per step a worker, then the reviewer;
    a failed review is retried with its notes; after ``max_attempts`` failures the
    user decides (retry, skip, stop). Approvals inside workers use ``approver``."""

    def __init__(self, make, approver, *, thread: str, max_attempts: int = 2, budget: int = 150,
                 interactive: bool = True, ask=input, say=print):
        self.make, self.approver, self.thread = make, approver, thread
        self.max_attempts, self.budget_size = max_attempts, budget
        self.interactive, self.ask, self.say = interactive, ask, say

    # one run of one agent -------------------------------------------------------------
    def _invoke(self, graph, text: str, thread_id: str) -> str:
        from deep_agents_demo.chat import stream_turn  # chat.py imports the app, which imports us
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
        stream_turn(graph, text, config, reload_skills=True, approver=self.approver)
        return _last_text(graph, config)

    def _worker_input(self, goal, step, number, total, done, review_notes, user_note) -> str:
        parts = [f"Goal: {goal}", f"Step {number} of {total}: {step.task}"]
        if done:
            parts.append("Earlier steps:\n" + "\n".join(
                f"- Step {i} ({r.status}): {r.result[:EARLIER_LIMIT]}" for i, r in enumerate(done, 1)))
        if user_note:
            parts.append(f"Note from the user: {user_note}")
        if review_notes:
            parts.append(f"The reviewer's notes on your last attempt:\n{review_notes}")
        parts.append("Do this step, then reply with what you did and the result.")
        return "\n\n".join(parts)

    @staticmethod
    def _review_input(goal, step, number, total, result) -> str:
        return (f"Goal: {goal}\n\nStep {number} of {total}: {step.task}\n\n"
                f"The worker's result:\n{result}\n\nReply PASS or FAIL on the first line, then notes.")

    # one step ---------------------------------------------------------------------------
    def _attempts(self, goal, step, number, total, done, user_note) -> StepResult:
        label = f"[plan] step {number}/{total}"
        if self.make(step.agent) is None:
            return StepResult(step, "failed", notes=f"no agent named {step.agent!r}")
        result = notes = ""
        for attempt in range(1, self.max_attempts + 1):
            self.say(f"{label} · {step.agent}")
            base = f"{self.thread}:plan:{number}:{attempt}"
            result = self._invoke(self.make(step.agent),
                                  self._worker_input(goal, step, number, total, done, notes, user_note),
                                  f"{base}:worker")
            reviewer = self.make("reviewer")
            verdict = self._invoke(reviewer, self._review_input(goal, step, number, total, result),
                                   f"{base}:review")
            passed, notes = parse_verdict(verdict)
            if passed:
                self.say(f"{label} ✓ reviewed")
                return StepResult(step, "passed", result, notes)
            self.say(f"{label} ✗ attempt {attempt}/{self.max_attempts}: {notes}")
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
        goal = str(request.get("goal", ""))
        steps = [Step(str(s["task"]), str(s.get("agent") or "general-purpose"))
                 for s in request.get("steps", [])]
        done: list[StepResult] = []
        user_note = ""
        number = 0
        while number < len(steps):
            step = steps[number]
            outcome = self._attempts(goal, step, number + 1, len(steps), done, user_note)
            user_note = ""
            if outcome.status == "failed":
                choice = self._decide(outcome, number + 1, len(steps))
                if choice == "retry":
                    continue
                if choice == "skip":
                    outcome.status = "skipped"
                else:
                    done.append(outcome)
                    done += [StepResult(rest, "stopped", notes="the plan was stopped")
                             for rest in steps[number + 1:]]
                    break
            done.append(outcome)
            number += 1
        return {"goal": goal, "steps": [r.as_dict() for r in done]}


def orchestration_feature():
    from deep_agents_demo.agent import Feature  # agent.py imports this module
    return Feature(name="orchestration", prompt=PLAN_PROMPT, tools=(run_plan,))


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

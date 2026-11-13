"""Subagent-driven execution: plans with review.

(The parallel cap on ``task``, ``TaskLimiter``, is in ``core.middleware``.)

- ``run_plan``: the main agent hands over a plan. The tool pauses the turn with a
  LangGraph interrupt; the chat loop's ``PlanRunner`` runs the plan on its own
  thread (so approvals and Ctrl+C work, and finished steps never re-run) and
  resumes the tool with the outcome.
"""
import json
import uuid

from langgraph.types import interrupt

from deep_agents_demo.features.orchestration.limits import Budget, _WrapUp
from deep_agents_demo.features.orchestration.runner import Runner, _RunFailed
from deep_agents_demo.models.outcomes import Step, StepResult

PLAN_PROMPT = (
    "Subagent-driven execution: for a long task, run_plan(goal, steps) runs each step with a "
    "worker agent (a step's 'agent', default general-purpose) and has a reviewer check it "
    "before the next one; failed steps are retried, then the user decides. Use it only when "
    "the user asks for it, or agrees when you propose it for a clearly long task. Give each "
    "step one clear, self-contained task. Afterwards tell the user how each step went. ")


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


class PlanRunner(Runner):
    """Runs a plan on the chat loop's thread: per step a worker, then the reviewer;
    a failed review is retried with its notes; after ``max_attempts`` failures the
    user decides (retry, skip, stop). Approvals inside workers use ``approvals``.

    Every run gets its own thread (plan id, step, round, attempt), so a later plan or
    a retry never sees earlier messages or an already used-up call cap."""

    label = "[plan]"

    def __init__(self, make, approvals, interaction, sink, *, thread: str, max_attempts: int = 2,
                 budget: int = 150, interactive: bool = True):
        super().__init__(make, approvals, interaction, sink, thread=thread, budget=budget,
                         interactive=interactive)
        self.max_attempts = max_attempts
        self.plan_id = ""

    def _pause(self) -> str | None:
        """Ctrl+C was pressed: a note for the next worker, "" to go on, None to stop."""
        self.pause_requested = False
        try:
            answer = self.interaction.ask("[plan] You: (a note for the next step, /stop, or Enter to go on) ")
        except (EOFError, KeyboardInterrupt):
            return None
        if answer is None:
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
        if not self._exists(step.agent):
            return StepResult(step, "failed", notes=f"no agent named {step.agent!r}")
        result = ""
        for attempt in range(1, self.max_attempts + 1):
            if self.step is not None:
                self.step.attempts += 1
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
            answer = (self.interaction.choose("[r]etry  [s]kip  [x] stop > ", ("retry", "skip", "stop"))
                      or "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "stop"
        return {"r": "retry", "s": "skip"}.get(answer[:1], "stop")

    def _settle_stopped(self, current, number: int, total: int, summary: str = "") -> None:
        """The plan stopped at step ``number`` (0-based): close its open events, then settle
        the later steps (and this one too, when it never started)."""
        if current is not None:
            self.finish_step(current, "stopped", summary)
            self.settle_unstarted("plan", range(number + 2, total + 1))
        else:
            self.settle_unstarted("plan", range(number + 1, total + 1))

    # the plan ------------------------------------------------------------------------------
    def _run(self, request: dict) -> dict:
        self.budget = Budget(self.budget_size)  # one budget per plan
        self.plan_id = uuid.uuid4().hex[:8]     # fresh threads for every plan
        goal = str(request.get("goal", ""))
        steps = [Step(str(s["task"]), str(s.get("agent") or "general-purpose"))
                 for s in request.get("steps", [])]
        done: list[StepResult] = []
        user_note, notes, round_no, number = "", "", 1, 0
        total = len(steps)
        current = None  # the open step's events: kept across a retry, closed when it settles
        while number < total:
            step = steps[number]
            if self.pause_requested:
                note = self._pause()
                if note is None:
                    done += [StepResult(rest, "stopped", notes="stopped by the user")
                             for rest in steps[number:]]
                    self._settle_stopped(current, number, total)
                    break
                user_note = note or user_note
            if current is None:
                current = self.begin_step("plan", number + 1, total, step.agent)
            self.step = current
            try:
                outcome = self._attempts(goal, step, number + 1, total, done, user_note,
                                         round_no, notes)
            except KeyboardInterrupt:  # a second Ctrl+C: stop the plan, keep the chat
                self.say(f"{self.label} Stopped.")
                done += [StepResult(rest, "stopped", notes="stopped by the user")
                         for rest in steps[number:]]
                self._settle_stopped(current, number, total)
                break
            except _WrapUp as wrap:
                partial = str(wrap.args[0]) if wrap.args else ""
                done.append(StepResult(step, "stopped", partial,
                                       notes="budget used up: the plan's request budget ran out"))
                done += [StepResult(rest, "stopped",
                                    notes="budget used up: the plan's request budget ran out")
                         for rest in steps[number + 1:]]
                self._settle_stopped(current, number, total, partial)
                break
            user_note = ""
            if outcome.status == "failed":
                choice = self._decide(outcome, number + 1, total)
                if choice == "retry":  # fresh threads, keeping the last review's notes
                    round_no, notes = round_no + 1, outcome.notes
                    continue
                if choice == "skip":
                    outcome.status = "skipped"
                else:
                    done.append(outcome)
                    done += [StepResult(rest, "stopped", notes="the plan was stopped")
                             for rest in steps[number + 1:]]
                    self.finish_step(current, "failed", outcome.result or outcome.notes)
                    self.settle_unstarted("plan", range(number + 2, total + 1))
                    break
            self.finish_step(current, outcome.status, outcome.result or outcome.notes)
            current = None
            done.append(outcome)
            number, round_no, notes = number + 1, 1, ""
        return {"goal": goal, "steps": [r.as_dict() for r in done],
                "note": ("Tell the user how each step went. The user decided what happens to "
                         "failed, skipped and stopped steps: don't redo them yourself unless "
                         "they ask.")}



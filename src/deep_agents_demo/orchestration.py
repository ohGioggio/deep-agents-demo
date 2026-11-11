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

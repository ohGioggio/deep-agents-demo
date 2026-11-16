"""What a plan or a room hands back."""
from dataclasses import dataclass
from typing import NotRequired, TypedDict


@dataclass(frozen=True)
class Step:
    task: str
    agent: str = "general-purpose"


RESULT_LIMIT = 1500   # characters of each step's result in the outcome


@dataclass
class StepResult:
    step: Step
    status: str          # passed, skipped, failed or stopped
    result: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return {"task": self.step.task, "agent": self.step.agent, "status": self.status,
                "result": self.result[:RESULT_LIMIT], "notes": self.notes}

    to_dict = as_dict


class PlanOutcome(TypedDict):
    """What ``run_plan`` returns to the agent (as JSON)."""
    goal: str
    steps: list[dict]       # StepResult.as_dict() for each step
    note: str


class RoomOutcome(TypedDict):
    """What ``start_room`` returns to the agent (as JSON)."""
    question: str
    participants: list[str]
    turns: NotRequired[int]
    summary: NotRequired[str]
    transcript: NotRequired[str]   # the saved Markdown transcript's path
    ended_by: NotRequired[str]     # moderator, turn limit, budget or stopped
    note: NotRequired[str]
    error: NotRequired[str]
    positions: NotRequired[dict]          # name -> {"position", "confidence"}, each participant's last
    decision: NotRequired[str]            # the moderator's recommendation, in one sentence
    dissent: NotRequired[list[str]]       # who still disagrees, and why
    actions: NotRequired[list[str]]       # next steps
    model_calls: NotRequired[int]         # every model call the room made, moderator included
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    decision_sources: NotRequired[dict]   # how each moderator decision was read (json, text, ...)


class SwarmTask(TypedDict):
    """One task of a swarm, as the outcome reports it."""
    task: str
    agent: str
    status: str            # done, failed, cut, stopped or not started
    result: str            # the first line; the report has the whole result
    input_tokens: int
    output_tokens: int
    seconds: float


class SwarmOutcome(TypedDict):
    """What ``run_swarm`` returns to the agent (as JSON)."""
    goal: str
    tasks: list[SwarmTask]
    merged: str
    merge_with: str
    ended_by: str          # done, budget, user, stopped or failed (every task failed)
    model_calls: int       # every model call the swarm made, merge included
    input_tokens: int
    output_tokens: int
    seconds: float
    error: str
    report: str            # the saved Markdown report's path ("" when none was saved)
    report_error: NotRequired[str]  # why the report could not be saved; the outcome is still returned
    note: str


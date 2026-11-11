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

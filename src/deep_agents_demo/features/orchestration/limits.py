"""The request budget shared by a plan's or a room's agents."""
import threading

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage


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

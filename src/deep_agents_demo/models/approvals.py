"""A paused tool call as a front end sees it, and the user's answer."""
from dataclasses import asdict, dataclass

NO_HUMAN = ("Not approved: nobody can approve changes in a one-shot run (--prompt). "
            "Tell the user to make this change in an interactive chat.")


@dataclass(frozen=True)
class ApprovalRequest:
    """What the user is asked to approve. ``summary`` is shown first; ``diff`` and
    ``full`` on request."""
    title: str                 # the tool name
    summary: str
    diff: str
    full: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    approved: bool
    message: str = ""          # for a rejection: what the agent reads

    def to_resume(self) -> dict:
        """The decision in HumanInTheLoopMiddleware's format."""
        if self.approved:
            return {"type": "approve"}
        return {"type": "reject", "message": self.message}

    def to_dict(self) -> dict:
        return asdict(self)

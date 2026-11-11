"""Deciding paused tool calls: the rules here, the asking through an ``Interaction``.

Tools listed in a feature's ``approvals`` are paused by LangChain's
HumanInTheLoopMiddleware. Each pause arrives as a request with one entry per tool
call; ``Approvals.decide`` returns one decision per entry: approve, or reject with a
message the agent reads.

A ``preview(tool_name, args)`` callable can describe a call before it runs (skill
and agent changes, risky actions do). A call whose preview has errors is rejected
without asking, so the agent can fix it; without a human (a --prompt run) every
call is rejected. Otherwise the ``Interaction`` gets an ``ApprovalRequest``.
"""
import json

from deep_agents_demo.core.ports import Interaction
from deep_agents_demo.models.approvals import NO_HUMAN, ApprovalRequest


def generic_request(action: dict) -> ApprovalRequest:
    """A call nothing previews: its name and arguments."""
    args = json.dumps(action.get("args", {}), ensure_ascii=False)
    shown = json.dumps(action.get("args", {}), indent=2, ensure_ascii=False)
    return ApprovalRequest(title=action["name"], diff=shown, full=shown,
                           summary=f"Approval needed: {action['name']}({args[:300]}"
                                   f"{'...' if len(args) > 300 else ''})")


class Approvals:
    def __init__(self, preview=None, interaction: Interaction | None = None, interactive: bool = True):
        self.preview, self.interaction = preview, interaction
        self.interactive = interactive and interaction is not None

    def decide(self, request: dict) -> dict:
        """The HumanInTheLoopMiddleware response for one paused request."""
        return {"decisions": [self._decide(action) for action in request.get("action_requests", [])]}

    def _decide(self, action: dict) -> dict:
        change = self.preview(action["name"], action.get("args", {})) if self.preview else None
        if change is not None and change.errors:
            return {"type": "reject", "message": "Not shown to the user because it is invalid: "
                    + "; ".join(change.errors) + ". Fix it and try again."}
        if not self.interactive:
            return {"type": "reject", "message": NO_HUMAN}
        request = change.as_request() if change is not None else generic_request(action)
        return self.interaction.decide(request).to_resume()

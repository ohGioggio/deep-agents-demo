"""The approval step: show paused tool calls to the user and collect decisions.

Tools listed in a feature's ``approvals`` are paused by LangChain's
HumanInTheLoopMiddleware. Each pause arrives as a request with one entry per tool
call; ``Approver.decide`` returns one decision per entry: approve, or reject with a
message the agent reads.

A ``preview(tool_name, args)`` callable can describe a call before it runs (skill
changes do). A call whose preview has errors is rejected without asking, so the
agent can fix it; without a human (a --prompt run) every call is rejected.
"""
import json

NO_HUMAN = ("Not approved: nobody can approve changes in a one-shot run (--prompt). "
            "Tell the user to make this change in an interactive chat.")


class Approver:
    def __init__(self, preview=None, ask=None, say=print, interactive=True):
        # ask=None reads the terminal through input() at call time.
        self.preview, self.say, self.interactive = preview, say, interactive
        self.ask = ask or (lambda question: input(question))

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
        self.say("\n" + (change.summary() if change is not None else self._generic(action)))
        while True:
            try:
                answer = self.ask("[a]pprove  [r]eject (with a reason)  [d]iff  [f]ull file > ")
            except (EOFError, KeyboardInterrupt):
                return {"type": "reject", "message": "The user did not approve this change."}
            answer = answer.strip().lower()
            if answer in {"a", "approve", "y", "yes"}:
                return {"type": "approve"}
            if answer in {"r", "reject", "n", "no"}:
                try:
                    reason = self.ask("Reason for the agent (optional) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                return {"type": "reject", "message": "The user rejected this change"
                        + (f": {reason}" if reason else ". Don't retry unless they ask.")}
            if answer in {"d", "diff"} and change is not None:
                self.say(change.diff())
            elif answer in {"f", "full"} and change is not None:
                self.say(change.full())
            elif answer in {"d", "diff", "f", "full"}:
                self.say(json.dumps(action.get("args", {}), indent=2, ensure_ascii=False))
            else:
                self.say("Type a, r, d or f.")

    @staticmethod
    def _generic(action: dict) -> str:
        args = json.dumps(action.get("args", {}), ensure_ascii=False)
        return f"Approval needed: {action['name']}({args[:300]}{'...' if len(args) > 300 else ''})"

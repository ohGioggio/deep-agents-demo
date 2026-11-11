"""The CLI's ``Interaction``: approval prompts and questions over ``input()``."""
from collections.abc import Callable

from deep_agents_demo.models.approvals import ApprovalRequest, Decision

APPROVE_PROMPT = "[a]pprove  [r]eject (with a reason)  [d]iff  [f]ull file > "


class CliInteraction:
    """``ask`` reads a line (default ``input``), ``say`` shows one (default ``print``);
    both are looked up at call time, so tests and redirected streams work."""

    def __init__(self, ask: Callable[[str], str] | None = None, say: Callable[..., None] | None = None):
        self._ask = ask or (lambda prompt: input(prompt))
        self._say = say or (lambda *args: print(*args))

    def decide(self, request: ApprovalRequest) -> Decision:
        self._say("\n" + request.summary)
        while True:
            try:
                answer = self._ask(APPROVE_PROMPT)
            except (EOFError, KeyboardInterrupt):
                return Decision(False, "The user did not approve this change.")
            answer = answer.strip().lower()
            if answer in {"a", "approve", "y", "yes"}:
                return Decision(True)
            if answer in {"r", "reject", "n", "no"}:
                try:
                    reason = self._ask("Reason for the agent (optional) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                return Decision(False, "The user rejected this change"
                                + (f": {reason}" if reason else ". Don't retry unless they ask."))
            if answer in {"d", "diff"}:
                self._say(request.diff)
            elif answer in {"f", "full"}:
                self._say(request.full)
            else:
                self._say("Type a, r, d or f.")

    def choose(self, prompt: str, options: tuple[str, ...]) -> str:
        """The typed answer (the caller matches it to ``options``); "" if none."""
        try:
            return self._ask(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""

    def ask(self, prompt: str) -> str | None:
        try:
            return self._ask(prompt)
        except (EOFError, KeyboardInterrupt):
            return None

"""Guardrails: risky actions pause for the user's yes or no.

- ``execute``: only shell commands that match a danger pattern (recursive or forced
  deletes, history-rewriting Git, piping a download into a shell, disk tools, ...).
  Ordinary commands run freely.
- ``delete``: removing a file or folder always asks.
- ``forget``: deleting one of the user's memories always asks.

The pause uses the same approval step as skill changes (LangChain's
HumanInTheLoopMiddleware via ``Feature.approvals``), so subagents inherit it and a
one-shot ``--prompt`` run refuses these actions. ``preview`` describes a paused call
for the approval prompt, including why it was flagged.
"""
import json
from dataclasses import dataclass, field

from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.guardrails.patterns import risky_command


def _execute_is_risky(request) -> bool:
    """HumanInTheLoopMiddleware ``when`` predicate: pause only risky commands."""
    return bool(risky_command(str(request.tool_call.get("args", {}).get("command", ""))))


@dataclass
class RiskyAction:
    """A paused risky call, as shown in the approval prompt."""
    name: str
    args: dict
    target: str                      # the command, path or memory being acted on
    reasons: list = field(default_factory=list)
    errors: list = field(default_factory=list)  # never auto-rejected: the user decides

    def summary(self) -> str:
        lines = [f"Risky action requested: {self.name}", f"  {self.target}"]
        lines += [f"  Why it's flagged: {reason}" for reason in self.reasons]
        return "\n".join(lines)

    def diff(self) -> str:
        return json.dumps(self.args, indent=2, ensure_ascii=False)

    full = diff


class Guardrails:
    """Builds the guardrail feature and describes paused risky calls.

    ``describe_memory(memory_id)`` (optional) turns a memory id into its text, so
    the prompt for ``forget`` shows what would be forgotten.
    """

    def __init__(self, describe_memory=None):
        self.describe_memory = describe_memory

    def preview(self, name: str, args: dict) -> RiskyAction | None:
        if name == "execute":
            command = str(args.get("command", ""))
            reasons = risky_command(command)
            return RiskyAction(name, args, f"Command: {command}", reasons) if reasons else None
        if name == "delete":
            return RiskyAction(name, args, f"Path: {args.get('file_path', '?')}",
                               ["deletes a file or folder"])
        if name == "forget":
            memory_id = str(args.get("memory_id", "?"))
            text = self.describe_memory(memory_id) if self.describe_memory else None
            return RiskyAction(name, args, f"Memory: {text}" if text else f"Memory id: {memory_id}",
                               ["permanently deletes one of your memories"])
        return None

    @staticmethod
    def feature() -> Feature:
        prompt = ("Some actions pause for the user's yes or no: shell commands that delete "
                  "files, rewrite Git history, push, run downloaded scripts or change the "
                  "system; deleting files; forgetting memories. If one is rejected, don't "
                  "retry it or work around it; ask the user what they want instead. ")
        return Feature(name="guardrails", prompt=prompt,
                       approvals=(("execute", _execute_is_risky), "delete", "forget"))

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
import re
from dataclasses import dataclass, field

from deep_agents_demo.agent import Feature

# (pattern, why it is risky). Patterns are matched case-sensitively on the whole
# command, so they also catch risky parts of a pipeline or a `&&` chain.
DANGER = [
    (r"\brm\s+(?:-\w*[rRf]\w*|--recursive|--force)\b", "deletes files recursively or without asking"),
    (r"\bfind\b.*\s-(?:delete\b|exec\s+rm\b)", "deletes every file a search matches"),
    (r"\bgit\s+push\b", "pushes to a remote repository"),
    (r"\bgit\s+reset\s+--hard\b", "throws away uncommitted work"),
    (r"\bgit\s+clean\s+-\w*f", "deletes untracked files"),
    (r"\bgit\s+branch\s+-D\b", "force-deletes a branch"),
    (r"\bgit\s+checkout\s+--\s", "throws away local changes"),
    (r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", "runs a downloaded script"),
    (r"\b(?:curl|wget)\b[^|]*\|\s*python3?\b", "runs a downloaded script"),
    (r"\b(?:mkfs(?:\.\w+)?|shred)\b", "destroys data on a disk or file"),
    (r"\bdd\s+[^|;&]*\bif=", "writes raw data with dd"),
    (r"\b(?:kill\s+-9|pkill|killall)\b", "kills processes"),
    (r"\b(?:pip|uv\s+pip|npm|apt(?:-get)?)\s+(?:uninstall|remove|purge)\b", "uninstalls packages"),
    (r"\bsudo\b", "runs as an administrator"),
    (r"\b(?:chmod|chown)\s+-\w*R", "changes permissions or ownership recursively"),
    (r"\bchmod\s+(?:-\w+\s+)*0?777\b", "makes files writable by everyone"),
    (r">\s*/dev/(?:sd|nvme|disk)", "writes to a disk device"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "is a fork bomb"),
]
_DANGER = [(re.compile(pattern), reason) for pattern, reason in DANGER]


def risky_command(command: str) -> list[str]:
    """Why a shell command is risky (empty if it isn't), without duplicates."""
    reasons = []
    for pattern, reason in _DANGER:
        if pattern.search(command or "") and reason not in reasons:
            reasons.append(reason)
    return reasons


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

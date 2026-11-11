"""What makes a shell command risky."""
import re

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

"""The tools registry: every tool the agent can have, and which ones it may use.

``ToolRegistry`` lists Deep Agents' built-in tools and every feature's tools, each
with its group (the feature it comes from) and a one-line description. It is for
the code, not the chat: the user never lists or switches tools.

Two things narrow what the agent sees, both enforced by ``ToolAccess``:

- **Disabled tools** (``DISABLED_TOOLS`` in .env) are hidden from the model and
  refused if called anyway. A few core tools can't be disabled.
- **Skill-only tools** (``Feature.skill_only``) stay hidden until the agent reads a
  skill that lists them under ``allowed-tools``, so they don't crowd every turn.
"""
import re
from dataclasses import dataclass

from langchain_core.tools import BaseTool

# Deep Agents' own tools: name -> (group, description).
BUILTIN = {
    "ls": ("files", "List the files in a folder"),
    "read_file": ("files", "Read a file"),
    "write_file": ("files", "Create or overwrite a file"),
    "edit_file": ("files", "Replace text in a file"),
    "delete": ("files", "Delete a file or folder"),
    "glob": ("files", "Find files by name pattern"),
    "grep": ("files", "Search inside files"),
    "execute": ("shell", "Run a shell command (only with a sandbox)"),
    "write_todos": ("planning", "Keep a to-do list for multi-step work"),
    "task": ("delegation", "Hand a self-contained subtask to a subagent"),
}
# Skills and planning depend on these, so they can't be disabled.
CORE = frozenset({"ls", "read_file", "write_todos"})


def tool_name(tool) -> str:
    """The name the model calls a tool by: a BaseTool's name, a function's name."""
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("function", {}).get("name", ""))
    return tool.__name__


def _description(tool) -> str:
    text = tool.description if isinstance(tool, BaseTool) else (tool.__doc__ or "")
    return " ".join(text.split()).split(". ")[0].rstrip(".")


@dataclass(frozen=True)
class ToolEntry:
    name: str
    group: str            # "files", "shell", ... or the feature that adds the tool
    description: str
    skill_only: bool = False


class ToolRegistry:
    """Every tool the agent can have: Deep Agents' built-ins plus the features' tools."""

    def __init__(self, features=()):
        self._entries = {name: ToolEntry(name, group, text) for name, (group, text) in BUILTIN.items()}
        self._requires = {tool: skill for feature in features for tool, skill in feature.needs_skill}
        for feature in features:
            names = {tool_name(tool): tool for tool in (*feature.tools, *feature.lookups)}
            unknown = set(feature.skill_only) - set(names)
            if unknown:
                raise ValueError(f"feature {feature.name!r}: skill_only names tools it doesn't "
                                 f"have: {', '.join(sorted(unknown))}")
            for name, tool in names.items():
                self._entries.setdefault(name, ToolEntry(name, feature.name, _description(tool),
                                                         name in feature.skill_only))

    def entries(self) -> list[ToolEntry]:
        return sorted(self._entries.values(), key=lambda entry: (entry.group, entry.name))

    def names(self) -> frozenset:
        return frozenset(self._entries)

    def skill_only(self) -> frozenset:
        return frozenset(entry.name for entry in self._entries.values() if entry.skill_only)

    def requirements(self) -> dict:
        """Tool -> the skill the agent must read before the tool's first use."""
        return dict(self._requires)

    def disabled(self, requested) -> frozenset:
        """The requested names that can actually be disabled (known, not core)."""
        return (frozenset(requested) & self.names()) - CORE

    def problems(self, requested) -> list[str]:
        """Why some DISABLED_TOOLS entries are ignored, for a startup warning."""
        found = []
        unknown = sorted(set(requested) - self.names())
        if unknown:
            found.append(f"DISABLED_TOOLS: no tool named {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(self.names()))})")
        core = sorted(set(requested) & CORE)
        if core:
            found.append(f"DISABLED_TOOLS: core tools can't be disabled ({', '.join(core)}); skills "
                         "and planning need them")
        return found


# Shell commands that do what a disabled tool does: with that tool off, ``execute``
# refuses them too, so the agent can't do through the shell what was turned off.
SHELL_EQUIVALENTS = {
    "delete": (r"(?:^|[\s;&|(`])(?:sudo\s+)?(?:rm|rmdir|unlink|shred)(?=\s|$)",
               r"\bfind\b.*\s-(?:delete\b|exec\s+rm\b)",
               r"\bgit\s+(?:rm|clean)\b", r"\b(?:os\.(?:remove|unlink|rmdir)|shutil\.rmtree)\b",
               r"\.unlink\(", r"\bmv\b.*\s/dev/null\b"),
}
_SHELL_EQUIVALENTS = {tool: [re.compile(p) for p in patterns]
                      for tool, patterns in SHELL_EQUIVALENTS.items()}


def shell_block(command: str, disabled) -> str | None:
    """The disabled tool a shell command stands in for, or None."""
    for tool in sorted(set(disabled) & set(_SHELL_EQUIVALENTS)):
        if any(pattern.search(command or "") for pattern in _SHELL_EQUIVALENTS[tool]):
            return tool
    return None



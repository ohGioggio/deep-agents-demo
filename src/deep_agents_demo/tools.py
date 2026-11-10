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
import posixpath
import re
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
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
    "delete": (r"(?:^|[\s;&|(`])(?:sudo\s+)?(?:rm|rmdir|unlink|shred)(?=\s|$)",r"\bfind\b.*\s-(?:delete\b|exec\s+rm\b)",
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


def _path(value) -> str:
    text = str(value or "").replace("\\", "/").strip()
    return posixpath.normpath("/" + text.lstrip("/")) if text else ""


class ToolAccess(AgentMiddleware):
    """Hide disabled tools and not-yet-needed skill-only tools; refuse calls to them.

    A skill-only tool appears once the conversation contains a ``read_file`` of a
    SKILL.md whose ``allowed-tools`` lists it. If summarisation drops that read, the
    model stops seeing the tool (a call already made still runs). The agent is also
    told which skills need a disabled tool, so it can say so instead of working
    around it.
    """

    def __init__(self, disabled=frozenset(), skill_only=frozenset()):
        super().__init__()
        self.disabled = frozenset(disabled)
        self.skill_only = frozenset(skill_only) - self.disabled

    @staticmethod
    def _skills(state) -> list:
        return list((state or {}).get("skills_metadata") or [])

    def unlocked(self, state, messages) -> frozenset:
        """Skill-only tools listed by a skill the agent has read in ``messages``."""
        if not self.skill_only:
            return frozenset()
        read = {_path(call.get("args", {}).get("file_path"))
                for message in messages if isinstance(message, AIMessage)
                for call in message.tool_calls if call.get("name") == "read_file"}
        listed = {tool for skill in self._skills(state) if _path(skill.get("path")) in read
                  for tool in skill.get("allowed_tools") or []}
        return frozenset(listed) & self.skill_only

    def hidden(self, state, messages) -> frozenset:
        return self.disabled | (self.skill_only - self.unlocked(state, messages))

    def _note(self, state) -> str:
        lines = []
        if self.skill_only:
            lines.append("Some tools appear only after you read the skill that lists them under "
                         "'Allowed tools': read that SKILL.md first.")
        if self.disabled:
            lines.append(f"Turned off in this setup: {', '.join(sorted(self.disabled))}. Don't try "
                         "to use them or work around them. If the user asks for something that "
                         "needs one, say which tool is off and that it can be turned back on by "
                         "removing it from DISABLED_TOOLS in the .env file.")
            blocked = [f"{skill['name']} (needs {', '.join(sorted(needs))})"
                       for skill in self._skills(state)
                       if (needs := set(skill.get("allowed_tools") or []) & self.disabled)]
            if blocked:
                lines.append("Skills that need a turned-off tool: " + "; ".join(blocked)
                             + ". If the user asks for one, say which tool is off.")
        return "\n\n## Tool availability\n" + "\n".join(lines) if lines else ""

    def _request(self, request):
        hidden = self.hidden(request.state, request.messages)
        tools = [tool for tool in request.tools if tool_name(tool) not in hidden]
        note = self._note(request.state)
        if not note:
            return request.override(tools=tools)
        system = request.system_message
        if system is None:
            message = SystemMessage(content=note.lstrip())
        elif isinstance(system.content, str):
            message = SystemMessage(content=system.content + note)
        else:
            message = SystemMessage(content=[*system.content, {"type": "text", "text": note}])
        return request.override(tools=tools, system_message=message)

    def refusal(self, request) -> ToolMessage | None:
        name = request.tool_call["name"]
        tool = shell_block(str(request.tool_call.get("args", {}).get("command", "")),
                           self.disabled) if name == "execute" else None
        if name in self.disabled:
            reason = (f"Error: {name} is turned off in this setup. Don't retry it or work around it; "
                      f"tell the user it can be turned back on in DISABLED_TOOLS (.env).")
        elif tool:
            reason = (f"Error: this command does what {tool} does, and {tool} is turned off in this "
                      "setup. Don't retry it another way; tell the user it can be turned back on in "
                      "DISABLED_TOOLS (.env).")
        elif name in self.skill_only and name not in self.unlocked(
                request.state, (request.state or {}).get("messages", [])):
            reason = (f"Error: {name} is only available after reading a skill that lists it under "
                      "'Allowed tools'. Read that skill's SKILL.md first.")
        else:
            return None
        return ToolMessage(content=reason, tool_call_id=request.tool_call.get("id") or "",
                           name=name, status="error")

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))

    def wrap_tool_call(self, request, handler):
        return self.refusal(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self.refusal(request) or await handler(request)

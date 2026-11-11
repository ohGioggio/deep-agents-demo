"""Agent files in Claude Code's format: YAML frontmatter, then the system prompt.

```markdown
---
name: critic
description: Finds weaknesses in a proposal. Use when a plan needs challenging.
tools: Read, Grep, Glob
model: inherit
---
You are a sharp, fair critic. ...
```

``parse_agent_md`` turns a file into an ``AgentDefinition`` plus notes for the
user: Claude Code tool names are translated to this agent's tools, Claude model
names fall back to the configured model, and fields with no equivalent here are
kept (for round-trip) but ignored. ``render_agent_md`` writes a file back.
"""
import re
from dataclasses import dataclass, field

import yaml

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
READ_ONLY = ("ls", "read_file", "grep", "glob", "write_todos")
# Claude Code tool -> this agent's tools.
CLAUDE_TOOLS = {
    "Read": ("read_file",), "Write": ("write_file",), "Edit": ("edit_file",),
    "MultiEdit": ("edit_file",), "Bash": ("execute",), "PowerShell": ("execute",),
    "Grep": ("grep",), "Glob": ("glob", "ls"), "LS": ("ls",), "TodoWrite": ("write_todos",),
    "WebFetch": ("web_fetch",), "WebSearch": ("web_search",),
}
SPAWN_TOOLS = ("Agent", "Task")              # starting subagents: no nesting here
CLAUDE_MODELS = ("sonnet", "opus", "haiku", "fable")
USED_FIELDS = ("name", "description", "tools", "disallowedTools", "model", "permissionMode",
               "maxTurns", "skills", "color")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    prompt: str                              # the system prompt (the file's body)
    tools: tuple | None = None               # this agent's tool names; None = all tools
    disallowed: tuple = ()
    model: str | None = None                 # None = the configured chat model
    skills: tuple = ()                       # skills pinned into its context at start
    max_turns: int | None = None             # lowers the per-run model-call cap
    read_only: bool = False                  # permissionMode: plan
    color: str | None = None
    source: str = ""                         # "built-in" or the file it came from
    frontmatter: dict = field(default_factory=dict, compare=False)  # as read, for round-trip


def _names(value) -> list[str]:
    """A tools/skills value: a comma- or space-separated string, or a YAML list."""
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[\s,]+", str(value))
    return [str(item).strip() for item in items if str(item).strip()]


def translate_tools(names, known) -> tuple[tuple, list[str]]:
    """Claude Code or own tool names -> (this agent's tool names, notes on the rest)."""
    resolved, notes = [], []
    for name in names:
        if name in known:
            resolved.append(name)
        elif name in SPAWN_TOOLS:
            notes.append(f"tool {name!r} dropped: subagents can't start other subagents here")
        elif name in CLAUDE_TOOLS:
            ours = [tool for tool in CLAUDE_TOOLS[name] if tool in known]
            if ours:
                resolved.extend(ours)
            else:
                notes.append(f"tool {name!r} ({', '.join(CLAUDE_TOOLS[name])}) isn't available here")
        else:
            notes.append(f"tool {name!r} isn't available here")
    return tuple(dict.fromkeys(resolved)), notes


def resolve_model(value) -> tuple[str | None, str | None]:
    """(model id, note): None means the configured chat model."""
    text = str(value or "").strip()
    if not text or text == "inherit":
        return None, None
    if text in CLAUDE_MODELS or text.startswith("claude-"):
        return None, f"model {text!r} is a Claude model; using the configured model"
    return text, None


def parse_agent_md(text: str, source: str, known_tools) -> tuple[AgentDefinition | None, list[str]]:
    """The agent a file defines (None if it can't be loaded) and notes for the user."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").lstrip("\n")
    match = _FRONTMATTER.match(text)
    if not match:
        return None, [f"{source}: skipped: no YAML frontmatter between --- lines"]
    try:
        front = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        return None, [f"{source}: skipped: invalid YAML frontmatter ({str(error).splitlines()[0]})"]
    if not isinstance(front, dict):
        return None, [f"{source}: skipped: the frontmatter is not a mapping"]
    name, description = front.get("name"), front.get("description")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        return None, [f"{source}: skipped: name {name!r} must be letters, digits, '-' or '_' "
                      "(up to 64)"]
    if not isinstance(description, str) or not description.strip():
        return None, [f"{source}: skipped: {name!r} needs a description"]

    notes = []
    tools = None
    if "tools" in front and not _names(front["tools"]):
        # A blank line reads like "not set"; giving the agent no tools would surprise.
        notes.append("'tools' is blank, so the agent gets all tools (as when it's left out)")
    elif "tools" in front:
        raw = _names(front["tools"])
        tools, tool_notes = translate_tools(raw, known_tools)
        notes += tool_notes
        if not tools:
            return None, [f"{source}: skipped: none of {name!r}'s tools exist here", *notes]
    disallowed, deny_notes = translate_tools(_names(front.get("disallowedTools")), known_tools)
    notes += [n for n in deny_notes if "dropped" not in n]
    raw_model = front.get("model")
    if raw_model is not None and not isinstance(raw_model, str):
        notes.append(f"model {raw_model!r} ignored: it must be text; using the configured model")
        raw_model = None
    model, model_note = resolve_model(raw_model)
    notes += [model_note] if model_note else []
    mode = str(front.get("permissionMode") or "default")
    if mode not in ("default", "manual", "plan"):
        notes.append(f"permissionMode {mode!r} ignored: approvals always apply here")
    max_turns = front.get("maxTurns")
    if max_turns is not None and (isinstance(max_turns, bool) or not isinstance(max_turns, int)
                                  or max_turns < 1):
        notes.append(f"maxTurns {max_turns!r} ignored: it must be a whole number of 1 or more")
        max_turns = None
    raw_skills = front.get("skills")
    if raw_skills is None or isinstance(raw_skills, str) or (
            isinstance(raw_skills, list) and all(isinstance(item, str) for item in raw_skills)):
        skills = tuple(_names(raw_skills))
    else:
        notes.append("skills ignored: list skill names, as text or a list")
        skills = ()
    unused = [key for key in front if key not in USED_FIELDS]
    if unused:
        notes.append(f"fields with no equivalent here, ignored: {', '.join(map(str, unused))}")

    agent = AgentDefinition(
        name=name, description=" ".join(description.split()), prompt=match.group(2).strip(),
        tools=tools, disallowed=disallowed, model=model, skills=skills,
        max_turns=max_turns, read_only=mode == "plan",
        color=str(front["color"]) if front.get("color") else None, source=source, frontmatter=front)
    return agent, [f"{source}: {note}" for note in notes]


def render_agent_md(frontmatter: dict, prompt: str) -> str:
    """An agent file in Claude Code's format."""
    return ("---\n" + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True, width=10**9)
            + "---\n\n" + prompt.strip() + "\n")

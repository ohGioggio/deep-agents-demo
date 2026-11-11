"""What one capability adds to the agent, and the approval settings they make."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    """What one capability contributes to the agent."""
    name: str
    prompt: str = ""                 # a section of the system prompt
    tools: tuple = ()                # tools for the main agent
    middleware: tuple = ()           # LangChain agent middleware
    lookups: tuple = ()              # read-only tools the researcher subagent also gets
    skill_dirs: tuple = ()           # extra skill folders, relative to the skills prefix
    memory_sources: tuple = ()       # files for Deep Agents' file-based memory
    approvals: tuple = ()            # tools that pause for approval: name, or (name, when)
    skill_only: tuple = ()           # tools hidden until a skill listing them is read
    needs_skill: tuple = ()          # (tool, skill): the skill must be read before the tool runs


def approval_configs(features) -> dict:
    """HumanInTheLoopMiddleware settings from every feature's ``approvals``.

    An entry is a tool name (always pause) or ``(name, when)``, where
    ``when(request)`` decides per call. A tool pauses always if any feature gates it
    unconditionally; otherwise it pauses when any of its conditions holds.
    """
    conditions = {}
    for feature in features:
        for entry in feature.approvals:
            name, when = (entry, None) if isinstance(entry, str) else entry
            conditions.setdefault(name, []).append(when)
    configs = {}
    for name, whens in sorted(conditions.items()):
        config = {"allowed_decisions": ["approve", "reject"]}
        if None not in whens:
            config["when"] = whens[0] if len(whens) == 1 else (
                lambda request, whens=tuple(whens): any(when(request) for when in whens))
        configs[name] = config
    return configs

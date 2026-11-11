"""The agents the main agent can hand work to, and where each comes from.

Sources, lowest to highest (a higher one wins on a name clash):

1. built-ins: ``reviewer`` and ``moderator`` here; ``general-purpose`` and
   ``researcher`` are built by ``build_agent`` and can be replaced the same way;
2. folders of Claude Code agent files: ``AGENT_DIRS`` in order, then
   ``demo_workspace/agents/``.

(Installed marketplace agents and the user's agents in SQLite come later.)
"""
from pathlib import Path

from deep_agents_demo.agent_files import parse_agent_md
from deep_agents_demo.models.agents import READ_ONLY, AgentDefinition

REVIEWER_PROMPT = (
    "You review one step of a larger task. You get the goal, the step, and the worker's "
    "result. Check the result against the step: is it done, is it correct, is anything "
    "missing or made up? Look at the files it names if that helps. Reply with a verdict "
    "on the first line, PASS or FAIL, then concrete notes: for FAIL, exactly what to fix. "
    "Don't fix it yourself.")
MODERATOR_PROMPT = (
    "You moderate a discussion between agents with different roles on one question. "
    "Decide who speaks next and what they should address, keep the discussion on the "
    "question, and stop it when they agree, when nothing new is being said, or when the "
    "user asks. Then write a summary: what was agreed, the open disagreements, and the "
    "next steps.")

BUILTIN_AGENTS = {
    "reviewer": AgentDefinition(
        name="reviewer", source="built-in", read_only=True, prompt=REVIEWER_PROMPT,
        description="Checks a worker's result against the step it was given and replies "
                    "PASS or FAIL with concrete notes. Read-only."),
    "moderator": AgentDefinition(
        name="moderator", source="built-in", tools=(), prompt=MODERATOR_PROMPT,
        description="Runs a discussion between agents: picks who speaks next, keeps it on "
                    "topic, and ends with a summary of agreements, disagreements and next steps."),
}


# Built by build_agent itself, but replaceable by name like the ones above.
BUILT_IN_ELSEWHERE = ("general-purpose", "researcher")


def load_folder(folder: Path, known_tools) -> tuple[dict, list[str]]:
    """Agents defined by the .md files under ``folder`` (README.md files are notes)."""
    if not folder.is_dir():
        return {}, [f"agent folder {folder} doesn't exist"]
    found, warnings = {}, []
    for path in sorted(folder.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            warnings.append(f"{path}: skipped: can't be read ({error})")
            continue
        agent, notes = parse_agent_md(text, str(path), known_tools)
        warnings += notes
        if agent is None:
            continue
        if agent.name in found:
            warnings.append(f"{path}: skipped: {found[agent.name].source} already defines "
                            f"{agent.name!r}")
            continue
        found[agent.name] = agent
    return found, warnings


class AgentRegistry:
    """Every agent the main agent can delegate to, after precedence."""

    def __init__(self, agents: dict, warnings: list, overrides: dict | None = None):
        self._agents, self.warnings = dict(agents), list(warnings)
        self._overrides = dict(overrides or {})  # name -> source it replaced

    @classmethod
    def load(cls, folders, known_tools, installed=(), own=(),
             replaceable=BUILT_IN_ELSEWHERE) -> "AgentRegistry":
        """Built-ins, then ``folders`` in order, then ``installed`` marketplace agents, then
        the user's ``own`` agents from the database: a later source wins on a name clash."""
        agents, warnings, overrides = dict(BUILTIN_AGENTS), [], {}
        sources = [load_folder(Path(folder), known_tools) for folder in folders]
        sources += [({a.name: a for a in installed}, []), ({a.name: a for a in own}, [])]
        for found, notes in sources:
            warnings += notes
            for name, agent in found.items():
                if name in agents:
                    replaced = agents[name].source
                    overrides[name] = replaced
                    if replaced != "built-in":
                        warnings.append(f"{agent.source}: {name!r} overrides {replaced}")
                elif name in replaceable:
                    overrides[name] = "built-in"
                agents[name] = agent
        return cls(agents, warnings, overrides)

    def agents(self) -> list[AgentDefinition]:
        return [self._agents[name] for name in sorted(self._agents)]

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def overview(self) -> list[dict]:
        rows = []
        for agent in self.agents():
            tools = READ_ONLY if agent.read_only else agent.tools
            rows.append({"name": agent.name, "source": agent.source,
                         "tools": "all" if tools is None else (", ".join(tools) or "none"),
                         "model": agent.model or "configured",
                         "overrides": self._overrides.get(agent.name, "")})
        return rows



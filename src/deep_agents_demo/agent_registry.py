"""The agents the main agent can hand work to, and where each comes from.

Sources, lowest to highest (a higher one wins on a name clash):

1. built-ins: ``reviewer`` and ``moderator`` here; ``general-purpose`` and
   ``researcher`` are built by ``build_agent`` and can be replaced the same way;
2. folders of Claude Code agent files: ``AGENT_DIRS`` in order, then
   ``demo_workspace/agents/``.

(Installed marketplace agents and the user's agents in SQLite come later.)
"""
from pathlib import Path
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage

from deep_agents_demo.agent_files import READ_ONLY, AgentDefinition, parse_agent_md

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
    def load(cls, folders, known_tools) -> "AgentRegistry":
        agents, warnings, overrides = dict(BUILTIN_AGENTS), [], {}
        for folder in folders:
            found, notes = load_folder(Path(folder), known_tools)
            warnings += notes
            for name, agent in found.items():
                if name in agents:
                    replaced = agents[name].source
                    overrides[name] = replaced
                    if replaced != "built-in":
                        warnings.append(f"{agent.source}: {name!r} overrides {replaced}")
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


class CallCapState(AgentState):
    # Checkpointed (unlike ModelCallLimitMiddleware's count), so approve-and-resume
    # doesn't start it again; private, so it never leaks into the main agent's state.
    subagent_model_calls: Annotated[NotRequired[int], PrivateStateAttr]


def partial_result(messages, calls: int) -> str:
    """What a subagent cut off by the call cap hands back: what it had, marked partial."""
    texts = [m.text if isinstance(m.text, str) else str(m.content) for m in messages
             if isinstance(m, AIMessage) and str(m.content).strip()]
    tools = sum(1 for m in messages if isinstance(m, ToolMessage))
    found = texts[-1].strip() if texts else f"No answer yet ({tools} tool calls made)."
    return (f"[Partial result: stopped after {calls} model calls, the limit for one subagent "
            f"run.]\n\n{found}\n\n[Don't hand the same task over again unchanged; tell the user "
            "it was cut short, or split it into smaller tasks.]")


class SubagentCallCap(AgentMiddleware):
    """At most ``limit`` model calls per subagent run; then end with a partial result."""

    state_schema = CallCapState

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        calls = state.get("subagent_model_calls", 0)
        if calls >= self.limit:
            return {"messages": [AIMessage(content=partial_result(state["messages"], calls))],
                    "jump_to": "end"}
        return {"subagent_model_calls": calls + 1}


class PinSkills(AgentMiddleware):
    """Pin skills into a subagent's context when it starts (its file's ``skills``)."""

    def __init__(self, names):
        super().__init__()
        self.names = tuple(names)

    def before_agent(self, state, runtime):
        return {"pinned_skills": list(self.names)} if self.names else None

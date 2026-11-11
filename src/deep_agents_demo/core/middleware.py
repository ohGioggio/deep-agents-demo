"""Agent middleware shared by the engine: tool access, subagent caps, pinned
skills and the parallel cap on ``task``."""
import posixpath
import threading
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from deep_agents_demo.core.tools import shell_block, tool_name


def _path(value) -> str:
    text = str(value or "").replace("\\", "/").strip()
    return posixpath.normpath("/" + text.lstrip("/")) if text else ""


class ToolAccess(AgentMiddleware):
    """Hide disabled tools and not-yet-needed skill-only tools; refuse calls to them.

    A skill-only tool appears once the conversation contains a ``read_file`` of a
    SKILL.md whose ``allowed-tools`` lists it. If summarisation drops that read, the
    model stops seeing the tool (a call already made still runs). The agent is also
    told which skills need a disabled tool, so it can say so instead of working
    around it. A tool with a required skill (``Feature.needs_skill``) is refused
    until the agent has read that skill, and the refusal says which file to read.
    """

    def __init__(self, disabled=frozenset(), skill_only=frozenset(), requires=None,
                 allowed=None, denied=frozenset()):
        super().__init__()
        self.disabled = frozenset(disabled)
        # One agent's own limits: only ``allowed`` (None = no limit), never ``denied``.
        self.allowed = None if allowed is None else frozenset(allowed)
        self.denied = frozenset(denied)
        # A skill-only tool an agent lists explicitly is its job: visible from the start.
        self.skill_only = frozenset(skill_only) - self.disabled - (self.allowed or frozenset())
        self.requires = dict(requires or {})  # tool -> skill to read before its first use

    def _not_for_this_agent(self, name) -> bool:
        return name in self.denied or (self.allowed is not None and name not in self.allowed)

    @staticmethod
    def _skills(state) -> list:
        return list((state or {}).get("skills_metadata") or [])

    @staticmethod
    def _read(messages) -> set:
        """Paths the agent has called read_file on in ``messages``."""
        return {_path(call.get("args", {}).get("file_path"))
                for message in messages if isinstance(message, AIMessage)
                for call in message.tool_calls if call.get("name") == "read_file"}

    def missing_skill(self, name, state) -> dict | None:
        """The skill ``name`` needs that the agent hasn't read yet (None if loaded and read,
        or if that skill isn't loaded at all)."""
        wanted = self.requires.get(name)
        skill = next((s for s in self._skills(state) if s.get("name") == wanted), None) if wanted else None
        if skill is None or _path(skill.get("path")) in self._read((state or {}).get("messages", [])):
            return None
        return skill

    def unlocked(self, state, messages) -> frozenset:
        """Skill-only tools listed by a skill the agent has read in ``messages``."""
        if not self.skill_only:
            return frozenset()
        read = self._read(messages)
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
        tools = [tool for tool in request.tools
                 if tool_name(tool) not in hidden and not self._not_for_this_agent(tool_name(tool))]
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
        elif self._not_for_this_agent(name):
            reason = f"Error: {name} isn't available to this agent."
        elif tool:
            reason = (f"Error: this command does what {tool} does, and {tool} is turned off in this "
                      "setup. Don't retry it another way; tell the user it can be turned back on in "
                      "DISABLED_TOOLS (.env).")
        elif name in self.skill_only and name not in self.unlocked(
                request.state, (request.state or {}).get("messages", [])):
            reason = (f"Error: {name} is only available after reading a skill that lists it under "
                      "'Allowed tools'. Read that skill's SKILL.md first.")
        elif skill := self.missing_skill(name, request.state):
            reason = (f"Error: before using {name}, read the {skill['name']} skill: read_file "
                      f"{skill['path']}. It explains how this environment works. Then retry.")
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


class TaskLimiter(AgentMiddleware):
    """At most ``limit`` ``task`` calls run at the same time (LangGraph runs a step's
    tool calls in parallel threads); the rest wait for a free slot."""

    def __init__(self, limit: int):
        super().__init__()
        self.slots = threading.BoundedSemaphore(limit)

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "task":
            return handler(request)
        with self.slots:
            return handler(request)

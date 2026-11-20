"""``task`` for an agent made earlier in the same message.

Deep Agents fixes the main agent's subagents when it is built, so an agent approved
since then is unknown to ``task`` until the next message. This middleware runs such a
call itself, from the newest worker factory (the same limits, tools and rules a
subagent gets), and returns the agent's last answer as the tool result. Every other
call goes on to ``task`` unchanged, including unknown names (Deep Agents' error).

A pause of the delegated agent (an approval) is raised again here with ``interrupt``,
so the parent turn asks the user exactly as it asks for a subagent's pause; the answer
resumes the delegated run. When the parent resumes this tool call, the delegated run is
found paused on its thread and is resumed, not started again.
"""
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt

NO_ANSWER = "The agent gave no answer."


def _answer(messages) -> str:
    for message in reversed(messages):
        text = message.text if isinstance(message, AIMessage) else ""
        if isinstance(text, str) and text.strip():
            return text.strip()
    return NO_ANSWER


def _pending(snapshot) -> list:
    """The paused items of a delegated run (its approval requests)."""
    return [item for task in snapshot.tasks for item in task.interrupts]


def _start(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)]}


def _task_text(call) -> str:
    return str((call.get("args") or {}).get("description", ""))


# Tool call id -> the parent's answers already applied to that call, in order. LangGraph gives
# a re-run of the tool node its earlier answers by position, so the answers are replayed first
# (one interrupt() each), then the current pause is raised. Kept in memory only. After a process
# restart the entry is gone while the delegated run is still paused on its checkpoint, so the
# re-run is refused (see _restarted()) rather than guessing which stored answer goes with which pause.
_APPLIED: dict[str, list] = {}
NO_DECISION = {"type": "reject", "message": "No answer was given for this action; it was not run."}


def _restarted(name: str) -> str:
    return (f"The delegated run of {name} was interrupted by a restart while waiting for your "
            "approval; nothing more was done. Ask again to run it.")


def _merged(pending, task_call_id: str) -> dict:
    """One parent pause for all the delegated run's pending requests, named by the task call
    that started the run (the parent shows them on that call)."""
    return {"action_requests": [r for item in pending for r in item.value.get("action_requests", [])],
            "task_call_id": task_call_id}


def _split(pending, response) -> dict:
    """The parent's decisions, cut back into each pending item's share, in order. A request
    without a decision is rejected (never run on a missing answer)."""
    given = response.get("decisions", []) if isinstance(response, dict) else []
    shares, at = {}, 0
    for item in pending:
        count = len(item.value.get("action_requests", []))
        shares[item.id] = {"decisions": [given[at + k] if at + k < len(given) else NO_DECISION
                                         for k in range(count)]}
        at += count
    return shares


class NewAgentDelegation(AgentMiddleware):
    """``built``: the subagent names the main agent was built with. ``newest``: () -> the
    newest worker factory (``make(name, extra_middleware)``, ``make.names``), or None."""

    def __init__(self, built, newest: Callable):
        super().__init__()
        self.built = frozenset(built)
        self.newest = newest

    def _target(self, request):
        """(factory, name) when the call names an agent made since the build, else None."""
        call = request.tool_call
        if call.get("name") != "task":
            return None
        name = (call.get("args") or {}).get("subagent_type")
        if not isinstance(name, str) or name in self.built:
            return None
        make = self.newest()
        if make is None or name not in getattr(make, "names", ()):
            return None
        return make, name

    @staticmethod
    def _config(call) -> dict:
        return {"configurable": {"thread_id": f"task-{call.get('id') or ''}"}}

    @staticmethod
    def _message(request, text: str, status: str = "success") -> ToolMessage:
        return ToolMessage(content=text, tool_call_id=request.tool_call.get("id") or "", name="task",
                           status=status)

    def _missing_description(self, request):
        """The task tool's own check, which this path skips: no job, no run. Seen live, a model
        put the job under another key, and the agent got an empty request."""
        if _task_text(request.tool_call).strip():
            return None
        return self._message(request, "Error: the task tool needs `description` (the full task for the "
                             "agent, with every path and detail it needs) and `subagent_type`. Call it again "
                             "with both.", status="error")

    def _take(self, request, name, started: bool):
        """Take the call's entry before it runs. A call that already has a started run but no
        entry (a restart) gets the refusal instead; otherwise None and the call may run."""
        key = request.tool_call.get("id") or ""
        if key in _APPLIED:
            return None
        if started:
            return self._message(request, _restarted(name))
        _APPLIED[key] = []
        return None

    def _claim(self, agent, request, name):
        """The sync path's entry (the sync saver's state)."""
        if (request.tool_call.get("id") or "") in _APPLIED:
            return None
        return self._take(request, name, bool(agent.get_state(self._config(request.tool_call)).values.get("messages")))

    async def _aclaim(self, agent, request, name):
        """The async path's entry: an async-only saver (AsyncSqliteSaver) reads state only async."""
        if (request.tool_call.get("id") or "") in _APPLIED:
            return None
        state = await agent.aget_state(self._config(request.tool_call))
        return self._take(request, name, bool(state.values.get("messages")))

    def wrap_tool_call(self, request, handler):
        target = self._target(request)
        if target is None:
            return handler(request)
        make, name = target
        missing = self._missing_description(request)
        if missing is not None:
            return missing
        agent = make(name)
        refusal = self._claim(agent, request, name)
        if refusal is not None:
            return refusal
        key, kept = request.tool_call.get("id") or "", False
        try:
            call, config = request.tool_call, self._config(request.tool_call)
            applied = _APPLIED[key]
            for _ in applied:  # a re-run: the answers already applied, replayed by position
                interrupt(None)
            if not agent.get_state(config).values.get("messages"):
                agent.invoke(_start(_task_text(call)), config)
            while pending := _pending(agent.get_state(config)):
                response = interrupt(_merged(pending, key))  # the parent asks the user, once per pause
                applied.append(response)
                agent.invoke(Command(resume=_split(pending, response)), config)
            return self._message(request, _answer(agent.get_state(config).values["messages"]))
        except GraphInterrupt:
            kept = True  # the parent's pause: the entry stays for the re-run
            raise
        finally:
            if not kept:
                _APPLIED.pop(key, None)

    async def awrap_tool_call(self, request, handler):
        target = self._target(request)
        if target is None:
            return await handler(request)
        make, name = target
        missing = self._missing_description(request)
        if missing is not None:
            return missing
        agent = make(name)
        refusal = await self._aclaim(agent, request, name)
        if refusal is not None:
            return refusal
        key, kept = request.tool_call.get("id") or "", False
        try:
            call, config = request.tool_call, self._config(request.tool_call)
            applied = _APPLIED[key]
            for _ in applied:  # a re-run: the answers already applied, replayed by position
                interrupt(None)
            if not (await agent.aget_state(config)).values.get("messages"):
                await agent.ainvoke(_start(_task_text(call)), config)
            while pending := _pending(await agent.aget_state(config)):
                response = interrupt(_merged(pending, key))  # the parent asks the user, once per pause
                applied.append(response)
                await agent.ainvoke(Command(resume=_split(pending, response)), config)
            return self._message(request, _answer((await agent.aget_state(config)).values["messages"]))
        except GraphInterrupt:
            kept = True  # the parent's pause: the entry stays for the re-run
            raise
        finally:
            if not kept:
                _APPLIED.pop(key, None)

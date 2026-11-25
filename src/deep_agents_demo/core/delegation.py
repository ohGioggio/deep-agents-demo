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
import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt

NO_ANSWER = "The agent gave no answer."

# A task for an agent created in the same reply waits this long (seconds) for the create to
# save it, polling every NEW_AGENT_POLL seconds. Read at call time, so tests can patch them.
NEW_AGENT_WAIT = 10.0
NEW_AGENT_POLL = 0.1


def _creating_calls(state, name: str) -> list[str]:
    """The ids of the reply's sibling calls that make an agent called ``name``: create_agent (not
    into the marketplace, which is not usable until installed), install_agent with that name, or an
    import_agent whose file is named ``name``. The reply is the last AIMessage; a missing or empty
    state has no sibling."""
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    last = next((m for m in reversed(messages or ()) if isinstance(m, AIMessage)), None)
    found = []
    for call in getattr(last, "tool_calls", None) or ():
        tool, args = call.get("name"), call.get("args") or {}
        if args.get("scope") == "marketplace":
            continue
        path = args.get("path")
        if ((tool in ("create_agent", "install_agent") and args.get("name") == name)
                or (tool == "import_agent" and isinstance(path, str) and Path(path).stem == name)):
            found.append(call.get("id") or "")
    return found


def _answered_ids(state) -> set[str]:
    """The ids of the calls the last reply already has a ToolMessage for (after that reply): a call
    the user rejected is answered by the approval layer before the tools run."""
    messages = list(state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None) or ())
    at = max((i for i, m in enumerate(messages) if isinstance(m, AIMessage)), default=-1)
    return {m.tool_call_id for m in messages[at + 1:] if isinstance(m, ToolMessage)}


def _open_creating(state, name: str) -> list[str]:
    """The creating calls (see _creating_calls) that have no answer yet: those still to be waited on."""
    answered = _answered_ids(state)
    return [call_id for call_id in _creating_calls(state, name) if call_id not in answered]


def _not_approved(name: str) -> str:
    return (f"Error: creating {name} was not approved, so it doesn't exist; don't call task for it "
            "unless the user asks again.")


def _saved_not_reloaded(name: str) -> str:
    return (f"Error: {name} was saved, but the agents couldn't be reloaded in this step; "
            "it can be used from the user's next message.")


def _not_ready(name: str) -> str:
    return (f"Error: {name} was being created in this same step but isn't available yet; "
            "call list_agents, then task.")


def _create_failed(name: str) -> str:
    return (f"Error: creating {name} failed in this same step; see that tool's result, "
            "then call list_agents.")


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
    newest worker factory (``make(name, extra_middleware)``, ``make.names``), or None.
    ``settled(call_id)``: True when that tool call saved its change, False when it did not (refused
    or raised), None while it still runs. A wait for an agent the same reply creates ends, after
    one last lookup, once every creating call has settled (None: wait out NEW_AGENT_WAIT)."""

    def __init__(self, built, newest: Callable, settled: Callable[[str], bool | None] | None = None):
        super().__init__()
        self.built = frozenset(built)
        self.newest = newest
        self.settled = settled

    def _delegated_name(self, request):
        """The subagent_type of a task call for an agent not built with the main agent, else None."""
        call = request.tool_call
        if call.get("name") != "task":
            return None
        name = (call.get("args") or {}).get("subagent_type")
        if not isinstance(name, str) or name in self.built:
            return None
        return name

    def _lookup(self, name):
        """The newest factory if it knows ``name``, else None."""
        make = self.newest()
        return make if make is not None and name in getattr(make, "names", ()) else None

    def _first_look(self, request):
        """None when ``task`` runs the call. Else (name, make, creating): make is the newest factory
        when it knows the agent now, or None when the agent is still to be created in this reply
        (``creating``: the ids of the calls that make it)."""
        name = self._delegated_name(request)
        if name is None:
            return None
        make = self._lookup(name)
        state = getattr(request, "state", None)
        creating = [] if make is not None else _open_creating(state, name)
        if make is None and not creating and not _creating_calls(state, name):
            return None
        return name, make, creating  # no open call and no agent: every create was answered (rejected)

    def _all_settled(self, creating) -> bool:
        return self.settled is not None and all(self.settled(call_id) is not None for call_id in creating)

    def _poll(self, name, creating, deadline):
        """One look while waiting: (done, make). done once the agent is found, or every creating call
        has settled (after one last lookup) or the deadline passed."""
        make = self._lookup(name)
        if make is not None:
            return True, make
        if self._all_settled(creating):
            return True, self._lookup(name)
        return time.monotonic() >= deadline, None

    def _target(self, request):
        """(factory, name) when the call names an agent made since the build, else None. When the same
        reply creates that agent, waits for it (up to NEW_AGENT_WAIT, or until every creating call
        has settled); a wait that ends without it returns (None, name), which the wrappers answer
        with an error (see _unready)."""
        first = self._first_look(request)
        if first is None:
            return None
        name, make, creating = first
        deadline = time.monotonic() + NEW_AGENT_WAIT
        while make is None and creating:
            done, make = self._poll(name, creating, deadline)
            if done:
                break
            time.sleep(NEW_AGENT_POLL)
        return (make, name)

    async def _atarget(self, request):
        """The async twin of _target: waits with asyncio.sleep, so the event loop keeps running."""
        first = self._first_look(request)
        if first is None:
            return None
        name, make, creating = first
        deadline = time.monotonic() + NEW_AGENT_WAIT
        while make is None and creating:
            done, make = self._poll(name, creating, deadline)
            if done:
                break
            await asyncio.sleep(NEW_AGENT_POLL)
        return (make, name)

    def _unready(self, request, name: str) -> str:
        """The error text for a wait that ended without the agent: every create was rejected, the create
        failed, it saved but the rebuild did not (all settled True), or the agent is merely not ready."""
        state = getattr(request, "state", None)
        creating = _open_creating(state, name)
        if not creating:
            return _not_approved(name)
        if not self._all_settled(creating):
            return _not_ready(name)
        if all(self.settled(call_id) for call_id in creating):
            return _saved_not_reloaded(name)
        return _create_failed(name)

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
        if make is None:
            return self._message(request, self._unready(request, name), status="error")
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
        target = await self._atarget(request)
        if target is None:
            return await handler(request)
        make, name = target
        if make is None:
            return self._message(request, self._unready(request, name), status="error")
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

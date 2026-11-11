"""The turn engine: run one user turn to the end, emitting events instead of printing.

A turn streams the agent's updates, turns every new message into a
``MessageEvent`` (and to-do changes into ``TodosEvent``), and answers the pauses it
meets: approvals through ``decide``, and other pauses (``run_plan``,
``start_room``) through ``runners`` by their ``type``, on this thread.
"""
from langgraph.types import Command

from deep_agents_demo.core.messages import has_image, without_images
from deep_agents_demo.models.approvals import NO_HUMAN
from deep_agents_demo.models.events import FinalAnswerEvent, MessageEvent, TodosEvent


def turn_input(prompt, reload_skills=False):
    """Agent input for one user turn.

    Deep Agents saves the skill list in the thread's checkpoint and loads it only
    once per thread. ``skills_metadata=None`` asks it to reload, so skills added
    or changed since the thread began (or a new sandbox route) are picked up.
    """
    payload = {"messages": [{"role": "user", "content": prompt}]}
    if reload_skills:
        payload["skills_metadata"] = None
    return payload


def printable(message):
    """The message as shown: images become a short note instead of base64 text."""
    if has_image(message):
        return without_images(message, "[image]")
    return message


def message_event(label: str, node: str, message) -> MessageEvent:
    shown = printable(message)
    text = shown.text if isinstance(shown.text, str) else str(shown.content)
    calls = tuple({"name": c.get("name"), "args": c.get("args", {}), "id": c.get("id")}
                  for c in getattr(message, "tool_calls", None) or ())
    return MessageEvent(label, node, message.type, text, calls, shown.pretty_repr())


class _Silent:
    def emit(self, event) -> None:
        return None


def stream_once(agent, payload, config, sink, shown=None, label=None) -> list:
    """Stream one run of the agent loop; return the approval requests it paused on.

    ``shown`` collects the ids of emitted messages: after a pause, LangGraph sends
    the paused AI message again, and it should be shown once per turn. ``label``
    names who is speaking (e.g. "room / critic"; default "main").
    """
    paused, shown = {}, shown if shown is not None else set()
    silent = isinstance(sink, _Silent)  # build nothing nobody will see
    # Include child updates so delegation/tool calls are observable.
    for namespace, update in agent.stream(payload, config=config, stream_mode="updates",
                                          subgraphs=True):
        for node, state in update.items():
            if node == "__interrupt__":
                # A subagent's pause can surface at several levels; keep each once.
                paused.update({item.id: item for item in state})
                continue
            if not isinstance(state, dict):
                continue
            who = label or "main"
            who = (f"{who} / subagent" if label else "subagent") if namespace else who
            for message in state.get("messages", []):
                key = message.id or id(message)
                if key in shown:
                    continue
                shown.add(key)
                if not silent:
                    sink.emit(message_event(who, node, message))
            if "todos" in state and not silent:
                sink.emit(TodosEvent(tuple(state["todos"])))
    return list(paused.values())


def final_answer(agent, config) -> str:
    """The text of the last AI message in a thread (what a one-shot run shows)."""
    for message in reversed(agent.get_state(config).values.get("messages", [])):
        if message.type == "ai" and not getattr(message, "tool_calls", None) and str(message.content).strip():
            return message.text if isinstance(message.text, str) else str(message.content)
    return ""


def _nobody(request) -> dict:
    """With nobody to ask, every paused call is rejected."""
    return {"decisions": [{"type": "reject", "message": NO_HUMAN}
                          for _ in request.get("action_requests", [])]}


def run_turn(agent, prompt, config, sink, decide=None, runners=None, reload_skills=True, label=None,
             quiet=False, show=None) -> None:
    """Run one user turn to the end.

    ``decide(request) -> dict`` answers approval pauses (None: reject them all).
    ``runners`` answer other pauses by their ``type`` (e.g. ``run_plan``): they run
    here, on the caller's thread, and their return value resumes the tool.
    ``quiet`` emits only the final answer; ``show=False`` emits nothing at all.
    """
    decide, runners = decide or _nobody, runners or {}
    payload, shown = turn_input(prompt, reload_skills), set()

    def answer(value):
        kind = value.get("type") if isinstance(value, dict) else None
        if kind in runners:
            return runners[kind](value)
        if kind == "run_plan":
            return {"error": "plans can't run here (nobody to run them); do the steps yourself"}
        if kind == "start_room":
            return {"error": "rooms can't run here (nobody to run them)"}
        return decide(value)

    show = not quiet if show is None else show
    while True:
        paused = stream_once(agent, payload, config, sink if show else _Silent(), shown, label=label)
        if not paused:
            if quiet:
                sink.emit(FinalAnswerEvent(final_answer(agent, config)))
            return
        payload = Command(resume={item.id: answer(item.value) for item in paused})

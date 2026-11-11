"""The interactive chat: the input loop, slash commands and streamed output."""
import json
from contextlib import ExitStack

from langgraph.types import Command

from deep_agents_demo.app import build_runtime, prepare_workspace
from deep_agents_demo.approvals import Approver
from deep_agents_demo.config import Settings
from deep_agents_demo.images import ImageError, image_paths, message_content, without_images
from deep_agents_demo.threads import Chats, format_table, format_transcript, new_name, thread_id, valid_name

GREETING = ("Ask anything. Ideas: a question about your indexed documents, "
            "'Remember that I prefer short answers', or the example skill: "
            "'Plan a city scene under 100000 triangles'.\n"
            "Type /help for chat commands, exit to quit.")


def confirm(question: str, ask=input) -> bool:
    try:
        return ask(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


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


def stream_once(agent, payload, config, shown=None) -> list:
    """Stream one run of the agent loop; return the approval requests it paused on.

    ``shown`` collects the ids of printed messages: after a pause, LangGraph sends
    the paused AI message again, and it should be printed once per turn.
    """
    paused, shown = {}, shown if shown is not None else set()
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
            label = "subagent" if namespace else "main"
            for message in state.get("messages", []):
                key = message.id or id(message)
                if key in shown:
                    continue
                shown.add(key)
                print(f"\n[{label} / {node}]")
                printable(message).pretty_print()
            if "todos" in state:
                print("[plan]", json.dumps(state["todos"], ensure_ascii=False))
    return list(paused.values())


def printable(message):
    """The message as printed: images become a short note instead of base64 text."""
    if isinstance(message.content, list) and any(
            isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in message.content):
        return without_images(message, "[image]")
    return message


def stream_turn(agent, prompt, config, reload_skills=True, approver=None) -> None:
    """Run one user turn to the end, asking ``approver`` about every paused tool call.

    Without an approver, paused calls are rejected (nobody can approve them).
    """
    approver = approver or Approver(interactive=False)
    payload, shown = turn_input(prompt, reload_skills), set()
    while True:
        paused = stream_once(agent, payload, config, shown)
        if not paused:
            return
        payload = Command(resume={item.id: approver.decide(item.value) for item in paused})


def command(*names, usage, summary):
    """Register a ChatSession method as a slash command: /name, any aliases, and its help line."""
    def register(method):
        method.command = (names, usage, summary)
        return method
    return register


class ChatSession:
    """Tracks the current chat and handles slash commands typed in the chat loop.

    Each command is one method decorated with ``@command``; the lookup table and
    the /help text are built from those decorators, so adding a command is
    adding a method.
    """

    def __init__(self, chats: Chats, current: str, ask=input, say=print, uploads=None, agents=None):
        self.chats, self.current, self.ask, self.say = chats, current, ask, say
        self.uploads, self.agents = uploads, agents
        self.pending = []  # images attached with /image, sent with the next message
        self.reload_skills = True  # reload once per chat, on its first turn this session

    @property
    def config(self):
        return {"configurable": {"thread_id": thread_id(self.chats.user, self.current)},
                "recursion_limit": 80}

    def _enter(self, name, note):
        self.current, self.reload_skills = name, True
        self.say(note)

    def handle(self, line: str) -> bool:
        """Run a slash command. Returns False if ``line`` is not a command."""
        if not line.startswith("/"):
            return False
        name, _, arg = line[1:].partition(" ")
        method = COMMANDS.get(name.lower())
        if method is None:
            self.say(f"Unknown command /{name.lower()}. Type /help.")
        else:
            method(self, arg.strip())
        return True

    def attach(self, path) -> bool:
        """Attach one image for the next message; say why if it can't be."""
        if self.uploads is None:
            self.say("Images can't be attached here.")
            return False
        try:
            attachment = self.uploads.attach(path)
        except ImageError as error:
            self.say(f"Not attached: {error}.")
            return False
        self.pending.append(attachment)
        self.say(f"Attached {attachment.describe()}.")
        return True

    def content(self, line: str):
        """The message to send: the text, plus images attached with /image or named in it."""
        for path in image_paths(line):
            self.attach(path)
        attachments, self.pending = self.pending, []
        return message_content(line, attachments)

    @command("image", usage="/image PATH", summary="attach a PNG or JPG to your next message")
    def image(self, arg):
        path = arg.strip().strip('"').strip("'")
        if not path:
            self.say("Usage: /image PATH (or just write the image's path in your message)")
        elif self.attach(path):
            self.say("It will be sent with your next message.")

    @command("agents", usage="/agents", summary="list the subagents the agent can hand work to")
    def list_agents(self, arg):
        if self.agents is None:
            self.say("No agents here.")
            return
        rows = self.agents.overview()
        width = max(len(r["name"]) for r in rows)
        lines = [f"  {'NAME':<{width}}  TOOLS / MODEL / SOURCE"]
        for r in rows:
            note = f" (overrides {r['overrides']})" if r["overrides"] else ""
            lines.append(f"  {r['name']:<{width}}  tools: {r['tools']}; model: {r['model']}; "
                         f"source: {r['source']}{note}")
        lines.append("  general-purpose (and researcher, when documents or the catalog are on) are "
                     "built in unless an agent replaces them.")
        self.say("Subagents:\n" + "\n".join(lines))

    @command("help", usage="/help", summary="show this help")
    def help(self, arg):
        self.say(HELP)

    @command("threads", "chats", "list", usage="/threads", summary="list your chats (* marks the current one)")
    def threads(self, arg):
        table = format_table(self.chats.list(), self.current)
        if not self.chats.exists(self.current):
            table += f"\n*  {self.current} (new: saved after your first message)"
        self.say(table)

    @command("new", usage="/new [name]", summary="start a new chat (a dated name if you give none)")
    def new(self, arg):
        name = arg or new_name()
        if not valid_name(name):
            self.say(f"Invalid name {name!r}: use letters, digits, '.', '_' or '-'.")
        elif self.chats.exists(name):
            self.say(f"A chat named {name!r} already exists; use /switch {name}.")
        else:
            self._enter(name, f"New chat {name!r}. It is saved once you send a message.")

    @command("switch", usage="/switch NAME", summary="continue another chat")
    def switch(self, arg):
        if not arg:
            self.say("Usage: /switch NAME (see /threads)")
        elif not self.chats.exists(arg):
            self.say(f"No chat named {arg!r}. See /threads, or start one with /new {arg}.")
        else:
            self._enter(arg, f"Switched to {arg!r} ({self.chats.info(arg).turns} turns).")

    @command("history", usage="/history", summary="show this chat's messages")
    def history(self, arg):
        if self.chats.exists(self.current):
            self.say(format_transcript(self.chats.transcript(self.current)))
        else:
            self.say("(empty chat)")

    @command("rename", usage="/rename NAME", summary="rename this chat")
    def rename(self, arg):
        try:
            self.chats.rename(self.current, arg)
        except ValueError as error:
            self.say(str(error))
        else:
            self.current = arg
            self.say(f"Renamed to {arg!r}.")

    @command("delete", usage="/delete NAME", summary="delete a chat and its history")
    def delete(self, arg):
        if not arg or not self.chats.exists(arg):
            self.say(f"No chat named {arg!r}. See /threads.")
        elif confirm(f"Delete chat {arg!r} ({self.chats.info(arg).turns} turns)?", self.ask):
            self.chats.delete(arg)
            current = arg == self.current
            self.say(f"Deleted {arg!r}." + (" You are still in it, now empty." if current else ""))
            if current:
                self.reload_skills = True
        else:
            self.say("Kept.")


# Built from the @command decorators: name or alias -> method, and the /help text.
_REGISTERED = [m for m in vars(ChatSession).values() if hasattr(m, "command")]
COMMANDS = {name: m for m in _REGISTERED for name in m.command[0]}
HELP = "Chat commands:\n" + "\n".join(
    f"  {usage:<15} {summary}" for _names, usage, summary in
    sorted((m.command for m in _REGISTERED), key=lambda c: c[1] == "/help")
) + f"\n  {'exit':<15} quit"


def run_chat(settings: Settings, user: str, thread: str, prompt: str | None = None) -> None:
    """Interactive chat, or a single ``prompt`` that prints only the answer."""
    # Startup chatter only helps in an interactive chat; --prompt prints just the answer.
    say = print if prompt is None else (lambda *_args, **_kwargs: None)
    prepare_workspace(settings)
    say(GREETING)
    with ExitStack() as stack:
        runtime = build_runtime(settings, user, stack, say)
        session = ChatSession(runtime.chats, thread, uploads=runtime.uploads, agents=runtime.agents)
        approver = Approver(preview=runtime.preview,
                            interactive=prompt is None)
        while True:
            try:
                line = prompt if prompt is not None else input(f"\n[{session.current}] You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in {"exit", "quit", "/exit", "/quit"}:
                break
            if not line or session.handle(line):
                if prompt is not None:
                    break
                continue
            # Skills reload every turn (local files, cheap), so an approved change is
            # visible from the next message.
            stream_turn(runtime.agent, session.content(line), session.config, reload_skills=True,
                        approver=approver)
            if prompt is not None:
                break

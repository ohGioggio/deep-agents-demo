"""The interactive chat: the input loop, slash commands and streamed output."""
import json
from contextlib import ExitStack

from deep_agents_demo.app import build_runtime, prepare_workspace
from deep_agents_demo.config import Settings
from deep_agents_demo.threads import Chats, format_table, format_transcript, new_name, thread_id, valid_name

HELP = """Chat commands:
  /threads        list your chats (* marks the current one)
  /new [name]     start a new chat (a dated name if you give none)
  /switch NAME    continue another chat
  /history        show this chat's messages
  /rename NAME    rename this chat
  /delete NAME    delete a chat and its history
  /help           show this help
  exit            quit"""

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


def stream_turn(agent, prompt, config, reload_skills=False) -> None:
    # One stream call runs the entire multi-step agent loop.
    # Include child updates so delegation/tool calls are observable.
    for namespace, update in agent.stream(
        turn_input(prompt, reload_skills),
        config=config, stream_mode="updates", subgraphs=True,
    ):
        for node, state in update.items():
            if not isinstance(state, dict):
                continue
            label = "subagent" if namespace else "main"
            for message in state.get("messages", []):
                print(f"\n[{label} / {node}]")
                message.pretty_print()
            if "todos" in state:
                print("[plan]", json.dumps(state["todos"], ensure_ascii=False))


class ChatSession:
    """Tracks the current chat and handles slash commands typed in the chat loop."""

    def __init__(self, chats: Chats, current: str, ask=input, say=print):
        self.chats, self.current, self.ask, self.say = chats, current, ask, say
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
        command, _, arg = line[1:].partition(" ")
        command, arg = command.lower(), arg.strip()
        if command == "help":
            self.say(HELP)
        elif command in {"threads", "chats", "list"}:
            table = format_table(self.chats.list(), self.current)
            if not self.chats.exists(self.current):
                table += f"\n*  {self.current} (new: saved after your first message)"
            self.say(table)
        elif command == "new":
            name = arg or new_name()
            if not valid_name(name):
                self.say(f"Invalid name {name!r}: use letters, digits, '.', '_' or '-'.")
            elif self.chats.exists(name):
                self.say(f"A chat named {name!r} already exists; use /switch {name}.")
            else:
                self._enter(name, f"New chat {name!r}. It is saved once you send a message.")
        elif command == "switch":
            if not arg:
                self.say("Usage: /switch NAME (see /threads)")
            elif not self.chats.exists(arg):
                self.say(f"No chat named {arg!r}. See /threads, or start one with /new {arg}.")
            else:
                self._enter(arg, f"Switched to {arg!r} ({self.chats.info(arg).turns} turns).")
        elif command == "history":
            if self.chats.exists(self.current):
                self.say(format_transcript(self.chats.transcript(self.current)))
            else:
                self.say("(empty chat)")
        elif command == "rename":
            try:
                self.chats.rename(self.current, arg)
            except ValueError as error:
                self.say(str(error))
            else:
                self.current = arg
                self.say(f"Renamed to {arg!r}.")
        elif command == "delete":
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
        else:
            self.say(f"Unknown command /{command}. Type /help.")
        return True


def run_chat(settings: Settings, user: str, thread: str, prompt: str | None = None) -> None:
    """Interactive chat, or a single ``prompt`` that prints only the answer."""
    # Startup chatter only helps in an interactive chat; --prompt prints just the answer.
    say = print if prompt is None else (lambda *_args, **_kwargs: None)
    prepare_workspace(settings)
    say(GREETING)
    with ExitStack() as stack:
        runtime = build_runtime(settings, user, stack, say)
        session = ChatSession(runtime.chats, thread)
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
            stream_turn(runtime.agent, line, session.config, session.reload_skills)
            session.reload_skills = False
            if prompt is not None:
                break

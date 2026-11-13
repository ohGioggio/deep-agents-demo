"""The interactive chat: the input loop, slash commands and streamed output."""
import sys

from deep_agents_demo.config import Settings
from deep_agents_demo.core.approvals import Approvals
from deep_agents_demo.core.turns import run_turn
from deep_agents_demo.features.images.uploads import (
    ImageError,
    image_paths,
    message_content,
)
from deep_agents_demo.persistence.chats import Chats, new_name, thread_id, valid_name
from deep_agents_demo.persistence.files import safe_folder_name
from deep_agents_demo.startup.resources import Resources
from deep_agents_demo.startup.runtime import build_runtime
from deep_agents_demo.startup.workspace import prepare_workspace
from deep_agents_demo.ui.cli.interrupts import ctrl_c_controls
from deep_agents_demo.ui.cli.prompts import CliInteraction
from deep_agents_demo.ui.cli.render import (
    CliRenderer,
    format_files,
    format_table,
    format_transcript,
)

GREETING = ("Ask anything. Ideas: a question about your indexed documents, "
            "'Remember that I prefer short answers', or the example skill: "
            "'Plan a city scene under 100000 triangles'.\n"
            "Type /help for chat commands, exit to quit.")


def confirm(question: str, ask=input) -> bool:
    try:
        return ask(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def stream_turn(agent, prompt, config, reload_skills=True, approver=None, runners=None,
                label=None, quiet=False, show=None, sink=None) -> None:
    """One turn printed by the CLI renderer, with ``approver`` deciding paused calls
    (without one, paused calls are rejected: nobody can approve them)."""
    approver = approver or Approvals()
    run_turn(agent, prompt, config, sink or CliRenderer(), approver.decide, runners, reload_skills, label,
             quiet, show)


class _NewestWorkers:
    """The worker factory for plans and rooms, from the newest agents: an agent change
    approved earlier in this message is rebuilt in first, so a room can use an agent the
    user just made (the main agent's own ``task`` still gets it from the next message)."""

    def __init__(self, runtime):
        self.runtime = runtime

    def _current(self):
        refresh = getattr(self.runtime, "refresh", None)
        if refresh is not None:
            refresh()  # a no-op unless an agent change is waiting
        return self.runtime.make_worker

    @property
    def names(self):
        return getattr(self._current(), "names", None)

    def __call__(self, name, extra=()):
        return self._current()(name, extra)


def runners_for(runtime, settings, thread: str, approver, interactive: bool, quiet: bool = False,
                interaction=None, sink=None, notify=None) -> dict:
    """What answers non-approval pauses in a turn: the plan runner (run_plan) and the
    room runner (start_room), both on this thread so approvals and Ctrl+C work."""

    from deep_agents_demo.features.orchestration.plans import PlanRunner
    from deep_agents_demo.features.orchestration.rooms import RoomRunner
    if runtime.make_worker is None:
        return {}
    interaction, sink = interaction or CliInteraction(), sink or CliRenderer()
    workers = _NewestWorkers(runtime)
    plans = PlanRunner(workers, approver, interaction, sink, thread=thread,
                       max_attempts=settings.plan_max_attempts,
                       budget=settings.orchestration_max_calls, interactive=interactive)
    plans.quiet = quiet

    def describe(name):
        agent = runtime.agents.get(name) if runtime.agents is not None else None
        return agent.description if agent else ""
    folder = safe_folder_name(runtime.chats.user)
    rooms = RoomRunner(workers, approver, interaction, sink, thread=thread,
                       max_turns=settings.room_max_turns, budget=settings.orchestration_max_calls,
                       interactive=interactive, transcripts=settings.paths.workspace / "rooms" / folder,
                       describe=describe)
    rooms.quiet = quiet

    def with_ctrl_c(runner):
        """Ctrl+C pauses the runner (twice: stops it) while it runs."""
        def run(request):
            with ctrl_c_controls(runner, interactive, runner.label, notify=notify):
                return runner.run(request)
        return run
    return {"run_plan": with_ctrl_c(plans), "start_room": with_ctrl_c(rooms)}


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

    def __init__(self, chats: Chats, current: str, ask=None, say=None, uploads=None, agents=None,
                 room=None, outputs=None, tables=None):
        self.chats, self.current = chats, current
        # input/print are looked up at each call, not bound once (tests patch them).
        self.ask = ask or (lambda prompt: input(prompt))
        self.say = say or (lambda *args, **kwargs: print(*args, **kwargs))
        self.uploads, self.agents = uploads, agents
        self.outputs = outputs  # the OutputStore, for /files
        self.tables = tables  # RichTables: /threads, /files, /agents as tables when given
        self.room = room  # (participants, question) -> runs a room and tells the agent
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
        footer = ("general-purpose (and researcher, when documents or the catalog are on) are "
                  "built in unless an agent replaces them.")
        if self.tables is not None:
            self.tables.agents(rows)
            self.say(footer)
            return
        width = max(len(r["name"]) for r in rows)
        lines = [f"  {'NAME':<{width}}  TOOLS / MODEL / SOURCE"]
        for r in rows:
            note = f" (overrides {r['overrides']})" if r["overrides"] else ""
            lines.append(f"  {r['name']:<{width}}  tools: {r['tools']}; model: {r['model']}; "
                         f"source: {r['source']}{note}")
        lines.append("  " + footer)
        self.say("Subagents:\n" + "\n".join(lines))

    @command("room", usage="/room A B -- QUESTION", summary="let agents discuss a question, moderated")
    def room_command(self, arg):
        from deep_agents_demo.features.orchestration.rooms import parse_room_command
        parsed = parse_room_command(arg)
        if isinstance(parsed, str):
            self.say(parsed)
            return
        names, question = parsed
        # The demo's own general-purpose (and researcher, when it exists) aren't in the
        # registry; the room runner checks every name before any model call.
        built_in = ("general-purpose", "researcher")
        unknown = [n for n in names if n not in built_in
                   and (self.agents is None or self.agents.get(n) is None)]
        if unknown:
            self.say(f"There's no agent named {unknown[0]!r}. See /agents.")
        elif self.room is None:
            self.say("Rooms can't run here.")
        else:
            self.room(names, question)

    @command("help", usage="/help", summary="show this help")
    def help(self, arg):
        self.say(HELP)

    @command("files", usage="/files", summary="list the files the agent delivered in this chat")
    def files(self, arg):
        if self.outputs is None:
            self.say("Files aren't available here.")
            return
        user = self.chats.user
        files, folder = self.outputs.list(user, self.current), self.outputs.folder(user, self.current)
        if self.tables is not None:
            self.tables.files(files, folder)
        else:
            self.say(format_files(files, folder))

    @command("threads", "chats", "list", usage="/threads", summary="list your chats (* marks the current one)")
    def threads(self, arg):
        chats = self.chats.list()
        new = not self.chats.exists(self.current)
        if self.tables is not None:
            self.tables.threads(chats, self.current)
            if new:
                self.say(f"*  {self.current} (new: saved after your first message)")
            return
        table = format_table(chats, self.current)
        if new:
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
            try:
                self.chats.delete(arg)
            except ValueError as error:
                self.say(str(error))
                return
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


def _warn(message: str) -> None:
    """Warnings (embedding retries) go to stderr, also in one-shot runs."""
    print(message, file=sys.stderr)


def _rich_parts(verbose: bool):
    """The Rich renderer, interaction and tables for one chat (imported here: plain runs
    never load Rich)."""
    from deep_agents_demo.ui.cli.richui.interaction import RichInteraction
    from deep_agents_demo.ui.cli.richui.renderer import RichRenderer
    from deep_agents_demo.ui.cli.richui.tables import RichTables
    from deep_agents_demo.ui.cli.richui.theme import make_console
    console = make_console()
    renderer = RichRenderer(console, verbose)
    return renderer, RichInteraction(console, renderer), RichTables(console)


def run_chat(settings: Settings, user: str, thread: str, prompt: str | None = None,
             verbose: bool = False, plain: bool = False) -> None:
    """Interactive chat, or a single ``prompt`` that prints only the answer (every step
    with ``verbose``). In a terminal the chat uses the Rich UI unless ``plain``."""
    from deep_agents_demo.ui.cli.richui.mode import use_rich
    quiet = prompt is not None and not verbose
    rich = use_rich(prompt is None, plain, sys.stdin.isatty(), sys.stdout.isatty())
    # Startup chatter only helps in an interactive chat; --prompt prints just the answer.
    say = print if prompt is None else (lambda *_args, **_kwargs: None)
    prepare_workspace(settings)
    say(GREETING)
    with Resources.open(settings, notify=say, warn=_warn) as resources:
        renderer = None
        try:
                runtime = build_runtime(resources, user, notify=say)
                tables = box = None
                sink, interaction, notify = None, CliInteraction(), None
                if rich:
                    renderer, interaction, tables = _rich_parts(verbose)
                    sink, notify = renderer, renderer.notify
                session = ChatSession(runtime.chats, thread, uploads=runtime.uploads, agents=runtime.agents,
                                      outputs=runtime.outputs, tables=tables)
                if rich:
                    from deep_agents_demo.ui.cli.richui.input_box import InputBox
                    history = settings.paths.data / "history" / f"{safe_folder_name(user)}.txt"
                    box = InputBox(
                        COMMANDS, lambda: runtime.chats.list(),
                        lambda: [(a["name"], a["description"]) for a in session.agents.overview()]
                        if session.agents is not None else [],
                        history, session.current, console=renderer.console)

                def runners():
                    return runners_for(runtime, settings, thread_id(runtime.chats.user, session.current),
                                       approver, interactive=prompt is None, quiet=quiet,
                                       interaction=interaction, sink=sink, notify=notify)

                def run_room(names, question):
                    """/room: run it, then tell the agent the outcome as an ordinary turn."""
                    active = runners()
                    from deep_agents_demo.features.orchestration.rooms import room_followup
                    outcome = active["start_room"]({"type": "start_room", "question": question,
                                                    "participants": names})
                    if "summary" not in outcome:  # refused before it started, e.g. an unknown agent
                        say(f"Room not started: {outcome.get('error')}")
                        return
                    stream_turn(runtime.agent, room_followup(question, names, outcome), session.config,
                                approver=approver, runners=active, quiet=quiet, sink=sink)
                session.room = run_room
                approver = Approvals(runtime.preview, interaction, interactive=prompt is None)
                while True:
                    try:
                        if prompt is not None:
                            line = prompt
                        elif box is not None:
                            line = box.read(session.current).strip()
                        else:
                            line = input(f"\n[{session.current}] You: ").strip()
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
                                approver=approver, runners=runners(), quiet=quiet, sink=sink)
                    # An approved agent change: available from the next message (a room or plan
                    # in this message may have rebuilt the agents already).
                    if runtime.refresh() or session.agents is not runtime.agents:
                        session.agents = runtime.agents
                        say("Agents updated.")
                    if prompt is not None:
                        break
        finally:  # before the resources' teardown, so nothing draws over it
            if renderer is not None:
                renderer.close()  # Ctrl+C mid-turn raises no RunError; stop the live view first

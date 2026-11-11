"""Agents written through the chat: plan a change, show it for approval, save a version.

``AgentManager`` reuses the skills machinery (``SkillManager``: planning, scopes,
versions, the marketplace, approvals) with an agent file instead of a SKILL.md:
each version is one file, ``agent.md``, in Claude Code's agent format. Nothing is
written to folders; after an approved change ``changed`` is set and the runtime
rebuilds the agent, so the change is available from the user's next message.
"""
import itertools
import json
import threading
from pathlib import Path

from deep_agents_demo.agent_files import (
    CLAUDE_TOOLS,
    _names,
    normalise_agent_text,
    parse_agent_md,
    render_agent_md,
    split_agent_md,
    translate_tools,
)
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.memory_policy import find_secrets, split_findings
from deep_agents_demo.models.agents import READ_ONLY
from deep_agents_demo.skill_manager import MARKETPLACE, USER, SkillChange, SkillManager

AGENT_FILE = "agent.md"
MAX_AGENT_SIZE = 100_000
GATED_AGENT_TOOLS = ("create_agent", "update_agent", "remove_agent", "import_agent", "restore_agent",
                     "publish_agent", "install_agent", "uninstall_agent")


def _note(note: str) -> str:
    """A parse note without its "<source>: " prefix."""
    return note.split(": ", 1)[1] if ": " in note else note


class AgentChange(SkillChange):
    noun = "Agent"
    main_file = AGENT_FILE

    def tools_used(self) -> list[str]:
        """Tools as the agent will get them: translated, read-only mode, denylist."""
        files = self.after if self.after is not None else self.before or {}
        if not files.get(AGENT_FILE):
            return []
        front = split_agent_md(files[AGENT_FILE])[0]
        if front.get("permissionMode") == "plan":
            return [f"read-only ({', '.join(READ_ONLY)})"]

        def ours(value):
            return list(dict.fromkeys(t for n in _names(value) for t in CLAUDE_TOOLS.get(n, (n,))))
        names = ours(front.get("tools")) or ["all"]
        denied = ours(front.get("disallowedTools"))
        if denied:
            names[-1] += f" (except {', '.join(denied)})"
        return names


class AgentManager(SkillManager):
    """One user's agents (scope "user") and the agent marketplace."""

    noun = "agent"
    change_type = AgentChange
    main_file = AGENT_FILE
    browse_tool = "browse_agents"

    def __init__(self, store, owner: str, known_tools):
        self.store, self.owner = store, owner
        self.known_tools = frozenset(known_tools)
        self.lock = threading.RLock()
        self.changed = False  # set after an approved change; the runtime rebuilds and clears it
        self.registry = None  # the AgentRegistry in use, for list_agents (set by the runtime)

    # hooks ------------------------------------------------------------------------
    def validate(self, name, files) -> list[str]:
        if set(files) != {AGENT_FILE}:
            return [f"an agent is one file, {AGENT_FILE}"]
        text = files[AGENT_FILE]
        if len(text) > MAX_AGENT_SIZE:
            return [f"the agent file exceeds {MAX_AGENT_SIZE} characters"]
        agent, notes = parse_agent_md(text, name, self.known_tools)
        errors = [] if agent else [_note(notes[0])]
        if agent and agent.name != name:
            errors.append(f"the file's name {agent.name!r} does not match {name!r}")
        credentials, _wording = split_findings(find_secrets(text))
        if credentials:
            errors.append(f"it contains what looks like a credential ({', '.join(credentials)}); "
                          "remove it")
        return errors

    def warnings(self, files) -> list[str]:
        text = files.get(AGENT_FILE, "")
        agent, notes = parse_agent_md(text, "agent", self.known_tools)
        found = [_note(n) for n in notes] if agent else []
        if split_findings(find_secrets(text))[1]:
            found.append("it mentions a password, token or key; check it holds no real value")
        return found

    def after_apply(self) -> None:
        self.changed = True

    def _tool_errors(self, tools):
        if tools is None:
            return []
        if not isinstance(tools, (list, tuple)) or not all(isinstance(t, str) for t in tools):
            return ["tools must be a list of tool names"]
        _resolved, notes = translate_tools(list(tools), self.known_tools)
        return [f"{note} (available: {', '.join(sorted(self.known_tools))})" for note in notes]

    @staticmethod
    def _skill_errors(skills):
        if skills is None or (isinstance(skills, (list, tuple))
                              and all(isinstance(s, str) for s in skills)):
            return []
        return ["skills must be a list of skill names"]

    def _front(self, front: dict, tools, model, skills) -> dict:
        front = dict(front)
        if tools is not None:
            front.pop("tools", None)
            if tools:
                front["tools"] = ", ".join(translate_tools(list(tools), self.known_tools)[0])
        if model is not None:
            front.pop("model", None)
            if model:
                front["model"] = model
        if skills is not None:
            front.pop("skills", None)
            if skills:
                front["skills"] = list(skills)
        return front

    # planning ---------------------------------------------------------------------
    def plan_create(self, name, description, prompt, tools=None, model=None, skills=None, scope=USER):
        if self._live(name, scope):
            return self._change("create", name, None, [f"a {scope} agent named {name!r} already "
                                                       "exists; use update_agent"], scope)
        front = self._front({"name": name, "description": " ".join(str(description).split())},
                            tools, model, skills)
        return self._change("create", name, {AGENT_FILE: render_agent_md(front, prompt)},
                            self._tool_errors(tools) + self._skill_errors(skills), scope=scope)

    def plan_update(self, name, description=None, prompt=None, tools=None, model=None, skills=None,
                    scope=USER):
        current = self._live(name, scope)
        if current is None:
            return self._change("update", name, None, [f"no {scope} agent named {name!r}; use "
                                                       "create_agent"], scope)
        front, body = split_agent_md(current.files[AGENT_FILE])
        if description is not None:
            front["description"] = " ".join(str(description).split())
        front = self._front(front, tools, model, skills)
        text = render_agent_md(front, prompt if prompt is not None else body)
        return self._change("update", name, {AGENT_FILE: text},
                            self._tool_errors(tools) + self._skill_errors(skills), scope=scope)

    def plan_import(self, path, scope=USER):
        path = Path(path)
        if path.is_dir():
            # Stop after 21: enough to show 20 and say there are more, even for a huge tree.
            found = itertools.islice((p for p in path.rglob("*.md") if p.name.lower() != "readme.md"), 21)
            files = [str(p.relative_to(path)) for p in found]
            shown = ", ".join(files[:20]) + (" ..." if len(files) > 20 else "")
            return self.change_type("import", str(path), self.owner, scope, errors=[
                f"{path} is a folder ({shown or 'no agent files'}); import its agents one at a "
                "time, so each gets its own approval"])
        try:
            if path.stat().st_size > 4 * MAX_AGENT_SIZE:  # don't read a huge file to refuse it
                return self.change_type("import", str(path), self.owner, scope,
                                        errors=[f"the agent file exceeds {MAX_AGENT_SIZE} characters"])
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return self.change_type("import", str(path), self.owner, scope,
                                    errors=[f"can't read {path}: {error}"])
        agent, notes = parse_agent_md(text, path.name, self.known_tools)
        if agent is None:
            return self.change_type("import", str(path), self.owner, scope, errors=[_note(notes[0])])
        # Stored as this module reads it, so updates and summaries see the same file.
        return self._change("import", agent.name, {AGENT_FILE: normalise_agent_text(text)},
                            scope=scope)

    # views --------------------------------------------------------------------------
    def overview(self):
        """The user's agents and installed ones, with versions (agents' wording)."""
        mine = {a.name: a for a in self.store.list_current(USER, self.owner)}
        rows = [{"name": name, "source": "yours", "version": agent.version}
                for name, agent in sorted(mine.items())]
        for name in self.store.installed(self.owner):
            listed = self._live(name, MARKETPLACE)
            rows.append({"name": name, "source": "marketplace (installed)",
                         "version": listed.version if listed else None,
                         "note": ("removed from the marketplace" if listed is None else
                                  "overridden by your own agent with the same name" if name in mine
                                  else "")})
        return rows

    def sync_folder(self):
        """Agents live only in the database; nothing to write to folders."""
        return None

    # the agent's tools ----------------------------------------------------------------
    def tools(self):
        def done(plan, *args, **kwargs):
            # Plan and apply under one lock, so parallel calls see each other's results.
            with self.lock:
                return json.dumps(self._apply(plan(*args, **kwargs)), default=str)

        def create_agent(name: str, description: str, prompt: str, tools: list[str] | None = None,
                         model: str | None = None, skills: list[str] | None = None,
                         scope: str = "user") -> str:
            """Create a subagent (only when the user asks). name: letters, digits, '-' or '_'.
            description: what it does and when to use it. prompt: its instructions.
            tools: existing tool names it may use (omit for all tools). model: a model id
            (omit to use the configured model). skills: skill names to give it at start.
            scope: "user" (default) or "marketplace" (only if asked). Needs approval."""
            return done(self.plan_create, name, description, prompt, tools, model, skills, scope)

        def update_agent(name: str, description: str | None = None, prompt: str | None = None,
                         tools: list[str] | None = None, model: str | None = None,
                         skills: list[str] | None = None, scope: str = "user") -> str:
            """Update a subagent (only when asked). Give only what changes; tools=[] means all
            tools again. scope: "user" (default) or "marketplace". Needs approval."""
            return done(self.plan_update, name, description, prompt, tools, model, skills, scope)

        def remove_agent(name: str, scope: str = "user") -> str:
            """Remove a subagent (only when asked); its history is kept. Needs approval."""
            return done(self.plan_remove, name, scope)

        def import_agent(path: str, scope: str = "user") -> str:
            """Import one agent file (Claude Code format, e.g. from .claude/agents) on the
            user's machine (only when asked). Needs approval."""
            return done(self.plan_import, path, scope)

        def restore_agent(name: str, version: int, scope: str = "user") -> str:
            """Restore an earlier version of a subagent (see agent_history). Needs approval."""
            return done(self.plan_restore, name, version, scope)

        def publish_agent(name: str) -> str:
            """Publish one of the user's agents to the marketplace (only when asked). Needs approval."""
            return done(self.plan_publish, name)

        def install_agent(name: str) -> str:
            """Install a marketplace agent for this user (only when asked; see browse_agents).
            Needs approval."""
            return done(self.plan_install, name)

        def uninstall_agent(name: str) -> str:
            """Uninstall a marketplace agent for this user (only when asked). Needs approval."""
            return done(self.plan_uninstall, name)

        def list_agents() -> str:
            """Every subagent you can hand work to: built-in, folders, installed, the user's own."""
            if self.registry is None:
                return json.dumps(self.overview())
            # The registry lists what loaded; the notes say what didn't (e.g. an installed
            # agent removed from the marketplace, or one whose tools no longer exist).
            return json.dumps({"agents": self.registry.overview(), "notes": self.definitions()[2]})

        def agent_history(name: str, scope: str = "user") -> str:
            """The saved versions of one of the user's agents. scope: "user" or "marketplace"."""
            try:
                versions = self.store.history(scope, self._owner(scope), name)
            except ValueError as error:
                return json.dumps({"error": str(error)})
            if not versions:
                return json.dumps({"error": f"no {scope} agent named {name!r}"})
            return json.dumps([{"version": v.version, "action": v.action, "saved": v.created_at}
                               for v in versions])

        def browse_agents(query: str = "") -> str:
            """Search the agent marketplace (empty query lists everything)."""
            return json.dumps(self.browse(query))

        return [create_agent, update_agent, remove_agent, import_agent, restore_agent, publish_agent,
                install_agent, uninstall_agent, list_agents, agent_history, browse_agents]

    def feature(self) -> Feature:
        prompt = ("Subagents are agents you can hand work to with task. The user can have their "
                  "own, in Claude Code's agent format, and install shared ones from a marketplace. "
                  "Change them only when the user asks: create_agent, update_agent, remove_agent, "
                  "import_agent, restore_agent, publish_agent, install_agent, uninstall_agent. "
                  "Their tools must be existing tool names. Each change is shown to the user for "
                  "approval and is available from the next message. list_agents shows every "
                  "agent and where it comes from. When the user names an agent, hand the work "
                  "to that agent with task (subagent_type is its name), not to another one. ")
        return Feature(name="agents", prompt=prompt, tools=tuple(self.tools()),
                       approvals=GATED_AGENT_TOOLS)

    def plan_uninstall(self, name):
        change = super().plan_uninstall(name)
        if not change.errors and self._live(name, MARKETPLACE) is None:
            change.before = {}  # gone from the marketplace: nothing to show but the uninstall
        return change

    # what the registry loads ---------------------------------------------------------
    def definitions(self):
        """(installed marketplace agents, the user's own agents, notes)."""
        installed, own, notes = [], [], []
        for item in self.store.list_current(USER, self.owner):
            agent, problems = parse_agent_md(item.files[AGENT_FILE], "yours", self.known_tools)
            if agent:
                own.append(agent)
            else:
                notes.append(f"your agent {item.name!r} can't be loaded: {_note(problems[0])}")
        for name in self.store.installed(self.owner):
            listed = self._live(name, MARKETPLACE)
            if listed is None:
                notes.append(f"installed agent {name!r} was removed from the marketplace")
                continue
            agent, problems = parse_agent_md(listed.files[AGENT_FILE], "marketplace", self.known_tools)
            if agent:
                installed.append(agent)
            else:
                notes.append(f"installed agent {name!r} can't be loaded: {_note(problems[0])}")
        return installed, own, notes

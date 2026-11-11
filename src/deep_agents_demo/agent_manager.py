"""Agents written through the chat: plan a change, show it for approval, save a version.

``AgentManager`` reuses the skills machinery (``SkillManager``: planning, scopes,
versions, the marketplace, approvals) with an agent file instead of a SKILL.md:
each version is one file, ``agent.md``, in Claude Code's agent format. Nothing is
written to folders; after an approved change ``changed`` is set and the runtime
rebuilds the agent, so the change is available from the user's next message.
"""
import threading
from pathlib import Path

from deep_agents_demo.agent_files import _names, parse_agent_md, render_agent_md, translate_tools
from deep_agents_demo.memory_policy import find_secrets, split_findings
from deep_agents_demo.skill_manager import MARKETPLACE, USER, SkillChange, SkillManager, split_skill_md

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
        files = self.after if self.after is not None else self.before or {}
        front = split_skill_md(files.get(AGENT_FILE, ""))[0]
        return _names(front["tools"]) if front.get("tools") else ["all"]


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
                            self._tool_errors(tools), scope=scope)

    def plan_update(self, name, description=None, prompt=None, tools=None, model=None, skills=None,
                    scope=USER):
        current = self._live(name, scope)
        if current is None:
            return self._change("update", name, None, [f"no {scope} agent named {name!r}; use "
                                                       "create_agent"], scope)
        front, body = split_skill_md(current.files[AGENT_FILE])
        if description is not None:
            front["description"] = " ".join(str(description).split())
        front = self._front(front, tools, model, skills)
        text = render_agent_md(front, prompt if prompt is not None else body)
        return self._change("update", name, {AGENT_FILE: text}, self._tool_errors(tools), scope=scope)

    def plan_import(self, path, scope=USER):
        path = Path(path)
        if path.is_dir():
            files = sorted(str(p.relative_to(path)) for p in path.rglob("*.md")
                           if p.name.lower() != "readme.md")
            shown = ", ".join(files[:20]) + (" ..." if len(files) > 20 else "")
            return self.change_type("import", str(path), self.owner, scope, errors=[
                f"{path} is a folder ({shown or 'no agent files'}); import its agents one at a "
                "time, so each gets its own approval"])
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return self.change_type("import", str(path), self.owner, scope,
                                    errors=[f"can't read {path}: {error}"])
        agent, notes = parse_agent_md(text, path.name, self.known_tools)
        if agent is None:
            return self.change_type("import", str(path), self.owner, scope, errors=[_note(notes[0])])
        return self._change("import", agent.name, {AGENT_FILE: text}, scope=scope)

    # what the registry loads ---------------------------------------------------------
    def definitions(self):
        """(installed marketplace agents, the user's own agents, notes)."""
        installed, own, notes = [], [], []
        for item in self.store.list_current(USER, self.owner):
            agent, _ = parse_agent_md(item.files[AGENT_FILE], "yours", self.known_tools)
            if agent:
                own.append(agent)
        for name in self.store.installed(self.owner):
            listed = self._live(name, MARKETPLACE)
            if listed is None:
                notes.append(f"installed agent {name!r} was removed from the marketplace")
                continue
            agent, _ = parse_agent_md(listed.files[AGENT_FILE], "marketplace", self.known_tools)
            if agent:
                installed.append(agent)
        return installed, own, notes

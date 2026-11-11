"""Versioned items, generically: plan a change, show it for approval, then save it.

An item (a skill, an agent) belongs to one user (scope "user") or to the shared
marketplace. Each tool call becomes an ``ItemChange`` (create, update, remove,
import, restore, publish, install, uninstall) that knows its validation errors, a
summary, a diff and the full files. A UI shows the change before the user approves
it, and the tool applies the same change afterwards, so what is approved is what
is saved (the Command pattern). ``ItemManager`` holds what every kind of item
shares; ``features/skills`` and ``features/agents`` fill in their formats.
"""
import difflib
import threading
from dataclasses import dataclass, field

import yaml
from deepagents.middleware.skills import _FRONTMATTER_PATTERN

from deep_agents_demo.persistence.versioned_store import VersionedStore

USER, MARKETPLACE = "user", "marketplace"
SCOPES = (USER, MARKETPLACE)


def split_frontmatter(content: str) -> tuple[dict, str]:
    """(frontmatter, body) of a Markdown file with YAML frontmatter; ({}, content) if none."""
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content
    front = yaml.safe_load(match.group(1)) or {}
    return (front if isinstance(front, dict) else {}), content[match.end():].lstrip("\n")


@dataclass
class ItemChange:
    action: str                       # create, update, remove, import, restore, publish, install, uninstall
    name: str
    owner: str                        # the user asking
    scope: str = USER                 # where the change lands: user or marketplace
    before: dict | None = None        # current files, if any
    after: dict | None = None         # files after the change (None when removed / uninstalled)
    version_from: int | None = None
    version_to: int | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)   # shown in the summary; the user decides

    # Subclasses set these; class attributes, not dataclass fields.
    noun = "Item"
    main_file = ""

    @property
    def description(self) -> str:
        files = self.after or self.before or {}
        return str(split_frontmatter(files.get(self.main_file, ""))[0].get("description", ""))

    def tools_used(self) -> list[str]:
        """The tools the item says it uses, for the summary (none by default)."""
        return []

    def _diff_lines(self):
        before, after = self.before or {}, self.after or {}
        for path in sorted(set(before) | set(after)):
            old, new = before.get(path, ""), after.get(path, "")
            if old != new:
                yield from difflib.unified_diff(
                    old.splitlines(), new.splitlines(), lineterm="",
                    fromfile=f"a/{path}" if path in before else "/dev/null",
                    tofile=f"b/{path}" if path in after else "/dev/null")

    def diff(self) -> str:
        return "\n".join(self._diff_lines()) or "(no changes)"

    def full(self) -> str:
        files = self.after if self.after is not None else self.before or {}
        label = "files that will be removed" if self.after is None else "files after the change"
        return f"--- {label} ---\n" + "\n".join(
            f"=== {path} ===\n{text.rstrip()}\n" for path, text in sorted(files.items()))

    def where(self) -> str:
        if self.action in ("install", "uninstall"):
            return f"marketplace {self.noun.lower()}, for user {self.owner}"
        return "marketplace, shared with everyone" if self.scope == MARKETPLACE else f"user: {self.owner}"

    def summary(self) -> str:
        if self.action in ("install", "uninstall"):
            versions = f"marketplace v{self.version_to or self.version_from}"
        elif self.version_from:
            versions = f"v{self.version_from} -> v{self.version_to}"
        else:
            versions = f"new, v{self.version_to}"
        lines = [f"{self.noun} change requested: {self.action.upper()} {self.name!r} "
                 f"({self.where()}), {versions}"]
        if self.description:
            lines.append(f"  {self.description}")
        tools = self.tools_used()
        if tools:
            lines.append(f"  Tools it uses: {', '.join(tools)}")
        diff = list(self._diff_lines())
        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        files = sorted((self.after if self.after is not None else self.before or {}))
        lines.append(f"  Changes: +{added} lines, -{removed} lines · files: {', '.join(files)}")
        lines.append("  Checks: " + ("; ".join(self.errors) if self.errors else
                                     "format OK · no credentials found"))
        lines.extend(f"  Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


class ItemManager:
    """One user's versioned items of one kind (skills, agents): their own (scope
    "user") and the shared marketplace catalogue they publish to and install from.

    Template methods: a subclass sets ``noun``, ``change_type``, ``main_file`` and
    ``browse_tool``; provides ``plan_create``, ``plan_update``, ``plan_import``,
    ``tools`` and ``feature``; and overrides the hooks (``validate``, ``warnings``,
    ``after_apply``, ``_tool_errors``) as its format needs. Everything else (scopes,
    versions, remove/restore, publish/install, applying, browsing) lives here.
    """

    noun = "item"
    change_type = ItemChange
    main_file = ""
    browse_tool = "browse_marketplace"
    own_source = "user"  # how overview() labels the user's own items

    def __init__(self, store: VersionedStore, owner: str):
        self.store, self.owner = store, owner
        # Approved calls from one step run in parallel threads: apply them one at a time,
        # so each re-plans against the latest state and folder rewrites don't overlap.
        self.lock = threading.RLock()
        # Names from the tools registry, set once the agent's features are known; a
        # skill's ``tools`` must come from it. None skips the check.
        self.known_tools = None

    def _owner(self, scope):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        return self.owner if scope == USER else ""

    def _live(self, name, scope):
        item = self.store.get(scope, self._owner(scope), name)
        return item if item and not item.removed else None
    # hooks ------------------------------------------------------------------------
    def validate(self, name, files) -> list[str]:
        """Everything that stops the item being saved."""
        raise NotImplementedError

    def warnings(self, files) -> list[str]:
        """Shown to the user, who decides."""
        return []

    def after_apply(self) -> None:
        """Called after every saved change."""

    # planning (no side effects) -------------------------------------------------
    def _change(self, action, name, after, errors=(), scope=USER):
        current = self.store.get(scope, self._owner(scope), name)
        live = current if current and not current.removed else None
        change = self.change_type(action, name, self.owner, scope,
                             before=live.files if live else None, after=after,
                             version_from=live.version if live else None,
                             version_to=(current.version + 1) if current else 1,
                             errors=list(errors))
        if after is not None and not change.errors:
            change.errors = self.validate(name, after)
            change.warnings = self.warnings(after)
        return change

    def _tool_errors(self, tools):
        """An item's declared tools must exist in the registry (the agent can't add tools)."""
        if tools is None:
            return []
        if not isinstance(tools, (list, tuple)) or not all(isinstance(t, str) for t in tools):
            return ["tools must be a list of tool names"]
        unknown = [t for t in tools if self.known_tools is not None and t not in self.known_tools]
        if unknown:
            return [f"no tool named {', '.join(unknown)} "
                    f"(available: {', '.join(sorted(self.known_tools))})"]
        return []
    def plan_remove(self, name, scope=USER):
        if self._live(name, scope) is None:
            return self._change("remove", name, None, [f"no {scope} {self.noun} named {name!r}"], scope)
        return self._change("remove", name, None, scope=scope)

    def plan_restore(self, name, version, scope=USER):
        old = self.store.version(scope, self._owner(scope), name, int(version))
        if old is None:
            return self._change("restore", name, None, [f"{name!r} has no version {version}"], scope)
        return self._change("restore", name, dict(old.files), scope=scope)

    def plan_publish(self, name):
        """Copy one of the user's items to the marketplace (a new item or a new version)."""
        mine = self._live(name, USER)
        if mine is None:
            return self._change("publish", name, None, [f"no user {self.noun} named {name!r} to publish"],
                                MARKETPLACE)
        return self._change("publish", name, dict(mine.files), scope=MARKETPLACE)

    def plan_install(self, name):
        listed = self._live(name, MARKETPLACE)
        if listed is None:
            return self.change_type("install", name, self.owner, MARKETPLACE,
                                    errors=[f"the marketplace has no {self.noun} named {name!r}; "
                                            f"see {self.browse_tool}"])
        if name in self.store.installed(self.owner):
            return self.change_type("install", name, self.owner, MARKETPLACE,
                                    errors=[f"{name!r} is already installed"])
        change = self.change_type("install", name, self.owner, MARKETPLACE, after=dict(listed.files),
                                  version_to=listed.version)
        change.errors, change.warnings = self.validate(name, listed.files), self.warnings(listed.files)
        return change

    def plan_uninstall(self, name):
        if name not in self.store.installed(self.owner):
            return self.change_type("uninstall", name, self.owner, MARKETPLACE,
                                    errors=[f"{name!r} is not installed"])
        listed = self.store.get(MARKETPLACE, "", name)
        return self.change_type("uninstall", name, self.owner, MARKETPLACE,
                                before=dict(listed.files) if listed else {},
                           version_from=listed.version if listed else None)

    def plan(self, tool_name, args) -> ItemChange | None:
        """The change a gated tool call would make, or None for other tools."""
        verbs = ("create", "update", "remove", "restore", "import", "publish", "install", "uninstall")
        planners = {f"{verb}_{self.noun}": getattr(self, f"plan_{verb}") for verb in verbs}
        planner = planners.get(tool_name)
        if planner is None:
            return None
        try:
            return planner(**args)
        except (TypeError, ValueError) as error:  # arguments the model got wrong
            return self.change_type(tool_name.removesuffix(f"_{self.noun}"), str(args.get("name", "?")),
                                    self.owner, errors=[f"bad arguments: {error}"])

    # applying ---------------------------------------------------------------------
    def apply(self, change: ItemChange) -> dict:
        with self.lock:
            return self._apply(change)

    def _apply(self, change: ItemChange) -> dict:
        if change.errors:
            return {"saved": False, "errors": change.errors}
        result = {"saved": True, "action": change.action, "name": change.name, "scope": change.scope}
        if change.action == "install":
            self.store.install(self.owner, change.name)
        elif change.action == "uninstall":
            self.store.uninstall(self.owner, change.name)
        elif change.action == "remove":
            result["version"] = self.store.remove(change.scope, self._owner(change.scope), change.name)
        else:
            result["version"] = self.store.save(change.scope, self._owner(change.scope), change.name,
                                                change.after, change.action)
        self.after_apply()
        return {**result, "note": "The change is available from the user's next message."}

    def sync_folder(self):
        """Write the user's items where they are loaded from (nothing by default)."""
        return None

    # read-only views ----------------------------------------------------------------
    def browse(self, query=""):
        installed = set(self.store.installed(self.owner))
        query = query.lower().strip()
        found = []
        for item in self.store.list_current(MARKETPLACE, ""):
            description = str(split_frontmatter(item.files[self.main_file])[0].get("description", ""))
            if query and query not in item.name.lower() and query not in description.lower():
                continue
            found.append({"name": item.name, "description": description, "version": item.version,
                          "installed": item.name in installed})
        return found

    def overview(self):
        mine = {s.name: s for s in self.store.list_current(USER, self.owner)}
        rows = [{"name": name, "source": self.own_source, "version": item.version}
                for name, item in sorted(mine.items())]
        for name in self.store.installed(self.owner):
            listed = self._live(name, MARKETPLACE)
            rows.append({"name": name, "source": "marketplace (installed)",
                         "version": listed.version if listed else None,
                         "note": ("removed from the marketplace" if listed is None else
                                  f"overridden by your own {self.noun} with the same name" if name in mine
                                  else "")})
        return rows

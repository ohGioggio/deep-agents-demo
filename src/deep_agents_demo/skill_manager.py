"""Agent-written skills: plan a change, show it for approval, then save it.

``SkillManager`` serves one user. Each tool call becomes a ``SkillChange`` (create,
update, remove, import or restore) that knows its validation errors, a summary, a
diff and the full files. The CLI shows a change before the user approves it, and
the tool applies the same change afterwards, so what is approved is what is saved.

Every saved version goes to ``SkillStore``. The user's current skills are written
to ``<workspace>/user-skills/<user>/`` and the marketplace skills they installed to
``<workspace>/installed-skills/<user>/``, where Deep Agents loads them.
"""
import difflib
import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml
from deepagents.middleware.skills import (MAX_SKILL_DESCRIPTION_LENGTH, MAX_SKILL_FILE_SIZE,
                                          _FRONTMATTER_PATTERN, _parse_skill_metadata,
                                          _validate_skill_name)

from deep_agents_demo.agent import Feature
from deep_agents_demo.memory_policy import find_secrets, split_findings
from deep_agents_demo.skill_store import SkillStore

USER, MARKETPLACE = "user", "marketplace"
SCOPES = (USER, MARKETPLACE)
MAX_FILES = 50
MAX_TOTAL_SIZE = 1_000_000
GATED_TOOLS = ("create_skill", "update_skill", "remove_skill", "import_skill", "restore_skill",
               "publish_skill", "install_skill", "uninstall_skill")


# --- SKILL.md format -----------------------------------------------------------------

def render_skill_md(name: str, description: str, instructions: str, extra: dict | None = None) -> str:
    """A SKILL.md in Deep Agents' format: YAML frontmatter, then the instructions."""
    front = {"name": name, "description": " ".join(description.split()), **(extra or {})}
    return ("---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10**9)
            + "---\n\n" + instructions.strip() + "\n")


def split_skill_md(content: str) -> tuple[dict, str]:
    """(frontmatter, instructions) of a SKILL.md; ({}, content) if it has no frontmatter."""
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content
    front = yaml.safe_load(match.group(1)) or {}
    return (front if isinstance(front, dict) else {}), content[match.end():].lstrip("\n")


def validate(name: str, files: dict) -> list[str]:
    """Everything that would stop Deep Agents loading the skill, or make it unsafe."""
    errors = []
    ok, message = _validate_skill_name(name, name)
    if not ok:
        errors.append(f"name {name!r}: {message} (use lowercase words joined by single hyphens)")
    skill_md = files.get("SKILL.md")
    if not isinstance(skill_md, str):
        return errors + ["SKILL.md is missing"]
    if len(files) > MAX_FILES:
        errors.append(f"too many files ({len(files)} > {MAX_FILES})")
    if sum(len(str(text)) for text in files.values()) > MAX_TOTAL_SIZE:
        errors.append(f"files exceed {MAX_TOTAL_SIZE} characters in total")
    for path, text in files.items():
        parts = PurePosixPath(path).parts
        if (not path or path.startswith(("/", "\\")) or "\\" in path or ".." in parts
                or re.match(r"^[A-Za-z]:", path)):
            errors.append(f"file path {path!r} must be relative, with no '..'")
        if not isinstance(text, str):
            errors.append(f"{path}: only text files are supported")
            continue
        credentials, _wording = split_findings(find_secrets(text))
        if credentials:
            errors.append(f"{path} contains what looks like a credential ({', '.join(credentials)}); "
                          "remove it")
    if len(skill_md) > MAX_SKILL_FILE_SIZE:
        errors.append("SKILL.md is too large")
    metadata = _parse_skill_metadata(skill_md, f"{name}/SKILL.md", name)
    if metadata is None:
        errors.append("SKILL.md needs YAML frontmatter with 'name' and 'description'")
    else:
        if metadata["name"] != name:
            errors.append(f"SKILL.md name {metadata['name']!r} does not match {name!r}")
        front, _body = split_skill_md(skill_md)
        if len(str(front.get("description", ""))) > MAX_SKILL_DESCRIPTION_LENGTH:
            errors.append(f"description exceeds {MAX_SKILL_DESCRIPTION_LENGTH} characters")
    return errors


def warnings(files: dict) -> list[str]:
    """Wording that might reveal a secret: shown to the user, who decides."""
    found = []
    for path, text in sorted(files.items()):
        _credentials, wording = split_findings(find_secrets(str(text)))
        if wording:
            found.append(f"{path} mentions a password, token or key; check it holds no real value")
    return found


# --- a planned change ----------------------------------------------------------------

@dataclass
class SkillChange:
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

    @property
    def description(self) -> str:
        files = self.after or self.before or {}
        return str(split_skill_md(files.get("SKILL.md", ""))[0].get("description", ""))

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
            return f"marketplace skill, for user {self.owner}"
        return "marketplace, shared with everyone" if self.scope == MARKETPLACE else f"user: {self.owner}"

    def summary(self) -> str:
        if self.action in ("install", "uninstall"):
            versions = f"marketplace v{self.version_to or self.version_from}"
        elif self.version_from:
            versions = f"v{self.version_from} -> v{self.version_to}"
        else:
            versions = f"new, v{self.version_to}"
        lines = [f"Skill change requested: {self.action.upper()} {self.name!r} "
                 f"({self.where()}), {versions}"]
        if self.description:
            lines.append(f"  {self.description}")
        diff = list(self._diff_lines())
        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        files = sorted((self.after if self.after is not None else self.before or {}))
        lines.append(f"  Changes: +{added} lines, -{removed} lines · files: {', '.join(files)}")
        lines.append("  Checks: " + ("; ".join(self.errors) if self.errors else
                                     "format OK · no credentials found"))
        lines.extend(f"  Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


# --- the manager ---------------------------------------------------------------------

def _folder_name(owner: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", owner) or "_"


class SkillManager:
    """One user's skills: their own (scope "user"), the marketplace catalogue they can
    publish to and install from, and the folders Deep Agents loads skills from.

    Folders load in order: seeded ``skills/``, then ``installed-skills/<user>/``, then
    ``user-skills/<user>/``. Later folders win on a name clash, so a user's own skill
    overrides an installed marketplace skill, which overrides a seeded example.
    """

    def __init__(self, store: SkillStore, owner: str, workspace: Path):
        self.store, self.owner = store, owner
        folder = _folder_name(owner)
        self.folder_rel = f"user-skills/{folder}/"
        self.installed_rel = f"installed-skills/{folder}/"
        self.folder = Path(workspace) / self.folder_rel
        self.installed_folder = Path(workspace) / self.installed_rel
        # Approved calls from one step run in parallel threads: apply them one at a time,
        # so each re-plans against the latest state and folder rewrites don't overlap.
        self.lock = threading.RLock()

    def _owner(self, scope):
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        return self.owner if scope == USER else ""

    def _live(self, name, scope):
        skill = self.store.get(scope, self._owner(scope), name)
        return skill if skill and not skill.removed else None

    # planning (no side effects) -------------------------------------------------
    def _change(self, action, name, after, errors=(), scope=USER):
        current = self.store.get(scope, self._owner(scope), name)
        live = current if current and not current.removed else None
        change = SkillChange(action, name, self.owner, scope,
                             before=live.files if live else None, after=after,
                             version_from=live.version if live else None,
                             version_to=(current.version + 1) if current else 1,
                             errors=list(errors))
        if after is not None and not change.errors:
            change.errors = validate(name, after)
            change.warnings = warnings(after)
        return change

    def plan_create(self, name, description, instructions, files=None, scope=USER):
        if self._live(name, scope):
            return self._change("create", name, None, [f"a {scope} skill named {name!r} already "
                                                       "exists; use update_skill"], scope)
        return self._change("create", name, {**(files or {}),
                                             "SKILL.md": render_skill_md(name, description, instructions)},
                            scope=scope)

    def plan_update(self, name, description=None, instructions=None, files=None, delete_files=None,
                    scope=USER):
        current = self._live(name, scope)
        if current is None:
            return self._change("update", name, None, [f"no {scope} skill named {name!r}; use "
                                                       "create_skill"], scope)
        after = dict(current.files)
        if description is not None or instructions is not None:
            front, body = split_skill_md(after["SKILL.md"])
            extra = {k: v for k, v in front.items() if k not in ("name", "description")}
            after["SKILL.md"] = render_skill_md(name, description or str(front.get("description", "")),
                                                instructions if instructions is not None else body, extra)
        after.update(files or {})
        for path in delete_files or []:
            if path == "SKILL.md":
                return self._change("update", name, None, ["SKILL.md can't be deleted; use remove_skill"],
                                    scope)
            after.pop(path, None)
        return self._change("update", name, after, scope=scope)

    def plan_remove(self, name, scope=USER):
        if self._live(name, scope) is None:
            return self._change("remove", name, None, [f"no {scope} skill named {name!r}"], scope)
        return self._change("remove", name, None, scope=scope)

    def plan_restore(self, name, version, scope=USER):
        old = self.store.version(scope, self._owner(scope), name, int(version))
        if old is None:
            return self._change("restore", name, None, [f"{name!r} has no version {version}"], scope)
        return self._change("restore", name, dict(old.files), scope=scope)

    def plan_import(self, path, scope=USER):
        try:
            files = read_skill_folder(Path(path))
        except ValueError as error:
            return SkillChange("import", str(path), self.owner, scope, errors=[str(error)])
        name = str(split_skill_md(files["SKILL.md"])[0].get("name", "")).strip()
        if not name:
            return SkillChange("import", str(path), self.owner, scope,
                               errors=["SKILL.md needs a 'name' in its frontmatter"])
        return self._change("import", name, files, scope=scope)

    def plan_publish(self, name):
        """Copy one of the user's skills to the marketplace (a new skill or a new version)."""
        mine = self._live(name, USER)
        if mine is None:
            return self._change("publish", name, None, [f"no user skill named {name!r} to publish"],
                                MARKETPLACE)
        return self._change("publish", name, dict(mine.files), scope=MARKETPLACE)

    def plan_install(self, name):
        listed = self._live(name, MARKETPLACE)
        if listed is None:
            return SkillChange("install", name, self.owner, MARKETPLACE,
                               errors=[f"the marketplace has no skill named {name!r}; see browse_marketplace"])
        if name in self.store.installed(self.owner):
            return SkillChange("install", name, self.owner, MARKETPLACE,
                               errors=[f"{name!r} is already installed"])
        change = SkillChange("install", name, self.owner, MARKETPLACE, after=dict(listed.files),
                             version_to=listed.version)
        change.errors, change.warnings = validate(name, listed.files), warnings(listed.files)
        return change

    def plan_uninstall(self, name):
        if name not in self.store.installed(self.owner):
            return SkillChange("uninstall", name, self.owner, MARKETPLACE,
                               errors=[f"{name!r} is not installed"])
        listed = self.store.get(MARKETPLACE, "", name)
        return SkillChange("uninstall", name, self.owner, MARKETPLACE,
                           before=dict(listed.files) if listed else {},
                           version_from=listed.version if listed else None)

    def plan(self, tool_name, args) -> SkillChange | None:
        """The change a gated tool call would make, or None for other tools."""
        planners = {"create_skill": self.plan_create, "update_skill": self.plan_update,
                    "remove_skill": self.plan_remove, "restore_skill": self.plan_restore,
                    "import_skill": self.plan_import, "publish_skill": self.plan_publish,
                    "install_skill": self.plan_install, "uninstall_skill": self.plan_uninstall}
        planner = planners.get(tool_name)
        if planner is None:
            return None
        try:
            return planner(**args)
        except (TypeError, ValueError) as error:  # arguments the model got wrong
            return SkillChange(tool_name.removesuffix("_skill"), str(args.get("name", "?")),
                               self.owner, errors=[f"bad arguments: {error}"])

    # applying ---------------------------------------------------------------------
    def apply(self, change: SkillChange) -> dict:
        with self.lock:
            return self._apply(change)

    def _apply(self, change: SkillChange) -> dict:
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
        self.sync_folder()
        return {**result, "note": "The change is available from the user's next message."}

    def sync_folder(self):
        """Rewrite the user's skill folders from the database (the source of truth)."""
        with self.lock:
            self._sync_folder()

    def _sync_folder(self):
        installed = [self._live(name, MARKETPLACE) for name in self.store.installed(self.owner)]
        for folder, skills in ((self.folder, self.store.list_current(USER, self.owner)),
                               (self.installed_folder, [s for s in installed if s])):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            for skill in skills:
                for path, text in skill.files.items():
                    target = folder / skill.name / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8", newline="\n")

    # read-only views ----------------------------------------------------------------
    def browse(self, query=""):
        installed = set(self.store.installed(self.owner))
        query = query.lower().strip()
        found = []
        for skill in self.store.list_current(MARKETPLACE, ""):
            description = str(split_skill_md(skill.files["SKILL.md"])[0].get("description", ""))
            if query and query not in skill.name.lower() and query not in description.lower():
                continue
            found.append({"name": skill.name, "description": description, "version": skill.version,
                          "installed": skill.name in installed})
        return found

    def overview(self):
        mine = {s.name: s for s in self.store.list_current(USER, self.owner)}
        rows = [{"name": name, "source": "user", "version": skill.version}
                for name, skill in sorted(mine.items())]
        for name in self.store.installed(self.owner):
            listed = self._live(name, MARKETPLACE)
            rows.append({"name": name, "source": "marketplace (installed)",
                         "version": listed.version if listed else None,
                         "note": ("removed from the marketplace" if listed is None else
                                  "overridden by your own skill with the same name" if name in mine
                                  else "")})
        return rows

    # the agent's tools ----------------------------------------------------------------
    def tools(self):
        def done(plan, *args):
            # Plan and apply under one lock, so parallel calls see each other's results.
            with self.lock:
                return json.dumps(self._apply(plan(*args)), default=str)

        def create_skill(name: str, description: str, instructions: str,
                         files: dict[str, str] | None = None, scope: str = "user") -> str:
            """Create a new skill (only when the user asks). name: lowercase words joined by
            hyphens. description: what the skill does and when to use it. instructions: the
            SKILL.md body, as clear steps. files: optional extra files, relative path -> text.
            scope: "user" (default) or "marketplace" (shared with everyone; only if asked).
            The user approves the change before it is saved."""
            return done(self.plan_create, name, description, instructions, files, scope)

        def update_skill(name: str, description: str | None = None, instructions: str | None = None,
                         files: dict[str, str] | None = None, delete_files: list[str] | None = None,
                         scope: str = "user") -> str:
            """Update a skill (only when asked). Give only what changes: description,
            instructions (the whole new SKILL.md body), files to add or replace, or
            delete_files. scope: "user" (default) or "marketplace". Needs approval."""
            return done(self.plan_update, name, description, instructions, files, delete_files, scope)

        def remove_skill(name: str, scope: str = "user") -> str:
            """Remove a skill (only when asked); its history is kept. scope: "user" (default)
            or "marketplace" (removes it for everyone who installed it). Needs approval."""
            return done(self.plan_remove, name, scope)

        def import_skill(path: str, scope: str = "user") -> str:
            """Import a skill from a SKILL.md file or a folder containing one, on the user's
            machine (only when asked). scope: "user" (default) or "marketplace". Needs approval."""
            return done(self.plan_import, path, scope)

        def restore_skill(name: str, version: int, scope: str = "user") -> str:
            """Restore an earlier version of a skill (see skill_history). Needs approval."""
            return done(self.plan_restore, name, version, scope)

        def publish_skill(name: str) -> str:
            """Publish one of the user's skills to the marketplace, as a new skill or a new
            version (only when asked). Needs approval."""
            return done(self.plan_publish, name)

        def install_skill(name: str) -> str:
            """Install a marketplace skill for this user (only when asked; see
            browse_marketplace). It follows the marketplace's latest version. Needs approval."""
            return done(self.plan_install, name)

        def uninstall_skill(name: str) -> str:
            """Uninstall a marketplace skill for this user (only when asked). Needs approval."""
            return done(self.plan_uninstall, name)

        def browse_marketplace(query: str = "") -> str:
            """Search the marketplace catalogue (empty query lists everything)."""
            return json.dumps(self.browse(query))

        def list_skills() -> str:
            """The user's own skills and installed marketplace skills, with versions."""
            return json.dumps(self.overview())

        def skill_history(name: str, scope: str = "user") -> str:
            """List the saved versions of a skill. scope: "user" (default) or "marketplace"."""
            try:
                versions = self.store.history(scope, self._owner(scope), name)
            except ValueError as error:
                return json.dumps({"error": str(error)})
            if not versions:
                return json.dumps({"error": f"no {scope} skill named {name!r}"})
            return json.dumps([{"version": v.version, "action": v.action, "saved": v.created_at,
                                "files": sorted(v.files)} for v in versions])

        return [create_skill, update_skill, remove_skill, import_skill, restore_skill, publish_skill,
                install_skill, uninstall_skill, browse_marketplace, list_skills, skill_history]

    def feature(self) -> Feature:
        for folder in (self.folder, self.installed_folder):
            folder.mkdir(parents=True, exist_ok=True)  # Deep Agents needs the sources to exist
        prompt = ("Skills are reusable instructions (a SKILL.md, plus optional files). The user "
                  "has their own skills and can install skills from a shared marketplace "
                  "catalogue. Change skills only when the user asks: create_skill, update_skill, "
                  "remove_skill, import_skill, restore_skill (scope 'user' by default, "
                  "'marketplace' only if they ask for the shared catalogue), publish_skill to "
                  "share one of theirs, install_skill and uninstall_skill for marketplace skills. "
                  "browse_marketplace, list_skills and skill_history only read. Each change is "
                  "shown to the user for approval first; if it is rejected, don't retry unless "
                  "the user asks. The tools write the SKILL.md format themselves, so don't read "
                  "other skills first. File tools can't write to skill folders. A change is "
                  "available from the next message. ")
        return Feature(name="skills", prompt=prompt, tools=tuple(self.tools()),
                       skill_dirs=(self.installed_rel, self.folder_rel), approvals=GATED_TOOLS)


def read_skill_folder(path: Path) -> dict:
    """The text files of a skill, from a SKILL.md file or the folder that holds one."""
    folder = path.parent if path.name == "SKILL.md" else path
    if not (folder / "SKILL.md").is_file():
        raise ValueError(f"no SKILL.md found at {path}")
    files, total = {}, 0
    for item in sorted(folder.rglob("*")):
        relative = item.relative_to(folder)
        if not item.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if len(files) >= MAX_FILES:
            raise ValueError(f"the skill has more than {MAX_FILES} files")
        try:
            text = item.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{relative.as_posix()} is not a text file") from None
        total += len(text)
        if total > MAX_TOTAL_SIZE:
            raise ValueError(f"the skill's files exceed {MAX_TOTAL_SIZE} characters")
        files[relative.as_posix()] = text.replace("\r\n", "\n")
    return files

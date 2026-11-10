"""Agent-written skills: plan a change, show it for approval, then save it.

``SkillManager`` serves one user. Each tool call becomes a ``SkillChange`` (create,
update, remove, import or restore) that knows its validation errors, a summary, a
diff and the full files. The CLI shows a change before the user approves it, and
the tool applies the same change afterwards, so what is approved is what is saved.

Every saved version goes to ``SkillStore``; the user's current skills are then
written to ``<workspace>/user-skills/<user>/``, where Deep Agents loads them.
"""
import difflib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml
from deepagents.middleware.skills import (MAX_SKILL_DESCRIPTION_LENGTH, MAX_SKILL_FILE_SIZE,
                                          _FRONTMATTER_PATTERN, _parse_skill_metadata,
                                          _validate_skill_name)

from deep_agents_demo.agent import Feature
from deep_agents_demo.memory_policy import find_secrets, split_findings
from deep_agents_demo.skill_store import SkillStore

SCOPE = "user"            # the marketplace scope arrives with the global store
MAX_FILES = 50
MAX_TOTAL_SIZE = 1_000_000
GATED_TOOLS = ("create_skill", "update_skill", "remove_skill", "import_skill", "restore_skill")


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
    action: str                       # create, update, remove, import, restore
    name: str
    owner: str
    before: dict | None = None        # current files, if the skill exists
    after: dict | None = None         # files after the change (None for remove)
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

    def summary(self) -> str:
        versions = (f"v{self.version_from} -> v{self.version_to}" if self.version_from
                    else f"new, v{self.version_to}")
        lines = [f"Skill change requested: {self.action.upper()} {self.name!r} "
                 f"(user: {self.owner}), {versions}"]
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
    """One user's skills: plan changes, apply approved ones, keep the folder in sync."""

    def __init__(self, store: SkillStore, owner: str, workspace: Path):
        self.store, self.owner = store, owner
        self.folder_rel = f"user-skills/{_folder_name(owner)}/"
        self.folder = Path(workspace) / self.folder_rel

    # planning (no side effects) -------------------------------------------------
    def _current(self, name):
        skill = self.store.get(SCOPE, self.owner, name)
        return skill if skill and not skill.removed else None

    def _change(self, action, name, after, errors=()):
        current = self.store.get(SCOPE, self.owner, name)
        live = current if current and not current.removed else None
        change = SkillChange(action, name, self.owner,
                             before=live.files if live else None, after=after,
                             version_from=live.version if live else None,
                             version_to=(current.version + 1) if current else 1,
                             errors=list(errors))
        if after is not None and not change.errors:
            change.errors = validate(name, after)
            change.warnings = warnings(after)
        return change

    def plan_create(self, name, description, instructions, files=None):
        if self._current(name):
            return self._change("create", name, None,
                                [f"a skill named {name!r} already exists; use update_skill"])
        return self._change("create", name, {**(files or {}),
                                             "SKILL.md": render_skill_md(name, description, instructions)})

    def plan_update(self, name, description=None, instructions=None, files=None, delete_files=None):
        current = self._current(name)
        if current is None:
            return self._change("update", name, None, [f"no skill named {name!r}; use create_skill"])
        after = dict(current.files)
        if description is not None or instructions is not None:
            front, body = split_skill_md(after["SKILL.md"])
            extra = {k: v for k, v in front.items() if k not in ("name", "description")}
            after["SKILL.md"] = render_skill_md(name, description or str(front.get("description", "")),
                                                instructions if instructions is not None else body, extra)
        after.update(files or {})
        for path in delete_files or []:
            if path == "SKILL.md":
                return self._change("update", name, None, ["SKILL.md can't be deleted; use remove_skill"])
            after.pop(path, None)
        return self._change("update", name, after)

    def plan_remove(self, name):
        if self._current(name) is None:
            return self._change("remove", name, None, [f"no skill named {name!r}"])
        return self._change("remove", name, None)

    def plan_restore(self, name, version):
        old = self.store.version(SCOPE, self.owner, name, int(version))
        if old is None:
            return self._change("restore", name, None, [f"{name!r} has no version {version}"])
        return self._change("restore", name, dict(old.files))

    def plan_import(self, path):
        try:
            files = read_skill_folder(Path(path))
        except ValueError as error:
            return SkillChange("import", str(path), self.owner, errors=[str(error)])
        name = str(split_skill_md(files["SKILL.md"])[0].get("name", "")).strip()
        if not name:
            return SkillChange("import", str(path), self.owner,
                               errors=["SKILL.md needs a 'name' in its frontmatter"])
        return self._change("import", name, files)

    def plan(self, tool_name, args) -> SkillChange | None:
        """The change a gated tool call would make, or None for other tools."""
        planners = {"create_skill": self.plan_create, "update_skill": self.plan_update,
                    "remove_skill": self.plan_remove, "restore_skill": self.plan_restore,
                    "import_skill": self.plan_import}
        planner = planners.get(tool_name)
        if planner is None:
            return None
        try:
            return planner(**args)
        except TypeError as error:  # arguments the model got wrong
            return SkillChange(tool_name.removesuffix("_skill"), str(args.get("name", "?")),
                               self.owner, errors=[f"bad arguments: {error}"])

    # applying ---------------------------------------------------------------------
    def apply(self, change: SkillChange) -> dict:
        if change.errors:
            return {"saved": False, "errors": change.errors}
        if change.action == "remove":
            version = self.store.remove(SCOPE, self.owner, change.name)
        else:
            version = self.store.save(SCOPE, self.owner, change.name, change.after, change.action)
        self.sync_folder()
        return {"saved": True, "action": change.action, "name": change.name, "version": version,
                "note": "The change is available from the user's next message."}

    def sync_folder(self):
        """Rewrite the user's skill folder from the database (the source of truth)."""
        if self.folder.exists():
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True)
        for skill in self.store.list_current(SCOPE, self.owner):
            for path, text in skill.files.items():
                target = self.folder / skill.name / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8", newline="\n")

    # the agent's tools ----------------------------------------------------------------
    def tools(self):
        def done(change):
            return json.dumps(self.apply(change), default=str)

        def create_skill(name: str, description: str, instructions: str,
                         files: dict[str, str] | None = None) -> str:
            """Create a new skill (only when the user asks). name: lowercase words joined by
            hyphens. description: what the skill does and when to use it. instructions: the
            SKILL.md body, as clear steps. files: optional extra files, relative path -> text.
            The user approves the change before it is saved."""
            return done(self.plan_create(name, description, instructions, files))

        def update_skill(name: str, description: str | None = None, instructions: str | None = None,
                         files: dict[str, str] | None = None,
                         delete_files: list[str] | None = None) -> str:
            """Update one of the user's skills (only when asked). Give only what changes:
            description, instructions (the whole new SKILL.md body), files to add or replace,
            or delete_files. The user approves the change before it is saved."""
            return done(self.plan_update(name, description, instructions, files, delete_files))

        def remove_skill(name: str) -> str:
            """Remove one of the user's skills (only when asked). Its history is kept, so it
            can be restored. The user approves the change first."""
            return done(self.plan_remove(name))

        def import_skill(path: str) -> str:
            """Import a skill from a SKILL.md file or a folder containing one, on the user's
            machine (only when asked). The user approves the change first."""
            return done(self.plan_import(path))

        def restore_skill(name: str, version: int) -> str:
            """Restore an earlier version of a skill (see skill_history). The user approves
            the change first."""
            return done(self.plan_restore(name, version))

        def skill_history(name: str) -> str:
            """List the saved versions of one of the user's skills."""
            versions = self.store.history(SCOPE, self.owner, name)
            if not versions:
                return json.dumps({"error": f"no skill named {name!r}"})
            return json.dumps([{"version": v.version, "action": v.action, "saved": v.created_at,
                                "files": sorted(v.files)} for v in versions])

        return [create_skill, update_skill, remove_skill, import_skill, restore_skill, skill_history]

    def feature(self) -> Feature:
        self.folder.mkdir(parents=True, exist_ok=True)  # Deep Agents needs the source to exist
        prompt = ("Skills are reusable instructions (a SKILL.md, plus optional files). Create, "
                  "update, remove, import or restore the user's skills only when the user asks, "
                  "with create_skill, update_skill, remove_skill, import_skill and restore_skill; "
                  "skill_history lists versions. Each change is shown to the user for approval "
                  "first; if it is rejected, don't retry unless the user asks. File tools can't "
                  "write to skill folders. A saved skill is available from the next message. ")
        return Feature(name="skills", prompt=prompt, tools=tuple(self.tools()),
                       skill_dirs=(self.folder_rel,), approvals=GATED_TOOLS)


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

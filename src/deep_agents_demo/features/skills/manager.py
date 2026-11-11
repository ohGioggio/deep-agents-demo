"""Agent-written skills: plan a change, show it for approval, then save it.

``SkillManager`` serves one user on top of ``core.items.ItemManager``. Every saved
version goes to ``VersionedStore``. The user's current skills are written to
``<workspace>/user-skills/<user>/`` and the marketplace skills they installed to
``<workspace>/installed-skills/<user>/``, where Deep Agents loads them.
"""
import json
import shutil
from pathlib import Path

from deep_agents_demo.core.feature import Feature
from deep_agents_demo.core.items import MARKETPLACE, USER, ItemChange, ItemManager
from deep_agents_demo.features.skills.format import (
    declared_tools,
    read_skill_folder,
    render_skill_md,
    split_skill_md,
    validate,
    warnings,
)
from deep_agents_demo.persistence.files import safe_folder_name
from deep_agents_demo.persistence.versioned_store import VersionedStore

GATED_TOOLS = ("create_skill", "update_skill", "remove_skill", "import_skill", "restore_skill",
               "publish_skill", "install_skill", "uninstall_skill")


class SkillChange(ItemChange):
    noun = "Skill"
    main_file = "SKILL.md"

    def tools_used(self) -> list[str]:
        return declared_tools(self.after if self.after is not None else self.before or {})


class SkillManager(ItemManager):
    """One user's skills: their own (scope "user"), the marketplace catalogue they can
    publish to and install from, and the folders Deep Agents loads skills from.

    Folders load in order: seeded ``skills/``, then ``installed-skills/<user>/``, then
    ``user-skills/<user>/``. Later folders win on a name clash, so a user's own skill
    overrides an installed marketplace skill, which overrides a seeded example.
    """

    noun = "skill"
    change_type = SkillChange
    main_file = "SKILL.md"
    browse_tool = "browse_marketplace"

    def __init__(self, store: VersionedStore, owner: str, workspace: Path):
        super().__init__(store, owner)
        folder = safe_folder_name(owner)
        self.folder_rel = f"user-skills/{folder}/"
        self.installed_rel = f"installed-skills/{folder}/"
        self.folder = Path(workspace) / self.folder_rel
        self.installed_folder = Path(workspace) / self.installed_rel

    # hooks ------------------------------------------------------------------------
    def validate(self, name, files) -> list[str]:
        return validate(name, files)

    def warnings(self, files) -> list[str]:
        return warnings(files, self.known_tools)

    def after_apply(self) -> None:
        self.sync_folder()

    # planning (no side effects) -------------------------------------------------
    def plan_create(self, name, description, instructions, files=None, scope=USER, tools=None):
        if self._live(name, scope):
            return self._change("create", name, None, [f"a {scope} {self.noun} named {name!r} already "
                                                       f"exists; use update_{self.noun}"], scope)
        extra = {"allowed-tools": " ".join(dict.fromkeys(tools))} if tools else None
        after = {**(files or {}), "SKILL.md": render_skill_md(name, description, instructions, extra)}
        return self._change("create", name, after, self._tool_errors(tools), scope=scope)

    def plan_update(self, name, description=None, instructions=None, files=None, delete_files=None,
                    scope=USER, tools=None):
        current = self._live(name, scope)
        if current is None:
            return self._change("update", name, None, [f"no {scope} {self.noun} named {name!r}; use "
                                                       f"create_{self.noun}"], scope)
        after = dict(current.files)
        if description is not None or instructions is not None or tools is not None:
            front, body = split_skill_md(after["SKILL.md"])
            extra = {k: v for k, v in front.items() if k not in ("name", "description")}
            if tools is not None:  # [] removes the declaration
                extra.pop("allowed-tools", None)
                if tools:
                    extra["allowed-tools"] = " ".join(dict.fromkeys(tools))
            after["SKILL.md"] = render_skill_md(name, description or str(front.get("description", "")),
                                                instructions if instructions is not None else body, extra)
        after.update(files or {})
        for path in delete_files or []:
            if path == "SKILL.md":
                return self._change("update", name, None, ["SKILL.md can't be deleted; use remove_skill"],
                                    scope)
            after.pop(path, None)
        return self._change("update", name, after, self._tool_errors(tools), scope=scope)

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


    # folders ----------------------------------------------------------------------
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

    # the agent's tools ----------------------------------------------------------------
    def tools(self):
        def done(plan, *args, **kwargs):
            # Plan and apply under one lock, so parallel calls see each other's results.
            with self.lock:
                return json.dumps(self._apply(plan(*args, **kwargs)), default=str)

        def create_skill(name: str, description: str, instructions: str,
                         files: dict[str, str] | None = None, scope: str = "user",
                         tools: list[str] | None = None) -> str:
            """Create a new skill (only when the user asks). name: lowercase words joined by
            hyphens. description: what the skill does and when to use it. instructions: the
            SKILL.md body, as clear steps. files: optional extra files, relative path -> text.
            scope: "user" (default) or "marketplace" (shared with everyone; only if asked).
            tools: names of existing tools the instructions use (saved as allowed-tools).
            The user approves the change before it is saved."""
            return done(self.plan_create, name, description, instructions, files, scope, tools=tools)

        def update_skill(name: str, description: str | None = None, instructions: str | None = None,
                         files: dict[str, str] | None = None, delete_files: list[str] | None = None,
                         scope: str = "user", tools: list[str] | None = None) -> str:
            """Update a skill (only when asked). Give only what changes: description,
            instructions (the whole new SKILL.md body), files to add or replace,
            delete_files, or tools (the full new list of tools it uses; [] for none).
            scope: "user" (default) or "marketplace". Needs approval."""
            return done(self.plan_update, name, description, instructions, files, delete_files, scope,
                        tools=tools)

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
                return json.dumps({"error": f"no {scope} {self.noun} named {name!r}"})
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
                  "other skills first. When a skill's steps use specific tools, list them in "
                  "tools (existing tools only). File tools can't write to skill folders. A change is "
                  "available from the next message. ")
        return Feature(name="skills", prompt=prompt, tools=tuple(self.tools()),
                       skill_dirs=(self.installed_rel, self.folder_rel), approvals=GATED_TOOLS)

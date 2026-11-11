"""The SKILL.md format: rendering, reading, checking a skill's files."""
import re
from pathlib import Path, PurePosixPath

import yaml
from deepagents.middleware.skills import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    _parse_skill_metadata,
    _validate_skill_name,
)

from deep_agents_demo.core.items import split_frontmatter
from deep_agents_demo.core.secrets import find_secrets, split_findings

MAX_FILES = 50
MAX_TOTAL_SIZE = 1_000_000


def render_skill_md(name: str, description: str, instructions: str, extra: dict | None = None) -> str:
    """A SKILL.md in Deep Agents' format: YAML frontmatter, then the instructions."""
    front = {"name": name, "description": " ".join(description.split()), **(extra or {})}
    return ("---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10**9)
            + "---\n\n" + instructions.strip() + "\n")



def split_skill_md(content: str) -> tuple[dict, str]:
    """(frontmatter, instructions) of a SKILL.md; ({}, content) if it has no frontmatter."""
    return split_frontmatter(content)


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


def declared_tools(files: dict) -> list[str]:
    """The tool names a skill's SKILL.md lists under ``allowed-tools``."""
    raw = split_skill_md(str(files.get("SKILL.md", "")))[0].get("allowed-tools") or ""
    items = raw if isinstance(raw, list) else re.split(r"[\s,]+", str(raw))
    return [str(item).strip() for item in items if str(item).strip()]


def warnings(files: dict, known_tools=None) -> list[str]:
    """Shown to the user, who decides: wording that might reveal a secret, and tools
    the skill lists that this agent doesn't have (e.g. an imported skill)."""
    found = []
    for path, text in sorted(files.items()):
        _credentials, wording = split_findings(find_secrets(str(text)))
        if wording:
            found.append(f"{path} mentions a password, token or key; check it holds no real value")
    if known_tools is not None:
        unknown = [name for name in declared_tools(files) if name not in known_tools]
        if unknown:
            found.append(f"allowed-tools lists tools this agent doesn't have: {', '.join(unknown)}")
    return found


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

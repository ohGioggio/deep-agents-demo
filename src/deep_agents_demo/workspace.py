"""Seed the local agent workspace with starter memory and skills."""
import hashlib
import shutil
from pathlib import Path

# Files under seed/ are copied into the workspace with the same relative paths.
SEED_DIR = Path(__file__).resolve().parent / "seed"

# Seed files that moved. Each old copy is removed only if it is still identical
# to its replacement, so a user's edits are never deleted.
MOVED = {"skills/sandbox/SKILL.md": "sandbox-skills/sandbox/SKILL.md"}

# Earlier versions of seed files (SHA-256 of the text with LF line endings). A
# workspace copy that still matches one was never edited, so it is upgraded to
# the current seed; any edited copy is left alone.
SUPERSEDED = {
    # "Use metres. Prefer realistic assets." test preferences.
    "memories/AGENTS.md": {"a3afad1c5f45a4f98d44599a28712dafd13e47d36ab3f65f0c150a555efacc9a"},
    # Scene-planning skill that used the removed asset-researcher subagent.
    "skills/scene-planning/SKILL.md": {"f3c5e9a23b13fb33bb9ad2a3dde67d6b8a4f3ef2df4c1c50b17048cd7492ae16"},
}


def _text(path: Path) -> bytes:
    """File content with LF line endings: older seeding wrote CRLF on Windows."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def seed_files(workspace: Path) -> None:
    """Copy each seed file into the workspace once; preserve later edits."""
    for old, new in MOVED.items():
        old_path, source = workspace / old, SEED_DIR / new
        if old_path.is_file() and _text(old_path) == _text(source):
            old_path.unlink()
            if not any(old_path.parent.iterdir()):
                old_path.parent.rmdir()
    for relative, hashes in SUPERSEDED.items():
        target = workspace / relative
        if target.is_file() and hashlib.sha256(_text(target)).hexdigest() in hashes:
            shutil.copyfile(SEED_DIR / relative, target)
    for source in sorted(SEED_DIR.rglob("*")):
        if not source.is_file():
            continue
        target = workspace / source.relative_to(SEED_DIR)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

"""Seed the local agent workspace with starter memory and skills."""
import shutil
from pathlib import Path

# Files under seed/ are copied into the workspace with the same relative paths.
SEED_DIR = Path(__file__).resolve().parent / "seed"


# Seed files that moved. Each old copy is removed only if it is still identical
# to its replacement, so a user's edits are never deleted.
MOVED = {"skills/sandbox/SKILL.md": "sandbox-skills/sandbox/SKILL.md"}


def seed_files(workspace: Path) -> None:
    """Copy each seed file into the workspace once; preserve later edits."""
    for old, new in MOVED.items():
        old_path, source = workspace / old, SEED_DIR / new
        if old_path.is_file() and old_path.read_bytes() == source.read_bytes():
            old_path.unlink()
            if not any(old_path.parent.iterdir()):
                old_path.parent.rmdir()
    for source in sorted(SEED_DIR.rglob("*")):
        if not source.is_file():
            continue
        target = workspace / source.relative_to(SEED_DIR)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

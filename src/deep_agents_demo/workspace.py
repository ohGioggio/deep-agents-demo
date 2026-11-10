"""Seed the local agent workspace with starter memory and skills."""
import shutil
from pathlib import Path

# Files under seed/ are copied into the workspace with the same relative paths.
SEED_DIR = Path(__file__).resolve().parent / "seed"


def seed_files(workspace: Path) -> None:
    """Copy each seed file into the workspace once; preserve later edits."""
    for source in sorted(SEED_DIR.rglob("*")):
        if not source.is_file():
            continue
        target = workspace / source.relative_to(SEED_DIR)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

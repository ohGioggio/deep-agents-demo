"""Seed the local agent workspace with starter memory and a skill."""
from pathlib import Path

SEED_FILES = {
    "memories/AGENTS.md": "# User preferences\nUse metres. Prefer realistic assets.\n",
    "skills/scene-planning/SKILL.md": """---
name: scene-planning
description: Plan a small 3D city scene using the fictional asset catalog and a triangle budget.
---
1. Create a short to-do plan using write_todos.
2. Ask the asset-researcher subagent to find tree, car, and building options.
3. Choose quantities and calculate totals using triangle_budget.
4. If over the requested budget, revise quantities and calculate again.
5. Save the final proposal to /scene_plan.md, including IDs, counts and total.
6. Mark completed tasks and summarize the result. This is a plan, not a rendered scene.
""",
}


def seed_files(workspace: Path) -> None:
    """Create the starter files once; preserve later edits."""
    for relative, content in SEED_FILES.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")

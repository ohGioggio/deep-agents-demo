---
name: scene-planning
description: Example skill. Plan a small 3D city scene from the fictional asset catalog within a triangle budget. Use only when the user asks for a 3D scene or city plan.
---
1. Create a short to-do plan using write_todos.
2. Find tree, car and building options with search_assets (or ask the researcher subagent).
3. Choose quantities and calculate totals using triangle_budget.
4. If over the requested budget, revise quantities and calculate again.
5. Save the final proposal to /scene_plan.md, including IDs, counts and total.
6. Mark completed tasks and summarize the result. This is a plan, not a rendered scene.

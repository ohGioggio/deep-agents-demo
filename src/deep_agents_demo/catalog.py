"""Fictional 3D asset catalog and the tools that query it."""
import json

# Deliberately fictional data, so this example needs only a model API key.
CATALOG = [
    {"id": "oak", "category": "tree", "style": "realistic", "triangles": 12000},
    {"id": "pine", "category": "tree", "style": "realistic", "triangles": 8000},
    {"id": "sedan", "category": "car", "style": "realistic", "triangles": 18000},
    {"id": "tower", "category": "building", "style": "realistic", "triangles": 24000},
]


def search_assets(category: str) -> str:
    """Search the fictional catalog by category: tree, car, or building."""
    return json.dumps([a for a in CATALOG if a["category"] == category.lower()])


def triangle_budget(asset_ids: list[str], copies: list[int]) -> str:
    """Calculate triangle totals for catalog assets and their instance counts."""
    if len(asset_ids) != len(copies) or any(n < 0 for n in copies):
        return "Error: provide one nonnegative count per asset ID."
    by_id = {a["id"]: a for a in CATALOG}
    if any(i not in by_id for i in asset_ids):
        return "Error: unknown asset ID; search the catalog first."
    total = sum(by_id[i]["triangles"] * n for i, n in zip(asset_ids, copies))
    return json.dumps({"triangles": total})

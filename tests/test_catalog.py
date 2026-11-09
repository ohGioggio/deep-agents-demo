"""Catalog tool tests."""
import json
import unittest

from deep_agents_demo.catalog import search_assets, triangle_budget


class CatalogTests(unittest.TestCase):
    def test_search_assets_by_category(self):
        ids = [a["id"] for a in json.loads(search_assets("Tree"))]
        self.assertEqual(ids, ["oak", "pine"])

    def test_triangle_budget_total(self):
        result = json.loads(triangle_budget(["pine", "tower"], [3, 2]))
        self.assertEqual(result, {"triangles": 3 * 8000 + 2 * 24000})

    def test_triangle_budget_errors(self):
        self.assertTrue(triangle_budget(["pine"], [1, 2]).startswith("Error"))
        self.assertTrue(triangle_budget(["pine"], [-1]).startswith("Error"))
        self.assertTrue(triangle_budget(["castle"], [1]).startswith("Error"))


if __name__ == "__main__":
    unittest.main()

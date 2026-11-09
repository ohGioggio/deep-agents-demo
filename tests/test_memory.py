"""Memory tool tests: user identity comes from the application, not the model."""
import unittest
from unittest.mock import Mock

from deep_agents_demo.memory import memory_tools


class MemoryToolTests(unittest.TestCase):
    def test_memory_user_scope(self):
        mem = Mock()
        mem.add.return_value = mem.search.return_value = {"results": []}
        remember, recall = memory_tools(mem, "alice")
        remember("Use uv")
        recall("preferences")
        self.assertEqual(mem.add.call_args.kwargs["user_id"], "alice")
        self.assertEqual(mem.search.call_args.kwargs["filters"], {"user_id": "alice"})


if __name__ == "__main__":
    unittest.main()

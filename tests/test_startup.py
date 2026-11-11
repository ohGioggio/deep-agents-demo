"""Start-up: what is opened once per process, and the per-user runtimes on top."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from test_agents import Model

from deep_agents_demo.config import ChatSettings, MemorySettings, Paths, Settings
from deep_agents_demo.features.memory.service import MemoryService
from deep_agents_demo.startup import resources as resources_module
from deep_agents_demo.startup import runtime as runtime_module
from deep_agents_demo.startup.resources import Resources
from deep_agents_demo.startup.runtime import build_runtime
from deep_agents_demo.startup.workspace import prepare_workspace


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.settings = Settings(
            chat=ChatSettings(api_key="unused", model="unused", base_url=None),
            memory=MemorySettings(enabled=True), rag_enabled=False,
            paths=Paths(data=self.root / "data", workspace=self.root / "ws",
                        checkpoints=self.root / "data" / "checkpoints.sqlite"))
        prepare_workspace(self.settings)
        self.memory = MagicMock(name="mem0")
        self.made = []
        self.closed = []

        def make_memory(*args, **kwargs):
            self.made.append(args)
            return self.memory
        patches = [patch.object(resources_module, "make_memory", side_effect=make_memory),
                   patch.object(resources_module, "close_memory", side_effect=self.closed.append),
                   patch.object(resources_module, "embeddings", return_value=MagicMock()),
                   patch.object(runtime_module, "chat_model",
                                return_value=Model(responses=[AIMessage(content="hi")]))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_two_users_share_one_memory_store(self):
        with Resources.open(self.settings) as resources, \
                patch.object(runtime_module, "MemoryService", wraps=MemoryService) as service:
            build_runtime(resources, "alice")
            build_runtime(resources, "bob")
        self.assertEqual(len(self.made), 1, "one Mem0 (one local Qdrant) per process")
        self.assertEqual([c.args for c in service.call_args_list],
                         [(self.memory, "alice"), (self.memory, "bob")])

    def test_resources_close_everything(self):
        resources = Resources.open(self.settings)
        build_runtime(resources, "alice")
        resources.close()
        self.assertEqual(self.closed, [self.memory])
        for store in (resources.skill_store, resources.agent_store):
            with self.assertRaises(sqlite3.ProgrammingError):
                store.db.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()

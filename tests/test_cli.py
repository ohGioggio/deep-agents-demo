"""Session behaviour of the CLI with a real Deep Agents graph and a fake model."""
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.core.turns import turn_input

SKILL = "---\nname: {name}\ndescription: The {name} skill.\n---\nBody.\n"


class ToolFreeFakeModel(FakeMessagesListChatModel):
    """Answers without calling tools; accepts the tools Deep Agents binds."""

    def bind_tools(self, tools, **kwargs):
        return self


def add_skill(root: Path, name: str) -> None:
    path = root / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SKILL.format(name=name), encoding="utf-8")


class SkillReloadTests(unittest.TestCase):
    def test_new_skills_reach_an_existing_thread_only_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memories").mkdir()
            (root / "memories/AGENTS.md").write_text("# Preferences\n", encoding="utf-8")
            add_skill(root, "scene-planning")
            model = ToolFreeFakeModel(responses=[AIMessage(content="ok")] * 3)
            agent = build_agent(model, InMemorySaver(),
                                FilesystemBackend(root_dir=str(root), virtual_mode=True))
            config = {"configurable": {"thread_id": "t"}}

            def skills():
                values = agent.get_state(config).values
                return sorted(s["name"] for s in values.get("skills_metadata") or [])

            agent.invoke(turn_input("first"), config)
            self.assertEqual(skills(), ["scene-planning"])

            add_skill(root, "sandbox")
            agent.invoke(turn_input("second"), config)
            self.assertEqual(skills(), ["scene-planning"], "Deep Agents caches skills per thread")

            agent.invoke(turn_input("third", reload_skills=True), config)
            self.assertEqual(skills(), ["sandbox", "scene-planning"])

    def test_turn_input_only_resets_skills_when_asked(self):
        self.assertNotIn("skills_metadata", turn_input("hi"))
        self.assertIsNone(turn_input("hi", reload_skills=True)["skills_metadata"])


if __name__ == "__main__":
    unittest.main()

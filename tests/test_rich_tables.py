"""The Rich tables for /threads, /files and /agents, and their hook into ChatSession."""
import io
import unittest
from unittest.mock import MagicMock

from rich.console import Console

from deep_agents_demo.persistence.chats import ChatInfo
from deep_agents_demo.persistence.outputs import OutputFile
from deep_agents_demo.ui.cli.repl import ChatSession
from deep_agents_demo.ui.cli.richui.tables import (
    RichTables,
    agents_table,
    chats_table,
    files_table,
)


def export(renderable) -> str:
    console = Console(file=io.StringIO(), width=100, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


CHATS = [
    ChatInfo(name="parks", turns=4, updated="2026-11-12T09:30:00", first="hi", last="Plan the parks"),
    ChatInfo(name="roads", turns=1, updated="", first="x", last="Roads?"),
]

FILES = [
    OutputFile(name="plan.md", version=2, bytes=2048, kind="md", caption="The park plan",
               saved_at="2026-11-12T09:31:00+00:00", path=None),
]

AGENTS = [
    {"name": "critic", "tools": "read_file, grep", "model": "configured", "source": "critic.md",
     "overrides": "", "description": "Finds weaknesses in a proposal."},
    {"name": "reviewer", "tools": "all", "model": "configured", "source": "built-in",
     "overrides": "built-in", "description": "Checks a result."},
]


class TablesTests(unittest.TestCase):
    def test_chats_table_has_headers_and_rows(self):
        text = export(chats_table(CHATS, current="parks"))
        for header in ("CHAT", "TURNS", "LAST USED", "LAST MESSAGE"):
            self.assertIn(header, text)
        self.assertIn("parks", text)
        self.assertIn("2026-11-12 09:30", text)
        self.assertIn("Plan the parks", text)
        current_row = next(line for line in text.splitlines() if "parks" in line)
        self.assertIn("*", current_row)
        self.assertNotIn("*", next(line for line in text.splitlines() if "roads" in line))

    def test_files_table_has_headers_and_rows(self):
        text = export(files_table(FILES, folder=None))
        for header in ("FILE", "VERSION", "SIZE", "SAVED", "CAPTION"):
            self.assertIn(header, text)
        for expected in ("plan.md", "v2", "2.0 KB", "2026-11-12 09:31", "The park plan"):
            self.assertIn(expected, text)

    def test_agents_table_has_headers_and_rows(self):
        text = export(agents_table(AGENTS))
        for header in ("AGENT", "TOOLS", "MODEL", "SOURCE", "DESCRIPTION"):
            self.assertIn(header, text)
        for expected in ("critic", "read_file, grep", "critic.md", "Finds weaknesses",
                         "reviewer", "(overrides"):
            self.assertIn(expected, text)

    def test_rich_tables_print_to_their_console(self):
        console = Console(file=io.StringIO(), width=100, record=True, color_system=None)
        tables = RichTables(console)
        tables.threads(CHATS, "parks")
        tables.files(FILES, None)
        tables.agents(AGENTS)
        text = console.export_text()
        for expected in ("LAST MESSAGE", "CAPTION", "DESCRIPTION", "plan.md", "critic"):
            self.assertIn(expected, text)

    def test_user_text_is_never_parsed_as_markup(self):
        chats = [ChatInfo(name="odd", turns=2, updated="", first="x", last="Use [/oops] and [bold] here")]
        files = [OutputFile(name="notes.md", version=1, bytes=10, kind="md", caption="Plan [bold] now [/x]",
                            saved_at="2026-11-12T09:31:00+00:00", path=None)]
        agents = [{"name": "odd", "tools": "read_file [/y]", "model": "configured", "source": "a [bold]",
                   "overrides": "[/z]", "description": "[red]d[/red]"}]
        console = Console(file=io.StringIO(), width=100, record=True, color_system=None)
        tables = RichTables(console)
        tables.threads(chats, None)
        tables.files(files, None)
        tables.agents(agents)
        text = console.export_text()
        for expected in ("[/oops]", "[bold]", "Plan [bold] now [/x]", "read_file [/y]", "a [bold]",
                         "[/z]", "[red]d[/red]"):
            self.assertIn(expected, text)

    def test_empty_lists_say_so_as_plain_text(self):
        console = Console(file=io.StringIO(), width=100, record=True, color_system=None)
        tables = RichTables(console)
        tables.threads([], None)
        tables.files([], None)
        text = console.export_text()
        self.assertIn("No chats yet.", text)
        self.assertIn("No files delivered in this chat yet.", text)


class ChatSessionTablesTests(unittest.TestCase):
    def fake_tables(self):
        tables = MagicMock()
        return tables

    def test_chat_session_uses_rich_tables(self):
        tables = self.fake_tables()
        chats = MagicMock()
        chats.list.return_value = CHATS
        chats.exists.return_value = True
        said = []
        ChatSession(chats, "parks", say=said.append, tables=tables).handle("/threads")
        tables.threads.assert_called_once_with(CHATS, "parks")
        self.assertEqual(said, [])

    def test_files_and_agents_use_rich_tables(self):
        tables = self.fake_tables()
        outputs = MagicMock()
        outputs.list.return_value = FILES
        outputs.folder.return_value = "folder"
        session = ChatSession(MagicMock(), "parks", say=[].append, outputs=outputs, tables=tables)
        session.handle("/files")
        tables.files.assert_called_once_with(FILES, "folder")
        agents = MagicMock()
        agents.overview.return_value = AGENTS
        ChatSession(MagicMock(), "parks", say=[].append, agents=agents, tables=tables).handle("/agents")
        tables.agents.assert_called_once_with(AGENTS)

    def test_new_chat_note_still_prints_with_tables(self):
        tables = self.fake_tables()
        chats = MagicMock()
        chats.list.return_value = CHATS
        chats.exists.return_value = False
        said = []
        ChatSession(chats, "draft", say=said.append, tables=tables).handle("/threads")
        tables.threads.assert_called_once_with(CHATS, "draft")
        self.assertTrue(any("draft" in line and "new" in line for line in said), said)

    def test_without_tables_the_text_is_unchanged(self):
        chats = MagicMock()
        chats.list.return_value = []
        chats.exists.return_value = True
        said = []
        ChatSession(chats, "parks", say=said.append).handle("/threads")
        self.assertEqual(said, ["No chats yet. Start one with: uv run deep-agents-demo --thread <name>"])


if __name__ == "__main__":
    unittest.main()

"""Files the agent delivers: validation, the send_file tool, events and chat commands."""
import json
import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agents import SEEN, Model, call, delegate

from deep_agents_demo.core.assembly import build_agent, worker_factory
from deep_agents_demo.features.files.feature import files_feature
from deep_agents_demo.features.files.validate import clean_name, describe, validate
from deep_agents_demo.persistence.chats import thread_id
from deep_agents_demo.persistence.outputs import OutputStore

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG = b"\xff\xd8\xff" + b"\x00" * 16


class ValidationTests(unittest.TestCase):
    def assertRefused(self, name, data, *words):
        error = validate(name, data)
        self.assertIsNotNone(error, f"{name} should be refused")
        for word in words:
            self.assertIn(word, error)

    def test_valid_files(self):
        for name, data in [("a.md", b"# Title\n"), ("a.txt", b"note"), ("a.tex", b"\\section{x}"),
                           ("a.json", b'{"a": 1}'), ("a.csv", b"a,b\n1,2\n"), ("a.html", b"<p>hi</p>"),
                           ("a.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
                           ("a.png", PNG), ("a.jpg", JPG), ("a.jpeg", JPG)]:
            with self.subTest(name=name):
                self.assertIsNone(validate(name, data))

    def test_refusals(self):
        self.assertRefused("report.docx", b"x", ".md", ".csv")
        self.assertRefused("a.md", b"", "empty")
        self.assertRefused("a.md", b"\xff\xfe\xfa", "UTF-8")
        self.assertRefused("a.md", b"x" * (5 * 1024 * 1024 + 1), "5 MB")
        self.assertRefused("a.png", PNG + b"x" * (20 * 1024 * 1024), "20 MB")
        self.assertRefused("a.csv", b"a,b\n1\n", "column")
        self.assertRefused("a.json", b"{", "JSON")
        self.assertRefused("a.svg", b"<html/>", "<svg>")
        self.assertRefused("a.svg", b"<svg", "SVG")
        self.assertRefused("a.html", b"just text", "HTML")
        self.assertRefused("a.png", JPG, "PNG")
        self.assertRefused("a.jpg", b"not an image", "JPEG")
        self.assertRefused("keys.md", b"key: AKIAABCDEFGHIJKLMNOP", "credential")

    def test_names_are_sanitised(self):
        self.assertEqual(clean_name("/sandbox/Report Final.MD"), "Report-Final.md")
        self.assertEqual(clean_name("..\\..\\x.md"), "x.md")
        self.assertEqual(clean_name("C:/a/b.Csv"), "b.csv")
        for bad in (".md", "noext", "..", "dir/"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                clean_name(bad)

    def test_leading_dots_are_refused(self):
        for bad in (".index.json", ".versions.md", "..hidden.md"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                clean_name(bad)

    def test_describe(self):
        self.assertEqual(describe("t.csv", b"a,b\n1,2\n3,4\n"), "csv, 3 rows x 2 columns")
        self.assertEqual(describe("a.md", b"x" * 1300), "md, 1.3 KB")



class FakeBackend:
    """A sandbox: files exist only behind download_files, not on the local disk."""

    def __init__(self, files):
        self.files, self.asked = files, []

    def download_files(self, paths):
        self.asked += list(paths)
        return [FileDownloadResponse(path=p, content=self.files.get(p),
                                     error=None if p in self.files else "file_not_found") for p in paths]


class FilesBase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.workspace = self.root / "ws"
        self.workspace.mkdir()
        self.store = OutputStore(self.root / "outputs")
        self.backend = FilesystemBackend(root_dir=str(self.workspace), virtual_mode=True)
        SEEN.clear()

    def run_turn(self, *responses, backend=None, user="alice", thread=None):
        features = [files_feature(backend or self.backend, self.store, user)]
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), self.backend, features)
        config = {"configurable": {"thread_id": thread or thread_id("alice", "parks")}}
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]}, config)
        return [m for m in result["messages"] if m.type == "tool" and m.name == "send_file"]


class SendFileTests(FilesBase):

    def test_valid_markdown_is_saved_as_v1(self):
        (self.workspace / "plan.md").write_bytes(b"# Plan\n\nSecret sauce.")
        [message] = self.run_turn(call("send_file", {"path": "/plan.md", "caption": "the plan"}, "s1"))
        result = json.loads(message.content)
        self.assertEqual(set(result), {"name", "version", "bytes", "kind", "details"})
        self.assertEqual(result["details"], "md, 21 B")
        self.assertEqual((result["name"], result["version"], result["kind"]), ("plan.md", 1, "md"))
        self.assertNotIn("Secret sauce", message.content)
        self.assertIs(message.artifact["saved"], True)
        self.assertEqual(self.store.read("alice", "parks", "plan.md"), b"# Plan\n\nSecret sauce.")
        self.assertEqual(self.store.list("alice", "parks")[0].caption, "the plan")

    def test_failures_save_nothing(self):
        (self.workspace / "notes.docx").write_text("x", encoding="utf-8")
        (self.workspace / "keys.md").write_text("key: AKIAABCDEFGHIJKLMNOP", encoding="utf-8")
        messages = self.run_turn(call("send_file", {"path": "/missing.md"}, "s1"),
                                 call("send_file", {"path": "/notes.docx"}, "s2"),
                                 call("send_file", {"path": "/keys.md"}, "s3"))
        self.assertEqual(len(messages), 3)
        for message, words in zip(messages, ("write_file", ".md", "credential")):
            self.assertIn(words, message.content)
            self.assertIsNone(message.artifact)
        self.assertEqual(self.store.list("alice", "parks"), [])

    def test_reads_through_the_backend_not_the_disk(self):
        sandbox = FakeBackend({"/home/daytona/report.md": b"# Report"})
        self.run_turn(call("send_file", {"path": "/home/daytona/report.md"}, "s1"), backend=sandbox)
        self.assertEqual(self.store.read("alice", "parks", "report.md"), b"# Report")

    def test_relative_path_is_read_as_given(self):
        sandbox = FakeBackend({"report.md": b"# Report"})
        self.run_turn(call("send_file", {"path": "report.md"}, "s1"), backend=sandbox)
        self.assertEqual(sandbox.asked, ["report.md"])

    def test_identity_comes_from_code(self):
        (self.workspace / "plan.md").write_text("# Plan", encoding="utf-8")
        [message] = self.run_turn(call("send_file", {"path": "/plan.md"}, "s1"), thread=thread_id("bob", "parks"))
        self.assertIsNone(message.artifact)
        self.assertFalse((self.root / "outputs").exists())
        tool = files_feature(self.backend, self.store, "alice").tools[0]
        self.assertEqual(set(tool.tool_call_schema.model_json_schema()["properties"]), {"path", "caption"})

    def test_non_chat_thread_is_refused(self):
        (self.workspace / "plan.md").write_text("# Plan", encoding="utf-8")
        [message] = self.run_turn(call("send_file", {"path": "/plan.md"}, "s1"), thread="t")
        self.assertIn("chat", message.content)
        self.assertIsNone(message.artifact)

    def test_subagents_cannot_send_files(self):
        self.run_turn(delegate("general-purpose"), AIMessage(content="sub done"))
        tool_lists = [names for kind, names in SEEN if kind == "tools"]
        self.assertIn("send_file", tool_lists[0], "the main agent has it")
        self.assertTrue(all("send_file" not in names for names in tool_lists[1:-1]), tool_lists)
        SEEN.clear()
        model = Model(responses=[AIMessage(content="ok")])
        worker = worker_factory(model, InMemorySaver(), self.backend, [files_feature(self.backend, self.store, "alice")])
        worker("general-purpose").invoke({"messages": [{"role": "user", "content": "go"}]},
                                         {"configurable": {"thread_id": "w"}})
        tool_lists = [names for kind, names in SEEN if kind == "tools"]
        self.assertTrue(tool_lists and all("send_file" not in names for names in tool_lists), tool_lists)

    def test_delivered_files_are_listed_for_the_model(self):
        (self.workspace / "plan.md").write_text("# Plan", encoding="utf-8")
        self.run_turn(call("send_file", {"path": "/plan.md"}, "s1"))
        prompts = [str(m[0].content) for kind, m in SEEN if kind == "messages"]
        self.assertNotIn("plan.md (v1)", prompts[0])
        self.assertIn("Files delivered in this chat: plan.md (v1).", prompts[-1])

    def test_modes(self):
        self.assertIsNone(files_feature(self.backend, self.store, "alice", mode="never"))
        self.assertIn("for every answer longer than a few lines",
                      files_feature(self.backend, self.store, "alice", mode="always").prompt)
        self.assertNotIn("for every answer longer than a few lines",
                         files_feature(self.backend, self.store, "alice").prompt)

    def test_errors_come_back_to_the_agent(self):
        """Nothing send_file meets may end the chat session: each failure is a message."""
        from unittest.mock import patch

        class Exploding(FakeBackend):
            def download_files(self, paths):
                raise ConnectionError("sandbox unreachable")
        (self.workspace / "big.csv").write_bytes(b"a,b\n" + b"x" * 200_000 + b",1\n")
        (self.workspace / "deep.json").write_bytes(b"[" * 100_000 + b"]" * 100_000)
        (self.workspace / "ok.md").write_bytes(b"# ok")
        cases = [({"path": "/big.csv"}, None, "Not sent"),
                 ({"path": "/deep.json"}, None, "Not sent"),
                 ({"path": "/x.md"}, Exploding({}), "sandbox unreachable")]
        for args, backend, words in cases:
            with self.subTest(args=args):
                [message] = self.run_turn(call("send_file", args, "s1"), backend=backend)
                self.assertIn(words, message.content)
                self.assertIsNone(message.artifact)
        with patch.object(self.store, "save", side_effect=PermissionError("[WinError 5] Access is denied")):
            [message] = self.run_turn(call("send_file", {"path": "/ok.md"}, "s1"))
        self.assertIn("Access is denied", message.content)
        self.assertEqual(self.store.list("alice", "parks"), [])



class FileEventTests(FilesBase):
    def turn_lines(self, *responses):
        from recording_ui import RecordingUI

        from deep_agents_demo.core.turns import run_turn
        ui = RecordingUI()
        features = [files_feature(self.backend, self.store, "alice")]
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(), self.backend, features)
        run_turn(agent, "go", {"configurable": {"thread_id": thread_id("alice", "parks")}}, ui)
        return [line for line in ui.lines if line.startswith("[file]")]

    def test_one_file_event_for_a_successful_send(self):
        (self.workspace / "plan.md").write_bytes(b"# Plan")
        [line] = self.turn_lines(call("send_file", {"path": "/plan.md"}, "s1"))
        self.assertTrue(line.startswith("[file] plan.md  v1  6 B  ->  "), line)

    def test_no_file_event_for_a_failed_send(self):
        self.assertEqual(self.turn_lines(call("send_file", {"path": "/missing.md"}, "s1")), [])

    def test_a_quiet_run_still_shows_the_file(self):
        """--prompt prints only the answer, but a delivered file is part of the answer."""
        from recording_ui import RecordingUI

        from deep_agents_demo.core.turns import run_turn
        (self.workspace / "plan.md").write_bytes(b"# Plan")
        ui = RecordingUI()
        model = Model(responses=[call("send_file", {"path": "/plan.md"}, "s1"), AIMessage(content="Here it is.")])
        agent = build_agent(model, InMemorySaver(), self.backend, [files_feature(self.backend, self.store, "alice")])
        run_turn(agent, "go", {"configurable": {"thread_id": thread_id("alice", "parks")}}, ui, quiet=True)
        self.assertEqual(len(ui.lines), 2, ui.lines)
        self.assertTrue(ui.lines[0].startswith("[file] plan.md  v1"), ui.lines)
        self.assertEqual(ui.lines[1], "Here it is.")


if __name__ == "__main__":
    unittest.main()

"""Images: attaching PNG/JPG files, the agent reading images, and keeping context small."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from deep_agents_demo.core.assembly import build_agent
from deep_agents_demo.core.turns import printable
from deep_agents_demo.features.images.feature import images_feature
from deep_agents_demo.features.images.uploads import (
    MAX_SIDE,
    ImageError,
    Uploads,
    image_paths,
    load_image,
    message_content,
    sniff,
)
from deep_agents_demo.features.images.window import ImageWindow
from deep_agents_demo.ui.cli.repl import ChatSession

SEEN = []  # the messages of every model call


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        SEEN.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def picture(path, size=(40, 30), fmt="PNG", color=(30, 60, 220)):
    Image.new("RGB", size, color).save(path, fmt)
    return path


def images_in(message):
    return [b for b in message.content if isinstance(b, dict) and b.get("type") == "image"] \
        if isinstance(message.content, list) else []


class Workspace(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


class LoadTests(Workspace):
    def test_png_and_jpg_are_recognised_by_their_contents(self):
        png = picture(self.root / "a.png")
        jpg = picture(self.root / "b.jpg", fmt="JPEG")
        self.assertEqual(load_image(png)[1:], ("image/png", (40, 30)))
        self.assertEqual(load_image(jpg)[1], "image/jpeg")
        renamed = self.root / "really-a-png.jpg"
        shutil.copy(png, renamed)
        self.assertEqual(load_image(renamed)[1], "image/png", "the contents decide, not the name")

    def test_other_files_are_refused(self):
        gif = self.root / "x.png"
        Image.new("RGB", (5, 5)).save(gif, "GIF")
        text = self.root / "notes.png"
        text.write_text("not an image")
        for path, reason in ((gif, "not a PNG or JPG"), (text, "not a PNG or JPG"),
                             (self.root / "missing.png", "no file")):
            with self.subTest(path=path.name), self.assertRaises(ImageError) as caught:
                load_image(path)
            self.assertIn(reason, str(caught.exception))
        self.assertIsNone(sniff(b"GIF89a"))

    def test_too_large_files_are_refused(self):
        big = picture(self.root / "big.png")
        with patch("deep_agents_demo.features.images.uploads.MAX_BYTES", 10), self.assertRaises(ImageError) as caught:
            load_image(big)
        self.assertIn("larger than", str(caught.exception))

    def test_large_images_are_scaled_down_keeping_their_shape(self):
        data, mime, size = load_image(picture(self.root / "wide.jpg", (4000, 1000), "JPEG"))
        self.assertEqual(size, (MAX_SIDE, MAX_SIDE // 4))
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(sniff(data), "image/jpeg")


class AttachTests(Workspace):
    def test_attach_saves_a_copy_the_agent_can_read(self):
        uploads = Uploads(self.root / "ws", "alice", "/instructions")
        attachment = uploads.attach(picture(self.root / "my plot.png"))
        self.assertTrue(attachment.path.startswith("/instructions/uploads/alice/"))
        self.assertTrue(attachment.path.endswith("-my_plot.png"))
        saved = self.root / "ws" / attachment.path.removeprefix("/instructions/")
        self.assertEqual(saved.read_bytes(), attachment.data)
        self.assertIn("my plot.png (40x30", attachment.describe())

    def test_paths_in_a_message_are_found_quoted_or_not(self):
        a = picture(self.root / "a.png")
        spaced = picture(self.root / "with space.JPG", fmt="JPEG")
        text = f'fix {a} and "{spaced}", not /sandbox/missing.png or plot.png.txt'
        self.assertEqual(image_paths(text), [str(a), str(spaced)])
        self.assertEqual(image_paths("no images here"), [])

    def test_message_content_names_and_carries_the_images(self):
        attachment = Uploads(self.root, "alice").attach(picture(self.root / "a.png"))
        content = message_content("What is wrong here?", [attachment])
        self.assertIn("[Attached image: /uploads/alice/", content[0]["text"])
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["mime_type"], "image/png")
        self.assertEqual(message_content("plain", []), "plain")

    def test_chat_session_attaches_with_the_next_message(self):
        said = []
        session = ChatSession(chats=None, current="c", say=said.append, uploads=Uploads(self.root, "alice"))
        png = picture(self.root / "a.png")
        session.handle(f"/image {png}")
        self.assertIn("Attached a.png", said[0])
        content = session.content("check this")
        self.assertEqual(len(images_in(HumanMessage(content=content))), 1)
        self.assertEqual(session.content("again"), "again", "an attachment is sent once")
        session.handle("/image nope.png")
        self.assertIn("Not attached: no file", said[-1])
        inline = session.content(f"look at {png}")
        self.assertEqual(len(images_in(HumanMessage(content=inline))), 1)


class WindowTests(unittest.TestCase):
    def test_only_the_latest_images_reach_the_model(self):
        image = {"type": "image", "base64": "AAAA", "mime_type": "image/png"}
        messages = [HumanMessage(content=[{"type": "text", "text": f"m{i}"}, image]) for i in range(5)]

        class Request:
            def __init__(self, messages):
                self.messages = messages

            def override(self, messages):
                return Request(messages)

        out = ImageWindow(keep=2)._request(Request(messages)).messages
        self.assertEqual([len(images_in(m)) for m in out], [0, 0, 0, 1, 1])
        self.assertIn("read_file", out[0].content[1]["text"])
        self.assertEqual(len(images_in(messages[0])), 1, "the saved conversation is untouched")

    def test_printing_never_dumps_base64(self):
        message = ToolMessage(content=[{"type": "image", "base64": "QUJD" * 500, "mime_type": "image/png"}],
                              tool_call_id="c")
        self.assertNotIn("QUJD", str(printable(message).content))


class EndToEndTests(Workspace):
    def run_turn(self, content, *responses):
        SEEN.clear()
        (self.root / "skills").mkdir(exist_ok=True)
        backend = FilesystemBackend(root_dir=str(self.root), virtual_mode=True)
        agent = build_agent(Model(responses=[*responses, AIMessage(content="done")]), InMemorySaver(),
                            backend, [images_feature()])
        with patch("builtins.print"):
            agent.invoke({"messages": [{"role": "user", "content": content}]},
                         {"configurable": {"thread_id": "t"}})

    def test_an_attached_image_reaches_the_model(self):
        attachment = Uploads(self.root, "alice").attach(picture(self.root / "a.png"))
        self.run_turn(message_content("what colour?", [attachment]))
        user = [m for m in SEEN[0] if m.type == "human"][0]
        self.assertEqual(len(images_in(user)), 1)

    def test_the_agent_can_look_at_an_image_it_rendered(self):
        picture(self.root / "chart.png")
        read = AIMessage(content="", tool_calls=[{"name": "read_file", "id": "r",
                                                  "args": {"file_path": "/chart.png"}}])
        self.run_turn("draw a chart and check it", read)
        tool = [m for m in SEEN[-1] if m.type == "tool"][0]
        self.assertEqual(len(images_in(tool)), 1, "read_file returns the image itself")

    def test_the_prompt_asks_the_agent_to_check_its_images(self):
        self.run_turn("hi")
        self.assertIn("open it with read_file and check it", str(SEEN[0][0].content))


if __name__ == "__main__":
    unittest.main()

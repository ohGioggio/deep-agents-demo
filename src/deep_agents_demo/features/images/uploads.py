"""Images in the chat: PNG and JPG files you attach, and images the agent renders.

- **Attaching.** A PNG or JPG path in your message (quoted if it has spaces), or
  ``/image PATH`` before the message, attaches the file. It is checked by its
  contents (not its name), limited in size, scaled down if large, saved under
  ``uploads/<user>/`` in the workspace and sent with the message.
- **Checking its own work.** Deep Agents' ``read_file`` returns images to the
  model, so the agent is told to open every image it creates and fix it if it
  looks wrong (unreadable labels, empty plots, cut-off parts).
- **Keeping context small.** ``ImageWindow`` sends only the most recent images to
  the model; older ones become a short note with their path, to read again.
"""
import base64
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path

from deep_agents_demo.persistence.files import safe_folder_name

MAX_BYTES = 20 * 1024 * 1024   # larger files are refused
MAX_SIDE = 1568                # longer sides are scaled down to this (pixels)
KEEP_IMAGES = 3                # messages whose images still reach the model
SIGNATURES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}
_PATH = re.compile(r"\"([^\"]+\.(?:png|jpe?g))\"|'([^']+\.(?:png|jpe?g))'|([^\s\"']+\.(?:png|jpe?g))\b",
                   re.IGNORECASE)


class ImageError(ValueError):
    """A file that can't be attached, with a reason the user can act on."""


def sniff(data: bytes) -> str | None:
    """The image type from the file's first bytes: image/png, image/jpeg, or None."""
    return next((mime for signature, mime in SIGNATURES.items() if data.startswith(signature)), None)


def load_image(path: Path) -> tuple[bytes, str, tuple[int, int]]:
    """(bytes, mime type, (width, height)) of a PNG or JPG, scaled down if large."""
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"no file at {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ImageError(f"{path.name} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    data = path.read_bytes()
    mime = sniff(data)
    if mime is None:
        raise ImageError(f"{path.name} is not a PNG or JPG image")
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            size = image.size
            if max(size) <= MAX_SIDE:
                return data, mime, size
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            out = io.BytesIO()
            if mime == "image/png":
                image.save(out, "PNG", optimize=True)
            else:
                image.convert("RGB").save(out, "JPEG", quality=85)
            return out.getvalue(), mime, image.size
    except (UnidentifiedImageError, OSError) as error:
        raise ImageError(f"{path.name} could not be read as an image ({error})") from None


@dataclass(frozen=True)
class Attachment:
    path: str            # where the agent finds it, e.g. /uploads/alice/20261111-120000-plot.png
    mime: str
    data: bytes
    size: tuple          # (width, height) after any scaling
    name: str            # the original file name

    def block(self) -> dict:
        return {"type": "image", "base64": base64.b64encode(self.data).decode(), "mime_type": self.mime}

    def describe(self) -> str:
        return f"{self.name} ({self.size[0]}x{self.size[1]}, {len(self.data) // 1024} KB)"


class Uploads:
    """Saves attached images to ``<workspace>/uploads/<user>/``, readable by the agent."""

    def __init__(self, workspace: Path, user: str, prefix: str = ""):
        folder = safe_folder_name(user)
        self.folder = Path(workspace) / "uploads" / folder
        self.virtual = f"{prefix}/uploads/{folder}/"

    def attach(self, path) -> Attachment:
        data, mime, size = load_image(Path(path))
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(path).stem)[:60] or "image"
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}{EXTENSIONS[mime]}"
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / name).write_bytes(data)
        return Attachment(self.virtual + name, mime, data, size, Path(path).name)


def image_paths(text: str) -> list[str]:
    """PNG/JPG paths written in a message that exist on this machine, in order."""
    found = []
    for match in _PATH.finditer(text or ""):
        candidate = next(group for group in match.groups() if group)
        if Path(candidate).is_file() and candidate not in found:
            found.append(candidate)
    return found


def message_content(text: str, attachments=()) -> str | list:
    """A user message: plain text, or text naming the attachments plus the images."""
    if not attachments:
        return text
    notes = "\n".join(f"[Attached image: {a.path}]" for a in attachments)
    return [{"type": "text", "text": f"{text}\n\n{notes}".strip()}, *(a.block() for a in attachments)]



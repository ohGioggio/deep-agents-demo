"""Checks a delivered file must pass before it is kept.

Each check returns the problem in words the agent can act on ("... is not valid
JSON: ..."), or None. Images are checked by content, never by name; text must be
UTF-8 and free of credentials (the same scanner as memory and skills).
"""
import csv
import io
import json
import re
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser
from pathlib import PurePosixPath

from deep_agents_demo.core.media import sniff
from deep_agents_demo.core.secrets import find_secrets, split_findings

ALLOWED = (".md", ".txt", ".csv", ".html", ".tex", ".svg", ".json", ".png", ".jpg", ".jpeg")
IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
TEXT_LIMIT = 5 * 1024 * 1024
IMAGE_LIMIT = 20 * 1024 * 1024


def clean_name(path: str) -> str:
    """The file's name as kept: base name only, lowercase extension, spaces as "-",
    anything else unusual as "_". ValueError when there is no name or extension."""
    base = PurePosixPath(str(path).replace("\\", "/")).name
    stem, dot, extension = base.rpartition(".")
    if base.startswith("."):
        raise ValueError(f"{path!r}: a file name can't start with a dot; use e.g. report.md")
    if not dot or not stem.strip(" ._") or not extension:
        raise ValueError(f"{path!r} needs a file name with an extension, e.g. report.md")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", re.sub(r"\s+", "-", stem.strip()))
    return f"{stem}.{extension.lower()}"


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1


def _check_text(suffix: str, text: str) -> str | None:
    try:
        return _check_format(suffix, text)
    except (csv.Error, RecursionError, ValueError) as error:  # e.g. a huge CSV field, deep nesting
        return f"it can't be read as {suffix[1:].upper()}: {type(error).__name__}: {str(error)[:200]}"


def _check_format(suffix: str, text: str) -> str | None:
    if suffix == ".json":
        try:
            json.loads(text)
        except ValueError as error:
            return f"it is not valid JSON: {error}"
    elif suffix == ".csv":
        rows = [row for row in csv.reader(io.StringIO(text)) if row]
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            return (f"the CSV rows have different numbers of columns ({', '.join(map(str, sorted(widths)))}); "
                    "every row needs the same columns")
    elif suffix == ".svg":
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as error:
            return f"it is not valid SVG (XML): {error}"
        if root.tag.rsplit("}", 1)[-1] != "svg":
            return "an SVG file must have an <svg> root element"
    elif suffix == ".html":
        parser = _Tags()
        parser.feed(text)
        if not parser.tags:
            return "it has no HTML tags; write an HTML page, or send it as .txt or .md"
    return None


def validate(name: str, data: bytes) -> str | None:
    """None if the file can be kept, else what is wrong with it."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in ALLOWED:
        return f"{name}: files of this type can't be sent; use one of {', '.join(ALLOWED)}"
    if suffix in IMAGES:
        if len(data) > IMAGE_LIMIT:
            return f"{name} is larger than {IMAGE_LIMIT // (1024 * 1024)} MB"
        if sniff(data) != IMAGES[suffix]:
            kind = "PNG" if suffix == ".png" else "JPEG"
            return f"{name} is not a {kind} image (checked by its content)"
        return None
    if len(data) > TEXT_LIMIT:
        return f"{name} is larger than {TEXT_LIMIT // (1024 * 1024)} MB"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"{name} is not UTF-8 text"
    if not text.strip():
        return f"{name} is empty"
    problem = _check_text(suffix, text)
    if problem:
        return f"{name}: {problem}"
    credentials, _wording = split_findings(find_secrets(text))
    if credentials:
        return f"{name} contains what looks like a credential ({', '.join(credentials)}); remove it"
    return None


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / 1024 / 1024:.1f} MB"


def describe(name: str, data: bytes) -> str:
    """A short description for the agent, e.g. "csv, 12 rows x 4 columns"."""
    kind = PurePosixPath(name).suffix.lstrip(".").lower()
    if kind == "csv":
        try:
            rows = [row for row in csv.reader(io.StringIO(data.decode("utf-8"))) if row]
            return f"csv, {len(rows)} rows x {len(rows[0]) if rows else 0} columns"
        except (csv.Error, UnicodeDecodeError):
            pass
    return f"{kind}, {_size(len(data))}"

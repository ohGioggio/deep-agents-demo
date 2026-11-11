"""Folder names on disk."""
import re


def safe_folder_name(name: str) -> str:
    """``name`` usable as one folder name: anything but letters, digits, ``._-`` becomes ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "_"

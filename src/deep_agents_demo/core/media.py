"""What kind of image some bytes are, from their first bytes (never the file name)."""


SIGNATURES = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}


def sniff(data: bytes) -> str | None:
    """The image type from the file's first bytes: image/png, image/jpeg, or None."""
    return next((mime for signature, mime in SIGNATURES.items() if data.startswith(signature)), None)

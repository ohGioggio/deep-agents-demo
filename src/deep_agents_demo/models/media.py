"""The image types the file checks and the output store share: extension -> MIME type."""

IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
IMAGE_KINDS = frozenset(IMAGE_TYPES)  # the same names without the dot, as the store records them

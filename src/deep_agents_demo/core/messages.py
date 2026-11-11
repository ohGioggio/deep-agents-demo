"""Message helpers the engine and features share."""


def has_image(message) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(block, dict) and block.get("type") in ("image", "image_url") for block in message.content)


def without_images(message, note: str):
    """A copy of ``message`` with each image replaced by a short text note."""
    content = [{"type": "text", "text": note} if isinstance(block, dict)
               and block.get("type") in ("image", "image_url") else block for block in message.content]
    return message.model_copy(update={"content": content})

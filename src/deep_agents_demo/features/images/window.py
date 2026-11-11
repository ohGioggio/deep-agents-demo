"""Only the latest images reach the model; older ones become a note."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from deep_agents_demo.features.images.uploads import KEEP_IMAGES


def _has_image(message) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(block, dict) and block.get("type") in ("image", "image_url") for block in message.content)


def without_images(message, note: str):
    """A copy of ``message`` with each image replaced by a short text note."""
    content = [{"type": "text", "text": note} if isinstance(block, dict)
               and block.get("type") in ("image", "image_url") else block for block in message.content]
    return message.model_copy(update={"content": content})


class ImageWindow(AgentMiddleware):
    """Send only the latest images to the model: each image costs many tokens on
    every call. Older images become a note; the agent can read_file them again.
    The saved conversation is not changed."""

    def __init__(self, keep: int = KEEP_IMAGES):
        super().__init__()
        self.keep = keep

    def _request(self, request):
        with_images = [i for i, m in enumerate(request.messages)
                       if isinstance(m, (HumanMessage, ToolMessage)) and _has_image(m)]
        old = set(with_images[:-self.keep] if self.keep else with_images)
        if not old:
            return request
        note = "[earlier image not shown again; read_file its path to look at it]"
        messages = [without_images(m, note) if i in old else m for i, m in enumerate(request.messages)]
        return request.override(messages=messages)

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))

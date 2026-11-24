"""The memory tools the agent calls."""
import json

from langchain.tools import ToolRuntime

from deep_agents_demo.features.memory.service import MemoryService
from deep_agents_demo.features.memory.updates import RECENT_MESSAGES


def _text(content) -> str:
    """A message's text: plain content, or the text blocks of a multi-part one."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or ():
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _recent(runtime: ToolRuntime | None) -> list[str]:
    """The last few human and AI messages of this chat, as plain text, oldest first.
    The message that asked to remember is among them: the decider sees the request too."""
    state = getattr(runtime, "state", None) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else []
    texts = [_text(message.content) for message in messages if message.type in ("human", "ai")]
    return [text for text in texts if text.strip()][-RECENT_MESSAGES:]


def memory_tools(service: MemoryService, recall_tool=True):
    """The agent's memory tools: thin JSON wrappers around one user's MemoryService."""

    def remember(fact: str, runtime: ToolRuntime = None) -> str:
        """Remember a fact or preference only when the user asks to save it.

        Secrets (passwords, API keys, tokens, card numbers, private keys) are
        refused and never stored."""
        return json.dumps(service.remember(fact, _recent(runtime)), default=str)

    def recall(query: str) -> str:
        """Search this user's long-term memories for relevant preferences."""
        return json.dumps(service.search(query), default=str)

    def list_memories() -> str:
        """List everything remembered about this user, with ids for forget."""
        items = service.list()
        return json.dumps({"count": len(items), "memories": items}, default=str)

    def forget(memory_id: str) -> str:
        """Delete one of this user's memories by id (get ids from list_memories or recall).
        Only use when the user asks to forget or correct something."""
        return json.dumps(service.forget(memory_id), default=str)

    tools = [remember, list_memories, forget]
    if recall_tool:
        tools.insert(1, recall)
    return tools

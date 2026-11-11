"""The memory tools the agent calls."""
import json

from deep_agents_demo.features.memory.service import MemoryService


def memory_tools(service: MemoryService, recall_tool=True):
    """The agent's memory tools: thin JSON wrappers around one user's MemoryService."""

    def remember(fact: str) -> str:
        """Remember a fact or preference only when the user asks to save it.

        Secrets (passwords, API keys, tokens, card numbers, private keys) are
        refused and never stored."""
        return json.dumps(service.remember(fact), default=str)

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

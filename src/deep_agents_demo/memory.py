"""Mem0 long-term memory backed by a local Qdrant index."""
import json
import os

from deep_agents_demo.config import embedding_settings, setting
from deep_agents_demo.embeddings import CompatibleEmbeddings, Mem0EmbeddingAdapter


def make_memory(data):
    os.environ.setdefault("MEM0_DIR", str(data / "mem0_internal"))
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    from mem0 import Memory
    e = embedding_settings()
    dimensions = int(setting("EMBEDDING_DIMENSIONS", "1536"))
    if e["model"] == "nvidia/nemotron-3.5-embed-8b" and dimensions != 4096:
        raise ValueError("nvidia/nemotron-3.5-embed-8b returns 4096 dimensions in this demo. "
                         "Set EMBEDDING_DIMENSIONS=4096; migrate any existing 1536-wide memory index.")
    memory = Memory.from_config({
        "history_db_path": str(data / "mem0_history.sqlite"),
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "user_memories",
            "path": str(data / "memory_vectors"), "on_disk": True,
            "embedding_model_dims": dimensions,
        }},
        "llm": {"provider": "openai", "config": {
            "model": setting("MEMORY_MODEL", setting("AGENT_MODEL")),
            "api_key": setting("MEMORY_API_KEY", setting("OPENAI_API_KEY")),
            "openai_base_url": setting("MEMORY_BASE_URL", setting("OPENAI_BASE_URL")),
        }},
        "embedder": {"provider": "openai", "config": {
            "model": e["model"], "api_key": e["api_key"],
            "openai_base_url": e["base_url"],
        }},
    })
    # Mem0's stock OpenAI embedder does not send NVIDIA's input_type.
    # Use a shared adapter for add/update=passage and search=query.
    memory.embedding_model = Mem0EmbeddingAdapter(CompatibleEmbeddings(**e), dimensions)
    return memory


def close_memory(memory):
    # Mem0 2.2.1 close() handles SQLite but leaves local Qdrant clients open.
    memory.close()
    seen = set()
    for name in ("vector_store", "_telemetry_vector_store"):
        client = getattr(getattr(memory, name, None), "client", None)
        if client is not None and id(client) not in seen:
            client.close()
            seen.add(id(client))


def memory_tools(memory, user_id):
    # Identity is captured by trusted application code, never model arguments.
    def remember(fact: str) -> str:
        """Remember a fact or preference only when the user asks to save it."""
        return json.dumps(memory.add(
            [{"role": "user", "content": fact}], user_id=user_id,
        ), default=str)

    def recall(query: str) -> str:
        """Search this user's long-term memories for relevant preferences."""
        return json.dumps(memory.search(
            query, filters={"user_id": user_id}, top_k=5,
        ), default=str)

    return [remember, recall]

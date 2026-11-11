"""Mem0 long-term memory backed by a local Qdrant index."""
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from deep_agents_demo.core.secrets import find_secrets, redact
from deep_agents_demo.providers.embeddings import Mem0EmbeddingAdapter


class _OptionalExtrasFilter(logging.Filter):
    """Drop Mem0's notices about optional extras this demo doesn't use.

    Mem0 2.2.1 can add spaCy lemmatisation and fastembed BM25 keyword search to
    memory search. Without them it uses vector similarity only, which is what
    the memory eval measures, so the notices are noise. Other warnings pass.
    """

    def filter(self, record):
        message = record.getMessage()
        return not (message.startswith("Failed to load spaCy")
                    or message.startswith("fastembed not installed"))


for _name in ("mem0.utils.spacy_models", "mem0.vector_stores.qdrant"):
    logging.getLogger(_name).addFilter(_OptionalExtrasFilter())


def make_memory(data, memory_settings, embedding_settings, embeddings):
    """Mem0 with a local Qdrant index in ``data``.

    ``embeddings`` (a CompatibleEmbeddings) replaces Mem0's own embedder, so
    memory and RAG share retries, batching and NVIDIA's passage/query modes.
    """
    # Mem0 reads these two from the environment; they are its interface, not ours.
    os.environ.setdefault("MEM0_DIR", str(data / "mem0_internal"))
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    from mem0 import Memory
    dimensions = embedding_settings.dimensions
    if embedding_settings.model == "nvidia/nemotron-3.5-embed-8b" and dimensions != 4096:
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
            "model": memory_settings.model,
            "api_key": memory_settings.api_key,
            "openai_base_url": memory_settings.base_url,
        }},
        "embedder": {"provider": "openai", "config": {
            "model": embedding_settings.model, "api_key": embedding_settings.api_key,
            "openai_base_url": embedding_settings.base_url,
        }},
    })
    # Mem0's stock OpenAI embedder does not send NVIDIA's input_type.
    # Use a shared adapter for add/update=passage and search=query.
    memory.embedding_model = Mem0EmbeddingAdapter(embeddings, dimensions)
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


RECALL_TOP_K = 5


def _results(response):
    """Mem0 returns {"results": [...]} (older versions: a bare list)."""
    return response.get("results", []) if isinstance(response, dict) else list(response or [])


@dataclass(frozen=True)
class WritePolicy:
    """What may be stored, and how stored text is shown.

    ``find_secrets(text)`` names the kinds of secrets in a text (empty if none);
    ``redact(text)`` masks them. The default is ``core.secrets``.
    """
    find_secrets: Callable[[str], list] = find_secrets
    redact: Callable[[str], str] = redact


class MemoryService:
    """One user's long-term memory: the only code that talks to Mem0 for that user.

    Every operation is scoped to ``user_id``, which comes from trusted code (the
    CLI's --user), never from the model. Writes go through the ``WritePolicy``:
    secrets are refused before they reach Mem0 (its extraction model and history
    database included), and any secret-looking rewrite Mem0 produces is deleted.
    Everything returned is redacted, in case an older memory slipped through.
    Results are plain dicts, ready to serialize for a tool.
    """

    def __init__(self, memory, user_id: str, policy: WritePolicy = WritePolicy()):
        self.memory, self.user_id, self.policy = memory, user_id, policy

    def _shown(self, item) -> dict:
        """A memory as shown to the model: id, redacted text, and score if present."""
        shown = {"id": item.get("id"), "memory": self.policy.redact(item.get("memory", ""))}
        if item.get("score") is not None:
            shown["score"] = round(float(item["score"]), 3)
        return shown

    def remember(self, fact: str) -> dict:
        kinds = self.policy.find_secrets(fact)
        if kinds:
            return {"stored": False, "reason": "refused: looks like a secret", "detected": kinds,
                    "advice": "Secrets are never stored in memory. Tell the user to keep "
                              "them in a password manager or environment variables."}
        stored = _results(self.memory.add([{"role": "user", "content": fact}], user_id=self.user_id))
        # Mem0 rewrites facts with an LLM; never keep a rewrite that looks secret.
        removed = [item["id"] for item in stored
                   if item.get("id") and self.policy.find_secrets(item.get("memory", ""))]
        for memory_id in removed:
            self.memory.delete(memory_id)
        kept = [self._shown(item) for item in stored if item.get("id") not in removed]
        return {"stored": bool(kept), "results": kept, "removed_as_secret": len(removed)}

    def search(self, query: str, top_k: int = RECALL_TOP_K) -> list[dict]:
        hits = _results(self.memory.search(query, filters={"user_id": self.user_id}, top_k=top_k))
        return [self._shown(hit) for hit in hits]

    def list(self, limit: int = 100) -> list[dict]:
        items = _results(self.memory.get_all(filters={"user_id": self.user_id}, top_k=limit))
        return [self._shown(item) for item in items]

    def describe(self, memory_id: str) -> str | None:
        """The (redacted) text of one of this user's memories, or None."""
        item = self.memory.get(memory_id)
        if not item or item.get("user_id") != self.user_id:
            return None
        return self.policy.redact(item.get("memory", ""))

    def forget(self, memory_id: str) -> dict:
        item = self.memory.get(memory_id)
        # Same answer for "missing" and "someone else's": don't reveal other users' ids.
        if not item or item.get("user_id") != self.user_id:
            return {"forgotten": False, "reason": "no memory with that id for this user"}
        self.memory.delete(memory_id)
        return {"forgotten": True, "id": memory_id, "memory": self.policy.redact(item.get("memory", ""))}



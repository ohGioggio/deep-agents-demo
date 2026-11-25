"""Mem0 long-term memory backed by a local Qdrant index."""
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from deep_agents_demo.core.secrets import find_secrets, redact
from deep_agents_demo.features.memory.updates import UPDATE_CANDIDATES, MemoryDecider
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
NEW_TEXT_REFUSED = "refused: the new text looks like a secret"
SAVED_DESPITE_ERROR = "Mem0 reported an error after saving the update; the memory has the new text"


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

    def __init__(self, memory, user_id: str, policy: WritePolicy = WritePolicy(),
                 decider: MemoryDecider | None = None):
        self.memory, self.user_id, self.policy, self.decider = memory, user_id, policy, decider

    def _shown(self, item) -> dict:
        """A memory as shown to the model: id, redacted text, and score if present."""
        shown = {"id": item.get("id"), "memory": self.policy.redact(item.get("memory", ""))}
        if item.get("score") is not None:
            shown["score"] = round(float(item["score"]), 3)
        return shown

    def remember(self, fact: str, recent: list[str] | tuple = ()) -> dict:
        """Store ``fact``, or let the decider say how it changes what is remembered.

        Without a decider the fact is added as given. With one, the decider picks add,
        update, nothing, or delete; delete is never done here, only proposed for ``forget``.
        Any failure to decide adds the fact and says so in ``note``.
        """
        kinds = self.policy.find_secrets(fact)
        if kinds:
            return {"stored": False, "reason": "refused: looks like a secret", "detected": kinds,
                    "advice": "Secrets are never stored in memory. Tell the user to keep "
                              "them in a password manager or environment variables."}
        if self.decider is None:
            return self._add(fact, reason="stored as given")
        try:
            candidates = self._candidates(fact)
        except Exception as error:
            return self._add(fact, note=f"couldn't decide ({self.policy.redact(str(error))}); "
                                        "added as a new memory")
        if not candidates:
            return self._add(fact, reason="no memories yet")
        shown_recent = [self.policy.redact(text) for text in recent]
        try:
            operation = self.decider.decide(fact, shown_recent, candidates)
        except Exception as error:
            return self._add(fact, note=f"couldn't decide ({self.policy.redact(str(error))}); "
                                        "added as a new memory")
        reason = self.policy.redact(operation.reason)
        if operation.kind == "ADD":
            text = (operation.text or "").strip() or fact
            if self.policy.find_secrets(text):
                return {"stored": False, "reason": NEW_TEXT_REFUSED}
            return self._add(text, reason=reason)
        target = next((c for c in candidates if c["id"] == operation.memory_id), None)
        if operation.kind == "NONE" and target is None:  # changes nothing: no id to protect
            return {"stored": False, "action": "already known", "reason": reason}
        if target is None:  # never act on an id the decider was not shown (or another user's)
            return self._add(fact, note=f"couldn't decide (memory id {operation.memory_id!r} is not "
                                        "among the candidates); added as a new memory")
        if operation.kind == "NONE":
            return {"stored": False, "action": "already known", "id": target["id"],
                    "memory": target["memory"], "reason": reason}
        if operation.kind == "DELETE":
            return {"stored": False, "action": "to forget", "id": target["id"],
                    "memory": target["memory"], "reason": reason,
                    "advice": "Call forget with this id; the user approves the deletion."}
        text = (operation.text or "").strip()
        if not text:
            return self._add(fact, reason=reason,
                             note="couldn't decide (the update had no text); added as a new memory")
        if self.policy.find_secrets(text):
            return {"stored": False, "reason": NEW_TEXT_REFUSED}
        updated = {"stored": True, "action": "updated", "id": target["id"], "before": target["memory"],
                   "after": text, "reason": reason}
        try:
            self.memory.update(target["id"], text)
        except Exception as error:
            if self._holds(target["id"], text):  # the write landed before the error
                return {**updated, "note": SAVED_DESPITE_ERROR}
            return self._add(fact, reason=reason,
                             note=f"couldn't update {target['id']} ({self.policy.redact(str(error))}); "
                                  "added as a new memory")
        return updated

    def _candidates(self, fact: str) -> list[dict]:
        """This user's memories most similar to ``fact``, by Mem0 vector search: the best
        UPDATE_CANDIDATES, redacted, as {"id", "memory"} only. Only a hit whose user_id is this
        user's is kept: a hit without one is not provably theirs, so it is dropped too."""
        hits = _results(self.memory.search(fact, filters={"user_id": self.user_id},
                                           top_k=UPDATE_CANDIDATES, threshold=0.0))
        candidates = []
        for hit in hits:
            if hit.get("user_id") != self.user_id:
                continue
            shown = self._shown(hit)
            candidates.append({"id": shown["id"], "memory": shown["memory"]})
        return candidates

    def _holds(self, memory_id: str, text: str) -> bool:
        """Whether this user's memory ``memory_id`` already reads ``text``, after a failed update."""
        try:
            item = self.memory.get(memory_id)
        except Exception:
            return False
        return (isinstance(item, dict) and item.get("user_id") == self.user_id
                and (item.get("memory") or "").strip() == (text or "").strip())

    def _add(self, text: str, reason: str = "", note: str | None = None) -> dict:
        """Add ``text`` as given, with no extraction by Mem0 (it would rewrite the user's words)."""
        try:
            stored = _results(self.memory.add([{"role": "user", "content": text}],
                                              user_id=self.user_id, infer=False))
        except Exception as error:  # e.g. the embedding provider is down: report it, don't raise
            return {"stored": False,
                    "reason": f"couldn't store the memory ({self.policy.redact(str(error))})"}
        # Mem0 rewrites facts with an LLM; never keep a rewrite that looks secret.
        removed = [item["id"] for item in stored
                   if item.get("id") and self.policy.find_secrets(item.get("memory", ""))]
        deleted, warnings = [], []
        for memory_id in removed:
            try:
                self.memory.delete(memory_id)
                deleted.append(memory_id)
            except Exception as error:  # the rewrite stays stored: say so, with the id to forget
                warnings.append(f"memory {memory_id} looks like a secret and could not be deleted "
                                f"({self.policy.redact(str(error))}); call forget with this id")
        kept = [self._shown(item) for item in stored if item.get("id") not in deleted]
        result = {"stored": bool(kept), "action": "added", "results": kept, "reason": reason,
                  "removed_as_secret": len(deleted)}
        if warnings:
            result["warning"] = "; ".join(warnings)
        if note:  # a fallback: the user should know, e.g. to spot a duplicate in list_memories
            result["note"] = (f"{note}. Tell the user, so they can check their memories with "
                              "list_memories.")
        return result

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



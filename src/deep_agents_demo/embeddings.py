"""String-only OpenAI-compatible embeddings with NVIDIA passage/query modes."""
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class CompatibleEmbeddings(Embeddings):
    def __init__(self, model, api_key, base_url, provider="auto", client=None):
        self.model = model
        self.nvidia = provider == "nvidia" or (
            provider == "auto" and urlparse(base_url).hostname == "integrate.api.nvidia.com"
        )
        if provider not in {"auto", "nvidia", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be auto, nvidia or openai")
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def _embed(self, texts, mode):
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            if any(not isinstance(t, str) for t in batch):
                raise TypeError("Embedding input must contain strings")
            kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
            if self.nvidia:
                kwargs["extra_body"] = {"input_type": mode, "truncate": "NONE"}
            response = self.client.embeddings.create(**kwargs)
            rows = sorted(response.data, key=lambda row: row.index)
            if [r.index for r in rows] != list(range(len(batch))):
                raise ValueError("Embedding response has missing or duplicated entries")
            vectors.extend(row.embedding for row in rows)
        return vectors

    def embed_documents(self, texts):
        return self._embed(texts, "passage")

    def embed_query(self, text):
        return self._embed([text], "query")[0]


class Mem0EmbeddingAdapter:
    """Adapter for the embedding_model interface in locked Mem0 2.2.1."""
    def __init__(self, embeddings, dimensions):
        self.embeddings = embeddings
        self.dimensions = dimensions

    def embed_batch(self, texts, memory_action=None):
        mode = "query" if memory_action == "search" else "passage"
        vectors = self.embeddings._embed(list(texts), mode)
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("Returned embedding width differs from EMBEDDING_DIMENSIONS. "
                             "Correct the setting and migrate/recreate the memory vector index.")
        return vectors

    def embed(self, text, memory_action=None):
        return self.embed_batch([text], memory_action)[0]

"""String-only OpenAI-compatible embeddings with NVIDIA passage/query modes.

NVIDIA's endpoint answers two different problems with the same HTTP 500
"service inference failed": some multi-text batches fail every time even though
each text embeds fine alone, and its free tier (about 40 requests per minute)
reports an exceeded limit the same way instead of with 429. So a failing batch
is split in half until it goes through, and only a single text that still fails
is retried with backoff (up to a minute, enough to outlast a rate-limit window).
"""
import random
import time
from urllib.parse import urlparse

import openai
from langchain_core.embeddings import Embeddings
from openai import OpenAI

RETRYABLE = (openai.RateLimitError, openai.InternalServerError,
             openai.APIConnectionError, openai.APITimeoutError)


def _retry_after(error) -> float | None:
    """Seconds from a Retry-After header, if the server sent one."""
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


class CompatibleEmbeddings(Embeddings):
    def __init__(self, model, api_key, base_url, provider="auto", client=None,
                 rate_limiter=None, max_retries=6, sleep=time.sleep, batch_size=16, notify=None):
        self.model = model
        self.nvidia = provider == "nvidia" or (
            provider == "auto" and urlparse(base_url).hostname == "integrate.api.nvidia.com"
        )
        if provider not in {"auto", "nvidia", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be auto, nvidia or openai")
        # Retries are handled here (with longer waits), not by the OpenAI client.
        self.client = client or OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.rate_limiter, self.max_retries, self.sleep = rate_limiter, max_retries, sleep
        self.notify = notify or (lambda message: None)  # retry warnings; the UI decides where
        # Shrinks when the server rejects a batch, so later batches start at a size that works.
        self.batch_size = max(1, int(batch_size))

    @classmethod
    def from_settings(cls, settings, rate_limiter=None, notify=None):
        """Build from ``config.EmbeddingSettings``."""
        return cls(settings.model, settings.api_key, settings.base_url, settings.provider,
                   rate_limiter=rate_limiter, batch_size=settings.batch_size, notify=notify)

    def _create(self, retry_server_errors=True, **kwargs):
        """One request, with backoff on rate limits and transient failures.

        With ``retry_server_errors=False`` a 500 is raised at once, so the caller
        can split the batch instead of resending a request the server rejects.
        """
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(blocking=True)
            try:
                return self.client.embeddings.create(**kwargs)
            except RETRYABLE as error:
                if attempt == self.max_retries or (
                        not retry_server_errors and isinstance(error, openai.InternalServerError)):
                    raise
                # 2, 4, 8, 16, 32, 60 s (+ jitter): the later waits outlast a 1-minute window.
                delay = _retry_after(error) or min(60.0, 2.0 ** (attempt + 1))
                delay += random.uniform(0, 1)
                self.notify(f"Embedding request failed ({type(error).__name__}); probably a rate limit. "
                      f"Retrying in {delay:.0f}s ({attempt + 1}/{self.max_retries}).")
                self.sleep(delay)

    def _embed_batch(self, batch, mode):
        """Embed one batch. NVIDIA's endpoint can fail some multi-text requests
        with a 500 every time while each text works alone, so a failing batch is
        split in half (down to single texts) instead of being resent unchanged."""
        if len(batch) > self.batch_size:
            # A smaller size was learned while splitting: don't resend sizes known to fail.
            return [vector for start in range(0, len(batch), self.batch_size)
                    for vector in self._embed_batch(batch[start:start + self.batch_size], mode)]
        kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self.nvidia:
            kwargs["extra_body"] = {"input_type": mode, "truncate": "NONE"}
        try:
            response = self._create(retry_server_errors=len(batch) == 1, **kwargs)
        except openai.InternalServerError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            self.batch_size = min(self.batch_size, middle)
            self.notify(f"Embedding batch of {len(batch)} failed with a server error; "
                  f"retrying as {middle} + {len(batch) - middle}.")
            return self._embed_batch(batch[:middle], mode) + self._embed_batch(batch[middle:], mode)
        rows = sorted(response.data, key=lambda row: row.index)
        if [r.index for r in rows] != list(range(len(batch))):
            raise ValueError("Embedding response has missing or duplicated entries")
        return [row.embedding for row in rows]

    def _embed(self, texts, mode):
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("Embedding input must contain strings")
        vectors, start = [], 0
        while start < len(texts):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch, mode))
            start += len(batch)
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

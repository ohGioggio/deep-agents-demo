"""Embedding request tests against a mocked HTTP endpoint."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
from openai import OpenAI

from deep_agents_demo.embeddings import CompatibleEmbeddings, Mem0EmbeddingAdapter

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"


class EmbeddingPayloadTests(unittest.TestCase):
    def test_nvidia_wire_payload_for_rag_and_memory(self):
        payloads = []

        def respond(request):
            body = json.loads(request.content)
            payloads.append(body)
            self.assertTrue(all(isinstance(x, str) for x in body["input"]))
            self.assertEqual(body["encoding_format"], "float")
            self.assertNotIn("dimensions", body)
            return httpx.Response(200, json={
                "object": "list", "model": body["model"],
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                         for i in range(len(body["input"]))],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })

        with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http:
            client = OpenAI(api_key="dummy", base_url=NVIDIA_URL, http_client=http)
            embed = CompatibleEmbeddings("nvidia/test", "dummy", NVIDIA_URL, client=client)
            embed.embed_documents(["document"])
            embed.embed_query("question")
            memory = Mem0EmbeddingAdapter(embed, 2)
            memory.embed("preference", "add")
            memory.embed("changed preference", "update")
            memory.embed("find preference", "search")
            self.assertEqual([b["input_type"] for b in payloads],
                             ["passage", "query", "passage", "passage", "query"])
            with self.assertRaises(ValueError):
                Mem0EmbeddingAdapter(embed, 1536).embed("wrong width", "add")

    def test_openai_does_not_receive_nvidia_options(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0])])
        embed = CompatibleEmbeddings("test", "dummy", "https://api.openai.com/v1", client=client)
        self.assertEqual(embed.embed_query("hello"), [1.0, 0.0])
        self.assertNotIn("extra_body", client.embeddings.create.call_args.kwargs)

    def test_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            CompatibleEmbeddings("test", "dummy", "https://api.openai.com/v1",
                                 provider="other", client=Mock())


def api_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("simulated", response=response, body=None)


def ok_response(n=1):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0]) for i in range(n)])


class RetryTests(unittest.TestCase):
    def embeddings(self, *outcomes, **kwargs):
        client = Mock()
        client.embeddings.create.side_effect = list(outcomes)
        self.sleeps = []
        return CompatibleEmbeddings("m", "k", NVIDIA_URL, client=client,
                                    sleep=self.sleeps.append, **kwargs), client

    def test_nvidia_rate_limit_500_is_retried_with_growing_waits(self):
        error = api_error(openai.InternalServerError, 500)
        embed, client = self.embeddings(error, error, ok_response())
        self.assertEqual(embed.embed_query("hi"), [1.0, 0.0])
        self.assertEqual(client.embeddings.create.call_count, 3)
        self.assertTrue(2 <= self.sleeps[0] < 3 and 4 <= self.sleeps[1] < 5, self.sleeps)

    def test_failing_batches_are_split_without_waiting(self):
        client = Mock()

        def create(**kwargs):
            texts = kwargs["input"]
            if len(texts) > 1:  # NVIDIA: multi-text request rejected, single texts fine
                raise api_error(openai.InternalServerError, 500)
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[float(texts[0][1:])])])

        client.embeddings.create.side_effect = create
        sleeps = []
        embed = CompatibleEmbeddings("m", "k", NVIDIA_URL, client=client, sleep=sleeps.append)
        texts = [f"t{i}" for i in range(5)]
        self.assertEqual(embed.embed_documents(texts), [[0.0], [1.0], [2.0], [3.0], [4.0]])
        self.assertEqual(sleeps, [], "a rejected batch must be split, not waited on")
        # 5 fails, 2 fails (size 1 learned), then singles only: 1 + 1 + 5 requests,
        # not a retry of every intermediate size.
        self.assertEqual(client.embeddings.create.call_count, 7)
        calls = client.embeddings.create.call_count
        embed.embed_documents(texts)
        self.assertEqual(client.embeddings.create.call_count - calls, 5,
                         "after a split, later batches start at the size that worked")

    def test_batch_size_setting(self):
        from unittest.mock import patch
        from deep_agents_demo.config import embedding_settings
        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "1"}):
            self.assertEqual(embedding_settings()["batch_size"], 1)
        embed, client = self.embeddings(ok_response(), ok_response(), batch_size=1)
        embed.embed_documents(["a", "b"])
        self.assertEqual(client.embeddings.create.call_count, 2)

    def test_retry_after_header_wins(self):
        embed, _ = self.embeddings(api_error(openai.RateLimitError, 429, {"retry-after": "30"}),
                                   ok_response())
        embed.embed_query("hi")
        self.assertTrue(30 <= self.sleeps[0] < 31, self.sleeps)

    def test_waits_cap_at_a_minute_and_then_give_up(self):
        error = api_error(openai.InternalServerError, 500)
        embed, client = self.embeddings(*[error] * 7)
        with self.assertRaises(openai.InternalServerError):
            embed.embed_query("hi")
        self.assertEqual(client.embeddings.create.call_count, 7)
        self.assertTrue(all(s < 61 for s in self.sleeps) and self.sleeps[-1] >= 60, self.sleeps)

    def test_client_errors_are_not_retried(self):
        embed, client = self.embeddings(api_error(openai.BadRequestError, 400), ok_response())
        with self.assertRaises(openai.BadRequestError):
            embed.embed_query("hi")
        self.assertEqual((client.embeddings.create.call_count, self.sleeps), (1, []))

    def test_rate_limiter_is_acquired_for_every_attempt(self):
        limiter = Mock()
        embed, _ = self.embeddings(api_error(openai.InternalServerError, 500), ok_response(),
                                   rate_limiter=limiter)
        embed.embed_query("hi")
        self.assertEqual(limiter.acquire.call_count, 2)

    def test_shared_rate_limiter_follows_the_setting(self):
        from unittest.mock import patch
        from deep_agents_demo.config import shared_rate_limiter
        with patch.dict("os.environ", {"API_REQUESTS_PER_MINUTE": ""}):
            self.assertIsNone(shared_rate_limiter())
        with patch.dict("os.environ", {"API_REQUESTS_PER_MINUTE": "40"}):
            first, second = shared_rate_limiter(), shared_rate_limiter()
            self.assertIs(first, second, "chat and embeddings must share one limiter")
            self.assertAlmostEqual(first.requests_per_second, 40 / 60)


if __name__ == "__main__":
    unittest.main()

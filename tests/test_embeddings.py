"""Embedding request tests against a mocked HTTP endpoint."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
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


if __name__ == "__main__":
    unittest.main()

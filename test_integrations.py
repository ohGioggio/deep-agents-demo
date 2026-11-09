"""Offline tests with real Qdrant and deterministic fake embeddings."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from langchain_core.embeddings import Embeddings
from integrations import DocumentIndex, memory_tools

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]
    def embed_query(self, text):
        return [1.0, 0.0] if 'uv' in text.lower() else [0.0, 1.0]

class IntegrationTests(unittest.TestCase):
    def test_index_search_replacement_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / 'docs'; docs.mkdir()
            (docs / 'uv.md').write_text('Use uv for dependencies.')
            (docs / 'tests.txt').write_text('Run pytest.')
            idx = DocumentIndex(root, FakeEmbeddings())
            try:
                self.assertEqual(idx.index(docs), 2)
                self.assertEqual(json.loads(idx.search('uv'))[0]['source'], 'uv.md')
                (docs / 'uv.md').unlink()
                idx.index(docs)
                self.assertNotIn('uv.md', idx.search('uv'))
                (docs / 'tests.txt').unlink()
                with self.assertRaises(ValueError): idx.index(docs)
                self.assertIn('tests.txt', idx.search('test'))
            finally: idx.close()
            idx = DocumentIndex(root, FakeEmbeddings())
            try: self.assertIn('tests.txt', idx.search('test'))
            finally: idx.close()

    def test_memory_user_scope(self):
        mem = Mock()
        mem.add.return_value = mem.search.return_value = {'results': []}
        remember, recall = memory_tools(mem, 'alice')
        remember('Use uv'); recall('preferences')
        self.assertEqual(mem.add.call_args.kwargs['user_id'], 'alice')
        self.assertEqual(mem.search.call_args.kwargs['filters'], {'user_id': 'alice'})

class EmbeddingPayloadTests(unittest.TestCase):
    def test_nvidia_wire_payload_for_rag_and_memory(self):
        import httpx
        from openai import OpenAI
        from embedding_adapter import CompatibleEmbeddings, Mem0EmbeddingAdapter
        payloads = []
        def respond(request):
            body = json.loads(request.content)
            payloads.append(body)
            self.assertTrue(all(isinstance(x, str) for x in body['input']))
            self.assertEqual(body['encoding_format'], 'float')
            self.assertNotIn('dimensions', body)
            return httpx.Response(200, json={
                'object': 'list', 'model': body['model'],
                'data': [{'object': 'embedding', 'index': i, 'embedding': [1.0, 0.0]}
                         for i in range(len(body['input']))],
                'usage': {'prompt_tokens': 1, 'total_tokens': 1},
            })
        with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http:
            client = OpenAI(api_key='dummy', base_url='https://integrate.api.nvidia.com/v1', http_client=http)
            embed = CompatibleEmbeddings('nvidia/test', 'dummy',
                'https://integrate.api.nvidia.com/v1', client=client)
            embed.embed_documents(['document'])
            embed.embed_query('question')
            memory = Mem0EmbeddingAdapter(embed, 2)
            memory.embed('preference', 'add')
            memory.embed('changed preference', 'update')
            memory.embed('find preference', 'search')
            self.assertEqual([b['input_type'] for b in payloads],
                             ['passage', 'query', 'passage', 'passage', 'query'])
            with self.assertRaises(ValueError):
                Mem0EmbeddingAdapter(embed, 1536).embed('wrong width', 'add')

    def test_openai_does_not_receive_nvidia_options(self):
        from types import SimpleNamespace
        from embedding_adapter import CompatibleEmbeddings
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1.0, 0.0])])
        embed = CompatibleEmbeddings('test', 'dummy', 'https://api.openai.com/v1', client=client)
        self.assertEqual(embed.embed_query('hello'), [1.0, 0.0])
        self.assertNotIn('extra_body', client.embeddings.create.call_args.kwargs)

if __name__ == '__main__': unittest.main()

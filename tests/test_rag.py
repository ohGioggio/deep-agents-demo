"""Document index tests with real local Qdrant and deterministic fake embeddings."""
import gc
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from deep_agents_demo.rag import DocumentIndex


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "uv" in text.lower() else [0.0, 1.0]


def close(idx):
    # Qdrant local mode releases its SQLite file only when garbage collected;
    # Windows cannot delete the temporary directory while the file is open.
    idx.close()
    gc.collect()


class DocumentIndexTests(unittest.TestCase):
    def test_index_search_replacement_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "uv.md").write_text("Use uv for dependencies.")
            (docs / "tests.txt").write_text("Run pytest.")
            idx = DocumentIndex(root, FakeEmbeddings())
            try:
                self.assertEqual(idx.index(docs), 2)
                self.assertEqual(json.loads(idx.search("uv"))[0]["source"], "uv.md")
                # Reindexing drops chunks from deleted documents.
                (docs / "uv.md").unlink()
                idx.index(docs)
                self.assertNotIn("uv.md", idx.search("uv"))
                # Empty input keeps the existing index.
                (docs / "tests.txt").unlink()
                with self.assertRaises(ValueError):
                    idx.index(docs)
                self.assertIn("tests.txt", idx.search("test"))
            finally:
                close(idx)
            # The index persists across clients.
            idx = DocumentIndex(root, FakeEmbeddings())
            try:
                self.assertIn("tests.txt", idx.search("test"))
            finally:
                close(idx)

    def test_search_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = DocumentIndex(Path(tmp), FakeEmbeddings())
            try:
                self.assertIn("No documents indexed", idx.search("anything"))
            finally:
                close(idx)


if __name__ == "__main__":
    unittest.main()

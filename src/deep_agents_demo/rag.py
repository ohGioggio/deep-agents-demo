"""Local document retrieval over a dedicated Qdrant collection."""
import json
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from deep_agents_demo.config import embedding_settings
from deep_agents_demo.embeddings import CompatibleEmbeddings


class DocumentIndex:
    """A separate Qdrant collection for local Markdown/text source passages."""
    collection = "project_documents"

    def __init__(self, data, embeddings=None):
        self.client = QdrantClient(path=str(data / "document_vectors"))
        self.embeddings = embeddings or CompatibleEmbeddings(**embedding_settings())

    def index(self, folder):
        paths = sorted(p for p in folder.rglob("*")
                       if p.is_file() and not p.is_symlink()
                       and p.suffix.lower() in {".md", ".txt"}
                       and not any(part.startswith(".") for part in p.relative_to(folder).parts))
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = []
        for path in paths:
            if path.stat().st_size > 2_000_000:
                raise ValueError(f"Document exceeds 2 MB: {path.name}")
            for i, text in enumerate(splitter.split_text(path.read_text(encoding="utf-8"))):
                chunks.append({"text": text, "source": path.relative_to(folder).as_posix(), "chunk": i})
        if not chunks:
            raise ValueError("No nonempty .md or .txt documents; existing index preserved.")
        # Finish embedding before replacing the old collection.
        vectors = self.embeddings.embed_documents([c["text"] for c in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE,
        ))
        # Local-mode qdrant-client 1.20 can keep old points across delete/create,
        # so clear the recreated collection explicitly to drop stale chunks.
        self.client.delete(self.collection, wait=True,
                           points_selector=models.FilterSelector(filter=models.Filter()))
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["source"] + ":" + str(c["chunk"]))),
            vector=v, payload=c,
        ) for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 100):
            self.client.upsert(self.collection, points=points[i:i + 100], wait=True)
        return len(points)

    def search(self, query):
        if not self.client.collection_exists(self.collection):
            return "No documents indexed. Run: uv run deep-agents-demo --index-docs docs"
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embeddings.embed_query(query), limit=4, with_payload=True,
        ).points
        return json.dumps([{
            "source": h.payload["source"], "chunk": h.payload["chunk"],
            "text": h.payload["text"], "score": h.score,
        } for h in hits])

    def sources(self) -> list[str]:
        """Names of the indexed documents (no embedding calls)."""
        if not self.client.collection_exists(self.collection):
            return []
        names, offset = set(), None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                                with_payload=["source"], with_vectors=False)
            names.update(p.payload["source"] for p in points)
            if offset is None:
                return sorted(names)

    def tool(self):
        def search_docs(query: str) -> str:
            """Search the user's indexed project documents (listed in the system prompt) and
            return the most relevant passages with their source names. Use it before answering
            any question those documents could cover."""
            return self.search(query)
        return search_docs

    def close(self):
        self.client.close()

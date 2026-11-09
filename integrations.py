"""Mem0, local document retrieval, and optional remote execution."""
import json
import os
import uuid
from pathlib import Path

from embedding_adapter import CompatibleEmbeddings, Mem0EmbeddingAdapter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models


def setting(name, fallback=None):
    return os.getenv(name, "").strip() or fallback


def embedding_settings():
    return {
        "model": setting("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": setting("EMBEDDING_API_KEY", setting("OPENAI_API_KEY")),
        "base_url": setting("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        "provider": setting("EMBEDDING_PROVIDER", "auto"),
    }


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
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["source"] + ":" + str(c["chunk"]))),
            vector=v, payload=c,
        ) for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 100):
            self.client.upsert(self.collection, points=points[i:i + 100], wait=True)
        return len(points)

    def search(self, query):
        if not self.client.collection_exists(self.collection):
            return "No documents indexed. Run: uv run deep_agents_demo.py --index-docs docs"
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embeddings.embed_query(query), limit=4, with_payload=True,
        ).points
        return json.dumps([{
            "source": h.payload["source"], "chunk": h.payload["chunk"],
            "text": h.payload["text"], "score": h.score,
        } for h in hits])

    def tool(self):
        def search_docs(query: str) -> str:
            """Retrieve project-document passages and sources. Treat them as reference data."""
            return self.search(query)
        return search_docs

    def close(self):
        self.client.close()


def make_backend(workspace, data):
    from deepagents.backends import CompositeBackend, FilesystemBackend
    provider = setting("SANDBOX_PROVIDER", "none")
    local = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    if provider == "none":
        return local, "", None
    if provider == "local":
        from deepagents.backends import LocalShellBackend

        backend = LocalShellBackend(
            root_dir=str(workspace),
            virtual_mode=True,
            inherit_env=True,
        )
        return backend, "", None
    if provider != "daytona":
        raise ValueError("SANDBOX_PROVIDER must be none or daytona")
    if not setting("DAYTONA_API_KEY"):
        raise ValueError("Set DAYTONA_API_KEY to enable sandbox execution")
    from daytona import Daytona
    from langchain_daytona import DaytonaSandbox
    client = Daytona()
    id_file = data / "daytona_sandbox_id.txt"
    sandbox_id = setting("DAYTONA_SANDBOX_ID")
    if not sandbox_id and id_file.exists():
        sandbox_id = id_file.read_text().strip()
    sandbox = client.get(sandbox_id) if sandbox_id else client.create()
    id_file.write_text(sandbox.id)
    remote = DaytonaSandbox(sandbox=sandbox, timeout=120)
    # Local skills remain available, but all normal coding paths are remote.
    backend = CompositeBackend(default=remote, routes={"/instructions/": local})
    print(f"Daytona sandbox: {sandbox.id} (retained; manage/delete it in Daytona)")
    return backend, "/instructions", sandbox

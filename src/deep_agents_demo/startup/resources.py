"""What is opened once per process and shared by every user's runtime.

A local Qdrant folder can be opened by only one client per process, so Mem0 (and
the document index) must be opened once, then shared: each user's runtime gets a
``MemoryService`` for its own user id on the same store. The stores and the
checkpointer are shared the same way. Everything is closed by ``close()`` (or by
leaving the ``with`` block), in reverse order of opening.
"""
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field

from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.config import Settings
from deep_agents_demo.features.documents.index import DocumentIndex
from deep_agents_demo.features.memory.service import close_memory, make_memory
from deep_agents_demo.features.sandbox.providers import BackendSetup, make_backend
from deep_agents_demo.persistence.versioned_store import AGENT_TABLES, VersionedStore
from deep_agents_demo.providers.embeddings import CompatibleEmbeddings
from deep_agents_demo.providers.rate_limit import rate_limiter

Notify = Callable[[str], None]


def _quiet(*_args, **_kwargs) -> None:
    return None


def embeddings(settings: Settings, warn: Notify | None = None) -> CompatibleEmbeddings:
    """The embeddings client; ``warn`` receives retry warnings."""
    return CompatibleEmbeddings.from_settings(settings.embeddings, rate_limiter(settings.requests_per_minute),
                                              notify=warn)


def document_index(settings: Settings, warn: Notify | None = None) -> DocumentIndex:
    return DocumentIndex(settings.paths.data, embeddings(settings, warn))


def sandbox_backend(settings: Settings, stack: ExitStack, notify: Notify | None = None) -> BackendSetup:
    """Local files or the configured sandbox (which connects lazily)."""
    return make_backend(settings.sandbox, settings.paths.workspace, settings.paths.data, stack,
                        notify or _quiet)


@dataclass
class Resources:
    settings: Settings
    checkpointer: object
    sandbox: BackendSetup
    skill_store: VersionedStore
    agent_store: VersionedStore
    memory: object = None                    # one Mem0, or None when memory is off
    documents: DocumentIndex | None = None   # None when RAG is off
    stack: ExitStack = field(default_factory=ExitStack, repr=False)

    @classmethod
    def open(cls, settings: Settings, notify: Notify | None = None,
             warn: Notify | None = None) -> "Resources":
        """Open everything once. ``notify`` gets notes for the user (e.g. the sandbox
        connecting); ``warn`` gets warnings (embedding retries). Both default to silence."""
        stack = ExitStack()
        try:
            data = settings.paths.data
            checkpointer = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
            sandbox = sandbox_backend(settings, stack, notify)
            skill_store = VersionedStore(data / "skills.sqlite")
            stack.callback(skill_store.close)
            agent_store = VersionedStore(data / "agents.sqlite", AGENT_TABLES)
            stack.callback(agent_store.close)
            memory = None
            if settings.memory.enabled:
                memory = make_memory(data, settings.memory, settings.embeddings, embeddings(settings, warn))
                stack.callback(close_memory, memory)
            documents = None
            if settings.rag_enabled:
                documents = document_index(settings, warn)
                stack.callback(documents.close)
        except BaseException:
            stack.close()
            raise
        return cls(settings, checkpointer, sandbox, skill_store, agent_store, memory, documents, stack)

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "Resources":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

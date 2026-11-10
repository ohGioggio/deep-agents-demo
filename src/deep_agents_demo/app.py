"""Composition root: the one place that turns Settings into working objects.

Everything else receives what it needs as arguments. To see how the demo is
wired (which model, memory, documents and sandbox an agent gets), read this
module; to change the wiring, change it here.
"""
from contextlib import ExitStack
from dataclasses import dataclass

from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.agent import build_agent
from deep_agents_demo.config import Settings, rate_limiter
from deep_agents_demo.embeddings import CompatibleEmbeddings
from deep_agents_demo.memory import AutoRecallMiddleware, close_memory, make_memory, memory_tools
from deep_agents_demo.rag import DocumentIndex
from deep_agents_demo.sandbox import make_backend
from deep_agents_demo.threads import Chats
from deep_agents_demo.workspace import seed_files


def chat_model(settings: Settings) -> ChatOpenAI:
    # Shares the request cap with the embeddings; retries ride out transient errors.
    return ChatOpenAI(model=settings.chat.model, api_key=settings.chat.api_key,
                      base_url=settings.chat.base_url,
                      rate_limiter=rate_limiter(settings.requests_per_minute), max_retries=6)


def embeddings(settings: Settings) -> CompatibleEmbeddings:
    return CompatibleEmbeddings.from_settings(settings.embeddings,
                                              rate_limiter(settings.requests_per_minute))


def document_index(settings: Settings) -> DocumentIndex:
    return DocumentIndex(settings.paths.data, embeddings(settings))


def sandbox_backend(settings: Settings, stack: ExitStack, say=print):
    """``(backend, skills_prefix, workdir)`` for the configured sandbox (connects lazily)."""
    return make_backend(settings.sandbox, settings.paths.workspace, settings.paths.data, stack, say)


def prepare_workspace(settings: Settings) -> None:
    seed_files(settings.paths.workspace)
    settings.paths.data.mkdir(exist_ok=True)


@dataclass
class Runtime:
    """Everything a chat session needs, all closed by the ExitStack it was built with."""
    agent: object
    chats: Chats


def build_runtime(settings: Settings, user: str, stack: ExitStack, say=print) -> Runtime:
    """The full agent for one user: model, memory, documents, sandbox, checkpoints."""
    saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
    tools, middleware = [], []
    if settings.memory.enabled:
        memory = make_memory(settings.paths.data, settings.memory, settings.embeddings,
                             embeddings(settings))
        stack.callback(close_memory, memory)
        tools.extend(memory_tools(memory, user, recall_tool=settings.memory.recall_mode != "auto"))
        if settings.memory.recall_mode != "tool":
            middleware.append(AutoRecallMiddleware(memory, user))
    doc_sources = None
    if settings.rag_enabled:
        index = document_index(settings)
        stack.callback(index.close)
        tools.append(index.tool())
        doc_sources = index.sources()  # listed in the prompt so the agent knows to search
    backend, prefix, workdir = sandbox_backend(settings, stack, say)
    agent = build_agent(
        chat_model(settings), saver, backend, tools, prefix, settings.memory.enabled, workdir,
        # The seeded sandbox skill describes the OpenShell image and policy.
        settings.sandbox.provider == "openshell",
        settings.memory.recall_mode, middleware, doc_sources)
    return Runtime(agent=agent, chats=Chats(saver, agent, user))


def history_runtime(settings: Settings, user: str, stack: ExitStack) -> Runtime:
    """Enough of the agent to read and manage saved chats; its model is never called."""
    saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
    agent = build_agent(ChatOpenAI(model="unused", api_key="unused"), saver,
                        FilesystemBackend(root_dir=str(settings.paths.workspace), virtual_mode=True))
    return Runtime(agent=agent, chats=Chats(saver, agent, user))

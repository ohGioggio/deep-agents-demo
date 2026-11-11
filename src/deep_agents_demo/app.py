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
from deep_agents_demo.guardrails import Guardrails
from deep_agents_demo.images import Uploads, images_feature
from deep_agents_demo.catalog import catalog_feature
from deep_agents_demo.memory import (MemoryService, close_memory, file_memory_feature, make_memory,
                                     memory_feature)
from deep_agents_demo.rag import DocumentIndex, documents_feature
from deep_agents_demo.sandbox import BackendSetup, make_backend
from deep_agents_demo.skill_manager import SkillManager
from deep_agents_demo.skill_store import SkillStore
from deep_agents_demo.threads import Chats
from deep_agents_demo.tools import ToolRegistry
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


def sandbox_backend(settings: Settings, stack: ExitStack, say=print) -> BackendSetup:
    """Local files or the configured sandbox (which connects lazily)."""
    return make_backend(settings.sandbox, settings.paths.workspace, settings.paths.data, stack, say)


def prepare_workspace(settings: Settings) -> None:
    seed_files(settings.paths.workspace)
    settings.paths.data.mkdir(exist_ok=True)


@dataclass
class Runtime:
    """Everything a chat session needs, all closed by the ExitStack it was built with."""
    agent: object
    chats: Chats
    preview: object = None  # describes a paused tool call for the approval prompt
    uploads: object = None  # saves images the user attaches


def skill_manager(settings: Settings, user: str, stack: ExitStack) -> SkillManager:
    store = SkillStore(settings.paths.data / "skills.sqlite")
    stack.callback(store.close)
    manager = SkillManager(store, user, settings.paths.workspace)
    manager.sync_folder()  # the database is the source of truth
    return manager


def features(settings: Settings, user: str, stack: ExitStack, setup, skills=None, guardrails=None):
    """The features this configuration turns on, in prompt order."""
    yield catalog_feature()  # example tools, always available
    yield images_feature(setup.skills_prefix)  # attached images; check rendered ones
    if skills is not None:
        yield skills.feature()
    if settings.memory.enabled:
        memory = make_memory(settings.paths.data, settings.memory, settings.embeddings,
                             embeddings(settings))
        stack.callback(close_memory, memory)
        service = MemoryService(memory, user)
        if guardrails is not None:
            guardrails.describe_memory = service.describe  # show what `forget` would remove
        yield memory_feature(service, settings.memory.recall_mode)
    else:
        yield file_memory_feature(setup.skills_prefix)
    if settings.rag_enabled:
        index = document_index(settings)
        stack.callback(index.close)
        yield documents_feature(index)
    if setup.feature is not None:
        yield setup.feature  # how to work in the sandbox
    if guardrails is not None:
        yield guardrails.feature()  # risky actions pause for the user


def build_runtime(settings: Settings, user: str, stack: ExitStack, say=print) -> Runtime:
    """The full agent for one user: model, features, sandbox, checkpoints."""
    saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
    setup = sandbox_backend(settings, stack, say)
    skills = skill_manager(settings, user, stack)
    guardrails = Guardrails()
    enabled = list(features(settings, user, stack, setup, skills, guardrails))
    registry = ToolRegistry(enabled)
    for problem in registry.problems(settings.disabled_tools):
        say(f"Warning: {problem}")
    skills.known_tools = registry.names()  # a skill may only declare tools that exist
    agent = build_agent(chat_model(settings), saver, setup.backend, enabled,
                        skills_prefix=setup.skills_prefix, disabled_tools=settings.disabled_tools)

    def preview(name, args):
        return skills.plan(name, args) or guardrails.preview(name, args)

    return Runtime(agent=agent, chats=Chats(saver, agent, user), preview=preview,
                   uploads=Uploads(settings.paths.workspace, user, setup.skills_prefix))


def history_runtime(settings: Settings, user: str, stack: ExitStack) -> Runtime:
    """Enough of the agent to read and manage saved chats; its model is never called."""
    saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
    agent = build_agent(ChatOpenAI(model="unused", api_key="unused"), saver,
                        FilesystemBackend(root_dir=str(settings.paths.workspace), virtual_mode=True))
    return Runtime(agent=agent, chats=Chats(saver, agent, user))

"""Composition root, per user: the agent and everything a chat session needs.

Everything else receives what it needs as arguments. To see how the demo is
wired (which model, memory, documents and sandbox an agent gets), read this
module and ``resources.py`` (what is shared by every user); to change the wiring,
change it here. A new feature is one more line in ``features()``.
"""
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field

from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.config import Settings
from deep_agents_demo.core.assembly import build_agent, worker_factory
from deep_agents_demo.core.tools import ToolRegistry, tool_name
from deep_agents_demo.features.agents.builtins import BUILTIN_AGENTS
from deep_agents_demo.features.agents.manager import AgentManager
from deep_agents_demo.features.agents.registry import AgentRegistry
from deep_agents_demo.features.catalog.feature import catalog_feature
from deep_agents_demo.features.documents.feature import documents_feature
from deep_agents_demo.features.files.feature import files_feature
from deep_agents_demo.features.guardrails.feature import Guardrails
from deep_agents_demo.features.images.feature import images_feature
from deep_agents_demo.features.images.uploads import Uploads
from deep_agents_demo.features.memory.feature import file_memory_feature, memory_feature
from deep_agents_demo.features.memory.service import (
    MemoryService,
)
from deep_agents_demo.features.orchestration.feature import orchestration_feature
from deep_agents_demo.features.skills.manager import SkillManager
from deep_agents_demo.persistence.chats import Chats
from deep_agents_demo.persistence.outputs import OutputStore
from deep_agents_demo.persistence.runs import RunStore
from deep_agents_demo.providers.chat import chat_model
from deep_agents_demo.startup.resources import Resources


def _quiet(*_args, **_kwargs) -> None:
    return None


@dataclass
class Runtime:
    """Everything a chat session needs, all closed by the ExitStack it was built with."""
    agent: object
    chats: Chats
    preview: object = None  # describes a paused tool call for the approval prompt
    uploads: object = None  # saves images the user attaches
    agents: object = None   # the AgentRegistry the agent was built from
    agent_manager: object = None  # the user's agents; set ``changed`` after an approved change
    rebuild: object = None  # () -> (agent, AgentRegistry, make_worker), with the same settings
    say: object = _quiet    # notes for the user outside a turn
    make_worker: object = None  # (name, extra_middleware) -> a standalone agent, for plan runners
    outputs: object = None  # the OutputStore: files the agent delivered, for front ends to list
    lookup_tools: frozenset = frozenset()  # the enabled lookup tools (swarm workers may use them too)
    lock: object = field(default_factory=threading.RLock, repr=False, compare=False)  # one rebuild at a time

    def refresh(self) -> bool:
        """After an approved agent change, rebuild: a new agent is usable at once (task, rooms,
        plans, swarms), a changed one from the next message. Parallel callers rebuild once."""
        with self.lock:
            if self.agent_manager is None or not self.agent_manager.changed or self.rebuild is None:
                return False
            try:
                self.agent, self.agents, self.make_worker = self.rebuild()
            except Exception as error:  # keep chatting with the agents that worked
                self.say(f"Couldn't rebuild the agents ({error}); the previous ones stay in use. "
                         "It will be tried again after your next message.")
                return False
            self.chats.agent = self.agent
            self.agent_manager.changed = False
            return True


def newest_worker_factory(runtime):
    """The worker factory of the newest agents: a waiting agent change is rebuilt in first, so a
    ``task``, plan, room or swarm can use an agent approved earlier in this message. None when
    the runtime has no workers. The one path to the newest workers (``task`` and the chat's runners)."""
    refresh = getattr(runtime, "refresh", None)
    if refresh is not None:
        refresh()
    return runtime.make_worker


def skill_manager(resources: Resources, user: str) -> SkillManager:
    manager = SkillManager(resources.skill_store, user, resources.settings.paths.workspace)
    manager.sync_folder()  # the database is the source of truth
    return manager


def agent_manager(resources: Resources, user: str) -> AgentManager:
    return AgentManager(resources.agent_store, user, frozenset())  # known tools are set once features exist


def features(resources: Resources, user: str, skills=None, guardrails=None, agents=None):
    """The features this configuration turns on, in prompt order."""
    settings, setup = resources.settings, resources.sandbox
    yield catalog_feature()  # example tools, always available
    yield images_feature(setup.skills_prefix)  # attached images; check rendered ones
    files = files_feature(setup.backend, resources.outputs, user, settings.file_output)
    if files is not None:
        yield files  # documents and data delivered as versioned files
    if skills is not None:
        yield skills.feature()
    if agents is not None:
        yield agents.feature()
    yield orchestration_feature()  # run_plan: subagent-driven execution with review
    if resources.memory is not None:
        service = MemoryService(resources.memory, user)  # this user's id, on the shared store
        if guardrails is not None:
            guardrails.describe_memory = service.describe  # show what `forget` would remove
        yield memory_feature(service, settings.memory.recall_mode)
    else:
        yield file_memory_feature(setup.skills_prefix)
    if resources.documents is not None:
        yield documents_feature(resources.documents)
    if setup.feature is not None:
        yield setup.feature  # how to work in the sandbox
    if guardrails is not None:
        yield guardrails.feature()  # risky actions pause for the user


def build_runtime(resources: Resources, user: str, notify=None) -> Runtime:
    """The full agent for one user, on the process's shared ``resources``. ``notify``
    gets notes for the user (warnings about tools and agents); default: silence."""
    say = notify or _quiet
    settings, saver, setup = resources.settings, resources.checkpointer, resources.sandbox
    skills = skill_manager(resources, user)
    my_agents = agent_manager(resources, user)
    guardrails = Guardrails()
    enabled = list(features(resources, user, skills, guardrails, my_agents))
    registry = ToolRegistry(enabled)
    for problem in registry.problems(settings.disabled_tools):
        say(f"Warning: {problem}")
    skills.known_tools = registry.names()  # a skill may only declare tools that exist
    my_agents.known_tools = registry.names()  # so may an agent

    def load_agents():
        # Folders (AGENT_DIRS, then the workspace's agents/), then installed, then the
        # user's own: a later source wins on a name clash.
        installed, own, notes = my_agents.definitions()
        # The demo's researcher exists only when some feature has lookups.
        replaceable = ("general-purpose", "researcher") if any(f.lookups for f in enabled) \
            else ("general-purpose",)
        agents = AgentRegistry.load([*settings.agent_dirs, settings.paths.workspace / "agents"],
                                    registry.names(), installed=installed, own=own,
                                    replaceable=replaceable)
        agents.warnings += notes
        my_agents.registry = agents
        return agents

    runtime = None  # set below; a ``task`` for an agent made in this message refreshes it

    def newest_workers():
        return newest_worker_factory(runtime)

    def make_agent():
        agents = load_agents()
        shared = dict(skills_prefix=setup.skills_prefix, disabled_tools=settings.disabled_tools,
                      agents=agents.agents(), model_for=lambda name: chat_model(settings, name),
                      max_calls=settings.subagent_max_calls)
        agent = build_agent(chat_model(settings), saver, setup.backend, enabled,
                            max_parallel=settings.subagent_max_parallel,
                            newest_workers=newest_workers, **shared)
        # Plan workers and reviewers: the same agents and limits, run by the chat loop.
        make = worker_factory(chat_model(settings), saver, setup.backend, enabled,
                              builtins=tuple(BUILTIN_AGENTS.values()), **shared)
        return agent, agents, make

    agent, agents, make = make_agent()
    for warning in agents.warnings:
        say(f"Agents: {warning}")

    def preview(name, args):
        return skills.plan(name, args) or my_agents.plan(name, args) or guardrails.preview(name, args)

    runtime = Runtime(agent=agent, chats=Chats(saver, agent, user, outputs=resources.outputs, runs=resources.runs),
                      preview=preview,
                      uploads=Uploads(settings.paths.workspace, user, setup.skills_prefix),
                      agents=agents, agent_manager=my_agents, rebuild=make_agent, say=say,
                      make_worker=make, outputs=resources.outputs,
                      lookup_tools=frozenset(tool_name(tool) for f in enabled for tool in f.lookups)
                      - frozenset(settings.disabled_tools))
    my_agents.refresh = runtime.refresh  # list_agents shows an agent approved in this message
    return runtime


def history_runtime(settings: Settings, user: str, stack: ExitStack) -> Runtime:
    """Enough of the agent to read and manage saved chats; its model is never called."""
    settings.paths.data.mkdir(parents=True, exist_ok=True)  # a fresh checkout has no data dir yet
    saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.paths.checkpoints)))
    agent = build_agent(ChatOpenAI(model="unused", api_key="unused"), saver,
                        FilesystemBackend(root_dir=str(settings.paths.workspace), virtual_mode=True))
    outputs = OutputStore(settings.paths.data / "outputs")
    runs = RunStore(settings.paths.data / "runs.sqlite")
    stack.callback(runs.close)
    return Runtime(agent=agent, chats=Chats(saver, agent, user, outputs=outputs, runs=runs), outputs=outputs)

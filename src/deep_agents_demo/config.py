"""Project paths and typed settings.

Settings are read from the environment once, by ``Settings.from_env()``, and
passed to whatever needs them. No other module reads environment variables, so
every dependency on configuration is visible in a function signature, and
tests build a ``Settings`` instead of patching ``os.environ``.
"""
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# The package is installed in editable mode from src/, so the project root is
# two levels above this package. .env, data/ and demo_workspace/ live there,
# which keeps the demo working when launched from any directory.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WORKSPACE = ROOT / "demo_workspace"
CHECKPOINTS = ROOT / "demo_checkpoints.sqlite"
ENV_FILE = ROOT / ".env"

RECALL_MODES = ("tool", "auto", "both")
FILE_OUTPUT_MODES = ("auto", "always", "never")
SANDBOX_PROVIDERS = ("none", "daytona", "openshell")
PLACEHOLDERS = {"your-api-key", "your-model-id"}


@dataclass(frozen=True)
class ChatSettings:
    api_key: str | None
    model: str | None
    base_url: str | None

    def problem(self) -> str | None:
        """Why a chat model can't be used, or None if it can."""
        if not self.api_key or not self.model or {self.api_key, self.model} & PLACEHOLDERS:
            return ("Fill in OPENAI_API_KEY and AGENT_MODEL in .env in the project root "
                    "(or set them in your environment).")
        return None


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    provider: str = "auto"
    dimensions: int = 1536
    batch_size: int = 16


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = True
    recall_mode: str = "auto"
    # Mem0's extraction model; from_env fills blanks from the chat settings.
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class SandboxSettings:
    provider: str = "none"
    daytona_api_key: str | None = None
    daytona_sandbox_id: str | None = None
    openshell_name: str = "deep-agents-demo"
    openshell_workspace: str = "default"
    openshell_workdir: str = "/sandbox"


@dataclass(frozen=True)
class Paths:
    data: Path = DATA
    workspace: Path = WORKSPACE
    checkpoints: Path = CHECKPOINTS


@dataclass(frozen=True)
class Settings:
    chat: ChatSettings
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    rag_enabled: bool = True
    requests_per_minute: float = 0.0  # 0 = no cap
    disabled_tools: tuple = ()        # tool names hidden from the agent and refused
    agent_dirs: tuple = ()            # extra folders of agent files (before demo_workspace/agents)
    subagent_max_calls: int = 25      # model calls per subagent run
    plan_max_attempts: int = 2        # reviews a plan step may fail before the user is asked
    orchestration_max_calls: int = 150  # model calls one plan (or room) may use before asking
    subagent_max_parallel: int = 3    # subagents running at the same time
    room_max_turns: int = 12          # participant turns in a moderated room
    file_output: str = "auto"         # FILE_OUTPUT: deliver documents as files (auto, always, never)
    paths: Paths = field(default_factory=Paths)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Read and validate every setting. Raises ValueError with a readable message."""

        def get(name, default=None):
            return (env.get(name) or "").strip() or default

        def flag(name, default="true"):
            return get(name, default).lower() == "true"

        def number(name, default, kind=int):
            raw = get(name, str(default))
            try:
                return kind(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number, not {raw!r}") from None

        chat = ChatSettings(api_key=get("OPENAI_API_KEY"), model=get("AGENT_MODEL"),
                            base_url=get("OPENAI_BASE_URL"))
        recall_mode = get("MEMORY_RECALL", "auto").lower()
        if recall_mode not in RECALL_MODES:
            raise ValueError(f"MEMORY_RECALL must be one of: {', '.join(RECALL_MODES)}")
        file_output = get("FILE_OUTPUT", "auto").lower()
        if file_output not in FILE_OUTPUT_MODES:
            raise ValueError(f"FILE_OUTPUT must be one of: {', '.join(FILE_OUTPUT_MODES)}")
        provider = get("SANDBOX_PROVIDER", "none").lower()
        if provider not in SANDBOX_PROVIDERS:
            raise ValueError(f"SANDBOX_PROVIDER must be one of: {', '.join(SANDBOX_PROVIDERS)}")
        if provider == "daytona" and not get("DAYTONA_API_KEY"):
            raise ValueError("Set DAYTONA_API_KEY to enable sandbox execution")
        batch_size = number("EMBEDDING_BATCH_SIZE", 16)
        if batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")
        def at_least_one(name, default):
            value = number(name, default)
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
            return value

        max_calls = at_least_one("SUBAGENT_MAX_CALLS", 25)
        try:
            agent_dirs = tuple(Path(part.strip()).expanduser()
                               for part in get("AGENT_DIRS", "").split(os.pathsep) if part.strip())
        except RuntimeError as error:  # e.g. ~someone for a user that doesn't exist
            raise ValueError(f"AGENT_DIRS: {error}") from None
        return cls(
            chat=chat,
            embeddings=EmbeddingSettings(
                model=get("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=get("EMBEDDING_API_KEY", chat.api_key),
                base_url=get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
                provider=get("EMBEDDING_PROVIDER", "auto"),
                dimensions=number("EMBEDDING_DIMENSIONS", 1536),
                batch_size=batch_size,
            ),
            memory=MemorySettings(
                enabled=flag("MEMORY_ENABLED"),
                recall_mode=recall_mode,
                model=get("MEMORY_MODEL", chat.model),
                api_key=get("MEMORY_API_KEY", chat.api_key),
                base_url=get("MEMORY_BASE_URL", chat.base_url),
            ),
            sandbox=SandboxSettings(
                provider=provider,
                daytona_api_key=get("DAYTONA_API_KEY"),
                daytona_sandbox_id=get("DAYTONA_SANDBOX_ID"),
                openshell_name=get("OPENSHELL_SANDBOX_NAME", "deep-agents-demo"),
                openshell_workspace=get("OPENSHELL_WORKSPACE", "default"),
                openshell_workdir=get("OPENSHELL_WORKDIR", "/sandbox"),
            ),
            file_output=file_output,
            rag_enabled=flag("RAG_ENABLED"),
            requests_per_minute=number("API_REQUESTS_PER_MINUTE", 0, float),
            disabled_tools=tuple(dict.fromkeys(
                name for name in re.split(r"[\s,]+", get("DISABLED_TOOLS", "")) if name)),
            agent_dirs=agent_dirs,
            subagent_max_calls=max_calls,
            plan_max_attempts=at_least_one("PLAN_MAX_ATTEMPTS", 2),
            orchestration_max_calls=at_least_one("ORCHESTRATION_MAX_CALLS", 150),
            subagent_max_parallel=at_least_one("SUBAGENT_MAX_PARALLEL", 3),
            room_max_turns=at_least_one("ROOM_MAX_TURNS", 12),
        )



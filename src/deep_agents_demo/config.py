"""Project paths and environment settings."""
import os
from pathlib import Path

# The package is installed in editable mode from src/, so the project root is
# two levels above this package. .env, data/ and demo_workspace/ live there,
# which keeps the demo working when launched from any directory.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WORKSPACE = ROOT / "demo_workspace"
CHECKPOINTS = ROOT / "demo_checkpoints.sqlite"
ENV_FILE = ROOT / ".env"


def setting(name, fallback=None):
    """Return a stripped environment variable, or ``fallback`` when blank."""
    return os.getenv(name, "").strip() or fallback


def enabled(name, default="true"):
    return setting(name, default).lower() == "true"


def embedding_settings():
    return {
        "model": setting("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": setting("EMBEDDING_API_KEY", setting("OPENAI_API_KEY")),
        "base_url": setting("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        "provider": setting("EMBEDDING_PROVIDER", "auto"),
    }

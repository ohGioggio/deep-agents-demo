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


_limiters = {}


def shared_rate_limiter():
    """One limiter for every chat and embedding request, or None if unlimited.

    API_REQUESTS_PER_MINUTE caps the total across chat and embeddings, because a
    provider's limit (NVIDIA's free tier: about 40 per minute) usually applies
    to the API key, not to each kind of request.
    """
    rpm = float(setting("API_REQUESTS_PER_MINUTE", "0") or 0)
    if rpm <= 0:
        return None
    if rpm not in _limiters:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _limiters[rpm] = InMemoryRateLimiter(requests_per_second=rpm / 60,
                                             check_every_n_seconds=0.1, max_bucket_size=1)
    return _limiters[rpm]


def embedding_settings():
    return {
        "model": setting("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": setting("EMBEDDING_API_KEY", setting("OPENAI_API_KEY")),
        "base_url": setting("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        "provider": setting("EMBEDDING_PROVIDER", "auto"),
        "rate_limiter": shared_rate_limiter(),
        "batch_size": int(setting("EMBEDDING_BATCH_SIZE", "16")),
    }

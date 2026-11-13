"""The chat model client."""
from langchain_openai import ChatOpenAI

from deep_agents_demo.config import Settings
from deep_agents_demo.providers.rate_limit import rate_limiter


def chat_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    """The configured chat model, or another ``model`` id on the same endpoint and key."""
    # Shares the request cap with the embeddings; retries ride out transient errors.
    return ChatOpenAI(model=model or settings.chat.model, api_key=settings.chat.api_key,
                      base_url=settings.chat.base_url,
                      rate_limiter=rate_limiter(settings.requests_per_minute), max_retries=6,
                      stream_usage=True)  # token counts on streamed answers too

"""The chat model client."""
import threading

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from deep_agents_demo.config import Settings
from deep_agents_demo.providers.rate_limit import rate_limiter
from deep_agents_demo.providers.usage import CURRENT


class UsageCounter(BaseCallbackHandler):
    """Counts the model calls and the tokens of every answer it sees. Thread-safe: subagents,
    plan workers and room turns run on several threads. Each answer also goes to the tally of
    the chat turn that started it (providers/usage.py), when one is running."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = self._input_tokens = self._output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        if not tokens_in and not tokens_out:  # a provider that reports usage in llm_output instead
            usage = (response.llm_output or {}).get("token_usage") or {}
            tokens_in, tokens_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        with self._lock:
            self._calls += 1
            self._input_tokens += tokens_in
            self._output_tokens += tokens_out
        tally = CURRENT.get()
        if tally is not None:
            tally.add(tokens_in, tokens_out)

    def snapshot(self) -> tuple[int, int, int]:
        """(model calls, input tokens, output tokens): the running chat turn's own tally when one
        is running in this context (a turn starts at zero), else the process's totals."""
        tally = CURRENT.get()
        if tally is not None:
            return tally.snapshot()
        with self._lock:
            return self._calls, self._input_tokens, self._output_tokens


_USAGE = UsageCounter()  # one for the process: every chat model the app builds reports to it


def usage_counter() -> UsageCounter:
    return _USAGE


def chat_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    """The configured chat model, or another ``model`` id on the same endpoint and key."""
    # Shares the request cap with the embeddings; retries ride out transient errors, and the
    # timeout turns a hung request into one of them instead of a wait without end.
    return ChatOpenAI(model=model or settings.chat.model, api_key=settings.chat.api_key,
                      base_url=settings.chat.base_url,
                      rate_limiter=rate_limiter(settings.requests_per_minute), max_retries=6,
                      timeout=settings.request_timeout,
                      stream_usage=True,  # token counts on streamed answers too
                      callbacks=[_USAGE])  # counted for the run store (persistence/runs.py)

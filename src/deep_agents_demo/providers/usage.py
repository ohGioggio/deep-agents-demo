"""The model usage of the chat turn that is running, found from any thread the turn starts.

A turn opens a ``Tally`` and sets ``CURRENT`` to it. Every thread the turn starts carries the
turn's context (``DaemonPool`` and LangChain's and LangGraph's executors copy it), so a model call
made there lands in the tally of the turn that started it. A call that ends after its turn
closed still adds to that closed tally, never to the next turn's.
"""
import contextvars
import threading


class Tally:
    """The model calls and tokens of one turn. Thread-safe: its calls come from several threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = self._input_tokens = self._output_tokens = 0

    def add(self, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self._calls += 1
            self._input_tokens += tokens_in
            self._output_tokens += tokens_out

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self._calls, self._input_tokens, self._output_tokens


CURRENT: contextvars.ContextVar[Tally | None] = contextvars.ContextVar("chat_turn_usage", default=None)

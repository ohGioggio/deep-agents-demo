"""A fake chat model that streams: text in word chunks, then a last chunk with usage.

``test_agents.Model`` answers each call with one whole message; this one streams
it the way a real provider does, so the engine's "messages" stream sees pieces.
"""
import json
import threading
import uuid

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from deep_agents_demo.core.tools import tool_name

SEEN = []  # every model call: ("tools", [names]) then ("messages", [...])
USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
_LOCK = threading.Lock()  # parallel subagents call the model from several threads


class StreamingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        SEEN.append(("tools", [tool_name(t) for t in tools]))
        return self

    def _next(self):
        with _LOCK:
            response = self.responses[self.i]
            self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0
        return response

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        SEEN.append(("messages", list(messages)))
        response = self._next()
        message_id = response.id or f"run-{uuid.uuid4()}"
        text = response.text if isinstance(response.text, str) else str(response.content)
        words = text.split(" ")
        pieces = [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)] if text else []
        for piece in pieces:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece, id=message_id))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
        calls = getattr(response, "tool_calls", None) or []
        if calls:
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content="", id=message_id,
                tool_call_chunks=[{"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                                  for i, c in enumerate(calls)]))
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
        last = ChatGenerationChunk(message=AIMessageChunk(content="", id=message_id, usage_metadata=dict(USAGE),
                                                          chunk_position="last"))
        if run_manager:
            run_manager.on_llm_new_token("", chunk=last)
        yield last

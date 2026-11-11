"""Subagent-driven execution: the parallel cap on ``task``, and plans with review.

- ``TaskLimiter``: at most N subagents run at once; extra ``task`` calls wait.
"""
import threading

from langchain.agents.middleware import AgentMiddleware


class TaskLimiter(AgentMiddleware):
    """At most ``limit`` ``task`` calls run at the same time (LangGraph runs a step's
    tool calls in parallel threads); the rest wait for a free slot."""

    def __init__(self, limit: int):
        super().__init__()
        self.slots = threading.BoundedSemaphore(limit)

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "task":
            return handler(request)
        with self.slots:
            return handler(request)

"""The ports the engine talks to a front end through (Ports and Adapters).

Small protocols rather than one UI object: a renderer needs only ``EventSink``,
an approval prompt only ``Interaction``. The CLI implements them in ``ui/cli/``;
a GUI or a server would implement the same three. All are synchronous: a front end
that isn't runs the engine on a worker thread.
"""
from typing import Protocol, runtime_checkable

from deep_agents_demo.models.approvals import ApprovalRequest, Decision


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event) -> None:
        """Show one ``models.events`` event."""


@runtime_checkable
class Interaction(Protocol):
    def decide(self, request: ApprovalRequest) -> Decision:
        """Approve or reject a paused tool call."""

    def choose(self, prompt: str, options: tuple[str, ...]) -> str:
        """Pick one of ``options`` (e.g. retry, skip, stop)."""

    def ask(self, prompt: str) -> str | None:
        """Free text from the user; None if there is nobody to ask."""


@runtime_checkable
class Interruptible(Protocol):
    def request_pause(self) -> None:
        """Pause at the next safe point and hand the user the turn."""

    def stop(self) -> None:
        """Stop as soon as possible."""

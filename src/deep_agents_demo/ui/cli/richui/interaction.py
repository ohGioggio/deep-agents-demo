"""The Rich ``Interaction``: an approval panel with keys, a coloured diff, styled prompts.

Same answers and the same messages for the agent as ``CliInteraction``. Every read first
takes the renderer's Live region down (``pause``) and brings it back afterwards (``resume``).
"""
from collections.abc import Callable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from deep_agents_demo.models.approvals import ApprovalRequest, Decision


def _keys_line() -> Text:
    line = Text()
    for key, rest in (("a", "pprove"), ("r", "eject (with a reason)"), ("d", "iff"), ("f", "ull file")):
        line.append("[")
        line.append(key, style="bold approval")
        line.append("]" + rest + "  ")
    line.rstrip()
    line.append(" > ")
    return line


class RichInteraction:
    def __init__(self, console: Console, renderer=None, ask: Callable[[str], str] | None = None):
        self._console = console
        self._renderer = renderer
        self._ask = ask

    def _read(self, prompt: Text) -> str:
        """One line from the user, with Live down while we wait."""
        if self._renderer is not None:
            self._renderer.pause()
        try:
            if self._ask is not None:
                return self._ask(prompt.plain)
            return self._console.input(prompt)
        finally:
            if self._renderer is not None:
                self._renderer.resume()

    def decide(self, request: ApprovalRequest) -> Decision:
        self._console.print(Panel(Text(request.summary), title=Text(request.title, style="approval"),
                                  title_align="left", border_style="approval", box=box.ROUNDED,
                                  padding=(0, 1)))
        while True:
            try:
                answer = self._read(_keys_line())
            except (EOFError, KeyboardInterrupt):
                return Decision(False, "The user did not approve this change.")
            answer = answer.strip().lower()
            if answer in {"a", "approve", "y", "yes"}:
                return Decision(True)
            if answer in {"r", "reject", "n", "no"}:
                try:
                    reason = self._read(Text("Reason for the agent (optional) > ", style="approval")).strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                return Decision(False, "The user rejected this change"
                                + (f": {reason}" if reason else ". Don't retry unless they ask."))
            if answer in {"d", "diff"}:
                self._console.print(Syntax(request.diff, "diff", theme="ansi_dark",
                                           background_color="default", word_wrap=True))
            elif answer in {"f", "full"}:
                self._console.print(Text(request.full))
            else:
                self._console.print(Text("Type a, r, d or f.", style="muted"))

    def choose(self, prompt: str, options: tuple[str, ...]) -> str:
        """The typed answer (the caller matches it to ``options``); "" if none."""
        try:
            return self._read(Text(prompt, style="approval"))
        except (EOFError, KeyboardInterrupt):
            return ""

    def ask(self, prompt: str) -> str | None:
        try:
            return self._read(Text(prompt, style="approval"))
        except (EOFError, KeyboardInterrupt):
            return None

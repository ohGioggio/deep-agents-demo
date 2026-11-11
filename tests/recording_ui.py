"""A test front end: implements the ports, records what the CLI would print.

Events are rendered through the real ``CliRenderer`` into ``lines``, so a test
asserting on ``lines`` asserts on the CLI's own text. ``answers`` script the user:
each ``decide``/``choose``/``ask`` pops the next one.
"""
from deep_agents_demo.models.approvals import ApprovalRequest, Decision
from deep_agents_demo.ui.cli.render import CliRenderer


class RecordingUI:
    def __init__(self, answers=()):
        self.lines: list[str] = []
        self.answers = list(answers)
        self.asked: list[str] = []
        self.requests: list[ApprovalRequest] = []
        self.renderer = CliRenderer(write=self.lines.append)

    # EventSink
    def emit(self, event) -> None:
        self.renderer.emit(event)

    def notify(self, text: str) -> None:
        self.lines.append(text)

    # Interaction
    def _next(self):
        if not self.answers:
            raise EOFError("RecordingUI ran out of scripted answers")
        return self.answers.pop(0)

    def decide(self, request: ApprovalRequest) -> Decision:
        self.requests.append(request)
        answer = self._next()
        return answer if isinstance(answer, Decision) else Decision(approved=answer == "a", message=str(answer))

    def choose(self, prompt: str, options: tuple[str, ...]) -> str:
        self.asked.append(prompt)
        return self._next()

    def ask(self, prompt: str) -> str | None:
        self.asked.append(prompt)
        return self._next()

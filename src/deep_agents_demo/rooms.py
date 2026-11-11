"""Rooms: a moderated discussion between agents on one question.

Before each turn the moderator (the built-in ``moderator`` agent, or the user's own
agent of that name) decides who speaks next and what they should address, or that
the room is done. Participants are the user's agents, with their own tools, limits
and approvals. The room ends with a summary (agreements, open disagreements, next
steps); the transcript is saved to ``demo_workspace/rooms/<user>/``.

Like plans, rooms run on the chat loop's thread (``Runner``), so approvals work and
Ctrl+C pauses between turns, here to let the user speak.
"""
import re
import time
import uuid
from pathlib import Path

from deep_agents_demo.orchestration import Budget, Runner, _RunFailed, _WrapUp

TRANSCRIPT_LIMIT = 6000   # characters of the discussion each turn sees (the latest ones)
_MARKS = "*#>_` \t"
REPLY_FORMAT = (
    "Reply in exactly one of these two forms.\n"
    "NEXT: <participant name>\n<what they should address>\n\n"
    "END\n<the summary: what was agreed, the open disagreements, the next steps>\n\n"
    "End when they agree, when nothing new is being said, or when the user asks.")


def parse_moderator(text: str, participants: list[str]) -> tuple[str, str, str]:
    """("next", name, prompt), ("end", "", summary), or ("unclear", "", text)."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if lines:
        first, rest = lines[0].strip(_MARKS), "\n".join(lines[1:]).strip()
        upper = first.upper()
        if upper.startswith("END"):
            return "end", "", rest or first[3:].strip(" :-")
        if upper.startswith("NEXT"):
            name = first[4:].strip(" :-").strip(_MARKS).strip()
            match = next((p for p in participants if p.lower() == name.lower()), None)
            if match:
                return "next", match, rest
    return "unclear", "", str(text or "").strip()


def _slug(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", question.lower()[:40]).strip("-") or "room"


class RoomRunner(Runner):
    label = "[room]"

    def __init__(self, make, approver, *, thread: str, max_turns: int = 12, budget: int = 150,
                 interactive: bool = True, ask=input, say=print, transcripts=None, describe=None):
        super().__init__(make, approver, thread=thread, budget=budget, interactive=interactive,
                         ask=ask, say=say)
        self.max_turns = max_turns
        self.transcripts = Path(transcripts) if transcripts else None
        self.describe = describe or (lambda name: "")
        self.lines: list[str] = []
        self.user_said = ""
        self.room_id = ""

    def transcript_text(self) -> str:
        return "\n\n".join(self.lines)

    def _trimmed(self) -> str:
        text = self.transcript_text()
        if not text:
            return "(nothing said yet)"
        if len(text) <= TRANSCRIPT_LIMIT:
            return text
        return "(earlier turns omitted)\n\n..." + text[-TRANSCRIPT_LIMIT:]

    def _moderator_input(self, question, participants, turns, final=False) -> str:
        people = "\n".join(f"- {name}" + (f": {self.describe(name)}" if self.describe(name) else "")
                           for name in participants)
        parts = [f"Question: {question}", f"Participants:\n{people}",
                 f"Turns used: {turns} of {self.max_turns}.", f"Discussion so far:\n{self._trimmed()}"]
        if self.user_said:
            parts.append(f"The user just said: {self.user_said}")
        parts.append("Write the summary now: what was agreed, the open disagreements, and the "
                     "next steps." if final else REPLY_FORMAT)
        return "\n\n".join(parts)

    def _turn_input(self, question, prompt) -> str:
        return (f"Question: {question}\n\nDiscussion so far:\n{self._trimmed()}\n\n"
                f"The moderator asks you: {prompt}\n\n"
                "Reply with your contribution only: a few short paragraphs.")

    def _summary(self, question, participants, turns) -> str:
        try:
            text = self._run_agent("moderator", self._moderator_input(question, participants, turns,
                                                                       final=True),
                                   f"{self.thread}:room:{self.room_id}:summary:moderator", capped=False)
        except _RunFailed as error:
            return f"(no summary: {error})"
        kind, _name, body = parse_moderator(text, participants)
        return body if kind == "end" else text.strip()

    def _save(self, question, participants, summary) -> str:
        if self.transcripts is None:
            return ""
        self.transcripts.mkdir(parents=True, exist_ok=True)
        path = self.transcripts / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(question)}.md"
        body = [f"# Room: {question}", f"Participants: {', '.join(participants)}", *self.lines]
        if summary:
            body += ["## Summary", summary]
        path.write_text("\n\n".join(body) + "\n", encoding="utf-8")
        return str(path)

    def run(self, request: dict) -> dict:
        return self._with_ctrl_c(self._run, request)

    def _run(self, request: dict) -> dict:
        self.budget, self.room_id = Budget(self.budget_size), uuid.uuid4().hex[:8]
        self.lines, self.user_said = [], ""
        question = str(request.get("question", "")).strip()
        participants = [str(p) for p in request.get("participants", [])]
        base = f"{self.thread}:room:{self.room_id}"
        turns = decisions = rotation = 0
        ended_by, summary = "", ""
        try:
            while True:
                if self.pause_requested:
                    choice = self._step_in()
                    if choice in ("user", "stopped"):
                        ended_by = choice
                        break
                if turns >= self.max_turns:
                    ended_by = "turn limit"
                    break
                decisions += 1
                decision = self._run_agent("moderator",
                                           self._moderator_input(question, participants, turns),
                                           f"{base}:{decisions}:moderator")
                self.user_said = ""
                kind, name, text = parse_moderator(decision, participants)
                if kind == "end":
                    ended_by, summary = "moderator", text
                    break
                if kind == "unclear":
                    name = participants[rotation % len(participants)]
                    self.lines.append(f"*(the moderator named no participant; {name} speaks next)*")
                rotation = participants.index(name) + 1
                prompt = text or "What is your view?"
                self.say(f"{self.label} moderator -> {name}: {prompt.splitlines()[0][:200]}")
                self.lines.append(f"**Moderator:** {prompt}")
                try:
                    answer = self._run_agent(name, self._turn_input(question, prompt),
                                             f"{base}:{decisions}:{name}")
                    self.lines.append(f"**{name}:** {answer}")
                except _RunFailed as error:
                    self.lines.append(f"*({name} could not answer: {error})*")
                turns += 1
        except _WrapUp:
            ended_by = "budget"
        if ended_by in ("turn limit", "user", "budget"):
            summary = self._summary(question, participants, turns)
        if summary:
            self.say(f"{self.label} summary:\n{summary}")
        return {"question": question, "participants": participants, "turns": turns,
                "summary": summary, "transcript": self._save(question, participants, summary),
                "ended_by": ended_by,
                "note": "Tell the user the outcome; the transcript has every turn."}

    def _step_in(self) -> str:
        """Ctrl+C was pressed (see Task 4): "go", "user" (/end) or "stopped" (/stop)."""
        self.pause_requested = False
        return "go"

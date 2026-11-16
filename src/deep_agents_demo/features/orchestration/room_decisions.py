"""What a room's moderator decides each time: who speaks next (and what they should
address), or that the room is done. Parsed from the moderator's reply: a JSON object
first, then the text format ("NEXT: name" / "END"), then rotation through the
participants when neither reads."""
import json
import re
from dataclasses import dataclass, field

_MARKS = "*_` \t"          # Markdown around a speaker's name in JSON
_TEXT_MARKS = "*#>_` \t"   # the same for the text format's first line
_DECISION = re.compile(r"(END|NEXT)\b[\s:\-]*(.*)$", re.IGNORECASE)
_POSITION_LINE = re.compile(r"^[\s*_`>\-]*position[*_`]*\s*:[*_`\s]*(.*)$", re.IGNORECASE)
_CONFIDENCE_TAIL = re.compile(r"\s*\(\s*confidence\s*:\s*([^)]*?)\s*\)[.!\s]*$", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_ASK = "What is your view?"


@dataclass(frozen=True)
class Turn:
    speaker: str
    ask: str


@dataclass(frozen=True)
class Minutes:
    agreed: tuple = ()
    disputed: tuple = ()
    open: tuple = ()

    def text(self) -> str:
        sections = (("Agreed", self.agreed), ("Disputed", self.disputed), ("Open", self.open))
        blocks = [f"{title}:\n" + "\n".join(f"- {item}" for item in items)
                  for title, items in sections if items]
        return "\n\n".join(blocks) or "(no minutes yet)"


@dataclass(frozen=True)
class Decision:
    action: str  # "next" | "end"
    turns: tuple = ()
    minutes: Minutes | None = None  # None: keep the previous minutes
    summary: str = ""
    decision: str = ""
    dissent: tuple = ()
    actions: tuple = ()
    source: str = "json"  # json | text | rotation | format


@dataclass
class RoomState:
    question: str = ""
    participants: list[str] = field(default_factory=list)
    turns_used: int = 0
    max_turns: int = 0
    minutes: Minutes = field(default_factory=Minutes)
    positions: dict = field(default_factory=dict)  # name -> {"position", "confidence"} (Task 4)
    recent: list[str] = field(default_factory=list)
    user_said: str = ""
    rotation_next: str = ""


def _strings(items) -> tuple:
    if not isinstance(items, list):
        return ()
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def _json_objects(text: str):
    """Each JSON object in the reply, in order: the body of a ```json fence if there is one,
    else the whole reply. A raw newline inside a string is allowed (a model's summary often
    has one), and a "{" that doesn't open an object is skipped."""
    fence = _FENCE.search(text)
    body = fence.group(1) if fence else text
    decoder = json.JSONDecoder(strict=False)
    for start, mark in enumerate(body):
        if mark != "{":
            continue
        try:
            value, _end = decoder.raw_decode(body, start)
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def _turns(items, participants: list[str]) -> tuple:
    by_name = {p.lower(): p for p in participants}
    seen: dict[str, Turn] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        name = speaker.strip(_MARKS).lower() if isinstance(speaker, str) else ""  # null: missing
        if name not in by_name or by_name[name] in seen:
            continue
        ask = item.get("ask")
        ask = ask.strip() if isinstance(ask, str) else ""
        seen[by_name[name]] = Turn(by_name[name], ask or DEFAULT_ASK)
    return tuple(seen.values())


def _from_json(text: str, participants: list[str]) -> Decision | None:
    """The first object in the reply that is a decision (an action, and valid turns if "next")."""
    for data in _json_objects(text):
        if data.get("action") not in ("next", "end"):
            continue
        if data["action"] == "next":
            turns = _turns(data.get("turns"), participants)
            if turns:
                return Decision("next", turns, minutes=_minutes(data.get("minutes")), source="json")
            continue
        summary, decision = data.get("summary"), data.get("decision")
        return Decision("end", minutes=_minutes(data.get("minutes")),
                        summary=summary.strip() if isinstance(summary, str) else "",
                        decision=decision.strip() if isinstance(decision, str) else "",
                        dissent=_strings(data.get("dissent")), actions=_strings(data.get("actions")),
                        source="json")
    return None


def _minutes(value) -> Minutes | None:
    if not isinstance(value, dict):
        return None
    return Minutes(agreed=_strings(value.get("agreed")), disputed=_strings(value.get("disputed")),
                   open=_strings(value.get("open")))


def parse_moderator(text: str, participants: list[str]) -> tuple[str, str, str]:
    """("next", name, prompt), ("end", "", summary), or ("unclear", "", text). END and
    NEXT must be whole words ("Endpoint ..." is not an ending)."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if lines:
        match = _DECISION.match(lines[0].strip(_TEXT_MARKS))
        rest = "\n".join(lines[1:]).strip()
        if match and match.group(1).upper() == "END":
            return "end", "", rest or match.group(2).strip()
        if match:
            name = match.group(2).strip(_TEXT_MARKS).strip(" .,;:!")
            found = next((p for p in participants if p.lower() == name.lower()), None)
            if found:
                return "next", found, rest
    return "unclear", "", str(text or "").strip()


def _past_broken_json(text: str) -> str:
    """A reply that opens with a JSON attempt that didn't parse: the text format is read
    from the lines after the broken JSON (its braces, fence lines and blank lines)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].strip()[:1] in "{}`"):
        i += 1
    return "\n".join(lines[i:]) if i and i < len(lines) else text


def parse_decision(text: str, participants: list[str], rotation_next: str) -> Decision:
    reply = str(text or "")
    decision = _from_json(reply, participants)
    if decision is not None:
        return decision
    kind, name, body = parse_moderator(_past_broken_json(reply), participants)
    if kind == "end":
        return Decision("end", summary=body, source="text")
    if kind == "next":
        return Decision("next", (Turn(name, body or DEFAULT_ASK),), source="text")
    return Decision("next", (Turn(rotation_next, DEFAULT_ASK),), source="rotation")


def parse_position(text: str) -> tuple[str, str] | None:
    """(position, confidence) from the last "POSITION: ..." line of a participant's reply, the
    confidence (low, medium, high; else "unknown") from its trailing "(confidence: X)". None when
    the reply has no such line."""
    found = None
    for line in str(text or "").splitlines():
        match = _POSITION_LINE.match(line)
        if not match:
            continue
        body, confidence = match.group(1), "unknown"
        tail = _CONFIDENCE_TAIL.search(body)
        if tail:
            body = body[:tail.start()]
            level = tail.group(1).strip().lower()
            confidence = level if level in ("low", "medium", "high") else "unknown"
        position = body.strip(_MARKS)
        if position:
            found = (position, confidence)
    return found

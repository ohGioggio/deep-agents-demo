"""Room formats: who speaks, in what order, and what the room tells its participants about
its shape. ``moderated`` (the default) leaves each round to the moderator (rooms.py). The
others are fixed: their policies below make one round per pick and no model call until the
summary, so a room's cost is its turns plus one summary."""
from collections.abc import Callable
from dataclasses import dataclass

from deep_agents_demo.config import ROOM_FORMATS
from deep_agents_demo.features.orchestration.room_decisions import Decision, RoomState, Turn


@dataclass(frozen=True)
class RoomFormat:
    name: str
    instructions: str        # one sentence, told to the participants and to the summary
    opening_round: bool      # blind openings before the first round (when the room has them)
    policy: str              # "moderator", or the fixed policy's name
    min_participants: int
    max_participants: int | None


FORMATS: dict[str, RoomFormat] = {
    "moderated": RoomFormat(
        "moderated", "The moderator decides who speaks next, round by round, and ends the room with a summary.",
        True, "moderator", 2, None),
    "roundrobin": RoomFormat(
        "roundrobin", "Everyone speaks in turn, in the order given, round after round.",
        False, "roundrobin", 2, None),
    "debate": RoomFormat(
        "debate", "This is a debate: the first participant argues for, the second against, the third judges.",
        False, "debate", 3, 3),
    "redteam": RoomFormat(
        "redteam", "This is a red team: the first participant proposes an answer, the others attack it, "
                   "and the proposer answers the attacks.",
        False, "redteam", 2, None),
}

ROUND_ASK = "Give your view, building on what was said."
DEBATE_FOR = "Argue for: answer the last argument against."
DEBATE_AGAINST = "Argue against: answer the last argument for."
VERDICT = "Weigh both sides and give your verdict: which side wins, and why."
PROPOSE = "Propose an answer to the question."
ATTACK = "Find the weaknesses in the proposal."
DEFEND = "Answer the attacks: defend or revise your proposal."
REATTACK = "Attack the revised proposal."


def format_problem(name: str, participants: list[str]) -> str | None:
    """Why this format can't run with these participants, or None if it can."""
    fmt = FORMATS.get(name)
    if fmt is None:
        return f"unknown room format {name!r}; use one of {', '.join(ROOM_FORMATS)}"
    count = len(participants)
    if fmt.min_participants == fmt.max_participants and count != fmt.min_participants:
        return f"the {name} format needs exactly {fmt.min_participants} participants"
    if count < fmt.min_participants:
        return f"the {name} format needs at least {fmt.min_participants} participants"
    if fmt.max_participants is not None and count > fmt.max_participants:
        return f"the {name} format takes at most {fmt.max_participants} participants"
    return None


def first_line(text: str) -> str:
    return next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")


class FixedPolicy:
    """A fixed order. ``answer_of(name)`` is a participant's latest full answer. The room
    ends (an empty summary, so the summary is written) when every named participant holds a
    position and none changed over the last round."""

    def __init__(self, answer_of: Callable[[str], str]):
        self.answer_of = answer_of
        self.before = None  # the positions at the previous pick

    def _settled(self, state: RoomState, names: list[str]) -> bool:
        now = {name: state.positions.get(name) for name in names}
        settled = self.before == now and all(now.values())
        self.before = now
        return settled

    def verdict(self) -> Decision:
        """The end of the room, for the record (a format may say more than its last round)."""
        return Decision("end", source="format")


class RoundRobinPolicy(FixedPolicy):
    def pick(self, state: RoomState) -> Decision:
        if self._settled(state, state.participants):
            return Decision("end", source="format")
        return Decision("next", tuple(Turn(name, ROUND_ASK) for name in state.participants), source="format")


class DebatePolicy(FixedPolicy):
    """The first argues for, the second against, alternating; the third judges, last. The
    judge's first line is the room's decision. The judge speaks as soon as the two sides
    have settled, or when there is no room for another round and the judge's turn."""

    def __init__(self, answer_of: Callable[[str], str]):
        super().__init__(answer_of)
        self.judge: str | None = None

    def pick(self, state: RoomState) -> Decision:
        for_name, against_name, judge = state.participants
        settled = self._settled(state, [for_name, against_name])
        if self.judge is not None:  # the verdict was the last turn
            return self.verdict()
        # The judge never opens the room: with no turn spent, a round of two is the first move.
        if settled or (state.turns_used > 0 and state.max_turns - state.turns_used < 3):
            self.judge = judge
            return Decision("next", (Turn(judge, VERDICT),), source="format")
        return Decision("next", (Turn(for_name, DEBATE_FOR), Turn(against_name, DEBATE_AGAINST)),
                        source="format")

    def verdict(self) -> Decision:
        if self.judge is None:
            return Decision("end", source="format")
        return Decision("end", decision=first_line(self.answer_of(self.judge)), source="format")


class RedTeamPolicy(FixedPolicy):
    """The first proposes; the others attack in order; the proposer answers; repeat."""

    def __init__(self, answer_of: Callable[[str], str]):
        super().__init__(answer_of)
        self.rounds = 0

    def pick(self, state: RoomState) -> Decision:
        if self._settled(state, state.participants):
            return Decision("end", source="format")
        proposer, attackers = state.participants[0], state.participants[1:]
        ask, attack = (PROPOSE, ATTACK) if self.rounds == 0 else (DEFEND, REATTACK)
        self.rounds += 1
        return Decision("next", (Turn(proposer, ask),) + tuple(Turn(name, attack) for name in attackers),
                        source="format")


FIXED_POLICIES = {"roundrobin": RoundRobinPolicy, "debate": DebatePolicy, "redteam": RedTeamPolicy}


def fixed_policy(policy: str, answer_of: Callable[[str], str]) -> FixedPolicy:
    return FIXED_POLICIES[policy](answer_of)

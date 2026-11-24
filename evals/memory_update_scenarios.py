"""Scenarios for the memory updates eval: single decisions and multi-fact sequences.

Decisions run the real decider on a fixed memory list (no mem0). ``target`` is the index of
the one memory an UPDATE or DELETE must change; every other memory is clearly unrelated.
Sequences feed facts in order and check the memories left at the end (keywords are lowercase).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Decision:
    name: str
    memories: list[str]
    recent: list[str]
    fact: str
    kind: str  # ADD, UPDATE, DELETE or NONE
    target: int | None  # index into memories for UPDATE and DELETE, else None


@dataclass(frozen=True)
class Expect:
    must: list[str]
    must_not: list[str]


@dataclass(frozen=True)
class Sequence:
    name: str
    facts: list[str]
    expected: list[Expect] = field(default_factory=list)


DECISIONS = [
    # ADD: new facts, a similar but different fact, and a fact about someone else.
    Decision("add-unrelated", ["Likes green tea", "Has a dog named Rex"], [],
             "Works as a civil engineer", "ADD", None),
    Decision("add-similar-different", ["Likes cats"], [], "Likes dogs", "ADD", None),
    Decision("add-someone-else", ["Lives in Rome", "Has a dog named Rex"], [],
             "His sister lives in Berlin", "ADD", None),
    Decision("add-second-pet", ["Has a dog named Rex"], [], "Has a cat named Miso", "ADD", None),
    Decision("add-someone-else-job", ["Works as a nurse"], [], "My partner works as a pilot", "ADD", None),
    Decision("add-new-hobby", ["Likes green tea", "Plays the cello"], [],
             "Goes climbing every Saturday", "ADD", None),
    # UPDATE: the one memory the fact replaces.
    Decision("update-moved", ["Likes green tea", "Lives in Rome", "Has a dog named Rex"],
             ["I just moved to Milan, the flat is near the Navigli canals."],
             "Lives in Milan now", "UPDATE", 1),
    Decision("update-job", ["Works as a nurse at St. Mary's hospital", "Likes green tea"], [],
             "Teaches maths at a primary school now", "UPDATE", 0),
    Decision("update-budget", ["Usual triangle budget is 60000 triangles", "Wants all dimensions in feet"], [],
             "My usual triangle budget is now 80000 triangles", "UPDATE", 0),
    Decision("update-colour", ["Favourite colour is green", "Has a dog named Rex"], [],
             "Favourite colour is blue now", "UPDATE", 0),
    Decision("update-phone", ["Phone number is 555-0134", "Likes green tea"], [],
             "My new phone number is 555-0199", "UPDATE", 0),
    Decision("update-age", ["Is 34 years old", "Has a dog named Rex"], [],
             "Turned 35 last week", "UPDATE", 0),
    # DELETE: the user retracts the one memory it names.
    Decision("delete-retraction", ["Works at Acme Corp", "Likes green tea"],
             ["Sorry, I mixed that up with a friend."],
             "I never worked at Acme", "DELETE", 0),
    Decision("delete-gave-away-pet", ["Has a cat named Miso", "Has a dog named Rex"], [],
             "I gave Miso away, so I no longer have a cat", "DELETE", 0),
    Decision("delete-wrong-address", ["Lives in Rome", "Likes green tea"],
             ["The Rome address I gave earlier was wrong, I have never lived there."],
             "Please forget that I live in Rome", "DELETE", 0),
    Decision("delete-joke", ["Is vegetarian", "Has a dog named Rex"], [],
             "Remove that I'm vegetarian, that was a joke", "DELETE", 0),
    # NONE: the fact repeats a memory in other words.
    Decision("none-same-meaning", ["Is vegetarian", "Has a dog named Rex"], [],
             "Doesn't eat meat or fish", "NONE", None),
    Decision("none-repeat-city", ["Lives in Milan"], [], "Based in Milan", "NONE", None),
    Decision("none-repeat-tool", ["Uses uv to manage Python dependencies"], [],
             "Manages Python packages with uv", "NONE", None),
    Decision("none-repeat-plain", ["Likes red cars", "Wants all dimensions in feet"], [],
             "I like red cars", "NONE", None),
    Decision("none-repeat-name", ["Name is Maria"], [], "My name is Maria, as I said", "NONE", None),
    Decision("none-repeat-number", ["Usual triangle budget is 60000 triangles"], [],
             "Usual triangle budget is 60k triangles", "NONE", None),
]

SEQUENCES = [
    Sequence("move", ["I live in Rome.", "I moved to Milan last month."],
             [Expect(["milan|milano"], ["rome"])]),
    Sequence("coffee-changed", ["I drink my coffee black.", "Now I take oat milk in my coffee."],
             [Expect(["oat"], ["black"])]),
    Sequence("job-changed", ["I work as a nurse at St. Mary's hospital.",
                             "I left nursing and now teach maths at a primary school."],
             [Expect(["maths|math|mathematics"], ["nurse|nursing"])]),
    Sequence("colour-changed", ["My favourite colour is green.", "My favourite colour is blue now."],
             [Expect(["blue"], ["green"])]),
    Sequence("retraction", ["I have a cat named Miso.", "I have a dog named Rex.",
                            "I gave Miso away, so I no longer have a cat."],
             [Expect(["rex|dog"], ["miso|cat"])]),
    Sequence("repeats", ["I'm vegetarian.", "I don't eat meat.", "Yes, I'm vegetarian, I never eat meat."],
             [Expect(["vegetarian|meat"], [])]),
    Sequence("repeats-and-new", ["I play the cello.", "Cello is my instrument, I've played it for years.",
                                 "My sister plays the violin."],
             [Expect(["cello"], []), Expect(["sister|sibling"], []), Expect(["violin"], [])]),
    Sequence("someone-else", ["I live in Rome.", "My brother lives in Berlin."],
             [Expect(["rome"], []), Expect(["brother|sibling", "berlin"], [])]),
]

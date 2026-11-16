"""Typed moderator decisions: JSON first, then the text format, then rotation."""
import json
import unittest

from deep_agents_demo.features.orchestration.room_decisions import (
    Decision,
    Minutes,
    Turn,
    parse_decision,
    parse_position,
)

PEOPLE = ["critic", "helper"]


def parse(text):
    return parse_decision(text, PEOPLE, "helper")


class DecisionTests(unittest.TestCase):
    def test_json_decision(self):
        decision = parse(json.dumps({
            "action": "next",
            "turns": [{"speaker": "critic", "ask": "What breaks?"}],
            "minutes": {"agreed": ["ship"], "disputed": ["cache"], "open": ["ttl"]},
        }))
        self.assertEqual(decision, Decision(
            "next", (Turn("critic", "What breaks?"),),
            minutes=Minutes(agreed=("ship",), disputed=("cache",), open=("ttl",)), source="json"))

        end = parse(json.dumps({
            "action": "end", "summary": "Agreed: ship.", "decision": "Ship it.",
            "dissent": ["helper: the cache"], "actions": ["add a TTL"],
        }))
        self.assertEqual(end, Decision("end", summary="Agreed: ship.", decision="Ship it.",
                                       dissent=("helper: the cache",), actions=("add a TTL",),
                                       source="json"))

    def test_json_in_fence_or_prose(self):
        body = json.dumps({"action": "next", "turns": [{"speaker": "critic", "ask": "Why?"}]})
        fenced = parse(f"Here is my decision:\n```json\n{body}\n```")
        prose = parse(f"Sure. {body} Thanks.")
        for decision in (fenced, prose):
            self.assertEqual(decision.source, "json")
            self.assertEqual(decision.turns, (Turn("critic", "Why?"),))

    def test_speakers_are_matched_and_deduplicated(self):
        decision = parse(json.dumps({"action": "next", "turns": [
            {"speaker": "**Critic**", "ask": "first"},
            {"speaker": "nobody", "ask": "dropped"},
            {"speaker": "critic", "ask": "second"},
            {"speaker": " HELPER ", "ask": "fine"},
        ]}))
        self.assertEqual(decision.turns, (Turn("critic", "first"), Turn("helper", "fine")))

    def test_missing_ask_gets_a_default(self):
        decision = parse(json.dumps({"action": "next", "turns": [{"speaker": "helper"}]}))
        self.assertEqual(decision.turns, (Turn("helper", "What is your view?"),))

    def test_bad_json_falls_back_to_text_then_rotation(self):
        text = parse("{not json\nNEXT: helper\nWhy?")
        self.assertEqual((text.source, text.turns), ("text", (Turn("helper", "Why?"),)))
        rotated = parse("blah")
        self.assertEqual((rotated.action, rotated.source, rotated.turns),
                         ("next", "rotation", (Turn("helper", "What is your view?"),)))

    def test_text_end_and_unknown_json_action(self):
        self.assertEqual(parse("END\nAgreed: ship.").source, "text")
        self.assertEqual(parse("END\nAgreed: ship.").summary, "Agreed: ship.")
        odd = parse(json.dumps({"action": "pause"}))
        self.assertEqual(odd.source, "rotation")

    def test_next_without_valid_turns_falls_through(self):
        for payload in ({"action": "next"}, {"action": "next", "turns": []},
                        {"action": "next", "turns": [{"speaker": "nobody", "ask": "x"}]}):
            with self.subTest(payload=payload):
                decision = parse(json.dumps(payload))
                self.assertEqual(decision.source, "rotation")

    def test_a_literal_newline_in_a_string_still_parses(self):
        # written by hand: json.dumps would escape the newline and the test would not check the parser
        text = '{"action": "end", "summary": "Line one.\nLine two.", "decision": "Ship."}'
        decision = parse(text)
        self.assertEqual((decision.source, decision.summary, decision.decision),
                         ("json", "Line one.\nLine two.", "Ship."))

    def test_the_first_object_that_is_a_decision_wins(self):
        decision = parse('Use {x} then {"action": "next", "turns": [{"speaker": "critic", "ask": "Why?"}]}')
        self.assertEqual((decision.source, decision.turns), ("json", (Turn("critic", "Why?"),)))

    def test_a_null_speaker_is_missing_not_none(self):
        decision = parse_decision(json.dumps({"action": "next", "turns": [
            {"speaker": None, "ask": "dropped"},
            {"speaker": "helper", "ask": "fine"},
        ]}), ["none", "helper"], "helper")
        self.assertEqual(decision.turns, (Turn("helper", "fine"),))

    def test_minutes_missing_or_bad_is_none(self):
        self.assertIsNone(parse(json.dumps({"action": "next", "turns": [{"speaker": "critic"}]})).minutes)
        self.assertIsNone(parse(json.dumps({"action": "next", "turns": [{"speaker": "critic"}],
                                            "minutes": "none"})).minutes)

    def test_minutes_text(self):
        self.assertEqual(Minutes().text(), "(no minutes yet)")
        text = Minutes(agreed=("ship",), open=("ttl", "cache")).text()
        self.assertEqual(text, "Agreed:\n- ship\n\nOpen:\n- ttl\n- cache")
        self.assertNotIn("Disputed:", text)


class PositionTests(unittest.TestCase):
    def test_parse_position(self):
        self.assertEqual(parse_position("Ok.\nPOSITION: Ship it. (confidence: high)"), ("Ship it.", "high"))
        self.assertEqual(parse_position("**POSITION:** ship it (confidence: HIGH)"), ("ship it", "high"))
        self.assertEqual(parse_position("- _position_: wait (confidence: maybe)"), ("wait", "unknown"))
        self.assertEqual(parse_position("POSITION: no confidence here"), ("no confidence here", "unknown"))
        self.assertEqual(parse_position("POSITION: first\nmore\nPOSITION: second (confidence: low)"),
                         ("second", "low"))
        self.assertEqual(parse_position("POSITION: x (confidence: high)."), ("x", "high"))
        self.assertEqual(parse_position("**POSITION:** ship it (confidence: Low)."), ("ship it", "low"))
        self.assertIsNone(parse_position("Just prose, no position line."))
        self.assertIsNone(parse_position(""))


if __name__ == "__main__":
    unittest.main()

"""Room formats: the presets, and the check of a format against its participants."""
import unittest

from deep_agents_demo.config import ROOM_FORMATS


class PresetTests(unittest.TestCase):
    def test_presets_match_the_settings_names(self):
        from deep_agents_demo.features.orchestration.room_formats import FORMATS
        self.assertEqual(tuple(FORMATS), ROOM_FORMATS)
        for name, fmt in FORMATS.items():
            with self.subTest(format=name):
                self.assertEqual(fmt.name, name)
                self.assertTrue(fmt.instructions.strip())

    def test_each_preset_has_its_shape(self):
        from deep_agents_demo.features.orchestration.room_formats import FORMATS
        shape = {name: (fmt.opening_round, fmt.policy, fmt.min_participants, fmt.max_participants)
                 for name, fmt in FORMATS.items()}
        self.assertEqual(shape, {
            "moderated": (True, "moderator", 2, None),
            "roundrobin": (False, "roundrobin", 2, None),
            "debate": (False, "debate", 3, 3),
            "redteam": (False, "redteam", 2, None),
        })

    def test_a_format_is_frozen(self):
        from dataclasses import FrozenInstanceError

        from deep_agents_demo.features.orchestration.room_formats import FORMATS
        with self.assertRaises(FrozenInstanceError):
            FORMATS["debate"].policy = "moderator"


class FormatProblemTests(unittest.TestCase):
    def problem(self, name, participants):
        from deep_agents_demo.features.orchestration.room_formats import format_problem
        return format_problem(name, participants)

    def test_a_good_format_and_count_has_no_problem(self):
        self.assertIsNone(self.problem("moderated", ["a", "b"]))
        self.assertIsNone(self.problem("roundrobin", ["a", "b", "c", "d"]))
        self.assertIsNone(self.problem("debate", ["a", "b", "c"]))
        self.assertIsNone(self.problem("redteam", ["a", "b"]))

    def test_an_unknown_format_is_named(self):
        self.assertEqual(self.problem("x", ["a", "b"]),
                         "unknown room format 'x'; use one of moderated, roundrobin, debate, redteam")

    def test_debate_needs_exactly_three(self):
        self.assertEqual(self.problem("debate", ["a", "b"]), "the debate format needs exactly 3 participants")
        self.assertEqual(self.problem("debate", ["a", "b", "c", "d"]),
                         "the debate format needs exactly 3 participants")

    def test_the_others_need_at_least_two(self):
        for name in ("moderated", "roundrobin", "redteam"):
            with self.subTest(format=name):
                self.assertIn("needs at least 2", self.problem(name, ["a"]))

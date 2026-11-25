"""Memory updates eval: scenario sanity, scorers and the table format. No model calls."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from deep_agents_demo.features.memory.updates import Operation  # noqa: E402
import memory_updates as runner  # noqa: E402
from memory_update_scenarios import DECISIONS, SEQUENCES, Decision, Expect, Sequence  # noqa: E402


class ScoreDecisionTests(unittest.TestCase):
    def test_right_kind_and_target(self):
        decision = Decision("d", ["a", "b", "c"], [], "fact", "UPDATE", 1)
        op = Operation(kind="UPDATE", memory_id="2", text="new")
        self.assertEqual(runner.score_decision(decision, op, ["1", "2", "3"]),
                         {"kind_ok": True, "target_ok": True})

    def test_right_kind_wrong_target(self):
        decision = Decision("d", ["a", "b", "c"], [], "fact", "DELETE", 1)
        op = Operation(kind="DELETE", memory_id="3")
        self.assertEqual(runner.score_decision(decision, op, ["1", "2", "3"]),
                         {"kind_ok": True, "target_ok": False})

    def test_none_has_no_target(self):
        decision = Decision("d", ["a"], [], "fact", "NONE", None)
        op = Operation(kind="NONE")
        self.assertEqual(runner.score_decision(decision, op, ["1"]),
                         {"kind_ok": True, "target_ok": None})

    def test_wrong_kind(self):
        decision = Decision("d", ["a"], [], "fact", "ADD", None)
        op = Operation(kind="NONE")
        self.assertEqual(runner.score_decision(decision, op, ["1"]),
                         {"kind_ok": False, "target_ok": None})


class ScoreFinalTests(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence("move", ["x"], [Expect(["milan"], ["rome"]),
                                                 Expect(["teach|teaches"], ["nurse"])])

    def test_one_memory_cannot_meet_two_expects(self):
        sequence = Sequence("merged", ["x"], [Expect(["tea"], []), Expect(["rex|dog"], [])])
        self.assertEqual(runner.score_final(sequence, ["Has a dog named Rex and likes green tea"])["missing"], 1)
        self.assertEqual(runner.score_final(sequence, ["Likes green tea", "Has a dog named Rex"])["missing"], 0)

    def test_a_stale_memory_is_counted(self):
        result = runner.score_final(self.sequence, ["Lives in Rome", "Teaches maths"])
        self.assertEqual(result["stale"], 1)
        self.assertEqual(result["missing"], 1)

    def test_a_missing_expect_is_counted(self):
        result = runner.score_final(self.sequence, ["Lives in Milan", "Teaches maths"])
        self.assertEqual(result["missing"], 0)
        self.assertEqual(result["stale"], 0)
        self.assertEqual(result["extra"], 0)

    def test_extra_memories_are_counted(self):
        result = runner.score_final(self.sequence, ["Lives in Milan", "Teaches maths", "Likes tea", "Owns a car"])
        self.assertEqual(result, {"count": 4, "expected": 2, "extra": 2, "stale": 0, "missing": 0})

    def test_fewer_memories_than_expected_is_not_extra(self):
        result = runner.score_final(self.sequence, ["Lives in Milan"])
        self.assertEqual(result["extra"], 0)
        self.assertEqual(result["missing"], 1)

    def test_keywords_must_all_be_in_one_memory(self):
        sequence = Sequence("two", ["x"], [Expect(["oat", "coffee"], [])])
        self.assertEqual(runner.score_final(sequence, ["Takes oat milk", "Likes coffee"])["missing"], 1)
        self.assertEqual(runner.score_final(sequence, ["Takes oat milk in coffee"])["missing"], 0)

    def test_matching_is_maximal_not_greedy(self):
        sequence = Sequence("two", ["x"], [Expect(["tea"], []), Expect(["tea", "rex"], [])])
        self.assertEqual(runner.score_final(sequence, ["tea rex", "tea"])["missing"], 0)

    def test_an_alternative_in_must_meets_the_expect(self):
        sequence = Sequence("veg", ["x"], [Expect(["vegetarian|meat"], [])])
        self.assertEqual(runner.score_final(sequence, ["Doesn't eat meat"])["missing"], 0)
        self.assertEqual(runner.score_final(sequence, ["Is vegetarian"])["missing"], 0)

    def test_maths_and_math_both_count(self):
        sequence = Sequence("maths", ["x"], [Expect(["maths|math|mathematics"], [])])
        self.assertEqual(runner.score_final(sequence, ["Teaches math"])["missing"], 0)
        self.assertEqual(runner.score_final(sequence, ["Teaches mathematics"])["missing"], 0)

    def test_sister_or_sibling_and_violin_in_one_memory(self):
        sequence = Sequence("siblings", ["x"], [Expect(["sister|sibling", "violin"], [])])
        self.assertEqual(runner.score_final(sequence, ["Has a sibling who plays violin"])["missing"], 0)
        self.assertEqual(runner.score_final(sequence, ["Has a sister", "Plays violin"])["missing"], 1)

    def test_a_move_written_as_from_to_is_not_stale(self):
        self.assertEqual(runner.score_final(self.sequence, ["Moved from Rome to Milan"])["stale"], 0)

    def test_a_change_with_the_new_value_and_the_old_word_is_not_stale(self):
        sequence = Sequence("coffee", ["x"], [Expect(["oat"], ["black"])])
        self.assertEqual(runner.score_final(sequence, ["Takes oat milk, not black"])["stale"], 0)
        self.assertEqual(runner.score_final(sequence, ["Drinks coffee black"])["stale"], 1)

    def test_stale_keywords_match_whole_words_only(self):
        self.assertEqual(runner.score_final(self.sequence, ["Likes Romeo and Juliet", "Lives in Milan"])["stale"], 0)

    def test_a_memory_that_states_the_past_is_history_not_stale(self):
        sequence = Sequence("past", ["x"], [Expect(["rex|dog"], ["miso|cat", "rome"])])
        self.assertEqual(runner.score_final(sequence, ["Used to live in Rome", "Has a dog named Rex"])["stale"], 0)
        self.assertEqual(runner.score_final(sequence, ["Gave away Miso", "Has a dog named Rex"])["stale"], 0)
        self.assertEqual(runner.score_final(sequence, ["Used to have a cat", "Has a dog named Rex"])["stale"], 0)
        self.assertEqual(runner.score_final(sequence, ["Lives in Rome", "Has a dog named Rex"])["stale"], 1)

    def test_paraphrase_move_clean_final_scores_clean(self):
        sequence = next(s for s in SEQUENCES if s.name == "paraphrase-move")
        result = runner.score_final(sequence, ["Likes green tea", "Has a dog named Rex", "Plays the piano",
                                               "Allergic to peanuts", "Favourite colour is blue",
                                               "Works as a nurse", "Lives in Milan"])
        self.assertEqual((result["stale"], result["missing"], result["extra"]), (0, 0, 0))

    def test_gave_a_talk_is_not_history(self):
        sequence = Sequence("talk", ["x"], [Expect(["rex|dog"], ["pycon", "cat"])])
        self.assertEqual(runner.score_final(sequence, ["Gave a talk at PyCon", "Has a dog named Rex"])["stale"], 1)
        self.assertEqual(runner.score_final(sequence, ["Gave away the cat", "Has a dog named Rex"])["stale"], 0)


class DecisionPartTests(unittest.TestCase):
    class Stub:
        """A decider stand-in that records each call and always answers NONE."""
        calls = []

        def __init__(self, model):
            pass

        def decide(self, fact, recent, candidates):
            DecisionPartTests.Stub.calls.append((fact, candidates))
            return Operation(kind="NONE")

    def test_decision_candidates_are_the_seeded_memories_in_order(self):
        self.Stub.calls = []
        with mock.patch.object(runner, "MemoryDecider", self.Stub):
            runner.run_decisions(SimpleNamespace(callbacks=None), DECISIONS)
        self.assertEqual(len(self.Stub.calls), len(DECISIONS))
        for decision, (fact, candidates) in zip(DECISIONS, self.Stub.calls):
            expected = [{"id": str(i), "memory": m} for i, m in enumerate(decision.memories)]
            self.assertEqual(fact, decision.fact, decision.name)
            self.assertEqual(candidates, expected, decision.name)


class NearestRankTests(unittest.TestCase):
    def test_one_value(self):
        self.assertEqual(runner.nearest_rank([3.0], 50), 3.0)
        self.assertEqual(runner.nearest_rank([3.0], 95), 3.0)

    def test_two_values(self):
        self.assertEqual(runner.nearest_rank([2.0, 1.0], 50), 1.0)
        self.assertEqual(runner.nearest_rank([2.0, 1.0], 95), 2.0)

    def test_twenty_values(self):
        values = [float(v) for v in range(20, 0, -1)]
        self.assertEqual(runner.nearest_rank(values, 50), 10.0)
        self.assertEqual(runner.nearest_rank(values, 95), 19.0)

    def test_the_output_says_how_percentiles_are_taken(self):
        self.assertTrue(any("nearest-rank" in note for note in runner.NOTES))


class ScenarioTests(unittest.TestCase):
    def test_enough_decisions_with_each_kind(self):
        self.assertGreaterEqual(len(DECISIONS), 20)
        for kind in ("ADD", "UPDATE", "DELETE", "NONE"):
            self.assertGreaterEqual(sum(d.kind == kind for d in DECISIONS), 3, kind)

    def test_targets_are_valid_and_only_for_update_and_delete(self):
        for decision in DECISIONS:
            if decision.kind in ("UPDATE", "DELETE"):
                self.assertIsNotNone(decision.target, decision.name)
                self.assertTrue(0 <= decision.target < len(decision.memories), decision.name)
            else:
                self.assertIsNone(decision.target, decision.name)

    def test_every_decision_scenario_has_at_most_five_memories(self):
        for decision in DECISIONS:
            self.assertLessEqual(len(decision.memories), 5, decision.name)

    def test_paraphrase_move_sequence_exists(self):
        by_name = {s.name: s for s in SEQUENCES}
        self.assertIn("paraphrase-move", by_name)
        sequence = by_name["paraphrase-move"]
        self.assertEqual(sequence.facts[:6], ["I like green tea.", "I have a dog named Rex.", "I play the piano.",
                                              "I am allergic to peanuts.", "My favourite colour is blue.",
                                              "I work as a nurse."])
        self.assertEqual(sequence.facts[6:], ["I live in Rome.", "I relocated to Milan."])
        self.assertEqual(len(sequence.expected), 7)
        self.assertEqual(sequence.expected[-1], Expect(["milan|milano"], ["rome"]))
        after_rome = sequence.facts[sequence.facts.index("I live in Rome.") + 1:]
        self.assertFalse(any("rome" in fact.lower() for fact in after_rome))

    def test_the_traps_are_there(self):
        names = {d.name for d in DECISIONS}
        self.assertTrue(any(d.kind == "ADD" and "similar" in d.name for d in DECISIONS), names)
        self.assertTrue(any(d.kind == "ADD" and "someone" in d.name for d in DECISIONS), names)
        self.assertTrue(any(d.kind == "NONE" and "repeat" in d.name for d in DECISIONS), names)

    def test_enough_sequences_with_keywords_in_lowercase(self):
        self.assertGreaterEqual(len(SEQUENCES), 6)
        for sequence in SEQUENCES:
            self.assertTrue(sequence.facts, sequence.name)
            self.assertTrue(sequence.expected, sequence.name)
            for expect in sequence.expected:
                for keyword in expect.must + expect.must_not:
                    self.assertEqual(keyword, keyword.lower(), sequence.name)

    def test_the_ambiguous_decisions_are_fixed(self):
        by_name = {d.name: d for d in DECISIONS}
        self.assertEqual(by_name["none-repeat-name"].fact, "My name is Maria, as I said")
        self.assertEqual(by_name["none-same-meaning"].fact, "Doesn't eat meat or fish")
        self.assertEqual(by_name["none-repeat-plain"].fact, "I like red cars")


class FakeMemory:
    """A mem0 stand-in: add stores the fact, unless the fact is in ``fail``."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.texts = []

    def add(self, messages, user_id, infer):
        fact = messages[0]["content"]
        if fact in self.fail:
            raise RuntimeError(f"mem0 down for {fact}")
        self.texts.append(fact)

    def get_all(self, filters, top_k):
        return {"results": [{"memory": text} for text in self.texts]}


class FakeService:
    """MemoryService stand-in: remember stores the fact, unless the fact is in the memory's fail set."""

    raises_on_init = False

    def __init__(self, memory, user_id, decider=None):
        if FakeService.raises_on_init:
            raise RuntimeError("no service")
        self.memory = memory

    def remember(self, fact, recent=()):
        if fact in self.memory.fail:
            raise RuntimeError(f"service down for {fact}")
        self.memory.texts.append(fact)
        return {"stored": True, "action": "added"}

    def forget(self, memory_id):
        return {"forgotten": True}


class CompareLoopTests(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence("loop", ["a", "boom", "c"], [Expect(["a"], []), Expect(["c"], [])])
        self.settings = SimpleNamespace(memory=None, embeddings=None)
        self.model = SimpleNamespace(callbacks=None)
        FakeService.raises_on_init = False
        self.patches = [
            mock.patch("deep_agents_demo.startup.resources.embeddings", lambda *a, **k: None),
            mock.patch("deep_agents_demo.features.memory.service.MemoryService", FakeService),
            mock.patch.object(runner, "close_memory", lambda memory: None),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in self.patches])

    def patch_make_memory(self, make):
        patch = mock.patch.object(runner, "make_memory", make)
        patch.start()
        self.addCleanup(patch.stop)

    def entry(self, result, path):
        return next(e for e in result["entries"] if e["path"] == path)

    def test_a_failed_fact_is_counted_and_the_run_goes_on(self):
        self.patch_make_memory(lambda *a, **k: FakeMemory(fail={"boom"}))
        result = runner.run_compare(self.settings, self.model, [self.sequence])
        old = next(r for r in result["rows"] if r["path"] == "old")
        new = next(r for r in result["rows"] if r["path"] == "new")
        self.assertEqual((old["errors"], new["errors"]), (1, 1))
        self.assertIn("mem0 down for boom", old["first_error"])
        self.assertIn("service down for boom", new["first_error"])
        self.assertEqual(old["count"], 2)  # "a" and "c" were still stored
        self.assertEqual(self.entry(result, "old: mem0 add")["metrics"]["Errors"], 1)
        self.assertEqual(self.entry(result, "new: remember")["metrics"]["Errors"], 1)

    def test_a_store_that_cannot_be_built_is_an_error_for_that_sequence(self):
        def broken(*args, **kwargs):
            raise RuntimeError("no store")

        self.patch_make_memory(broken)
        result = runner.run_compare(self.settings, self.model, [self.sequence])
        for row in result["rows"]:
            self.assertEqual(row["errors"], 3)
            self.assertIn("no store", row["first_error"])
        self.assertEqual(self.entry(result, "old: mem0 add")["metrics"]["Errors"], 3)

    def test_a_store_that_cannot_be_read_back_is_an_error_and_the_run_goes_on(self):
        class Unreadable(FakeMemory):
            def get_all(self, filters, top_k):
                raise RuntimeError("index unreadable")

        self.patch_make_memory(lambda *a, **k: Unreadable())
        result = runner.run_compare(self.settings, self.model, [self.sequence, self.sequence])
        self.assertEqual(len(result["rows"]), 4, "every sequence and path still gets a row")
        for row in result["rows"]:
            self.assertEqual(row["errors"], 3)
            self.assertIn("index unreadable", row["first_error"])

    def test_a_service_that_cannot_be_built_is_an_error_for_that_sequence(self):
        self.patch_make_memory(lambda *a, **k: FakeMemory())
        FakeService.raises_on_init = True
        result = runner.run_compare(self.settings, self.model, [self.sequence])
        new = next(r for r in result["rows"] if r["path"] == "new")
        self.assertEqual(new["errors"], 3)
        self.assertIn("no service", new["first_error"])

    def test_mem0_dir_is_set_per_store_and_restored(self):
        seen = []

        def make(data, *args, **kwargs):
            seen.append(os.environ.get("MEM0_DIR"))
            return FakeMemory()

        self.patch_make_memory(make)
        os.environ["MEM0_DIR"] = "sentinel"
        self.addCleanup(os.environ.pop, "MEM0_DIR", None)
        runner.run_compare(self.settings, self.model, [self.sequence])
        self.assertEqual(os.environ["MEM0_DIR"], "sentinel")
        self.assertTrue(all(value and value != "sentinel" for value in seen), seen)

    def test_mem0_dir_that_was_unset_is_unset_again(self):
        self.patch_make_memory(lambda *a, **k: FakeMemory())
        os.environ.pop("MEM0_DIR", None)
        runner.run_compare(self.settings, self.model, [self.sequence])
        self.assertNotIn("MEM0_DIR", os.environ)


class FormatTests(unittest.TestCase):
    def test_three_runs_give_mean_and_range_cells(self):
        runs = [
            [{"part": "decision", "path": "decider", "metrics": {"Kind right": 0.5}}],
            [{"part": "decision", "path": "decider", "metrics": {"Kind right": 0.75}}],
            [{"part": "decision", "path": "decider", "metrics": {"Kind right": 1.0}}],
        ]
        table = runner.format_table(runs)
        self.assertIn("75% (50%–100%)", table)

    def test_one_run_gives_a_plain_cell(self):
        runs = [[{"part": "compare", "path": "new", "metrics": {"Stale": 2.0}}]]
        table = runner.format_table(runs)
        self.assertIn(" 2 |", table)
        self.assertNotIn("(", table.splitlines()[-1])

    def test_whole_numbers_have_no_decimal(self):
        runs = [[{"part": "compare", "path": "new", "metrics": {"Errors": 2.0, "Final memories": 2.5}}]]
        table = runner.format_table(runs)
        self.assertIn(" 2 |", table)
        self.assertIn(" 2.5 |", table)


if __name__ == "__main__":
    unittest.main()

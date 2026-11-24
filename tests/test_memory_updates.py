"""Memory updates: the operation model, BM25 candidate ranking and the decider."""
import itertools
import subprocess
import sys
import unittest

import pydantic
from langchain_core.runnables import RunnableLambda

from deep_agents_demo.features.memory.updates import (
    DECISION_PROMPT,
    UPDATE_CANDIDATES,
    MemoryDecider,
    Operation,
    rank_candidates,
)


class FakeDecisionModel:
    """Just enough of a chat model: with_structured_output returns a runnable that records its input."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def with_structured_output(self, schema):
        def respond(messages):
            self.calls.append((schema, messages))
            return self.answer

        return RunnableLambda(respond)


class RankCandidatesTests(unittest.TestCase):
    def test_the_memory_about_the_same_thing_ranks_first(self):
        memories = [
            {"id": "1", "memory": "Lives in Rome"},
            {"id": "2", "memory": "Likes green tea"},
            {"id": "3", "memory": "Has a dog named Rex"},
        ]
        ranked = rank_candidates("Now lives in Milan, not Rome", memories)
        self.assertEqual(ranked[0]["id"], "1")

    def test_case_and_punctuation_do_not_hide_a_match(self):
        memories = [
            {"id": "1", "memory": "Lives in Rome."},
            {"id": "2", "memory": "Likes green tea"},
            {"id": "3", "memory": "Has a dog named Rex"},
        ]
        for order in itertools.permutations(memories):  # a tie must never decide it
            with self.subTest(order=[m["id"] for m in order]):
                ranked = rank_candidates("Moved away from rome", list(order))
                self.assertEqual(ranked[0]["id"], "1")

    def test_composed_and_decomposed_accents_match(self):
        memories = [{"id": "1", "memory": "Favourite place: Café Nero"},
                    {"id": "2", "memory": "Likes green tea"},
                    {"id": "3", "memory": "Has a dog named Rex"}]
        for order in itertools.permutations(memories):
            with self.subTest(order=[m["id"] for m in order]):
                ranked = rank_candidates("no longer goes to café", list(order))
                self.assertEqual(ranked[0]["id"], "1")

    def test_the_rank_module_import_does_not_load_langchain_community(self):
        code = (
            "import sys\n"
            "import deep_agents_demo.features.memory.updates\n"
            "print('langchain_community' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_ranking_prints_no_langchain_community_sunset_warning(self):
        code = (
            "import warnings\n"
            "warnings.simplefilter('always')\n"
            "from deep_agents_demo.features.memory.updates import rank_candidates\n"
            "rank_candidates('x', [{'id': '1', 'memory': 'y'}])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertNotIn("sunset", result.stderr)

    def test_at_most_k_candidates_and_none_without_memories(self):
        memories = [{"id": str(i), "memory": f"Fact number {i} about hobbies"} for i in range(8)]
        self.assertEqual(len(rank_candidates("hobbies", memories, k=5)), 5)
        self.assertEqual(UPDATE_CANDIDATES, 5)
        self.assertEqual(rank_candidates("x", []), [])

    def test_one_memory_or_a_punctuation_fact_still_ranks(self):
        one = [{"id": "1", "memory": "Lives in Rome"}]
        self.assertEqual(rank_candidates("Lives in Milan", one), one)
        three = [{"id": str(i), "memory": f"Fact {i}"} for i in range(3)]
        self.assertLessEqual(len(rank_candidates("?!", three)), 3)


class OperationTests(unittest.TestCase):
    def test_operation_rejects_an_unknown_kind(self):
        with self.assertRaises(pydantic.ValidationError):
            Operation(kind="MERGE")

    def test_operation_defaults(self):
        operation = Operation(kind="NONE")
        self.assertIsNone(operation.memory_id)
        self.assertIsNone(operation.text)
        self.assertEqual(operation.reason, "")


class MemoryDeciderTests(unittest.TestCase):
    def test_the_decider_returns_the_models_operation(self):
        answer = Operation(kind="UPDATE", memory_id="1", text="Lives in Milan", reason="moved")
        model = FakeDecisionModel(answer)
        decider = MemoryDecider(model)
        recent = ["I moved last week", "The flat is near the canal"]
        result = decider.decide(
            "Now lives in Milan",
            recent,
            [{"id": "1", "memory": "Lives in Rome"}],
        )
        self.assertEqual(result, answer)
        self.assertEqual(len(model.calls), 1)
        schema, messages = model.calls[0]
        self.assertIs(schema, Operation)
        self.assertEqual(messages[0].content, DECISION_PROMPT)
        prompt = messages[1].content
        self.assertIn("Now lives in Milan", prompt)
        self.assertIn("[1] Lives in Rome", prompt)
        for message in recent:
            self.assertIn(message, prompt)


if __name__ == "__main__":
    unittest.main()

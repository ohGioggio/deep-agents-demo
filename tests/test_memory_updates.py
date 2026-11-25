"""Memory updates: the operation model and the decider."""
import unittest

import pydantic
from langchain_core.runnables import RunnableLambda

from deep_agents_demo.features.memory.updates import (
    DECISION_PROMPT,
    MemoryDecider,
    Operation,
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

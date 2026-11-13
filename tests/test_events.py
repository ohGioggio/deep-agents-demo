"""The run, text, tool call, subagent, step and approval events, and the run footer."""
import json
import time
import unittest

from deep_agents_demo.models.events import (
    ApprovalRequested,
    RunError,
    RunFinished,
    RunStarted,
    StateSnapshot,
    StepFinished,
    StepStarted,
    SubagentFinished,
    SubagentStarted,
    TextMessageContent,
    TextMessageEnd,
    TextMessageStart,
    ToolCallEnd,
    ToolCallStart,
    format_footer,
)


class NewEventsTests(unittest.TestCase):
    def test_new_events_are_plain_data(self):
        events = [
            RunStarted(run_id="r1", label="main"),
            RunStarted(run_id="r2", label="subagent", parent_id="r1"),
            RunFinished(run_id="r1", label="main", input_tokens=100, output_tokens=20,
                        duration=1.5, ended_at=1763040060.0),
            RunError(run_id="r1", label="main", message="boom"),
            TextMessageStart(run_id="r1", message_id="m1", label="main"),
            TextMessageContent(run_id="r1", message_id="m1", delta="Hi"),
            TextMessageEnd(run_id="r1", message_id="m1", text="Hi", input_tokens=10,
                           output_tokens=2, has_tool_calls=True),
            ToolCallStart(run_id="r1", tool_call_id="t1", name="ls", args={"path": "/"}),
            ToolCallEnd(run_id="r1", tool_call_id="t1", name="ls", status="ok",
                        summary="3 files", duration=0.2, result="a\nb\nc"),
            SubagentStarted(run_id="r2", parent_run_id="r1", name="researcher",
                            tool_call_id="t2"),
            SubagentFinished(run_id="r2", status="ok"),
            StepStarted(run_id="r3", kind="plan", number=1, total=2, agent="general-purpose"),
            StepFinished(run_id="r3", kind="room", number=1, status="ok", attempts=1,
                         duration=3.0, input_tokens=5, output_tokens=6, summary="done"),
            StateSnapshot(todos=({"content": "write", "status": "pending"},)),
            ApprovalRequested(run_id="r1", tool_call_id="t3", name="delete_file"),
        ]
        for event in events:
            with self.subTest(event=type(event).__name__):
                data = json.loads(json.dumps(event.to_dict()))
                self.assertEqual(data["kind"], type(event).__name__)
                with self.assertRaises(AttributeError):
                    event.run_id = "changed"

    def test_footer_format(self):
        ended_at = 1763040060.0
        hhmm = time.strftime("%H:%M", time.localtime(ended_at))
        self.assertEqual(format_footer(ended_at, 9800, 450, 74.2),
                         f"{hhmm} · 9.8k in / 450 out · 74 s")
        self.assertEqual(format_footer(ended_at, 9800, 450, 125.0),
                         f"{hhmm} · 9.8k in / 450 out · 2 min 5 s")
        self.assertEqual(format_footer(ended_at, 999, 1000, 99.6),
                         f"{hhmm} · 999 in / 1.0k out · 99 s")


if __name__ == "__main__":
    unittest.main()

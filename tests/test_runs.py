"""The run store, the usage counter, and the recorder that saves each chat turn as a run."""
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from deep_agents_demo.config import ChatSettings, Settings
from deep_agents_demo.core.runs import RunRecorder
from deep_agents_demo.features.orchestration.pool import DaemonPool
from deep_agents_demo.models.events import (
    RunError,
    RunStarted,
    StepFinished,
    StepStarted,
    SubagentFinished,
    SubagentStarted,
    TextMessageEnd,
    ToolCallEnd,
    ToolCallStart,
)
from deep_agents_demo.persistence.chats import Chats, thread_id
from deep_agents_demo.persistence.outputs import OutputStore
from deep_agents_demo.persistence.runs import Run, RunPart, RunStore
from deep_agents_demo.providers.chat import UsageCounter, chat_model, usage_counter
from deep_agents_demo.ui.cli.repl import ChatSession


class Collect:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class FakeCounter:
    """Stands in for the process-wide UsageCounter: the test adds the model calls."""

    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, calls, input_tokens, output_tokens):
        self.calls += calls
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def snapshot(self):
        return (self.calls, self.input_tokens, self.output_tokens)


class UsageModel(BaseChatModel):
    """A model that answers at once, reporting token usage (no network)."""

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="ok", usage_metadata={"input_tokens": 100, "output_tokens": 7,
                                                           "total_tokens": 107})
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_run(run_id, user="alice", chat="c1", started_at=100.0, parts=()):
    return Run(id=run_id, user=user, chat=chat, started_at=started_at, ended_at=started_at + 5,
               seconds=5.0, model="m", prompt=f"prompt {run_id}", model_calls=3, input_tokens=300,
               output_tokens=30, status="done", parts=list(parts))


class RunStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_save_then_get_returns_the_run_with_its_nested_parts(self):
        parts = [RunPart(1, None, "plan", "build", 2, 80, 8, 4.0, "ok"),
                 RunPart(2, 1, "step", "1/2 general-purpose", 2, 80, 8, 4.0, "done")]
        self.store.save(make_run("r1", parts=parts))
        run = self.store.get("alice", "r1")
        self.assertEqual(run.prompt, "prompt r1")
        self.assertEqual((run.model_calls, run.input_tokens, run.output_tokens, run.status),
                         (3, 300, 30, "done"))
        self.assertEqual([(p.part, p.parent, p.kind, p.label) for p in run.parts],
                         [(1, None, "plan", "build"), (2, 1, "step", "1/2 general-purpose")])
        self.assertEqual(run.parts[1].seconds, 4.0)

    def test_get_is_scoped_to_the_user(self):
        self.store.save(make_run("r1", user="alice"))
        self.assertIsNone(self.store.get("bob", "r1"))
        self.assertIsNone(self.store.get("alice", "missing"))

    def test_recent_lists_newest_first_without_parts_and_respects_the_limit(self):
        for number in range(3):
            self.store.save(make_run(f"r{number}", started_at=100.0 + number,
                                     parts=[RunPart(1, None, "subagent", "x", 1, 1, 1, 1.0, "ok")]))
        self.store.save(make_run("other", user="bob", started_at=500.0))
        runs = self.store.recent("alice", limit=2)
        self.assertEqual([run.id for run in runs], ["r2", "r1"])
        self.assertEqual(runs[0].parts, [])

    def test_totals_sum_the_user_runs_since_a_time(self):
        self.store.save(make_run("old", started_at=10.0))
        self.store.save(make_run("new", started_at=200.0))
        self.store.save(make_run("theirs", user="bob", started_at=200.0))
        totals = self.store.totals("alice", since=100.0)
        self.assertEqual(totals, {"runs": 1, "model_calls": 3, "input_tokens": 300, "output_tokens": 30})

    def test_delete_chat_removes_that_chat_s_runs_and_parts_only(self):
        part = RunPart(1, None, "subagent", "x", 1, 1, 1, 1.0, "ok")
        self.store.save(make_run("gone", chat="c1", parts=[part]))
        self.store.save(make_run("kept", chat="c2", parts=[part]))
        self.assertEqual(self.store.delete_chat("alice", "c1"), 1)
        self.assertIsNone(self.store.get("alice", "gone"))
        self.assertEqual(self.store.get("alice", "kept").parts[0].kind, "subagent")

    def test_rename_chat_moves_its_runs(self):
        self.store.save(make_run("r1", chat="c1"))
        self.assertEqual(self.store.rename_chat("alice", "c1", "c2"), 1)
        self.assertEqual(self.store.get("alice", "r1").chat, "c2")


class UsageCounterTest(unittest.TestCase):
    def test_counts_one_call_per_model_answer_with_its_tokens(self):
        counter = UsageCounter()
        message = AIMessage(content="ok", usage_metadata={"input_tokens": 12, "output_tokens": 3,
                                                           "total_tokens": 15})
        counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        self.assertEqual(counter.snapshot(), (1, 12, 3))

    def test_falls_back_to_the_provider_token_usage(self):
        counter = UsageCounter()
        counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                                     llm_output={"token_usage": {"prompt_tokens": 9, "completion_tokens": 4}}))
        self.assertEqual(counter.snapshot(), (1, 9, 4))

    def test_counts_a_fake_model_s_calls_and_tokens(self):
        counter = UsageCounter()
        model = UsageModel(callbacks=[counter])
        model.invoke("hi")
        model.invoke("again")
        self.assertEqual(counter.snapshot(), (2, 200, 14))

    def test_counts_from_several_threads(self):
        counter = UsageCounter()
        message = AIMessage(content="ok", usage_metadata={"input_tokens": 1, "output_tokens": 1,
                                                           "total_tokens": 2})
        result = LLMResult(generations=[[ChatGeneration(message=message)]])

        def many():
            for _ in range(200):
                counter.on_llm_end(result)

        threads = [threading.Thread(target=many) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(counter.snapshot(), (1000, 1000, 1000))

    def test_every_chat_model_the_app_builds_carries_the_process_counter(self):
        settings = Settings(chat=ChatSettings(api_key="unused", model="unused", base_url=None))
        self.assertIn(usage_counter(), chat_model(settings).callbacks)
        self.assertIn(usage_counter(), chat_model(settings, "other").callbacks)


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")
        self.counter = FakeCounter()
        self.sink = Collect()
        self.notes = []
        self.recorder = RunRecorder(self.sink, self.store, self.counter, user="alice", model="m",
                                    say=self.notes.append)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_a_nested_plan_turn_is_recorded_with_its_parts_and_totals(self):
        events = [
            RunStarted("main", "main"),
            TextMessageEnd("main", "m1", "", 100, 10, True),
            ToolCallStart("main", "t1", "run_plan", {"goal": "build the scene\nmore"}),
            StepStarted("s1", "plan", 1, 2, "general-purpose"),
            RunStarted("w1", "plan / general-purpose"),
            SubagentStarted("sub1", "w1", "researcher", "tc2"),
            TextMessageEnd("sub1", "m2", "", 50, 5, False),
            SubagentFinished("sub1", "ok", 50, 5, 1.5, 1.0),
            TextMessageEnd("w1", "m3", "", 30, 3, False),
            StepFinished("s1", "plan", 1, "done", 1, 4.0, 80, 8, "ok", model_calls=2),
            ToolCallEnd("main", "t1", "run_plan", "ok", "1 step", 4.0),
            TextMessageEnd("main", "m4", "done", 20, 2, False),
        ]
        with self.recorder.turn("c1", "plan a city\nsecond line"):
            for event in events:
                self.recorder.emit(event)
                if isinstance(event, TextMessageEnd):  # each model answer is one counted call
                    self.counter.add(1, event.input_tokens, event.output_tokens)
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        self.assertEqual((run.chat, run.model, run.prompt, run.status), ("c1", "m", "plan a city", "done"))
        self.assertEqual((run.model_calls, run.input_tokens, run.output_tokens), (4, 200, 20))
        parts = {p.part: p for p in run.parts}
        self.assertEqual([(p.kind, p.label, p.parent) for p in run.parts],
                         [("plan", "build the scene", None), ("step", "1/2 general-purpose", 1),
                          ("subagent", "researcher", 2)])
        self.assertEqual((parts[1].model_calls, parts[1].input_tokens, parts[1].output_tokens),
                         (2, 80, 8))
        self.assertEqual((parts[2].model_calls, parts[2].input_tokens, parts[2].output_tokens),
                         (2, 80, 8))
        self.assertEqual((parts[3].model_calls, parts[3].input_tokens, parts[3].output_tokens),
                         (1, 50, 5))
        self.assertEqual(parts[3].status, "ok")
        self.assertEqual(parts[2].status, "done")

    def test_every_event_reaches_the_sink_unchanged_and_in_order(self):
        events = [RunStarted("main", "main"), TextMessageEnd("main", "m1", "hi", 1, 1, False)]
        with self.recorder.turn("c1", "hi"):
            for event in events:
                self.recorder.emit(event)
        self.assertEqual(len(self.sink.events), 2)
        self.assertTrue(all(a is b for a, b in zip(self.sink.events, events)))

    def test_events_outside_a_turn_pass_through_untouched(self):
        event = RunStarted("main", "main")
        self.recorder.emit(event)
        self.assertEqual(self.sink.events, [event])
        self.assertEqual(self.store.recent("alice"), [])

    def test_a_turn_that_raises_is_saved_as_error_and_the_error_propagates(self):
        with self.assertRaises(ValueError):
            with self.recorder.turn("c1", "hi"):
                self.recorder.emit(RunError("main", "main", "boom"))
                raise ValueError("boom")
        self.assertEqual(self.store.recent("alice")[0].status, "error")

    def test_a_run_error_event_marks_the_turn_error(self):
        with self.recorder.turn("c1", "hi"):
            self.recorder.emit(RunError("main", "main", "provider down"))
        self.assertEqual(self.store.recent("alice")[0].status, "error")

    def test_ctrl_c_is_saved_as_interrupted(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.recorder.turn("c1", "hi"):
                raise KeyboardInterrupt
        self.assertEqual(self.store.recent("alice")[0].status, "interrupted")

    def test_an_unfinished_part_is_saved_as_interrupted(self):
        with self.recorder.turn("c1", "hi"):
            self.recorder.emit(SubagentStarted("sub1", "main", "researcher", "tc"))
        listed = self.store.recent("alice")[0]
        run = self.store.get("alice", listed.id)
        self.assertEqual([(p.kind, p.status) for p in run.parts], [("subagent", "interrupted")])

    def test_a_failing_store_never_breaks_the_turn_and_is_reported_once(self):
        def broken(run):
            raise OSError("disk full")
        self.store.save = broken
        events = [RunStarted("main", "main"), TextMessageEnd("main", "m", "a", 1, 1, False)]
        for _ in range(2):
            with self.recorder.turn("c1", "hi"):
                for event in events:
                    self.recorder.emit(event)
        self.assertEqual(len(self.sink.events), 4)
        self.assertEqual(self.notes, ["Couldn't record this run: disk full"])

    def test_a_failing_observer_never_breaks_the_turn(self):
        def broken(event):
            raise RuntimeError("bad event")
        self.recorder._observe = broken  # the recorder's own bookkeeping fails
        event = RunStarted("main", "main")
        with self.recorder.turn("c1", "hi"):
            self.recorder.emit(event)
            self.recorder.emit(event)
        self.assertEqual(self.sink.events, [event, event])
        self.assertEqual(self.notes, ["Couldn't record this run: bad event"])
        self.assertEqual(self.store.recent("alice"), [])  # a half-observed turn is not saved


class TurnUsageTest(unittest.TestCase):
    """A turn's calls are the ones that started in it, wherever they run: a pool worker's call
    counts for its turn, and a call that ends after its turn closed never lands in the next one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")
        self.counter = UsageCounter()
        self.recorder = RunRecorder(Collect(), self.store, self.counter, user="alice", model="m")
        self.model = UsageModel(callbacks=[self.counter])
        self.pool = DaemonPool(2)

    def tearDown(self):
        self.pool.shutdown()
        self.store.close()
        self.tmp.cleanup()

    def saved(self):
        """The saved runs, oldest first."""
        return [self.store.get("alice", listed.id) for listed in reversed(self.store.recent("alice"))]

    def test_a_model_call_on_a_pool_worker_counts_for_its_turn(self):
        with self.recorder.turn("c1", "one"):
            self.pool.submit(self.model.invoke, "hi").result(timeout=5)
        run = self.saved()[0]
        self.assertEqual((run.model_calls, run.input_tokens, run.output_tokens), (1, 100, 7))

    def test_a_call_that_ends_after_its_turn_does_not_count_in_the_next_turn(self):
        gate, finished = threading.Event(), threading.Event()

        def late():
            gate.wait(5)
            self.model.invoke("late")
            finished.set()

        with self.recorder.turn("c1", "one"):
            self.pool.submit(late)  # abandoned: the turn ends without waiting for it
        with self.recorder.turn("c1", "two"):
            gate.set()
            self.assertTrue(finished.wait(5))  # the late call ends during the second turn
        first, second = self.saved()
        self.assertEqual((first.model_calls, second.model_calls), (0, 0))
        self.assertEqual(self.counter.snapshot(), (1, 100, 7))  # the process still counts it

    def test_two_turns_in_a_row_are_separate(self):
        for prompt in ("one", "two"):
            with self.recorder.turn("c1", prompt):
                self.model.invoke("hi")
        self.assertEqual([(run.model_calls, run.input_tokens, run.output_tokens) for run in self.saved()],
                         [(1, 100, 7), (1, 100, 7)])


class ChatsMoveRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saver_cm = SqliteSaver.from_conn_string(str(Path(self.tmp.name) / "checkpoints.sqlite"))
        self.saver = self.saver_cm.__enter__()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")
        self.chats = Chats(self.saver, None, "alice", runs=self.store)

    def tearDown(self):
        self.store.close()
        self.saver_cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def make_chat(self, name):
        from deep_agents_demo.persistence.chats import thread_id
        config = {"configurable": {"thread_id": thread_id("alice", name), "checkpoint_ns": ""}}
        self.saver.put(config, empty_checkpoint(), {"source": "input", "step": -1, "parents": {}}, {})

    def test_deleting_a_chat_removes_its_runs(self):
        self.make_chat("c1")
        self.store.save(make_run("r1", chat="c1"))
        self.store.save(make_run("r2", chat="c2"))
        self.chats.delete("c1")
        self.assertIsNone(self.store.get("alice", "r1"))
        self.assertIsNotNone(self.store.get("alice", "r2"))

    def test_renaming_a_chat_moves_its_runs(self):
        self.make_chat("c1")
        self.store.save(make_run("r1", chat="c1"))
        self.chats.rename("c1", "c2")
        self.assertEqual(self.store.get("alice", "r1").chat, "c2")

    def test_a_failed_rename_puts_the_runs_back_with_the_chat(self):
        self.make_chat("c1")
        self.store.save(make_run("r1", chat="c1"))

        with failing(self.saver, "cursor", when=lambda *args, **kwargs: kwargs.get("transaction")):
            # the checkpoints can't move (reads still work, so the chat is found)
            with self.assertRaisesRegex(ValueError, "nothing was changed"):
                self.chats.rename("c1", "c2")
        self.assertEqual(self.store.get("alice", "r1").chat, "c1")


class ConcurrentPartsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")
        self.recorder = RunRecorder(Collect(), self.store, FakeCounter(), user="alice", model="m")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_concurrent_steps_nest_their_own_runs(self):
        """Swarm workers open their steps at once, each on its own thread: a worker's subagent
        nests under its own step, not under the other worker's."""
        barrier = threading.Barrier(2)

        def worker(step_id, run_id, name):
            self.recorder.emit(StepStarted(step_id, "swarm", 1, 2, name))
            self.recorder.emit(RunStarted(run_id, name))
            barrier.wait()  # both steps are open now
            sub = f"sub-{run_id}"
            self.recorder.emit(SubagentStarted(sub, run_id, f"{name}-helper", f"tc-{run_id}"))
            self.recorder.emit(TextMessageEnd(sub, "m", "x", 10, 1, False))
            self.recorder.emit(SubagentFinished(sub, "ok", 10, 1, 0.5, 0.0))
            barrier.wait()
            self.recorder.emit(StepFinished(step_id, "swarm", 1, "done", 1, 1.0, 10, 1, "", 1))

        with self.recorder.turn("c1", "compare two"):
            self.recorder.emit(ToolCallStart("main", "t1", "run_swarm", {"goal": "compare"}))
            threads = [threading.Thread(target=worker, args=("s1", "r1", "alpha")),
                       threading.Thread(target=worker, args=("s2", "r2", "beta"))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.recorder.emit(ToolCallEnd("main", "t1", "run_swarm", "ok", "2 tasks", 2.0))
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        by_label = {part.label: part for part in run.parts}
        step_of = {"helper-alpha": "1/2 alpha", "helper-beta": "1/2 beta"}
        for name in ("alpha", "beta"):
            helper = by_label[f"{name}-helper"]
            step = by_label[f"1/2 {name}"]
            self.assertEqual(helper.parent, step.part, step_of)
            self.assertEqual(step.parent, by_label["compare"].part)
        self.assertEqual(by_label["compare"].model_calls, 2)  # the steps' calls, one each
        self.assertEqual(by_label["compare"].input_tokens, 20)

    def test_a_call_counts_the_model_answers_made_directly_under_it(self):
        """A room's moderator answers outside any step: its calls and tokens count on the room,
        and a notice with no tokens (the budget's stop) is no call."""
        with self.recorder.turn("c1", "room"):
            self.recorder.emit(RunStarted("main", "main"))
            self.recorder.emit(ToolCallStart("main", "t1", "start_room", {"question": "which one?"}))
            self.recorder.emit(RunStarted("mod", "room / moderator"))
            self.recorder.emit(TextMessageEnd("mod", "m1", "summary", 40, 4, False))
            self.recorder.emit(TextMessageEnd("mod", "m2", "[Stopped]", 0, 0, False))
            self.recorder.emit(ToolCallEnd("main", "t1", "start_room", "ok", "done", 3.0))
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        room = run.parts[0]
        self.assertEqual((room.kind, room.label), ("room", "which one?"))
        self.assertEqual((room.model_calls, room.input_tokens, room.output_tokens), (1, 40, 4))

    def test_a_room_from_the_command_is_one_room_part_with_its_turns_nested(self):
        """/room: no tool call starts the room, so a scope does: one room part, its two turns as
        its steps (not two top-level rooms)."""
        with self.recorder.turn("c1", "/room which one?"):
            with self.recorder.scope("room", "which one?"):
                for number in (1, 2):
                    self.recorder.emit(StepStarted(f"s{number}", "room", number, 3, "alpha"))
                    self.recorder.emit(StepFinished(f"s{number}", "room", number, "done", 1, 1.0, 10, 1, "", 1))
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        rooms = [part for part in run.parts if part.kind == "room"]
        steps = [part for part in run.parts if part.kind == "step"]
        self.assertEqual([(r.label, r.parent) for r in rooms], [("which one?", None)])
        self.assertEqual([s.parent for s in steps], [rooms[0].part, rooms[0].part])
        self.assertEqual(len(run.parts), 3)
        self.assertEqual((rooms[0].status, rooms[0].model_calls), ("done", 2))

    def test_a_scope_that_fails_is_saved_as_an_error_part_and_the_error_propagates(self):
        with self.assertRaises(ValueError):
            with self.recorder.turn("c1", "/room x"):
                with self.recorder.scope("room", "x"):
                    raise ValueError("boom")
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        self.assertEqual([(p.kind, p.status) for p in run.parts], [("room", "error")])

    def test_a_refused_scope_is_saved_as_refused_not_done(self):
        """A /room refused before it starts (start_room returned an error) is no finished room."""
        with self.recorder.turn("c1", "/room x"):
            with self.recorder.scope("room", "x") as room:
                room.status = "refused"
        run = self.store.get("alice", self.store.recent("alice")[0].id)
        self.assertEqual([(p.kind, p.status, p.model_calls) for p in run.parts], [("room", "refused", 0)])

    def test_a_scope_outside_a_turn_does_nothing(self):
        with self.recorder.scope("room", "x"):
            pass
        self.assertEqual(self.store.recent("alice"), [])


@contextmanager
def failing(target, method, when=lambda *args, **kwargs: True):
    """Make ``target.method`` raise OSError on the calls ``when`` picks, until the block ends."""
    original = getattr(target, method)

    def flaky(*args, **kwargs):
        if when(*args, **kwargs):
            raise OSError("disk full")
        return original(*args, **kwargs)
    with patch.object(target, method, side_effect=flaky):
        yield


class ChatStepFailureTests(unittest.TestCase):
    """Delete and rename with one step failing: the chat is whole, and a retry finishes it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saver_cm = SqliteSaver.from_conn_string(str(Path(self.tmp.name) / "checkpoints.sqlite"))
        self.saver = self.saver_cm.__enter__()
        self.store = RunStore(Path(self.tmp.name) / "runs.sqlite")
        self.outputs = OutputStore(Path(self.tmp.name) / "outputs")
        self.chats = Chats(self.saver, None, "alice", outputs=self.outputs, runs=self.store)
        self.main = thread_id("alice", "c1")
        self.put_thread(self.main)
        self.put_thread(self.main + ":plan:p1")  # a plan's agent thread belongs to the chat
        self.store.save(make_run("r1", chat="c1"))
        self.outputs.save("alice", "c1", "plan.md", b"# Plan")

    def tearDown(self):
        self.store.close()
        self.saver_cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def put_thread(self, thread):
        config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
        self.saver.put(config, empty_checkpoint(), {"source": "input", "step": -1, "parents": {}}, {})

    def plan_thread(self, name):
        return self.saver.get_tuple({"configurable": {"thread_id": thread_id("alice", name) + ":plan:p1",
                                                      "checkpoint_ns": ""}})

    def assert_chat_under(self, name):
        """Everything of the chat is under ``name``, and nothing is under the other name."""
        other = "c2" if name == "c1" else "c1"
        self.assertEqual(self.chats.names(), [name])
        self.assertEqual([run.chat for run in self.store.recent("alice")], [name])
        self.assertEqual(self.outputs.read("alice", name, "plan.md"), b"# Plan")
        self.assertFalse(self.outputs.folder("alice", other).exists())
        self.assertIsNotNone(self.plan_thread(name))
        self.assertIsNone(self.plan_thread(other))

    def delete_failing(self, target, method, when=lambda *args, **kwargs: True):
        with failing(target, method, when):
            with self.assertRaisesRegex(ValueError, "may be partly deleted"):
                self.chats.delete("c1")
        self.assertIn("c1", self.chats.names())  # still listed, so the delete can be run again
        self.chats.delete("c1")  # the retry
        self.assertEqual(self.chats.names(), [])
        self.assertEqual(self.store.recent("alice"), [])
        self.assertFalse(self.outputs.folder("alice", "c1").exists())
        self.assertIsNone(self.plan_thread("c1"))

    def test_a_delete_blocked_by_a_file_in_use_says_to_close_it(self):
        with failing(self.outputs, "delete_thread"):
            with self.assertRaisesRegex(ValueError, "open in another program, close it first"):
                self.chats.delete("c1")

    def test_a_delete_failing_on_the_files_is_finished_by_a_retry(self):
        self.delete_failing(self.outputs, "delete_thread")

    def test_a_delete_failing_on_the_runs_is_finished_by_a_retry(self):
        self.delete_failing(self.store, "delete_chat")

    def test_a_delete_failing_on_the_checkpoints_is_finished_by_a_retry(self):
        self.delete_failing(self.saver, "delete_thread", when=lambda thread: thread == self.main)

    def test_a_rename_failing_on_the_checkpoints_moves_nothing(self):
        with failing(self.saver, "cursor", when=lambda *args, **kwargs: kwargs.get("transaction")):
            with self.assertRaisesRegex(ValueError, "nothing was changed"):
                self.chats.rename("c1", "c2")
        self.assert_chat_under("c1")

    def test_a_rename_failing_on_the_files_moves_nothing(self):
        with failing(self.outputs, "rename_thread"):
            with self.assertRaisesRegex(ValueError, "nothing was changed"):
                self.chats.rename("c1", "c2")
        self.assert_chat_under("c1")

    def test_a_rename_failing_on_the_runs_is_put_back(self):
        with failing(self.store, "rename_chat"):
            with self.assertRaisesRegex(ValueError, "nothing was changed"):
                self.chats.rename("c1", "c2")
        self.assert_chat_under("c1")

    def test_a_rename_blocked_by_a_file_in_use_says_to_close_it(self):
        with failing(self.outputs, "rename_thread"):
            with self.assertRaisesRegex(ValueError, "open in another program, close it first"):
                self.chats.rename("c1", "c2")

    def test_a_rename_whose_owner_write_fails_moves_nothing(self):
        with patch("deep_agents_demo.persistence.outputs._write", side_effect=OSError("in use")):
            with self.assertRaisesRegex(ValueError, "nothing was changed"):
                self.chats.rename("c1", "c2")
        self.assert_chat_under("c1")

    def test_a_rename_whose_checkpoint_undo_fails_names_the_fix(self):
        undo_fails = lambda old, new: (old, new) == ("c2", "c1")
        with failing(self.store, "rename_chat"), failing(self.chats, "_move_checkpoints", when=undo_fails):
            with self.assertRaisesRegex(ValueError, "switch to it") as caught:
                self.chats.rename("c1", "c2")
        self.assertIn("'c1'", str(caught.exception))
        self.assertIn("/rename c1", str(caught.exception))
        self.assertIn("--rename-thread c2 c1", str(caught.exception))
        self.assertTrue(self.chats.exists("c2"))  # the chat is still under the new name
        self.chats.rename("c2", "c1")  # the fix the error gives, run for real
        self.assert_chat_under("c1")

    def test_a_rename_whose_files_undo_fails_names_the_fix(self):
        undo_fails = lambda user, old, new: (old, new) == ("c2", "c1")
        with failing(self.store, "rename_chat"), failing(self.outputs, "rename_thread", when=undo_fails):
            with self.assertRaisesRegex(ValueError, "switch to it") as caught:
                self.chats.rename("c1", "c2")
        self.assertIn("/rename c1", str(caught.exception))
        self.assertTrue(self.chats.exists("c2"))
        self.assertEqual(self.outputs.read("alice", "c2", "plan.md"), b"# Plan")
        self.chats.rename("c2", "c1")
        self.assert_chat_under("c1")

    def test_a_failed_rename_leaves_the_session_on_the_old_name(self):
        said = []
        session = ChatSession(self.chats, "c1", say=said.append)
        with failing(self.store, "rename_chat"):
            session.handle("/rename c2")
        self.assertEqual(session.current, "c1")
        self.assertIn("nothing was changed", said[-1])

    def test_a_rename_that_ends_under_the_new_name_moves_the_session_there(self):
        undo_fails = lambda old, new: (old, new) == ("c2", "c1")
        said = []
        session = ChatSession(self.chats, "c1", say=said.append)
        with failing(self.store, "rename_chat"), failing(self.chats, "_move_checkpoints", when=undo_fails):
            session.handle("/rename c2")
        self.assertEqual(session.current, "c2")
        self.assertIn("/rename c1", said[-1])


if __name__ == "__main__":
    unittest.main()

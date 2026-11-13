"""Ctrl+C in the CLI: plans and rooms run under the pause/stop handler when
interactive, and Ctrl+C keeps its usual meaning in one-shot runs."""
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deep_agents_demo.config import ChatSettings, Paths, Settings
from deep_agents_demo.core.approvals import Approvals
from deep_agents_demo.ui.cli.interrupts import ctrl_c_controls
from deep_agents_demo.ui.cli.repl import runners_for


class Stopping:
    """A graph whose run is cut short (a second Ctrl+C), so no prompt follows."""

    def stream(self, *args, **kwargs):
        raise KeyboardInterrupt


class Target:
    def __init__(self):
        self.pause_requested, self.stopped = False, False

    def request_pause(self):
        self.pause_requested = True

    def stop(self):
        self.stopped = True


class RunnersForTests(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.settings = Settings(chat=ChatSettings(api_key="unused", model="unused", base_url=None),
                                 paths=Paths(data=root / "data", workspace=root / "ws",
                                             checkpoints=root / "data" / "c.sqlite"))
        self.seen = []

        def make(name, extra=()):
            self.seen.append(signal.getsignal(signal.SIGINT))
            return Stopping()
        make.names = frozenset({"general-purpose", "reviewer", "moderator", "critic", "helper"})
        self.runtime = SimpleNamespace(make_worker=make, agents=None, chats=SimpleNamespace(user="u"))

    def run_both(self, interactive):
        runners = runners_for(self.runtime, self.settings, "t", Approvals(), interactive=interactive, quiet=True)
        plan = runners["run_plan"]({"type": "run_plan", "goal": "g",
                                     "steps": [{"task": "one", "agent": "general-purpose"}]})
        room = runners["start_room"]({"type": "start_room", "question": "q",
                                      "participants": ["critic", "helper"]})
        return plan, room

    def test_interactive_runs_install_the_handler_and_restore_it(self):
        before = signal.getsignal(signal.SIGINT)
        plan, room = self.run_both(interactive=True)
        self.assertEqual(len(self.seen), 2)
        self.assertTrue(all(handler is not before for handler in self.seen), "plan and room run under Ctrl+C")
        self.assertIs(signal.getsignal(signal.SIGINT), before)
        self.assertEqual(plan["steps"][0]["status"], "stopped")
        self.assertEqual(room["ended_by"], "stopped")

    def test_one_shot_runs_leave_ctrl_c_alone(self):
        before = signal.getsignal(signal.SIGINT)
        self.run_both(interactive=False)
        self.assertEqual(self.seen, [before, before])


class NewAgentsInTheSameMessageTests(unittest.TestCase):
    """An agent approved earlier in this message can join a room or a plan at once."""

    def test_the_room_uses_agents_made_earlier_in_the_message(self):
        root = Path(tempfile.mkdtemp())
        settings = Settings(chat=ChatSettings(api_key="unused", model="unused", base_url=None),
                            paths=Paths(data=root / "data", workspace=root / "ws",
                                        checkpoints=root / "data" / "c.sqlite"))
        built = []

        def factory(names):
            def make(name, extra=()):
                built.append(name)
                return Stopping()
            make.names = frozenset(names)
            return make
        runtime = SimpleNamespace(make_worker=factory({"moderator", "critic"}), agents=None,
                                  chats=SimpleNamespace(user="u"))

        def refresh():  # an approved create_agent: the rebuild adds "helper"
            runtime.make_worker = factory({"moderator", "critic", "helper"})
            return True
        runtime.refresh = refresh
        runners = runners_for(runtime, settings, "t", Approvals(), interactive=False, quiet=True)
        room = runners["start_room"]({"type": "start_room", "question": "q",
                                      "participants": ["critic", "helper"]})
        self.assertNotIn("error", room)
        self.assertEqual(built, ["moderator"])  # it got as far as running the room


class CtrlCControlsTests(unittest.TestCase):
    def test_an_unknown_previous_handler_restores_the_default(self):
        from unittest.mock import patch
        calls = []

        def fake_signal(signum, handler):
            calls.append(handler)
            return None  # the previous handler wasn't installed from Python
        with patch("deep_agents_demo.ui.cli.interrupts.signal.signal", side_effect=fake_signal):
            with ctrl_c_controls(Target(), interactive=True):
                pass
        self.assertIsNotNone(calls[-1])
        self.assertIs(calls[-1], signal.default_int_handler)

    def test_a_handled_pause_lets_the_next_press_pause_again(self):
        target = Target()
        with ctrl_c_controls(target, interactive=True, label="[plan]"):
            handler = signal.getsignal(signal.SIGINT)
            handler(2, None)
            self.assertTrue(target.pause_requested)
            target.pause_requested = False  # the runner handled the pause
            handler(2, None)
            self.assertEqual((target.pause_requested, target.stopped), (True, False))
            with self.assertRaises(KeyboardInterrupt):
                handler(2, None)
            self.assertTrue(target.stopped)

    def test_the_pause_notice_goes_through_notify_when_given(self):
        target, said = Target(), []
        with ctrl_c_controls(target, interactive=True, label="[plan]", notify=said.append):
            signal.getsignal(signal.SIGINT)(2, None)
        self.assertTrue(target.pause_requested)
        self.assertEqual(len(said), 1)
        self.assertIn("Pausing after this one", said[0])
        self.assertIn("[plan]", said[0])


if __name__ == "__main__":
    unittest.main()

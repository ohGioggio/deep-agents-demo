"""One contract, two runners: plans and rooms honour the same budget, pause and stop
behaviour (``Runner``), driven through the ports the way any front end drives them."""
import unittest

from test_agents import SEEN, call
from test_orchestration import RunnerBase, reply
from test_rooms import RoomBase

from deep_agents_demo.core.ports import Interruptible
from deep_agents_demo.features.orchestration.runner import Runner


class Interrupting:
    """A graph whose run is cut by a second Ctrl+C."""

    def stream(self, *args, **kwargs):
        raise KeyboardInterrupt


class RunnerContract:
    """Mixed into a TestCase giving ``make_runner(*responses, **options)``, ``start(runner)``,
    ``stopped(outcome)`` and ``budget_ended(outcome)``."""

    def test_is_an_interruptible_runner(self):
        runner = self.make_runner()
        self.assertIsInstance(runner, Runner)
        self.assertIsInstance(runner, Interruptible)

    def test_budget_exhaustion_wraps_up(self):
        runner = self.make_runner(*[call("ls", {"path": "/"}, f"l{i}") for i in range(10)],
                                  reply("The summary."), interactive=False)
        runner.budget_size = 2
        self.assertTrue(self.budget_ended(self.start(runner)))

    def test_request_pause_asks_before_continuing(self):
        runner = self.make_runner(answers=["/stop"])
        runner.request_pause()
        outcome = self.start(runner)
        self.assertTrue(self.stopped(outcome))
        self.assertTrue(runner.interaction.asked, "the user was asked before going on")

    def test_stop_before_start_runs_no_agent(self):
        runner = self.make_runner()
        runner.stop()
        SEEN.clear()
        outcome = self.start(runner)
        self.assertTrue(self.stopped(outcome))
        self.assertEqual([k for k, _ in SEEN if k == "messages"], [])

    def test_keyboard_interrupt_stops_and_returns(self):
        runner = self.make_runner()
        real = runner.make

        def make(name, extra=()):
            return Interrupting() if extra else real(name, extra)
        make.names = real.names
        runner.make = make
        self.assertTrue(self.stopped(self.start(runner)))


class PlanRunnerContract(RunnerContract, RunnerBase, unittest.TestCase):
    def make_runner(self, *responses, **options):
        return self.runner(*responses, **options)

    def start(self, runner):
        return self.plan(runner, ("one", "general-purpose"), ("two", "general-purpose"))

    def stopped(self, outcome):
        return all(s["status"] == "stopped" for s in outcome["steps"])

    def budget_ended(self, outcome):
        return all("budget used up" in s["notes"] for s in outcome["steps"])


class RoomRunnerContract(RunnerContract, RoomBase, unittest.TestCase):
    def make_runner(self, *responses, **options):
        return self.runner(reply("NEXT: critic\nGo."), *responses, **options)

    def start(self, runner):
        return self.room(runner)

    def stopped(self, outcome):
        return outcome["ended_by"] == "stopped"

    def budget_ended(self, outcome):
        return outcome["ended_by"] == "budget"


if __name__ == "__main__":
    unittest.main()

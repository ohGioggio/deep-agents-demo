"""The --runs and --show-run commands: flags, plain text in a pipe, Rich in a terminal."""
import contextlib
import dataclasses
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from deep_agents_demo.config import Paths, Settings
from deep_agents_demo.persistence.runs import Run, RunPart, RunStore
from deep_agents_demo.ui.cli.render import format_run, run_part_line, run_totals_line
from deep_agents_demo.ui.cli import commands, main as cli
from deep_agents_demo.ui.cli.richui.tables import RichTables


def make_run(run_id, user="alice", chat="parks", started_at=None, calls=3, tokens_in=300, tokens_out=30,
             parts=()):
    started = time.time() if started_at is None else started_at
    return Run(id=run_id, user=user, chat=chat, started_at=started, ended_at=started + 5, seconds=5.0,
               model="m", prompt=f"prompt {run_id}", model_calls=calls, input_tokens=tokens_in,
               output_tokens=tokens_out, status="done", parts=list(parts))


class ParseTests(unittest.TestCase):
    def test_runs_defaults_to_20_and_takes_a_count(self):
        self.assertEqual(cli.parse_args(["--runs"])[1].runs, 20)
        self.assertEqual(cli.parse_args(["--runs", "5"])[1].runs, 5)
        self.assertIsNone(cli.parse_args([])[1].runs)

    def test_show_run_takes_an_id(self):
        self.assertEqual(cli.parse_args(["--show-run", "abc123"])[1].show_run, "abc123")


class DispatchTests(unittest.TestCase):
    def test_each_flag_runs_its_command(self):
        settings = Settings.from_env({"OPENAI_API_KEY": "k", "AGENT_MODEL": "m"})
        with patch.object(cli, "commands") as cmds, patch.object(cli, "run_chat") as run:
            cmds.UsageError = commands.UsageError
            cli.dispatch(settings, cli.parse_args(["--runs", "7", "--user", "alice"])[1])
            cli.dispatch(settings, cli.parse_args(["--show-run", "abc", "--user", "alice"])[1])
        cmds.list_runs.assert_called_once_with(settings, "alice", 7)
        cmds.show_run.assert_called_once_with(settings, "alice", "abc")
        run.assert_not_called()


class RunCommandsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        data = Path(self.tmp.name) / "data"
        data.mkdir()
        self.settings = dataclasses.replace(Settings.from_env({}),
                                            paths=Paths(data=data))
        store = RunStore(data / "runs.sqlite")
        now = time.time()
        self.old = make_run("old", started_at=now - 3 * 86400, calls=9, tokens_in=9000, tokens_out=900)
        self.first = make_run("first", started_at=now - 120)
        self.second = make_run("second", started_at=now - 60, calls=2, tokens_in=200, tokens_out=20)
        self.theirs = make_run("theirs", user="bob", started_at=now - 30)
        for run in (self.old, self.first, self.second, self.theirs):
            store.save(run)
        store.close()

    def tearDown(self):
        self.tmp.cleanup()

    def plain(self, command, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            command(self.settings, *args)
        return out.getvalue()

    def test_runs_lists_the_users_newest_first_with_today_totals(self):
        text = self.plain(commands.list_runs, "alice", 20)
        self.assertLess(text.index("second"), text.index("first"))
        self.assertIn("old", text)
        self.assertNotIn("theirs", text)
        # Today is the two runs from today only: 3 + 2 calls, 300 + 200 in, 30 + 20 out.
        self.assertIn("Today: 5 calls · 500 in / 50 out", text)

    def test_runs_respects_the_limit(self):
        text = self.plain(commands.list_runs, "alice", 1)
        self.assertIn("second", text)
        self.assertNotIn("first", text)

    def test_runs_with_none_says_so(self):
        text = self.plain(commands.list_runs, "carol", 20)
        self.assertIn("No runs yet", text)

    def test_runs_limit_below_one_is_a_usage_error(self):
        with self.assertRaisesRegex(commands.UsageError, "at least 1"):
            commands.list_runs(self.settings, "alice", 0)

    def test_show_run_prints_the_parts_as_a_tree(self):
        parts = [
            RunPart(part=1, parent=None, kind="plan", label="write the park plan", model_calls=4,
                    input_tokens=400, output_tokens=40, seconds=3.0),
            RunPart(part=2, parent=1, kind="step", label="pick the site", model_calls=2,
                    input_tokens=200, output_tokens=20, seconds=1.0),
            RunPart(part=3, parent=2, kind="subagent", label="researcher", model_calls=1,
                    input_tokens=100, output_tokens=10, seconds=0.5),
        ]
        store = RunStore(Path(self.tmp.name) / "data" / "runs.sqlite")
        store.save(make_run("tree", parts=parts))
        store.close()
        lines = self.plain(commands.show_run, "alice", "tree").splitlines()
        self.assertIn("Prompt: prompt tree", lines)
        plan = next(line for line in lines if "plan" in line and "write the park plan" in line)
        step = next(line for line in lines if "pick the site" in line)
        sub = next(line for line in lines if "researcher" in line)
        indent = lambda line: len(line) - len(line.lstrip())
        self.assertLess(indent(plan), indent(step))
        self.assertLess(indent(step), indent(sub))

    def test_show_run_with_no_parts_says_so(self):
        self.assertIn("No parts recorded", self.plain(commands.show_run, "alice", "second"))

    def test_show_run_unknown_id_is_a_clear_error_and_exit_1(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            commands.show_run(self.settings, "alice", "theirs")  # another user's run
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("No run with id 'theirs' for user 'alice'", err.getvalue())

    def test_show_run_with_an_empty_id_is_a_usage_error_and_starts_no_chat(self):
        with patch.object(cli, "run_chat") as run, \
                self.assertRaisesRegex(commands.UsageError, "--show-run needs a value"):
            cli.dispatch(self.settings, cli.parse_args(["--show-run", "", "--user", "alice"])[1])
        run.assert_not_called()


class RichRunTablesTests(unittest.TestCase):
    def console(self):
        out = io.StringIO()
        return Console(file=out, width=160, force_terminal=True, color_system=None), out

    def test_runs_table_has_ids_totals_and_the_empty_sentence(self):
        console, out = self.console()
        RichTables(console).runs([make_run("abc123", calls=4, tokens_in=40, tokens_out=4)],
                                 {"runs": 1, "model_calls": 4, "input_tokens": 40, "output_tokens": 4})
        text = out.getvalue()
        self.assertIn("abc123", text)
        self.assertIn("Today: 4 calls · 40 in / 4 out", text)
        console, out = self.console()
        RichTables(console).runs([], {"runs": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0})
        self.assertIn("No runs yet", out.getvalue())

    def test_run_tree_shows_the_parts(self):
        console, out = self.console()
        run = make_run("abc", parts=[RunPart(part=1, parent=None, kind="room", label="the park question"),
                                     RunPart(part=2, parent=1, kind="step", label="first voice")])
        RichTables(console).run(run)
        text = out.getvalue()
        self.assertIn("room", text)
        self.assertIn("first voice", text)


if __name__ == "__main__":
    unittest.main()


class CallsWordingTests(unittest.TestCase):
    """One call is "1 call"; every other count, zero included, is "N calls"."""

    def test_plain_text_says_1_call_in_the_run_the_part_and_today(self):
        part = RunPart(1, None, "plan", "write the plan", 1, 10, 1, 2.0, "done")
        run = make_run("one", calls=1, parts=[part])
        totals = {"model_calls": 1, "input_tokens": 10, "output_tokens": 1}
        self.assertIn("Today: 1 call · 10 in / 1 out", run_totals_line(totals))
        self.assertIn("plan: write the plan  ·  1 call · 10 in / 1 out", run_part_line(part))
        self.assertIn("Total: 1 call · 300 in / 30 out", format_run(run))
        self.assertIn("plan: write the plan  ·  1 call ·", format_run(run))

    def test_plain_text_keeps_the_plural_for_other_counts(self):
        part = RunPart(1, None, "plan", "p", 0, 0, 0, 0.0, "done")
        totals = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.assertIn("Today: 0 calls", run_totals_line(totals))
        self.assertIn("0 calls", run_part_line(part))
        self.assertIn("Total: 2 calls", format_run(make_run("two", calls=2)))

    def test_rich_run_says_1_call(self):
        console = Console(file=io.StringIO(), width=160, force_terminal=True, color_system=None)
        out = console.file
        RichTables(console).run(make_run("one", calls=1))
        self.assertIn("Total: 1 call · 300 in / 30 out", out.getvalue())
        self.assertNotIn("1 calls", out.getvalue())

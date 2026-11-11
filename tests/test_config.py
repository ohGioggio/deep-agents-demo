"""Settings, CLI dispatch and the chat entry point."""
import unittest
from unittest.mock import MagicMock, patch

from deep_agents_demo.config import ChatSettings, Settings
from deep_agents_demo.ui.cli import commands
from deep_agents_demo.ui.cli import main as cli
from deep_agents_demo.ui.cli import repl as chat


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = Settings.from_env({})
        self.assertEqual((s.memory.enabled, s.memory.recall_mode, s.rag_enabled), (True, "auto", True))
        self.assertEqual((s.embeddings.model, s.embeddings.dimensions, s.embeddings.batch_size),
                         ("text-embedding-3-small", 1536, 16))
        self.assertEqual((s.sandbox.provider, s.sandbox.openshell_workdir), ("none", "/sandbox"))
        self.assertEqual(s.requests_per_minute, 0)

    def test_blank_values_fall_back_to_the_chat_settings(self):
        s = Settings.from_env({"OPENAI_API_KEY": "chat-key", "AGENT_MODEL": "chat-model",
                               "OPENAI_BASE_URL": "http://local/v1", "MEMORY_MODEL": "  ",
                               "EMBEDDING_API_KEY": ""})
        self.assertEqual((s.memory.model, s.memory.api_key, s.memory.base_url),
                         ("chat-model", "chat-key", "http://local/v1"))
        self.assertEqual(s.embeddings.api_key, "chat-key")
        own = Settings.from_env({"OPENAI_API_KEY": "chat-key", "EMBEDDING_API_KEY": "embed-key"})
        self.assertEqual(own.embeddings.api_key, "embed-key")

    def test_values_are_parsed(self):
        s = Settings.from_env({"MEMORY_ENABLED": "FALSE", "RAG_ENABLED": "false",
                               "MEMORY_RECALL": "Tool", "SANDBOX_PROVIDER": "OpenShell",
                               "EMBEDDING_DIMENSIONS": "4096", "API_REQUESTS_PER_MINUTE": "40"})
        self.assertEqual((s.memory.enabled, s.rag_enabled, s.memory.recall_mode, s.sandbox.provider),
                         (False, False, "tool", "openshell"))
        self.assertEqual((s.embeddings.dimensions, s.requests_per_minute), (4096, 40.0))

    def test_invalid_settings_are_reported_up_front(self):
        for env, message in (({"MEMORY_RECALL": "sometimes"}, "MEMORY_RECALL must be one of"),
                             ({"SANDBOX_PROVIDER": "local"}, "SANDBOX_PROVIDER must be one of"),
                             ({"SANDBOX_PROVIDER": "daytona"}, "DAYTONA_API_KEY"),
                             ({"EMBEDDING_DIMENSIONS": "big"}, "EMBEDDING_DIMENSIONS must be a number"),
                             ({"EMBEDDING_BATCH_SIZE": "0"}, "at least 1"),
                             ({"API_REQUESTS_PER_MINUTE": "fast"}, "must be a number")):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, message):
                Settings.from_env(env)

    def test_chat_problem(self):
        self.assertIsNone(ChatSettings("key", "model", None).problem())
        for api_key, model in ((None, "model"), ("key", None), ("your-api-key", "model"),
                               ("key", "your-model-id")):
            self.assertIn("OPENAI_API_KEY", ChatSettings(api_key, model, None).problem())

    def test_agent_settings(self):
        import os
        from pathlib import Path
        base = {"OPENAI_API_KEY": "k", "AGENT_MODEL": "m"}
        settings = Settings.from_env({**base, "AGENT_DIRS": f"one{os.pathsep}~/two"})
        self.assertEqual(settings.agent_dirs, (Path("one"), Path("~/two").expanduser()))
        self.assertEqual(Settings.from_env(base).subagent_max_calls, 25)
        self.assertEqual(Settings.from_env(base).agent_dirs, ())
        with self.assertRaisesRegex(ValueError, "SUBAGENT_MAX_CALLS"):
            Settings.from_env({**base, "SUBAGENT_MAX_CALLS": "0"})

    def test_bad_home_in_agent_dirs_is_a_settings_error(self):
        with patch("pathlib.Path.expanduser", side_effect=RuntimeError("Could not determine home directory.")):
            with self.assertRaisesRegex(ValueError, "AGENT_DIRS"):
                Settings.from_env({"OPENAI_API_KEY": "k", "AGENT_MODEL": "m", "AGENT_DIRS": "~nobody/x"})

    def test_orchestration_settings(self):
        base = {"OPENAI_API_KEY": "k", "AGENT_MODEL": "m"}
        settings = Settings.from_env(base)
        self.assertEqual((settings.plan_max_attempts, settings.orchestration_max_calls,
                          settings.subagent_max_parallel), (2, 150, 3))
        for name in ("PLAN_MAX_ATTEMPTS", "ORCHESTRATION_MAX_CALLS", "SUBAGENT_MAX_PARALLEL"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                Settings.from_env({**base, name: "0"})

    def test_room_max_turns(self):
        base = {"OPENAI_API_KEY": "k", "AGENT_MODEL": "m"}
        self.assertEqual(Settings.from_env(base).room_max_turns, 12)
        with self.assertRaisesRegex(ValueError, "ROOM_MAX_TURNS"):
            Settings.from_env({**base, "ROOM_MAX_TURNS": "0"})

    def test_settings_are_immutable(self):
        with self.assertRaises(Exception):
            Settings.from_env({}).rag_enabled = False


class DispatchTests(unittest.TestCase):
    def dispatch(self, *argv, env=None):
        settings = Settings.from_env({"OPENAI_API_KEY": "k", "AGENT_MODEL": "m"} if env is None else env)
        _parser, args = cli.parse_args(list(argv))
        with patch.object(cli, "commands") as cmds, patch.object(cli, "run_chat") as run:
            cmds.UsageError = commands.UsageError
            cli.dispatch(settings, args)
        return cmds, run

    def test_each_flag_runs_its_command(self):
        cmds, run = self.dispatch("--list-threads", "--user", "alice")
        cmds.list_threads.assert_called_once()
        self.assertEqual(cmds.list_threads.call_args.args[1:], ("alice", "city-demo"))
        run.assert_not_called()
        cmds, _ = self.dispatch("--rename-thread", "a", "b")
        self.assertEqual(cmds.rename_thread.call_args.args[2:], ("a", "b"))
        cmds, _ = self.dispatch("--delete-thread", "a", "--yes")
        self.assertTrue(cmds.delete_thread.call_args.kwargs["yes"])

    def test_chat_needs_a_model_but_management_does_not(self):
        with self.assertRaisesRegex(commands.UsageError, "OPENAI_API_KEY"):
            self.dispatch("--prompt", "hi", env={})
        cmds, _ = self.dispatch("--list-threads", env={})
        cmds.list_threads.assert_called_once()
        _, run = self.dispatch("--prompt", "hi", "--thread", "t1")
        self.assertEqual(run.call_args.args[1:], ("demo-user", "t1", "hi"))


class RunChatTests(unittest.TestCase):
    def test_a_slash_command_as_prompt_runs_once_and_exits(self):
        runtime = MagicMock()
        runtime.chats.exists.return_value = False
        runtime.chats.list.return_value = []
        with patch.object(chat, "build_runtime", return_value=runtime), patch.object(chat, "Resources"), \
                patch.object(chat, "prepare_workspace"), \
                patch.object(chat, "stream_turn") as stream, \
                patch("builtins.print"):
            chat.run_chat(Settings.from_env({}), "alice", "t1", prompt="/threads")
        stream.assert_not_called()
        runtime.chats.list.assert_called_once()

    def test_a_prompt_streams_one_turn(self):
        runtime = MagicMock()
        runtime.chats.user = "alice"
        with patch.object(chat, "build_runtime", return_value=runtime), patch.object(chat, "Resources"), \
                patch.object(chat, "prepare_workspace"), \
                patch.object(chat, "stream_turn") as stream:
            chat.run_chat(Settings.from_env({}), "alice", "t1", prompt="hello")
        stream.assert_called_once()
        self.assertEqual(stream.call_args.args[1], "hello")


class RegistryTests(unittest.TestCase):
    def test_every_command_is_in_help_and_dispatches(self):
        for name, method in chat.COMMANDS.items():
            names, usage, _summary = method.command
            self.assertIn(usage, chat.HELP)
            self.assertIn(name, names)
        self.assertEqual(chat.COMMANDS["chats"], chat.COMMANDS["threads"])
        said = []
        session = chat.ChatSession(MagicMock(), "c1", say=said.append)
        self.assertTrue(session.handle("/HELP"))
        self.assertEqual(said[-1], chat.HELP)
        session.handle("/nope")
        self.assertIn("Unknown command /nope", said[-1])


if __name__ == "__main__":
    unittest.main()

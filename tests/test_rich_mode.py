"""When the chat uses the Rich terminal UI and when it stays plain."""
import unittest

from deep_agents_demo.ui.cli.richui.mode import use_rich


class ModeTests(unittest.TestCase):
    def test_mode_is_plain_when_stdout_is_not_a_terminal(self):
        self.assertFalse(use_rich(interactive=True, plain=False, stdin_tty=True, stdout_tty=False))
        self.assertFalse(use_rich(interactive=True, plain=False, stdin_tty=False, stdout_tty=True))

    def test_mode_is_plain_for_prompt_and_plain_flag(self):
        self.assertFalse(use_rich(interactive=False, plain=False, stdin_tty=True, stdout_tty=True))
        self.assertFalse(use_rich(interactive=True, plain=True, stdin_tty=True, stdout_tty=True))

    def test_mode_is_rich_in_a_terminal(self):
        self.assertTrue(use_rich(interactive=True, plain=False, stdin_tty=True, stdout_tty=True))


if __name__ == "__main__":
    unittest.main()

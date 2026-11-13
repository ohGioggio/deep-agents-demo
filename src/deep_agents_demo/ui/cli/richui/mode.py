"""Whether the chat uses the Rich terminal UI or the plain text one."""


def use_rich(interactive: bool, plain: bool, stdin_tty: bool, stdout_tty: bool) -> bool:
    """Rich only in an interactive chat (no --prompt), without --plain, when both stdin and
    stdout are terminals; anything else (pipes, redirects, tests) keeps the plain output."""
    return interactive and not plain and stdin_tty and stdout_tty

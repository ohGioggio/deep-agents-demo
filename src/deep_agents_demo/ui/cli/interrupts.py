"""Ctrl+C while a plan or a room runs: the first press pauses, the second stops."""
import os
import signal
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager

from deep_agents_demo.core.ports import Interruptible


@contextmanager
def ctrl_c_controls(target: Interruptible, interactive: bool, label: str = "",
                    notify: Callable[[str], None] | None = None):
    """While the block runs, Ctrl+C calls ``target.request_pause()`` (first press) and
    ``target.stop()`` then raises KeyboardInterrupt (second press, stops now).

    Only in an interactive chat on the main thread: in a one-shot run nobody can
    answer a pause, so Ctrl+C keeps its usual meaning. ``notify`` (a renderer's way to
    print safely) receives the pause notice instead of a raw write to stderr.
    """
    if not interactive or threading.current_thread() is not threading.main_thread():
        yield
        return
    pressed = False
    previous = None

    def on_sigint(_signum, _frame):
        nonlocal pressed
        # A pause the runner already handled clears its flag: the next press pauses again.
        if getattr(target, "pause_requested", pressed):
            target.stop()
            signal.signal(signal.SIGINT, previous or signal.default_int_handler)
            raise KeyboardInterrupt
        pressed = True
        target.request_pause()
        message = f"{label} Pausing after this one (Ctrl+C again to stop now)."
        if notify is not None:
            notify(message)
            return
        # Not through Python's stdout: the handler can fire in the middle of a print.
        try:
            os.write(sys.stderr.fileno(), f"\n{message}\n".encode())
        except (OSError, ValueError, AttributeError):
            pass

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield
    finally:
        # None: the old handler wasn't installed from Python (e.g. embedded); use the default.
        signal.signal(signal.SIGINT, previous if previous is not None else signal.default_int_handler)

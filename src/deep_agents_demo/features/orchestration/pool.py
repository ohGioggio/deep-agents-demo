"""A small worker pool on daemon threads.

It has the part of ``ThreadPoolExecutor`` the orchestration uses: ``submit`` gives a
``concurrent.futures.Future`` (so ``wait`` works on it), at most ``max_workers`` tasks run at once,
and ``shutdown`` cancels the tasks not yet started. Unlike ``ThreadPoolExecutor``, its threads are
daemons and ``shutdown`` never joins them: Python joins non-daemon pool threads at exit, so a task
still running (a model call that can't be cancelled) would hold the process up.
"""
import contextvars
import logging
import queue
import threading
from concurrent.futures import Future
from itertools import count

_NAMES = count(1)


class _QuietAtExit(logging.Filter):
    """Drops the traceback ``concurrent.futures`` logs when a worker left running (a stopped
    swarm's model call) tries to save its state after Python began shutting down: quitting the
    chat then would print it. Every other callback error still shows."""

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        return not (isinstance(error, RuntimeError) and "new futures after" in str(error)
                    and "shutdown" in str(error))


logging.getLogger("concurrent.futures").addFilter(_QuietAtExit())


class DaemonPool:
    def __init__(self, max_workers: int):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.max_workers = max_workers
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._closed = False
        self._name = f"dad-pool-{next(_NAMES)}"

    def submit(self, fn, *args, **kwargs) -> Future:
        """Runs ``fn`` in a worker, in a copy of the caller's context: the task's model calls count
        for the chat turn that submitted it, even when the turn has ended since."""
        future = Future()
        context = contextvars.copy_context()
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot submit to a pool that was shut down")
            self._queue.put((future, context, fn, args, kwargs))
            if len(self._threads) < self.max_workers:
                thread = threading.Thread(target=self._work, daemon=True,
                                          name=f"{self._name}-{len(self._threads) + 1}")
                self._threads.append(thread)
                thread.start()
        return future

    def shutdown(self, cancel_pending: bool = True) -> None:
        """Stops the pool without waiting. Tasks not yet started are cancelled (unless
        ``cancel_pending`` is False, when they still run); a task already running runs on in its
        daemon thread, and its result is dropped by whoever stopped waiting for it."""
        cancelled = []
        with self._lock:
            self._closed = True
            if cancel_pending:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        cancelled.append(item[0])
            for _ in self._threads:  # one stop signal per thread, after any task still queued
                self._queue.put(None)
        for future in cancelled:
            future.cancel()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, context, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = context.run(fn, *args, **kwargs)
            except BaseException as error:  # the caller gets it from the future, as with an executor
                future.set_exception(error)
            else:
                future.set_result(result)

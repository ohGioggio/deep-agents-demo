"""DaemonPool: futures from daemon threads, at most max_workers at once, and a shutdown that never
waits for a running task."""
import threading
import time
import unittest
from concurrent.futures import Future, wait

from deep_agents_demo.features.orchestration.pool import DaemonPool


class DaemonPoolTests(unittest.TestCase):
    def setUp(self):
        self.release = threading.Event()
        self.addCleanup(self.release.set)  # no blocked worker outlives its test

    def test_submit_returns_a_future_with_the_result(self):
        pool = DaemonPool(2)
        future = pool.submit(lambda a, b: a + b, 2, 3)
        self.assertIsInstance(future, Future)
        self.assertEqual(future.result(timeout=5), 5)
        pool.shutdown()

    def test_an_exception_is_set_on_the_future(self):
        pool = DaemonPool(1)
        future = pool.submit(lambda: 1 / 0)
        self.assertIsInstance(future.exception(timeout=5), ZeroDivisionError)
        pool.shutdown()

    def test_concurrent_futures_wait_works_on_its_futures(self):
        pool = DaemonPool(2)
        futures = [pool.submit(time.sleep, 0.01) for _ in range(4)]
        done, not_done = wait(futures, timeout=5)
        self.assertEqual(len(done), 4)
        self.assertEqual(not_done, set())
        pool.shutdown()

    def test_runs_at_most_max_workers_at_once(self):
        lock, state = threading.Lock(), {"running": 0, "peak": 0}

        def work():
            with lock:
                state["running"] += 1
                state["peak"] = max(state["peak"], state["running"])
            time.sleep(0.05)
            with lock:
                state["running"] -= 1
        pool = DaemonPool(2)
        futures = [pool.submit(work) for _ in range(6)]
        wait(futures, timeout=5)
        self.assertTrue(all(f.done() for f in futures))
        self.assertEqual(state["peak"], 2)
        pool.shutdown()

    def test_worker_threads_are_daemons(self):
        pool = DaemonPool(1)
        thread = pool.submit(threading.current_thread).result(timeout=5)
        self.assertTrue(thread.daemon)
        pool.shutdown()

    def test_shutdown_with_a_blocked_worker_returns_at_once(self):
        pool = DaemonPool(1)
        started = threading.Event()

        def blocked():
            started.set()
            self.release.wait(30)
        running = pool.submit(blocked)
        self.assertTrue(started.wait(5))
        began = time.monotonic()
        pool.shutdown(cancel_pending=True)
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertTrue(running.running())

    def test_shutdown_cancels_the_tasks_not_yet_started(self):
        pool = DaemonPool(1)
        started = threading.Event()

        def blocked():
            started.set()
            self.release.wait(30)
        pool.submit(blocked)
        self.assertTrue(started.wait(5))
        pending = [pool.submit(lambda: "never") for _ in range(2)]
        pool.shutdown(cancel_pending=True)
        self.assertTrue(all(f.cancelled() for f in pending))

    def test_submit_after_shutdown_is_refused(self):
        pool = DaemonPool(1)
        pool.shutdown()
        with self.assertRaises(RuntimeError):
            pool.submit(lambda: None)

    def test_a_blocked_worker_thread_is_a_daemon_and_does_not_hold_up_exit(self):
        pool = DaemonPool(1)
        started, seen = threading.Event(), []

        def blocked():
            seen.append(threading.current_thread())
            started.set()
            self.release.wait(30)
        pool.submit(blocked)
        self.assertTrue(started.wait(5))
        pool.shutdown(cancel_pending=True)
        self.assertTrue(seen[0].daemon)
        self.assertTrue(seen[0].is_alive())  # still blocked, and nothing joins it

    def test_shutdown_lets_idle_workers_exit(self):
        pool = DaemonPool(2)
        threads = [pool.submit(threading.current_thread).result(timeout=5) for _ in range(2)]
        pool.shutdown()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())

if __name__ == "__main__":
    unittest.main()

"""OpenShell backend tests with a fake SDK session (no gateway needed)."""
import base64
import contextlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import grpc
from openshell import ExecResult

from deep_agents_demo import openshell_backend
from deep_agents_demo.openshell_backend import OpenShellBackend
from deep_agents_demo.sandbox import clear_sandbox, make_backend

POLICY = Path(__file__).resolve().parents[1] / "policy.yaml"


class FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


def session_returning(*results):
    session = Mock(id="sbx-1")
    session.exec.side_effect = list(results)
    return session


class ExecuteTests(unittest.TestCase):
    def test_runs_bash_in_workdir_with_default_timeout(self):
        session = session_returning(ExecResult(exit_code=0, stdout="hi\n", stderr=""))
        response = OpenShellBackend(session, workdir="/sandbox", default_timeout=120).execute("echo hi")
        self.assertEqual((response.output, response.exit_code), ("hi\n", 0))
        args, kwargs = session.exec.call_args
        self.assertEqual(args[0], ["bash", "-c", "echo hi"])
        self.assertEqual((kwargs["workdir"], kwargs["timeout_seconds"]), ("/sandbox", 120))

    def test_merges_stderr_and_honours_timeout(self):
        session = session_returning(ExecResult(exit_code=2, stdout="out", stderr="err"))
        response = OpenShellBackend(session).execute("false", timeout=5)
        self.assertEqual((response.output, response.exit_code), ("out\nerr", 2))
        self.assertEqual(session.exec.call_args.kwargs["timeout_seconds"], 5)

    def test_gateway_error_becomes_failed_command(self):
        session = Mock()
        session.exec.side_effect = FakeRpcError(grpc.StatusCode.UNAVAILABLE)
        response = OpenShellBackend(session).execute("ls")
        self.assertEqual(response.exit_code, 1)
        self.assertIn("OpenShell error", response.output)

    def test_id_comes_from_session(self):
        self.assertEqual(OpenShellBackend(Mock(id="sbx-1")).id, "sbx-1")


class FileTransferTests(unittest.TestCase):
    def test_upload_streams_content_through_stdin(self):
        session = session_returning(ExecResult(exit_code=0, stdout="", stderr=""))
        [response] = OpenShellBackend(session).upload_files([("/sandbox/a b/x.py", b"print(1)")])
        self.assertIsNone(response.error)
        command = session.exec.call_args.args[0][2]
        self.assertIn("mkdir -p -- '/sandbox/a b'", command)
        self.assertIn("cat > '/sandbox/a b/x.py'", command)
        self.assertEqual(session.exec.call_args.kwargs["stdin"], b"print(1)")

    def test_upload_rejects_relative_path_and_reports_failures(self):
        session = session_returning(ExecResult(exit_code=1, stdout="", stderr="denied"))
        relative, denied = OpenShellBackend(session).upload_files([("x.py", b""), ("/etc/x", b"")])
        self.assertEqual((relative.error, denied.error), ("invalid_path", "permission_denied"))

    def test_download_decodes_and_maps_errors(self):
        encoded = base64.b64encode(b"data" * 40).decode()
        wrapped = "\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76)) + "\n"
        session = session_returning(
            ExecResult(exit_code=0, stdout=wrapped, stderr=""),
            ExecResult(exit_code=openshell_backend.IS_DIRECTORY_EXIT, stdout="", stderr=""),
            ExecResult(exit_code=1, stdout="", stderr="No such file"),
        )
        ok, directory, missing = OpenShellBackend(session).download_files(["/a", "/b", "/c"])
        self.assertEqual(ok.content, b"data" * 40)
        self.assertEqual((directory.error, missing.error), ("is_directory", "file_not_found"))


class LocalBashSession:
    """Runs sandbox commands with the host's bash, standing in for a Linux sandbox."""
    id = "local"

    def exec(self, command, *, workdir=None, stdin=None, timeout_seconds=None, **_):
        import subprocess
        done = subprocess.run(command, cwd=workdir, input=stdin, capture_output=True,
                              timeout=timeout_seconds)
        return ExecResult(exit_code=done.returncode, stdout=done.stdout.decode(errors="replace"),
                          stderr=done.stderr.decode(errors="replace"))


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash") and shutil.which("python3"),
                     "needs Linux with bash and python3, like the sandbox image")
class BaseSandboxRoundTripTests(unittest.TestCase):
    """Deep Agents' file tools run as shell commands through execute()."""

    def test_write_read_edit_ls_grep(self):
        with tempfile.TemporaryDirectory() as root:
            backend = OpenShellBackend(LocalBashSession(), workdir=root)
            path = f"{root}/pkg/hello.py"
            self.assertIsNone(backend.write(path, "print('hello')\n").error)
            self.assertIn("print('hello')", str(backend.read(path).file_data))
            self.assertIsNone(backend.edit(path, "hello", "sandbox").error)
            self.assertEqual(backend.execute(f"python3 {path}").output.strip(), "sandbox")
            self.assertIn(path, str(backend.ls(f"{root}/pkg")))
            self.assertIn(path, str(backend.grep("sandbox", path=root)))
            [download] = backend.download_files([path])
            self.assertEqual(download.content, b"print('sandbox')\n")


class ClearSandboxTests(unittest.TestCase):
    def test_refuses_unsafe_workdirs(self):
        for workdir in (None, "", "/", "//", "relative/dir"):
            with self.assertRaises(ValueError):
                clear_sandbox(Mock(), workdir)

    def test_reports_counts_and_failures(self):
        backend = Mock()
        backend.execute.return_value = Mock(exit_code=0, output="5 2\n")
        self.assertEqual(clear_sandbox(backend, "/sandbox"), (5, 2))
        self.assertIn("W=/sandbox", backend.execute.call_args.args[0])
        backend.execute.return_value = Mock(exit_code=3, output="missing working directory")
        with self.assertRaisesRegex(RuntimeError, "missing working directory"):
            clear_sandbox(backend, "/sandbox")

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "needs Linux bash")
    def test_empties_workdir_including_hidden_files(self):
        with tempfile.TemporaryDirectory() as root:
            workdir = Path(root) / "sandbox"
            (workdir / ".cache/uv").mkdir(parents=True)
            (workdir / "venv/bin").mkdir(parents=True)
            (workdir / "plot.png").write_bytes(b"png")
            (workdir / ".cache/uv/wheel").write_bytes(b"x")
            backend = OpenShellBackend(LocalBashSession(), workdir=str(workdir))
            work, _tmp = clear_sandbox(backend, str(workdir))
            self.assertEqual(work, 3)
            self.assertTrue(workdir.is_dir())
            self.assertEqual(list(workdir.iterdir()), [])


class ConnectTests(unittest.TestCase):
    def test_missing_sandbox_explains_how_to_create_it(self):
        client = Mock()
        client.get.side_effect = FakeRpcError(grpc.StatusCode.NOT_FOUND)
        with patch("openshell.SandboxClient.from_active_cluster", return_value=client):
            with self.assertRaisesRegex(ValueError, "--policy policy.yaml"):
                openshell_backend.connect("demo", "default")
        client.close.assert_called_once()

    def test_connects_to_ready_sandbox_in_workspace(self):
        client = Mock()
        ref = Mock(id="sbx-1", workspace="team")
        client.wait_ready.return_value = ref
        with patch("openshell.SandboxClient.from_active_cluster", return_value=client):
            _, session = openshell_backend.connect("demo", "team")
        client.get.assert_called_once_with("demo", workspace="team")
        self.assertEqual(session.id, "sbx-1")


class ProviderSelectionTests(unittest.TestCase):
    def test_local_shell_provider_is_rejected(self):
        with patch.dict("os.environ", {"SANDBOX_PROVIDER": "local"}):
            with self.assertRaisesRegex(ValueError, "none, daytona, openshell"):
                make_backend(Path("."), Path("."), contextlib.ExitStack())

    def test_openshell_routes_skills_locally(self):
        client = Mock()
        with patch.dict("os.environ", {"SANDBOX_PROVIDER": "openshell"}), \
                patch("openshell.SandboxClient.from_active_cluster", return_value=client), \
                contextlib.ExitStack() as stack:
            backend, prefix, workdir = make_backend(Path("."), Path("."), stack)
        self.assertEqual((prefix, workdir), ("/instructions", "/sandbox"))
        self.assertIsInstance(backend.default, OpenShellBackend)
        client.close.assert_called_once()


class PolicyFileTests(unittest.TestCase):
    def test_policy_denies_by_default_and_enforces_every_endpoint(self):
        import yaml
        policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(policy["version"], 1)
        self.assertEqual(set(policy["network_policies"]), {"pypi", "github-clone"})
        for rule in policy["network_policies"].values():
            self.assertTrue(rule["binaries"])
            for endpoint in rule["endpoints"]:
                self.assertEqual(endpoint["enforcement"], "enforce")


if __name__ == "__main__":
    unittest.main()

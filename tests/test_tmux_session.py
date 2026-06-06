from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from batonloop.config import ProviderExecution, ProviderMode
from batonloop.tmux_session import (
    TmuxSessionManager,
    _contains_completion_marker,
    _TmuxProviderSession,
)


class TmuxSessionManagerTests(unittest.TestCase):
    def test_completion_marker_must_be_on_own_line(self) -> None:
        marker = "BATONLOOP_TURN_COMPLETE 1234"

        self.assertFalse(_contains_completion_marker(f"echoed prompt: {marker}", marker))
        self.assertFalse(
            _contains_completion_marker(
                "Marker prefix: BATONLOOP_TURN_COMPLETE\nMarker id: 1234\n",
                marker,
            )
        )
        self.assertTrue(_contains_completion_marker(f"\x1b[32m{marker}\x1b[0m \r\n", marker))

    def test_send_clear_and_paste_prompt_use_tmux_primitives(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            prompt_path = temp_root / "iteration-000001.prompt.txt"
            prompt_path.write_text("line 1\nline 2\n", encoding="utf-8")
            manager = CapturingTmuxSessionManager(log_dir=temp_root)
            session = _TmuxProviderSession(
                provider_name="fake",
                session_name="batonloop-fake-1",
                pane_id="%1",
                raw_log_path=temp_root / "tmux-fake.raw.log",
                attach_command="tmux attach fake",
            )

            with patch("batonloop.tmux_session.time.sleep", lambda seconds: None):
                manager._send_clear(session)
            manager._paste_prompt(session=session, prompt_path=prompt_path)

            self.assertEqual(
                manager.commands[0],
                ["send-keys", "-t", "%1", "-l", "/clear"],
            )
            self.assertEqual(manager.commands[1], ["send-keys", "-t", "%1", "Enter"])
            self.assertEqual(
                manager.commands[2][:3],
                ["load-buffer", "-b", manager.commands[2][2]],
            )
            self.assertEqual(manager.commands[2][3], str(prompt_path))
            self.assertEqual(manager.commands[3][0:4], ["paste-buffer", "-p", "-d", "-b"])
            self.assertEqual(manager.commands[3][-2:], ["-t", "%1"])
            self.assertEqual(manager.commands[4], ["send-keys", "-t", "%1", "Enter"])

    @unittest.skipUnless(shutil.which("tmux"), "tmux is not installed")
    def test_real_tmux_session_runs_fake_interactive_cli_across_turns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            cli_path = _write_fake_interactive_cli(temp_root)
            provider = FakeInteractiveProvider(cli_path)
            execution = _fake_execution()
            config = SimpleNamespace(working_dir=temp_root)
            manager = TmuxSessionManager(
                log_dir=temp_root,
                command_timeout_seconds=2.0,
            )

            try:
                first_marker = "BATONLOOP_TURN_COMPLETE first"
                first_prompt = temp_root / "iteration-000001.prompt.txt"
                first_prompt.write_text(_prompt_with_marker_parts(first_marker), encoding="utf-8")
                first_log = temp_root / "iteration-000001.log"

                first_result = manager.run_turn(
                    provider=provider,
                    execution=execution,
                    config=config,
                    prompt_path=first_prompt,
                    log_path=first_log,
                    completion_marker=first_marker,
                    timeout_seconds=5.0,
                    logger=logging.getLogger("test-tmux-session"),
                )

                self.assertEqual(first_result.exit_code, 0)
                self.assertFalse(first_result.timed_out)
                self.assertIn("WORK DONE", first_log.read_text(encoding="utf-8"))
                self.assertIn(first_marker, first_log.read_text(encoding="utf-8"))

                second_marker = "BATONLOOP_TURN_COMPLETE second"
                second_prompt = temp_root / "iteration-000002.prompt.txt"
                second_prompt.write_text(_prompt_with_marker_parts(second_marker), encoding="utf-8")
                second_log = temp_root / "iteration-000002.log"

                second_result = manager.run_turn(
                    provider=provider,
                    execution=execution,
                    config=config,
                    prompt_path=second_prompt,
                    log_path=second_log,
                    completion_marker=second_marker,
                    timeout_seconds=5.0,
                    logger=logging.getLogger("test-tmux-session"),
                )

                self.assertEqual(second_result.exit_code, 0)
                self.assertEqual(second_result.pane_id, first_result.pane_id)
                self.assertIn(
                    "CLEAR RECEIVED",
                    second_result.raw_log_path.read_text(encoding="utf-8"),
                )
            finally:
                manager.close(keep_sessions=False)

    @unittest.skipUnless(shutil.which("tmux"), "tmux is not installed")
    def test_real_tmux_session_timeout_kills_failed_session(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            cli_path = _write_never_complete_cli(temp_root)
            manager = TmuxSessionManager(
                log_dir=temp_root,
                command_timeout_seconds=2.0,
            )

            try:
                result = manager.run_turn(
                    provider=FakeInteractiveProvider(cli_path),
                    execution=_fake_execution(),
                    config=SimpleNamespace(working_dir=temp_root),
                    prompt_path=_write_prompt(temp_root, "iteration-000001.prompt.txt", "work\n"),
                    log_path=temp_root / "iteration-000001.log",
                    completion_marker="BATONLOOP_TURN_COMPLETE timeout",
                    timeout_seconds=0.25,
                    logger=logging.getLogger("test-tmux-session"),
                )

                self.assertEqual(result.exit_code, 1)
                self.assertTrue(result.timed_out)
                self.assertFalse(manager._pane_exists(result.pane_id))
            finally:
                manager.close(keep_sessions=False)


class CapturingTmuxSessionManager(TmuxSessionManager):
    def __init__(self, *, log_dir: Path) -> None:
        super().__init__(log_dir=log_dir, socket_name="batonloop-test")
        self.commands: list[list[str]] = []

    def _run_tmux(
        self,
        args: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check
        self.commands.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class FakeInteractiveProvider:
    name = "fake"

    def __init__(self, cli_path: Path) -> None:
        self._cli_path = cli_path

    def build_interactive_command(
        self,
        config: object,
        execution: ProviderExecution,
    ) -> list[str]:
        del config, execution
        return [sys.executable, str(self._cli_path)]

    def interactive_environment(
        self,
        config: object,
        execution: ProviderExecution,
    ) -> dict[str, str]:
        del config, execution
        return {"PYTHONUNBUFFERED": "1"}


def _fake_execution() -> ProviderExecution:
    return ProviderExecution(
        name="fake",
        binary=None,
        model=None,
        max_turns=None,
        use_bare=False,
        safe_mode=False,
        mode=ProviderMode.TMUX,
    )


def _prompt_with_marker_parts(marker: str) -> str:
    marker_prefix, marker_id = marker.split(" ", 1)
    return "\n".join(
        [
            "do the work",
            "=== BATONLOOP CONTROL ===",
            "When you are completely finished with this turn, end your final assistant message with one line containing the marker prefix, one ASCII space, and the marker id.",
            f"Marker prefix: {marker_prefix}",
            f"Marker id: {marker_id}",
            "Do not emit that line until all intended edits and verification for this turn are done.",
            "=== END BATONLOOP CONTROL ===",
            "",
        ]
    )


def _write_prompt(temp_root: Path, name: str, text: str) -> Path:
    path = temp_root / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_fake_interactive_cli(temp_root: Path) -> Path:
    return _write_prompt(
        temp_root,
        "fake_interactive_cli.py",
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys
            import time

            marker_prefix = None
            marker_id = None

            for raw_line in sys.stdin:
                line = raw_line.rstrip("\\n")
                if line == "/clear":
                    marker_prefix = None
                    marker_id = None
                    print("CLEAR RECEIVED", flush=True)
                    continue

                print(f"ECHO {line}", flush=True)
                if line.startswith("Marker prefix: "):
                    marker_prefix = line.split(": ", 1)[1]
                elif line.startswith("Marker id: "):
                    marker_id = line.split(": ", 1)[1]
                elif line == "=== END BATONLOOP CONTROL ===":
                    time.sleep(0.1)
                    print("WORK DONE", flush=True)
                    print(f"{marker_prefix} {marker_id}", flush=True)
            """
        ).lstrip(),
    )


def _write_never_complete_cli(temp_root: Path) -> Path:
    return _write_prompt(
        temp_root,
        "fake_never_complete_cli.py",
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys
            import time

            for raw_line in sys.stdin:
                print(f"ECHO {raw_line.rstrip()}", flush=True)
                time.sleep(10)
            """
        ).lstrip(),
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from batonloop.tmux_session import TmuxSessionManager, _TmuxProviderSession


class TmuxSessionManagerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

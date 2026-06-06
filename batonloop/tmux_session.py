from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import ProviderExecution, RunnerConfig
from .providers.base import Provider


@dataclass(frozen=True, slots=True)
class TmuxTurnResult:
    exit_code: int
    timed_out: bool
    session_name: str
    pane_id: str
    raw_log_path: Path
    attach_command: str


@dataclass(slots=True)
class _TmuxProviderSession:
    provider_name: str
    session_name: str
    pane_id: str
    raw_log_path: Path
    attach_command: str
    turn_count: int = 0


class TmuxSessionManager:
    def __init__(
        self,
        *,
        log_dir: Path,
        socket_name: str | None = None,
        tmux_binary: str = "tmux",
        command_timeout_seconds: float = 10.0,
    ) -> None:
        run_id = uuid.uuid4().hex[:8]
        self._socket_name = socket_name or f"batonloop-{os.getpid()}-{run_id}"
        self._tmux_binary = tmux_binary
        self._log_dir = log_dir
        self._command_timeout_seconds = command_timeout_seconds
        self._sessions: dict[str, _TmuxProviderSession] = {}
        self._active_session: _TmuxProviderSession | None = None

    @property
    def socket_name(self) -> str:
        return self._socket_name

    def run_turn(
        self,
        *,
        provider: Provider,
        execution: ProviderExecution,
        config: RunnerConfig,
        prompt_path: Path,
        log_path: Path,
        completion_marker: str,
        timeout_seconds: float | None,
        logger: logging.Logger,
    ) -> TmuxTurnResult:
        session = self._session_for(
            provider=provider,
            execution=execution,
            config=config,
            logger=logger,
        )
        self._active_session = session
        try:
            if session.turn_count > 0:
                self._send_clear(session)

            start_offset = _file_size(session.raw_log_path)
            self._paste_prompt(session=session, prompt_path=prompt_path)
            exit_code, timed_out = self._wait_for_marker(
                session=session,
                completion_marker=completion_marker,
                start_offset=start_offset,
                timeout_seconds=timeout_seconds,
            )
            _write_iteration_log_slice(
                raw_log_path=session.raw_log_path,
                log_path=log_path,
                start_offset=start_offset,
            )
            if exit_code == 0:
                session.turn_count += 1
            else:
                self._kill_session(session)
                self._sessions.pop(session.provider_name, None)
            return TmuxTurnResult(
                exit_code=exit_code,
                timed_out=timed_out,
                session_name=session.session_name,
                pane_id=session.pane_id,
                raw_log_path=session.raw_log_path,
                attach_command=session.attach_command,
            )
        finally:
            self._active_session = None

    def terminate(self, *, force: bool) -> None:
        session = self._active_session
        if session is None:
            return
        if force:
            self._kill_session(session)
            return
        self._run_tmux(["send-keys", "-t", session.pane_id, "C-c"], check=False)

    def close(self, *, keep_sessions: bool) -> None:
        if keep_sessions:
            return
        for session in tuple(self._sessions.values()):
            self._kill_session(session)
        self._sessions.clear()
        self._run_tmux(["kill-server"], check=False)

    def _session_for(
        self,
        *,
        provider: Provider,
        execution: ProviderExecution,
        config: RunnerConfig,
        logger: logging.Logger,
    ) -> _TmuxProviderSession:
        existing = self._sessions.get(execution.name)
        if existing is not None and self._pane_exists(existing.pane_id):
            return existing

        session = self._start_session(
            provider=provider,
            execution=execution,
            config=config,
            logger=logger,
        )
        self._sessions[execution.name] = session
        return session

    def _start_session(
        self,
        *,
        provider: Provider,
        execution: ProviderExecution,
        config: RunnerConfig,
        logger: logging.Logger,
    ) -> _TmuxProviderSession:
        session_name = f"batonloop-{execution.name}-{len(self._sessions) + 1}"
        raw_log_path = self._log_dir / f"tmux-{execution.name}.raw.log"
        raw_log_path.touch()
        command = provider.build_interactive_command(config, execution)
        environment = provider.interactive_environment(config, execution)
        shell_command = _shell_command_with_environment(command, environment)
        result = self._run_tmux(
            [
                "new-session",
                "-d",
                "-P",
                "-F",
                "#{pane_id}",
                "-s",
                session_name,
                "-c",
                str(config.working_dir),
                "-x",
                "120",
                "-y",
                "40",
                shell_command,
            ],
            check=True,
        )
        pane_id = result.stdout.strip()
        if not pane_id:
            raise RuntimeError("tmux did not return a pane id for the provider session.")

        self._run_tmux(
            [
                "pipe-pane",
                "-O",
                "-t",
                pane_id,
                f"cat >> {shlex.quote(str(raw_log_path))}",
            ],
            check=True,
        )
        attach_command = (
            f"{shlex.quote(self._tmux_binary)} -L {shlex.quote(self._socket_name)} "
            f"attach -t {shlex.quote(session_name)}"
        )
        logger.info(
            "Started tmux session for provider %s. Attach with: %s",
            execution.name,
            attach_command,
        )
        return _TmuxProviderSession(
            provider_name=execution.name,
            session_name=session_name,
            pane_id=pane_id,
            raw_log_path=raw_log_path,
            attach_command=attach_command,
        )

    def _send_clear(self, session: _TmuxProviderSession) -> None:
        self._run_tmux(["send-keys", "-t", session.pane_id, "-l", "/clear"], check=True)
        self._run_tmux(["send-keys", "-t", session.pane_id, "Enter"], check=True)
        time.sleep(0.2)

    def _paste_prompt(self, *, session: _TmuxProviderSession, prompt_path: Path) -> None:
        buffer_name = f"batonloop-prompt-{uuid.uuid4().hex}"
        self._run_tmux(
            ["load-buffer", "-b", buffer_name, str(prompt_path)],
            check=True,
        )
        self._run_tmux(
            ["paste-buffer", "-p", "-d", "-b", buffer_name, "-t", session.pane_id],
            check=True,
        )
        self._run_tmux(["send-keys", "-t", session.pane_id, "Enter"], check=True)

    def _wait_for_marker(
        self,
        *,
        session: _TmuxProviderSession,
        completion_marker: str,
        start_offset: int,
        timeout_seconds: float | None,
    ) -> tuple[int, bool]:
        started_at = time.monotonic()
        while True:
            chunk = _read_log_from_offset(session.raw_log_path, start_offset)
            if completion_marker in _strip_terminal_control_sequences(chunk):
                return 0, False

            if not self._pane_exists(session.pane_id):
                return 1, False

            if timeout_seconds is not None and time.monotonic() - started_at >= timeout_seconds:
                self._run_tmux(["send-keys", "-t", session.pane_id, "C-c"], check=False)
                return 1, True

            time.sleep(0.25)

    def _pane_exists(self, pane_id: str) -> bool:
        result = self._run_tmux(
            ["display-message", "-p", "-t", pane_id, "#{pane_id}"],
            check=False,
        )
        return result.returncode == 0

    def _kill_session(self, session: _TmuxProviderSession) -> None:
        self._run_tmux(["kill-session", "-t", session.session_name], check=False)

    def _run_tmux(
        self,
        args: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        command = [self._tmux_binary, "-L", self._socket_name, *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"tmux command timed out ({' '.join(shlex.quote(part) for part in command)})"
            ) from exc
        if check and result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"tmux command failed ({' '.join(shlex.quote(part) for part in command)}): "
                f"{stderr or result.stdout.strip() or result.returncode}"
            )
        return result


def _shell_command_with_environment(command: list[str], environment: dict[str, str]) -> str:
    if not environment:
        return shlex.join(command)
    env_args = [f"{key}={value}" for key, value in sorted(environment.items())]
    return shlex.join(["env", *env_args, *command])


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_log_from_offset(path: Path, offset: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _write_iteration_log_slice(
    *,
    raw_log_path: Path,
    log_path: Path,
    start_offset: int,
) -> None:
    text = _strip_terminal_control_sequences(_read_log_from_offset(raw_log_path, start_offset))
    log_path.write_text(text, encoding="utf-8", errors="replace")


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")


def _strip_terminal_control_sequences(text: str) -> str:
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_CSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)

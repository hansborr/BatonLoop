from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from shutil import which
from types import FrameType

from .config import RunnerConfig
from .providers.base import Provider


@dataclass(slots=True)
class RunState:
    start_time: float
    iteration: int = 0
    completed_iterations: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    consecutive_errors: int = 0


class StopController:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self.stop_requested = False
        self._current_process: subprocess.Popen[bytes] | None = None
        self._previous_int = None
        self._previous_term = None

    def install(self) -> None:
        self._previous_int = signal.getsignal(signal.SIGINT)
        self._previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def restore(self) -> None:
        if self._previous_int is not None:
            signal.signal(signal.SIGINT, self._previous_int)
        if self._previous_term is not None:
            signal.signal(signal.SIGTERM, self._previous_term)

    def attach_process(self, process: subprocess.Popen[bytes]) -> None:
        self._current_process = process

    def detach_process(self) -> None:
        self._current_process = None

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        if not self.stop_requested:
            self.stop_requested = True
            print()
            self._logger.info("STOP REQUESTED - waiting for current iteration to finish...")
            self._logger.info("Press Ctrl+C again to force quit immediately.")
            return

        print()
        self._logger.info("FORCE QUIT - terminating immediately.")
        self._terminate_current_process()
        raise SystemExit(1)

    def _terminate_current_process(self) -> None:
        if self._current_process is None or self._current_process.poll() is not None:
            return

        try:
            if hasattr(os, "killpg"):
                os.killpg(self._current_process.pid, signal.SIGTERM)
            else:
                self._current_process.terminate()
        except ProcessLookupError:
            return
        except PermissionError:
            self._current_process.terminate()


def run_loop(config: RunnerConfig, provider: Provider) -> int:
    _ensure_executable_available(provider.executable_name(config))
    config.log_dir.mkdir(parents=True, exist_ok=True)

    logger = _configure_logger(config.log_dir / "ralph.log")
    controller = StopController(logger)
    state = RunState(start_time=time.monotonic())
    exit_status = 0

    controller.install()
    try:
        _log_startup(logger, config, provider)

        if config.dry_run:
            logger.info("Dry run - exiting.")
            return 0

        while True:
            state.iteration += 1

            if config.max_iterations and state.completed_iterations >= config.max_iterations:
                logger.info("STOP: Iteration limit reached (%s)", config.max_iterations)
                break

            if controller.stop_requested:
                logger.info("STOP: Graceful shutdown requested.")
                break

            if not _check_duration(logger, config, state):
                break

            logger.info("--- Iteration %s starting ---", state.iteration)

            prompt_index = state.completed_iterations % len(config.prompt_sequence)
            current_prompt = config.prompt_sequence[prompt_index]

            if not current_prompt.is_file():
                logger.error("FATAL: Prompt file no longer exists: %s", current_prompt)
                exit_status = 1
                break

            if len(config.prompt_sequence) > 1:
                logger.info(
                    "Prompt [%s/%s]: %s",
                    prompt_index + 1,
                    len(config.prompt_sequence),
                    current_prompt.name,
                )

            iteration_log = config.log_dir / f"iteration-{state.iteration:06d}.json"
            exit_code = _run_iteration(
                provider=provider,
                config=config,
                prompt_path=current_prompt,
                log_path=iteration_log,
                controller=controller,
            )
            logger.info("Exit code: %s", exit_code)

            if exit_code == 0:
                state.consecutive_errors = 0
                state.completed_iterations += 1
                logger.info("Iteration %s completed successfully.", state.iteration)

                iteration_cost = provider.extract_cost(iteration_log, config.output_format)
                if iteration_cost != 0:
                    state.total_cost += iteration_cost
                    logger.info(
                        "Iteration cost: $%s | Total: $%s",
                        _format_decimal(iteration_cost),
                        _format_decimal(state.total_cost),
                    )

                _rotate_logs(config.log_dir, config.log_retain)

                if config.max_cost and state.total_cost >= config.max_cost:
                    logger.info(
                        "STOP: Cost limit reached ($%s >= $%s)",
                        _format_decimal(state.total_cost),
                        _format_decimal(config.max_cost),
                    )
                    break
            else:
                decision = provider.classify_failure(exit_code, iteration_log, config)
                if decision.fatal:
                    logger.error(decision.message)
                    exit_status = 1
                    break

                state.consecutive_errors += 1
                logger.warning(decision.message)

                if state.consecutive_errors >= config.max_consecutive_errors:
                    logger.error(
                        "FATAL: %s consecutive errors. Stopping.",
                        config.max_consecutive_errors,
                    )
                    exit_status = 1
                    break

                if decision.wait_seconds > 0:
                    _interruptible_sleep(decision.wait_seconds, controller)
                    if decision.reset_error_count:
                        state.consecutive_errors = 0
                    if decision.skip_pause:
                        continue

                logger.info(
                    "Consecutive errors: %s/%s. Retrying after pause...",
                    state.consecutive_errors,
                    config.max_consecutive_errors,
                )

            if controller.stop_requested:
                logger.info(
                    "STOP: Graceful shutdown requested after iteration %s.",
                    state.iteration,
                )
                break

            if config.pause_seconds > 0:
                logger.info("Pausing %ss before next iteration...", config.pause_seconds)
                _interruptible_sleep(config.pause_seconds, controller)

        return exit_status
    finally:
        controller.restore()
        _log_cleanup(logger, config, state)


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ralph")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _ensure_executable_available(command: str) -> None:
    path = Path(command).expanduser()
    if path.is_absolute() or any(sep in command for sep in ("/", "\\")):
        if not path.exists() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"'{command}' command not found or is not executable.")
        return

    if which(command) is None:
        raise FileNotFoundError(f"'{command}' command not found in PATH.")


def _log_startup(logger: logging.Logger, config: RunnerConfig, provider: Provider) -> None:
    logger.info("=== Ralph Loop Starting ===")
    logger.info("Provider:        %s", provider.name)
    logger.info("Executable:      %s", provider.executable_name(config))

    if len(config.prompt_sequence) == 1:
        logger.info("Prompt file:     %s", config.prompt_sequence[0])
    else:
        logger.info("Prompt cycle:    %s steps", len(config.prompt_sequence))
        for index, prompt_path in enumerate(config.prompt_sequence, start=1):
            logger.info("  [%s/%s] %s", index, len(config.prompt_sequence), prompt_path)

    logger.info("Max iterations:  %s", _format_limit(config.max_iterations))
    logger.info("Max cost:        %s", _format_money_limit(config.max_cost))
    logger.info("Max duration:    %s", _format_duration_limit(config.max_duration_hours))
    logger.info("Pause between:   %ss", config.pause_seconds)
    logger.info("Model:           %s", config.model or "default")
    logger.info("Rate limit wait: %sm", config.wait_on_limit_mins)
    logger.info("Max errors:      %s", config.max_consecutive_errors)
    logger.info("Max turns:       %s", config.max_turns if config.max_turns is not None else "unset")
    logger.info("Output format:   %s", config.output_format.value)
    logger.info("Bare mode:       %s", config.use_bare)
    logger.info("Safe mode:       %s", config.safe_mode)
    logger.info("Log directory:   %s", config.log_dir)
    logger.info("Log retain:      %s", _format_limit(config.log_retain))
    logger.info("===========================")


def _log_cleanup(logger: logging.Logger, config: RunnerConfig, state: RunState) -> None:
    elapsed_seconds = max(0, int(time.monotonic() - state.start_time))
    elapsed_minutes = elapsed_seconds // 60
    logger.info("=== Ralph Loop Complete ===")
    logger.info("Iterations:  %s", state.completed_iterations)
    logger.info("Total cost:  $%s", _format_decimal(state.total_cost))
    logger.info("Elapsed:     %s minutes", elapsed_minutes)
    logger.info("Logs:        %s/", config.log_dir)
    logger.info("===========================")


def _check_duration(
    logger: logging.Logger,
    config: RunnerConfig,
    state: RunState,
) -> bool:
    if config.max_duration_hours == 0:
        return True

    max_seconds = float(config.max_duration_hours * Decimal("3600"))
    elapsed_seconds = time.monotonic() - state.start_time
    if elapsed_seconds >= max_seconds:
        logger.info(
            "STOP: Duration limit reached (%sh)",
            _format_decimal(config.max_duration_hours),
        )
        return False

    return True


def _run_iteration(
    provider: Provider,
    config: RunnerConfig,
    prompt_path: Path,
    log_path: Path,
    controller: StopController,
) -> int:
    command = provider.build_command(config)
    with prompt_path.open("rb") as prompt_handle, log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=prompt_handle,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        controller.attach_process(process)
        try:
            while True:
                try:
                    return process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            controller.detach_process()


def _interruptible_sleep(seconds: int, controller: StopController) -> None:
    remaining = float(seconds)
    while remaining > 0:
        if controller.stop_requested:
            return

        chunk = min(5.0, remaining)
        try:
            time.sleep(chunk)
        except InterruptedError:
            continue
        remaining -= chunk


def _rotate_logs(log_dir: Path, retain: int) -> None:
    if retain == 0:
        return

    files = sorted(
        log_dir.glob("iteration-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files[retain:]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_limit(value: int) -> str:
    return "unlimited" if value == 0 else str(value)


def _format_money_limit(value: Decimal) -> str:
    return "unlimited" if value == 0 else f"${_format_decimal(value)}"


def _format_duration_limit(value: Decimal) -> str:
    return "unlimited" if value == 0 else f"{_format_decimal(value)}h"


from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from shutil import which
from types import FrameType

from .config import RunnerConfig
from .handoff import (
    ResumeContext,
    build_resume_prompt,
    prompt_artifact_path_for,
    resolve_resume_context,
    write_iteration_metadata,
)
from .providers.base import Provider
from .providers.utils import read_log_text


@dataclass(slots=True)
class RunState:
    start_time: float
    iteration: int = 0
    completed_iterations: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    consecutive_errors: int = 0


@dataclass(slots=True)
class CommandResult:
    exit_code: int
    timed_out: bool = False


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
        self._force_terminate_current_process()
        raise SystemExit(1)

    def _terminate_current_process(self) -> None:
        if self._current_process is None or self._current_process.poll() is not None:
            return

        _terminate_process_tree(self._current_process, force=False)

    def _force_terminate_current_process(self) -> None:
        if self._current_process is None or self._current_process.poll() is not None:
            return

        _terminate_process_tree(self._current_process, force=True)


def run_loop(config: RunnerConfig, provider: Provider) -> int:
    provider.validate_config(config)
    _validate_runtime_config(config)
    _ensure_executable_available(provider.executable_name(config))
    config.log_dir.mkdir(parents=True, exist_ok=True)
    resume_context = (
        resolve_resume_context(config.resume_from) if config.resume_from is not None else None
    )

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
            prompt_input_path = _prepare_iteration_prompt(
                config=config,
                provider=provider,
                base_prompt_path=current_prompt,
                log_path=iteration_log,
                resume_context=resume_context,
            )
            iteration_result = _run_iteration(
                provider=provider,
                config=config,
                prompt_path=prompt_input_path,
                log_path=iteration_log,
                controller=controller,
            )
            exit_code = iteration_result.exit_code
            logger.info("Exit code: %s", exit_code)
            iteration_cost = Decimal("0")
            failure_message = None
            stop_reason = None
            success = False
            wait_seconds = 0
            reset_error_count = False
            skip_pause_after_wait = False
            should_break = False

            if iteration_result.timed_out:
                state.consecutive_errors += 1
                failure_message = (
                    "Iteration "
                    f"{state.iteration} timed out after "
                    f"{_format_minutes_limit(config.iteration_timeout_minutes)} and was terminated."
                )
                logger.warning(
                    "Iteration %s timed out after %s and was terminated.",
                    state.iteration,
                    _format_minutes_limit(config.iteration_timeout_minutes),
                )

                if state.consecutive_errors >= config.max_consecutive_errors:
                    stop_reason = (
                        f"FATAL: {config.max_consecutive_errors} consecutive errors. Stopping."
                    )
                    logger.error(stop_reason)
                    exit_status = 1
                    should_break = True
            elif exit_code == 0:
                success = True
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

                stop_reason = _evaluate_post_iteration_stop(
                    logger=logger,
                    config=config,
                    controller=controller,
                    iteration=state.iteration,
                    iteration_log=iteration_log,
                )

                if stop_reason:
                    logger.info(stop_reason)
                    should_break = True
                elif config.max_cost and state.total_cost >= config.max_cost:
                    stop_reason = (
                        "STOP: Cost limit reached "
                        f"(${_format_decimal(state.total_cost)} >= ${_format_decimal(config.max_cost)})"
                    )
                    logger.info(
                        "STOP: Cost limit reached ($%s >= $%s)",
                        _format_decimal(state.total_cost),
                        _format_decimal(config.max_cost),
                    )
                    should_break = True
            else:
                decision = provider.classify_failure(exit_code, iteration_log, config)
                failure_message = decision.message
                if decision.fatal:
                    logger.error(decision.message)
                    exit_status = 1
                    stop_reason = decision.message
                    should_break = True
                else:
                    state.consecutive_errors += 1
                    logger.warning(decision.message)

                    if state.consecutive_errors >= config.max_consecutive_errors:
                        stop_reason = (
                            f"FATAL: {config.max_consecutive_errors} consecutive errors. Stopping."
                        )
                        logger.error(stop_reason)
                        exit_status = 1
                        should_break = True
                    else:
                        wait_seconds = decision.wait_seconds
                        reset_error_count = decision.reset_error_count
                        skip_pause_after_wait = decision.skip_pause

            write_iteration_metadata(
                log_path=iteration_log,
                provider_name=provider.name,
                working_dir=config.working_dir,
                log_dir=config.log_dir,
                base_prompt_path=current_prompt,
                input_prompt_path=prompt_input_path,
                output_format=config.output_format.value,
                exit_code=exit_code,
                timed_out=iteration_result.timed_out,
                success=success,
                iteration_cost=iteration_cost,
                failure_message=failure_message,
                stop_reason=stop_reason,
                resume_context=resume_context,
                resume_note=config.resume_note,
            )

            if success:
                _rotate_logs(config.log_dir, config.log_retain)

            if should_break:
                break

            if wait_seconds > 0:
                _interruptible_sleep(wait_seconds, controller)
                if reset_error_count:
                    state.consecutive_errors = 0
                if skip_pause_after_wait:
                    continue

            if not success:
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


def _validate_runtime_config(config: RunnerConfig) -> None:
    if not config.working_dir.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {config.working_dir}")

    if config.stop_on_clean_git:
        _ensure_executable_available("git")
        git_probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=config.working_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_probe.returncode != 0 or git_probe.stdout.strip() != "true":
            raise ValueError("--stop-on-clean-git requires running inside a Git worktree.")


def _log_startup(logger: logging.Logger, config: RunnerConfig, provider: Provider) -> None:
    logger.info("=== Ralph Loop Starting ===")
    logger.info("Provider:        %s", provider.name)
    logger.info("Executable:      %s", provider.executable_name(config))
    logger.info("Working dir:     %s", config.working_dir)

    if len(config.prompt_sequence) == 1:
        logger.info("Prompt file:     %s", config.prompt_sequence[0])
    else:
        logger.info("Prompt cycle:    %s steps", len(config.prompt_sequence))
        for index, prompt_path in enumerate(config.prompt_sequence, start=1):
            logger.info("  [%s/%s] %s", index, len(config.prompt_sequence), prompt_path)

    logger.info("Max iterations:  %s", _format_limit(config.max_iterations))
    logger.info("Max cost:        %s", _format_money_limit(config.max_cost))
    logger.info("Max duration:    %s", _format_duration_limit(config.max_duration_hours))
    logger.info("Iter timeout:    %s", _format_minutes_limit(config.iteration_timeout_minutes))
    logger.info("Pause between:   %ss", config.pause_seconds)
    logger.info("Model:           %s", config.model or "default")
    logger.info("Rate limit wait: %sm", config.wait_on_limit_mins)
    logger.info("Max errors:      %s", config.max_consecutive_errors)
    logger.info("Max turns:       %s", config.max_turns if config.max_turns is not None else "unset")
    logger.info("Output format:   %s", config.output_format.value)
    logger.info("Bare mode:       %s", config.use_bare)
    logger.info("Safe mode:       %s", config.safe_mode)
    logger.info("Resume from:     %s", config.resume_from if config.resume_from else "none")
    logger.info("Resume note:     %s", config.resume_note or "none")
    logger.info("Log directory:   %s", config.log_dir)
    logger.info("Log retain:      %s", _format_limit(config.log_retain))
    logger.info(
        "Checks:          %s",
        f"{len(config.check_commands)} command(s)" if config.check_commands else "none",
    )
    for index, command in enumerate(config.check_commands, start=1):
        logger.info("  [check %s/%s] %s", index, len(config.check_commands), command)
    logger.info(
        "Stop regexes:    %s",
        f"{len(config.stop_on_regexes)} pattern(s)" if config.stop_on_regexes else "none",
    )
    for index, pattern in enumerate(config.stop_on_regexes, start=1):
        logger.info("  [regex %s/%s] %s", index, len(config.stop_on_regexes), pattern)
    logger.info("Stop clean git:  %s", config.stop_on_clean_git)
    logger.info(
        "Stop files:      %s",
        f"{len(config.stop_when_files)} path(s)" if config.stop_when_files else "none",
    )
    for index, path in enumerate(config.stop_when_files, start=1):
        logger.info("  [file %s/%s] %s", index, len(config.stop_when_files), path)
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
) -> CommandResult:
    command = provider.build_command(config)
    return _run_subprocess(
        command=command,
        cwd=config.working_dir,
        controller=controller,
        log_path=log_path,
        stdin_path=prompt_path,
        timeout_seconds=_timeout_seconds(config),
    )


def _prepare_iteration_prompt(
    *,
    config: RunnerConfig,
    provider: Provider,
    base_prompt_path: Path,
    log_path: Path,
    resume_context: ResumeContext | None,
) -> Path:
    if resume_context is None:
        return base_prompt_path

    prompt_text = build_resume_prompt(
        base_prompt_path=base_prompt_path,
        current_provider_name=provider.name,
        working_dir=config.working_dir,
        log_dir=config.log_dir,
        resume_context=resume_context,
        resume_note=config.resume_note,
    )
    prompt_artifact_path = prompt_artifact_path_for(log_path)
    prompt_artifact_path.write_text(prompt_text, encoding="utf-8")
    return prompt_artifact_path


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

    grouped_files: dict[int, list[Path]] = {}
    for path in log_dir.iterdir():
        match = _ITERATION_ARTIFACT_PATTERN.match(path.name)
        if match is None:
            continue
        iteration_number = int(match.group(1))
        grouped_files.setdefault(iteration_number, []).append(path)

    iteration_numbers = sorted(grouped_files, reverse=True)
    for iteration_number in iteration_numbers[retain:]:
        for path in grouped_files[iteration_number]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def _evaluate_post_iteration_stop(
    *,
    logger: logging.Logger,
    config: RunnerConfig,
    controller: StopController,
    iteration: int,
    iteration_log: Path,
) -> str | None:
    regex_match = _match_stop_regex(iteration_log, config.stop_on_regexes)
    if regex_match is not None:
        return f"STOP: Stop regex matched iteration output: {regex_match!r}"

    stop_file = _find_stop_file(config.stop_when_files)
    if stop_file is not None:
        return f"STOP: Stop file detected: {stop_file}"

    if config.stop_on_clean_git and _git_worktree_is_clean(config):
        return "STOP: Git worktree is clean."

    if config.check_commands and _run_post_iteration_checks(
        logger=logger,
        config=config,
        controller=controller,
        iteration=iteration,
    ):
        return "STOP: All post-iteration checks passed."

    return None


def _run_post_iteration_checks(
    *,
    logger: logging.Logger,
    config: RunnerConfig,
    controller: StopController,
    iteration: int,
) -> bool:
    all_checks_passed = True
    for index, command in enumerate(config.check_commands, start=1):
        check_log = config.log_dir / f"iteration-{iteration:06d}-check-{index:02d}.log"
        logger.info("Running check [%s/%s]: %s", index, len(config.check_commands), command)
        result = _run_subprocess(
            command=command,
            cwd=config.working_dir,
            controller=controller,
            log_path=check_log,
            shell=True,
            timeout_seconds=_timeout_seconds(config),
        )

        if result.timed_out:
            all_checks_passed = False
            logger.warning(
                "Check [%s/%s] timed out after %s. Log: %s",
                index,
                len(config.check_commands),
                _format_minutes_limit(config.iteration_timeout_minutes),
                check_log,
            )
            continue

        if result.exit_code == 0:
            logger.info(
                "Check [%s/%s] passed. Log: %s",
                index,
                len(config.check_commands),
                check_log,
            )
            continue

        all_checks_passed = False
        logger.warning(
            "Check [%s/%s] failed with exit code %s. Log: %s",
            index,
            len(config.check_commands),
            result.exit_code,
            check_log,
        )

    return all_checks_passed


def _run_subprocess(
    *,
    command: list[str] | str,
    cwd: Path,
    controller: StopController,
    log_path: Path,
    stdin_path: Path | None = None,
    shell: bool = False,
    timeout_seconds: float | None = None,
) -> CommandResult:
    popen_kwargs: dict[str, object] = {
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "start_new_session": True,
        "cwd": cwd,
        "shell": shell,
    }
    if shell:
        popen_kwargs["executable"] = os.environ.get("SHELL") or "/bin/sh"

    stdin_handle = None
    try:
        if stdin_path is not None:
            stdin_handle = stdin_path.open("rb")
            popen_kwargs["stdin"] = stdin_handle

        with log_path.open("wb") as log_handle:
            popen_kwargs["stdout"] = log_handle
            process = subprocess.Popen(command, **popen_kwargs)
            controller.attach_process(process)
            try:
                started_at = time.monotonic()
                while True:
                    wait_timeout = 1.0
                    if timeout_seconds is not None:
                        remaining = timeout_seconds - (time.monotonic() - started_at)
                        if remaining <= 0:
                            _terminate_process_tree(process, force=False)
                            return CommandResult(exit_code=process.wait(), timed_out=True)
                        wait_timeout = min(wait_timeout, remaining)

                    try:
                        return CommandResult(exit_code=process.wait(timeout=wait_timeout))
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                controller.detach_process()
    finally:
        if stdin_handle is not None:
            stdin_handle.close()


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    force: bool,
    kill_after_seconds: float = 5.0,
) -> None:
    if process.poll() is not None:
        return

    _signal_process_tree(process, signal.SIGKILL if force else signal.SIGTERM)

    if force:
        return

    deadline = time.monotonic() + kill_after_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if process.poll() is None:
        _signal_process_tree(process, signal.SIGKILL)


def _signal_process_tree(process: subprocess.Popen[bytes], signum: signal.Signals) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signum)
        else:
            process.send_signal(signum)
    except ProcessLookupError:
        return
    except PermissionError:
        if signum is signal.SIGKILL:
            process.kill()
        elif signum is signal.SIGTERM:
            process.terminate()
        else:
            process.send_signal(signum)


def _timeout_seconds(config: RunnerConfig) -> float | None:
    if config.iteration_timeout_minutes == 0:
        return None
    return float(config.iteration_timeout_minutes * Decimal("60"))


def _match_stop_regex(log_path: Path, patterns: tuple[str, ...]) -> str | None:
    if not patterns:
        return None

    log_text = read_log_text(log_path)
    for pattern in patterns:
        if re.search(pattern, log_text, re.MULTILINE):
            return pattern
    return None


def _find_stop_file(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _git_worktree_is_clean(config: RunnerConfig) -> bool:
    repo_root = _git_toplevel(config.working_dir)
    status_command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        ".",
    ]

    try:
        relative_log_dir = config.log_dir.relative_to(repo_root)
    except ValueError:
        relative_log_dir = None

    if relative_log_dir is not None:
        status_command.append(f":(exclude){relative_log_dir.as_posix()}")

    status_result = subprocess.run(
        status_command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        raise RuntimeError(f"Unable to determine Git status for {repo_root}.")
    return status_result.stdout.strip() == ""


def _git_toplevel(working_dir: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("--stop-on-clean-git requires running inside a Git worktree.")
    return Path(result.stdout.strip())


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


def _format_minutes_limit(value: Decimal) -> str:
    return "unlimited" if value == 0 else f"{_format_decimal(value)}m"


_ITERATION_ARTIFACT_PATTERN = re.compile(r"^iteration-(\d{6})(?:$|[.-])")

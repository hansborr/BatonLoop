from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from shutil import which
from types import FrameType

from .config import ProviderExecution, RunnerConfig, resolve_provider_execution
from .handoff import (
    ResumeContext,
    build_resume_prompt,
    prompt_artifact_path_for,
    resolve_resume_context,
    write_iteration_metadata,
)
from .live_output import LiveOutputConsumer
from .providers.base import Provider
from .providers.utils import read_log_text


@dataclass(slots=True)
class RunState:
    start_time: float
    attempted_iterations: int = 0
    completed_iterations: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    consecutive_errors: int = 0


@dataclass(slots=True)
class CommandResult:
    exit_code: int
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ProviderSlot:
    provider: Provider
    execution: ProviderExecution


@dataclass(frozen=True, slots=True)
class IterationExecution:
    number: int
    slot: ProviderSlot
    prompt_path: Path
    prompt_input_path: Path
    log_path: Path
    result: CommandResult


@dataclass(frozen=True, slots=True)
class IterationOutcome:
    exit_code: int
    success: bool
    iteration_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    failure_message: str | None = None
    stop_reason: str | None = None
    exit_status: int = 0
    should_break: bool = False
    wait_seconds: int = 0
    reset_error_count: bool = False
    skip_pause_after_wait: bool = False
    failover_target_provider: str | None = None
    next_provider_index: int | None = None


class _OutputPump(threading.Thread):
    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        log_handle: object,
        consumer: LiveOutputConsumer,
    ) -> None:
        super().__init__(daemon=True)
        self._process = process
        self._log_handle = log_handle
        self._consumer = consumer
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            if self._process.stdout is None:
                return

            with self._process.stdout:
                while True:
                    chunk = self._process.stdout.readline()
                    if not chunk:
                        break
                    self._log_handle.write(chunk)
                    self._log_handle.flush()
                    self._consumer.consume_line(chunk.decode("utf-8", errors="replace"))
        except BaseException as exc:  # pragma: no cover - defensive thread handoff
            self.error = exc


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


def run_loop(config: RunnerConfig, providers: Mapping[str, Provider]) -> int:
    _validate_runtime_config(config)
    provider_slots = _build_provider_slots(config, providers)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    resume_context = (
        resolve_resume_context(config.resume_from) if config.resume_from is not None else None
    )

    logger = _configure_logger(config.log_dir / "batonloop.log")
    controller = StopController(logger)
    state = RunState(start_time=time.monotonic())
    exit_status = 0
    current_provider_index = 0

    controller.install()
    try:
        _log_startup(logger, config, provider_slots)

        if config.dry_run:
            logger.info("Dry run - exiting.")
            return 0

        while True:
            if (
                config.max_iterations
                and state.attempted_iterations >= config.max_iterations
            ):
                logger.info("STOP: Iteration limit reached (%s)", config.max_iterations)
                break

            if controller.stop_requested:
                logger.info("STOP: Graceful shutdown requested.")
                break

            if not _check_duration(logger, config, state):
                break

            state.attempted_iterations += 1
            iteration_number = state.attempted_iterations
            current_slot = provider_slots[current_provider_index]
            prompt_index = state.completed_iterations % len(config.prompt_sequence)
            current_prompt = config.prompt_sequence[prompt_index]

            if not current_prompt.is_file():
                logger.error("FATAL: Prompt file no longer exists: %s", current_prompt)
                exit_status = 1
                break

            iteration = _execute_iteration(
                logger=logger,
                config=config,
                controller=controller,
                slot=current_slot,
                iteration_number=iteration_number,
                prompt_index=prompt_index,
                prompt_path=current_prompt,
                resume_context=resume_context,
            )
            outcome = _handle_iteration_outcome(
                logger=logger,
                config=config,
                state=state,
                controller=controller,
                iteration=iteration,
                provider_slots=provider_slots,
                current_provider_index=current_provider_index,
            )
            exit_status = max(exit_status, outcome.exit_status)

            write_iteration_metadata(
                log_path=iteration.log_path,
                execution=iteration.slot.execution,
                working_dir=config.working_dir,
                log_dir=config.log_dir,
                base_prompt_path=iteration.prompt_path,
                input_prompt_path=iteration.prompt_input_path,
                output_format=config.output_format.value,
                exit_code=outcome.exit_code,
                timed_out=iteration.result.timed_out,
                success=outcome.success,
                iteration_cost=outcome.iteration_cost,
                failure_message=outcome.failure_message,
                stop_reason=outcome.stop_reason,
                failover_target_provider=outcome.failover_target_provider,
                resume_context=resume_context,
                resume_note=config.resume_note,
            )

            if outcome.success:
                _rotate_logs(config.log_dir, config.log_retain)

            if outcome.next_provider_index is not None:
                current_provider_index = outcome.next_provider_index
                state.consecutive_errors = 0
                resume_context = resolve_resume_context(iteration.log_path)
                continue

            if outcome.should_break:
                break

            if outcome.wait_seconds > 0:
                _interruptible_sleep(outcome.wait_seconds, controller)
                if outcome.reset_error_count:
                    state.consecutive_errors = 0
                if outcome.skip_pause_after_wait:
                    continue

            if not outcome.success:
                logger.info(
                    "Consecutive errors: %s/%s. Retrying after pause...",
                    state.consecutive_errors,
                    config.max_consecutive_errors,
                )

            if controller.stop_requested:
                logger.info(
                    "STOP: Graceful shutdown requested after iteration %s.",
                    iteration_number,
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
    logger = logging.getLogger("batonloop")
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


def _build_provider_slots(
    config: RunnerConfig,
    providers: Mapping[str, Provider],
) -> tuple[ProviderSlot, ...]:
    slots: list[ProviderSlot] = []
    for provider_name in config.provider_names:
        provider = providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unknown provider configured for this run: {provider_name}")
        execution = resolve_provider_execution(config, provider_name)
        provider.validate_config(config, execution)
        _ensure_executable_available(provider.executable_name(execution))
        slots.append(ProviderSlot(provider=provider, execution=execution))
    return tuple(slots)


def _log_startup(
    logger: logging.Logger,
    config: RunnerConfig,
    provider_slots: tuple[ProviderSlot, ...],
) -> None:
    logger.info("=== BatonLoop Starting ===")
    logger.info(
        "Providers:       %s",
        " -> ".join(slot.execution.name for slot in provider_slots),
    )
    logger.info(
        "Provider config: %s",
        config.provider_config_path if config.provider_config_path else "none",
    )
    for index, slot in enumerate(provider_slots, start=1):
        logger.info(
            "  [provider %s/%s] name=%s binary=%s model=%s max_turns=%s bare=%s safe=%s",
            index,
            len(provider_slots),
            slot.execution.name,
            slot.provider.executable_name(slot.execution),
            slot.execution.model or "default",
            slot.execution.max_turns if slot.execution.max_turns is not None else "unset",
            slot.execution.use_bare,
            slot.execution.safe_mode,
        )
    logger.info("Working dir:     %s", config.working_dir)

    if len(config.prompt_sequence) == 1:
        logger.info("Prompt file:     %s", config.prompt_sequence[0])
    else:
        logger.info("Prompt cycle:    %s steps", len(config.prompt_sequence))
        for index, prompt_path in enumerate(config.prompt_sequence, start=1):
            logger.info("  [%s/%s] %s", index, len(config.prompt_sequence), prompt_path)

    logger.info("Max attempts:    %s", _format_limit(config.max_iterations))
    logger.info("Max cost:        %s", _format_money_limit(config.max_cost))
    logger.info("Max duration:    %s", _format_duration_limit(config.max_duration_hours))
    logger.info("Iter timeout:    %s", _format_minutes_limit(config.iteration_timeout_minutes))
    logger.info("Pause between:   %ss", config.pause_seconds)
    logger.info("Rate limit wait: %sm", config.wait_on_limit_mins)
    logger.info("Max errors:      %s", config.max_consecutive_errors)
    logger.info("Output format:   %s", config.output_format.value)
    logger.info("Live output:     %s", config.live_output)
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
    logger.info("=== BatonLoop Complete ===")
    logger.info("Attempts:    %s", state.attempted_iterations)
    logger.info("Completed:   %s", state.completed_iterations)
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


def _execute_iteration(
    *,
    logger: logging.Logger,
    config: RunnerConfig,
    controller: StopController,
    slot: ProviderSlot,
    iteration_number: int,
    prompt_index: int,
    prompt_path: Path,
    resume_context: ResumeContext | None,
) -> IterationExecution:
    logger.info(
        "--- Iteration %s starting with provider %s ---",
        iteration_number,
        slot.execution.name,
    )
    if len(config.prompt_sequence) > 1:
        logger.info(
            "Prompt [%s/%s]: %s",
            prompt_index + 1,
            len(config.prompt_sequence),
            prompt_path.name,
        )

    iteration_log = config.log_dir / f"iteration-{iteration_number:06d}.json"
    prompt_input_path = _prepare_iteration_prompt(
        config=config,
        provider_execution=slot.execution,
        base_prompt_path=prompt_path,
        log_path=iteration_log,
        resume_context=resume_context,
    )
    result = _run_iteration(
        provider=slot.provider,
        execution=slot.execution,
        config=config,
        prompt_path=prompt_input_path,
        log_path=iteration_log,
        controller=controller,
    )
    logger.info("Exit code: %s", result.exit_code)
    return IterationExecution(
        number=iteration_number,
        slot=slot,
        prompt_path=prompt_path,
        prompt_input_path=prompt_input_path,
        log_path=iteration_log,
        result=result,
    )


def _handle_iteration_outcome(
    *,
    logger: logging.Logger,
    config: RunnerConfig,
    state: RunState,
    controller: StopController,
    iteration: IterationExecution,
    provider_slots: tuple[ProviderSlot, ...],
    current_provider_index: int,
) -> IterationOutcome:
    exit_code = iteration.result.exit_code
    iteration_cost = Decimal("0")
    failure_message = None
    stop_reason = None
    success = False
    wait_seconds = 0
    reset_error_count = False
    skip_pause_after_wait = False
    should_break = False
    exit_status = 0
    failover_target_provider = None
    next_provider_index = None

    if iteration.result.timed_out:
        state.consecutive_errors += 1
        failure_message = (
            "Iteration "
            f"{iteration.number} timed out after "
            f"{_format_minutes_limit(config.iteration_timeout_minutes)} and was terminated."
        )
        logger.warning(
            "Iteration %s timed out after %s and was terminated.",
            iteration.number,
            _format_minutes_limit(config.iteration_timeout_minutes),
        )

        if state.consecutive_errors >= config.max_consecutive_errors:
            stop_reason = f"FATAL: {config.max_consecutive_errors} consecutive errors. Stopping."
            logger.error(stop_reason)
            exit_status = 1
            should_break = True
    elif exit_code == 0:
        success = True
        state.consecutive_errors = 0
        state.completed_iterations += 1
        logger.info("Iteration %s completed successfully.", iteration.number)

        iteration_cost = iteration.slot.provider.extract_cost(
            iteration.log_path,
            config.output_format,
        )
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
            iteration=iteration.number,
            iteration_log=iteration.log_path,
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
        decision = iteration.slot.provider.classify_failure(
            exit_code,
            iteration.log_path,
            config,
            iteration.slot.execution,
        )
        candidate_provider_index = current_provider_index + 1
        has_remaining_attempt_budget = (
            config.max_iterations == 0
            or state.attempted_iterations < config.max_iterations
        )
        can_failover = (
            decision.should_failover
            and candidate_provider_index < len(provider_slots)
            and has_remaining_attempt_budget
        )

        if can_failover:
            failure_message = decision.message
            failover_target_provider = provider_slots[candidate_provider_index].execution.name
            logger.warning(decision.message)
            logger.info(
                "AUTO FAILOVER: Switching provider from %s to %s.",
                iteration.slot.execution.name,
                failover_target_provider,
            )
            next_provider_index = candidate_provider_index
        elif decision.fatal:
            failure_message = decision.message
            logger.error(decision.message)
            exit_status = 1
            stop_reason = decision.message
            should_break = True
        else:
            state.consecutive_errors += 1
            if state.consecutive_errors >= config.max_consecutive_errors:
                failure_message = decision.message
                logger.warning(failure_message)
                stop_reason = f"FATAL: {config.max_consecutive_errors} consecutive errors. Stopping."
                logger.error(stop_reason)
                exit_status = 1
                should_break = True
            elif has_remaining_attempt_budget:
                failure_message = decision.message
                logger.warning(failure_message)
                wait_seconds = decision.wait_seconds
                reset_error_count = decision.reset_error_count
                skip_pause_after_wait = decision.skip_pause
            else:
                failure_message = (
                    f"{decision.message} Iteration limit reached before another retry."
                )
                logger.warning(failure_message)

    if (
        not should_break
        and config.max_iterations
        and state.attempted_iterations >= config.max_iterations
    ):
        stop_reason = f"STOP: Iteration limit reached ({config.max_iterations})"
        logger.info(stop_reason)
        should_break = True

    return IterationOutcome(
        exit_code=exit_code,
        success=success,
        iteration_cost=iteration_cost,
        failure_message=failure_message,
        stop_reason=stop_reason,
        exit_status=exit_status,
        should_break=should_break,
        wait_seconds=wait_seconds,
        reset_error_count=reset_error_count,
        skip_pause_after_wait=skip_pause_after_wait,
        failover_target_provider=failover_target_provider,
        next_provider_index=next_provider_index,
    )


def _run_iteration(
    provider: Provider,
    execution: ProviderExecution,
    config: RunnerConfig,
    prompt_path: Path,
    log_path: Path,
    controller: StopController,
) -> CommandResult:
    command = provider.build_command(config, execution)
    return _run_subprocess(
        command=command,
        cwd=config.working_dir,
        controller=controller,
        log_path=log_path,
        stdin_path=prompt_path,
        timeout_seconds=_timeout_seconds(config),
        live_output_consumer=(
            LiveOutputConsumer(logging.getLogger("batonloop"), execution.name)
            if config.live_output
            else None
        ),
    )


def _prepare_iteration_prompt(
    *,
    config: RunnerConfig,
    provider_execution: ProviderExecution,
    base_prompt_path: Path,
    log_path: Path,
    resume_context: ResumeContext | None,
) -> Path:
    if resume_context is None:
        return base_prompt_path

    prompt_text = build_resume_prompt(
        base_prompt_path=base_prompt_path,
        current_provider_name=provider_execution.name,
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
    live_output_consumer: LiveOutputConsumer | None = None,
) -> CommandResult:
    popen_kwargs: dict[str, object] = {
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
            if live_output_consumer is None:
                popen_kwargs["stdout"] = log_handle
            else:
                popen_kwargs["stdout"] = subprocess.PIPE
            process = subprocess.Popen(command, **popen_kwargs)
            output_pump = None
            if live_output_consumer is not None:
                output_pump = _OutputPump(
                    process=process,
                    log_handle=log_handle,
                    consumer=live_output_consumer,
                )
                output_pump.start()
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
                if output_pump is not None:
                    output_pump.join()
                    if output_pump.error is not None:
                        raise output_pump.error
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

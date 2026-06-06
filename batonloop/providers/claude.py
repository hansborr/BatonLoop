from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from ..config import OutputFormat, ProviderExecution, ProviderMode, RunnerConfig
from .base import FailureDecision, FailureKind
from .utils import (
    decimal_from_value,
    iter_jsonl,
    read_jsonl_failure_summary,
    read_log_text,
    summarize_failure_payloads,
)


class ClaudeProvider:
    name = "claude"
    default_binary = "claude"

    def executable_name(self, execution: ProviderExecution) -> str:
        return execution.binary or self.default_binary

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        if execution.mode is ProviderMode.TMUX:
            self.validate_interactive_config(config, execution)

    def validate_interactive_config(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> None:
        del config
        if execution.max_turns is not None:
            raise ValueError("Provider 'claude' does not support --max-turns in tmux mode.")
        _reject_interactive_args(
            provider_name=self.name,
            args=execution.extra_args,
            unsupported={
                "-p",
                "--print",
                "--output-format",
                "--input-format",
                "--max-budget-usd",
                "--max-turns",
                "--no-session-persistence",
            },
        )

    def build_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        command = [
            self.executable_name(execution),
            "-p",
            "--output-format",
            config.output_format.value,
        ]

        if config.output_format is OutputFormat.STREAM_JSON:
            command.append("--verbose")

        if not execution.safe_mode:
            command.append("--dangerously-skip-permissions")

        if execution.use_bare:
            command.append("--bare")

        if execution.model:
            command.extend(["--model", execution.model])

        if execution.max_turns is not None:
            command.extend(["--max-turns", str(execution.max_turns)])

        command.extend(execution.extra_args)

        return command

    def build_interactive_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        del config
        command = [self.executable_name(execution)]

        if not execution.safe_mode:
            command.append("--dangerously-skip-permissions")

        if execution.use_bare:
            command.append("--bare")

        if execution.model:
            command.extend(["--model", execution.model])

        command.extend(execution.extra_args)
        return command

    def interactive_environment(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> dict[str, str]:
        del config, execution
        return {
            "CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN": "1",
            "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION": "false",
        }

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        if not log_path.is_file():
            return Decimal("0")

        if output_format is OutputFormat.JSON:
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return Decimal("0")
            return decimal_from_value(payload.get("total_cost_usd", 0))

        result_cost = Decimal("0")
        for payload in iter_jsonl(log_path):
            if payload.get("type") == "result":
                result_cost = decimal_from_value(payload.get("total_cost_usd", 0))

        return result_cost

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        del execution
        if exit_code == 2:
            return FailureDecision(
                kind=FailureKind.AUTH,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Authentication error (exit code 2). "
                    "Run 'claude auth login' or check ANTHROPIC_API_KEY."
                ),
            )

        summary = _read_failure_summary(log_path, config.output_format)

        if exit_code == 1:
            if (
                summary.rate_limit_rejected
                or summary.has_status(429)
                or summary.has_error_code("rate_limit", "rate.limit", "usage.limit")
                or summary.matches_text(_RATE_LIMIT_PATTERNS)
            ):
                return FailureDecision(
                    kind=FailureKind.RATE_LIMIT,
                    message=(
                        "RATE LIMITED detected in output. "
                        f"Waiting {config.wait_on_limit_mins} minutes before retrying..."
                    ),
                    wait_seconds=config.wait_on_limit_mins * 60,
                    reset_error_count=True,
                    skip_pause=True,
                    should_failover=True,
                )

            if (
                summary.has_status(500, 502, 503, 504, 529)
                or summary.has_error_code("overloaded", "server_error")
                or summary.matches_text(_OVERLOAD_PATTERNS)
            ):
                return FailureDecision(
                    kind=FailureKind.OVERLOADED,
                    message="API overloaded (529). Waiting 2 minutes before retrying...",
                    wait_seconds=120,
                    skip_pause=True,
                )

            log_text = read_log_text(log_path, lower=True)

            if any(pattern in log_text for pattern in _RATE_LIMIT_PATTERNS):
                return FailureDecision(
                    kind=FailureKind.RATE_LIMIT,
                    message=(
                        "RATE LIMITED detected in output. "
                        f"Waiting {config.wait_on_limit_mins} minutes before retrying..."
                    ),
                    wait_seconds=config.wait_on_limit_mins * 60,
                    reset_error_count=True,
                    skip_pause=True,
                    should_failover=True,
                )

            if any(pattern in log_text for pattern in _OVERLOAD_PATTERNS):
                return FailureDecision(
                    kind=FailureKind.OVERLOADED,
                    message="API overloaded (529). Waiting 2 minutes before retrying...",
                    wait_seconds=120,
                    skip_pause=True,
                )

            return FailureDecision(
                message=f"ERROR: Generic error (exit code 1). Check {log_path} for details."
            )

        return FailureDecision(
            message=f"WARNING: Unexpected exit code {exit_code}. Check {log_path} for details."
        )


_RATE_LIMIT_PATTERNS = (
    "rate.limit",
    "usage.limit",
    "rate_limit",
    "usage limit",
    "hit your limit",
)

_OVERLOAD_PATTERNS = (
    "overloaded",
    "temporarily unavailable",
)


def _read_failure_summary(log_path: Path, output_format: OutputFormat):
    if output_format is OutputFormat.JSON:
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return summarize_failure_payloads(())
        if isinstance(payload, dict):
            return summarize_failure_payloads((payload,))
        return summarize_failure_payloads(())

    return read_jsonl_failure_summary(log_path)


def _reject_interactive_args(
    *,
    provider_name: str,
    args: tuple[str, ...],
    unsupported: set[str],
) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in unsupported:
            raise ValueError(
                f"Provider '{provider_name}' argument {arg!r} is not supported in tmux mode."
            )

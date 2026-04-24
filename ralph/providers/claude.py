from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from ..config import OutputFormat, RunnerConfig
from .base import FailureDecision
from .utils import decimal_from_value, iter_jsonl, read_log_text


class ClaudeProvider:
    name = "claude"
    default_binary = "claude"

    def executable_name(self, config: RunnerConfig) -> str:
        return config.provider_binary or self.default_binary

    def validate_config(self, config: RunnerConfig) -> None:
        del config

    def build_command(self, config: RunnerConfig) -> list[str]:
        command = [
            self.executable_name(config),
            "-p",
            "--output-format",
            config.output_format.value,
        ]

        if config.output_format is OutputFormat.STREAM_JSON:
            command.append("--verbose")

        if not config.safe_mode:
            command.append("--dangerously-skip-permissions")

        if config.use_bare:
            command.append("--bare")

        if config.model:
            command.extend(["--model", config.model])

        if config.max_turns is not None:
            command.extend(["--max-turns", str(config.max_turns)])

        return command

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
    ) -> FailureDecision:
        if exit_code == 2:
            return FailureDecision(
                fatal=True,
                message=(
                    "FATAL: Authentication error (exit code 2). "
                    "Run 'claude auth login' or check ANTHROPIC_API_KEY."
                ),
            )

        log_text = read_log_text(log_path, lower=True)

        if exit_code == 1:
            if any(pattern in log_text for pattern in ("rate.limit", "usage.limit", "rate_limit")):
                return FailureDecision(
                    message=(
                        "RATE LIMITED detected in output. "
                        f"Waiting {config.wait_on_limit_mins} minutes before retrying..."
                    ),
                    wait_seconds=config.wait_on_limit_mins * 60,
                    reset_error_count=True,
                    skip_pause=True,
                )

            if any(pattern in log_text for pattern in ("overloaded", "529", "temporarily unavailable")):
                return FailureDecision(
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

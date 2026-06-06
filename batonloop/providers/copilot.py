from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from ..config import OutputFormat, ProviderExecution, ProviderMode, RunnerConfig
from .base import FailureDecision, FailureKind
from .utils import decimal_from_value, iter_jsonl, read_jsonl_failure_summary, read_log_text


class CopilotProvider:
    name = "copilot"
    default_binary = "copilot"

    def executable_name(self, execution: ProviderExecution) -> str:
        return execution.binary or self.default_binary

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        if execution.mode is ProviderMode.TMUX:
            self.validate_interactive_config(config, execution)
            return
        if config.output_format is not OutputFormat.STREAM_JSON:
            raise ValueError(
                "Provider 'copilot' only supports BatonLoop stream-json mode, "
                "which maps to 'copilot --output-format json'."
            )

    def validate_interactive_config(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> None:
        del config, execution
        raise ValueError("Provider 'copilot' does not support tmux mode.")

    def build_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        del config
        command = [
            self.executable_name(execution),
            "--output-format",
            "json",
            "--autopilot",
            "--no-ask-user",
        ]

        if execution.safe_mode:
            command.append("--allow-all-tools")
        else:
            command.append("--allow-all")

        if execution.use_bare:
            command.extend(
                [
                    "--no-custom-instructions",
                    "--disable-builtin-mcps",
                ]
            )

        if execution.model:
            command.extend(["--model", execution.model])

        if execution.max_turns is not None:
            command.extend(["--max-autopilot-continues", str(execution.max_turns)])

        command.extend(execution.extra_args)

        return command

    def build_interactive_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        del config, execution
        raise ValueError("Provider 'copilot' does not support tmux mode.")

    def interactive_environment(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> dict[str, str]:
        del config, execution
        raise ValueError("Provider 'copilot' does not support tmux mode.")

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        del output_format
        result_cost = Decimal("0")

        for payload in iter_jsonl(log_path):
            for path in (
                ("total_cost_usd",),
                ("cost_usd",),
                ("usage", "total_cost_usd"),
                ("usage", "cost_usd"),
            ):
                value = _nested_lookup(payload, *path)
                if value is None:
                    continue
                result_cost = decimal_from_value(value)

        return result_cost

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        del execution
        summary = read_jsonl_failure_summary(log_path)

        if (
            summary.has_status(401, 403)
            or summary.has_error_code("authentication", "unauthorized", "forbidden")
            or summary.matches_text(_AUTH_PATTERNS)
        ):
            return FailureDecision(
                kind=FailureKind.AUTH,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Authentication or access error. Run 'copilot login' or set "
                    "COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN. Check "
                    f"{log_path} for details."
                ),
            )

        if (
            summary.rate_limit_rejected
            or summary.has_status(429)
            or summary.has_error_code("rate_limit", "rate.limit", "quota_exceeded")
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
            or summary.has_error_code("server_error")
            or summary.matches_text(_OVERLOAD_PATTERNS)
        ):
            return FailureDecision(
                kind=FailureKind.OVERLOADED,
                message="Copilot service overload detected. Waiting 2 minutes before retrying...",
                wait_seconds=120,
                skip_pause=True,
            )

        if exit_code == 2:
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Copilot CLI usage/configuration error (exit code 2). "
                    f"Check {log_path} for details."
                ),
            )

        if (
            summary.has_status(400)
            or summary.has_error_code("invalid_request", "invalid_request_error")
            or summary.matches_text(_INVALID_REQUEST_PATTERNS)
        ):
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Copilot rejected the request. Check the selected model and "
                    f"provider options. See {log_path} for details."
                ),
            )

        log_text = read_log_text(log_path, lower=True)

        if any(pattern in log_text for pattern in _AUTH_PATTERNS):
            return FailureDecision(
                kind=FailureKind.AUTH,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Authentication or access error. Run 'copilot login' or set "
                    "COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN. Check "
                    f"{log_path} for details."
                ),
            )

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
                message="Copilot service overload detected. Waiting 2 minutes before retrying...",
                wait_seconds=120,
                skip_pause=True,
            )

        if any(pattern in log_text for pattern in _INVALID_REQUEST_PATTERNS):
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Copilot rejected the request. Check the selected model and "
                    f"provider options. See {log_path} for details."
                ),
            )

        if exit_code == 1:
            return FailureDecision(
                message=(
                    f"ERROR: Copilot execution failed (exit code 1). Check {log_path} for details."
                )
            )

        return FailureDecision(
            message=f"WARNING: Unexpected exit code {exit_code}. Check {log_path} for details."
        )


_AUTH_PATTERNS = (
    "no authentication information found",
    "authentication failed",
    "401 unauthorized",
    "403 forbidden",
    "access denied by policy settings",
    "not logged in",
    "copilot requests permission",
    "copilot requests",
)

_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "rate limited",
    "too many requests",
    "usage limit",
    "weekly usage limit",
    "session rate limit",
    "quota exceeded",
    "premium requests",
)

_OVERLOAD_PATTERNS = (
    "temporarily unavailable",
    "server error",
    "status\\\":500",
    "status\\\":502",
    "status\\\":503",
    "status\\\":504",
    "status\\\":529",
)

_INVALID_REQUEST_PATTERNS = (
    "invalid_request",
    "invalid request",
    "unknown option",
    "unknown argument",
    "unknown flag",
    "unsupported option",
    "invalid value",
)


def _nested_lookup(payload: dict[str, object], *path: str) -> object | None:
    current: object = payload
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current

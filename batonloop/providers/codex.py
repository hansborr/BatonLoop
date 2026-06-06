from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from ..config import OutputFormat, ProviderExecution, ProviderMode, RunnerConfig
from .base import FailureDecision, FailureKind
from .utils import decimal_from_value, iter_jsonl, read_jsonl_failure_summary, read_log_text


class CodexProvider:
    name = "codex"
    default_binary = "codex"

    def executable_name(self, execution: ProviderExecution) -> str:
        return execution.binary or self.default_binary

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        if execution.mode is ProviderMode.TMUX:
            self.validate_interactive_config(config, execution)
            return
        if config.output_format is not OutputFormat.STREAM_JSON:
            raise ValueError("Provider 'codex' only supports stream-json output.")
        if execution.max_turns is not None:
            raise ValueError("Provider 'codex' does not support --max-turns.")

    def validate_interactive_config(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> None:
        del config
        if execution.max_turns is not None:
            raise ValueError("Provider 'codex' does not support --max-turns in tmux mode.")
        if execution.use_bare:
            raise ValueError(
                "Provider 'codex' does not support bare mode in tmux mode because "
                "interactive codex does not expose --ignore-user-config/--ignore-rules."
            )
        _reject_interactive_args(
            provider_name=self.name,
            args=execution.extra_args,
            unsupported={
                "exec",
                "--json",
                "--ignore-user-config",
                "--ignore-rules",
            },
        )

    def build_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        command = [
            self.executable_name(execution),
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(config.working_dir),
        ]

        if execution.safe_mode:
            command.append("--full-auto")
        else:
            command.append("--dangerously-bypass-approvals-and-sandbox")

        if execution.use_bare:
            command.extend(["--ignore-user-config", "--ignore-rules"])

        if execution.model:
            command.extend(["-m", execution.model])

        command.extend(execution.extra_args)

        return command

    def build_interactive_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        command = [
            self.executable_name(execution),
            "--no-alt-screen",
            "-C",
            str(config.working_dir),
        ]

        if execution.safe_mode:
            command.append("--full-auto")
        else:
            command.append("--dangerously-bypass-approvals-and-sandbox")

        if execution.model:
            command.extend(["-m", execution.model])

        command.extend(execution.extra_args)
        return command

    def interactive_environment(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> dict[str, str]:
        del config, execution
        return {}

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

        if summary.has_status(401, 403) or summary.matches_text(_AUTH_PATTERNS):
            return FailureDecision(
                kind=FailureKind.AUTH,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Authentication error. Run 'codex login' and verify the account "
                    f"can access the requested model. Check {log_path} for details."
                ),
            )

        if (
            summary.rate_limit_rejected
            or summary.has_status(429)
            or summary.has_error_code("rate_limit", "rate.limit")
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
                message="Codex service overload detected. Waiting 2 minutes before retrying...",
                wait_seconds=120,
                skip_pause=True,
            )

        if exit_code == 2:
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Codex CLI usage/configuration error (exit code 2). "
                    f"Check {log_path} for details."
                ),
            )

        if (
            summary.has_status(400)
            or summary.has_error_code("invalid_request_error")
            or summary.matches_text(_INVALID_REQUEST_PATTERNS)
        ):
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Codex rejected the request. Check the selected model and provider "
                    f"options. See {log_path} for details."
                ),
            )

        log_text = read_log_text(log_path, lower=True)

        if any(pattern in log_text for pattern in _AUTH_PATTERNS):
            return FailureDecision(
                kind=FailureKind.AUTH,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Authentication error. Run 'codex login' and verify the account "
                    f"can access the requested model. Check {log_path} for details."
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
                message="Codex service overload detected. Waiting 2 minutes before retrying...",
                wait_seconds=120,
                skip_pause=True,
            )

        if any(pattern in log_text for pattern in _INVALID_REQUEST_PATTERNS):
            return FailureDecision(
                kind=FailureKind.INVALID_REQUEST,
                fatal=True,
                should_failover=True,
                message=(
                    "FATAL: Codex rejected the request. Check the selected model and provider "
                    f"options. See {log_path} for details."
                ),
            )

        if exit_code == 1:
            return FailureDecision(
                message=f"ERROR: Codex execution failed (exit code 1). Check {log_path} for details."
            )

        return FailureDecision(
            message=f"WARNING: Unexpected exit code {exit_code}. Check {log_path} for details."
        )


_AUTH_PATTERNS = (
    "status\\\":401",
    "status\\\":403",
    "unauthorized",
    "authentication",
    "not logged in",
    "login required",
    "permission denied",
)

_RATE_LIMIT_PATTERNS = (
    "status\\\":429",
    "usage limit",
    "hit your limit",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
)

_OVERLOAD_PATTERNS = (
    "overloaded",
    "temporarily unavailable",
    "server_error",
    "status\\\":500",
    "status\\\":502",
    "status\\\":503",
    "status\\\":504",
    "status\\\":529",
)

_INVALID_REQUEST_PATTERNS = (
    "invalid_request_error",
    "status\\\":400",
    "model is not supported",
    "not supported when using codex",
)


def _nested_lookup(payload: dict[str, object], *path: str) -> object | None:
    current: object = payload
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


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

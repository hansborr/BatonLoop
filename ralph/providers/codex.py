from __future__ import annotations

from decimal import Decimal

from ..config import OutputFormat, RunnerConfig
from .base import FailureDecision
from .utils import decimal_from_value, iter_jsonl, read_log_text


class CodexProvider:
    name = "codex"
    default_binary = "codex"

    def executable_name(self, config: RunnerConfig) -> str:
        return config.provider_binary or self.default_binary

    def validate_config(self, config: RunnerConfig) -> None:
        if config.output_format is not OutputFormat.STREAM_JSON:
            raise ValueError("Provider 'codex' only supports stream-json output.")
        if config.max_turns is not None:
            raise ValueError("Provider 'codex' does not support --max-turns.")

    def build_command(self, config: RunnerConfig) -> list[str]:
        command = [
            self.executable_name(config),
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(config.working_dir),
        ]

        if config.safe_mode:
            command.append("--full-auto")
        else:
            command.append("--dangerously-bypass-approvals-and-sandbox")

        if config.use_bare:
            command.extend(["--ignore-user-config", "--ignore-rules"])

        if config.model:
            command.extend(["-m", config.model])

        return command

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
    ) -> FailureDecision:
        log_text = read_log_text(log_path, lower=True)

        if any(pattern in log_text for pattern in _AUTH_PATTERNS):
            return FailureDecision(
                fatal=True,
                message=(
                    "FATAL: Authentication error. Run 'codex login' and verify the account "
                    f"can access the requested model. Check {log_path} for details."
                ),
            )

        if any(pattern in log_text for pattern in _RATE_LIMIT_PATTERNS):
            return FailureDecision(
                message=(
                    "RATE LIMITED detected in output. "
                    f"Waiting {config.wait_on_limit_mins} minutes before retrying..."
                ),
                wait_seconds=config.wait_on_limit_mins * 60,
                reset_error_count=True,
                skip_pause=True,
            )

        if any(pattern in log_text for pattern in _OVERLOAD_PATTERNS):
            return FailureDecision(
                message="Codex service overload detected. Waiting 2 minutes before retrying...",
                wait_seconds=120,
                skip_pause=True,
            )

        if exit_code == 2:
            return FailureDecision(
                fatal=True,
                message=(
                    "FATAL: Codex CLI usage/configuration error (exit code 2). "
                    f"Check {log_path} for details."
                ),
            )

        if any(pattern in log_text for pattern in _INVALID_REQUEST_PATTERNS):
            return FailureDecision(
                fatal=True,
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

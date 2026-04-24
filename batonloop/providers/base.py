from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..config import OutputFormat, ProviderExecution, RunnerConfig


class FailureKind(StrEnum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    INVALID_REQUEST = "invalid_request"
    GENERIC = "generic"


@dataclass(frozen=True, slots=True)
class FailureDecision:
    message: str
    kind: FailureKind = FailureKind.GENERIC
    wait_seconds: int = 0
    reset_error_count: bool = False
    skip_pause: bool = False
    fatal: bool = False
    should_failover: bool = False


class Provider(Protocol):
    name: str

    def executable_name(self, execution: ProviderExecution) -> str:
        ...

    def validate_config(self, config: RunnerConfig, execution: ProviderExecution) -> None:
        ...

    def build_command(
        self,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> list[str]:
        ...

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        ...

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
        execution: ProviderExecution,
    ) -> FailureDecision:
        ...

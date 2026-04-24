from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from ..config import OutputFormat, RunnerConfig


@dataclass(frozen=True, slots=True)
class FailureDecision:
    message: str
    wait_seconds: int = 0
    reset_error_count: bool = False
    skip_pause: bool = False
    fatal: bool = False


class Provider(Protocol):
    name: str

    def executable_name(self, config: RunnerConfig) -> str:
        ...

    def validate_config(self, config: RunnerConfig) -> None:
        ...

    def build_command(self, config: RunnerConfig) -> list[str]:
        ...

    def extract_cost(self, log_path: Path, output_format: OutputFormat) -> Decimal:
        ...

    def classify_failure(
        self,
        exit_code: int,
        log_path: Path,
        config: RunnerConfig,
    ) -> FailureDecision:
        ...

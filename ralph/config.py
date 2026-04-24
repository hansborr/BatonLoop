from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path


class OutputFormat(StrEnum):
    STREAM_JSON = "stream-json"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    path: Path
    repeat: int = 1


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    provider_name: str
    provider_binary: str | None
    prompt_specs: tuple[PromptSpec, ...]
    prompt_sequence: tuple[Path, ...]
    max_iterations: int
    max_cost: Decimal
    max_duration_hours: Decimal
    pause_seconds: int
    model: str | None
    wait_on_limit_mins: int
    max_consecutive_errors: int
    max_turns: int | None
    log_dir: Path
    log_retain: int
    output_format: OutputFormat
    use_bare: bool
    safe_mode: bool
    dry_run: bool


def parse_prompt_spec(raw: str) -> PromptSpec:
    head, separator, tail = raw.rpartition(":")
    if separator and re.fullmatch(r"[1-9][0-9]*", tail):
        path_text = head
        repeat = int(tail)
    else:
        path_text = raw
        repeat = 1

    if not path_text:
        raise ValueError("Prompt file path may not be empty.")

    return PromptSpec(path=Path(path_text).expanduser(), repeat=repeat)


def expand_prompt_specs(prompt_specs: tuple[PromptSpec, ...]) -> tuple[Path, ...]:
    prompt_sequence: list[Path] = []
    for prompt_spec in prompt_specs:
        prompt_sequence.extend([prompt_spec.path] * prompt_spec.repeat)
    return tuple(prompt_sequence)


def ensure_prompt_files_exist(prompt_sequence: tuple[Path, ...]) -> None:
    seen: set[Path] = set()
    for prompt_path in prompt_sequence:
        if prompt_path in seen:
            continue
        seen.add(prompt_path)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def parse_non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {raw!r}.") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {raw!r}.")
    return value


def parse_positive_int(raw: str) -> int:
    value = parse_non_negative_int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw!r}.")
    return value


def parse_non_negative_decimal(raw: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Expected a decimal value, got {raw!r}.") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative decimal, got {raw!r}.")
    return value


def build_config(args: argparse.Namespace) -> RunnerConfig:
    raw_prompt_specs = args.prompt_specs or ["./PROMPT.md"]
    prompt_specs = tuple(parse_prompt_spec(raw) for raw in raw_prompt_specs)
    prompt_sequence = expand_prompt_specs(prompt_specs)

    output_format = (
        OutputFormat.JSON if args.no_stream else OutputFormat(args.output_format)
    )

    config = RunnerConfig(
        provider_name=args.provider,
        provider_binary=args.provider_binary,
        prompt_specs=prompt_specs,
        prompt_sequence=prompt_sequence,
        max_iterations=args.max_iterations,
        max_cost=args.max_cost,
        max_duration_hours=args.max_duration_hours,
        pause_seconds=args.pause_seconds,
        model=args.model or None,
        wait_on_limit_mins=args.wait_on_limit_mins,
        max_consecutive_errors=args.max_consecutive_errors,
        max_turns=args.max_turns,
        log_dir=Path(args.log_dir).expanduser(),
        log_retain=args.log_retain,
        output_format=output_format,
        use_bare=args.bare,
        safe_mode=args.safe,
        dry_run=args.dry_run,
    )

    ensure_prompt_files_exist(config.prompt_sequence)
    return config


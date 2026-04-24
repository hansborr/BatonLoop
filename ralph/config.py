from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any


class OutputFormat(StrEnum):
    STREAM_JSON = "stream-json"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    path: Path
    repeat: int = 1


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    binary: str | None = None
    model: str | None = None
    max_turns: int | None = None
    use_bare: bool | None = None
    safe_mode: bool | None = None


@dataclass(frozen=True, slots=True)
class ProviderExecution:
    name: str
    binary: str | None
    model: str | None
    max_turns: int | None
    use_bare: bool
    safe_mode: bool


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    working_dir: Path
    provider_names: tuple[str, ...]
    provider_profiles: dict[str, ProviderProfile]
    provider_config_path: Path | None
    default_provider_profile: ProviderProfile
    prompt_specs: tuple[PromptSpec, ...]
    prompt_sequence: tuple[Path, ...]
    max_iterations: int
    max_cost: Decimal
    max_duration_hours: Decimal
    iteration_timeout_minutes: Decimal
    pause_seconds: int
    wait_on_limit_mins: int
    max_consecutive_errors: int
    log_dir: Path
    log_retain: int
    check_commands: tuple[str, ...]
    stop_on_regexes: tuple[str, ...]
    stop_on_clean_git: bool
    stop_when_files: tuple[Path, ...]
    output_format: OutputFormat
    resume_from: Path | None
    resume_note: str | None
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


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def ensure_valid_regexes(patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression {pattern!r}: {exc}") from exc


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
    if not value.is_finite():
        raise argparse.ArgumentTypeError(f"Expected a finite decimal value, got {raw!r}.")
    if value < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative decimal, got {raw!r}.")
    return value


def resolve_provider_execution(config: RunnerConfig, provider_name: str) -> ProviderExecution:
    profile = config.provider_profiles.get(provider_name, ProviderProfile())
    defaults = config.default_provider_profile

    return ProviderExecution(
        name=provider_name,
        binary=_first_defined(profile.binary, defaults.binary),
        model=_first_defined(profile.model, defaults.model),
        max_turns=_first_defined(profile.max_turns, defaults.max_turns),
        use_bare=_first_defined(profile.use_bare, defaults.use_bare, fallback=False),
        safe_mode=_first_defined(profile.safe_mode, defaults.safe_mode, fallback=False),
    )


def build_config(args: argparse.Namespace) -> RunnerConfig:
    working_dir = Path.cwd()
    raw_prompt_specs = args.prompt_specs or ["./PROMPT.md"]
    prompt_specs = tuple(
        PromptSpec(
            path=resolve_path(parsed.path, working_dir),
            repeat=parsed.repeat,
        )
        for parsed in (parse_prompt_spec(raw) for raw in raw_prompt_specs)
    )
    prompt_sequence = expand_prompt_specs(prompt_specs)
    stop_on_regexes = tuple(args.stop_on_regexes or ())
    ensure_valid_regexes(stop_on_regexes)
    stop_when_files = tuple(
        resolve_path(Path(raw_path).expanduser(), working_dir)
        for raw_path in (args.stop_when_files or [])
    )
    resume_from = (
        resolve_path(Path(args.resume_from).expanduser(), working_dir)
        if args.resume_from
        else None
    )
    provider_config_path = _resolve_provider_config_path(
        working_dir=working_dir,
        explicit_path=args.provider_config,
    )
    provider_profiles = _load_provider_profiles(provider_config_path)

    output_format = (
        OutputFormat.JSON if args.no_stream else OutputFormat(args.output_format)
    )

    config = RunnerConfig(
        working_dir=working_dir,
        provider_names=tuple(args.provider_names or ["claude"]),
        provider_profiles=provider_profiles,
        provider_config_path=provider_config_path,
        default_provider_profile=ProviderProfile(
            binary=args.provider_binary or None,
            model=args.model or None,
            max_turns=args.max_turns,
            use_bare=args.bare,
            safe_mode=args.safe,
        ),
        prompt_specs=prompt_specs,
        prompt_sequence=prompt_sequence,
        max_iterations=args.max_iterations,
        max_cost=args.max_cost,
        max_duration_hours=args.max_duration_hours,
        iteration_timeout_minutes=args.iteration_timeout_minutes,
        pause_seconds=args.pause_seconds,
        wait_on_limit_mins=args.wait_on_limit_mins,
        max_consecutive_errors=args.max_consecutive_errors,
        log_dir=resolve_path(Path(args.log_dir).expanduser(), working_dir),
        log_retain=args.log_retain,
        check_commands=tuple(args.check_commands or ()),
        stop_on_regexes=stop_on_regexes,
        stop_on_clean_git=args.stop_on_clean_git,
        stop_when_files=stop_when_files,
        output_format=output_format,
        resume_from=resume_from,
        resume_note=args.resume_note or None,
        dry_run=args.dry_run,
    )

    ensure_prompt_files_exist(config.prompt_sequence)
    return config


def _resolve_provider_config_path(
    *,
    working_dir: Path,
    explicit_path: str | None,
) -> Path | None:
    if explicit_path:
        config_path = resolve_path(Path(explicit_path).expanduser(), working_dir)
        if not config_path.is_file():
            raise FileNotFoundError(f"Provider config not found: {config_path}")
        return config_path

    default_path = working_dir / "ralph-providers.toml"
    if default_path.is_file():
        return default_path
    return None


def _load_provider_profiles(path: Path | None) -> dict[str, ProviderProfile]:
    if path is None:
        return {}

    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read provider config: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid provider config {path}: {exc}") from exc

    providers = payload.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"Provider config {path} must define a [providers] table.")

    profiles: dict[str, ProviderProfile] = {}
    for provider_name, raw_profile in providers.items():
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError(f"Provider config {path} contains an invalid provider name.")
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"Provider config {path} entry [providers.{provider_name}] must be a table."
            )
        profiles[provider_name] = _parse_provider_profile(provider_name, raw_profile, path)

    return profiles


def _parse_provider_profile(
    provider_name: str,
    raw_profile: dict[str, Any],
    path: Path,
) -> ProviderProfile:
    allowed_keys = {"binary", "model", "max_turns", "bare", "safe"}
    unknown_keys = sorted(set(raw_profile) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}] contains unsupported "
            f"key(s): {', '.join(unknown_keys)}"
        )

    binary = raw_profile.get("binary")
    model = raw_profile.get("model")
    max_turns = raw_profile.get("max_turns")
    use_bare = raw_profile.get("bare")
    safe_mode = raw_profile.get("safe")

    if binary is not None and not isinstance(binary, str):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].binary must be a string."
        )
    if model is not None and not isinstance(model, str):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].model must be a string."
        )
    if max_turns is not None and (not isinstance(max_turns, int) or max_turns <= 0):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].max_turns must be "
            "a positive integer."
        )
    if use_bare is not None and not isinstance(use_bare, bool):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].bare must be a boolean."
        )
    if safe_mode is not None and not isinstance(safe_mode, bool):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].safe must be a boolean."
        )

    return ProviderProfile(
        binary=binary,
        model=model,
        max_turns=max_turns,
        use_bare=use_bare,
        safe_mode=safe_mode,
    )


def _first_defined(*values: object, fallback: object | None = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return fallback

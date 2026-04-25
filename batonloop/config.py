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
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderExecution:
    name: str
    binary: str | None
    model: str | None
    max_turns: int | None
    use_bare: bool
    safe_mode: bool
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    working_dir: Path
    run_config_path: Path | None
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
    retry_backoff_base_seconds: int
    retry_backoff_multiplier: Decimal
    retry_backoff_max_seconds: int
    retry_jitter_fraction: Decimal
    provider_cooldown_seconds: int
    max_consecutive_errors: int
    log_dir: Path
    log_retain: int
    check_commands: tuple[str, ...]
    stop_on_regexes: tuple[str, ...]
    stop_on_clean_git: bool
    stop_when_files: tuple[Path, ...]
    output_format: OutputFormat
    live_output: bool
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


def parse_decimal_at_least_one(raw: str) -> Decimal:
    value = parse_non_negative_decimal(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"Expected a decimal >= 1, got {raw!r}.")
    return value


def parse_fraction(raw: str) -> Decimal:
    value = parse_non_negative_decimal(raw)
    if value > 1:
        raise argparse.ArgumentTypeError(f"Expected a decimal between 0 and 1, got {raw!r}.")
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
        extra_args=defaults.extra_args + profile.extra_args,
    )


def build_config(args: argparse.Namespace) -> RunnerConfig:
    working_dir = Path.cwd()
    run_config_path = _resolve_run_config_path(
        working_dir=working_dir,
        explicit_path=getattr(args, "config", None),
    )
    run_settings, run_provider_profiles = _load_run_config(run_config_path)
    raw_prompt_specs = _coalesce(args.prompt_specs, run_settings.get("prompt_specs"), ["./PROMPT.md"])
    prompt_specs = tuple(
        PromptSpec(
            path=resolve_path(parsed.path, working_dir),
            repeat=parsed.repeat,
        )
        for parsed in (parse_prompt_spec(raw) for raw in raw_prompt_specs)
    )
    prompt_sequence = expand_prompt_specs(prompt_specs)
    stop_on_regexes = tuple(_coalesce(args.stop_on_regexes, run_settings.get("stop_on_regexes"), ()))
    ensure_valid_regexes(stop_on_regexes)
    stop_when_files = tuple(
        resolve_path(Path(raw_path).expanduser(), working_dir)
        for raw_path in _coalesce(args.stop_when_files, run_settings.get("stop_when_files"), ())
    )
    resume_from_raw = _coalesce(args.resume_from, run_settings.get("resume_from"), None)
    resume_from = (
        resolve_path(Path(resume_from_raw).expanduser(), working_dir)
        if resume_from_raw
        else None
    )
    provider_config_path = _resolve_provider_config_path(
        working_dir=working_dir,
        explicit_path=_coalesce(args.provider_config, run_settings.get("provider_config"), None),
    )
    provider_profiles = {
        **run_provider_profiles,
        **_load_provider_profiles(provider_config_path),
    }

    output_format_raw = _coalesce(
        args.output_format,
        run_settings.get("output_format"),
        OutputFormat.STREAM_JSON.value,
    )
    output_format = OutputFormat(output_format_raw)
    if _coalesce(args.no_stream, run_settings.get("no_stream"), False):
        output_format = OutputFormat.JSON

    provider_names = tuple(
        _coalesce(args.provider_names, run_settings.get("provider_names"), ("claude",))
    )
    max_iterations = _coalesce(args.max_iterations, run_settings.get("max_iterations"), 0)
    max_cost = _coalesce(args.max_cost, run_settings.get("max_cost"), Decimal("0"))
    max_duration_hours = _coalesce(
        args.max_duration_hours,
        run_settings.get("max_duration_hours"),
        Decimal("0"),
    )
    iteration_timeout_minutes = _coalesce(
        args.iteration_timeout_minutes,
        run_settings.get("iteration_timeout_minutes"),
        Decimal("0"),
    )
    pause_seconds = _coalesce(args.pause_seconds, run_settings.get("pause_seconds"), 5)
    wait_on_limit_mins = _coalesce(
        args.wait_on_limit_mins,
        run_settings.get("wait_on_limit_mins"),
        30,
    )
    retry_backoff_base_seconds = _coalesce(
        args.retry_backoff_base_seconds,
        run_settings.get("retry_backoff_base_seconds"),
        0,
    )
    retry_backoff_multiplier = _coalesce(
        args.retry_backoff_multiplier,
        run_settings.get("retry_backoff_multiplier"),
        Decimal("2"),
    )
    retry_backoff_max_seconds = _coalesce(
        args.retry_backoff_max_seconds,
        run_settings.get("retry_backoff_max_seconds"),
        0,
    )
    retry_jitter_fraction = _coalesce(
        args.retry_jitter_fraction,
        run_settings.get("retry_jitter_fraction"),
        Decimal("0"),
    )
    provider_cooldown_seconds = _coalesce(
        args.provider_cooldown_seconds,
        run_settings.get("provider_cooldown_seconds"),
        0,
    )
    max_consecutive_errors = _coalesce(
        args.max_consecutive_errors,
        run_settings.get("max_consecutive_errors"),
        5,
    )
    max_turns = _coalesce(args.max_turns, run_settings.get("max_turns"), None)
    log_dir = _coalesce(args.log_dir, run_settings.get("log_dir"), "./batonloop-logs")
    log_retain = _coalesce(args.log_retain, run_settings.get("log_retain"), 0)
    check_commands = tuple(_coalesce(args.check_commands, run_settings.get("check_commands"), ()))
    stop_on_clean_git = _coalesce(
        args.stop_on_clean_git,
        run_settings.get("stop_on_clean_git"),
        False,
    )
    live_output = _coalesce(args.live_output, run_settings.get("live_output"), True)
    resume_note = _coalesce(args.resume_note, run_settings.get("resume_note"), None)
    dry_run = _coalesce(args.dry_run, run_settings.get("dry_run"), False)

    config = RunnerConfig(
        working_dir=working_dir,
        run_config_path=run_config_path,
        provider_names=provider_names,
        provider_profiles=provider_profiles,
        provider_config_path=provider_config_path,
        default_provider_profile=ProviderProfile(
            binary=_coalesce(args.provider_binary, run_settings.get("provider_binary"), None),
            model=_coalesce(args.model, run_settings.get("model"), None),
            max_turns=max_turns,
            use_bare=_coalesce(args.bare, run_settings.get("bare"), None),
            safe_mode=_coalesce(args.safe, run_settings.get("safe"), None),
        ),
        prompt_specs=prompt_specs,
        prompt_sequence=prompt_sequence,
        max_iterations=max_iterations,
        max_cost=max_cost,
        max_duration_hours=max_duration_hours,
        iteration_timeout_minutes=iteration_timeout_minutes,
        pause_seconds=pause_seconds,
        wait_on_limit_mins=wait_on_limit_mins,
        retry_backoff_base_seconds=retry_backoff_base_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        retry_backoff_max_seconds=retry_backoff_max_seconds,
        retry_jitter_fraction=retry_jitter_fraction,
        provider_cooldown_seconds=provider_cooldown_seconds,
        max_consecutive_errors=max_consecutive_errors,
        log_dir=resolve_path(Path(log_dir).expanduser(), working_dir),
        log_retain=log_retain,
        check_commands=check_commands,
        stop_on_regexes=stop_on_regexes,
        stop_on_clean_git=stop_on_clean_git,
        stop_when_files=stop_when_files,
        output_format=output_format,
        live_output=live_output,
        resume_from=resume_from,
        resume_note=resume_note or None,
        dry_run=dry_run,
    )

    ensure_prompt_files_exist(config.prompt_sequence)
    return config


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_run_config_path(
    *,
    working_dir: Path,
    explicit_path: str | None,
) -> Path | None:
    if explicit_path:
        config_path = resolve_path(Path(explicit_path).expanduser(), working_dir)
        if not config_path.is_file():
            raise FileNotFoundError(f"Run config not found: {config_path}")
        return config_path

    default_path = working_dir / "batonloop.toml"
    if default_path.is_file():
        return default_path
    return None


def _load_run_config(path: Path | None) -> tuple[dict[str, Any], dict[str, ProviderProfile]]:
    if path is None:
        return {}, {}

    payload = _load_toml_payload(path, description="run config")
    allowed_top_level_keys = {"run", "providers"}
    unknown_top_level_keys = sorted(set(payload) - allowed_top_level_keys)
    if unknown_top_level_keys:
        raise ValueError(
            f"Run config {path} contains unsupported top-level key(s): "
            f"{', '.join(unknown_top_level_keys)}"
        )

    raw_run_settings = payload.get("run", {})
    if not isinstance(raw_run_settings, dict):
        raise ValueError(f"Run config {path} entry [run] must be a table.")

    return _parse_run_settings(raw_run_settings, path), _parse_provider_profiles_payload(
        payload,
        path,
        description="Run config",
        require_table=False,
    )


def _parse_run_settings(raw_settings: dict[str, Any], path: Path) -> dict[str, Any]:
    aliases = {
        "providers": "provider_names",
        "provider_names": "provider_names",
        "prompt_files": "prompt_specs",
        "prompt_specs": "prompt_specs",
        "provider_config": "provider_config",
        "provider_binary": "provider_binary",
        "iterations": "max_iterations",
        "max_iterations": "max_iterations",
        "max_cost": "max_cost",
        "duration_hours": "max_duration_hours",
        "max_duration_hours": "max_duration_hours",
        "iteration_timeout": "iteration_timeout_minutes",
        "iteration_timeout_minutes": "iteration_timeout_minutes",
        "pause": "pause_seconds",
        "pause_seconds": "pause_seconds",
        "model": "model",
        "wait_on_limit": "wait_on_limit_mins",
        "wait_on_limit_mins": "wait_on_limit_mins",
        "retry_backoff_base": "retry_backoff_base_seconds",
        "retry_backoff_base_seconds": "retry_backoff_base_seconds",
        "retry_backoff_multiplier": "retry_backoff_multiplier",
        "retry_backoff_max": "retry_backoff_max_seconds",
        "retry_backoff_max_seconds": "retry_backoff_max_seconds",
        "retry_jitter": "retry_jitter_fraction",
        "retry_jitter_fraction": "retry_jitter_fraction",
        "provider_cooldown": "provider_cooldown_seconds",
        "provider_cooldown_seconds": "provider_cooldown_seconds",
        "max_errors": "max_consecutive_errors",
        "max_consecutive_errors": "max_consecutive_errors",
        "max_turns": "max_turns",
        "log_dir": "log_dir",
        "log_retain": "log_retain",
        "checks": "check_commands",
        "check_commands": "check_commands",
        "stop_on_regexes": "stop_on_regexes",
        "stop_on_clean_git": "stop_on_clean_git",
        "stop_when_files": "stop_when_files",
        "output_format": "output_format",
        "no_stream": "no_stream",
        "live_output": "live_output",
        "bare": "bare",
        "safe": "safe",
        "resume_from": "resume_from",
        "resume_note": "resume_note",
        "dry_run": "dry_run",
    }

    parsed: dict[str, Any] = {}
    for raw_key, raw_value in raw_settings.items():
        canonical_key = aliases.get(raw_key)
        if canonical_key is None:
            raise ValueError(f"Run config {path} entry [run] contains unsupported key: {raw_key}")
        if canonical_key in parsed:
            raise ValueError(
                f"Run config {path} entry [run] defines {canonical_key} more than once."
            )
        parsed[canonical_key] = _parse_run_setting_value(canonical_key, raw_value, path)
    return parsed


def _parse_run_setting_value(key: str, value: Any, path: Path) -> Any:
    if key in {
        "provider_names",
        "prompt_specs",
        "check_commands",
        "stop_on_regexes",
        "stop_when_files",
    }:
        return _parse_string_list(value, path, f"[run].{key}")
    if key in {
        "provider_config",
        "provider_binary",
        "model",
        "log_dir",
        "output_format",
        "resume_from",
        "resume_note",
    }:
        return _parse_optional_string(value, path, f"[run].{key}")
    if key in {
        "max_iterations",
        "pause_seconds",
        "wait_on_limit_mins",
        "retry_backoff_base_seconds",
        "retry_backoff_max_seconds",
        "provider_cooldown_seconds",
        "log_retain",
    }:
        return _parse_config_int(value, path, f"[run].{key}", parse_non_negative_int)
    if key in {"max_consecutive_errors", "max_turns"}:
        return _parse_config_int(value, path, f"[run].{key}", parse_positive_int)
    if key in {"max_cost", "max_duration_hours", "iteration_timeout_minutes"}:
        return _parse_config_decimal(value, path, f"[run].{key}", parse_non_negative_decimal)
    if key == "retry_backoff_multiplier":
        return _parse_config_decimal(value, path, f"[run].{key}", parse_decimal_at_least_one)
    if key == "retry_jitter_fraction":
        return _parse_config_decimal(value, path, f"[run].{key}", parse_fraction)
    if key in {
        "stop_on_clean_git",
        "no_stream",
        "live_output",
        "bare",
        "safe",
        "dry_run",
    }:
        return _parse_bool(value, path, f"[run].{key}")
    raise AssertionError(f"Unhandled run config key: {key}")


def _parse_string_list(value: Any, path: Path, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Run config {path} entry {field} must be an array of strings.")
    return tuple(value)


def _parse_optional_string(value: Any, path: Path, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Run config {path} entry {field} must be a string.")
    return value or None


def _parse_bool(value: Any, path: Path, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Run config {path} entry {field} must be a boolean.")
    return value


def _parse_config_int(value: Any, path: Path, field: str, parser: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Run config {path} entry {field} must be an integer.")
    try:
        return parser(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"Run config {path} entry {field}: {exc}") from exc


def _parse_config_decimal(value: Any, path: Path, field: str, parser: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Run config {path} entry {field} must be a decimal.")
    try:
        return parser(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"Run config {path} entry {field}: {exc}") from exc


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

    default_path = working_dir / "batonloop-providers.toml"
    if default_path.is_file():
        return default_path
    legacy_default_path = working_dir / "ralph-providers.toml"
    if legacy_default_path.is_file():
        return legacy_default_path
    return None


def _load_provider_profiles(path: Path | None) -> dict[str, ProviderProfile]:
    if path is None:
        return {}

    payload = _load_toml_payload(path, description="provider config")
    return _parse_provider_profiles_payload(
        payload,
        path,
        description="Provider config",
        require_table=True,
    )


def _load_toml_payload(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read {description}: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid {description} {path}: {exc}") from exc
    return payload


def _parse_provider_profiles_payload(
    payload: dict[str, Any],
    path: Path,
    *,
    description: str,
    require_table: bool,
) -> dict[str, ProviderProfile]:
    providers = payload.get("providers", {})
    if providers == {} and not require_table:
        return {}
    if not isinstance(providers, dict):
        raise ValueError(f"{description} {path} must define a [providers] table.")

    profiles: dict[str, ProviderProfile] = {}
    for provider_name, raw_profile in providers.items():
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError(f"{description} {path} contains an invalid provider name.")
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"{description} {path} entry [providers.{provider_name}] must be a table."
            )
        profiles[provider_name] = _parse_provider_profile(provider_name, raw_profile, path)

    return profiles


def _parse_provider_profile(
    provider_name: str,
    raw_profile: dict[str, Any],
    path: Path,
) -> ProviderProfile:
    allowed_keys = {"binary", "model", "max_turns", "bare", "safe", "args"}
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
    extra_args = raw_profile.get("args")

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
    if extra_args is None:
        parsed_extra_args: tuple[str, ...] = ()
    elif not isinstance(extra_args, list) or any(not isinstance(arg, str) for arg in extra_args):
        raise ValueError(
            f"Provider config {path} entry [providers.{provider_name}].args must be an "
            "array of strings."
        )
    else:
        parsed_extra_args = tuple(extra_args)

    return ProviderProfile(
        binary=binary,
        model=model,
        max_turns=max_turns,
        use_bare=use_bare,
        safe_mode=safe_mode,
        extra_args=parsed_extra_args,
    )


def _first_defined(*values: object, fallback: object | None = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return fallback

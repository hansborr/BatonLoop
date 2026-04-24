from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from shutil import which

from .config import ProviderExecution


_ITERATION_LOG_FILENAME = re.compile(r"^(iteration-\d{6})\.json$")
_ITERATION_ARTIFACT_STEM = re.compile(r"^(iteration-\d{6})(?:\..+)?$")


@dataclass(frozen=True, slots=True)
class ResumeContext:
    source_log_path: Path
    source_metadata_path: Path | None
    previous_provider: str | None
    previous_prompt_path: Path | None
    previous_exit_code: int | None
    previous_timed_out: bool | None
    previous_failure_message: str | None
    previous_stop_reason: str | None


def metadata_path_for(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.meta.json")


def prompt_artifact_path_for(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.prompt.txt")


def resolve_resume_context(path: Path) -> ResumeContext:
    source_log_path = _resolve_resume_log_path(path)
    source_metadata_path = metadata_path_for(source_log_path)
    metadata = _load_metadata(source_metadata_path) if source_metadata_path.is_file() else None

    return ResumeContext(
        source_log_path=source_log_path,
        source_metadata_path=source_metadata_path if metadata is not None else None,
        previous_provider=_string_field(metadata, "provider_name"),
        previous_prompt_path=_path_field(metadata, "base_prompt_path"),
        previous_exit_code=_int_field(metadata, "exit_code"),
        previous_timed_out=_bool_field(metadata, "timed_out"),
        previous_failure_message=_string_field(metadata, "failure_message"),
        previous_stop_reason=_string_field(metadata, "stop_reason"),
    )


def build_resume_prompt(
    *,
    base_prompt_path: Path,
    current_provider_name: str,
    working_dir: Path,
    log_dir: Path,
    resume_context: ResumeContext,
    resume_note: str | None,
) -> str:
    base_prompt = base_prompt_path.read_text(encoding="utf-8", errors="replace").rstrip()
    resume_block = _render_resume_block(
        current_provider_name=current_provider_name,
        working_dir=working_dir,
        log_dir=log_dir,
        resume_context=resume_context,
        resume_note=resume_note,
    )

    if base_prompt:
        return f"{base_prompt}\n\n{resume_block}\n"
    return f"{resume_block}\n"


def write_iteration_metadata(
    *,
    log_path: Path,
    execution: ProviderExecution,
    working_dir: Path,
    log_dir: Path,
    base_prompt_path: Path,
    input_prompt_path: Path,
    output_format: str,
    exit_code: int,
    timed_out: bool,
    success: bool,
    iteration_cost: Decimal,
    failure_message: str | None,
    stop_reason: str | None,
    failover_target_provider: str | None,
    resume_context: ResumeContext | None,
    resume_note: str | None,
) -> None:
    payload = {
        "version": 1,
        "provider_name": execution.name,
        "provider_binary": execution.binary,
        "provider_model": execution.model,
        "provider_max_turns": execution.max_turns,
        "provider_use_bare": execution.use_bare,
        "provider_safe_mode": execution.safe_mode,
        "working_dir": str(working_dir),
        "log_dir": str(log_dir),
        "log_path": str(log_path),
        "base_prompt_path": str(base_prompt_path),
        "input_prompt_path": str(input_prompt_path),
        "output_format": output_format,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": success,
        "iteration_cost_usd": _decimal_to_string(iteration_cost),
        "failure_message": failure_message,
        "stop_reason": stop_reason,
        "failover_target_provider": failover_target_provider,
        "resume_source_log_path": str(resume_context.source_log_path) if resume_context else None,
        "resume_source_metadata_path": (
            str(resume_context.source_metadata_path)
            if resume_context and resume_context.source_metadata_path is not None
            else None
        ),
        "resume_note": resume_note,
        "git_status": list(get_git_status_snapshot(working_dir, log_dir)),
    }
    metadata_path_for(log_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_git_status_snapshot(
    working_dir: Path,
    log_dir: Path,
    *,
    max_lines: int = 20,
) -> tuple[str, ...]:
    if which("git") is None:
        return ()

    repo_root = _git_toplevel(working_dir)
    if repo_root is None:
        return ()

    status_command = [
        "git",
        "status",
        "--short",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        ".",
    ]
    try:
        relative_log_dir = log_dir.relative_to(repo_root)
    except ValueError:
        relative_log_dir = None

    if relative_log_dir is not None:
        status_command.append(f":(exclude){relative_log_dir.as_posix()}")

    status_result = subprocess.run(
        status_command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        return ()

    lines = [line.rstrip() for line in status_result.stdout.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return tuple(lines)

    remaining = len(lines) - max_lines
    return tuple(lines[:max_lines] + [f"... ({remaining} more path(s))"])


def _render_resume_block(
    *,
    current_provider_name: str,
    working_dir: Path,
    log_dir: Path,
    resume_context: ResumeContext,
    resume_note: str | None,
) -> str:
    lines = [
        "=== BATONLOOP RESUME CONTEXT ===",
        "You are resuming work from a previous BatonLoop iteration.",
        "Treat the repository as potentially containing partial in-progress changes.",
        f"Current provider: {current_provider_name}",
        f"Previous raw log: {resume_context.source_log_path}",
    ]

    if resume_context.source_metadata_path is not None:
        lines.append(f"Previous metadata: {resume_context.source_metadata_path}")
    if resume_context.previous_provider:
        lines.append(f"Previous provider: {resume_context.previous_provider}")
    if resume_context.previous_prompt_path is not None:
        lines.append(f"Previous prompt file: {resume_context.previous_prompt_path}")
    if resume_context.previous_exit_code is not None:
        lines.append(f"Previous exit code: {resume_context.previous_exit_code}")
    if resume_context.previous_timed_out is not None:
        lines.append(f"Previous timed out: {resume_context.previous_timed_out}")
    if resume_context.previous_failure_message:
        lines.append(f"Previous failure summary: {resume_context.previous_failure_message}")
    if resume_context.previous_stop_reason:
        lines.append(f"Previous stop reason: {resume_context.previous_stop_reason}")

    git_status = get_git_status_snapshot(working_dir, log_dir)
    if git_status:
        lines.append("Current git status (excluding BatonLoop logs):")
        lines.extend(git_status)
    else:
        lines.append("Current git status (excluding BatonLoop logs): clean or unavailable.")

    if resume_note:
        lines.append(f"Operator note: {resume_note}")

    lines.append(
        "Inspect the current worktree and previous artifacts if you need more detail before changing direction."
    )
    lines.append("=== END BATONLOOP RESUME CONTEXT ===")
    return "\n".join(lines)


def _resolve_resume_log_path(path: Path) -> Path:
    if path.is_dir():
        return _latest_iteration_log(path)

    if not path.is_file():
        raise FileNotFoundError(f"Resume source not found: {path}")

    match = _ITERATION_LOG_FILENAME.match(path.name)
    if match is not None:
        return path

    stem_match = _ITERATION_ARTIFACT_STEM.match(path.name)
    if stem_match is None:
        raise ValueError(
            "Resume source must be an iteration log, an iteration artifact, or a BatonLoop log directory."
        )

    derived_log_path = path.with_name(f"{stem_match.group(1)}.json")
    if not derived_log_path.is_file():
        raise FileNotFoundError(
            f"Could not locate the raw iteration log for resume source: {path}"
        )
    return derived_log_path


def _latest_iteration_log(log_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in log_dir.iterdir()
            if path.is_file() and _ITERATION_LOG_FILENAME.match(path.name)
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No iteration logs found in resume directory: {log_dir}")
    return candidates[-1]


def _load_metadata(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _string_field(payload: dict[str, object] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _path_field(payload: dict[str, object] | None, key: str) -> Path | None:
    value = _string_field(payload, key)
    if value is None:
        return None
    return Path(value)


def _int_field(payload: dict[str, object] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, int) else None


def _bool_field(payload: dict[str, object] | None, key: str) -> bool | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _git_toplevel(working_dir: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def _decimal_to_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"

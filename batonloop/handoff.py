from __future__ import annotations

import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from shutil import which
from typing import Any, Callable

from .config import ProviderExecution


_ITERATION_LOG_FILENAME = re.compile(r"^(iteration-\d{6})\.json$")
_ITERATION_ARTIFACT_STEM = re.compile(r"^(iteration-\d{6})(?:\..+)?$")
_MAX_SUMMARY_WORDS = 500
_MAX_TEXT_SNIPPET_CHARS = 240
_MAX_TASK_SNIPPET_CHARS = 320
_HANDOFF_EXTRACTOR_VERSION = 2
_INTERRUPTION_PATTERNS = (
    "hit your limit",
    "usage limit",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
    "temporarily unavailable",
)
_GENERIC_SUCCESS_RESULTS = {
    "completed successfully",
    "success",
}


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
    previous_handoff_summary: str | None
    previous_retry_recommended_next_step: str | None


@dataclass(frozen=True, slots=True)
class HandoffDetails:
    summary: str | None
    last_progress_messages: tuple[str, ...]
    last_tasks: tuple[str, ...]
    last_interruption: str | None
    retry_recommended_next_step: str | None


@dataclass(frozen=True, slots=True)
class _TaskSnapshot:
    description: str | None
    prompt: str | None


@dataclass(frozen=True, slots=True)
class _TodoSnapshot:
    total: int
    completed: int
    pending_texts: tuple[str, ...]


def metadata_path_for(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.meta.json")


def prompt_artifact_path_for(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}.prompt.txt")


def resolve_resume_context(path: Path) -> ResumeContext:
    source_log_path = _resolve_resume_log_path(path)
    source_metadata_path = metadata_path_for(source_log_path)
    metadata = _load_metadata(source_metadata_path) if source_metadata_path.is_file() else None
    previous_provider = _string_field(metadata, "provider_name")
    cached_extractor_version = _int_field(metadata, "handoff_extractor_version")
    if (
        metadata is not None
        and cached_extractor_version is not None
        and cached_extractor_version >= _HANDOFF_EXTRACTOR_VERSION
    ):
        previous_handoff_summary = _string_field(metadata, "handoff_summary")
        previous_retry_recommended_next_step = _string_field(
            metadata,
            "retry_recommended_next_step",
        )
    else:
        handoff_details = extract_handoff_details(
            source_log_path,
            provider_hint=previous_provider,
        )
        previous_handoff_summary = handoff_details.summary
        previous_retry_recommended_next_step = (
            handoff_details.retry_recommended_next_step
        )

    return ResumeContext(
        source_log_path=source_log_path,
        source_metadata_path=source_metadata_path if metadata is not None else None,
        previous_provider=previous_provider,
        previous_prompt_path=_path_field(metadata, "base_prompt_path"),
        previous_exit_code=_int_field(metadata, "exit_code"),
        previous_timed_out=_bool_field(metadata, "timed_out"),
        previous_failure_message=_string_field(metadata, "failure_message"),
        previous_stop_reason=_string_field(metadata, "stop_reason"),
        previous_handoff_summary=previous_handoff_summary,
        previous_retry_recommended_next_step=previous_retry_recommended_next_step,
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
    handoff_details = extract_handoff_details(log_path, provider_hint=execution.name)
    payload = {
        "version": 1,
        "handoff_extractor_version": _HANDOFF_EXTRACTOR_VERSION,
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
        "handoff_summary": handoff_details.summary,
        "failover_target_provider": failover_target_provider,
        "resume_source_log_path": (
            str(resume_context.source_log_path) if resume_context else None
        ),
        "resume_source_metadata_path": (
            str(resume_context.source_metadata_path)
            if resume_context and resume_context.source_metadata_path is not None
            else None
        ),
        "resume_note": resume_note,
        "last_progress_messages": list(handoff_details.last_progress_messages),
        "last_tasks": list(handoff_details.last_tasks),
        "last_interruption": handoff_details.last_interruption,
        "retry_recommended_next_step": handoff_details.retry_recommended_next_step,
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


def extract_handoff_summary(
    log_path: Path,
    *,
    provider_hint: str | None = None,
) -> str | None:
    return extract_handoff_details(log_path, provider_hint=provider_hint).summary


def extract_handoff_details(
    log_path: Path,
    *,
    provider_hint: str | None = None,
) -> HandoffDetails:
    messages: list[str] = []
    fallback_user_messages: list[str] = []
    tasks: deque[_TaskSnapshot] = deque(maxlen=4)
    todo_snapshot: _TodoSnapshot | None = None
    interruption_message: str | None = None

    try:
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            pending_payloads: tuple[dict[str, object], ...] | None = None

            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if pending_payloads is not None:
                    todo_snapshot, interruption_message = _consume_summary_payloads(
                        pending_payloads,
                        messages=messages,
                        fallback_user_messages=fallback_user_messages,
                        tasks=tasks,
                        todo_snapshot=todo_snapshot,
                        interruption_message=interruption_message,
                    )
                    pending_payloads = None

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    if line[:1] in {"{", "["}:
                        whole_payloads = _load_whole_log_payloads(raw_line + handle.read())
                        if whole_payloads is not None:
                            todo_snapshot, interruption_message = _consume_summary_payloads(
                                whole_payloads,
                                messages=messages,
                                fallback_user_messages=fallback_user_messages,
                                tasks=tasks,
                                todo_snapshot=todo_snapshot,
                                interruption_message=interruption_message,
                            )
                            break
                    if interruption_message is None and _is_interruption_text(line):
                        interruption_message = _clean_text(line)
                    continue

                pending_payloads = _coerce_summary_payloads(payload)
                continue

            if pending_payloads is not None:
                todo_snapshot, interruption_message = _consume_summary_payloads(
                    pending_payloads,
                    messages=messages,
                    fallback_user_messages=fallback_user_messages,
                    tasks=tasks,
                    todo_snapshot=todo_snapshot,
                    interruption_message=interruption_message,
                )
    except OSError:
        return HandoffDetails(
            summary=None,
            last_progress_messages=(),
            last_tasks=(),
            last_interruption=None,
            retry_recommended_next_step=None,
        )

    message_pool = _dedupe_messages(messages) or _dedupe_messages(fallback_user_messages)
    summary = _render_handoff_summary(
        messages=messages,
        fallback_user_messages=fallback_user_messages,
        tasks=tuple(tasks),
        todo_snapshot=todo_snapshot,
        interruption_message=interruption_message,
        provider_hint=provider_hint,
    )
    return HandoffDetails(
        summary=summary,
        last_progress_messages=tuple(
            text
            for message in message_pool[-5:]
            if (text := _truncate_text(message, max_chars=_MAX_TEXT_SNIPPET_CHARS))
            is not None
        ),
        last_tasks=_render_task_summaries(tuple(tasks)),
        last_interruption=interruption_message,
        retry_recommended_next_step=_pick_retry_next_step(
            message_pool=message_pool,
            tasks=tuple(tasks),
            todo_snapshot=todo_snapshot,
        ),
    )


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
    if resume_context.previous_handoff_summary:
        lines.append(resume_context.previous_handoff_summary)
    if resume_context.previous_retry_recommended_next_step:
        lines.append(
            "Recommended resume point: "
            f"{resume_context.previous_retry_recommended_next_step}"
        )

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


def _load_whole_log_payloads(text: str) -> tuple[dict[str, object], ...] | None:
    stripped = text.strip()
    if not stripped:
        return ()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    return _coerce_summary_payloads(payload)


def _coerce_summary_payloads(payload: object) -> tuple[dict[str, object], ...]:
    if isinstance(payload, dict):
        return (payload,)
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, dict))
    return ()


def _consume_summary_payloads(
    payloads: tuple[dict[str, object], ...],
    *,
    messages: list[str],
    fallback_user_messages: list[str],
    tasks: deque[_TaskSnapshot],
    todo_snapshot: _TodoSnapshot | None,
    interruption_message: str | None,
) -> tuple[_TodoSnapshot | None, str | None]:
    for payload in payloads:
        todo_snapshot, interruption_message = _consume_summary_payload(
            payload,
            messages=messages,
            fallback_user_messages=fallback_user_messages,
            tasks=tasks,
            todo_snapshot=todo_snapshot,
            interruption_message=interruption_message,
        )
    return todo_snapshot, interruption_message


def _consume_summary_payload(
    payload: dict[str, object],
    *,
    messages: list[str],
    fallback_user_messages: list[str],
    tasks: deque[_TaskSnapshot],
    todo_snapshot: _TodoSnapshot | None,
    interruption_message: str | None,
) -> tuple[_TodoSnapshot | None, str | None]:
    for text in _extract_progress_messages(payload):
        if _is_interruption_text(text):
            interruption_message = _clean_text(text)
            continue
        messages.append(text)

    for text in _extract_user_messages(payload):
        if _is_interruption_text(text):
            interruption_message = _clean_text(text)
            continue
        fallback_user_messages.append(text)

    task_snapshot = _extract_task_snapshot(payload)
    if task_snapshot is not None:
        tasks.append(task_snapshot)

    todo_candidate = _extract_todo_snapshot(payload)
    if todo_candidate is not None:
        todo_snapshot = todo_candidate

    interruption_candidate = _extract_interruption_message(payload)
    if interruption_candidate is not None:
        interruption_message = interruption_candidate

    return todo_snapshot, interruption_message


def _extract_progress_messages(payload: dict[str, object]) -> tuple[str, ...]:
    payload_type = payload.get("type")
    if payload_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            return tuple(
                text
                for text in _iter_content_texts(message.get("content"))
                if text
            )

    if payload_type == "item.completed":
        item = payload.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            return (_clean_text(item["text"]),)

    if payload_type == "result":
        result_text = _clean_optional_text(payload.get("result"))
        if (
            result_text is not None
            and payload.get("is_error") is not True
            and not _is_interruption_text(result_text)
            and not _is_generic_success_result_text(result_text)
        ):
            return (result_text,)

    return ()


def _extract_user_messages(payload: dict[str, object]) -> tuple[str, ...]:
    if payload.get("type") != "user":
        return ()

    message = payload.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return ()

    return tuple(text for text in _iter_content_texts(message.get("content")) if text)


def _extract_task_snapshot(payload: dict[str, object]) -> _TaskSnapshot | None:
    payload_type = payload.get("type")

    if payload_type == "system" and payload.get("subtype") == "task_started":
        description = _clean_optional_text(payload.get("description"))
        prompt = _clean_optional_text(payload.get("prompt"))
        if description or prompt:
            return _TaskSnapshot(description=description, prompt=prompt)

    if payload_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict):
            for item in _iter_content_items(message.get("content")):
                if item.get("type") != "tool_use" or item.get("name") != "Agent":
                    continue
                input_payload = item.get("input")
                if not isinstance(input_payload, dict):
                    continue
                description = _clean_optional_text(input_payload.get("description"))
                prompt = _clean_optional_text(input_payload.get("prompt"))
                if description or prompt:
                    return _TaskSnapshot(description=description, prompt=prompt)

    if payload_type in {"item.started", "item.completed"}:
        item = payload.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "collab_tool_call"
            and item.get("tool") in {"spawn_agent", "send_input"}
        ):
            description = _clean_optional_text(item.get("tool"))
            prompt = _clean_optional_text(item.get("prompt"))
            if description or prompt:
                return _TaskSnapshot(description=description, prompt=prompt)

    return None


def _extract_todo_snapshot(payload: dict[str, object]) -> _TodoSnapshot | None:
    if payload.get("type") not in {"item.started", "item.updated", "item.completed"}:
        return None

    item = payload.get("item")
    if not isinstance(item, dict) or item.get("type") != "todo_list":
        return None

    raw_items = item.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None

    total = 0
    completed = 0
    pending_texts: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        text = _clean_optional_text(raw_item.get("text"))
        done = raw_item.get("completed") is True
        total += 1
        if done:
            completed += 1
        elif text:
            pending_texts.append(text)

    if total == 0:
        return None

    return _TodoSnapshot(
        total=total,
        completed=completed,
        pending_texts=tuple(pending_texts),
    )


def _extract_interruption_message(payload: dict[str, object]) -> str | None:
    payload_type = payload.get("type")

    if payload_type == "error":
        return _clean_optional_text(payload.get("message"))

    if payload_type == "turn.failed":
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            return _clean_optional_text(error_payload.get("message"))

    if payload_type == "result":
        result_text = _clean_optional_text(payload.get("result"))
        if result_text is None:
            return None
        if payload.get("is_error") is True or _is_interruption_text(result_text):
            return result_text
        return None

    if payload_type == "rate_limit_event":
        rate_limit_info = payload.get("rate_limit_info")
        if isinstance(rate_limit_info, dict) and rate_limit_info.get("status") == "rejected":
            rate_limit_type = _clean_optional_text(rate_limit_info.get("rateLimitType"))
            if rate_limit_type:
                return f"Rate limit rejected ({rate_limit_type})."
            return "Rate limit rejected."

    if payload_type == "user":
        for text in _iter_content_texts(
            payload.get("message", {}).get("content")
            if isinstance(payload.get("message"), dict)
            else None
        ):
            if _is_interruption_text(text):
                return text

    return None


def _iter_content_texts(content: object) -> tuple[str, ...]:
    texts: list[str] = []
    for item in _iter_content_items(content):
        if item.get("type") != "text":
            continue
        text = _clean_optional_text(item.get("text"))
        if text:
            texts.append(text)
    if isinstance(content, str):
        text = _clean_optional_text(content)
        if text:
            texts.append(text)
    return tuple(texts)


def _iter_content_items(content: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(content, list):
        return ()
    return tuple(item for item in content if isinstance(item, dict))


def _render_handoff_summary(
    *,
    messages: list[str],
    fallback_user_messages: list[str],
    tasks: tuple[_TaskSnapshot, ...],
    todo_snapshot: _TodoSnapshot | None,
    interruption_message: str | None,
    provider_hint: str | None,
) -> str | None:
    del provider_hint
    message_pool = _dedupe_messages(messages) or _dedupe_messages(fallback_user_messages)
    lines: list[str] = []

    state = _pick_state_message(message_pool)
    if state:
        lines.append(f"- State: {state}")

    checkpoint = _pick_verification_checkpoint(message_pool)
    if checkpoint and checkpoint != state:
        lines.append(f"- Progress checkpoint: {checkpoint}")

    if todo_snapshot is not None:
        todo_line = _render_todo_summary(todo_snapshot)
        if todo_line:
            lines.append(f"- Checklist: {todo_line}")

    task_line = _render_task_summary(tasks)
    if task_line:
        lines.append(f"- In-flight task: {task_line}")

    last_activity = _pick_last_activity(message_pool)
    if (
        last_activity
        and last_activity not in {state, checkpoint}
        and (task_line is None or _is_actionable_recovery_message(last_activity))
    ):
        lines.append(f"- Last activity: {last_activity}")

    if interruption_message:
        lines.append(f"- Interruption: {interruption_message}")

    if not lines:
        return None

    summary = "Previous iteration summary:\n" + "\n".join(lines)
    return _truncate_to_word_limit(summary, max_words=_MAX_SUMMARY_WORDS)


def _pick_state_message(messages: list[str]) -> str | None:
    if not messages:
        return None

    terminal = _pick_best_message(messages, _is_terminal_state_message)
    if terminal is not None:
        return terminal

    recovery = _pick_best_message(
        messages,
        _is_actionable_recovery_message,
        priority=_recovery_message_priority,
    )
    if recovery is not None:
        return recovery

    next_work = _pick_best_message(messages, _is_explicit_next_work_message)
    if next_work is not None:
        return next_work

    non_generic = _pick_best_message(
        messages,
        lambda text: not _is_generic_context_progress_message(text),
        priority=_state_message_priority,
    )
    if non_generic is not None:
        return non_generic

    return _truncate_text(messages[-1], max_chars=_MAX_TEXT_SNIPPET_CHARS)


def _pick_best_message(
    messages: list[str],
    predicate: Callable[[str], bool],
    *,
    priority: Callable[[str], float] | None = None,
) -> str | None:
    best_index: int | None = None
    best_score = float("-inf")
    for index, text in enumerate(messages):
        if not predicate(text):
            continue
        score = priority(text) if priority is not None else 0.0
        score += min(index, 40) * 0.01
        if score >= best_score:
            best_score = score
            best_index = index
    if best_index is None:
        return None
    return _truncate_text(messages[best_index], max_chars=_MAX_TEXT_SNIPPET_CHARS)


def _pick_verification_checkpoint(messages: list[str]) -> str | None:
    for text in reversed(messages):
        lowered = text.lower()
        if any(token in lowered for token in ("test", "tests", "lint", "typecheck", "verification")):
            if any(
                token in lowered
                for token in ("pass", "passed", "clean", "blocked", "caught", "failed")
            ):
                return _truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS)
    return None


def _pick_last_activity(messages: list[str]) -> str | None:
    for text in reversed(messages):
        if _is_interruption_text(text):
            continue
        return _truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS)
    return None


def _render_todo_summary(todo_snapshot: _TodoSnapshot) -> str | None:
    if todo_snapshot.total <= 0:
        return None

    if todo_snapshot.pending_texts:
        pending = "; ".join(todo_snapshot.pending_texts[:2])
        return (
            f"{todo_snapshot.completed}/{todo_snapshot.total} complete; remaining: "
            f"{_truncate_text(pending, max_chars=_MAX_TEXT_SNIPPET_CHARS)}"
        )

    return f"All {todo_snapshot.total} items were complete."


def _render_task_summary(tasks: tuple[_TaskSnapshot, ...]) -> str | None:
    for snapshot in reversed(tasks):
        description = _truncate_text(snapshot.description, max_chars=120)
        prompt = _compact_task_prompt(snapshot.prompt)
        if description and prompt:
            if description.lower() in prompt.lower():
                return prompt
            return _truncate_text(
                f"{description}: {prompt}",
                max_chars=_MAX_TASK_SNIPPET_CHARS,
            )
        if prompt:
            return prompt
        if description:
            return description
    return None


def _render_task_summaries(tasks: tuple[_TaskSnapshot, ...]) -> tuple[str, ...]:
    rendered: list[str] = []
    for snapshot in tasks[-5:]:
        summary = _render_task_summary((snapshot,))
        if summary:
            rendered.append(summary)
    return tuple(rendered)


def _pick_retry_next_step(
    *,
    message_pool: list[str],
    tasks: tuple[_TaskSnapshot, ...],
    todo_snapshot: _TodoSnapshot | None,
) -> str | None:
    if todo_snapshot is not None and todo_snapshot.pending_texts:
        return _truncate_text(
            todo_snapshot.pending_texts[0],
            max_chars=_MAX_TEXT_SNIPPET_CHARS,
        )

    for text in reversed(message_pool):
        if (
            _is_actionable_recovery_message(text)
            or _is_terminal_state_message(text)
            or _is_explicit_next_work_message(text)
        ):
            return _truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS)

    task_summary = _render_task_summary(tasks)
    if task_summary is not None:
        return task_summary

    for text in reversed(message_pool):
        if not _is_interruption_text(text) and not _is_generic_context_progress_message(text):
            return _truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS)

    return _pick_last_activity(message_pool)


def _compact_task_prompt(prompt: str | None) -> str | None:
    if not prompt:
        return None

    text = _clean_text(prompt)
    text = re.sub(r"^You are [^.]+?\.\s*", "", text, flags=re.IGNORECASE)
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return _truncate_text(text, max_chars=_MAX_TASK_SNIPPET_CHARS)

    selected: list[str] = [sentences[0]]
    for sentence in sentences[1:4]:
        lowered = sentence.lower()
        if any(token in lowered for token in ("focus", "check", "flag", "specifically")):
            selected.append(sentence)
            break
        if len(selected) == 1 and len(selected[0]) < 100:
            selected.append(sentence)
            break

    return _truncate_text(" ".join(selected), max_chars=_MAX_TASK_SNIPPET_CHARS)


def _dedupe_messages(messages: list[str]) -> list[str]:
    deduped: list[str] = []
    for text in messages:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        if deduped and deduped[-1] == cleaned:
            continue
        deduped.append(cleaned)
    return deduped


def _state_message_priority(text: str) -> float:
    lowered = text.lower()
    score = min(len(text), 240) / 120
    if _is_explicit_next_work_message(text):
        score += 7
    if any(token in lowered for token in ("plan is", "switching to", "the queue confirms")):
        score += 5
    if any(token in lowered for token in ("phase ", "wire ", "implement", "add ", "update ", "fix ")):
        score += 2
    if any(token in lowered for token in ("checking the repo state", "reading", "let me explore")):
        score -= 1
    return score


def _is_terminal_state_message(text: str) -> bool:
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "working tree clean",
            "worktree is clean",
            "nothing further to do",
            "no further action",
        )
    ):
        return True
    return any(
        re.search(pattern, lowered) is not None
        for pattern in (
            r"\b(is|are|was|were) shipped\b",
            r"\bshipped and committed\b",
            r"\b(is|are|was|were) committed\b",
            r"\bcommitted\b",
            r"\b(is|are|was|were) implemented\b",
            r"\bimplemented and committed\b",
            r"\b(is|are|was|were) landed\b",
            r"\blanded and committed\b",
        )
    )


def _is_explicit_next_work_message(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "next recommended task",
            "next leaf task",
            "next task is",
            "current hot task",
            "queue points to",
            "queue confirms",
            "next up",
            "resume point",
        )
    )


def _recovery_message_priority(text: str) -> float:
    lowered = text.lower()
    if any(token in lowered for token in ("reviewer flagged", "review flagged", "blocker")):
        return 4.0
    if any(token in lowered for token in ("commit failed", "failed to commit")):
        return 4.0
    if any(
        token in lowered
        for token in (
            "failed verification",
            "verification failed",
            "tests failed",
            "test failed",
        )
    ):
        return 3.0
    if any(token in lowered for token in ("apply", "fix")):
        return 1.0
    return 2.0


def _is_actionable_recovery_message(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "reviewer flagged",
            "review flagged",
            "flagged two",
            "blocker",
            "blocked",
            "commit failed",
            "failed to commit",
            "failed verification",
            "verification failed",
            "tests failed",
            "test failed",
            "now i'll apply",
            "now i will apply",
            "apply fixes",
            "apply the fix",
            "apply the fixes",
            "apply the two fixes",
            "fix the blocker",
            "fix the blockers",
            "wait for git commit",
            "all tests pass. now commit",
            "all verification passes. now commit",
        )
    )


def _is_generic_context_progress_message(text: str) -> bool:
    lowered = text.lower()
    if (
        _is_terminal_state_message(text)
        or _is_actionable_recovery_message(text)
        or _is_explicit_next_work_message(text)
        or "plan is" in lowered
    ):
        return False
    return any(
        token in lowered
        for token in (
            "now i have enough context",
            "i have enough context",
            "now i have the full picture",
            "now i have a full picture",
            "now i have the complete picture",
            "now i have a complete picture",
            "now i have what i need",
            "now i have enough context. let me start implementing",
            "now i have enough context. let me start the implementation",
            "now i have a complete picture. let me start implementing",
            "now i have a complete picture. let me start the implementation",
            "now i have what i need. let me create",
            "now let me check",
            "now let me read",
            "now let me open",
            "now let me look",
            "now let me grep",
            "let me grep",
            "i'll start by reading",
            "i will start by reading",
            "i’ll start by reading",
        )
    )


def _is_interruption_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _INTERRUPTION_PATTERNS)


def _is_generic_success_result_text(text: str) -> bool:
    return text.lower() in _GENERIC_SUCCESS_RESULTS


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = _clean_text(value)
    return text or None


def _truncate_text(text: str | None, *, max_chars: int) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


def _truncate_to_word_limit(text: str, *, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "…"


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

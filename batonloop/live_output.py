from __future__ import annotations

import json
import logging
import re
from typing import Any

_MAX_TEXT_SNIPPET_CHARS = 240
_MAX_TASK_SNIPPET_CHARS = 320
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
_NOISE_PLAIN_LINES = {
    "Reading prompt from stdin...",
}
_PLAIN_ERROR_HINTS = (
    "error",
    "failed",
    "unauthorized",
    "authentication",
    "not logged in",
    "login required",
    "permission denied",
    "rate limit",
    "quota exceeded",
    "too many requests",
    "temporarily unavailable",
)


class LiveOutputConsumer:
    def __init__(self, logger: logging.Logger, provider_name: str) -> None:
        self._logger = logger
        self._provider_name = provider_name
        self._last_emitted: str | None = None
        self._last_todo: str | None = None
        self._last_task: str | None = None
        self._last_interruption: str | None = None

    def consume_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        payload = _parse_payload(line)
        if payload is None:
            self._consume_plain_line(line)
            return

        for text in _extract_progress_messages(payload):
            self._emit(text)

        todo_summary = _extract_todo_summary(payload)
        if todo_summary and todo_summary != self._last_todo:
            self._emit(f"Checklist: {todo_summary}")
            self._last_todo = todo_summary

        task_summary = _extract_task_summary(payload)
        if task_summary and task_summary != self._last_task:
            self._emit(f"Task: {task_summary}")
            self._last_task = task_summary

        interruption = _extract_interruption_message(payload)
        if interruption and interruption != self._last_interruption:
            self._emit(f"Interruption: {interruption}")
            self._last_interruption = interruption

    def _consume_plain_line(self, line: str) -> None:
        cleaned = _clean_text(line)
        if not cleaned or cleaned in _NOISE_PLAIN_LINES:
            return

        if _is_interruption_text(cleaned):
            if cleaned != self._last_interruption:
                self._emit(f"Interruption: {cleaned}")
                self._last_interruption = cleaned
            return

        lowered = cleaned.lower()
        if any(token in lowered for token in _PLAIN_ERROR_HINTS):
            self._emit(cleaned)

    def _emit(self, text: str) -> None:
        cleaned = _clean_text(text)
        if not cleaned or cleaned == self._last_emitted:
            return
        self._logger.info("[%s] %s", self._provider_name, cleaned)
        self._last_emitted = cleaned


def _parse_payload(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
            return (_truncate_text(_clean_text(item["text"]), max_chars=_MAX_TEXT_SNIPPET_CHARS),)

    if payload_type == "result":
        result_text = _clean_optional_text(payload.get("result"))
        if (
            result_text is not None
            and payload.get("is_error") is not True
            and not _is_interruption_text(result_text)
            and not _is_generic_success_result_text(result_text)
        ):
            return (_truncate_text(result_text, max_chars=_MAX_TEXT_SNIPPET_CHARS),)

    return ()


def _extract_todo_summary(payload: dict[str, object]) -> str | None:
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

    if pending_texts:
        pending = "; ".join(pending_texts[:2])
        return (
            f"{completed}/{total} complete; remaining: "
            f"{_truncate_text(pending, max_chars=_MAX_TEXT_SNIPPET_CHARS)}"
        )

    return f"All {total} items were complete."


def _extract_task_summary(payload: dict[str, object]) -> str | None:
    payload_type = payload.get("type")

    if payload_type == "system" and payload.get("subtype") == "task_started":
        return _render_task_summary(
            _clean_optional_text(payload.get("description")),
            _clean_optional_text(payload.get("prompt")),
        )

    if payload_type == "assistant":
        message = payload.get("message")
        if isinstance(message, dict):
            for item in _iter_content_items(message.get("content")):
                if item.get("type") != "tool_use":
                    continue
                input_payload = item.get("input")
                if not isinstance(input_payload, dict):
                    continue
                description = _clean_optional_text(input_payload.get("description"))
                prompt = _clean_optional_text(input_payload.get("prompt"))
                if description or prompt:
                    return _render_task_summary(description, prompt)

    if payload_type in {"item.started", "item.completed"}:
        item = payload.get("item")
        if (
            isinstance(item, dict)
            and item.get("type") == "collab_tool_call"
            and item.get("tool") in {"spawn_agent", "send_input"}
        ):
            description = _clean_optional_text(item.get("tool"))
            prompt = _clean_optional_text(item.get("prompt"))
            return _render_task_summary(description, prompt)

    return None


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
        message = payload.get("message")
        if isinstance(message, dict):
            for text in _iter_content_texts(message.get("content")):
                if _is_interruption_text(text):
                    return text

    return None


def _render_task_summary(description: str | None, prompt: str | None) -> str | None:
    description_text = _truncate_text(description, max_chars=120)
    prompt_text = _compact_task_prompt(prompt)
    if description_text and prompt_text:
        if description_text.lower() in prompt_text.lower():
            return prompt_text
        return _truncate_text(
            f"{description_text}: {prompt_text}",
            max_chars=_MAX_TASK_SNIPPET_CHARS,
        )
    if prompt_text:
        return prompt_text
    if description_text:
        return description_text
    return None


def _compact_task_prompt(prompt: str | None) -> str | None:
    if not prompt:
        return None

    text = _clean_text(prompt)
    text = re.sub(r"^You are [^.]+?\.\s*", "", text, flags=re.IGNORECASE)
    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", text)
        if segment.strip()
    ]
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


def _iter_content_texts(content: object) -> tuple[str, ...]:
    texts: list[str] = []
    for item in _iter_content_items(content):
        if item.get("type") != "text":
            continue
        text = _clean_optional_text(item.get("text"))
        if text:
            texts.append(_truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS))
    if isinstance(content, str):
        text = _clean_optional_text(content)
        if text:
            texts.append(_truncate_text(text, max_chars=_MAX_TEXT_SNIPPET_CHARS))
    return tuple(texts)


def _iter_content_items(content: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(content, list):
        return ()
    return tuple(item for item in content if isinstance(item, dict))


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
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."

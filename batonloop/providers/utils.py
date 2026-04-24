from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True, slots=True)
class FailureLogSummary:
    status_codes: frozenset[int]
    error_codes: frozenset[str]
    texts: tuple[str, ...]
    rate_limit_rejected: bool = False

    def has_status(self, *codes: int) -> bool:
        return any(code in self.status_codes for code in codes)

    def has_error_code(self, *codes: str) -> bool:
        normalized = {code.lower() for code in codes}
        return not self.error_codes.isdisjoint(normalized)

    def matches_text(self, patterns: Iterable[str]) -> bool:
        return any(pattern in text for pattern in patterns for text in self.texts)


def read_log_text(log_path: Path, *, lower: bool = False) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if lower:
        return text.lower()
    return text


def decimal_from_value(value: object) -> Decimal:
    if value in (None, "", "null"):
        return Decimal("0")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def iter_jsonl(log_path: Path) -> Iterator[dict[str, Any]]:
    try:
        with log_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


def read_jsonl_failure_summary(log_path: Path) -> FailureLogSummary:
    return summarize_failure_payloads(iter_jsonl(log_path))


def summarize_failure_payloads(payloads: Iterable[dict[str, Any]]) -> FailureLogSummary:
    status_codes: set[int] = set()
    error_codes: set[str] = set()
    texts: list[str] = []
    rate_limit_rejected = False

    for payload in payloads:
        if _collect_payload_failure_data(payload, status_codes, error_codes, texts):
            rate_limit_rejected = True

    return FailureLogSummary(
        status_codes=frozenset(status_codes),
        error_codes=frozenset(error_codes),
        texts=tuple(texts),
        rate_limit_rejected=rate_limit_rejected,
    )


def _collect_payload_failure_data(
    payload: dict[str, Any],
    status_codes: set[int],
    error_codes: set[str],
    texts: list[str],
) -> bool:
    rejected_rate_limit = False
    payload_type = _clean_optional_text(payload.get("type"))

    if payload_type == "rate_limit_event":
        rate_limit_info = payload.get("rate_limit_info")
        if isinstance(rate_limit_info, dict):
            status = _clean_optional_text(rate_limit_info.get("status"))
            if status == "rejected":
                rejected_rate_limit = True

    if payload_type == "error":
        _collect_message_text(payload.get("message"), status_codes, error_codes, texts)
    elif payload_type == "turn.failed":
        _collect_failure_object(payload.get("error"), status_codes, error_codes, texts)
    elif payload_type == "result":
        _collect_status_code(payload.get("api_error_status"), status_codes)
        if payload.get("is_error") is True:
            _collect_message_text(payload.get("result"), status_codes, error_codes, texts)
    elif payload_type is None:
        _collect_message_text(payload.get("message"), status_codes, error_codes, texts)
        if (
            payload.get("is_error") is True
            or payload.get("api_error_status") is not None
            or payload.get("error") is not None
        ):
            _collect_message_text(payload.get("result"), status_codes, error_codes, texts)

    top_level_error = payload.get("error")
    if payload_type not in {"turn.failed"} and top_level_error is not None:
        _collect_error_value(top_level_error, status_codes, error_codes, texts)

    if payload_type != "result":
        _collect_status_code(payload.get("api_error_status"), status_codes)

    return rejected_rate_limit


def _collect_failure_object(
    value: object,
    status_codes: set[int],
    error_codes: set[str],
    texts: list[str],
    *,
    allow_error_fields: bool = False,
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"status", "api_error_status"}:
                _collect_status_code(nested, status_codes)
            elif key in {"message", "result"}:
                _collect_message_text(nested, status_codes, error_codes, texts)
            elif key == "error":
                _collect_error_value(nested, status_codes, error_codes, texts)
            elif allow_error_fields and key in {"type", "code"}:
                _collect_error_code(nested, error_codes)
            elif isinstance(nested, (dict, list)):
                _collect_failure_object(
                    nested,
                    status_codes,
                    error_codes,
                    texts,
                    allow_error_fields=allow_error_fields,
                )
        return

    if isinstance(value, list):
        for item in value:
            _collect_failure_object(
                item,
                status_codes,
                error_codes,
                texts,
                allow_error_fields=allow_error_fields,
            )


def _collect_error_value(
    value: object,
    status_codes: set[int],
    error_codes: set[str],
    texts: list[str],
) -> None:
    if isinstance(value, str):
        _collect_error_code(value, error_codes)
        embedded_payload = _parse_embedded_json(value)
        if embedded_payload is not None:
            _collect_failure_object(
                embedded_payload,
                status_codes,
                error_codes,
                texts,
            )
        return

    _collect_failure_object(value, status_codes, error_codes, texts, allow_error_fields=True)


def _collect_message_text(
    value: object,
    status_codes: set[int],
    error_codes: set[str],
    texts: list[str],
) -> None:
    text = _clean_optional_text(value)
    if text is None:
        return
    texts.append(text)
    embedded_payload = _parse_embedded_json(text)
    if embedded_payload is not None:
        _collect_failure_object(embedded_payload, status_codes, error_codes, texts)


def _collect_error_code(value: object, error_codes: set[str]) -> None:
    text = _clean_optional_text(value)
    if text is None:
        return
    error_codes.add(text)


def _collect_status_code(value: object, status_codes: set[int]) -> None:
    status = _coerce_int(value)
    if status is None:
        return
    status_codes.add(status)


def _parse_embedded_json(value: str) -> dict[str, Any] | None:
    if not value.startswith("{"):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    return text


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None

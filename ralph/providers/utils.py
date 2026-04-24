from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


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

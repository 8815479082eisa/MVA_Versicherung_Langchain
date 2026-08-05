from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]+=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.IGNORECASE)


def sanitize_diagnostic_text(value: Any) -> str:
    text = str(value or "")
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub("[SECRET_REDACTED]", text)
    text = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    text = _IBAN_PATTERN.sub("[IBAN_REDACTED]", text)
    return text


def sanitize_diagnostic_value(value: Any, *, key: str = "") -> Any:
    normalized_key = key.lower().replace("-", "_")
    key_parts = set(normalized_key.split("_"))
    if normalized_key in _SECRET_KEYS or (
        "api" in key_parts and "key" in key_parts
    ) or "password" in key_parts or "authorization" in key_parts or "secret" in key_parts:
        return "[SECRET_REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_diagnostic_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_diagnostic_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_diagnostic_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_diagnostic_text(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_diagnostic_value(payload)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(sanitized, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

"""Shared helpers for security-focused benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from config.models import load_model_settings
    from core.safety_adapter import create_safety_checker
except Exception:
    from src.config.models import load_model_settings
    from src.core.safety_adapter import create_safety_checker


def load_security_cases(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def derive_case_label(case: Dict[str, Any]) -> str:
    explicit = str(case.get("label") or "").strip().lower()
    if explicit in {"attack", "benign"}:
        return explicit

    expected = str(case.get("expected_decision") or "").strip().lower()
    if expected in {"allow", "review_or_allow"}:
        return "benign"
    return "attack"


def evaluate_security_cases(
    cases: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    settings = load_model_settings()
    checker = create_safety_checker(settings.safety)

    rows: List[Dict[str, Any]] = []
    total_attack = 0
    total_benign = 0
    stopped_attack = 0
    allowed_benign = 0

    for case in cases:
        query = str(case.get("query") or "").strip()
        label = derive_case_label(case)
        result = checker.check_query_safety(query, [])

        details = dict(result.details or {})
        decision_owner = details.get("pre_query_decision_owner")
        fallback_used = bool(details.get("legacy_fallback_used", False))

        stopped = (not result.allow) or (result.action in {"block", "fallback"})
        allowed = result.allow and result.action == "allow"

        attack_block = None
        benign_allow = None

        if label == "attack":
            total_attack += 1
            attack_block = 1.0 if stopped else 0.0
            if stopped:
                stopped_attack += 1
        else:
            total_benign += 1
            benign_allow = 1.0 if allowed else 0.0
            if allowed:
                allowed_benign += 1

        rows.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "label": label,
                "expected_decision": case.get("expected_decision"),
                "query": query,
                "allow": result.allow,
                "action": result.action,
                "risk_level": result.risk_level,
                "reasons": list(result.reasons),
                "scores": dict(result.scores),
                "decision_owner": decision_owner,
                "legacy_fallback_used": fallback_used,
                "details": details,
                "metrics": {
                    "attack_block": attack_block,
                    "benign_allow": benign_allow,
                },
            }
        )

    summary = summarize_security_rows(rows)
    metadata = {
        "scope": "input_query_safety",
        "metrics": ["attack_block_rate", "benign_allow_rate"],
        "attack_stop_definition": ["block", "fallback", "allow=false"],
        "benign_allow_definition": ["allow=true", "action=allow"],
        "safety_backend": settings.safety.backend,
        "safety_mode": settings.safety.mode,
    }
    return rows, summary, metadata


def summarize_security_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    attack_metrics = [
        float(row["metrics"]["attack_block"])
        for row in rows
        if row.get("metrics", {}).get("attack_block") is not None
    ]
    benign_metrics = [
        float(row["metrics"]["benign_allow"])
        for row in rows
        if row.get("metrics", {}).get("benign_allow") is not None
    ]

    false_negative_count = sum(1 for value in attack_metrics if value < 1.0)
    false_positive_count = sum(1 for value in benign_metrics if value < 1.0)

    return {
        "total_queries": len(rows),
        "total_attack": len(attack_metrics),
        "total_benign": len(benign_metrics),
        "stopped_attack": sum(1 for value in attack_metrics if value >= 1.0),
        "allowed_benign": sum(1 for value in benign_metrics if value >= 1.0),
        "attack_block_rate": sum(attack_metrics) / len(attack_metrics) if attack_metrics else 0.0,
        "benign_allow_rate": sum(benign_metrics) / len(benign_metrics) if benign_metrics else 0.0,
        "false_negative_count": false_negative_count,
        "false_positive_count": false_positive_count,
    }


def write_security_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

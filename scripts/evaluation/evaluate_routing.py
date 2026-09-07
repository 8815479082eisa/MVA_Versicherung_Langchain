"""Evaluate the four-way insurance request routing benchmark.

The local mode reproduces the deterministic API front door: hard safety and
secret-request checks run first, followed by the lexical insurance router.
The API mode sends the same cases to ``/api/ask`` and reports routing accuracy
separately from operational pipeline success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.insurance_tool_routing import plan_insurance_query
from src.core.safety_audit import (
    detect_query_hard_injection_signals,
    detect_system_secret_request_signals,
)


DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "benchmarks" / "routing" / "routing_eval_80.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "routing-evaluation"
ROUTES = ("crm_only", "retrieval_only", "combined", "denied")


def normalize_route(value: Any) -> str:
    """Convert API and enum spellings to the benchmark's canonical labels."""

    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "crm": "crm_only",
        "rag": "retrieval_only",
        "rag_only": "retrieval_only",
        "retrieval": "retrieval_only",
    }
    return aliases.get(normalized, normalized) or "unknown"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            case["_line_number"] = line_number
            cases.append(case)
    return cases


def predict_local_route(question: str) -> tuple[str, str, list[str]]:
    """Reproduce routing up to the point where external services are called."""

    secret_signals = detect_system_secret_request_signals(question)
    injection_signals = detect_query_hard_injection_signals(question)
    signals = list(dict.fromkeys([*secret_signals, *injection_signals]))
    if signals:
        return "denied", "safety_precheck", signals

    plan = plan_insurance_query(question)
    return normalize_route(plan.mode.value), "route_planner", [plan.reason]


def evaluate_local(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        actual_route, actual_stage, reasons = predict_local_route(case["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        expected_route = normalize_route(case["expected_route"])
        route_correct = actual_route == expected_route
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "subcategory": case["subcategory"],
                "expected_route": expected_route,
                "actual_route": actual_route,
                "expected_stage": case.get("expected_stage"),
                "actual_stage": actual_stage,
                "route_correct": route_correct,
                "stage_correct": actual_stage == case.get("expected_stage"),
                "downstream_operational_success": True,
                "operational_success": route_correct,
                "status_code": None,
                "latency_ms": round(latency_ms, 4),
                "reasons": reasons,
                "error": None,
            }
        )
    return rows


def _extract_route(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    candidates: list[Any] = [payload.get("route")]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        candidates.extend((detail.get("route"), detail.get("diagnostics", {}).get("route") if isinstance(detail.get("diagnostics"), dict) else None))
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        candidates.append(diagnostics.get("route"))
    for candidate in candidates:
        route = normalize_route(candidate)
        if route in ROUTES:
            return route
    return "unknown"


def _api_operational_success(expected_route: str, status_code: int) -> bool:
    if expected_route == "denied":
        return status_code in {200, 403}
    return status_code in {200, 206}


def normalize_saved_operational_fields(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upgrade saved result rows to the strict end-to-end metric definition."""
    normalized_rows: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        status_code = row.get("status_code")
        if status_code is None:
            downstream_success = True
        else:
            downstream_success = _api_operational_success(
                row["expected_route"], int(status_code)
            )
        row["downstream_operational_success"] = downstream_success
        row["operational_success"] = bool(
            row.get("route_correct") and downstream_success
        )
        normalized_rows.append(row)
    return normalized_rows


def evaluate_api(
    cases: Sequence[dict[str, Any]],
    *,
    endpoint: str,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    health: dict[str, Any] = {}
    base_url = endpoint.rsplit("/api/ask", 1)[0]

    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            response = client.get(f"{base_url}/health")
            health = {
                "reachable": True,
                "status_code": response.status_code,
                "payload": response.json(),
            }
        except Exception as exc:  # pragma: no cover - depends on live service
            health = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}

        for case in cases:
            expected_route = normalize_route(case["expected_route"])
            started = time.perf_counter()
            status_code = 0
            payload: Any = {}
            error: str | None = None
            try:
                response = client.post(
                    endpoint,
                    json={"question": case["question"]},
                )
                status_code = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"raw_text": response.text[:1000]}
            except Exception as exc:  # pragma: no cover - depends on live service
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000
            actual_route = _extract_route(payload)
            downstream_operational_success = _api_operational_success(
                expected_route, status_code
            )
            route_correct = actual_route == expected_route
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "category": case["category"],
                    "subcategory": case["subcategory"],
                    "expected_route": expected_route,
                    "actual_route": actual_route,
                    "expected_stage": case.get("expected_stage"),
                    "actual_stage": None,
                    "route_correct": route_correct,
                    "stage_correct": None,
                    "downstream_operational_success": downstream_operational_success,
                    # Strict end-to-end success requires both correct routing and
                    # a successful downstream HTTP outcome.
                    "operational_success": (
                        route_correct and downstream_operational_success
                    ),
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 4),
                    "error": error,
                    "response_status": payload.get("status") if isinstance(payload, dict) else None,
                    "response_error_code": (
                        payload.get("detail", {}).get("errorCode")
                        if isinstance(payload, dict)
                        and isinstance(payload.get("detail"), dict)
                        else None
                    ),
                }
            )
    return rows, health


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_counts = Counter(row["expected_route"] for row in rows)
    actual_counts = Counter(row["actual_route"] for row in rows)
    status_counts = Counter(str(row.get("status_code")) for row in rows)
    operational_error_counts = Counter(
        str(row.get("response_error_code") or row.get("error") or row.get("status_code"))
        for row in rows
        if not row.get(
            "downstream_operational_success",
            row.get("operational_success"),
        )
    )
    confusion: dict[str, dict[str, int]] = {
        expected: {actual: 0 for actual in (*ROUTES, "unknown")}
        for expected in ROUTES
    }
    for row in rows:
        expected = row["expected_route"]
        actual = row["actual_route"]
        confusion.setdefault(expected, {}).setdefault(actual, 0)
        confusion[expected][actual] += 1

    per_route: dict[str, Any] = {}
    for route in ROUTES:
        true_positive = sum(
            1
            for row in rows
            if row["expected_route"] == route and row["actual_route"] == route
        )
        false_positive = sum(
            1
            for row in rows
            if row["expected_route"] != route and row["actual_route"] == route
        )
        false_negative = sum(
            1
            for row in rows
            if row["expected_route"] == route and row["actual_route"] != route
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_route[route] = {
            "support": expected_counts[route],
            "predicted": actual_counts[route],
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "operational_success": sum(
                1
                for row in rows
                if row["expected_route"] == route and row.get("operational_success")
            ),
            "operational_success_rate": _safe_ratio(
                sum(
                    1
                    for row in rows
                    if row["expected_route"] == route
                    and row.get("operational_success")
                ),
                expected_counts[route],
            ),
            "downstream_operational_success": sum(
                1
                for row in rows
                if row["expected_route"] == route
                and row.get(
                    "downstream_operational_success",
                    row.get("operational_success"),
                )
            ),
            "downstream_operational_success_rate": _safe_ratio(
                sum(
                    1
                    for row in rows
                    if row["expected_route"] == route
                    and row.get(
                        "downstream_operational_success",
                        row.get("operational_success"),
                    )
                ),
                expected_counts[route],
            ),
        }

    correct = sum(bool(row["route_correct"]) for row in rows)
    operational = sum(bool(row["operational_success"]) for row in rows)
    downstream_operational = sum(
        bool(row.get("downstream_operational_success", row["operational_success"]))
        for row in rows
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    stage_rows = [row for row in rows if row.get("stage_correct") is not None]
    return {
        "total": len(rows),
        "correct": correct,
        "routing_accuracy": _safe_ratio(correct, len(rows)),
        "macro_f1": _safe_ratio(
            sum(values["f1"] for values in per_route.values()), len(ROUTES)
        ),
        "operational_success_count": operational,
        "operational_success_rate": _safe_ratio(operational, len(rows)),
        "downstream_operational_success_count": downstream_operational,
        "downstream_operational_success_rate": _safe_ratio(
            downstream_operational, len(rows)
        ),
        "stage_accuracy": (
            _safe_ratio(
                sum(bool(row["stage_correct"]) for row in stage_rows),
                len(stage_rows),
            )
            if stage_rows
            else None
        ),
        "mean_latency_ms": _safe_ratio(sum(latencies), len(latencies)),
        "expected_counts": dict(expected_counts),
        "actual_counts": dict(actual_counts),
        "http_status_counts": dict(status_counts),
        "operational_error_counts": dict(operational_error_counts),
        "per_route": per_route,
        "confusion_matrix": confusion,
        "misclassified_ids": [row["id"] for row in rows if not row["route_correct"]],
        "misclassifications": [
            {
                "id": row["id"],
                "question": row["question"],
                "expected_route": row["expected_route"],
                "actual_route": row["actual_route"],
                "status_code": row.get("status_code"),
                "response_error_code": row.get("response_error_code"),
            }
            for row in rows
            if not row["route_correct"]
        ],
        "operational_failure_ids": [
            row["id"] for row in rows if not row["operational_success"]
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Routing Evaluation Report",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Dataset: `{payload['dataset']}`",
        f"Cases: {payload['dataset_case_count']} (20 per route)",
        "",
        "## Method",
        "",
        "- `LOCAL` reproduces deterministic safety prechecks and the lexical route planner without external services.",
        "- `API` calls `/api/ask`; route correctness is measured independently from downstream operational success.",
        "- A denied request is operationally successful with HTTP 200 (safety fallback) or 403 (planner denial).",
        "- Non-denied requests are operationally successful with HTTP 200 or 206.",
        "",
    ]
    for mode in ("local", "api"):
        result = payload.get(mode)
        if not result:
            continue
        summary = result["summary"]
        lines.extend(
            [
                f"## {mode.upper()} evaluation",
                "",
                f"- Cases: {summary['total']}",
                f"- Routing accuracy: {_percent(summary['routing_accuracy'])}",
                f"- Macro F1: {_percent(summary['macro_f1'])}",
                f"- Operational success rate: {_percent(summary['operational_success_rate'])}",
                f"- Downstream operational completion: {_percent(summary['downstream_operational_success_rate'])}",
                f"- Stage accuracy: {_percent(summary['stage_accuracy'])}",
                f"- Mean latency: {summary['mean_latency_ms']:.3f} ms",
                "",
                "### Per-route metrics",
                "",
                "| Route | Support | Correct | Precision | Recall | F1 | Operational |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for route in ROUTES:
            metric = summary["per_route"][route]
            lines.append(
                f"| `{route}` | {metric['support']} | {metric['correct']} | "
                f"{_percent(metric['precision'])} | {_percent(metric['recall'])} | "
                f"{_percent(metric['f1'])} | {_percent(metric['operational_success_rate'])} |"
            )
        lines.extend(
            [
                "",
                "### Confusion matrix",
                "",
                "| Expected / Actual | crm_only | retrieval_only | combined | denied | unknown |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        matrix = summary["confusion_matrix"]
        for expected in ROUTES:
            lines.append(
                f"| `{expected}` | "
                + " | ".join(str(matrix[expected].get(actual, 0)) for actual in (*ROUTES, "unknown"))
                + " |"
            )
        lines.extend(
            [
                "",
                f"Misclassified IDs: {', '.join(summary['misclassified_ids']) or 'none'}",
                "",
                f"Operational failure IDs: {', '.join(summary['operational_failure_ids']) or 'none'}",
                "",
                "### Operational error counts",
                "",
                "```json",
                json.dumps(summary.get("operational_error_counts", {}), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
        if mode == "api":
            lines.extend(
                [
                    "### API health snapshot",
                    "",
                    "```json",
                    json.dumps(result.get("health", {}), indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("local", "api", "both"), default="local")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/ask")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "routing_summary.json"
    payload: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(existing.get("dataset")) == str(args.dataset):
                payload = existing
        except (OSError, ValueError):
            payload = {}
    payload.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_case_count": len(cases),
    })

    if args.mode in {"local", "both"}:
        local_rows = evaluate_local(cases)
        write_jsonl(args.output_dir / "router_only_results.jsonl", local_rows)
        payload["local"] = {"summary": summarize_rows(local_rows)}

    if args.mode in {"api", "both"}:
        api_rows, health = evaluate_api(
            cases,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
        )
        write_jsonl(args.output_dir / "api_results.jsonl", api_rows)
        payload["api"] = {"summary": summarize_rows(api_rows), "health": health}

    # Recompute a preserved mode after evaluator changes without repeating live
    # requests. The row files are the canonical per-case evidence.
    if args.mode == "local" and (args.output_dir / "api_results.jsonl").is_file():
        api_rows = normalize_saved_operational_fields(
            load_cases(args.output_dir / "api_results.jsonl")
        )
        write_jsonl(args.output_dir / "api_results.jsonl", api_rows)
        previous_health = payload.get("api", {}).get("health", {})
        payload["api"] = {
            "summary": summarize_rows(api_rows),
            "health": previous_health,
        }
    if args.mode == "api" and (args.output_dir / "router_only_results.jsonl").is_file():
        local_rows = normalize_saved_operational_fields(
            load_cases(args.output_dir / "router_only_results.jsonl")
        )
        write_jsonl(args.output_dir / "router_only_results.jsonl", local_rows)
        payload["local"] = {"summary": summarize_rows(local_rows)}

    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "routing_report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

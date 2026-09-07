"""Audit a completed ``run_pdf_rag_200.py`` collection without API calls."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "benchmarks" / "pdf_rag" / "pdf_rag_200.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "pdf-rag-200-evaluation-sequential"
TOKEN_RE = re.compile(r"[a-z0-9]+")
FALLBACK = "I could not generate an answer that is sufficiently supported by the available documents."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(value: Any) -> set[str]:
    return set(TOKEN_RE.findall(str(value or "").casefold()))


def overlap_recall(reference: str, evidence: str) -> float | None:
    expected = tokens(reference)
    if not expected:
        return None
    return len(expected & tokens(evidence)) / len(expected)


def source_matches(source: Mapping[str, Any], expected: str) -> bool:
    expected = expected.casefold()
    return any(expected in str(source.get(key) or "").casefold() for key in ("documentTitle", "documentId", "source"))


def classify(row: Mapping[str, Any]) -> str:
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    answer = str(payload.get("answer") or "")
    if int(row.get("status_code") or 0) >= 500:
        detail = payload.get("detail") if isinstance(payload.get("detail"), Mapping) else {}
        return str(detail.get("errorCode") or "http_error")
    if answer.strip().casefold() == FALLBACK.casefold():
        return "groundedness_or_insufficient_evidence_fallback"
    if answer.startswith("I cannot include or expose personal or sensitive information"):
        return "safety_pii_false_positive_or_block"
    if not answer.strip():
        return "empty_answer"
    return "generated_answer"


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def metric_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["classification"] == "generated_answer"]
    all_scores = [float(row["groundedness_score"]) for row in rows if row.get("groundedness_score") is not None]
    return {
        "cases": len(rows),
        "generated_answer_cases": len(successful),
        "generated_answer_rate": len(successful) / len(rows) if rows else None,
        "fallback_or_block_rate": sum(row["classification"] != "generated_answer" for row in rows) / len(rows) if rows else None,
        "expected_source_hit_rate_all": mean(rows, "expected_source_found"),
        "expected_source_hit_rate_generated": mean(successful, "expected_source_found"),
        "expected_exact_page_hit_rate_all": mean(rows, "expected_page_found"),
        "expected_exact_page_hit_rate_generated": mean(successful, "expected_page_found"),
        "reference_context_recall_all": mean(rows, "reference_context_recall"),
        "reference_context_recall_generated": mean(successful, "reference_context_recall"),
        "reference_answer_recall_generated": mean(successful, "reference_token_recall"),
        "claim_support_generated": mean(successful, "claim_support_rate"),
        "groundedness_score_all": statistics.mean(all_scores) if all_scores else None,
        "groundedness_pass_rate_all": mean(rows, "groundedness_pass"),
        "groundedness_pass_rate_generated": mean(successful, "groundedness_pass"),
        "citation_presence_generated": sum(bool(row.get("citation_count")) for row in successful) / len(successful) if successful else None,
        "citation_link_precision_generated": mean(successful, "citation_link_precision"),
        "latency_ms_all_mean": mean(rows, "latency_ms"),
        "latency_ms_generated_mean": mean(successful, "latency_ms"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = {str(row["id"]): row for row in load_jsonl(args.dataset)}
    responses = load_jsonl(args.output_dir / "responses_full.jsonl")
    base_rows = load_jsonl(args.output_dir / "per_case_results.jsonl")
    by_id = {str(row["id"]): row for row in base_rows}
    audit_rows: list[dict[str, Any]] = []
    for response in responses:
        case = cases[str(response["id"])]
        evaluated = dict(by_id[str(response["id"])])
        payload = response.get("payload") if isinstance(response.get("payload"), Mapping) else {}
        sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
        snippets = " ".join(str(source.get("snippet") or "") for source in sources if isinstance(source, Mapping))
        evaluated["classification"] = classify(response)
        evaluated["error_code"] = ((payload.get("detail") or {}).get("errorCode") if isinstance(payload.get("detail"), Mapping) else None)
        evaluated["timeout_stage"] = ((payload.get("detail") or {}).get("diagnostics") or {}).get("timeoutStage") if isinstance(payload.get("detail"), Mapping) else evaluated.get("timeout_stage")
        evaluated["reference_context_recall"] = overlap_recall(str(case.get("reference_answer") or ""), snippets)
        evaluated["returned_pages"] = sorted({int(source.get("page")) for source in sources if isinstance(source, Mapping) and str(source.get("page") or "").isdigit()})
        audit_rows.append(evaluated)
    audit_rows.sort(key=lambda row: str(row["id"]))
    classes = Counter(row["classification"] for row in audit_rows)
    stages = Counter(str(row.get("timeout_stage")) for row in audit_rows if row.get("timeout_stage"))
    per_pdf: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped[str(cases[str(row["id"])] ["subcategory"])].append(row)
    for name, rows in sorted(grouped.items()):
        per_pdf[name] = metric_group(rows)
    latency = [float(row["latency_ms"]) for row in audit_rows if row.get("latency_ms") is not None]
    summary = {
        "method_label": "automated_pdf_reference_and_runtime_validation_audit",
        "human_validated": False,
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "total_cases": len(audit_rows),
        "overall": metric_group(audit_rows),
        "latency_ms": {
            "mean": statistics.mean(latency) if latency else None,
            "median": statistics.median(latency) if latency else None,
            "p90": pctl(latency, 0.90),
            "p95": pctl(latency, 0.95),
            "p99": pctl(latency, 0.99),
            "min": min(latency) if latency else None,
            "max": max(latency) if latency else None,
        },
        "classification_counts": dict(classes),
        "timeout_stage_counts": dict(stages),
        "per_pdf": per_pdf,
        "failure_cases": [
            {
                "id": row["id"],
                "source": cases[str(row["id"])]["expected_source"],
                "question": row["question"],
                "classification": row["classification"],
                "status_code": row["status_code"],
                "groundedness_score": row.get("groundedness_score"),
                "reference_context_recall": row.get("reference_context_recall"),
                "reference_token_recall": row.get("reference_token_recall"),
                "returned_pages": row.get("returned_pages"),
                "timeout_stage": row.get("timeout_stage"),
            }
            for row in audit_rows
            if row["classification"] != "generated_answer" or row.get("groundedness_pass") is False or not row.get("expected_source_found")
        ],
    }
    (args.output_dir / "detailed_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "detailed_audit_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

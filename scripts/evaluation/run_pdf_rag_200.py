"""Run and audit the 200-case PDF-only RAG benchmark.

The runner intentionally keeps the full API payload for every case and writes
deterministic metrics that can be recomputed from the saved JSONL responses.
It does not call CRM and every benchmark case is expected to route to
``retrieval_only``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "benchmarks" / "pdf_rag" / "pdf_rag_200.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "pdf-rag-200-evaluation"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/ask"
FALLBACK = "I could not generate an answer that is sufficiently supported by the available documents."
GROUNDING_THRESHOLD = 0.7888
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\s*([^\],]+\.pdf)\s*,?\s*page\s+(\d+)\s*\]", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_route(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().casefold().replace("-", "_")


def _tokens(value: Any) -> list[str]:
    return _TOKEN_RE.findall(str(value or "").casefold())


def _token_recall(answer: str, reference: str) -> float | None:
    expected = set(_tokens(reference))
    if not expected:
        return None
    actual = set(_tokens(answer))
    return len(expected & actual) / len(expected)


def _statement_support(answer: str, snippets: Iterable[str]) -> tuple[float | None, int, int]:
    source_tokens = set(_tokens(" ".join(snippets)))
    statements = [part.strip() for part in _SENTENCE_RE.split(answer or "") if len(_tokens(part)) >= 4]
    if not statements:
        return None, 0, 0
    supported = sum(bool(set(_tokens(statement)) & source_tokens) for statement in statements)
    return supported / len(statements), supported, len(statements)


def _source_matches(source: Mapping[str, Any], expected_filename: str) -> bool:
    expected = expected_filename.casefold()
    return any(
        expected == str(source.get(key) or "").casefold()
        or expected in str(source.get(key) or "").casefold()
        for key in ("documentTitle", "documentId", "source", "name")
    )


def _citation_rows(answer: str) -> list[dict[str, Any]]:
    return [{"source": source.strip(), "page": int(page)} for source, page in _CITATION_RE.findall(answer or "")]


def _extract_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def _evaluate_row(case: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _extract_payload(row)
    answer = str(payload.get("answer") or "")
    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    expected_source = str(case["expected_source"])
    expected_page = int((case.get("source_pages") or [0])[0])
    matching_sources = [source for source in sources if isinstance(source, Mapping) and _source_matches(source, expected_source)]
    exact_page = any(int(source.get("page") or -1) == expected_page for source in matching_sources)
    route = payload.get("route") or ((payload.get("diagnostics") or {}).get("route") if isinstance(payload.get("diagnostics"), Mapping) else None)
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    evidence = diagnostics.get("evidence") if isinstance(diagnostics.get("evidence"), Mapping) else {}
    groundedness = evidence.get("groundedness") if isinstance(evidence.get("groundedness"), Mapping) else {}
    score = groundedness.get("score")
    if score is None:
        score = ((payload.get("diagnostics") or {}).get("evidence") or {}).get("groundedness", {}).get("score") if isinstance(payload.get("diagnostics"), Mapping) else None
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    snippets = [str(source.get("snippet") or "") for source in sources if isinstance(source, Mapping)]
    support_rate, supported_statements, statement_count = _statement_support(answer, snippets)
    citations = _citation_rows(answer)
    returned_markers = 0
    linked_markers = 0
    for citation in citations:
        returned_markers += 1
        if any(
            citation["source"].casefold() == str(source.get("documentTitle") or "").casefold()
            and int(source.get("page") or -1) == citation["page"]
            for source in sources
            if isinstance(source, Mapping)
        ):
            linked_markers += 1
    status = int(row.get("status_code") or 0)
    fallback = answer.strip().casefold() == FALLBACK.casefold()
    return {
        "id": case["id"],
        "question": case["question"],
        "subcategory": case.get("subcategory"),
        "expected_route": "retrieval_only",
        "actual_route": _normalize_route(route),
        "status_code": status,
        "latency_ms": row.get("latency_ms"),
        "answer": answer,
        "answer_present": bool(answer),
        "fallback": fallback,
        "source_count": len(sources),
        "expected_source_found": bool(matching_sources),
        "expected_page_found": exact_page,
        "reference_token_recall": _token_recall(answer, str(case.get("reference_answer") or "")),
        "citation_count": len(citations),
        "citation_link_precision": linked_markers / returned_markers if returned_markers else None,
        "groundedness_score": score,
        "groundedness_pass": score >= GROUNDING_THRESHOLD if score is not None else None,
        "claim_support_rate": support_rate,
        "supported_statement_count": supported_statements,
        "statement_count": statement_count,
        "timeout_stage": diagnostics.get("timeoutStage"),
        "operational_success": status in {200, 206},
        "route_correct": _normalize_route(route) == "retrieval_only",
        "error": row.get("collection_error"),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize(rows: list[dict[str, Any]], cases: list[dict[str, Any]], *, dataset: Path, output: Path, health: Mapping[str, Any], workers: int) -> dict[str, Any]:
    latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    substantive = [row for row in rows if row.get("operational_success")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subcategory") or "unknown")].append(row)
    per_pdf = {}
    for name, group in sorted(grouped.items()):
        per_pdf[name] = {
            "cases": len(group),
            "overall_operational_success": _mean(group, "operational_success"),
            "fallback_rate": _mean(group, "fallback"),
            "expected_source_hit_rate": _mean(group, "expected_source_found"),
            "expected_page_hit_rate": _mean(group, "expected_page_found"),
            "groundedness_pass_rate": _mean(group, "groundedness_pass"),
            "mean_groundedness": _mean(group, "groundedness_score"),
            "mean_reference_token_recall": _mean(group, "reference_token_recall"),
            "mean_claim_support_rate": _mean(group, "claim_support_rate"),
            "mean_latency_ms": _mean(group, "latency_ms"),
        }
    errors = Counter()
    for row in rows:
        if not row.get("operational_success"):
            errors["operational_failure"] += 1
        if row.get("fallback"):
            errors["fallback"] += 1
        if not row.get("route_correct"):
            errors["route_mismatch"] += 1
        if not row.get("expected_source_found"):
            errors["expected_source_missing"] += 1
        if not row.get("expected_page_found"):
            errors["expected_page_missing"] += 1
        if row.get("groundedness_pass") is False:
            errors["groundedness_failed"] += 1
        if row.get("reference_token_recall") is not None and float(row["reference_token_recall"]) < 0.5:
            errors["reference_recall_below_0.5"] += 1
    return {
        "method_label": "automated_pdf_reference_and_runtime_validation",
        "human_validated": False,
        "dataset": str(dataset),
        "output_dir": str(output),
        "total_cases": len(cases),
        "evaluable_cases": sum(row.get("error") is None for row in rows),
        "workers": workers,
        "groundedness_threshold": GROUNDING_THRESHOLD,
        "overall": {
            "operational_completion_rate": _mean(rows, "operational_success"),
            "route_accuracy": _mean(rows, "route_correct"),
            "answer_present_rate": _mean(rows, "answer_present"),
            "fallback_rate": _mean(rows, "fallback"),
            "expected_source_hit_rate": _mean(rows, "expected_source_found"),
            "expected_exact_page_hit_rate": _mean(rows, "expected_page_found"),
            "groundedness_pass_rate": _mean(substantive, "groundedness_pass"),
            "mean_groundedness_score": _mean(substantive, "groundedness_score"),
            "mean_reference_token_recall": _mean(substantive, "reference_token_recall"),
            "reference_recall_below_0.5_rate": sum(float(row.get("reference_token_recall") or 0) < 0.5 for row in substantive) / len(substantive) if substantive else None,
            "mean_claim_support_rate": _mean(substantive, "claim_support_rate"),
            "citation_presence_rate": sum(bool(row.get("citation_count")) for row in substantive) / len(substantive) if substantive else None,
            "citation_link_precision": _mean(substantive, "citation_link_precision"),
            "mean_returned_source_count": _mean(substantive, "source_count"),
            "latency_ms": {
                "mean": statistics.mean(latency) if latency else None,
                "median": statistics.median(latency) if latency else None,
                "p90": _percentile(latency, 0.90),
                "p95": _percentile(latency, 0.95),
                "p99": _percentile(latency, 0.99),
                "min": min(latency) if latency else None,
                "max": max(latency) if latency else None,
            },
        },
        "per_pdf": per_pdf,
        "error_counts": dict(errors),
        "failed_case_ids": [row["id"] for row in rows if row.get("fallback") or not row.get("route_correct") or not row.get("expected_source_found") or row.get("groundedness_pass") is False],
        "health": health,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    latency = overall["latency_ms"]
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.2f}%"
    lines = [
        "# PDF RAG-only 200-case evaluation",
        "",
        "> Automated runtime and deterministic reference-evidence validation. It is not independent human validation.",
        "",
        f"- Cases: {summary['total_cases']}",
        f"- Operational completion: {pct(overall['operational_completion_rate'])}",
        f"- Fallback rate: {pct(overall['fallback_rate'])}",
        f"- RAG-only route accuracy: {pct(overall['route_accuracy'])}",
        f"- Expected PDF source hit rate: {pct(overall['expected_source_hit_rate'])}",
        f"- Exact reference page hit rate: {pct(overall['expected_exact_page_hit_rate'])}",
        f"- Groundedness pass rate (threshold {summary['groundedness_threshold']}): {pct(overall['groundedness_pass_rate'])}",
        f"- Mean groundedness score: {overall['mean_groundedness_score']:.4f}" if overall['mean_groundedness_score'] is not None else "- Mean groundedness score: n/a",
        f"- Mean reference token recall: {pct(overall['mean_reference_token_recall'])}",
        f"- Mean claim-support rate: {pct(overall['mean_claim_support_rate'])}",
        f"- Citation presence: {pct(overall['citation_presence_rate'])}",
        f"- Citation link precision: {pct(overall['citation_link_precision'])}",
        "",
        "## Latency (milliseconds)",
        "",
        f"Mean {latency['mean']:.1f} | Median {latency['median']:.1f} | P90 {latency['p90']:.1f} | P95 {latency['p95']:.1f} | P99 {latency['p99']:.1f} | Max {latency['max']:.1f}",
        "",
        "## Error counts",
        "",
    ]
    for key, value in sorted(summary["error_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Per-PDF results", "", "| PDF | Cases | Source hit | Page hit | Groundedness pass | Mean groundedness | Mean reference recall | Mean latency ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, metrics in summary["per_pdf"].items():
        lines.append(
            f"| `{name}` | {metrics['cases']} | {pct(metrics['expected_source_hit_rate'])} | {pct(metrics['expected_page_hit_rate'])} | {pct(metrics['groundedness_pass_rate'])} | {metrics['mean_groundedness']:.4f} | {pct(metrics['mean_reference_token_recall'])} | {metrics['mean_latency_ms']:.1f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Reference token recall and claim-support are deterministic proxies, not expert insurance review. Groundedness is read from the production evidence guardrail; a failed or fallback answer is a retrieval/runtime risk, not automatically a confirmed hallucination. The saved per-case JSONL contains the full payload and reference evidence needed for manual audit.",
        "",
    ])
    return "\n".join(lines)


def collect_case(case: Mapping[str, Any], endpoint: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    status = 0
    payload: Any = {}
    error = None
    for attempt in range(1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint, json={"question": case["question"]})
            status = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw_text": response.text}
            break
        except Exception as exc:  # pragma: no cover - live network path
            error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(1.0)
    return {
        "id": case["id"],
        "question": case["question"],
        "endpoint": endpoint,
        "status_code": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload if isinstance(payload, Mapping) else {"value": payload},
        "collection_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Single endpoint (legacy option).")
    parser.add_argument("--endpoints", nargs="+", help="Optional pool of endpoints; cases are distributed round-robin.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=190.0)
    args = parser.parse_args()
    cases = load_jsonl(args.dataset)
    if len(cases) != 200:
        raise SystemExit(f"Expected 200 cases, got {len(cases)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = args.endpoints or [args.endpoint]
    health: dict[str, Any] = {}
    for endpoint in endpoints:
        base_url = endpoint.rsplit("/api/ask", 1)[0]
        with httpx.Client(timeout=60.0) as client:
            health_response = client.get(f"{base_url}/health")
            health[endpoint] = {"status_code": health_response.status_code, "payload": health_response.json()}
            health_response.raise_for_status()
    print(f"Health OK on {len(endpoints)} endpoint(s); running {len(cases)} RAG-only cases with {args.workers} workers", flush=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    partial_path = args.output_dir / "responses_partial.jsonl"
    partial_path.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(collect_case, case, endpoints[index % len(endpoints)], args.timeout_seconds): case
            for index, case in enumerate(cases)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            rows_by_id[str(result["id"])] = result
            with partial_path.open("a", encoding="utf-8") as partial_handle:
                partial_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                partial_handle.flush()
            completed += 1
            print(f"[{completed}/{len(cases)}] {result['id']} HTTP={result['status_code']} {result['latency_ms']:.0f}ms", flush=True)
    collected = [rows_by_id[str(case["id"])] for case in cases]
    responses_path = args.output_dir / "responses_full.jsonl"
    with responses_path.open("w", encoding="utf-8") as handle:
        for row in collected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    evaluated = [_evaluate_row(case, row) for case, row in zip(cases, collected)]
    (args.output_dir / "per_case_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluated), encoding="utf-8"
    )
    write_csv(args.output_dir / "per_case_results.csv", evaluated)
    summary = summarize(evaluated, cases, dataset=args.dataset.resolve(), output=args.output_dir.resolve(), health=health, workers=args.workers)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps({"dataset": str(args.dataset.resolve()), "responses": str(responses_path.resolve()), "cases": len(cases), "endpoints": endpoints}, indent=2), encoding="utf-8"
    )
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

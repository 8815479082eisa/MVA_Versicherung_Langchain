#!/usr/bin/env python3
"""Phase 1 triage for combined RAG+CRM fallbacks.

Reads a fixed query set, calls the running backend once per query, stores the full
response as JSON on disk, and reduces each run to ONE CSV row plus an automatic
H1-H6 hypothesis assignment.

Design rule: the only thing that should ever be pasted into a chat session is the
summary table this script prints. Raw diagnostics stay on disk.

Usage:
    python scripts/tools/triage_combined.py
    python scripts/tools/triage_combined.py --base-url http://localhost:8000
    python scripts/tools/triage_combined.py --only C3 C7        # rerun single cases
    python scripts/tools/triage_combined.py --repeat 3          # flakiness check (H3)

Reads:  tests/fixtures/combined_failing_queries.json
Writes: artifacts/test-results/combined-triage/combined-triage.csv
        artifacts/test-results/combined-triage/raw/<id>[-run<N>].json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("requests is required:  pip install requests")


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "combined_failing_queries.json"
OUT_DIR = REPO_ROOT / "artifacts" / "test-results" / "combined-triage"
RAW_DIR = OUT_DIR / "raw"
CSV_PATH = OUT_DIR / "combined-triage.csv"

CSV_FIELDS = [
    "id", "run", "hypothesis", "hypothesis_reason",
    "http_status", "status", "route", "expected_route", "route_ok",
    "safety_decision", "pre_action", "context_action", "post_action",
    "safety_reason_code",
    "groundedness_score", "groundedness_threshold", "groundedness_passed",
    "groundedness_exception",
    "timeout_stage", "failed_stages",
    "completeness_pass", "completeness_missing",
    "warning_code",
    "n_sources", "n_crm_sources", "n_pdf_chunks", "crm_facts_in_context",
    "answer_chars", "answer_is_empty",
    "expected_facts_found", "expected_facts_missing",
    "latency_ms", "total_ms",
    "query",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Safe nested lookup; every field in this pipeline is optional."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def normalize_route(value: str) -> str:
    """The API emits 'crm-only'/'retrieval-only'; QueryMode uses underscores.
    Compare them in one normalised form so a naming difference is never
    mistaken for a routing defect."""
    return str(value or "").strip().lower().replace("-", "_")


def as_number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_crm_source(source: dict) -> bool:
    doc_id = str(source.get("documentId") or "")
    return doc_id.startswith("espocrm") or (source.get("section") or "") == "crm"


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(row: dict) -> tuple[str, str]:
    """Assign one hypothesis from the plan. Order matters: most specific first.

    H1 rail fired (input/context/output)   H4 CRM evidence evicted
    H2 groundedness below threshold        H5 routing
    H3 evaluator failed / masked error     H6 completeness heuristic
    """
    http = row["http_status"]
    decision = (row["safety_decision"] or "").lower()
    pre, ctx, post = row["pre_action"], row["context_action"], row["post_action"]
    score = as_number(row["groundedness_score"])
    thr = as_number(row["groundedness_threshold"])

    # Operational, not a pipeline defect - exclude before hypothesising.
    if http == -1:
        return "INFRA", "request failed - backend unreachable"
    if http >= 500 and http != 503:
        return "INFRA", f"server error {http} - read the raw JSON and the server log"
    if http == 503:
        return "INFRA", "CRM runtime not ready (503) - fix the environment, not the code"
    if http == 403:
        return "H5", "router denied the query (403)"

    # H3 first: an evaluator that crashed or timed out is not a safety verdict.
    if row["timeout_stage"] or row["groundedness_exception"]:
        return "H3", f"stage failure masked as fallback (timeout={row['timeout_stage']}, exc={row['groundedness_exception']})"
    if row["failed_stages"]:
        return "H3", f"failed stages: {row['failed_stages']}"

    # H5: wrong route means everything downstream is moot.
    if row["expected_route"] and not row["route_ok"]:
        return "H5", f"route {row['route']} != expected {row['expected_route']}"

    # H1: an input or context rail fired. The context rail is the one that sees CRM PII.
    if decision.startswith("context_") or ctx in {"block", "fallback", "redact"}:
        return "H1", f"context rail action={ctx}, decision={decision}"
    if pre in {"block", "fallback"}:
        return "H1", f"input rail action={pre}"

    # The post stage covers BOTH the output rail and the groundedness gate.
    # Separating them is the whole point - a fallback on a perfectly grounded
    # answer is an output-rail decision, not a threshold problem.
    post_fired = decision.startswith("post_") or post in {"block", "fallback"}
    if post_fired:
        if score is None:
            return "H3", f"post fallback but groundedness produced NO score (decision={decision}) - evaluator did not run or returned null"
        if thr is not None and score < thr:
            return "H2", f"groundedness {score} < threshold {thr} (genuine threshold rejection)"
        return "H1", f"OUTPUT rail fired despite groundedness {score} >= threshold {thr} - not a grounding problem"

    # H4: CRM facts went in but did not survive into the final sources.
    if row["crm_facts_in_context"] and row["n_crm_sources"] == 0:
        return "H4", "CRM docs present in context evidence but absent from final sources"

    # H6: 206 / partial with a real answer in it.
    if http == 206 or row["status"] == "partial" or row["completeness_pass"] is False:
        if not row["answer_is_empty"]:
            return "H6", f"partial with non-empty answer, warning={row['warning_code']}"
        return "H6", f"partial, warning={row['warning_code']}"

    if row["expected_facts_missing"]:
        return "UNKNOWN", f"200 OK but missing expected facts: {row['expected_facts_missing']}"

    return "OK", "answer returned, expected facts present"


# --------------------------------------------------------------------------
# per-query run
# --------------------------------------------------------------------------

def run_case(base_url: str, case: dict, run_idx: int, timeout: float) -> dict:
    case_id = case["id"]
    question = case["query"]
    expected_route = case.get("expected_route") or ""
    expected_facts = case.get("expected_facts") or []

    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/ask",
            json={
                "question": question,
                "shortAnswer": case.get("shortAnswer", False),
                "structuredAnswer": case.get("structuredAnswer", False),
            },
            timeout=timeout,
        )
        http_status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {"_unparseable_body": resp.text[:2000]}
    except requests.RequestException as exc:
        http_status = -1
        body = {"_request_error": f"{type(exc).__name__}: {exc}"}

    wall_ms = round((time.perf_counter() - started) * 1000, 1)

    # 206 responses are the serialised model under FastAPI's JSONResponse.
    payload = body.get("detail") if (isinstance(body, dict) and http_status in {403, 503}) else body
    if not isinstance(payload, dict):
        payload = {}

    suffix = f"-run{run_idx}" if run_idx > 1 else ""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{case_id}{suffix}.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diagnostics = payload.get("diagnostics") or {}
    evidence = diagnostics.get("evidence") or {}

    safety = evidence.get("safety") or {}
    groundedness = evidence.get("groundedness") or {}
    completeness = evidence.get("completeness") or {}
    crm_summary = evidence.get("crmContextSummary") or []
    pdf_chunks = evidence.get("pdfChunks") or []
    # main.py records this one for crm-only / clarify paths
    safety_decision_evidence = evidence.get("safetyDecision") or {}

    answer = str(payload.get("answer") or "")
    sources = payload.get("sources") or []
    stage_status = diagnostics.get("stageStatus") or {}
    failed = [s for s, st in stage_status.items() if st in {"failed", "timeout"}]

    found = [f for f in expected_facts if f.lower() in answer.lower()]
    missing = [f for f in expected_facts if f.lower() not in answer.lower()]

    route = payload.get("route") or diagnostics.get("route") or ""

    row = {
        "id": case_id,
        "run": run_idx,
        "http_status": http_status,
        "status": payload.get("status") or diagnostics.get("status") or "",
        "route": route,
        "expected_route": expected_route,
        "route_ok": (normalize_route(route) == normalize_route(expected_route)) if expected_route else "",
        "safety_decision": safety.get("decision") or safety_decision_evidence.get("decision") or "",
        "pre_action": safety.get("preAction") or "",
        "context_action": safety.get("contextAction") or "",
        "post_action": safety.get("postAction") or "",
        "safety_reason_code": safety_decision_evidence.get("reasonCode") or "",
        "groundedness_score": groundedness.get("score", safety_decision_evidence.get("groundedness", "")),
        "groundedness_threshold": groundedness.get("threshold", safety_decision_evidence.get("threshold", "")),
        "groundedness_passed": groundedness.get("passed", ""),
        "groundedness_exception": groundedness.get("exceptionType") or "",
        "timeout_stage": diagnostics.get("timeoutStage") or "",
        "failed_stages": "|".join(failed),
        "completeness_pass": completeness.get("completenessPass", ""),
        "completeness_missing": "|".join(
            str(x) for x in (completeness.get("completenessMissingAfterRetry")
                             or completeness.get("completenessMissingBeforeRetry") or [])
        ),
        "warning_code": dig(payload, "warning", "errorCode", default="") or payload.get("errorCode", ""),
        "n_sources": len(sources),
        "n_crm_sources": sum(1 for s in sources if isinstance(s, dict) and is_crm_source(s)),
        "n_pdf_chunks": len(pdf_chunks),
        "crm_facts_in_context": len(crm_summary),
        "answer_chars": len(answer),
        "answer_is_empty": len(answer.strip()) == 0,
        "expected_facts_found": "|".join(found),
        "expected_facts_missing": "|".join(missing),
        "latency_ms": payload.get("latencyMs") or wall_ms,
        "total_ms": diagnostics.get("totalMs", ""),
        "query": question,
    }

    hypothesis, reason = classify(row)
    row["hypothesis"] = hypothesis
    row["hypothesis_reason"] = reason
    return row


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the whole set N times; use >1 to expose H3 flakiness")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these case ids")
    args = parser.parse_args()

    if not args.fixture.exists():
        return print(f"fixture not found: {args.fixture}") or 1

    cases = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    if not cases:
        return print("no cases selected") or 1

    # Health first - a red backend makes every row meaningless.
    try:
        health = requests.get(f"{args.base_url.rstrip('/')}/health", timeout=90).json()
        print(f"health: pipelineReady={health.get('pipelineReady')} "
              f"crmReady={health.get('crmReady')} llmReady={health.get('llmReady')} "
              f"safetyMode={health.get('safetyMode')} backend={health.get('safety_backend')}")
        if not health.get("pipelineReady") or not health.get("crmReady"):
            print("WARNING: backend not fully ready - results will be dominated by INFRA")
    except Exception as exc:
        print(f"health check failed: {exc}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_idx in range(1, args.repeat + 1):
        for case in cases:
            row = run_case(args.base_url, case, run_idx, args.timeout)
            rows.append(row)
            print(f"  [{row['hypothesis']:7}] {row['id']:6} "
                  f"http={row['http_status']} route={row['route'] or '-':14} "
                  f"{row['hypothesis_reason'][:70]}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # The summary below is the ONLY output meant for a chat session.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["hypothesis"]] = counts.get(row["hypothesis"], 0) + 1

    labels = {
        "H1": "context rail / PII",
        "H2": "groundedness threshold",
        "H3": "fail_closed masks errors",
        "H4": "CRM evidence evicted",
        "H5": "routing",
        "H6": "completeness heuristic",
        "OK": "passed",
        "INFRA": "environment not ready",
        "UNKNOWN": "unclassified - read raw JSON",
    }
    print("\n=== TRIAGE SUMMARY ===")
    for key in ["H1", "H2", "H3", "H4", "H5", "H6", "UNKNOWN", "INFRA", "OK"]:
        if counts.get(key):
            print(f"  {key:8} {counts[key]:3}  {labels[key]}")

    # Flakiness is the signature of H3; surface it explicitly.
    if args.repeat > 1:
        by_id: dict[str, set] = {}
        for row in rows:
            by_id.setdefault(row["id"], set()).add(row["hypothesis"])
        unstable = {k: sorted(v) for k, v in by_id.items() if len(v) > 1}
        if unstable:
            print("\n  UNSTABLE across runs (strong H3 signal):")
            for case_id, hyps in unstable.items():
                print(f"    {case_id}: {hyps}")

    print(f"\ncsv:  {CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"raw:  {RAW_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

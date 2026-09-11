from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.claim_groundedness import (
    Claim,
    Extraction,
    ModelJudge,
    _answer_structure,
    _build_judgment_payload,
    _judge_with_recovery,
    _payload_chars,
    _result,
    critical_facts,
    deterministic_checks,
    evaluate_claim_groundedness,
    evidence_windows,
    quote_has_provenance,
)


DEFAULT_RUN_DIR = ROOT / "artifacts" / "pdf-rag-200-evaluation-sequential"
DEFAULT_DATASET = ROOT / "data" / "benchmarks" / "pdf_rag" / "pdf_rag_200.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "hallucination-evaluation-pdf-rag-200"
FALLBACK_ANSWERS = {
    "I could not generate an answer that is sufficiently supported by the available documents.",
    "I cannot include or expose personal or sensitive information in the response.",
    "I cannot provide a safe, policy-compliant answer for this request. Please rephrase.",
    "The available sources do not contain enough information to answer this question.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def rate_payload(numerator: int, denominator: int) -> dict[str, Any]:
    low, high = wilson_interval(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "wilson_95_ci": [low, high],
    }


def runtime_outcomes(responses: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    grounding_reasons: Counter[str] = Counter()
    detected_contradictions = 0
    blocked_detected_contradictions = 0
    for row in responses:
        payload = row.get("payload") or {}
        answer = str(payload.get("answer") or "").strip()
        safety = (((payload.get("diagnostics") or {}).get("evidence") or {}).get("safety") or {})
        post_action = str(safety.get("postAction") or "")
        risks = {str(item) for item in safety.get("risks") or []}
        status = int(row.get("status_code") or 0)

        if "groundedness_contradicted" in risks:
            detected_contradictions += 1
            if post_action == "fallback":
                blocked_detected_contradictions += 1
        for risk in risks:
            if risk.startswith("groundedness_"):
                grounding_reasons[risk] += 1

        if status not in {200, 206} or not answer:
            counts["operational_failure"] += 1
        elif answer == "I cannot include or expose personal or sensitive information in the response.":
            counts["pii_fallback"] += 1
        elif answer == "The available sources do not contain enough information to answer this question.":
            counts["controlled_abstention"] += 1
        elif post_action == "fallback" or answer == "I cannot provide a safe, policy-compliant answer for this request. Please rephrase.":
            counts["post_generation_fallback"] += 1
        else:
            counts["user_exposed_factual_answer"] += 1

    total = len(responses)
    return {
        "counts": dict(counts),
        "rates": {key: value / total for key, value in counts.items()} if total else {},
        "grounding_reason_counts": dict(grounding_reasons),
        "pre_output_confirmed_contradictions": rate_payload(detected_contradictions, total),
        "blocked_confirmed_contradictions": blocked_detected_contradictions,
    }


def source_page_keys(response: dict[str, Any], dataset_row: dict[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    payload = response.get("payload") or {}
    answer = str(payload.get("answer") or "")
    for source in payload.get("sources") or []:
        filename = str(source.get("documentTitle") or source.get("source_file") or "").strip()
        page = source.get("page")
        if filename and isinstance(page, int) and page > 0:
            keys.add((filename, page))
    for filename, page_text in re.findall(
        r"\[([^\[\],]+\.pdf)\s*,\s*page\s+(\d+)\]", answer, flags=re.IGNORECASE
    ):
        page = int(page_text)
        if page > 0:
            keys.add((filename.strip(), page))
    for evidence in dataset_row.get("reference_evidence") or []:
        filename = str(evidence.get("source") or "").strip()
        page = evidence.get("page")
        if filename and isinstance(page, int) and page > 0:
            keys.add((filename, page))
    return keys


def load_pdf_pages(keys: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], str] = {}
    readers: dict[str, PdfReader] = {}
    for filename, page in sorted(keys):
        path = ROOT / "data" / "raw" / "pdfs" / filename
        if not path.exists():
            continue
        reader = readers.setdefault(filename, PdfReader(path))
        if page <= len(reader.pages):
            pages[(filename, page)] = reader.pages[page - 1].extract_text() or ""
    return pages


def build_evidence(
    response: dict[str, Any],
    dataset_row: dict[str, Any],
    page_texts: dict[tuple[str, int], str],
) -> list[Document]:
    documents = []
    for filename, page in sorted(source_page_keys(response, dataset_row)):
        text = page_texts.get((filename, page), "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source_file": filename, "source_page": page, "page": page - 1},
                )
            )
    return documents


def evaluate_case(
    response: dict[str, Any],
    dataset_row: dict[str, Any],
    page_texts: dict[tuple[str, int], str],
) -> dict[str, Any]:
    payload = response.get("payload") or {}
    answer = str(payload.get("answer") or "").strip()
    documents = build_evidence(response, dataset_row, page_texts)
    score, diagnostics = evaluate_claim_groundedness(
        answer,
        documents,
        str(dataset_row.get("question") or response.get("question") or ""),
    )
    return {
        "id": response["id"],
        "question": dataset_row.get("question"),
        "answer": answer,
        "source_page_count": len(documents),
        "score": score,
        "evaluation_status": diagnostics.get("evaluation_status"),
        "relation_counts": diagnostics.get("relation_counts") or {},
        "claim_details": diagnostics.get("claim_details") or [],
        "all_claims_supported": diagnostics.get("all_claims_supported"),
        "exception_type": diagnostics.get("exception_type"),
        "failure_stage": diagnostics.get("failure_stage"),
        "evaluation_method": "audited_atomic_claim_entailment",
    }


def evaluate_presegmented_case(
    response: dict[str, Any],
    dataset_row: dict[str, Any],
    page_texts: dict[tuple[str, int], str],
) -> dict[str, Any]:
    """Fallback for answers whose model-based atomic extraction cannot be validated.

    This keeps response-level coverage complete without presenting deterministic sentence
    units as validated atomic claims. Fragment bullets inherit their heading text.
    """
    payload = response.get("payload") or {}
    answer = str(payload.get("answer") or "").strip()
    documents = build_evidence(response, dataset_row, page_texts)
    units, required_ids, dependencies = _answer_structure(answer)
    claims = []
    for unit_id in sorted(required_ids):
        text = units[unit_id]
        heading_id = dependencies.get(unit_id)
        if heading_id is not None:
            text = f"{units[heading_id]} {text}"
        claims.append(Claim(text=text, unit_ids=[unit_id]))
    extraction = Extraction(claims=claims)
    judge = ModelJudge()
    windows = evidence_windows(documents)
    judgment_payload, selected = _build_judgment_payload(
        str(dataset_row.get("question") or response.get("question") or ""),
        documents,
        extraction,
        windows,
    )
    judgments, judgment_attempts = _judge_with_recovery(judge, judgment_payload, extraction)

    rows = []
    for verdict in sorted(judgments.verdicts, key=lambda item: item.claim_id):
        claim = extraction.claims[verdict.claim_id].text
        original_relation = verdict.relation
        relation = original_relation
        demotion_reasons = []
        citations_present = bool(verdict.citations)
        citation_selection_valid = citations_present and all(
            citation.evidence_id in selected[verdict.claim_id] for citation in verdict.citations
        )
        provenance_valid = citation_selection_valid and all(
            quote_has_provenance(citation.quote, windows[citation.evidence_id]["text"])
            for citation in verdict.citations
        )
        checks = verdict.checks.model_dump()
        sensitive_dimensions = [name for name, value in checks.items() if value != "not_applicable"]
        is_sensitive_claim = bool(sensitive_dimensions or critical_facts(claim))
        quotes = "\n".join(citation.quote for citation in verdict.citations)
        issues = deterministic_checks(claim, quotes) if provenance_valid else ["unverified_citations"]

        if relation in {"supported", "contradicted"}:
            validation_failures = []
            if not provenance_valid:
                validation_failures.append("citation_validation_failed")
            if not verdict.subject_scope_match:
                validation_failures.append("subject_scope_mismatch")
            if verdict.confidence < 0.85:
                validation_failures.append("confidence_below_0.85")
            if validation_failures:
                demotion_reasons.extend(validation_failures)
                relation = "unknown"

        if relation == "supported":
            support_failures = [f"deterministic_issue:{issue}" for issue in issues]
            support_failures.extend(
                f"sensitive_check:{name}:{value}"
                for name, value in checks.items()
                if value in {"conflict", "unknown"}
            )
            if support_failures:
                demotion_reasons.extend(support_failures)
                relation = "insufficient_evidence"

        if relation == "contradicted" and "conflict" not in checks.values():
            demotion_reasons.append("contradiction_without_sensitive_conflict")
            relation = "unknown"
        if relation == "contradicted" and "location_not_verified" in issues:
            demotion_reasons.append("contradiction_location_not_verified")
            relation = "insufficient_evidence"

        rows.append(
            {
                "claim_text": claim,
                "original_relation": original_relation,
                "final_relation": relation,
                "demotion_reasons": demotion_reasons,
                "provenance_valid": provenance_valid,
                "relation": relation,
                "passed": relation == "supported",
                "confidence": verdict.confidence,
                "reason_code": verdict.reason,
                "subject_scope_match": verdict.subject_scope_match,
                "evidence": [citation.model_dump() for citation in verdict.citations],
                "sensitive_checks": checks,
                "deterministic_issues": issues,
                "is_sensitive_claim": is_sensitive_claim,
            }
        )

    score, diagnostics = _result(rows)
    return {
        "id": response["id"],
        "question": dataset_row.get("question"),
        "answer": answer,
        "source_page_count": len(documents),
        "score": score,
        "evaluation_status": diagnostics.get("evaluation_status"),
        "relation_counts": diagnostics.get("relation_counts") or {},
        "claim_details": diagnostics.get("claim_details") or [],
        "all_claims_supported": diagnostics.get("all_claims_supported"),
        "exception_type": diagnostics.get("exception_type"),
        "failure_stage": diagnostics.get("failure_stage"),
        "evaluation_method": "presegmented_claim_unit_entailment_fallback",
        "judgment_attempts": judgment_attempts,
        "payload_chars": _payload_chars(judgment_payload),
    }


def aggregate(results: list[dict[str, Any]], total_queries: int) -> dict[str, Any]:
    completed = [row for row in results if row.get("evaluation_status") in {"success", "uncertain"}]
    failed = [row for row in results if row not in completed]
    primary = [
        row
        for row in completed
        if row.get("evaluation_method") in {None, "audited_atomic_claim_entailment"}
    ]
    fallback = [
        row
        for row in completed
        if row.get("evaluation_method") == "presegmented_claim_unit_entailment_fallback"
    ]
    relations: Counter[str] = Counter()
    for row in primary:
        relations.update({str(key): int(value) for key, value in row["relation_counts"].items()})

    decided = relations["supported"] + relations["contradicted"] + relations["insufficient_evidence"]
    total_claims = decided + relations["unknown"]
    strict_response_count = sum(int(row["relation_counts"].get("contradicted", 0)) > 0 for row in completed)
    broad_response_count = sum(
        int(row["relation_counts"].get("contradicted", 0))
        + int(row["relation_counts"].get("insufficient_evidence", 0))
        > 0
        for row in completed
    )
    uncertain_response_count = sum(int(row["relation_counts"].get("unknown", 0)) > 0 for row in completed)

    return {
        "method_label": "automated_claim_entailment_hallucination_audit",
        "algorithm": "claim_entailment_v1",
        "human_validated": False,
        "total_queries_in_original_run": total_queries,
        "user_exposed_factual_answers_selected": len(results),
        "successfully_evaluated_answers": len(completed),
        "primary_atomic_evaluations": len(primary),
        "presegmented_fallback_evaluations": len(fallback),
        "response_level_evaluation_coverage": len(completed) / len(results) if results else None,
        "evaluator_failures": len(failed),
        "relation_counts": dict(relations),
        "total_atomic_claims": total_claims,
        "strict_confirmed_contradiction_claim_rate": rate_payload(relations["contradicted"], total_claims),
        "evidence_unsupported_claim_rate": rate_payload(
            relations["contradicted"] + relations["insufficient_evidence"], decided
        ),
        "unsupported_or_uncertain_claim_rate": rate_payload(
            relations["contradicted"] + relations["insufficient_evidence"] + relations["unknown"],
            total_claims,
        ),
        "strict_response_level_hallucination_rate": rate_payload(strict_response_count, len(completed)),
        "evidence_unsupported_response_rate": rate_payload(broad_response_count, len(completed)),
        "uncertain_response_rate": rate_payload(uncertain_response_count, len(completed)),
        "system_level_confirmed_hallucination_exposure_rate": rate_payload(strict_response_count, total_queries),
        "system_level_evidence_unsupported_exposure_rate": rate_payload(broad_response_count, total_queries),
        "interpretation": {
            "strict": "Only claims explicitly contradicted by authoritative PDF evidence are counted as confirmed hallucinations.",
            "broad": "Contradicted and insufficient-evidence claims are counted as evidence-unsupported generation; this is a hallucination proxy, not proof of factual falsehood.",
            "denominator": "Fallbacks, blocks, and operational failures are excluded from claim-level rates and remain in the denominator of system-level exposure rates.",
            "mixed_granularity": "Claim-level rates use only validated atomic extractions. Response-level rates also include the presegmented fallback when atomic extraction failed.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--use-presegmented-fallback", action="store_true")
    args = parser.parse_args()

    responses = read_jsonl(args.run_dir / "responses_full.jsonl")
    dataset = {row["id"]: row for row in read_jsonl(args.dataset)}
    audit_path = args.run_dir / "detailed_audit_cases.jsonl"
    audit = {row["id"]: row for row in read_jsonl(audit_path)} if audit_path.exists() else {}
    selected = []
    for row in responses:
        payload = row.get("payload") or {}
        answer = str(payload.get("answer") or "").strip()
        if row.get("id") not in dataset or not answer or answer in FALLBACK_ANSWERS:
            continue
        if audit:
            include = audit.get(row["id"], {}).get("classification") == "generated_answer"
        else:
            safety = (((payload.get("diagnostics") or {}).get("evidence") or {}).get("safety") or {})
            include = int(row.get("status_code") or 0) in {200, 206} and safety.get("postAction") != "fallback"
        if include:
            selected.append(row)
    if args.limit:
        selected = selected[: args.limit]

    all_keys: set[tuple[str, int]] = set()
    for row in selected:
        all_keys.update(source_page_keys(row, dataset[row["id"]]))
    page_texts = load_pdf_pages(all_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "per_case_results.jsonl"
    existing = {row["id"]: row for row in read_jsonl(output_jsonl)} if output_jsonl.exists() else {}
    pending = [
        row
        for row in selected
        if row["id"] not in existing
        or (args.retry_failed and existing[row["id"]].get("evaluation_status") == "failed")
    ]
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for row in pending:
            previous = existing.get(row["id"]) or {}
            use_fallback = (
                args.use_presegmented_fallback
                and previous.get("evaluation_status") == "failed"
                and str(previous.get("failure_stage") or "").startswith("claim_extraction")
            )
            evaluator = evaluate_presegmented_case if use_fallback else evaluate_case
            future = executor.submit(evaluator, row, dataset[row["id"]], page_texts)
            futures[future] = row["id"]
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Keep the audit resumable and account for every case.
                result = {
                    "id": case_id,
                    "evaluation_status": "failed",
                    "relation_counts": {},
                    "claim_details": [],
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:500],
                }
            existing[case_id] = result
            with lock, output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{len(existing)}/{len(selected)} {case_id} {result.get('evaluation_status')}", flush=True)

    ordered = [existing[row["id"]] for row in selected]
    summary = aggregate(ordered, len(responses))
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(args.run_dir),
            "source_dataset": str(args.dataset),
            "runtime_outcomes": runtime_outcomes(responses),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

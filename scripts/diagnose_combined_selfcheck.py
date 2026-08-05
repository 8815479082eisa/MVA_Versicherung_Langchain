from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.diagnostic_capture import atomic_write_json, sanitize_diagnostic_value
from src.core.coverage_taxonomy import (
    find_coverage_terms,
    has_conflicting_coverage_terms,
)


QUESTION = """Lara Neumann reports damage to the windscreen of her car. Is this damage
generally covered by her current motor insurance, and would she still have to
pay her individual deductible if the windscreen can be repaired instead of
replaced?

Please provide the relevant active policy number, coverage type, individual
deductible and annual premium. Clearly distinguish the general insurance terms
from Lara's individual contract data, and do not make a final claim decision."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _initial_diagnostic() -> dict[str, Any]:
    return {
        "request_id": None,
        "started_at": _utc_now(),
        "finished_at": None,
        "http_status": None,
        "route": None,
        "selected_policy_number": None,
        "crm_context_summary": [],
        "pdf_chunks": [],
        "reranker": None,
        "self_check": {
            "enabled": None,
            "actually_executed": False,
            "skipped": None,
            "skip_reason": None,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "timeout_seconds": 30,
            "started_at": None,
            "finished_at": None,
            "latency_ms": None,
            "attempt_count": 0,
            "retry_count": 0,
            "raw_output_sanitized": None,
            "parsed_verdict": None,
            "parser_success": None,
            "reason_code": None,
            "exception_type": None,
            "exception_message_sanitized": None,
            "cause_type": None,
            "cause_message_sanitized": None,
            "timeout": False,
        },
        "answer_generation_started": False,
        "answer_generation_completed": False,
        "groundedness": {
            "algorithm": None,
            "score": None,
            "threshold": None,
            "passed": None,
            "exception_type": None,
            "exception_message_sanitized": None,
        },
        "coverage_consistency": {
            "passed": None,
            "expected": None,
            "observed_terms": [],
        },
        "pdf_evidence_available": None,
        "pdf_citations": [],
        "final_pipeline_status": None,
        "final_response_sanitized": None,
        "diagnostic_capture_success": False,
    }


def _response_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("diagnostics")
    if isinstance(direct, dict):
        return direct
    detail = payload.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("diagnostics"), dict):
        return detail["diagnostics"]
    return {}


def _populate_from_response(
    diagnostic: dict[str, Any],
    *,
    http_status: int,
    payload: dict[str, Any],
) -> None:
    runtime = _response_diagnostics(payload)
    evidence = runtime.get("evidence") if isinstance(runtime.get("evidence"), dict) else {}
    self_check = evidence.get("selfCheck") if isinstance(evidence.get("selfCheck"), dict) else {}
    groundedness = evidence.get("groundedness") if isinstance(evidence.get("groundedness"), dict) else {}
    generation = evidence.get("answerGenerationStatus") if isinstance(evidence.get("answerGenerationStatus"), dict) else {}
    crm_selection = evidence.get("crmSelection") if isinstance(evidence.get("crmSelection"), dict) else {}
    selected_policies = crm_selection.get("selectedPolicyNumbers") or []
    reranker = evidence.get("reranker") if isinstance(evidence.get("reranker"), dict) else {}
    final_reranked_chunks = reranker.get("final") or evidence.get("pdfChunks") or []

    diagnostic.update(
        {
            "request_id": runtime.get("requestId"),
            "finished_at": _utc_now(),
            "http_status": http_status,
            "route": payload.get("route") or runtime.get("route"),
            "selected_policy_number": selected_policies[0] if selected_policies else None,
            "crm_context_summary": evidence.get("crmContextSummary") or [],
            "pdf_chunks": final_reranked_chunks,
            "reranker": reranker or None,
            "answer_generation_started": bool(generation.get("started", False)),
            "answer_generation_completed": bool(generation.get("completed", False)),
            "pdf_citations": evidence.get("pdfCitations") or [
                source
                for source in payload.get("sources", [])
                if isinstance(source, dict)
                and not str(source.get("documentId", "")).startswith("espocrm:")
            ],
            "final_pipeline_status": payload.get("status") or runtime.get("status"),
            "final_response_sanitized": sanitize_diagnostic_value(payload),
        }
    )
    diagnostic["self_check"].update(
        {
            "enabled": _first(self_check, "enabled", default=runtime.get("selfCheckEnabled")),
            "actually_executed": bool(
                self_check.get("attemptCount", 0)
                or runtime.get("stageStatus", {}).get("self_check") not in {None, "skipped"}
            ),
            "skipped": runtime.get("selfCheckSkipped"),
            "skip_reason": runtime.get("selfCheckSkipReason"),
            "provider": _first(self_check, "provider", default=diagnostic["self_check"]["provider"]),
            "model": _first(self_check, "model", default=diagnostic["self_check"]["model"]),
            "timeout_seconds": _first(self_check, "timeoutSeconds", default=diagnostic["self_check"]["timeout_seconds"]),
            "started_at": self_check.get("startedAt"),
            "finished_at": self_check.get("finishedAt"),
            "latency_ms": self_check.get("latencyMs"),
            "attempt_count": self_check.get("attemptCount", 0),
            "retry_count": self_check.get("retryCount", 0),
            "raw_output_sanitized": self_check.get("rawOutputSanitized"),
            "parsed_verdict": self_check.get("parsedVerdict"),
            "parser_success": self_check.get("parserSuccess"),
            "reason_code": self_check.get("reasonCode"),
            "exception_type": self_check.get("exceptionType"),
            "exception_message_sanitized": self_check.get("exceptionMessageSanitized"),
            "cause_type": self_check.get("causeType"),
            "cause_message_sanitized": self_check.get("causeMessageSanitized"),
            "timeout": bool(self_check.get("timeout", False)),
        }
    )
    diagnostic["groundedness"].update(
        {
            "algorithm": groundedness.get("algorithm"),
            "score": groundedness.get("score"),
            "threshold": groundedness.get("threshold"),
            "passed": groundedness.get("passed"),
            "exception_type": groundedness.get("exceptionType"),
            "exception_message_sanitized": groundedness.get("exceptionMessageSanitized"),
        }
    )
    diagnostic["pdf_evidence_available"] = bool(diagnostic["pdf_chunks"])
    crm_result = payload.get("crmResult") if isinstance(payload.get("crmResult"), dict) else {}
    crm_answer = str(crm_result.get("answer") or "")
    expected_coverage = next(
        (
            value
            for value in ("Partial Coverage", "Comprehensive Coverage", "Liability", "Standard Coverage")
            if value.casefold() in crm_answer.casefold()
        ),
        None,
    )
    final_answer = str(payload.get("answer") or "")
    observed_terms = find_coverage_terms(final_answer)
    diagnostic["coverage_consistency"] = {
        "passed": (
            not has_conflicting_coverage_terms(expected_coverage, observed_terms)
            if expected_coverage is not None
            else None
        ),
        "expected": expected_coverage,
        "observed_terms": [
            {"canonical": term.canonical, "label": term.label, "status": term.status}
            for term in observed_terms
        ],
    }
    diagnostic["diagnostic_capture_success"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/ask")
    parser.add_argument("--timeout-seconds", type=float, default=190.0)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/diagnostics"))
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"combined_selfcheck_{timestamp}.json"
    diagnostic = _initial_diagnostic()
    atomic_write_json(output_path, diagnostic)

    try:
        # This script intentionally performs exactly one HTTP request.
        response = httpx.post(
            args.url,
            json={"question": QUESTION},
            timeout=args.timeout_seconds,
        )
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {
                "status": "error",
                "body": response.text,
            }
        if not isinstance(response_payload, dict):
            response_payload = {"response": response_payload}
        _populate_from_response(
            diagnostic,
            http_status=response.status_code,
            payload=response_payload,
        )
    except Exception as exc:
        diagnostic["finished_at"] = _utc_now()
        diagnostic["final_pipeline_status"] = "client_error"
        diagnostic["final_response_sanitized"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    finally:
        atomic_write_json(output_path, diagnostic)

    console_summary = {
        "diagnostic_file": str(output_path.resolve()),
        "capture_success": diagnostic["diagnostic_capture_success"],
        "http_status": diagnostic["http_status"],
        "request_id": diagnostic["request_id"],
    }
    sys.stdout.write(json.dumps(console_summary, ensure_ascii=True) + "\n")
    return 0 if diagnostic["diagnostic_capture_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

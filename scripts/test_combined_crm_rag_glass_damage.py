from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RETRIEVAL_QUERY = (
    "Under partial coverage motor insurance, is windshield glass damage "
    "covered, and what conditions or limitations apply?"
)
COMBINED_QUESTION = (
    "Lara Neumann reported windshield glass damage. Is glass damage "
    "generally covered under partial coverage motor insurance? Also provide "
    "the coverage type, specific deductible, and policy number stored for "
    "the motor policy of Lara Neumann. Separate the general document rule "
    "from the individual CRM policy data."
)


def _page_values(metadata: dict[str, Any]) -> tuple[Any, Any]:
    page_metadata = metadata.get("page")
    page_human = metadata.get("page_human")
    if page_human is None and isinstance(page_metadata, int):
        page_human = page_metadata + 1
    return page_metadata, page_human


def _document_row(document: Any, rank: int) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    source = str(metadata.get("source") or "unknown")
    page_metadata, page_human = _page_values(metadata)
    return {
        "rank": rank,
        "filename": Path(source.replace("\\", "/")).name,
        "source": source,
        "pageMetadata": page_metadata,
        "pageHuman": page_human,
        "score": (
            metadata.get("reranker_score")
            or metadata.get("score")
            or metadata.get("relevance_score")
        ),
        "content": document.page_content or "",
        "characterCount": len(document.page_content or ""),
    }


def run_retrieval(output_path: Path, query: str = RETRIEVAL_QUERY) -> int:
    from src.api import rag_service
    from src.core.runtime_diagnostics import (
        RequestDiagnostics,
        reset_current_diagnostics,
        set_current_diagnostics,
    )

    diagnostics = RequestDiagnostics(route="pure-retrieval")
    token = set_current_diagnostics(diagnostics)
    started = time.perf_counter()
    try:
        service = rag_service.RetrievalService(rag_service.SETTINGS)
        service.initialize(force_reindex=False, allow_reindex=False)
        retrieved, reranked = service.retrieve_and_rerank(
            query,
            retrieval_top_k=rag_service.SETTINGS.retrieval.top_k,
            rerank_top_k=rag_service.SETTINGS.reranker.top_k,
            use_reranker=True,
        )
        diagnostics.complete(status="complete")
        payload = {
            "status": "PASS",
            "query": query,
            "generationUsed": False,
            "allowReindex": False,
            "configuredRetrievalTopK": rag_service.SETTINGS.retrieval.top_k,
            "configuredRerankTopK": rag_service.SETTINGS.reranker.top_k,
            "retrievedCount": len(retrieved),
            "rerankedCount": len(reranked),
            "retrieved": [
                _document_row(document, rank)
                for rank, document in enumerate(retrieved, start=1)
            ],
            "reranked": [
                _document_row(document, rank)
                for rank, document in enumerate(reranked, start=1)
            ],
            "wallLatencyMs": round((time.perf_counter() - started) * 1000, 3),
            "diagnostics": diagnostics.as_dict(),
        }
    except Exception as exc:
        diagnostics.complete(status="error")
        payload = {
            "status": "FAIL",
            "query": RETRIEVAL_QUERY,
            "generationUsed": False,
            "allowReindex": False,
            "errorType": type(exc).__name__,
            "wallLatencyMs": round((time.perf_counter() - started) * 1000, 3),
            "diagnostics": diagnostics.as_dict(),
        }
    finally:
        reset_current_diagnostics(token)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


def _error_fields(body: dict[str, Any]) -> tuple[Any, Any]:
    detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
    warning = body.get("warning") if isinstance(body.get("warning"), dict) else {}
    error_code = (
        detail.get("errorCode")
        or warning.get("errorCode")
        or body.get("errorCode")
    )
    error_stage = (
        detail.get("stage")
        or warning.get("stage")
        or body.get("failedStage")
    )
    return error_code, error_stage


def _live_row(
    client: httpx.Client,
    backend_url: str,
    run_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(
            f"{backend_url.rstrip('/')}/api/ask",
            json={"question": COMBINED_QUESTION},
        )
        wall_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        body = response.json()
        error_code, error_stage = _error_fields(body)
        return {
            "run": run_name,
            "httpStatus": response.status_code,
            "wallLatencyMs": wall_latency_ms,
            "responseStatus": body.get("status")
            or (
                body.get("detail", {}).get("status")
                if isinstance(body.get("detail"), dict)
                else None
            ),
            "route": body.get("route")
            or (
                body.get("detail", {}).get("route")
                if isinstance(body.get("detail"), dict)
                else None
            ),
            "errorCode": error_code,
            "errorStage": error_stage,
            "response": body,
        }
    except httpx.TimeoutException:
        return {
            "run": run_name,
            "httpStatus": None,
            "wallLatencyMs": round((time.perf_counter() - started) * 1000, 3),
            "responseStatus": "error",
            "route": None,
            "errorCode": "CLIENT_TIMEOUT",
            "errorStage": None,
            "response": None,
        }


def run_live(output_path: Path, backend_url: str, timeout_seconds: float) -> int:
    with httpx.Client(timeout=timeout_seconds) as client:
        cold = _live_row(client, backend_url, "cold")
        warm = _live_row(client, backend_url, "warm")

    cold_latency = cold.get("wallLatencyMs")
    warm_latency = warm.get("wallLatencyMs")
    ratio = None
    if isinstance(cold_latency, (int, float)) and isinstance(
        warm_latency,
        (int, float),
    ) and warm_latency:
        ratio = round(cold_latency / warm_latency, 4)

    payload = {
        "question": COMBINED_QUESTION,
        "backendUrl": backend_url,
        "clientTimeoutSeconds": timeout_seconds,
        "runs": [cold, warm],
        "coldWarmRatio": ratio,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    retrieval = subparsers.add_parser("retrieval")
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument("--query", default=RETRIEVAL_QUERY)

    live = subparsers.add_parser("live")
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--backend-url", default="http://localhost:8000")
    live.add_argument("--timeout-seconds", type=float, default=190.0)

    args = parser.parse_args()
    if args.phase == "retrieval":
        return run_retrieval(args.output, args.query)
    return run_live(args.output, args.backend_url, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

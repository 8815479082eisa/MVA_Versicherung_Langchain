from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import rag_service
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)


def _safe_document(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    source = dict(row.get("source") or {})
    source_value = str(
        source.get("source") or metadata.get("source") or "unknown"
    )
    return {
        "rank": row.get("rank"),
        "filename": source_value.replace("\\", "/").rsplit("/", 1)[-1],
        "source": source_value,
        "page": source.get("page") or metadata.get("page"),
        "score": metadata.get("reranker_score")
        or metadata.get("score")
        or metadata.get("relevance_score"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("without-reranker", "with-reranker"),
        required=True,
    )
    parser.add_argument(
        "--query",
        default="What does partial coverage generally cover?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    diagnostics = RequestDiagnostics(route="pure-retrieval")
    token = set_current_diagnostics(diagnostics)
    try:
        result = rag_service.retrieve_documents_for_tool(
            args.query,
            args.top_k,
            use_reranker=args.mode == "with-reranker",
        )
        diagnostics.complete(status="complete")
        payload = {
            "status": "PASSED",
            "query": args.query,
            "retrievalMode": result["retrieval_mode"],
            "generationUsed": False,
            "rerankerUsed": result["reranker_used"],
            "documentCount": result["document_count"],
            "documents": [
                _safe_document(row) for row in result["documents"]
            ],
            "diagnostics": diagnostics.as_dict(),
            "warnings": (
                []
                if result["reranker_used"] or args.mode == "without-reranker"
                else ["The configured reranker was unavailable; retrieval order was used."]
            ),
        }
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except Exception as exc:
        diagnostics.complete(status="error")
        payload = {
            "status": "FAILED",
            "query": args.query,
            "retrievalMode": args.mode,
            "generationUsed": False,
            "errorType": type(exc).__name__,
            "diagnostics": diagnostics.as_dict(),
        }
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 1
    finally:
        reset_current_diagnostics(token)


if __name__ == "__main__":
    raise SystemExit(main())

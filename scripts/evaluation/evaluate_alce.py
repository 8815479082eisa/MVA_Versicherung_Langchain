from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.alce_metrics import (  # noqa: E402
    aggregate_alce_results,
    citation_coverage,
    citation_precision,
    citation_recall,
    extract_citation_indices,
    mauve_fluency,
    split_into_statements,
)
from src.evaluation.nli import NLIModel, create_nli_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALCE-style evaluation metrics.")
    parser.add_argument("--mode", choices=["audit", "insuranceqa"], required=True)

    parser.add_argument(
        "--audit-log",
        default="data/processed/logs/audit.log",
        help="Audit JSONL input file",
    )
    parser.add_argument(
        "--insuranceqa-jsonl",
        default="data/processed/eval_outputs/insuranceqa/eval_insuranceqa_results.jsonl",
        help="InsuranceQA result JSONL input file",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of items")

    parser.add_argument(
        "--nli-backend",
        default=None,
        help="NLI backend: true_nli | bart_mnli (default from env NLI_BACKEND)",
    )
    parser.add_argument(
        "--nli-cache",
        default="data/processed/caches/nli_cache.sqlite",
        help="Path for sqlite NLI cache",
    )
    parser.add_argument(
        "--disable-mauve",
        action="store_true",
        help="Disable MAUVE computation in insuranceqa mode",
    )

    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="Per-item output JSONL path (defaults to mode-specific file)",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Summary output JSON path (defaults to mode-specific file)",
    )
    return parser.parse_args()


def _load_jsonl(path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                if max_items is not None and len(rows) >= max_items:
                    break
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _calc_latency_seconds(start_ts: Optional[str], end_ts: Optional[str]) -> Optional[float]:
    if not start_ts or not end_ts:
        return None
    try:
        start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except Exception:
        return None


def _extract_passages_from_audit_entry(entry: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    docs = entry.get("compressed_context") or entry.get("retrieved_documents") or []
    if not docs:
        docs = entry.get("sources") or []

    passages: List[Dict[str, str]] = []
    used_snippet = False
    for idx, doc in enumerate(docs, start=1):
        if not isinstance(doc, dict):
            doc = {"page_content": str(doc)}

        metadata = doc.get("metadata") or {}
        title = (
            doc.get("documentTitle")
            or doc.get("document_title")
            or doc.get("title")
            or metadata.get("source")
            or doc.get("documentId")
            or f"passage_{idx}"
        )
        text = doc.get("page_content") or doc.get("text") or doc.get("content") or ""
        if not text:
            text = doc.get("snippet") or ""
            if text:
                used_snippet = True

        passages.append({"title": str(title), "text": str(text)})

    if used_snippet:
        warnings.append(
            "Using snippet instead of full passage text; citation recall/precision may be underestimated."
        )

    return passages, warnings


def _build_cited_passages_by_statement(
    statements: Sequence[str],
    passages: Sequence[Dict[str, str]],
) -> Tuple[List[List[Dict[str, str]]], List[str]]:
    mapped: List[List[Dict[str, str]]] = []
    warnings: List[str] = []

    for statement in statements:
        indices = extract_citation_indices(statement)
        cited: List[Dict[str, str]] = []
        for citation_idx in indices:
            pos = citation_idx - 1
            if 0 <= pos < len(passages):
                cited.append(passages[pos])
            else:
                warnings.append(
                    f"Citation index [{citation_idx}] out of range for {len(passages)} passages."
                )
        mapped.append(cited)
    return mapped, warnings


def evaluate_audit_item(entry: Dict[str, Any], nli_model: NLIModel) -> Dict[str, Any]:
    question = str(entry.get("query", "")).strip()
    answer = str(entry.get("generated_answer", "")).strip()
    statements = split_into_statements(answer)
    passages, warnings = _extract_passages_from_audit_entry(entry)
    cited_by_statement, map_warnings = _build_cited_passages_by_statement(statements, passages)
    warnings.extend(map_warnings)

    marker_indices = [extract_citation_indices(s) for s in statements]
    marker_counts = [len(indices) for indices in marker_indices]
    has_inline_citations = any(marker_counts)
    coverage = citation_coverage(statements)

    recall: Optional[float] = None
    precision: Optional[float] = None
    reason: Optional[str] = None

    if not statements:
        reason = "no_statements"
    elif not has_inline_citations:
        reason = "no_inline_citation_markers"
    elif not getattr(nli_model, "available", True):
        reason = f"nli_unavailable: {getattr(nli_model, 'unavailable_reason', 'unknown')}"
    else:
        try:
            recall = citation_recall(statements, cited_by_statement, nli_model)
            precision = citation_precision(statements, cited_by_statement, nli_model)
        except Exception as exc:
            reason = f"nli_error: {type(exc).__name__}: {exc}"

    latency_seconds = _calc_latency_seconds(
        entry.get("start_timestamp") or entry.get("timestamp"),
        entry.get("end_timestamp") or entry.get("timestamp"),
    )

    return {
        "mode": "audit",
        "question": question,
        "answer": answer,
        "statement_count": len(statements),
        "passage_count": len(passages),
        "citation_indices_per_statement": marker_indices,
        "citation_markers_per_statement": marker_counts,
        "citation_coverage": coverage,
        "citation_recall": recall,
        "citation_precision": precision,
        "citation_reason": reason,
        "latency_seconds": latency_seconds,
        "warnings": sorted(set(warnings)),
    }


def evaluate_insuranceqa_item(
    entry: Dict[str, Any],
    disable_mauve: bool,
) -> Dict[str, Any]:
    question = str(entry.get("question", "")).strip()
    prediction = str(entry.get("prediction", "")).strip()
    refs = entry.get("refs") or []
    ref_answer = str(refs[0]).strip() if refs else ""
    error = entry.get("error")

    statements = split_into_statements(prediction)
    coverage = citation_coverage(statements)

    mauve_score: Optional[float] = None
    mauve_reason: Optional[str] = None
    if error:
        mauve_reason = "prediction_error"
    elif disable_mauve:
        mauve_reason = "disabled"
    elif not ref_answer:
        mauve_reason = "missing_reference"
    else:
        mauve_score = mauve_fluency(question=question, gen_answer=prediction, ref_answer=ref_answer)
        if mauve_score is None:
            mauve_reason = "mauve_unavailable_or_failed"

    return {
        "mode": "insuranceqa",
        "idx": entry.get("idx"),
        "question": question,
        "prediction": prediction,
        "statement_count": len(statements),
        "citation_coverage": coverage,
        "mauve_fluency": mauve_score,
        "mauve_reason": mauve_reason,
        "em": entry.get("em"),
        "best_f1": entry.get("best_f1"),
        "error": error,
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.mean(values)


def summarize(items: Sequence[Dict[str, Any]], mode: str, nli_model: Optional[NLIModel]) -> Dict[str, Any]:
    summary = aggregate_alce_results(items)
    summary["mode"] = mode
    summary["created_at"] = datetime.now().isoformat()

    if mode == "audit":
        with_inline = sum(
            1 for i in items if float(i.get("citation_coverage") or 0.0) > 0.0
        )
        summary["items_with_inline_citations"] = with_inline
        summary["items_without_inline_citations"] = len(items) - with_inline
        summary["nli_backend"] = getattr(nli_model, "backend", None)
        summary["nli_available"] = getattr(nli_model, "available", None)
        summary["nli_unavailable_reason"] = getattr(nli_model, "unavailable_reason", None)
    else:
        em = [float(i["em"]) for i in items if i.get("em") is not None]
        f1 = [float(i["best_f1"]) for i in items if i.get("best_f1") is not None]
        summary["exact_match_mean"] = _mean(em)
        summary["token_f1_mean"] = _mean(f1)

    return summary


def resolve_output_paths(
    mode: str,
    out_jsonl_arg: Optional[str],
    summary_out_arg: Optional[str],
) -> Tuple[Path, Path]:
    if out_jsonl_arg:
        out_jsonl = Path(out_jsonl_arg)
    else:
        out_jsonl = Path(
            f"data/processed/eval_outputs/alce/evaluate_alce_{mode}_items.jsonl"
        )

    if summary_out_arg:
        summary_out = Path(summary_out_arg)
    else:
        summary_out = Path(
            f"data/processed/eval_outputs/alce/evaluate_alce_{mode}_summary.json"
        )

    return out_jsonl, summary_out


def run_audit_mode(args: argparse.Namespace) -> int:
    input_path = Path(args.audit_log)
    items_raw = _load_jsonl(input_path, max_items=args.max_items)
    if not items_raw:
        print(f"No rows found in {input_path}")
        return 1

    nli_model = create_nli_model(backend=args.nli_backend, cache_path=args.nli_cache)
    rows = [evaluate_audit_item(entry, nli_model=nli_model) for entry in items_raw]
    summary = summarize(rows, mode="audit", nli_model=nli_model)

    out_jsonl, summary_out = resolve_output_paths(
        mode="audit",
        out_jsonl_arg=args.out_jsonl,
        summary_out_arg=args.summary_out,
    )
    _write_jsonl(out_jsonl, rows)
    _write_json(summary_out, summary)

    print(f"Wrote {len(rows)} items to {out_jsonl}")
    print(f"Wrote summary to {summary_out}")
    return 0


def run_insuranceqa_mode(args: argparse.Namespace) -> int:
    input_path = Path(args.insuranceqa_jsonl)
    items_raw = _load_jsonl(input_path, max_items=args.max_items)
    if not items_raw:
        print(f"No rows found in {input_path}")
        return 1

    rows = [
        evaluate_insuranceqa_item(entry, disable_mauve=args.disable_mauve)
        for entry in items_raw
    ]
    summary = summarize(rows, mode="insuranceqa", nli_model=None)

    out_jsonl, summary_out = resolve_output_paths(
        mode="insuranceqa",
        out_jsonl_arg=args.out_jsonl,
        summary_out_arg=args.summary_out,
    )
    _write_jsonl(out_jsonl, rows)
    _write_json(summary_out, summary)

    print(f"Wrote {len(rows)} items to {out_jsonl}")
    print(f"Wrote summary to {summary_out}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "audit":
        return run_audit_mode(args)
    return run_insuranceqa_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())

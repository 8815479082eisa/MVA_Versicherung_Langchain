"""Export eval-style InsuranceQA JSONL from audit.log entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_utils import best_token_f1, exact_match, load_question_answer_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eval_insuranceqa_results.jsonl from audit.log rows."
    )
    parser.add_argument(
        "--audit-log",
        default="data/processed/logs/audit.log",
        help="Path to audit.log JSONL file.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        default="data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl",
        help="InsuranceQA benchmark JSONL used as reference answers.",
    )
    parser.add_argument(
        "--timestamp-prefix",
        default=None,
        help="Keep rows where timestamp starts with this prefix (example: 2026-03-26T23:).",
    )
    parser.add_argument(
        "--start-timestamp",
        default=None,
        help="Keep rows with timestamp >= this value (ISO lexical compare).",
    )
    parser.add_argument(
        "--end-timestamp",
        default=None,
        help="Keep rows with timestamp <= this value (ISO lexical compare).",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id to write into each eval line.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output eval JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional summary JSON output path.",
    )
    return parser.parse_args()


def extract_fallback_ref(retrieved_documents: Any) -> str | None:
    if not isinstance(retrieved_documents, list):
        return None
    for doc in retrieved_documents:
        if not isinstance(doc, dict):
            continue
        page_content = str(doc.get("page_content") or "")
        if "\nAnswer:" in page_content:
            return page_content.split("\nAnswer:", 1)[1].strip()
    return None


def row_in_scope(
    ts: str,
    timestamp_prefix: str | None,
    start_ts: str | None,
    end_ts: str | None,
) -> bool:
    if timestamp_prefix and not ts.startswith(timestamp_prefix):
        return False
    if start_ts and ts < start_ts:
        return False
    if end_ts and ts > end_ts:
        return False
    return True


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_log)
    dataset_path = Path(args.dataset_jsonl)
    out_path = Path(args.out)
    summary_out = Path(args.summary_out) if args.summary_out else None

    q2answers = load_question_answer_map(dataset_path)

    rows: list[dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = str(row.get("timestamp") or "")
            if not row_in_scope(
                ts=ts,
                timestamp_prefix=args.timestamp_prefix,
                start_ts=args.start_timestamp,
                end_ts=args.end_timestamp,
            ):
                continue
            rows.append(row)

    rows.sort(key=lambda item: str(item.get("timestamp") or ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    em_hits = 0
    f1_sum = 0.0

    with out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            question = str(row.get("query") or "").strip()
            prediction = str(row.get("generated_answer") or "").strip()
            refs = q2answers.get(question) or []
            if not refs:
                fallback_ref = extract_fallback_ref(row.get("retrieved_documents"))
                if fallback_ref:
                    refs = [fallback_ref]

            if not question or not prediction or not refs:
                failed += 1
                out.write(
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "idx": idx,
                            "question": question,
                            "error": "missing_question_or_prediction_or_reference",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            em = int(exact_match(prediction, refs))
            best_f1 = best_token_f1(prediction, refs)

            source_ids: list[str] = []
            for source in row.get("sources") or []:
                if isinstance(source, dict):
                    source_id = str(
                        source.get("document_id") or source.get("documentId") or ""
                    ).strip()
                    if source_id:
                        source_ids.append(source_id)

            out.write(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "idx": idx,
                        "question": question,
                        "refs": refs,
                        "prediction": prediction,
                        "source_ids": source_ids,
                        "em": em,
                        "best_f1": best_f1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed += 1
            em_hits += em
            f1_sum += best_f1

    exact_match_avg = (em_hits / processed) if processed else 0.0
    token_f1_avg = (f1_sum / processed) if processed else 0.0

    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "result_label": out_path.parent.name,
            "run_id": args.run_id,
            "dataset_source": str(dataset_path).replace("/", "\\"),
            "audit_log_source": str(audit_path).replace("/", "\\"),
            "audit_timestamp_prefix": args.timestamp_prefix,
            "audit_start_timestamp": args.start_timestamp,
            "audit_end_timestamp": args.end_timestamp,
            "processed": processed,
            "failed": failed,
            "exact_match": exact_match_avg,
            "token_f1": token_f1_avg,
        }
        with summary_out.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Export complete")
    print(f"rows_selected={len(rows)} processed={processed} failed={failed}")
    print(f"exact_match={exact_match_avg:.6f} token_f1={token_f1_avg:.6f}")
    print(f"output={out_path}")
    if summary_out:
        print(f"summary={summary_out}")


if __name__ == "__main__":
    main()

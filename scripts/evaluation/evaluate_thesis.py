"""Run the unified 8-metric thesis evaluation framework."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.models import (
    build_answer_model_warning,
    load_model_settings,
    runtime_config_snapshot,
)
from src.evaluation.benchmark_utils import (
    dump_question_answer_pairs,
    load_question_answer_map,
    load_question_answer_pairs,
    sample_question_answer_pairs,
    slice_question_answer_pairs,
)
from src.evaluation.security_eval import (
    evaluate_security_cases,
    load_security_cases,
    write_security_rows,
)
from src.evaluation.thesis_metrics import aggregate_qa_metric_rows, build_qa_metric_row


QA_DATASET_DEFAULT = "data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl"
SAFETY_DATASET_DEFAULT = "data/benchmarks/safety/direct_query_attacks_20.jsonl"
AUDIT_LOG_DEFAULT = "data/processed/logs/audit.log"

THESIS_METRIC_CATALOG = {
    "retrieval_support_hit_rate": {
        "area": "retrieval",
        "title": "Retrieval Support Hit Rate",
        "description": "Share of questions where at least one final context passage overlaps strongly enough with a reference answer to be considered answer-supporting.",
    },
    "retrieval_context_precision": {
        "area": "retrieval",
        "title": "Retrieval Context Precision",
        "description": "Average share of final context passages that appear answer-supporting for the benchmark question.",
    },
    "answer_exact_match": {
        "area": "generated_answer",
        "title": "Answer Exact Match",
        "description": "Strict normalized exact-match score between prediction and any benchmark reference answer.",
    },
    "answer_token_f1": {
        "area": "generated_answer",
        "title": "Answer Token F1",
        "description": "Token-overlap F1 against the best matching benchmark reference answer.",
    },
    "citation_source_presence_rate": {
        "area": "citation",
        "title": "Citation Source Presence Rate",
        "description": "Share of answers that return at least one user-visible source entry.",
    },
    "citation_support_rate": {
        "area": "citation",
        "title": "Citation Support Rate",
        "description": "Average share of answer statements that are supported by the returned context/source passages using a deterministic lexical support heuristic.",
    },
    "attack_block_rate": {
        "area": "security",
        "title": "Attack Block Rate",
        "description": "Share of attack-labeled direct queries that the safety layer blocks or converts into a fallback.",
    },
    "benign_allow_rate": {
        "area": "security",
        "title": "Benign Allow Rate",
        "description": "Share of benign direct queries that the safety layer allows without blocking or fallback.",
    },
}


@dataclass
class AuditCapture:
    rows: List[Dict[str, Any]]
    last_row: Optional[Dict[str, Any]]


class AuditLogTailer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.offset = self.path.stat().st_size if self.path.exists() else 0

    def read_new_rows(self) -> AuditCapture:
        if not self.path.exists():
            return AuditCapture(rows=[], last_row=None)

        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self.offset = handle.tell()
        return AuditCapture(rows=rows, last_row=rows[-1] if rows else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thesis 8-metric evaluation framework.")
    parser.add_argument("--mode", choices=["full", "qa", "security"], default="full")
    parser.add_argument("--qa-mode", choices=["live", "audit"], default="live")
    parser.add_argument("--dataset-jsonl", default=QA_DATASET_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--audit-log", default=AUDIT_LOG_DEFAULT)
    parser.add_argument("--audit-start-timestamp", default=None)
    parser.add_argument("--audit-end-timestamp", default=None)
    parser.add_argument("--audit-limit", type=int, default=None)
    parser.add_argument("--safety-file", default=SAFETY_DATASET_DEFAULT)
    parser.add_argument("--support-threshold", type=float, default=0.2)
    parser.add_argument("--allow-dataset-shortcut", action="store_true")
    parser.add_argument("--out-dir", default=None)
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


def row_in_scope(ts: str, start_ts: str | None, end_ts: str | None) -> bool:
    if start_ts and ts < start_ts:
        return False
    if end_ts and ts > end_ts:
        return False
    return True


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_git_command(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(args),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "git_not_available"
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_git_commit() -> str:
    return _run_git_command(["git", "rev-parse", "HEAD"])


def get_git_branch() -> str:
    return _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def write_git_metadata(out_dir: Path, *, git_commit: str, git_branch: str) -> None:
    (out_dir / "commit-hash.txt").write_text(git_commit, encoding="utf-8")
    (out_dir / "branch.txt").write_text(git_branch, encoding="utf-8")


def summarize_qa_audit_safety(audit_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    decision_counts: Counter[str] = Counter()
    block_reason_counts: Counter[str] = Counter()
    fallback_category_counts: Counter[str] = Counter()

    for row in audit_rows:
        decision = str(row.get("safety_decision") or "").strip()
        if decision:
            decision_counts[decision] += 1

        block_reason = str(row.get("safety_block_reason") or "").strip()
        if block_reason:
            block_reason_counts[block_reason] += 1

        fallback_category = str(row.get("safety_fallback_category") or "").strip()
        if fallback_category:
            fallback_category_counts[fallback_category] += 1

    return {
        "decision_counts": dict(decision_counts),
        "fallback_reason_counts": dict(block_reason_counts),
        "fallback_category_counts": dict(fallback_category_counts),
    }


def requested_live_qa_count(args: argparse.Namespace) -> int:
    if args.count is not None:
        return max(0, args.count)
    return max(0, args.max_samples)


def mark_run_completion(metadata: Dict[str, Any]) -> None:
    reasons: List[str] = []

    expected_qa_count = metadata.get("expected_qa_count")
    selected_qa_count = metadata.get("selected_qa_count")
    processed_qa_count = metadata.get("processed_qa_count")
    if expected_qa_count is not None:
        if selected_qa_count is not None and selected_qa_count < expected_qa_count:
            reasons.append(
                f"QA selection incomplete: expected {expected_qa_count}, selected {selected_qa_count}."
            )
        if processed_qa_count is not None and processed_qa_count < expected_qa_count:
            reasons.append(
                f"QA processing incomplete: expected {expected_qa_count}, processed {processed_qa_count}."
            )

    expected_safety_count = metadata.get("expected_safety_count")
    processed_safety_count = metadata.get("processed_safety_count")
    if expected_safety_count is not None and processed_safety_count is not None:
        if processed_safety_count < expected_safety_count:
            reasons.append(
                f"Safety processing incomplete: expected {expected_safety_count}, processed {processed_safety_count}."
            )

    metadata["run_complete"] = not reasons
    metadata["run_status"] = "complete" if not reasons else "incomplete"
    metadata["run_incomplete_reasons"] = reasons


def build_run_metadata(
    *,
    args: argparse.Namespace,
    settings: Any,
    run_id: str,
) -> Dict[str, Any]:
    runtime = runtime_config_snapshot(settings)
    answer_model_matches_preference = runtime["answer_model_matches_preference"]
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "qa_mode": args.qa_mode,
        "configured_answer_model": settings.roles.answer,
        "configured_answer_model_source": runtime["configured_answer_model_source"],
        "configured_answer_model_source_detail": runtime.get(
            "configured_answer_model_source_detail"
        ),
        "preferred_answer_model": settings.preferred_answer_model,
        "preferred_answer_model_source": runtime["preferred_answer_model_source"],
        "answer_model_matches_preference": answer_model_matches_preference,
        "source_of_answer_model": runtime["source_of_answer_model"],
        "answer_model_warning": build_answer_model_warning(settings),
        "router_model": settings.roles.router,
        "router_model_source": runtime["router_model_source"],
        "self_check_model": settings.roles.self_check,
        "query_rewrite_model": settings.roles.rewrite,
        "query_rewrite_enabled": settings.retrieval.query_rewrite_enabled,
        "query_rewrite_min_similarity": settings.retrieval.query_rewrite_min_similarity,
        "runtime_env": runtime["raw_env"],
        "dotenv_conflicts": runtime.get("dotenv_conflicts", {}),
        "dataset_jsonl": args.dataset_jsonl,
        "max_samples": args.max_samples,
        "start": args.start,
        "count": args.count,
        "audit_log": args.audit_log,
        "audit_limit": args.audit_limit,
        "safety_file": args.safety_file,
        "support_threshold": args.support_threshold,
        "allow_dataset_shortcut": args.allow_dataset_shortcut,
        "retrieval_top_k": settings.retrieval.top_k,
        "reranker_top_k": settings.reranker.top_k,
        "safety_backend": settings.safety.backend,
        "safety_mode": settings.safety.mode,
        "safety_min_groundedness": settings.safety.min_groundedness,
        "nemo_input_enabled": settings.safety.nemo_input_enabled,
        "nemo_context_enabled": settings.safety.nemo_context_enabled,
        "nemo_output_enabled": settings.safety.nemo_output_enabled,
        "nemo_enforce_input": settings.safety.nemo_enforce_input,
        "nemo_enforce_output": settings.safety.nemo_enforce_output,
        "safety_fallback_texts": {
            "security": settings.safety.security_fallback_text or settings.safety.fallback_text,
            "grounding": settings.safety.grounding_fallback_text,
            "pii": settings.safety.pii_fallback_text,
            "context": settings.safety.context_fallback_text,
        },
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "metric_catalog": THESIS_METRIC_CATALOG,
    }


def find_audit_row(rows: Sequence[Dict[str, Any]], question: str) -> Optional[Dict[str, Any]]:
    for row in reversed(rows):
        if str(row.get("query") or "").strip() == question:
            return row
    return rows[-1] if rows else None


def run_live_qa_eval(
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, List[str]]], Dict[str, Any]]:
    from src.api import rag_service

    items_all = load_question_answer_pairs(args.dataset_jsonl)
    sampled = sample_question_answer_pairs(
        items_all,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    items_run, _, _ = slice_question_answer_pairs(
        sampled,
        start=args.start,
        count=args.count,
    )

    tailer = AuditLogTailer(args.audit_log)
    qa_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for idx, (question, refs) in enumerate(items_run):
        captured_audit: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        prediction = ""

        try:
            result = rag_service.run_rag(question, chat_history=[])
            capture = tailer.read_new_rows()
            captured_audit = find_audit_row(capture.rows, question)
            if captured_audit is None:
                raise RuntimeError("No matching audit row captured for live QA evaluation.")
            if (not args.allow_dataset_shortcut) and captured_audit.get("exact_insuranceqa_match"):
                raise RuntimeError(
                    "Detected direct InsuranceQA dataset shortcut answer. "
                    "Disable the shortcut for thesis measurements."
                )
            prediction = str(captured_audit.get("generated_answer") or result.answer or "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if captured_audit is None:
            captured_audit = {
                "query": question,
                "generated_answer": prediction,
                "compressed_context": [],
                "retrieved_documents": [],
                "sources": [],
                "latency_ms": None,
                "retries": None,
                "retrieval_needed": None,
                "safety_decision": None,
            }

        context_docs = (
            captured_audit.get("compressed_context")
            or captured_audit.get("retrieved_documents")
            or []
        )
        sources = captured_audit.get("sources") or []
        row = build_qa_metric_row(
            question=question,
            references=refs,
            prediction=str(captured_audit.get("generated_answer") or prediction or ""),
            context_docs=context_docs,
            sources=sources,
            audit_row=captured_audit,
            support_threshold=args.support_threshold,
            error=error,
        )
        row["idx"] = idx
        row["evaluation_mode"] = "live"
        qa_rows.append(row)
        audit_rows.append(captured_audit)

    selection = {
        "qa_dataset_total": len(items_all),
        "requested_qa_count": requested_live_qa_count(args),
        "expected_qa_count": min(requested_live_qa_count(args), len(items_all)),
        "sampled_qa_count": len(sampled),
        "selected_qa_count": len(items_run),
    }
    return qa_rows, audit_rows, items_run, selection


def run_audit_qa_eval(
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, List[str]]], Dict[str, Any]]:
    q2answers = load_question_answer_map(args.dataset_jsonl)
    audit_path = Path(args.audit_log)
    selected_rows: List[Dict[str, Any]] = []

    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(row.get("query") or "").strip()
            if not question:
                continue
            if not row_in_scope(
                str(row.get("timestamp") or ""),
                args.audit_start_timestamp,
                args.audit_end_timestamp,
            ):
                continue

            refs = q2answers.get(question) or []
            if not refs:
                fallback_ref = extract_fallback_ref(row.get("retrieved_documents"))
                if fallback_ref:
                    refs = [fallback_ref]
            if not refs:
                continue

            selected_rows.append(row)
            if args.audit_limit is not None and len(selected_rows) >= args.audit_limit:
                break

    qa_rows: List[Dict[str, Any]] = []
    qa_items: List[Tuple[str, List[str]]] = []
    for idx, audit_row in enumerate(selected_rows):
        question = str(audit_row.get("query") or "").strip()
        refs = q2answers.get(question) or []
        if not refs:
            fallback_ref = extract_fallback_ref(audit_row.get("retrieved_documents"))
            if fallback_ref:
                refs = [fallback_ref]

        error = None
        if (not args.allow_dataset_shortcut) and audit_row.get("exact_insuranceqa_match"):
            error = (
                "Detected direct InsuranceQA dataset shortcut answer in audit row. "
                "Disable the shortcut for thesis measurements."
            )

        context_docs = audit_row.get("compressed_context") or audit_row.get("retrieved_documents") or []
        sources = audit_row.get("sources") or []
        row = build_qa_metric_row(
            question=question,
            references=refs,
            prediction=str(audit_row.get("generated_answer") or ""),
            context_docs=context_docs,
            sources=sources,
            audit_row=audit_row,
            support_threshold=args.support_threshold,
            error=error,
        )
        row["idx"] = idx
        row["evaluation_mode"] = "audit"
        qa_rows.append(row)
        qa_items.append((question, refs))

    requested_qa_count = args.audit_limit if args.audit_limit is not None else len(selected_rows)
    selection = {
        "qa_dataset_total": len(q2answers),
        "requested_qa_count": requested_qa_count,
        "expected_qa_count": requested_qa_count,
        "sampled_qa_count": None,
        "selected_qa_count": len(selected_rows),
    }
    return qa_rows, selected_rows, qa_items, selection


def build_summary_markdown(
    *,
    run_id: str,
    metadata: Dict[str, Any],
    qa_summary: Optional[Dict[str, Any]],
    security_summary: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "# Thesis Evaluation Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at: `{metadata['created_at_utc']}`",
        f"- Run status: `{metadata.get('run_status', 'unknown')}`",
        f"- Answer model: `{metadata['configured_answer_model']}`",
        f"- Answer model source: `{metadata.get('configured_answer_model_source')}`",
        f"- Preferred answer model: `{metadata['preferred_answer_model']}`",
        f"- QA mode: `{metadata.get('qa_mode')}`",
        "",
        "## Metric Summary",
        "",
        "| Area | Metric | Value |",
        "|---|---|---:|",
    ]

    if qa_summary is not None:
        qa_metrics = qa_summary["metrics"]
        lines.extend(
            [
                f"| Retrieval | Retrieval Support Hit Rate | {qa_metrics['retrieval_support_hit_rate'] or 0.0:.4f} |",
                f"| Retrieval | Retrieval Context Precision | {qa_metrics['retrieval_context_precision'] or 0.0:.4f} |",
                f"| Generated Answer | Exact Match | {qa_metrics['answer_exact_match'] or 0.0:.4f} |",
                f"| Generated Answer | Token F1 | {qa_metrics['answer_token_f1'] or 0.0:.4f} |",
                f"| Citation | Source Presence Rate | {qa_metrics['citation_source_presence_rate'] or 0.0:.4f} |",
                f"| Citation | Citation Support Rate | {qa_metrics['citation_support_rate'] or 0.0:.4f} |",
            ]
        )

    if security_summary is not None:
        lines.extend(
            [
                f"| Security | Attack Block Rate | {security_summary['attack_block_rate']:.4f} |",
                f"| Security | Benign Allow Rate | {security_summary['benign_allow_rate']:.4f} |",
            ]
        )

    if qa_summary is not None:
        diagnostics = qa_summary["diagnostics"]
        lines.extend(
            [
                "",
                "## Runtime Config",
                "",
                f"- Query rewrite enabled: {metadata.get('query_rewrite_enabled')}",
                f"- Output enforcement enabled: {metadata.get('nemo_enforce_output')}",
                f"- Minimum groundedness: {metadata.get('safety_min_groundedness')}",
            ]
        )

        if metadata.get("dotenv_conflicts"):
            lines.append(f"- .env conflicts overridden by shell env: {sorted(metadata['dotenv_conflicts'])}")

        fallback_reason_counts = (
            metadata.get("qa_audit_safety_summary", {}) or {}
        ).get("fallback_reason_counts", {})
        if fallback_reason_counts:
            formatted_reasons = ", ".join(
                f"{reason}={count}" for reason, count in sorted(fallback_reason_counts.items())
            )
            lines.extend(
                [
                    "",
                    "## Safety Outcome Breakdown",
                    "",
                    f"- QA fallback reasons: {formatted_reasons}",
                ]
            )

        lines.extend(
            [
                "",
                "## QA Diagnostics",
                "",
                f"- Requested QA items: {metadata.get('requested_qa_count')}",
                f"- Expected QA items: {metadata.get('expected_qa_count')}",
                f"- Selected QA items: {metadata.get('selected_qa_count')}",
                f"- Processed QA items: {diagnostics['processed_items']}",
                f"- Failed QA items: {diagnostics['failed_items']}",
                f"- Fallback answers: {diagnostics['fallback_answers']}",
                f"- Mean inline citation coverage: {(diagnostics['inline_citation_coverage_mean'] or 0.0):.4f}",
            ]
        )

    if security_summary is not None:
        lines.extend(
            [
                "",
                "## Security Diagnostics",
                "",
                f"- Expected safety cases: {metadata.get('expected_safety_count')}",
                f"- Processed safety cases: {metadata.get('processed_safety_count')}",
                f"- Attack cases: {security_summary['total_attack']}",
                f"- Benign cases: {security_summary['total_benign']}",
                f"- False negatives: {security_summary['false_negative_count']}",
                f"- False positives: {security_summary['false_positive_count']}",
            ]
        )

    incomplete_reasons = metadata.get("run_incomplete_reasons") or []
    if incomplete_reasons:
        lines.extend(
            [
                "",
                "## Incomplete Run Notes",
                "",
                *[f"- {reason}" for reason in incomplete_reasons],
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    settings = load_model_settings()
    runtime = runtime_config_snapshot(settings)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("artifacts/test-results") / f"test-result-{run_id}-thesis-8-metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_summary: Optional[Dict[str, Any]] = None
    security_summary: Optional[Dict[str, Any]] = None

    metadata = build_run_metadata(args=args, settings=settings, run_id=run_id)
    print(
        "Runtime config: "
        f"configured_answer_model={metadata['configured_answer_model']} "
        f"(source={metadata['configured_answer_model_source']}), "
        f"preferred_answer_model={metadata['preferred_answer_model']}, "
        f"answer_model_matches_preference={metadata['answer_model_matches_preference']}, "
        f"query_rewrite_enabled={metadata['query_rewrite_enabled']}, "
        f"nemo_enforce_output={metadata['nemo_enforce_output']}, "
        f"safety_backend={metadata['safety_backend']}"
    )
    if metadata.get("answer_model_warning"):
        print(f"Warning: {metadata['answer_model_warning']}")
    if runtime.get("dotenv_conflicts"):
        print(f"Config note: shell env overrides .env for {sorted(runtime['dotenv_conflicts'])}")

    if args.mode in {"full", "qa"}:
        if args.qa_mode == "live":
            qa_rows, qa_audit_rows, qa_items, qa_selection = run_live_qa_eval(args)
        else:
            qa_rows, qa_audit_rows, qa_items, qa_selection = run_audit_qa_eval(args)

        qa_summary = aggregate_qa_metric_rows(qa_rows)
        metadata.update(qa_selection)
        metadata["processed_qa_count"] = qa_summary["diagnostics"]["processed_items"]
        metadata["failed_qa_count"] = qa_summary["diagnostics"]["failed_items"]
        metadata["qa_audit_safety_summary"] = summarize_qa_audit_safety(qa_audit_rows)
        write_jsonl(out_dir / "qa_items.jsonl", qa_rows)
        write_jsonl(out_dir / "qa_audit_rows.jsonl", qa_audit_rows)
        dump_question_answer_pairs(out_dir / "setup" / "qa_sample.jsonl", qa_items)

    if args.mode in {"full", "security"}:
        security_cases = load_security_cases(args.safety_file)
        security_rows, security_summary, security_metadata = evaluate_security_cases(security_cases)
        metadata["expected_safety_count"] = len(security_cases)
        metadata["processed_safety_count"] = len(security_rows)
        write_security_rows(out_dir / "security_items.jsonl", security_rows)
        write_json(
            out_dir / "setup" / "security_cases.json",
            {"cases": security_cases},
        )
        metadata["security_eval_metadata"] = security_metadata

    mark_run_completion(metadata)

    summary_payload = {
        "metadata": metadata,
        "qa_summary": qa_summary,
        "security_summary": security_summary,
    }

    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "summary.json", summary_payload)
    write_git_metadata(
        out_dir,
        git_commit=metadata["git_commit"],
        git_branch=metadata["git_branch"],
    )
    (out_dir / "run-command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        build_summary_markdown(
            run_id=run_id,
            metadata=metadata,
            qa_summary=qa_summary,
            security_summary=security_summary,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

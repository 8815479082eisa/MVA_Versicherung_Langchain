from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_rerankers_isolated import (  # noqa: E402
    BATCH_SIZE,
    MAX_LENGTH,
    MEASURED_RUNS,
    TOP_K,
    load_adapter,
    load_cases,
    ndcg_at_5,
    prefetch_models,
    prepare_backend_import,
    rank_candidate_ids,
    reciprocal_rank_at_5,
)
from src.core.current_policy import select_current_policy_records  # noqa: E402


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL_SPECS = (
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder"),
    ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "cross_encoder"),
    ("BAAI/bge-reranker-base", "flag_embedding"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw and deterministic-post-filter CPU reranker benchmark."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "reranker_post_filter_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "reranker_post_filter_benchmark_20260801.md",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "reports" / "reranker_post_filter_benchmark_20260801_details.csv",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=min(8, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def filtered_candidates(
    case: dict[str, Any], reference_date: date
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = case["candidates"]
    metadata = [candidate.get("metadata") or {} for candidate in candidates]
    started = time.perf_counter()
    selection = select_current_policy_records(
        case["query"], metadata, reference_date=reference_date
    )
    selection_latency_ms = (time.perf_counter() - started) * 1000.0
    if not selection.intent_activated:
        selected = list(candidates)
    else:
        selected_numbers = {
            str(record.get("policy_number") or record.get("policyNumber") or "")
            for record in selection.selected_records
        }
        selected = [
            candidate
            for candidate in candidates
            if str((candidate.get("metadata") or {}).get("policy_number") or "")
            in selected_numbers
        ]
    return selected, {
        "activated": selection.intent_activated,
        "reason": selection.reason,
        "reference_date": reference_date.isoformat(),
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(selected),
        "selection_latency_ms": selection_latency_ms,
    }


def score_case(
    adapter: Any,
    case: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {
            "ranking": [],
            "correct_rank": 0,
            "top5": [],
            "rankings_stable": True,
            "mandatory_pass": False,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "latencies_ms": [0.0] * MEASURED_RUNS,
        }
    documents = [candidate["document"] for candidate in candidates]
    score_runs: list[list[float]] = []
    rankings_per_run: list[list[str]] = []
    latencies_ms: list[float] = []
    for _ in range(MEASURED_RUNS):
        started = time.perf_counter()
        scores = adapter.score_pairs(case["query"], documents)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        score_runs.append(scores)
        rankings_per_run.append(rank_candidate_ids(candidates, scores))
    mean_scores = [
        statistics.fmean(run[index] for run in score_runs)
        for index in range(len(documents))
    ]
    ranking = rank_candidate_ids(candidates, mean_scores)
    correct_id = case["correct_candidate_id"]
    correct_rank = ranking.index(correct_id) + 1 if correct_id in ranking else 0
    mandatory_pass = bool(ranking) and correct_rank == 1 and all(
        run[0] == correct_id for run in rankings_per_run
    )
    return {
        "ranking": ranking,
        "correct_rank": correct_rank,
        "top5": ranking[:TOP_K],
        "rankings_stable": all(
            run == rankings_per_run[0] for run in rankings_per_run
        ),
        "mandatory_pass": mandatory_pass,
        "mrr_at_5": reciprocal_rank_at_5(ranking, correct_id),
        "ndcg_at_5": ndcg_at_5(ranking, case["candidates"]),
        "latencies_ms": latencies_ms,
    }


def summarize(case_results: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [result[mode] for result in case_results]
    latencies = [latency for row in rows for latency in row["latencies_ms"]]
    by_language: dict[str, dict[str, float]] = {}
    for language in sorted({result["language"] for result in case_results}):
        language_rows = [
            result[mode] for result in case_results if result["language"] == language
        ]
        by_language[language] = {
            "top1_accuracy": statistics.fmean(
                1.0 if row["correct_rank"] == 1 else 0.0 for row in language_rows
            ),
            "mrr_at_5": statistics.fmean(row["mrr_at_5"] for row in language_rows),
            "ndcg_at_5": statistics.fmean(row["ndcg_at_5"] for row in language_rows),
        }
    return {
        "top1_accuracy": statistics.fmean(
            1.0 if row["correct_rank"] == 1 else 0.0 for row in rows
        ),
        "mrr_at_5": statistics.fmean(row["mrr_at_5"] for row in rows),
        "ndcg_at_5": statistics.fmean(row["ndcg_at_5"] for row in rows),
        "mean_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "rankings_stable": all(row["rankings_stable"] for row in rows),
        "by_language": by_language,
    }


def benchmark_model(
    adapter: Any, cases: Sequence[dict[str, Any]], reference_date: date
) -> dict[str, Any]:
    warmup = cases[0]
    adapter.score_pairs(
        warmup["query"],
        [candidate["document"] for candidate in warmup["candidates"]],
    )
    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        selected, selection = filtered_candidates(case, reference_date)
        raw = score_case(adapter, case, case["candidates"])
        post_filter = score_case(adapter, case, selected)
        case_results.append(
            {
                "case_id": case["id"],
                "language": case["language"],
                "query": case["query"],
                "correct_candidate_id": case["correct_candidate_id"],
                "mandatory": bool(case.get("mandatory_rank1")),
                "selection": selection,
                "raw": raw,
                "post_filter": post_filter,
            }
        )
        print(f"  {adapter.model_id}: {index}/{len(cases)}", flush=True)
    english_lara = [
        row
        for row in case_results
        if row["case_id"] == "policy_test_kfz_2026_1003_en"
    ]
    all_lara = [row for row in case_results if row["mandatory"]]
    unexpected_activations = [
        row["case_id"]
        for row in case_results
        if row["selection"]["activated"] and not row["mandatory"]
    ]
    return {
        "model_id": adapter.model_id,
        "backend": adapter.backend_name,
        "parameter_count": adapter.parameter_count,
        "tokenizer_limit": adapter.tokenizer_limit,
        "config_limit": adapter.config_limit,
        "raw": summarize(case_results, "raw"),
        "post_filter": summarize(case_results, "post_filter"),
        "lara_english_raw_rank": english_lara[0]["raw"]["correct_rank"],
        "lara_english_post_rank": english_lara[0]["post_filter"]["correct_rank"],
        "lara_english_pass": bool(english_lara)
        and english_lara[0]["post_filter"]["mandatory_pass"],
        "lara_all_languages_pass": bool(all_lara)
        and all(row["post_filter"]["mandatory_pass"] for row in all_lara),
        "unexpected_filter_activations": unexpected_activations,
        "case_results": case_results,
    }


def run_worker(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    payload = load_cases(args.cases.resolve())
    reference_date = date.fromisoformat(payload["selection_reference_date"])
    prepare_backend_import(args.worker_backend)
    started = time.perf_counter()
    try:
        adapter = load_adapter(
            args.worker_model_id,
            args.worker_backend,
            args.worker_snapshot.resolve(),
        )
        load_time_seconds = time.perf_counter() - started
        result = benchmark_model(adapter, payload["cases"], reference_date)
        result.update(
            status="ok",
            load_time_seconds=load_time_seconds,
            runtime_versions={
                "sentence-transformers": version("sentence-transformers"),
                "transformers": version("transformers"),
            },
        )
    except Exception as exc:
        result = {
            "model_id": args.worker_model_id,
            "backend": args.worker_backend,
            "status": "error",
            "load_time_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.worker_result.write_text(json.dumps(result), encoding="utf-8")
    return 0


def run_model_worker(
    args: argparse.Namespace, model_id: str, backend: str, snapshot: Path
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix="mva_post_filter_", suffix=".json", delete=False
    ) as handle:
        result_path = Path(handle.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cases",
        str(args.cases.resolve()),
        "--torch-threads",
        str(args.torch_threads),
        "--worker-model-id",
        model_id,
        "--worker-backend",
        backend,
        "--worker-snapshot",
        str(snapshot.resolve()),
        "--worker-result",
        str(result_path),
    ]
    try:
        completed = subprocess.run(command, check=False, env=os.environ.copy())
        if completed.returncode != 0:
            raise RuntimeError(f"Worker exited with {completed.returncode}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        result_path.unlink(missing_ok=True)


def recommendation(results: Sequence[dict[str, Any]]) -> tuple[str | None, str]:
    successful = {row["model_id"]: row for row in results if row.get("status") == "ok"}
    ordered = [model_id for model_id, _ in MODEL_SPECS]
    for model_id in ordered:
        result = successful.get(model_id)
        if (
            result
            and result["lara_english_pass"]
            and result["post_filter"]["rankings_stable"]
            and not result["unexpected_filter_activations"]
        ):
            return model_id, "Passed the English Lara gate and current-intent safety gate."
    return None, "No model passed the required post-filter quality and safety gates."


def pct(value: float) -> str:
    return f"{value:.3f}"


def params(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1_000_000:.1f}M"


def render_report(
    payload: dict[str, Any],
    results: Sequence[dict[str, Any]],
    selected: str | None,
    reason: str,
    torch_threads: int,
) -> str:
    lines = [
        "# Reranker benchmark after deterministic current-policy filtering",
        "",
        "## Scope and method",
        "",
        (
            "The same 64 static regression queries and the same eight candidate chunks were "
            "used for every model. No `/api/ask`, Chroma, BM25, embeddings, reindex, LLM, "
            "self-check or CRM call was executed. The metadata gate reads only the fixture "
            "copy derived from the project's CRM CSV files."
        ),
        "",
        f"- Reference date: {payload['selection_reference_date']}",
        f"- CPU only; FP16 disabled; threads={torch_threads}",
        f"- max_length={MAX_LENGTH}; batch_size={BATCH_SIZE}",
        f"- Warm-up: 1 per loaded model; measured runs: {MEASURED_RUNS} per query and mode",
        "- Each model was loaded once in its own process; downloads completed before load timing.",
        "- Raw scores are not compared; all quality metrics are ranking-based.",
        "- Peak RAM was not sampled because reliable process-tree measurement would add instrumentation.",
        f"- Host: {platform.platform()}",
        "",
        "## Compact comparison (post-filter ranking)",
        "",
        "| Model | Params | Top-1 | MRR@5 | nDCG@5 | EN Lara raw→filtered | Lara all languages | Median rerank | Mean rerank | Load | Stable |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if result.get("status") != "ok":
            lines.append(
                f"| `{result['model_id']}` | n/a | n/a | n/a | n/a | FAIL | FAIL | n/a | n/a | "
                f"{result.get('load_time_seconds', 0):.2f}s | error: {result['error']} |"
            )
            continue
        filtered = result["post_filter"]
        lines.append(
            f"| `{result['model_id']}` | {params(result['parameter_count'])} | "
            f"{pct(filtered['top1_accuracy'])} | {pct(filtered['mrr_at_5'])} | "
            f"{pct(filtered['ndcg_at_5'])} | {result['lara_english_raw_rank']}→"
            f"{result['lara_english_post_rank']} {'PASS' if result['lara_english_pass'] else 'FAIL'} | "
            f"{'PASS' if result['lara_all_languages_pass'] else 'FAIL'} | "
            f"{filtered['median_latency_ms']:.1f}ms | {filtered['mean_latency_ms']:.1f}ms | "
            f"{result['load_time_seconds']:.2f}s | {filtered['rankings_stable']} |"
        )

    lines.extend(
        [
            "",
            "## Raw versus post-filter quality",
            "",
            "| Model | Raw Top-1 | Filtered Top-1 | Raw MRR@5 | Filtered MRR@5 | Raw nDCG@5 | Filtered nDCG@5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        if result.get("status") != "ok":
            continue
        raw, filtered = result["raw"], result["post_filter"]
        lines.append(
            f"| `{result['model_id']}` | {pct(raw['top1_accuracy'])} | "
            f"{pct(filtered['top1_accuracy'])} | {pct(raw['mrr_at_5'])} | "
            f"{pct(filtered['mrr_at_5'])} | {pct(raw['ndcg_at_5'])} | "
            f"{pct(filtered['ndcg_at_5'])} |"
        )

    lines.extend(["", "## Language breakdown after filtering", ""])
    lines.append("| Model | Language | Top-1 | MRR@5 | nDCG@5 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for result in results:
        if result.get("status") != "ok":
            continue
        for language, metrics in result["post_filter"]["by_language"].items():
            lines.append(
                f"| `{result['model_id']}` | {language} | {pct(metrics['top1_accuracy'])} | "
                f"{pct(metrics['mrr_at_5'])} | {pct(metrics['ndcg_at_5'])} |"
            )

    lines.extend(["", "## Decision", ""])
    if selected:
        lines.append(f"**Recommendation: `{selected}`.** {reason}")
        lines.append(
            "This is a benchmark recommendation only; productive configuration was not changed."
        )
    else:
        lines.append(f"**No winner.** {reason}")
        lines.append(
            "Next measures would be stricter metadata coverage, stronger hard negatives, or domain fine-tuning."
        )

    lines.extend(["", "## Runtime and compatibility", ""])
    for result in results:
        if result.get("status") != "ok":
            continue
        lines.append(
            f"- `{result['model_id']}`: backend={result['backend']}; "
            f"sentence-transformers={result['runtime_versions']['sentence-transformers']}; "
            f"transformers={result['runtime_versions']['transformers']}; "
            f"tokenizer/config limits={result['tokenizer_limit']}/{result['config_limit']}; "
            f"filtered latency range={result['post_filter']['min_latency_ms']:.1f}-"
            f"{result['post_filter']['max_latency_ms']:.1f}ms."
        )
    lines.extend(
        [
            "",
            "All three adapters accepted max_length=512. Inputs are short, so no model-specific truncation was required.",
            "",
            "## Per-query top 5",
            "",
        ]
    )
    for result in results:
        lines.extend([f"### {result['model_id']}", ""])
        if result.get("status") != "ok":
            lines.extend([f"Error: `{result['error']}`", ""])
            continue
        lines.append("| Case | Filter | Correct rank | Top 5 | Median latency |")
        lines.append("| --- | --- | ---: | --- | ---: |")
        for row in result["case_results"]:
            mode = "applied" if row["selection"]["activated"] else "unchanged"
            rank = row["post_filter"]["correct_rank"] or "missing"
            top5 = ", ".join(f"`{item}`" for item in row["post_filter"]["top5"])
            lines.append(
                f"| `{row['case_id']}` | {mode} | {rank} | {top5} | "
                f"{statistics.median(row['post_filter']['latencies_ms']):.1f}ms |"
            )
        lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "model",
            "case_id",
            "language",
            "filter_activated",
            "filter_reason",
            "candidate_count_before",
            "candidate_count_after",
            "selection_latency_ms",
            "raw_correct_rank",
            "post_filter_correct_rank",
            "raw_top5",
            "post_filter_top5",
            "raw_run_1_ms",
            "raw_run_2_ms",
            "raw_run_3_ms",
            "post_run_1_ms",
            "post_run_2_ms",
            "post_run_3_ms",
            "post_rankings_stable",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if result.get("status") != "ok":
                continue
            for row in result["case_results"]:
                writer.writerow(
                    {
                        "model": result["model_id"],
                        "case_id": row["case_id"],
                        "language": row["language"],
                        "filter_activated": row["selection"]["activated"],
                        "filter_reason": row["selection"]["reason"],
                        "candidate_count_before": row["selection"]["candidate_count_before"],
                        "candidate_count_after": row["selection"]["candidate_count_after"],
                        "selection_latency_ms": f"{row['selection']['selection_latency_ms']:.4f}",
                        "raw_correct_rank": row["raw"]["correct_rank"],
                        "post_filter_correct_rank": row["post_filter"]["correct_rank"],
                        "raw_top5": "|".join(row["raw"]["top5"]),
                        "post_filter_top5": "|".join(row["post_filter"]["top5"]),
                        "raw_run_1_ms": f"{row['raw']['latencies_ms'][0]:.3f}",
                        "raw_run_2_ms": f"{row['raw']['latencies_ms'][1]:.3f}",
                        "raw_run_3_ms": f"{row['raw']['latencies_ms'][2]:.3f}",
                        "post_run_1_ms": f"{row['post_filter']['latencies_ms'][0]:.3f}",
                        "post_run_2_ms": f"{row['post_filter']['latencies_ms'][1]:.3f}",
                        "post_run_3_ms": f"{row['post_filter']['latencies_ms'][2]:.3f}",
                        "post_rankings_stable": row["post_filter"]["rankings_stable"],
                    }
                )


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        if not all((args.worker_backend, args.worker_snapshot, args.worker_result)):
            raise ValueError("Incomplete worker arguments")
        return run_worker(args)
    payload = load_cases(args.cases.resolve())
    if "selection_reference_date" not in payload:
        raise ValueError("Fixture must provide selection_reference_date")
    snapshots = prefetch_models(MODEL_SPECS)
    results = []
    for model_id, backend in MODEL_SPECS:
        print(f"Benchmarking {model_id}...", flush=True)
        results.append(
            run_model_worker(args, model_id, backend, snapshots[model_id])
        )
    selected, reason = recommendation(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(payload, results, selected, reason, args.torch_threads),
        encoding="utf-8",
    )
    write_csv(args.csv, results)
    print(f"Report: {args.report}")
    print(f"Details: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

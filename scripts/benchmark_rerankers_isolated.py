from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence


# This script is intentionally standalone. It must not import the project's RAG,
# retrieval, embedding, LLM, CRM, or pipeline modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL_SPECS = (
    ("BAAI/bge-reranker-base", "flag_embedding"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder"),
    ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "cross_encoder"),
    ("cross-encoder/ettin-reranker-32m-v1", "cross_encoder"),
    ("cross-encoder/ettin-reranker-68m-v1", "cross_encoder"),
)
MAX_LENGTH = 512
BATCH_SIZE = 8
MEASURED_RUNS = 3
TOP_K = 5


def validate_benchmark_runtime(
    model_specs: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Require the newer runtime only when an Ettin checkpoint is selected."""

    from packaging.version import Version

    versions = {
        "sentence-transformers": version("sentence-transformers"),
        "transformers": version("transformers"),
    }
    requires_ettin = any("ettin-reranker" in model_id for model_id, _ in model_specs)
    minimums = (
        {
            "sentence-transformers": Version("5.4.1"),
            "transformers": Version("5.7.0"),
        }
        if requires_ettin
        else {}
    )
    incompatible = [
        f"{name}={installed} (requires >= {minimums[name]})"
        for name, installed in versions.items()
        if name in minimums
        if Version(installed) < minimums[name]
    ]
    if incompatible:
        raise RuntimeError(
            "The Ettin modular CrossEncoder checkpoints require a newer isolated "
            "benchmark runtime: " + "; ".join(incompatible)
        )
    return versions


@dataclass
class ModelAdapter:
    model_id: str
    backend_name: str
    backend: Any
    parameter_count: int | None
    tokenizer_limit: int | None
    config_limit: int | None

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        """Common adapter interface required by the isolated benchmark."""

        pairs = [(query, document) for document in documents]
        if self.backend_name == "flag_embedding":
            raw_scores = self.backend.compute_score(pairs)
        else:
            raw_scores = self.backend.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        if not isinstance(raw_scores, (list, tuple)):
            try:
                raw_scores = raw_scores.tolist()
            except AttributeError:
                raw_scores = [raw_scores]
        if not isinstance(raw_scores, list):
            raw_scores = list(raw_scores)
        if raw_scores and isinstance(raw_scores[0], list):
            raw_scores = [row[-1] for row in raw_scores]
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(documents):
            raise RuntimeError(
                f"{self.model_id} returned {len(scores)} scores for "
                f"{len(documents)} documents"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"{self.model_id} returned a non-finite score")
        return scores


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CPU-only, retrieval-free benchmark for local cross-encoder rerankers."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=root / "tests" / "fixtures" / "reranker_benchmark_cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "reranker_benchmark_isolated.md",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "reports" / "reranker_benchmark_isolated_details.csv",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated subset of configured model IDs. By default all models "
            "are benchmarked."
        ),
    )
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--worker-snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_cases(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if len(cases) < 5:
        raise ValueError("At least five benchmark cases are required")
    expected_count = int(payload["candidate_count_per_query"])
    if expected_count != BATCH_SIZE:
        raise ValueError(
            f"Fixture candidate count {expected_count} must match batch size {BATCH_SIZE}"
        )
    for case in cases:
        candidates = case.get("candidates") or []
        if len(candidates) != expected_count:
            raise ValueError(
                f"Case {case.get('id')} has {len(candidates)} candidates; "
                f"expected {expected_count}"
            )
        candidate_ids = [item["id"] for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Duplicate candidate IDs in case {case.get('id')}")
        if case["correct_candidate_id"] not in candidate_ids:
            raise ValueError(f"Missing correct candidate in case {case.get('id')}")
        correct = next(
            item for item in candidates if item["id"] == case["correct_candidate_id"]
        )
        if int(correct["relevance"]) != max(int(item["relevance"]) for item in candidates):
            raise ValueError(f"Correct candidate is not maximally relevant in {case.get('id')}")
    return payload


def selected_model_specs(raw_models: str | None) -> tuple[tuple[str, str], ...]:
    if not raw_models:
        return MODEL_SPECS
    requested = [item.strip() for item in raw_models.split(",") if item.strip()]
    known = dict(MODEL_SPECS)
    unknown = [model_id for model_id in requested if model_id not in known]
    if unknown:
        raise ValueError(f"Unknown benchmark model(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("--models must contain at least one model ID")
    return tuple((model_id, known[model_id]) for model_id in requested)


def prefetch_models(model_specs: Sequence[tuple[str, str]]) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    snapshots: dict[str, Path] = {}
    print("Prefetching model snapshots; download time is excluded from all timings.", flush=True)
    for model_id, _ in model_specs:
        started = time.perf_counter()
        snapshots[model_id] = Path(snapshot_download(repo_id=model_id))
        print(
            f"Prefetched {model_id} in {time.perf_counter() - started:.2f}s -> "
            f"{snapshots[model_id]}",
            flush=True,
        )
    return snapshots


def _safe_limit(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 0 < parsed < 1_000_000 else None


def _parameter_count(model: Any) -> int | None:
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def load_adapter(model_id: str, backend_name: str, snapshot: Path) -> ModelAdapter:
    if backend_name == "flag_embedding":
        from FlagEmbedding import FlagReranker

        backend = FlagReranker(
            str(snapshot),
            use_fp16=False,
            devices="cpu",
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
    else:
        from sentence_transformers import CrossEncoder

        backend = CrossEncoder(
            str(snapshot),
            max_length=MAX_LENGTH,
            device="cpu",
            local_files_only=True,
        )

    model = getattr(backend, "model", None)
    tokenizer = getattr(backend, "tokenizer", None)
    config = getattr(model, "config", None)
    return ModelAdapter(
        model_id=model_id,
        backend_name=backend_name,
        backend=backend,
        parameter_count=_parameter_count(model),
        tokenizer_limit=_safe_limit(getattr(tokenizer, "model_max_length", None)),
        config_limit=_safe_limit(getattr(config, "max_position_embeddings", None)),
    )


def prepare_backend_import(backend_name: str) -> None:
    """Exclude one-time library import cost from per-model load timing."""

    if backend_name == "flag_embedding":
        from FlagEmbedding import FlagReranker  # noqa: F401
    else:
        from sentence_transformers import CrossEncoder  # noqa: F401


def rank_candidate_ids(candidates: Sequence[dict[str, Any]], scores: Sequence[float]) -> list[str]:
    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda index: (-float(scores[index]), index),
    )
    return [candidates[index]["id"] for index in ranked_indices]


def reciprocal_rank_at_5(ranking: Sequence[str], correct_id: str) -> float:
    try:
        rank = ranking.index(correct_id) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank if rank <= TOP_K else 0.0


def ndcg_at_5(ranking: Sequence[str], candidates: Sequence[dict[str, Any]]) -> float:
    relevance = {item["id"]: int(item["relevance"]) for item in candidates}

    def dcg(values: Sequence[int]) -> float:
        return sum(
            (2**value - 1) / math.log2(index + 2)
            for index, value in enumerate(values[:TOP_K])
        )

    observed = dcg([relevance[candidate_id] for candidate_id in ranking])
    ideal = dcg(sorted(relevance.values(), reverse=True))
    return observed / ideal if ideal else 0.0


def benchmark_model(adapter: ModelAdapter, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    warmup_case = cases[0]
    adapter.score_pairs(
        warmup_case["query"],
        [candidate["document"] for candidate in warmup_case["candidates"]],
    )

    case_results: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []
    for case in cases:
        documents = [candidate["document"] for candidate in case["candidates"]]
        score_runs: list[list[float]] = []
        rankings_per_run: list[list[str]] = []
        latencies_ms: list[float] = []
        for _ in range(MEASURED_RUNS):
            started = time.perf_counter()
            scores = adapter.score_pairs(case["query"], documents)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            score_runs.append(scores)
            rankings_per_run.append(rank_candidate_ids(case["candidates"], scores))
            latencies_ms.append(elapsed_ms)
            all_latencies_ms.append(elapsed_ms)

        mean_scores = [
            statistics.fmean(run[index] for run in score_runs)
            for index in range(len(documents))
        ]
        ranking = rank_candidate_ids(case["candidates"], mean_scores)
        correct_rank = ranking.index(case["correct_candidate_id"]) + 1
        stable = all(run_ranking == rankings_per_run[0] for run_ranking in rankings_per_run)
        mandatory_pass = (
            correct_rank == 1
            and all(
                run_ranking[0] == case["correct_candidate_id"]
                for run_ranking in rankings_per_run
            )
        )
        case_results.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "correct_candidate_id": case["correct_candidate_id"],
                "correct_rank": correct_rank,
                "top5": ranking[:TOP_K],
                "rankings_stable": stable,
                "mandatory": bool(case.get("mandatory_rank1")),
                "mandatory_pass": mandatory_pass,
                "mrr_at_5": reciprocal_rank_at_5(ranking, case["correct_candidate_id"]),
                "ndcg_at_5": ndcg_at_5(ranking, case["candidates"]),
                "latencies_ms": latencies_ms,
            }
        )

    mandatory_results = [item for item in case_results if item["mandatory"]]
    return {
        "model_id": adapter.model_id,
        "backend": adapter.backend_name,
        "parameter_count": adapter.parameter_count,
        "tokenizer_limit": adapter.tokenizer_limit,
        "config_limit": adapter.config_limit,
        "top1_accuracy": statistics.fmean(
            1.0 if item["correct_rank"] == 1 else 0.0 for item in case_results
        ),
        "mrr_at_5": statistics.fmean(item["mrr_at_5"] for item in case_results),
        "ndcg_at_5": statistics.fmean(item["ndcg_at_5"] for item in case_results),
        "lara_pass": bool(mandatory_results)
        and all(item["mandatory_pass"] for item in mandatory_results),
        "rankings_stable": all(item["rankings_stable"] for item in case_results),
        "mean_latency_ms": statistics.fmean(all_latencies_ms),
        "median_latency_ms": statistics.median(all_latencies_ms),
        "min_latency_ms": min(all_latencies_ms),
        "max_latency_ms": max(all_latencies_ms),
        "case_results": case_results,
    }


def run_worker(args: argparse.Namespace) -> int:
    """Run one backend in a dependency-compatible subprocess and return JSON."""

    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cases_payload = load_cases(args.cases.resolve())
    prepare_backend_import(args.worker_backend)
    load_started = time.perf_counter()
    try:
        adapter = load_adapter(
            args.worker_model_id,
            args.worker_backend,
            args.worker_snapshot.resolve(),
        )
        load_time_seconds = time.perf_counter() - load_started
        result = benchmark_model(adapter, cases_payload["cases"])
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
            "load_time_seconds": time.perf_counter() - load_started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.worker_result.parent.mkdir(parents=True, exist_ok=True)
    args.worker_result.write_text(json.dumps(result), encoding="utf-8")
    return 0


def run_flag_embedding_worker(
    args: argparse.Namespace,
    model_id: str,
    backend_name: str,
    snapshot: Path,
) -> dict[str, Any]:
    """Use the unchanged project runtime required by FlagEmbedding."""

    with tempfile.NamedTemporaryFile(
        prefix="mva_reranker_worker_",
        suffix=".json",
        delete=False,
    ) as handle:
        result_path = Path(handle.name)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
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
        backend_name,
        "--worker-snapshot",
        str(snapshot.resolve()),
        "--worker-result",
        str(result_path),
    ]
    try:
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"FlagEmbedding worker exited with {completed.returncode}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        result_path.unlink(missing_ok=True)


def select_winner(results: Sequence[dict[str, Any]]) -> tuple[str | None, str]:
    eligible = [result for result in results if result.get("status") == "ok" and result["lara_pass"]]
    if not eligible:
        return None, (
            "No model passed the Lara Neumann rank-1 requirement in every measured run. "
            "No winner is selected."
        )

    best_top1 = max(result["top1_accuracy"] for result in eligible)
    quality_pool = [result for result in eligible if result["top1_accuracy"] == best_top1]
    best_mrr = max(result["mrr_at_5"] for result in quality_pool)
    quality_pool = [result for result in quality_pool if best_mrr - result["mrr_at_5"] <= 0.01]
    best_ndcg = max(result["ndcg_at_5"] for result in quality_pool)
    quality_pool = [result for result in quality_pool if best_ndcg - result["ndcg_at_5"] <= 0.01]
    winner = min(quality_pool, key=lambda result: result["median_latency_ms"])
    reason = (
        "All mandatory-case failures were excluded first. Ranking quality was compared "
        "by Top-1 Accuracy, MRR@5 and nDCG@5; median latency broke near-equal quality ties."
    )
    return winner["model_id"], reason


def _format_parameters(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1_000_000:.1f}M"


def _format_limit(value: int | None) -> str:
    return "not exposed" if value is None else str(value)


def render_report(
    cases_payload: dict[str, Any],
    results: Sequence[dict[str, Any]],
    winner: str | None,
    winner_reason: str,
    torch_threads: int,
    runtime_versions: dict[str, str],
) -> str:
    lines = [
        "# Isolated local reranker benchmark",
        "",
        "## Scope",
        "",
        (
            "This benchmark scored static query-document pairs only. It did not import or "
            "invoke the project RAG pipeline, Chroma, BM25, embeddings, LLMs, self-check, "
            "CRM, reindexing, or `/api/ask`."
        ),
        "",
        f"- Cases: {len(cases_payload['cases'])}",
        f"- Candidates per case: {cases_payload['candidate_count_per_query']} (production `RETRIEVE_TOP_K` default)",
        f"- Top documents reported: {TOP_K}",
        f"- Warm-up runs: 1 per model",
        f"- Measured runs: {MEASURED_RUNS} per query and model",
        f"- Device: CPU only; PyTorch threads: {torch_threads}; FP16: disabled",
        (
            f"- CrossEncoder runtime: sentence-transformers={runtime_versions['sentence-transformers']}; "
            f"transformers={runtime_versions['transformers']}"
        ),
        (
            "- FlagEmbedding runs in a subprocess using the unchanged project runtime because "
            "FlagEmbedding and the Ettin-required Transformers version are mutually incompatible."
        ),
        "- FlagEmbedding's built-in per-call progress handling could not be disabled through its public scoring API and is included in BGE latency.",
        f"- Shared settings: max_length={MAX_LENGTH}, batch_size={BATCH_SIZE}",
        "- Model downloads completed before model-load and inference timing.",
        "- Peak RAM was not sampled to avoid background instrumentation affecting this short latency test.",
        f"- Host: {platform.platform()}",
        "",
        "## Comparison",
        "",
        "| Model | Parameters | Top-1 Accuracy | MRR@5 | nDCG@5 | Lara case | Mean latency | Median latency | Load time | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if result.get("status") != "ok":
            lines.append(
                f"| `{result['model_id']}` | n/a | n/a | n/a | n/a | FAIL | n/a | n/a | "
                f"{result.get('load_time_seconds', 0.0):.2f}s | Load/benchmark error: {result['error']} |"
            )
            continue
        notes = (
            f"backend={result['backend']}; rankings stable={result['rankings_stable']}; "
            f"latency range {result['min_latency_ms']:.1f}-{result['max_latency_ms']:.1f}ms; "
            f"tokenizer/config limits={_format_limit(result['tokenizer_limit'])}/"
            f"{_format_limit(result['config_limit'])}; "
            f"runtime ST/TF={result['runtime_versions']['sentence-transformers']}/"
            f"{result['runtime_versions']['transformers']}"
        )
        lines.append(
            f"| `{result['model_id']}` | {_format_parameters(result['parameter_count'])} | "
            f"{result['top1_accuracy']:.3f} | {result['mrr_at_5']:.3f} | "
            f"{result['ndcg_at_5']:.3f} | {'PASS' if result['lara_pass'] else 'FAIL'} | "
            f"{result['mean_latency_ms']:.1f}ms | {result['median_latency_ms']:.1f}ms | "
            f"{result['load_time_seconds']:.2f}s | {notes} |"
        )

    lines.extend(["", "## Selection", ""])
    if winner is None:
        lines.append(f"**No winner selected.** {winner_reason}")
        lines.append(
            "Further measures should focus on metadata filtering, stronger domain hard negatives, "
            "or domain-specific fine-tuning."
        )
    else:
        lines.append(f"**Winner: `{winner}`.** {winner_reason}")

    lines.extend(["", "## Per-query rankings", ""])
    for result in results:
        lines.extend([f"### {result['model_id']}", ""])
        if result.get("status") != "ok":
            lines.extend([f"Benchmark failed: `{result['error']}`", ""])
            continue
        lines.append("| Query case | Correct rank | Top 5 candidate IDs | Median query latency |")
        lines.append("| --- | ---: | --- | ---: |")
        for case in result["case_results"]:
            lines.append(
                f"| `{case['case_id']}` | {case['correct_rank']} | "
                f"{', '.join(f'`{item}`' for item in case['top5'])} | "
                f"{statistics.median(case['latencies_ms']):.1f}ms |"
            )
        lines.append("")

    lines.extend(
        [
            "## Input-length compatibility",
            "",
            (
                f"All adapters were configured with max_length={MAX_LENGTH}. The table notes the "
                "tokenizer and model-configuration limits exposed at runtime. No model required a "
                "lower benchmark limit; inputs were short and the shared limit applied uniformly."
            ),
            "",
            "## Metric interpretation",
            "",
            "Raw scores are intentionally omitted because score scales differ between models. "
            "All quality metrics use rankings only. nDCG@5 uses fixture relevance grades 0, 1 and 3.",
            "",
        ]
    )
    return "\n".join(lines)


def write_detail_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model",
                "case_id",
                "correct_rank",
                "top5_candidate_ids",
                "run_1_ms",
                "run_2_ms",
                "run_3_ms",
                "rankings_stable",
                "mandatory_pass",
            ),
        )
        writer.writeheader()
        for result in results:
            if result.get("status") != "ok":
                continue
            for case in result["case_results"]:
                writer.writerow(
                    {
                        "model": result["model_id"],
                        "case_id": case["case_id"],
                        "correct_rank": case["correct_rank"],
                        "top5_candidate_ids": "|".join(case["top5"]),
                        "run_1_ms": f"{case['latencies_ms'][0]:.3f}",
                        "run_2_ms": f"{case['latencies_ms'][1]:.3f}",
                        "run_3_ms": f"{case['latencies_ms'][2]:.3f}",
                        "rankings_stable": case["rankings_stable"],
                        "mandatory_pass": case["mandatory_pass"],
                    }
                )


def main() -> int:
    args = parse_args()
    if args.worker_model_id:
        if not all(
            (
                args.worker_backend,
                args.worker_snapshot,
                args.worker_result,
            )
        ):
            raise ValueError("Incomplete worker arguments")
        return run_worker(args)

    model_specs = selected_model_specs(args.models)
    cases_payload = load_cases(args.cases.resolve())
    runtime_versions = validate_benchmark_runtime(model_specs)

    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    snapshots = prefetch_models(model_specs)
    prepare_backend_import("cross_encoder")
    results: list[dict[str, Any]] = []
    for model_id, backend_name in model_specs:
        print(f"Loading {model_id} from local cache...", flush=True)
        load_started = time.perf_counter()
        try:
            if backend_name == "flag_embedding":
                result = run_flag_embedding_worker(
                    args,
                    model_id,
                    backend_name,
                    snapshots[model_id],
                )
            else:
                adapter = load_adapter(model_id, backend_name, snapshots[model_id])
                load_time_seconds = time.perf_counter() - load_started
                print(
                    f"Loaded {model_id} in {load_time_seconds:.2f}s; benchmarking...",
                    flush=True,
                )
                result = benchmark_model(adapter, cases_payload["cases"])
                result.update(
                    status="ok",
                    load_time_seconds=load_time_seconds,
                    runtime_versions=dict(runtime_versions),
                )
            if result.get("status") != "ok":
                raise RuntimeError(result.get("error", "backend worker failed"))
            print(
                f"Completed {model_id}: Top1={result['top1_accuracy']:.3f}, "
                f"MRR@5={result['mrr_at_5']:.3f}, Lara={result['lara_pass']}, "
                f"median={result['median_latency_ms']:.1f}ms",
                flush=True,
            )
        except Exception as exc:
            result = {
                "model_id": model_id,
                "backend": backend_name,
                "status": "error",
                "load_time_seconds": time.perf_counter() - load_started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAILED {model_id}: {result['error']}", flush=True)
        results.append(result)
        if "adapter" in locals():
            del adapter
        gc.collect()

    winner, winner_reason = select_winner(results)
    report = render_report(
        cases_payload,
        results,
        winner,
        winner_reason,
        torch.get_num_threads(),
        runtime_versions,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_detail_csv(args.csv, results)
    print(f"Report: {args.report.resolve()}", flush=True)
    print(f"Details: {args.csv.resolve()}", flush=True)
    return 0 if any(result.get("status") == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

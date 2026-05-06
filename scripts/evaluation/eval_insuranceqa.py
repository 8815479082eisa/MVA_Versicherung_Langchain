import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "Missing dependency `datasets`. Install with: pip install datasets"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_utils import (
    best_token_f1,
    exact_match,
    load_question_answer_pairs,
    sample_question_answer_pairs,
    slice_question_answer_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate insuranceQA-v2 against /api/ask")
    parser.add_argument("--api-url", default="http://localhost:8000/api/ask")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-jsonl",
        default="data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl",
        help="Local InsuranceQA JSONL to evaluate against; used by default when present",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0, help="Start index inside sampled set")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="How many questions to run from --start (default: all remaining)",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--health-url",
        default=None,
        help="Health endpoint to poll before starting (default: derived from --api-url)",
    )
    parser.add_argument(
        "--wait-for-api-seconds",
        type=int,
        default=180,
        help="Wait up to this many seconds for the API to become reachable before sending questions",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing JSONL file instead of starting a clean single-run output",
    )
    parser.add_argument(
        "--allow-dataset-shortcut",
        action="store_true",
        help="Allow direct dataset shortcut answers (for debugging only; not for real benchmark runs)",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=20,
        help="Stop early if this many failures happen in a row",
    )
    parser.add_argument(
        "--out",
        default="data/processed/eval_outputs/insuranceqa/eval_insuranceqa_results.jsonl",
        help="Append per-item results to this JSONL file",
    )
    return parser.parse_args()


def _load_hf_sample(split: str) -> list[tuple[str, list[str]]]:
    ds = load_dataset("deccan-ai/insuranceQA-v2", split=split)

    q2answers: dict[str, list[str]] = {}
    for row in ds:
        question = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if not question or not answer:
            continue
        q2answers.setdefault(question, []).append(answer)
    return list(q2answers.items())


def build_sample(
    split: str,
    max_samples: int,
    seed: int,
    dataset_jsonl: str,
) -> tuple[list[tuple[str, list[str]]], str]:
    dataset_path = Path(dataset_jsonl)
    if dataset_path.exists():
        items_all = load_question_answer_pairs(dataset_path)
        source_label = str(dataset_path)
    else:
        items_all = _load_hf_sample(split)
        source_label = f"hf://deccan-ai/insuranceQA-v2[{split}]"

    sampled = sample_question_answer_pairs(
        items_all,
        max_samples=max_samples,
        seed=seed,
    )
    return sampled, source_label


def post_with_retry(
    api_url: str,
    question: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(api_url, json={"question": question}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"API request failed after {retries} attempts: {last_err}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_health_url(api_url: str, explicit_health_url: str | None) -> str:
    if explicit_health_url:
        return explicit_health_url
    if api_url.endswith("/api/ask"):
        return api_url[: -len("/api/ask")] + "/health"
    return api_url.rstrip("/") + "/health"


def wait_for_api(health_url: str, wait_seconds: int, poll_interval: float = 2.0) -> None:
    deadline = time.time() + max(0, wait_seconds)
    last_error = "health endpoint did not respond"
    while time.time() <= deadline:
        try:
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status", "")).lower()
            if status == "ok":
                print(
                    "API reachable:"
                    f" health={health_url}"
                    f" | pipelineReady={payload.get('pipelineReady')}"
                    f" | pipelineInitializing={payload.get('pipelineInitializing')}"
                )
                return
            last_error = f"unexpected health payload: {payload}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(poll_interval)
    raise RuntimeError(
        f"API did not become reachable within {wait_seconds}s via {health_url}: {last_error}"
    )


def prepare_output_file(path: Path, append: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not append:
        path.unlink()
        return "overwritten"
    return "appending" if path.exists() else "new"


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    health_url = derive_health_url(args.api_url, args.health_url)
    output_mode = prepare_output_file(out_path, append=args.append)

    wait_for_api(health_url=health_url, wait_seconds=args.wait_for_api_seconds)

    items, dataset_source = build_sample(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        dataset_jsonl=args.dataset_jsonl,
    )
    items_run, start_idx, end_idx = slice_question_answer_pairs(
        items,
        start=args.start,
        count=args.count,
    )

    print(f"Sample size: {len(items)} | split={args.split} | seed={args.seed}")
    print(f"Dataset source: {dataset_source}")
    print(f"Running indexes: [{start_idx}, {end_idx}) -> {len(items_run)} items")
    print(f"Output JSONL: {out_path} | mode={output_mode} | run_id={run_id}")

    em_hits = 0
    f1_sum = 0.0
    processed = 0
    failed = 0
    consecutive_failures = 0

    for i, (question, refs) in enumerate(items_run, start=start_idx):
        try:
            data = post_with_retry(
                api_url=args.api_url,
                question=question,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            prediction = str(data.get("answer", ""))
            sources = data.get("sources") or []
            source_ids = []
            for source in sources:
                if isinstance(source, dict):
                    source_ids.append(
                        str(source.get("documentId") or source.get("document_id") or "")
                    )

            if (not args.allow_dataset_shortcut) and any(
                source_id == "insuranceqa_v2_local" for source_id in source_ids
            ):
                raise RuntimeError(
                    "Detected direct InsuranceQA dataset shortcut answer "
                    "(source documentId=insuranceqa_v2_local)."
                )

            em = int(exact_match(prediction, refs))
            best_f1 = best_token_f1(prediction, refs)

            em_hits += em
            f1_sum += best_f1
            processed += 1
            consecutive_failures = 0

            append_jsonl(
                out_path,
                {
                    "run_id": run_id,
                    "idx": i,
                    "question": question,
                    "refs": refs,
                    "prediction": prediction,
                    "source_ids": source_ids,
                    "em": em,
                    "best_f1": best_f1,
                },
            )
        except Exception as exc:
            failed += 1
            consecutive_failures += 1
            append_jsonl(
                out_path,
                {"run_id": run_id, "idx": i, "question": question, "error": str(exc)},
            )
            print(f"[ERROR] idx={i}: {exc}")
            if consecutive_failures >= args.max_consecutive_failures:
                print(
                    f"Stopping early after {consecutive_failures} consecutive failures. "
                    "Check backend/API key/quota."
                )
                break

        if (i - start_idx + 1) % 20 == 0:
            print(
                f"{i - start_idx + 1}/{len(items_run)} done | processed={processed} | failed={failed}"
            )

    print("\n=== RESULT ===")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    if processed:
        print(f"Exact Match: {em_hits / processed:.6f}")
        print(f"Token F1: {f1_sum / processed:.6f}")
    else:
        print("No successful samples to score.")


if __name__ == "__main__":
    main()

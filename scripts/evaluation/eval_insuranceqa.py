import argparse
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "Missing dependency `datasets`. Install with: pip install datasets"
    ) from exc


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


def norm(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def token_f1(pred: str, gold: str) -> float:
    p = norm(pred).split()
    g = norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1

    inter = 0
    for w in g:
        if common.get(w, 0) > 0:
            inter += 1
            common[w] -= 1

    if inter == 0:
        return 0.0

    precision = inter / len(p)
    recall = inter / len(g)
    return 2 * precision * recall / (precision + recall)


def _load_local_sample(path: Path) -> list[tuple[str, list[str]]]:
    q2answers: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question") or row.get("input") or "").strip()
            answer = str(row.get("answer") or row.get("output") or "").strip()
            if question and answer:
                q2answers[question].append(answer)
    return list(q2answers.items())


def _load_hf_sample(split: str) -> list[tuple[str, list[str]]]:
    ds = load_dataset("deccan-ai/insuranceQA-v2", split=split)

    q2answers: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        question = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if question and answer:
            q2answers[question].append(answer)
    return list(q2answers.items())


def build_sample(
    split: str,
    max_samples: int,
    seed: int,
    dataset_jsonl: str,
) -> tuple[list[tuple[str, list[str]]], str]:
    dataset_path = Path(dataset_jsonl)
    if dataset_path.exists():
        items_all = _load_local_sample(dataset_path)
        source_label = str(dataset_path)
    else:
        items_all = _load_hf_sample(split)
        source_label = f"hf://deccan-ai/insuranceQA-v2[{split}]"

    random.seed(seed)
    k = min(max_samples, len(items_all))
    sampled = random.sample(items_all, k=k)
    return sampled, source_label


def batched_slice(
    items: list[tuple[str, list[str]]], start: int, count: int | None
) -> tuple[list[tuple[str, list[str]]], int, int]:
    safe_start = max(0, min(start, len(items)))
    end = len(items) if count is None else min(len(items), safe_start + max(0, count))
    return items[safe_start:end], safe_start, end


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
            r = requests.post(api_url, json={"question": question}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"API request failed after {retries} attempts: {last_err}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    items_run, start_idx, end_idx = batched_slice(items, start=args.start, count=args.count)

    print(f"Sample size: {len(items)} | split={args.split} | seed={args.seed}")
    print(f"Dataset source: {dataset_source}")
    print(f"Running indexes: [{start_idx}, {end_idx}) -> {len(items_run)} items")
    print(f"Output JSONL: {out_path} | mode={output_mode} | run_id={run_id}")

    em_hits = 0
    f1_sum = 0.0
    processed = 0
    failed = 0
    consecutive_failures = 0

    for i, (q, refs) in enumerate(items_run, start=start_idx):
        try:
            data = post_with_retry(
                api_url=args.api_url,
                question=q,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            pred = str(data.get("answer", ""))
            sources = data.get("sources") or []
            source_ids = []
            for src in sources:
                if isinstance(src, dict):
                    source_ids.append(str(src.get("documentId") or src.get("document_id") or ""))

            if (not args.allow_dataset_shortcut) and any(
                sid == "insuranceqa_v2_local" for sid in source_ids
            ):
                raise RuntimeError(
                    "Detected direct InsuranceQA dataset shortcut answer "
                    "(source documentId=insuranceqa_v2_local)."
                )

            pred_n = norm(pred)
            refs_n = [norm(x) for x in refs]
            em = 1 if pred_n in refs_n else 0
            best_f1 = max(token_f1(pred, ref) for ref in refs)

            em_hits += em
            f1_sum += best_f1
            processed += 1
            consecutive_failures = 0

            append_jsonl(
                out_path,
                {
                    "run_id": run_id,
                    "idx": i,
                    "question": q,
                    "refs": refs,
                    "prediction": pred,
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
                {"run_id": run_id, "idx": i, "question": q, "error": str(exc)},
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

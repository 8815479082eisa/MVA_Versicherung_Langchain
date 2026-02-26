import argparse
import json
import random
import re
import time
from collections import defaultdict
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


def build_sample(split: str, max_samples: int, seed: int) -> list[tuple[str, list[str]]]:
    ds = load_dataset("deccan-ai/insuranceQA-v2", split=split)

    q2answers: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        question = str(row.get("input", "")).strip()
        answer = str(row.get("output", "")).strip()
        if question and answer:
            q2answers[question].append(answer)

    items_all = list(q2answers.items())
    random.seed(seed)
    k = min(max_samples, len(items_all))
    sampled = random.sample(items_all, k=k)
    return sampled


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


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    items = build_sample(split=args.split, max_samples=args.max_samples, seed=args.seed)
    items_run, start_idx, end_idx = batched_slice(items, start=args.start, count=args.count)

    print(f"Sample size: {len(items)} | split={args.split} | seed={args.seed}")
    print(f"Running indexes: [{start_idx}, {end_idx}) -> {len(items_run)} items")
    print(f"Output JSONL: {out_path}")

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
                    "idx": i,
                    "question": q,
                    "refs": refs,
                    "prediction": pred,
                    "em": em,
                    "best_f1": best_f1,
                },
            )
        except Exception as exc:
            failed += 1
            consecutive_failures += 1
            append_jsonl(
                out_path,
                {"idx": i, "question": q, "error": str(exc)},
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

import argparse
import json
import random
from pathlib import Path

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit("Missing dependency `datasets`. Install with: pip install datasets") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract random Q/A pairs from deccan-ai/insuranceQA-v2 into local JSONL"
    )
    parser.add_argument("--split", default="train", help="Dataset split: train|validation|test")
    parser.add_argument("--count", type=int, default=1000, help="Number of unique questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out",
        default="data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl",
        help="Output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("deccan-ai/insuranceQA-v2", split=args.split)

    grouped: dict[str, list[str]] = {}
    for row in ds:
        q = str(row.get("input", "")).strip()
        a = str(row.get("output", "")).strip()
        if not q or not a:
            continue
        grouped.setdefault(q, []).append(a)

    questions = list(grouped.keys())
    random.seed(args.seed)
    k = min(args.count, len(questions))
    sampled_questions = random.sample(questions, k=k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for q in sampled_questions:
            # Use the first reference answer for lightweight FAQ mode.
            answer = grouped[q][0]
            row = {"question": q, "answer": answer}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Split: {args.split}")
    print(f"Unique questions in split: {len(questions)}")
    print(f"Saved rows: {written}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

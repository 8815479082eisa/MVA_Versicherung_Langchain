from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.evaluation.benchmark_utils import load_question_answer_pairs, normalize_text
except Exception:
    from evaluation.benchmark_utils import load_question_answer_pairs, normalize_text  # type: ignore


INPUT_DEFAULT = Path("data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl")
OUTPUT_DEFAULT = Path("data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl")


def classify_question(question: str) -> str:
    q = normalize_text(question)

    if q.startswith("what is ") or q.startswith("what are ") or q.startswith("what s "):
        return "general_definition"

    if any(
        token in q
        for token in (
            "premium",
            "deductible",
            "retirement",
            "benefit",
            "policy",
            "copay",
            "coinsurance",
        )
    ):
        return "policy_premium_deductible_retirement_benefits"

    if any(
        token in q
        for token in (
            "how much",
            "how long",
            "how to",
            "can i",
            "does ",
            "should i",
            "when can",
            "who sells",
        )
    ):
        return "dataset_like_benchmark_style"

    if any(
        token in q
        for token in (
            "insurance",
            "medicare",
            "medigap",
            "annuity",
            "life ",
            "auto ",
            "home ",
            "disability",
            "health ",
        )
    ):
        return "domain_specific_insurance"

    return "benign_informational"


def take_quota(
    *,
    bucketed: Dict[str, List[Tuple[str, str]]],
    category: str,
    count: int,
    chosen: Dict[str, Tuple[str, str, str]],
) -> None:
    for question, answer in bucketed.get(category, []):
        if len([1 for _, _, c in chosen.values() if c == category]) >= count:
            break
        key = normalize_text(question)
        if key in chosen:
            continue
        chosen[key] = (question, answer, category)


def main() -> None:
    items = load_question_answer_pairs(INPUT_DEFAULT)
    rows: List[Tuple[str, str, str]] = []

    for question, refs in items:
        if not refs:
            continue
        rows.append((question.strip(), refs[0].strip(), classify_question(question)))

    rows = [row for row in rows if row[0] and row[1]]
    rows.sort(key=lambda x: normalize_text(x[0]))

    bucketed: Dict[str, List[Tuple[str, str]]] = {}
    for question, answer, category in rows:
        bucketed.setdefault(category, []).append((question, answer))

    target_quota = {
        "general_definition": 40,
        "policy_premium_deductible_retirement_benefits": 50,
        "dataset_like_benchmark_style": 45,
        "domain_specific_insurance": 45,
        "benign_informational": 20,
    }

    chosen: Dict[str, Tuple[str, str, str]] = {}

    for category, count in target_quota.items():
        take_quota(bucketed=bucketed, category=category, count=count, chosen=chosen)

    if len(chosen) < 200:
        for question, answer, category in rows:
            key = normalize_text(question)
            if key in chosen:
                continue
            chosen[key] = (question, answer, category)
            if len(chosen) >= 200:
                break

    selected = sorted(chosen.values(), key=lambda x: normalize_text(x[0]))[:200]

    OUTPUT_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DEFAULT.open("w", encoding="utf-8") as handle:
        for idx, (question, answer, category) in enumerate(selected, start=1):
            payload = {
                "id": f"qa-thesis-{idx:03d}",
                "question": question,
                "answer": answer,
                "category": category,
                "expected_behavior": "allow_and_answer_from_dataset_or_retrieved_context",
                "expected_decision": "allow",
                "notes": "Deterministic thesis QA subset derived from InsuranceQA benchmark.",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "output": str(OUTPUT_DEFAULT),
        "count": len(selected),
        "category_counts": {
            category: sum(1 for _, _, c in selected if c == category)
            for category in sorted({c for _, _, c in selected})
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

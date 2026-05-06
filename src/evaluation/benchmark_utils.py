"""Shared benchmark utilities for InsuranceQA-style evaluation."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


QuestionAnswerPairs = List[Tuple[str, List[str]]]


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_references = [normalize_text(reference) for reference in references]
    return 1.0 if normalized_prediction and normalized_prediction in normalized_references else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common: Dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    intersection = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            intersection += 1
            common[token] -= 1

    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_tokens)
    recall = intersection / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    return max(token_f1(prediction, reference) for reference in references)


def load_question_answer_pairs(dataset_jsonl: str | Path) -> QuestionAnswerPairs:
    dataset_path = Path(dataset_jsonl)
    q2answers: dict[str, list[str]] = defaultdict(list)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question") or row.get("input") or "").strip()
            answer = str(row.get("answer") or row.get("output") or "").strip()
            if question and answer:
                q2answers[question].append(answer)
    return list(q2answers.items())


def load_question_answer_map(dataset_jsonl: str | Path) -> dict[str, list[str]]:
    return dict(load_question_answer_pairs(dataset_jsonl))


def sample_question_answer_pairs(
    items: Sequence[Tuple[str, List[str]]],
    *,
    max_samples: int,
    seed: int,
) -> QuestionAnswerPairs:
    if not items:
        return []
    random.seed(seed)
    sample_size = min(max_samples, len(items))
    return random.sample(list(items), k=sample_size)


def slice_question_answer_pairs(
    items: Sequence[Tuple[str, List[str]]],
    *,
    start: int,
    count: int | None,
) -> tuple[QuestionAnswerPairs, int, int]:
    safe_start = max(0, min(start, len(items)))
    end = len(items) if count is None else min(len(items), safe_start + max(0, count))
    return list(items[safe_start:end]), safe_start, end


def dump_question_answer_pairs(
    path: str | Path,
    items: Iterable[Tuple[str, Sequence[str]]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for question, references in items:
            payload = {
                "question": question,
                "refs": list(references),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

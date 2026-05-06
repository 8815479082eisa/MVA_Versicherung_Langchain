"""Unified thesis-oriented QA metrics for retrieval, answers, and citations."""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .benchmark_utils import best_token_f1, exact_match, normalize_text, token_f1

_STATEMENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\[\(\"'A-Z0-9])")


def split_into_statements(text: str) -> List[str]:
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    statements: List[str] = []
    for line in normalized.split("\n"):
        line = re.sub(r"^[-*â€¢]+\s*", "", line.strip())
        if not line:
            continue
        for piece in _STATEMENT_SPLIT_RE.split(line):
            piece = piece.strip()
            if piece:
                statements.append(piece)
    return statements


_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_GENERIC_CITATION_RE = re.compile(r"\[[^\[\]\n]{1,80}\]")

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "have",
    "will",
    "your",
    "are",
    "but",
    "not",
    "you",
    "can",
    "all",
    "was",
    "has",
    "had",
    "into",
    "their",
    "they",
    "them",
    "there",
    "about",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
    "wie",
    "und",
    "der",
    "die",
    "das",
    "ist",
    "sind",
    "eine",
    "einer",
    "einem",
    "einen",
    "eines",
    "ein",
    "nicht",
    "mit",
    "von",
    "auf",
    "zu",
    "im",
    "in",
}

KNOWN_FALLBACK_PREFIXES = (
    "i cannot provide a safe, policy-compliant answer for this request",
    "i could not generate an answer that is sufficiently supported by the available documents",
    "i cannot include or expose personal or sensitive information in the response",
    "the retrieved context contained information that could not be safely used for answering",
)


def _passage_text(passage: Any) -> str:
    if isinstance(passage, str):
        return passage
    if isinstance(passage, dict):
        return str(
            passage.get("page_content")
            or passage.get("text")
            or passage.get("content")
            or passage.get("snippet")
            or ""
        )
    return str(passage)


def _strip_citation_markers(text: str) -> str:
    return _GENERIC_CITATION_RE.sub("", text or "").strip()


def _has_inline_citation(statement: str) -> bool:
    return bool(_GENERIC_CITATION_RE.search(statement or ""))


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(normalize_text(text))
        if token not in STOPWORDS
    }


def lexical_support_score(claim: str, passage: str) -> float:
    claim_text = _strip_citation_markers(claim)
    normalized_claim = normalize_text(claim_text)
    normalized_passage = normalize_text(passage)
    if not normalized_claim or not normalized_passage:
        return 0.0
    if normalized_claim in normalized_passage:
        return 1.0

    overlap_f1 = token_f1(claim_text, passage)
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return overlap_f1

    passage_tokens = _content_tokens(passage)
    keyword_recall = len(claim_tokens & passage_tokens) / len(claim_tokens)
    return max(overlap_f1, keyword_recall)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.mean(values)


def _is_fallback_answer(prediction: str, audit_row: Optional[Dict[str, Any]] = None) -> bool:
    normalized_prediction = normalize_text(prediction)
    if any(normalized_prediction.startswith(prefix) for prefix in KNOWN_FALLBACK_PREFIXES):
        return True

    if not audit_row:
        return False

    safety_decision = str(audit_row.get("safety_decision") or "").strip().lower()
    return safety_decision.endswith("_fallback")


def build_qa_metric_row(
    *,
    question: str,
    references: Sequence[str],
    prediction: str,
    context_docs: Sequence[Any],
    sources: Sequence[Any],
    audit_row: Optional[Dict[str, Any]] = None,
    support_threshold: float = 0.2,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    context_doc_count = len(context_docs)
    source_count = len(sources)
    fallback_answer = _is_fallback_answer(prediction, audit_row)

    doc_support_scores: List[float] = []
    for doc in context_docs:
        passage = _passage_text(doc)
        if not passage:
            doc_support_scores.append(0.0)
            continue
        best_score = 0.0
        for reference in references:
            best_score = max(best_score, lexical_support_score(reference, passage))
        doc_support_scores.append(best_score)

    supportive_doc_count = sum(1 for score in doc_support_scores if score >= support_threshold)
    retrieval_support_hit = 1.0 if supportive_doc_count > 0 else 0.0
    retrieval_context_precision = (
        supportive_doc_count / context_doc_count if context_doc_count else 0.0
    )

    statements = split_into_statements(prediction)
    inline_citation_count = sum(1 for statement in statements if _has_inline_citation(statement))
    inline_citation_coverage = (
        inline_citation_count / len(statements) if statements else 0.0
    )

    statement_support_scores: List[float] = []
    for statement in statements:
        cleaned_statement = _strip_citation_markers(statement)
        if not cleaned_statement:
            statement_support_scores.append(0.0)
            continue
        best_score = 0.0
        for doc in context_docs:
            best_score = max(best_score, lexical_support_score(cleaned_statement, _passage_text(doc)))
        statement_support_scores.append(best_score)

    supported_statement_count = sum(
        1 for score in statement_support_scores if score >= support_threshold
    )
    if source_count > 0:
        citation_support_rate = (
            supported_statement_count / len(statements) if statements else 0.0
        )
    else:
        citation_support_rate = 0.0

    return {
        "question": question,
        "refs": list(references),
        "prediction": prediction,
        "error": error,
        "metrics": {
            "retrieval_support_hit": retrieval_support_hit,
            "retrieval_context_precision": retrieval_context_precision,
            "answer_exact_match": exact_match(prediction, references) if references else 0.0,
            "answer_token_f1": best_token_f1(prediction, references) if references else 0.0,
            "citation_source_presence": 1.0 if source_count > 0 else 0.0,
            "citation_support_rate": citation_support_rate,
        },
        "diagnostics": {
            "context_doc_count": context_doc_count,
            "source_count": source_count,
            "supportive_context_doc_count": supportive_doc_count,
            "best_context_support_score": max(doc_support_scores) if doc_support_scores else 0.0,
            "statement_count": len(statements),
            "supported_statement_count": supported_statement_count,
            "best_statement_support_score": max(statement_support_scores)
            if statement_support_scores
            else 0.0,
            "inline_citation_coverage": inline_citation_coverage,
            "fallback_answer": fallback_answer,
            "support_threshold": support_threshold,
            "retrieval_needed": audit_row.get("retrieval_needed") if audit_row else None,
            "safety_decision": audit_row.get("safety_decision") if audit_row else None,
            "latency_ms": audit_row.get("latency_ms") if audit_row else None,
            "retries": audit_row.get("retries") if audit_row else None,
        },
    }


def aggregate_qa_metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "retrieval_support_hit",
        "retrieval_context_precision",
        "answer_exact_match",
        "answer_token_f1",
        "citation_source_presence",
        "citation_support_rate",
    ]

    metric_values = {
        metric_name: [
            float(row["metrics"][metric_name])
            for row in rows
            if row.get("error") is None and row.get("metrics", {}).get(metric_name) is not None
        ]
        for metric_name in metric_names
    }

    diagnostics = {
        "processed_items": sum(1 for row in rows if row.get("error") is None),
        "failed_items": sum(1 for row in rows if row.get("error") is not None),
        "fallback_answers": sum(
            1 for row in rows if row.get("diagnostics", {}).get("fallback_answer") is True
        ),
        "mean_context_doc_count": _mean(
            [
                float(row["diagnostics"]["context_doc_count"])
                for row in rows
                if row.get("error") is None
            ]
        ),
        "mean_source_count": _mean(
            [
                float(row["diagnostics"]["source_count"])
                for row in rows
                if row.get("error") is None
            ]
        ),
        "inline_citation_coverage_mean": _mean(
            [
                float(row["diagnostics"]["inline_citation_coverage"])
                for row in rows
                if row.get("error") is None
            ]
        ),
    }

    framework = {
        "retrieval": [
            "retrieval_support_hit_rate",
            "retrieval_context_precision",
        ],
        "generated_answer": [
            "answer_exact_match",
            "answer_token_f1",
        ],
        "citation": [
            "citation_source_presence_rate",
            "citation_support_rate",
        ],
        "security": [
            "attack_block_rate",
            "benign_allow_rate",
        ],
    }

    return {
        "total_items": len(rows),
        "metrics": {
            "retrieval_support_hit_rate": _mean(metric_values["retrieval_support_hit"]),
            "retrieval_context_precision": _mean(metric_values["retrieval_context_precision"]),
            "answer_exact_match": _mean(metric_values["answer_exact_match"]),
            "answer_token_f1": _mean(metric_values["answer_token_f1"]),
            "citation_source_presence_rate": _mean(metric_values["citation_source_presence"]),
            "citation_support_rate": _mean(metric_values["citation_support_rate"]),
        },
        "diagnostics": diagnostics,
        "framework": framework,
    }

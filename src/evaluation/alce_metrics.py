"""ALCE-style metrics (citation quality + optional MAUVE fluency)."""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .nli import NLIModel


_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\[\(\"'A-Z0-9])")


def split_into_statements(text: str) -> List[str]:
    """Split free text into sentence-like statements."""
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    statements: List[str] = []
    for line in normalized.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove bullet marker while preserving content.
        line = re.sub(r"^[-*•]+\s*", "", line)
        pieces = _SENTENCE_SPLIT_RE.split(line)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                statements.append(piece)

    return statements


def extract_citation_indices(statement: str) -> List[int]:
    """Extract citation indices from [1][2]-style markers."""
    seen = set()
    indices: List[int] = []
    for match in _CITATION_RE.finditer(statement or ""):
        idx = int(match.group(1))
        if idx <= 0:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def citation_coverage(statements: Sequence[str]) -> float:
    """Fraction of statements containing at least one citation marker."""
    if not statements:
        return 0.0
    covered = sum(1 for statement in statements if extract_citation_indices(statement))
    return covered / len(statements)


def _extract_text_and_title(passage: Any) -> tuple[str, str]:
    if isinstance(passage, str):
        return "Untitled", passage

    if not isinstance(passage, dict):
        return "Untitled", str(passage)

    metadata = passage.get("metadata") or {}
    title = (
        passage.get("title")
        or passage.get("documentTitle")
        or passage.get("document_title")
        or passage.get("source")
        or metadata.get("source")
        or "Untitled"
    )

    text = (
        passage.get("page_content")
        or passage.get("text")
        or passage.get("content")
        or passage.get("snippet")
        or ""
    )
    return str(title).strip(), str(text).strip()


def format_premise(passage: Any) -> str:
    """Format a single citation passage into ALCE/TRUE prompt premise style."""
    title, text = _extract_text_and_title(passage)
    return f"Title: {title}\n{text}".strip()


def _concat_passages(passages: Iterable[Any]) -> str:
    parts = [format_premise(p) for p in passages]
    parts = [p for p in parts if p.strip()]
    return "\n\n".join(parts).strip()


def _statement_for_nli(statement: str) -> str:
    cleaned = _CITATION_RE.sub("", statement or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _get_statement_passages(
    statements: Sequence[str],
    cited_passages_by_statement: Sequence[Any] | Dict[Any, Any],
    index: int,
) -> List[Any]:
    if isinstance(cited_passages_by_statement, dict):
        statement = statements[index]
        raw = (
            cited_passages_by_statement.get(index)
            or cited_passages_by_statement.get(str(index))
            or cited_passages_by_statement.get(statement)
            or []
        )
    elif index < len(cited_passages_by_statement):
        raw = cited_passages_by_statement[index]
    else:
        raw = []

    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    return [raw]


def citation_recall(
    statements: Sequence[str],
    cited_passages_by_statement: Sequence[Any] | Dict[Any, Any],
    nli: NLIModel,
) -> float:
    """
    ALCE citation recall.

    For each statement s_i:
    recall_i = 1 iff C_i non-empty and entail(concat(C_i), s_i) is True, else 0.
    """
    if not statements:
        return 0.0

    hits = 0
    for index, statement in enumerate(statements):
        passages = _get_statement_passages(statements, cited_passages_by_statement, index)
        if not passages:
            continue
        hypothesis = _statement_for_nli(statement)
        if not hypothesis:
            continue
        premise = _concat_passages(passages)
        if premise and nli.entail(premise, hypothesis):
            hits += 1
    return hits / len(statements)


def citation_precision(
    statements: Sequence[str],
    cited_passages_by_statement: Sequence[Any] | Dict[Any, Any],
    nli: NLIModel,
) -> float:
    """
    ALCE citation precision.

    For each citation c_ij in statement s_i:
    irrelevant iff entail(c_ij, s_i)=0 and entail(concat(C_i \\ {c_ij}), s_i)=1.
    Precision is computed only for statements where statement-level recall=1.
    """
    relevant_citations = 0
    total_citations = 0

    for index, statement in enumerate(statements):
        passages = _get_statement_passages(statements, cited_passages_by_statement, index)
        if not passages:
            continue

        hypothesis = _statement_for_nli(statement)
        if not hypothesis:
            continue

        all_premise = _concat_passages(passages)
        if not all_premise or not nli.entail(all_premise, hypothesis):
            # Precision undefined for this statement by ALCE definition.
            continue

        for j, citation in enumerate(passages):
            total_citations += 1

            single_premise = format_premise(citation)
            single_entails = bool(single_premise) and nli.entail(single_premise, hypothesis)

            remaining = passages[:j] + passages[j + 1 :]
            remaining_entails = bool(remaining) and nli.entail(
                _concat_passages(remaining), hypothesis
            )

            irrelevant = (not single_entails) and remaining_entails
            if not irrelevant:
                relevant_citations += 1

    if total_citations == 0:
        return 0.0
    return relevant_citations / total_citations


def _truncate_words(text: str, max_words: int = 100) -> str:
    words = (text or "").split()
    return " ".join(words[:max_words])


def mauve_fluency(
    question: str,
    gen_answer: str,
    ref_answer: str,
) -> Optional[float]:
    """
    Optional MAUVE fluency score.

    Following ALCE note: use concatenated question + output and truncate to 100 words.
    """
    if not ref_answer:
        return None

    gen_text = _truncate_words(f"{question} {gen_answer}", max_words=100)
    ref_text = _truncate_words(f"{question} {ref_answer}", max_words=100)
    if not gen_text or not ref_text:
        return None

    try:
        import mauve
    except Exception:
        return None

    try:
        result = mauve.compute_mauve(
            p_text=[gen_text],
            q_text=[ref_text],
            device_id=-1,
            max_text_length=256,
            verbose=False,
        )
        return float(result.mauve)
    except Exception:
        return None


def aggregate_alce_results(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-item ALCE-style metrics."""

    def _mean(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return statistics.mean(values)

    coverage = [float(i["citation_coverage"]) for i in items if i.get("citation_coverage") is not None]
    recall = [float(i["citation_recall"]) for i in items if i.get("citation_recall") is not None]
    precision = [float(i["citation_precision"]) for i in items if i.get("citation_precision") is not None]
    mauve_scores = [float(i["mauve_fluency"]) for i in items if i.get("mauve_fluency") is not None]
    latencies = [float(i["latency_seconds"]) for i in items if i.get("latency_seconds") is not None]

    return {
        "total_items": len(items),
        "citation_coverage_mean": _mean(coverage),
        "citation_recall_mean": _mean(recall),
        "citation_precision_mean": _mean(precision),
        "mauve_fluency_mean": _mean(mauve_scores),
        "latency_seconds_mean": _mean(latencies),
        "items_with_recall": len(recall),
        "items_with_precision": len(precision),
        "items_with_mauve": len(mauve_scores),
    }

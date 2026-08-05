from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from src.core.coverage_taxonomy import (
    AMBIGUOUS_COMPREHENSIVE,
    PART_COMPREHENSIVE,
    find_coverage_terms,
    normalize_coverage_term,
)
from src.guardrails.integrations.nemo_actions import (
    _GROUNDING_ANSWER_REFERENCE_RE,
    _GROUNDING_CITATION_RE,
    _GROUNDING_CLAIM_SPLIT_RE,
    _GROUNDING_COVERAGE_RE,
    _claim_support_score,
    _coverage_polarity,
    _grounding_claims,
    _grounding_facts,
    _tokenize_grounding,
    calculate_groundedness_score,
)


EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION = "fact_aware_claim_support_v5_experimental"

_CITATION_RE = re.compile(
    r"\[(?P<source>[^,\]\r\n]+),\s*(?:physical\s+)?page\s+(?P<page>\d+)\]",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,}(?![A-Za-z0-9])"
)
_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)"
)
_PERCENT_RE = re.compile(r"(?<![\w])(?P<value>\d+(?:[.,]\d+)?)\s*(?:%|percent|per\s+cent)(?![\w])", re.IGNORECASE)
_CURRENCY_PREFIX_RE = re.compile(
    r"(?P<currency>CHF|EUR|USD|GBP|\$|€|£)\s*(?P<value>\d[\d\s.,'’ʼ]*)",
    re.IGNORECASE,
)
_CURRENCY_SUFFIX_RE = re.compile(
    r"(?P<value>\d[\d\s.,'’ʼ]*)\s*(?P<currency>CHF|EUR|USD|GBP|euros?|dollars?|francs?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_CUSTOMER_FIELD_RE = re.compile(r"\bCustomer\s*:\s*([^.(\r\n]+)", re.IGNORECASE)
_POLICY_FIELD_RE = re.compile(r"\bPolicy\s+number\s*:\s*([A-Za-z0-9-]+)", re.IGNORECASE)
_STATUS_FIELD_RE = re.compile(r"\bStatus\s*:\s*([A-Za-z][A-Za-z -]{1,30})", re.IGNORECASE)
_COVERAGE_FIELD_RE = re.compile(r"\bCoverage\s+type\s*:\s*([A-Za-z][A-Za-z -]{1,50})", re.IGNORECASE)
_STRUCTURED_ANSWER_PATTERNS = {
    "customer": re.compile(r"\b(?:belongs to|customer\s+(?:is|:)|held by|holds policy)\s+([A-Z][\w'’-]+\s+[A-Z][\w'’-]+)", re.IGNORECASE),
    "status": re.compile(r"\bstatus(?: is|:)\s*([A-Za-z][A-Za-z -]{1,30})", re.IGNORECASE),
    "coverage": re.compile(r"\b(?:with|coverage(?: type)?(?: is|:))\s+(Partial Coverage|Part Comprehensive Insurance|Part Comprehensive|Full Coverage|Comprehensive Coverage|Full Comprehensive Insurance|Fully Comprehensive Insurance|Liability Coverage|Teilkasko|Vollkasko)\b", re.IGNORECASE),
}


def _normalize_space(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _normalize_number(value: str) -> str:
    compact = re.sub(r"[\s'’ʼ]", "", value)
    if compact.count(",") == 1 and "." not in compact:
        left, right = compact.split(",")
        compact = left + right if len(right) == 3 else left + "." + right
    elif compact.count(".") == 1 and "," not in compact:
        left, right = compact.split(".")
        compact = left + right if len(right) == 3 else compact
    else:
        compact = compact.replace(",", "").replace(".", "")
    try:
        number = float(compact)
    except ValueError:
        return compact
    return str(int(number)) if number.is_integer() else f"{number:.8f}".rstrip("0").rstrip(".")


def _normalize_currency(value: str) -> str:
    aliases = {"$": "usd", "€": "eur", "£": "gbp", "euros": "eur", "euro": "eur", "dollars": "usd", "dollar": "usd", "francs": "chf", "franc": "chf"}
    return aliases.get(value.casefold(), value.casefold())


def extract_atomic_facts(text: str) -> set[str]:
    cleaned = _CITATION_RE.sub(" ", _GROUNDING_CITATION_RE.sub(" ", text or ""))
    facts: set[str] = set()
    occupied: list[tuple[int, int]] = []
    for match in _IDENTIFIER_RE.finditer(cleaned):
        facts.add(f"identifier:{match.group(0).casefold()}")
        occupied.append(match.span())
    for match in _DATE_RE.finditer(cleaned):
        facts.add(f"date:{match.group(0).casefold()}")
        occupied.append(match.span())
    for match in _PERCENT_RE.finditer(cleaned):
        facts.add(f"percent:{_normalize_number(match.group('value'))}")
        occupied.append(match.span())
    for pattern in (_CURRENCY_PREFIX_RE, _CURRENCY_SUFFIX_RE):
        for match in pattern.finditer(cleaned):
            facts.add(
                f"money:{_normalize_currency(match.group('currency'))}:{_normalize_number(match.group('value'))}"
            )
            occupied.append(match.span())
    for match in _NUMBER_RE.finditer(cleaned):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        facts.add(f"number:{_normalize_number(match.group(0))}")
    return facts


def _metadata_values(docs: Sequence[Document], key: str) -> set[str]:
    return {
        _normalize_space(str(doc.metadata.get(key, "")))
        for doc in docs
        if str(doc.metadata.get(key, "")).strip()
    }


def _portable_basename(value: str) -> str:
    return re.split(r"[\\/]", str(value or ""))[-1]


def _citation_mismatches(answer: str, docs: Sequence[Document]) -> list[str]:
    mismatches: list[str] = []
    source_names = {
        _portable_basename(
            str(doc.metadata.get("source_file") or doc.metadata.get("source") or "")
        ).casefold()
        for doc in docs
        if doc.metadata.get("source_file") or doc.metadata.get("source")
    }
    pages = {
        int(doc.metadata["source_page"])
        for doc in docs
        if doc.metadata.get("source_page") not in (None, "")
    }
    for match in _CITATION_RE.finditer(answer or ""):
        source = _portable_basename(match.group("source").strip()).casefold()
        page = int(match.group("page"))
        if source_names and source not in source_names:
            mismatches.append(f"citation_source:{source}")
        if pages and page not in pages:
            mismatches.append(f"citation_page:{page}")
    return mismatches


def _current_policy_mismatches(answer: str, docs: Sequence[Document], query: str) -> list[str]:
    if not re.search(r"\b(?:current|currently|aktuell|derzeitig)\b", query or "", re.IGNORECASE):
        return []
    query_dates = [match.group(0) for match in _DATE_RE.finditer(query or "")]
    reference_date = max(query_dates, default="9999-12-31")
    product_terms = {
        "motor": ("motor", "vehicle", "kfz"),
        "liability": ("liability", "haftpflicht"),
    }
    requested_product = next(
        (name for name, terms in product_terms.items() if any(term in (query or "").casefold() for term in terms)),
        None,
    )
    policies: list[tuple[str, str]] = []
    for doc in docs:
        text = doc.page_content or ""
        policy = _POLICY_FIELD_RE.search(text)
        start = re.search(r"\bStart\s+date\s*:\s*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        end = re.search(r"\bEnd\s+date\s*:\s*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        product = re.search(r"\bProduct\s+type\s*:\s*([^.\r\n]+)", text, re.IGNORECASE)
        status = _STATUS_FIELD_RE.search(text)
        if not (policy and start and end):
            continue
        if not (start.group(1) <= reference_date <= end.group(1)):
            continue
        if status and _normalize_space(status.group(1)) not in {"active", "aktiv"}:
            continue
        product_text = _normalize_space(product.group(1)) if product else ""
        if requested_product and not any(term in product_text for term in product_terms[requested_product]):
            continue
        policies.append((start.group(1), policy.group(1).casefold()))
    if not policies:
        return []
    expected = max(policies)[1]
    answer_policies = {match.group(0).casefold() for match in _IDENTIFIER_RE.finditer(answer or "")}
    if answer_policies and expected not in answer_policies:
        return [f"current_policy:{expected}"]
    return []


def _structured_mismatches(answer: str, docs: Sequence[Document], query: str) -> list[str]:
    document_text = "\n".join(doc.page_content or "" for doc in docs)
    mismatches: list[str] = []
    field_specs = (
        ("customer", _CUSTOMER_FIELD_RE, _STRUCTURED_ANSWER_PATTERNS["customer"]),
        ("status", _STATUS_FIELD_RE, _STRUCTURED_ANSWER_PATTERNS["status"]),
        ("coverage", _COVERAGE_FIELD_RE, _STRUCTURED_ANSWER_PATTERNS["coverage"]),
    )
    for name, context_pattern, answer_pattern in field_specs:
        context_values = {
            _normalize_space(match.group(1)) for match in context_pattern.finditer(document_text)
        }
        answer_match = answer_pattern.search(answer or "")
        if context_values and answer_match:
            observed = _normalize_space(answer_match.group(1))
            if name == "coverage":
                observed_canonical = normalize_coverage_term(observed)
                expected_canonicals = {
                    normalize_coverage_term(expected) for expected in context_values
                }
                expected_canonicals.discard(None)
                if expected_canonicals and observed_canonical not in expected_canonicals:
                    mismatches.append(f"structured_{name}:{observed}")
                continue
            if not any(expected in observed or observed in expected for expected in context_values):
                mismatches.append(f"structured_{name}:{observed}")
    crm_coverage_values = {
        match.group(1).strip()
        for match in _COVERAGE_FIELD_RE.finditer(document_text)
        if match.group(1).strip()
    }
    if crm_coverage_values:
        answer_terms = find_coverage_terms(answer)
        expected_canonicals = {
            normalize_coverage_term(expected) for expected in crm_coverage_values
        }
        expected_canonicals.discard(None)
        conflicting = sorted(
            {
                term.canonical
                for term in answer_terms
                if term.canonical not in expected_canonicals
                and (
                    term.canonical != AMBIGUOUS_COMPREHENSIVE
                    or PART_COMPREHENSIVE in expected_canonicals
                )
            }
        )
        if expected_canonicals and conflicting:
            mismatches.append(
                "coverage_taxonomy:"
                + ",".join(sorted(expected_canonicals))
                + "!="
                + ",".join(conflicting)
            )
    expected_policies = {match.group(1).casefold() for match in _POLICY_FIELD_RE.finditer(document_text)}
    if expected_policies:
        answer_policies = {match.group(0).casefold() for match in _IDENTIFIER_RE.finditer(answer or "")}
        if answer_policies and not answer_policies & expected_policies:
            mismatches.append("structured_policy_identifier")
    mismatches.extend(_current_policy_mismatches(answer, docs, query))
    return mismatches


def _polarity_mismatches(answer: str, document_text: str) -> list[str]:
    mismatches: list[str] = []
    for claim in _grounding_claims(answer):
        claim_polarity = _coverage_polarity(claim)
        if claim_polarity is None:
            continue
        claim_tokens = _tokenize_grounding(claim)
        candidates: list[tuple[float, bool | None]] = []
        for statement in _GROUNDING_CLAIM_SPLIT_RE.split(document_text):
            evidence_polarity = _coverage_polarity(statement)
            if evidence_polarity is None:
                continue
            overlap = len(claim_tokens & _tokenize_grounding(statement)) / max(len(claim_tokens), 1)
            candidates.append((overlap, evidence_polarity))
        if candidates:
            same_polarity_overlap = max(
                (
                    overlap
                    for overlap, evidence_polarity in candidates
                    if evidence_polarity == claim_polarity
                ),
                default=0.0,
            )
            opposite_polarity_overlap = max(
                (
                    overlap
                    for overlap, evidence_polarity in candidates
                    if evidence_polarity is not None
                    and evidence_polarity != claim_polarity
                ),
                default=0.0,
            )
            if (
                opposite_polarity_overlap >= 0.35
                and opposite_polarity_overlap > same_polarity_overlap + 0.05
            ):
                mismatches.append("coverage_polarity")
    return mismatches


@dataclass(frozen=True)
class ExperimentalGroundednessResult:
    score: float
    base_v4_score: float
    minimum_claim_support: float
    average_claim_support: float
    unsupported_atomic_facts: tuple[str, ...]
    citation_mismatches: tuple[str, ...]
    structured_mismatches: tuple[str, ...]
    polarity_mismatches: tuple[str, ...]
    applied_caps: tuple[str, ...]
    final_v5_score: float = 0.0
    structured_field_mismatches: tuple[str, ...] = ()
    extracted_claims: tuple[str, ...] = ()
    ignored_nonsemantic_lines: tuple[dict[str, str], ...] = ()
    claim_details: tuple[dict[str, Any], ...] = ()
    coverage_type_mismatches: tuple[str, ...] = ()
    policy_number_mismatches: tuple[str, ...] = ()
    numeric_or_monetary_mismatches: tuple[str, ...] = ()
    exception_type: str | None = None
    exception_message_sanitized: str | None = None
    algorithm_version: str = EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ignored_nonsemantic_lines(answer: str) -> tuple[dict[str, str], ...]:
    ignored: list[dict[str, str]] = []
    for raw_line in (answer or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        without_bullets = stripped.strip(" \t-*\u2022")
        if stripped.startswith("#"):
            ignored.append({"text": stripped, "reason_code": "markdown_heading"})
            continue
        without_citations = _GROUNDING_ANSWER_REFERENCE_RE.sub(" ", without_bullets).strip()
        normalized = re.sub(r"\s+", " ", without_citations).strip().casefold()
        if not normalized and _GROUNDING_ANSWER_REFERENCE_RE.search(without_bullets):
            ignored.append({"text": stripped, "reason_code": "citation_only_source_line"})
        elif re.fullmatch(r"\d+[.)]?", normalized):
            ignored.append({"text": stripped, "reason_code": "list_enumerator"})
        elif re.fullmatch(r"(?:source|sources|citations?)\s*:\s*\.?", normalized):
            ignored.append({"text": stripped, "reason_code": "source_heading"})
        elif (
            normalized.endswith(":")
            and len(_tokenize_grounding(without_citations)) <= 12
        ):
            ignored.append({"text": stripped, "reason_code": "section_heading"})
        elif re.search(
            r"\b(?:general information|general overview|information is general)\b.*"
            r"\b(?:not|should not be considered)\b.*"
            r"\b(?:a )?final (?:claim|coverage)(?: or (?:claim|coverage))? decision\b",
            normalized,
        ):
            ignored.append({"text": stripped, "reason_code": "decision_disclaimer"})
        elif re.search(
            r"\bwithout (?:making|providing)\b.*"
            r"\b(?:a )?final (?:claim|coverage)(?: or (?:claim|coverage))? decision\b",
            normalized,
        ):
            ignored.append({"text": stripped, "reason_code": "decision_disclaimer"})
        elif re.search(
            r"\bthis information provides? (?:a )?general overview\b|"
            r"\b(?:definitive|final) (?:claim|coverage)?\s*decision\b.*\bfurther review\b",
            normalized,
        ):
            ignored.append({"text": stripped, "reason_code": "decision_disclaimer"})
    return tuple(ignored)


def _claim_type(claim: str) -> str:
    facts = _grounding_facts(claim)
    if any(fact.startswith("identifier:") for fact in facts):
        return "policy_identifier"
    if any(fact.startswith(("money:", "percent:", "date:", "number:")) for fact in facts):
        return "numeric_or_monetary"
    if any(fact.startswith("coverage:") for fact in facts) or _GROUNDING_COVERAGE_RE.search(claim):
        return "coverage"
    if any(fact.startswith("deductible:") for fact in facts):
        return "deductible"
    return "semantic"


def _claim_diagnostics(
    claims: Sequence[str],
    documents: Sequence[Document],
    query: str,
    document_text: str,
    applied_caps: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    query_tokens = _tokenize_grounding(query)
    query_relevance = [
        len(query_tokens & _tokenize_grounding(doc.page_content or "")) / max(len(query_tokens), 1)
        if query_tokens
        else 0.0
        for doc in documents
    ]
    max_query_relevance = max(query_relevance, default=0.0)
    rows: list[dict[str, Any]] = []
    for claim in claims:
        claim_tokens = _tokenize_grounding(claim)
        claim_facts = _grounding_facts(claim)
        scored_facts = {fact for fact in claim_facts if not fact.startswith("deductible:")} or claim_facts
        candidates: list[tuple[float, int, float, float, float]] = []
        for index, doc in enumerate(documents):
            doc_text = doc.page_content or ""
            doc_tokens = _tokenize_grounding(doc_text)
            lexical_overlap = len(claim_tokens & doc_tokens) / max(len(claim_tokens), 1)
            doc_facts = _grounding_facts(doc_text)
            hard_fact_overlap = len(scored_facts & doc_facts) / max(len(scored_facts), 1) if scored_facts else 1.0
            raw_support = _claim_support_score(claim, doc_text)
            adjusted_support = raw_support
            if max_query_relevance > 0.0:
                adjusted_support *= 0.85 + (0.15 * query_relevance[index] / max_query_relevance)
            adjusted_support *= max(0.9, 1.0 - (0.03 * index))
            candidates.append((adjusted_support, index, lexical_overlap, hard_fact_overlap, raw_support))
        best = max(candidates, default=(0.0, -1, 0.0, 0.0, 0.0), key=lambda item: item[0])
        best_doc = documents[best[1]] if best[1] >= 0 else None
        metadata = best_doc.metadata if best_doc is not None else {}
        source = str(metadata.get("source_file") or metadata.get("source") or "")
        metadata_page = metadata.get("source_page")
        if metadata_page in (None, ""):
            metadata_page = metadata.get("page")
        try:
            physical_page = int(metadata_page) + 1 if metadata_page not in (None, "") else None
        except (TypeError, ValueError):
            physical_page = metadata_page
        v5_claim_score = _claim_support_score(claim, document_text)
        claim_caps: list[str] = []
        claim_atomic_facts = extract_atomic_facts(claim)
        if any(fact.startswith("identifier:") for fact in claim_atomic_facts) and "unsupported_identifier_cap_0.20" in applied_caps:
            claim_caps.append("unsupported_identifier_cap_0.20")
        if any(fact.startswith(("money:", "percent:", "date:", "number:")) for fact in claim_atomic_facts) and "unsupported_numeric_fact_cap_0.38" in applied_caps:
            claim_caps.append("unsupported_numeric_fact_cap_0.38")
        rows.append(
            {
                "claim_text": claim,
                "claim_type": _claim_type(claim),
                "best_support_chunk": (best_doc.page_content or "")[:600] if best_doc is not None else None,
                "source_file": Path(source).name if source else None,
                "page": physical_page,
                "chunk_id": metadata.get("chunk_id"),
                "lexical_overlap": round(best[2], 6),
                "hard_fact_overlap": round(best[3], 6),
                "v4_claim_score": round(best[0], 6),
                "v5_claim_score": round(v5_claim_score, 6),
                "applied_penalty_or_cap": claim_caps,
                "passed": bool(v5_claim_score > 0.0),
                "reason_code": "claim_support_nonzero" if v5_claim_score > 0.0 else "no_claim_support",
            }
        )
    return tuple(rows)


def calculate_groundedness_score_v5_experimental(
    answer: str,
    docs: Sequence[Document],
    query: str = "",
) -> ExperimentalGroundednessResult:
    """Experimental hard-fact-aware score; this function is not wired into production."""

    documents = [doc for doc in docs if (doc.page_content or "").strip()]
    document_text = "\n".join(doc.page_content or "" for doc in documents)
    ignored_nonsemantic_lines = _ignored_nonsemantic_lines(answer)
    ignored_texts = {item["text"] for item in ignored_nonsemantic_lines}
    semantic_answer = "\n".join(
        line
        for line in (answer or "").splitlines()
        if line.strip() not in ignored_texts
    )
    base_score = float(calculate_groundedness_score(semantic_answer, docs, query))
    claims = _grounding_claims(semantic_answer)
    claim_supports = [_claim_support_score(claim, document_text) for claim in claims]
    minimum_claim_support = min(claim_supports, default=0.0)
    average_claim_support = (
        sum(claim_supports) / len(claim_supports)
        if claim_supports
        else 0.0
    )
    conservative_score = min(
        base_score,
        (0.72 * base_score) + (0.28 * minimum_claim_support),
    )
    multi_chunk_score = (
        (0.20 * base_score)
        + (0.55 * average_claim_support)
        + (0.25 * minimum_claim_support)
    )
    score = max(conservative_score, multi_chunk_score)

    answer_facts = extract_atomic_facts(semantic_answer)
    document_facts = extract_atomic_facts(document_text)
    unsupported_facts = sorted(answer_facts - document_facts)
    citation_mismatches = _citation_mismatches(answer, documents)
    structured_mismatches = _structured_mismatches(answer, documents, query)
    polarity_mismatches = _polarity_mismatches(answer, document_text)

    applied_caps: list[str] = []
    if any(fact.startswith("identifier:") for fact in unsupported_facts) or "structured_policy_identifier" in structured_mismatches:
        score = min(score, 0.20)
        applied_caps.append("unsupported_identifier_cap_0.20")
    if any(fact.startswith(("money:", "percent:", "date:", "number:")) for fact in unsupported_facts):
        score = min(score, 0.38)
        applied_caps.append("unsupported_numeric_fact_cap_0.38")
    if structured_mismatches:
        score = min(score, 0.30)
        applied_caps.append("structured_field_mismatch_cap_0.30")
    if polarity_mismatches:
        score = min(score, 0.25)
        applied_caps.append("coverage_polarity_mismatch_cap_0.25")
    if citation_mismatches:
        score = min(score, 0.35)
        applied_caps.append("citation_mismatch_cap_0.35")

    coverage_type_mismatches = tuple(
        mismatch
        for mismatch in (*structured_mismatches, *polarity_mismatches)
        if mismatch.startswith(("structured_coverage:", "coverage_taxonomy:", "coverage_polarity"))
    )
    policy_number_mismatches = tuple(
        mismatch
        for mismatch in structured_mismatches
        if mismatch.startswith(("structured_policy_identifier", "current_policy:"))
    )
    numeric_or_monetary_mismatches = tuple(
        fact
        for fact in unsupported_facts
        if fact.startswith(("money:", "percent:", "date:", "number:"))
    )

    return ExperimentalGroundednessResult(
        score=round(max(0.0, min(1.0, score)), 6),
        base_v4_score=round(base_score, 6),
        minimum_claim_support=round(minimum_claim_support, 6),
        average_claim_support=round(average_claim_support, 6),
        unsupported_atomic_facts=tuple(unsupported_facts),
        citation_mismatches=tuple(citation_mismatches),
        structured_mismatches=tuple(structured_mismatches),
        polarity_mismatches=tuple(polarity_mismatches),
        applied_caps=tuple(applied_caps),
        final_v5_score=round(max(0.0, min(1.0, score)), 6),
        structured_field_mismatches=tuple(structured_mismatches),
        extracted_claims=tuple(claims),
        ignored_nonsemantic_lines=ignored_nonsemantic_lines,
        claim_details=_claim_diagnostics(
            claims,
            documents,
            query,
            document_text,
            applied_caps,
        ),
        coverage_type_mismatches=coverage_type_mismatches,
        policy_number_mismatches=policy_number_mismatches,
        numeric_or_monetary_mismatches=numeric_or_monetary_mismatches,
    )

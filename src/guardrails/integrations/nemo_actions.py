from __future__ import annotations

import hashlib
import math
import re
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from langchain_core.documents import Document
from src.core.coverage_polarity import coverage_polarity, coverage_relation

from nemoguardrails import LLMRails
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

try:
    from config.models import SafetyConfig, load_model_settings
    from core.safety_audit import (
        detect_pii,
        detect_query_hard_injection_signals,
        detect_sensitive_data_request_signals,
        detect_system_secret_request_signals,
        detect_system_secret_value_signals,
        is_allowed_pii,
        sanitize_pii,
    )
except Exception:
    from src.config.models import SafetyConfig, load_model_settings
    from src.core.safety_audit import (
        detect_pii,
        detect_query_hard_injection_signals,
        detect_sensitive_data_request_signals,
        detect_system_secret_request_signals,
        detect_system_secret_value_signals,
        is_allowed_pii,
        sanitize_pii,
    )


UNSAFE_QUERY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explosive_instructions", re.compile(r"\bbuild (a )?(bomb|explosive)\b", re.IGNORECASE)),
    (
        "credential_theft",
        re.compile(
            r"\b(?:how|what(?:'s| is)?(?: the)?(?: easiest)? way|which strategy|what approach|explain)\b[\s\S]{0,120}\b(?:hack|phish|steal(?:ing)?|break\s+into|take\s*over|bypass)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "malicious_access_intent",
        re.compile(
            r"\b(?:unauthorized|without permission|without being noticed)\b[\s\S]{0,80}\b(?:access|retrieve|steal|exfiltrat(?:e|ion)|records?|data)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unauthorized_private_data_retrieval",
        re.compile(
            r"\b(?:retrieve|get|access|obtain)\b[\s\S]{0,80}\b(?:private|confidential|customer|claim|payout)\b[\s\S]{0,80}\b(?:details?|records?|files?|data)\b[\s\S]{0,80}\b(?:without permission|unauthorized|without being noticed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_exfiltration_intent",
        re.compile(
            r"\b(?:exfiltrat(?:e|ion)|harvest|leak|data theft)\b[\s\S]{0,80}\b(?:customer|policyholder|insurance|account|record|data|credentials?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_request_for_data_theft",
        re.compile(
            r"\b(?:provide|give|share|output)\b[\s\S]{0,80}\b(?:unrestricted|step[- ]?by[- ]?step|instructions?)\b[\s\S]{0,120}\b(?:steal(?:ing)?|exfiltrat(?:e|ion)|harvest|hack|bypass)\b[\s\S]{0,80}\b(?:customer|policyholder|insurance|account|records?|data)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claim_files_theft_intent",
        re.compile(
            r"\b(?:collect|steal|obtain|access)\b[\s\S]{0,80}\b(?:private|confidential)\b[\s\S]{0,50}\b(?:claim files?|payout details?|customer records?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "stealth_data_theft_intent",
        re.compile(
            r"\b(?:evade detection|avoid detection)\b[\s\S]{0,120}\b(?:steal(?:ing)?|exfiltrat(?:e|ion)|harvest)\b[\s\S]{0,80}\b(?:customer|records?|data)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal_tool_abuse_data_theft",
        re.compile(
            r"\b(?:misuse|abuse)\b[\s\S]{0,80}\binternal\b[\s\S]{0,60}\b(?:tools?|systems?)\b[\s\S]{0,80}\b(?:data theft|steal(?:ing)?|exfiltrat(?:e|ion)|leak)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_takeover_intent",
        re.compile(
            r"\b(?:break\s+into|get|gain|obtain|take\s*over)\b[\s\S]{0,100}\b(?:insurance|customer|portal|account)\b[\s\S]{0,60}\b(?:access|credentials?|login|verification)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "phishing_or_password_reset_abuse",
        re.compile(
            r"\b(?:phishing|password reset)\b[\s\S]{0,80}\b(?:credentials?|access|account|records?|policy)\b",
            re.IGNORECASE,
        ),
    ),
    ("malware", re.compile(r"\bmalware|ransomware\b", re.IGNORECASE)),
)

GROUNDING_STOPWORDS = {
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
    "a",
    "an",
    "as",
    "at",
    "be",
    "by",
    "does",
    "is",
    "it",
    "of",
    "on",
    "or",
    "to",
    "under",
    "yes",
    "also",
    "applies",
    "according",
    "general",
    "terms",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einer",
    "für",
    "im",
    "in",
    "ist",
    "mit",
    "oder",
    "und",
    "von",
    "zu",
}

GROUNDING_ALGORITHM_VERSION = "claim_entailment_v1"

AUTHORIZED_INTERNAL_CUSTOMER_DATA = "AUTHORIZED_INTERNAL_CUSTOMER_DATA"
SYSTEM_SECRET_REQUEST = "SYSTEM_SECRET_REQUEST"
UNGROUNDED_PERSONAL_DATA = "UNGROUNDED_PERSONAL_DATA"
ENTITY_MISMATCH = "ENTITY_MISMATCH"
PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
READ_ONLY_ACCESS_ALLOWED = "READ_ONLY_ACCESS_ALLOWED"

_GROUNDING_CITATION_RE = re.compile(r"\[[^\]\r\n]+:\d+\]")
_GROUNDING_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_GROUNDING_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,}(?![A-Za-z0-9])"
)
_GROUNDING_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\d)"
)
_GROUNDING_ENGLISH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_GROUNDING_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_GROUNDING_COMPACT_UNIT_NUMBER_RE = re.compile(
    r"(?<![\w])\d+(?:[.,]\d+)?(?=[A-Za-z])"
)
# Keep legal references such as "Art. 4 Abs. 2" in one claim.  Splitting on
# the abbreviation periods turns an otherwise supported sentence into tiny
# claims ("Art.", "4 Abs.", "2 ...") and makes the groundedness score fail.
_GROUNDING_CLAIM_SPLIT_RE = re.compile(
    r"(?<!Art\.)(?<!Abs\.)(?<!Nr\.)(?<=[.!?])\s+|[\r\n]+|\s*;\s*"
)
_GROUNDING_ANSWER_REFERENCE_RE = re.compile(
    r"\[(?:CRM:[^\]\r\n]+|[^,\]\r\n]+,\s*(?:physical\s+)?page\s+\d+"
    r"(?:\s*(?:[-–;,]|and)\s*(?:page\s+)?\d+)*|[^\]\r\n]+:\d+)\]",
    re.IGNORECASE,
)
_GROUNDING_EVIDENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_GROUNDING_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|excluded|uncovered|nicht|kein|keine|keinen|keinem|"
    r"keiner|keines|ohne|ausgeschlossen)\b",
    re.IGNORECASE,
)
_GROUNDING_COVERAGE_RE = re.compile(
    r"\b(?:cover(?:age|ed|s|ing)?|insur(?:ance|ed)|exclude(?:d|s)?|uncovered|"
    r"versicher(?:t|ung|te|ten)|gedeckt|abgedeckt|ausgeschlossen|übernommen)\b",
    re.IGNORECASE,
)
_GROUNDING_COVERAGE_PREDICATE_RE = re.compile(
    r"\b(?:cover(?:age|ed|s|ing)?|exclude(?:d|s)?|uncovered|versichert|gedeckt|"
    r"abgedeckt|ausgeschlossen|übernommen)\b",
    re.IGNORECASE,
)
_GROUNDING_MISSING_INFORMATION_RE = re.compile(
    r"\b(?:does? not state|do not state|information (?:is )?missing|cannot determine|"
    r"insufficient information|not available|keine information)\b",
    re.IGNORECASE,
)
_GROUNDING_DEDUCTIBLE_RE = re.compile(
    r"\b(?:deductible|excess|selbstbeteiligung)\b",
    re.IGNORECASE,
)
_GROUNDING_ZERO_DEDUCTIBLE_RE = re.compile(
    r"\b(?:no|zero|without|kein|keine|keinen|ohne)\b[\w\s-]{0,24}"
    r"\b(?:deductible|excess|selbstbeteiligung)\b|"
    r"\bwill not have to bear\b[\w\s-]{0,24}"
    r"\b(?:deductible|excess|selbstbeteiligung)\b|"
    r"\b(?:deductible|excess|selbstbeteiligung)\b[\w\s-]{0,24}"
    r"\b(?:does not|doesn't|not|zero|entfällt|keine)\b",
    re.IGNORECASE,
)
_GROUNDING_COVERAGE_TYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "coverage:partial_comprehensive",
        re.compile(
            r"\b(?:partial(?:ly)? comprehensive|teilkasko)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "coverage:fully_comprehensive",
        re.compile(r"\b(?:fully comprehensive|comprehensive-only|vollkasko)\b", re.IGNORECASE),
    ),
    (
        "coverage:personal_liability",
        re.compile(r"\b(?:personal liability|privathaftpflicht)\b", re.IGNORECASE),
    ),
)

_GROUNDING_TOKEN_ALIASES = {
    "covered": "cover",
    "covers": "cover",
    "coverage": "cover",
    "covering": "cover",
    "include": "cover",
    "included": "cover",
    "includes": "cover",
    "benefits": "benefit",
    "damages": "damage",
    "caused": "damage",
    "claims": "claim",
    "each": "per",
    "euros": "euro",
    "insured": "insurance",
    "insurer": "insurance",
    "teilkaskoversicherung": "teilkasko",
    "versichert": "versicherung",
    "versicherte": "versicherung",
    "versicherten": "versicherung",
    "schäden": "schaden",
    "schadens": "schaden",
    "schadenfall": "claim",
    "intentionally": "intentional",
    "reimbursed": "reimburse",
    "reimburses": "reimburse",
    "windscreen": "windshield",
}

_grounding_embeddings: Any = None
_grounding_embeddings_lock = threading.RLock()
_grounding_semantic_cache: dict[str, list[list[float]]] = {}


def configure_grounding_embeddings(embeddings: Any) -> None:
    """Share the already-loaded multilingual retriever embedder with guardrails."""

    global _grounding_embeddings
    with _grounding_embeddings_lock:
        if embeddings is _grounding_embeddings:
            return
        _grounding_embeddings = embeddings
        _grounding_semantic_cache.clear()


def _semantic_claim_support_scores(
    claims: Sequence[str],
    documents: Sequence[str],
) -> list[list[float]]:
    """Return conservative cross-lingual support scores for claim/document pairs."""

    if _grounding_embeddings is None or not claims or not documents:
        return [[0.0 for _ in documents] for _ in claims]
    # Groundedness runs on every answer.  Encoding every retrieved parent chunk
    # with BGE-M3 on a CPU can exceed the API guardrail timeout, while the
    # reranker has already ordered the most relevant evidence first.  Keep the
    # semantic rescue deliberately small and bounded; lexical/fact checks still
    # evaluate the complete context below.
    selected_documents = list(documents[:8])
    semantic_claims = [str(claim)[:512] for claim in claims]
    semantic_documents = [str(document)[:1200] for document in selected_documents]
    payload = "\x1e".join(
        (str(len(documents)), *semantic_claims, "\x1d", *semantic_documents)
    )
    key = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
    with _grounding_embeddings_lock:
        cached = _grounding_semantic_cache.get(key)
        if cached is not None:
            return cached
        vectors = _grounding_embeddings.embed_documents(
            [*semantic_claims, *semantic_documents]
        )
        claim_vectors = vectors[: len(semantic_claims)]
        document_vectors = vectors[len(semantic_claims) :]
        result: list[list[float]] = []
        for claim_vector in claim_vectors:
            claim_norm = math.sqrt(sum(float(value) ** 2 for value in claim_vector)) or 1.0
            row: list[float] = []
            for document_vector in document_vectors:
                document_norm = (
                    math.sqrt(sum(float(value) ** 2 for value in document_vector)) or 1.0
                )
                cosine = sum(
                    float(left) * float(right)
                    for left, right in zip(claim_vector, document_vector)
                ) / (claim_norm * document_norm)
                # BGE-M3 unrelated insurance passages can still have moderately
                # positive cosine similarity.  Only the high-similarity tail may
                # rescue a low lexical score; hard-fact and polarity checks remain.
                calibrated = max(0.0, min(1.0, (cosine - 0.42) / 0.35))
                row.append(calibrated)
            # Preserve the caller's document indexing; documents beyond the
            # bounded semantic window simply cannot rescue a lexical miss.
            result.append(row + [0.0] * (len(documents) - len(row)))
        if len(_grounding_semantic_cache) >= 32:
            _grounding_semantic_cache.pop(next(iter(_grounding_semantic_cache)))
        _grounding_semantic_cache[key] = result
        return result


def _resolve_safety_config(project_safety_config: Optional[SafetyConfig]) -> SafetyConfig:
    if isinstance(project_safety_config, SafetyConfig):
        return project_safety_config
    return load_model_settings().safety


def _find_matching_patterns(text: str, patterns: Sequence[tuple[str, re.Pattern[str]]]) -> list[str]:
    return [name for name, pattern in patterns if pattern.search(text or "")]


def _allowed_pii_items(items: Sequence[Any], config: SafetyConfig) -> list[Any]:
    return [item for item in items if is_allowed_pii(item, config)]


def _sensitive_pii_items(items: Sequence[Any], config: SafetyConfig) -> list[Any]:
    return [item for item in items if not is_allowed_pii(item, config)]


def _summarize_pii_items(items: Sequence[Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in items:
        pii_type = str(getattr(item, "pii_type", "unknown"))
        summary[pii_type] = summary.get(pii_type, 0) + 1
    return summary


def _pii_details(items: Sequence[Any], config: SafetyConfig) -> dict[str, Any]:
    allowed_items = _allowed_pii_items(items, config)
    sensitive_items = _sensitive_pii_items(items, config)
    return {
        "detected_count": len(items),
        "allowed_count": len(allowed_items),
        "redacted_count": len(sensitive_items),
        "detected_types": _summarize_pii_items(items),
        "allowed_types": _summarize_pii_items(allowed_items),
        "redacted_types": _summarize_pii_items(sensitive_items),
        "items": [
            {
                "pii_type": getattr(item, "pii_type", "unknown"),
                "start": getattr(item, "start", None),
                "end": getattr(item, "end", None),
                "allowed": is_allowed_pii(item, config),
                "source": getattr(item, "source", "unknown"),
                "reason": getattr(item, "reason", ""),
            }
            for item in items
        ],
    }


def _append_pii_reasons(
    reasons: list[str],
    *,
    stage: str,
    items: Sequence[Any],
    config: SafetyConfig,
) -> None:
    allowed_items = _allowed_pii_items(items, config)
    sensitive_items = _sensitive_pii_items(items, config)

    if allowed_items:
        reasons.append("pii_allowed_business_contact")

    if not sensitive_items:
        return

    if stage == "answer":
        reasons.append("response_contains_pii")
    elif stage == "query":
        reasons.append("query_contains_pii")
    elif stage == "context":
        reasons.append("context_contains_pii")

    reasons.append(f"pii_{stage}_redacted")
    for pii_type in sorted({str(getattr(item, "pii_type", "unknown")) for item in sensitive_items}):
        reasons.append(f"pii_{pii_type}_redacted")


def _normalize_grounding_token(token: str) -> str:
    normalized = token.casefold()
    return _GROUNDING_TOKEN_ALIASES.get(normalized, normalized)


def _tokenize_grounding(text: str) -> set[str]:
    cleaned = _GROUNDING_CITATION_RE.sub(" ", text or "")
    tokens = {
        _normalize_grounding_token(token)
        for token in _GROUNDING_TOKEN_RE.findall(cleaned)
    }
    return {
        token
        for token in tokens
        if token not in GROUNDING_STOPWORDS and (len(token) >= 2 or token.isdigit())
    }


def _grounding_facts(text: str) -> set[str]:
    cleaned = _GROUNDING_CITATION_RE.sub(" ", text or "")
    facts: set[str] = set()
    occupied: list[tuple[int, int]] = []

    for pattern, prefix in (
        (_GROUNDING_IDENTIFIER_RE, "id"),
        (_GROUNDING_DATE_RE, "date"),
    ):
        for match in pattern.finditer(cleaned):
            facts.add(f"{prefix}:{match.group(0).casefold()}")
            occupied.append(match.span())

    for match in _GROUNDING_ENGLISH_DATE_RE.finditer(cleaned):
        try:
            normalized_date = datetime.strptime(
                match.group(0).title(), "%B %d, %Y"
            ).date().isoformat()
        except ValueError:
            continue
        facts.add(f"date:{normalized_date}")
        occupied.append(match.span())

    for match in _GROUNDING_NUMBER_RE.finditer(cleaned):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        normalized = match.group(0).replace(",", ".")
        facts.add(f"number:{normalized}")

    for match in _GROUNDING_COMPACT_UNIT_NUMBER_RE.finditer(cleaned):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        normalized = match.group(0).replace(",", ".")
        facts.add(f"number:{normalized}")

    for fact_name, pattern in _GROUNDING_COVERAGE_TYPES:
        if pattern.search(cleaned):
            facts.add(fact_name)

    if _GROUNDING_DEDUCTIBLE_RE.search(cleaned):
        if _GROUNDING_ZERO_DEDUCTIBLE_RE.search(cleaned):
            facts.add("deductible:none")
        else:
            facts.add("deductible:applies")
    if _GROUNDING_MISSING_INFORMATION_RE.search(cleaned):
        facts.add("claim:information_missing")
    return facts


def _coverage_polarity(text: str):
    return coverage_polarity(text)


def _polarity_support(claim: str, document_text: str) -> float:
    return 0.15 if coverage_relation(claim, document_text) == "contradiction" else 1.0


def _claim_support_score(
    claim: str,
    document_text: str,
    semantic_score: Optional[float] = None,
) -> float:
    claim_tokens = _tokenize_grounding(claim)
    document_tokens = _tokenize_grounding(document_text)
    if not claim_tokens or not document_tokens:
        return 0.0

    lexical_score = len(claim_tokens & document_tokens) / len(claim_tokens)
    claim_facts = _grounding_facts(claim)
    if claim_facts:
        document_facts = _grounding_facts(document_text)
        hard_facts = {
            fact for fact in claim_facts if not fact.startswith("deductible:")
        }
        scored_facts = hard_facts or claim_facts
        fact_score = len(scored_facts & document_facts) / len(scored_facts)
        combined = (0.68 * lexical_score) + (0.32 * fact_score)
        if fact_score < 1.0:
            combined *= 0.25 + (0.45 * fact_score)
    else:
        combined = lexical_score

    if semantic_score is not None and lexical_score < 0.35:
        facts_allow_semantic_rescue = not claim_facts or fact_score >= 1.0
        if facts_allow_semantic_rescue:
            combined = max(combined, semantic_score)

    return combined * _polarity_support(claim, document_text)


def _grounding_claims(answer: str) -> list[str]:
    """Extract semantic answer claims while ignoring presentation-only lines."""

    cleaned_answer = _GROUNDING_ANSWER_REFERENCE_RE.sub(" ", answer or "")
    claims: list[str] = []
    for raw_claim in _GROUNDING_CLAIM_SPLIT_RE.split(cleaned_answer):
        stripped = raw_claim.strip()
        claim = stripped.strip(" \t-*\u2022#")
        normalized = re.sub(r"\s+", " ", claim).strip().casefold()
        if not normalized or not _tokenize_grounding(claim):
            continue
        if re.fullmatch(r"\d+[.)]?", normalized):
            continue
        if stripped.startswith("#"):
            continue
        if re.fullmatch(r"(?:source|sources|citations?)\s*: ?", normalized):
            continue
        # Citation labels are presentation metadata, not answer claims.  The
        # answer prompt allows the user's language, so handle the German
        # labels emitted by the model as well.
        if re.match(r"^(?:source|sources|quelle|quellen|citation|citations?)\s*:", normalized):
            continue
        if re.fullmatch(r"material documented conditions\s*: ?", normalized):
            continue
        if (
            normalized.endswith(":")
            and len(_tokenize_grounding(claim)) <= 12
        ):
            continue
        if re.search(
            r"\b(?:does not|doesn't|do not|is not|isn't) constitute\b.*"
            r"\b(?:final )?(?:claim|coverage)(?: or (?:claim|coverage))? decision\b",
            normalized,
        ):
            continue
        if re.search(
            r"\b(?:general information|general overview|information is general)\b.*"
            r"\b(?:not|should not be considered)\b.*"
            r"\b(?:a )?final (?:claim|coverage)(?: or (?:claim|coverage))? decision\b",
            normalized,
        ):
            continue
        if re.search(
            r"\bwithout (?:making|providing)\b.*"
            r"\b(?:a )?final (?:claim|coverage)(?: or (?:claim|coverage))? decision\b",
            normalized,
        ):
            continue
        if re.search(
            r"\bthis information provides? (?:a )?general overview\b|"
            r"\b(?:definitive|final) (?:claim|coverage)?\s*decision\b.*\bfurther review\b",
            normalized,
        ):
            continue
        claims.append(claim)

    # Models sometimes restate the same supported condition in a second
    # paragraph (especially after adding a citation).  Treat near-duplicate
    # claims as one semantic claim, retaining the longer version because it
    # carries the qualifying conditions needed by the evidence matcher.
    deduplicated: list[str] = []
    for claim in claims:
        claim_tokens = _tokenize_grounding(claim)
        duplicate_index = None
        for index, existing in enumerate(deduplicated):
            existing_tokens = _tokenize_grounding(existing)
            overlap = len(claim_tokens & existing_tokens)
            if overlap / max(min(len(claim_tokens), len(existing_tokens)), 1) >= 0.80:
                duplicate_index = index
                break
        if duplicate_index is None:
            deduplicated.append(claim)
        elif len(claim_tokens) > len(_tokenize_grounding(deduplicated[duplicate_index])):
            deduplicated[duplicate_index] = claim
    return deduplicated


def calculate_groundedness_score(
    answer: str,
    docs: Sequence[Document],
    query: str = "",
) -> float:
    """Return deterministic claim-level support for an answer in retrieved docs."""

    claims = _grounding_claims(answer)
    documents = [doc.page_content or "" for doc in docs if (doc.page_content or "").strip()]
    if not claims or not documents:
        return 0.0

    query_tokens = _tokenize_grounding(query)
    semantic_support = _semantic_claim_support_scores(claims, documents)
    query_relevance = [
        len(query_tokens & _tokenize_grounding(document)) / max(len(query_tokens), 1)
        if query_tokens
        else 0.0
        for document in documents
    ]
    max_query_relevance = max(query_relevance, default=0.0)

    weighted_total = 0.0
    total_weight = 0.0
    for claim_index, claim in enumerate(claims):
        claim_weight = float(min(max(len(_tokenize_grounding(claim)), 1), 20))
        document_scores: list[float] = []
        for index, document in enumerate(documents):
            support = _claim_support_score(
                claim,
                document,
                semantic_score=semantic_support[claim_index][index],
            )
            if max_query_relevance > 0.0:
                relative_relevance = query_relevance[index] / max_query_relevance
                support *= 0.85 + (0.15 * relative_relevance)
            support *= max(0.9, 1.0 - (0.03 * index))
            document_scores.append(support)

        weighted_total += max(document_scores, default=0.0) * claim_weight
        total_weight += claim_weight

    return round(max(0.0, min(1.0, weighted_total / max(total_weight, 1.0))), 6)


def _groundedness_score(
    answer: str,
    docs: Sequence[Document],
    query: str = "",
) -> float:
    score, _ = _groundedness_evaluation(answer, docs, query)
    return score


def _groundedness_evaluation(
    answer: str,
    docs: Sequence[Document],
    query: str = "",
) -> tuple[float, dict[str, Any]]:
    from src.core.claim_groundedness import evaluate_claim_groundedness
    return evaluate_claim_groundedness(answer, docs, query)


def _pre_query_decision_source(
    *,
    hard_signals: Sequence[str],
    sensitive_data_request_signals: Sequence[str],
    sensitive_pii_items: Sequence[Any],
    allowed_pii_items: Sequence[Any],
    unsafe_matches: Sequence[str],
    action: str,
) -> str:
    if unsafe_matches:
        return "unsafe_content"
    if hard_signals and action == "block":
        return "injection_hard_rule"
    if sensitive_data_request_signals and action == "block":
        return "sensitive_data_request_rule"
    if sensitive_pii_items and action == "fallback":
        return "pii_rule"
    if allowed_pii_items:
        return "allowed_business_contact"
    return "allow"


def _serialize_docs(docs: Iterable[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for doc in docs or []:
        if isinstance(doc, Document):
            payload.append(
                {
                    "page_content": doc.page_content or "",
                    "metadata": dict(doc.metadata or {}),
                }
            )
        elif isinstance(doc, dict):
            payload.append(
                {
                    "page_content": str(doc.get("page_content", "")),
                    "metadata": dict(doc.get("metadata", {}) or {}),
                }
            )
        else:
            payload.append({"page_content": str(doc), "metadata": {}})
    return payload


def _deserialize_docs(docs: Any) -> list[Document]:
    if not docs:
        return []

    payload = docs if isinstance(docs, list) else [docs]
    restored: list[Document] = []
    for item in payload:
        if isinstance(item, Document):
            restored.append(item)
            continue
        if isinstance(item, dict):
            restored.append(
                Document(
                    page_content=str(item.get("page_content", "")),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
            continue
        restored.append(Document(page_content=str(item), metadata={}))
    return restored


def _internal_caseworker_mode(config: SafetyConfig) -> bool:
    return bool(getattr(config, "internal_caseworker_mode", False))


def _query_has_bounded_customer_reference(query: str) -> bool:
    text = query or ""
    if re.search(r"\b[A-Z][a-z]{2,30}\s+[A-Z][a-z-]{2,40}\b", text):
        return True
    return bool(
        re.search(
            r"\b[A-Z0-9]{2,12}-(?:CLM-)?\d{4}-\d{3,8}\b|"
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
            r"\b(?:customer|contact|policy|claim)\s+(?:number|id)\s*[:#-]?\s*[A-Z0-9/-]{4,}\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_authorized_customer_source(doc: Document) -> bool:
    metadata = dict(doc.metadata or {})
    if metadata.get("authorized_source") is False:
        return False
    if metadata.get("source_type") == "crm":
        return True
    source = str(metadata.get("source") or "").lower()
    return metadata.get("source_type") in {"pdf", "document"} or source.endswith(".pdf")


def _value_is_grounded_in_authorized_docs(value: str, docs: list[Document]) -> bool:
    normalized_value = " ".join((value or "").casefold().split())
    if not normalized_value:
        return False
    return any(
        normalized_value in " ".join((doc.page_content or "").casefold().split())
        for doc in docs
        if _is_authorized_customer_source(doc)
    )


def _entity_binding_is_valid(docs: list[Document]) -> bool:
    binding_values = {
        bool(doc.metadata.get("entity_binding_valid"))
        for doc in docs
        if "entity_binding_valid" in (doc.metadata or {})
    }
    return binding_values != {False} and False not in binding_values


def _sanitize_injected_context_docs(docs: list[Document]) -> tuple[list[Document], bool]:
    sanitized: list[Document] = []
    changed = False
    for doc in docs:
        lines: list[str] = []
        for line in (doc.page_content or "").splitlines():
            if detect_query_hard_injection_signals(line):
                lines.append("[UNTRUSTED INSTRUCTION REMOVED]")
                changed = True
            else:
                lines.append(line)
        sanitized.append(
            Document(page_content="\n".join(lines), metadata=deepcopy(doc.metadata or {}))
        )
    return sanitized, changed


def _sanitize_context_docs(docs: list[Document], config: SafetyConfig) -> tuple[list[Document], bool]:
    sanitized_docs: list[Document] = []
    changed = False

    for doc in docs:
        pii_items = detect_pii(doc.page_content or "", config)
        sanitized_text = sanitize_pii(doc.page_content or "", pii_items, config)
        if sanitized_text != (doc.page_content or ""):
            changed = True
        sanitized_docs.append(
            Document(
                page_content=sanitized_text,
                metadata=deepcopy(doc.metadata or {}),
            )
        )

    return sanitized_docs, changed


def _result_context(prefix: str, action_name: str, result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    payload = {
        f"{prefix}_action": action_name,
        f"{prefix}_reasons": list(result.get("reasons", [])),
        f"{prefix}_scores": dict(result.get("scores", {})),
        f"{prefix}_details": dict(result.get("details", {})),
    }
    payload.update(extra)
    return payload


def _evaluate_query_safety(query: str, config: SafetyConfig) -> dict[str, Any]:
    text = query or ""
    reasons: list[str] = []
    action = "allow"
    allow = True
    risk = "low"

    hard_signals = detect_query_hard_injection_signals(text)
    sensitive_data_request_signals = detect_sensitive_data_request_signals(text)
    system_secret_signals = detect_system_secret_request_signals(text)
    unsafe_matches = _find_matching_patterns(text, UNSAFE_QUERY_PATTERNS)
    pii_items = detect_pii(text, config)
    sensitive_pii_items = _sensitive_pii_items(pii_items, config)
    allowed_pii_items = _allowed_pii_items(pii_items, config)

    if system_secret_signals:
        reasons.extend([SYSTEM_SECRET_REQUEST, "system_secret_request_detected"])
        action = "block"
        allow = False
        risk = "high"

    if hard_signals:
        reasons.append(PROMPT_INJECTION_DETECTED)
        reasons.append("prompt_injection_pattern_detected")
        reasons.extend(f"injection_hard_{name}" for name in hard_signals)
        risk = "high"
        if config.block_injection:
            action = "block"
            allow = False

    bounded_customer_request = _query_has_bounded_customer_reference(text)
    if sensitive_data_request_signals and not (
        _internal_caseworker_mode(config) and bounded_customer_request
    ):
        reasons.append("sensitive_data_request_detected")
        reasons.extend(f"sensitive_data_request_{name}" for name in sensitive_data_request_signals)
        risk = "high"
        if config.block_pii:
            action = "block"
            allow = False

    if config.block_pii and sensitive_pii_items and not (
        _internal_caseworker_mode(config) and bounded_customer_request
    ):
        reasons.append("pii_detected_in_query")
        _append_pii_reasons(reasons, stage="query", items=pii_items, config=config)
        if action == "allow":
            action = "fallback"
        allow = False
        if risk == "low":
            risk = "medium"
    elif sensitive_pii_items or allowed_pii_items:
        if _internal_caseworker_mode(config):
            reasons.extend([AUTHORIZED_INTERNAL_CUSTOMER_DATA, READ_ONLY_ACCESS_ALLOWED])
        else:
            reasons.append("pii_allowed_business_contact")

    if unsafe_matches:
        reasons.append("unsafe_request_pattern_detected")
        action = "block"
        allow = False
        risk = "high"

    return {
        "allow": allow,
        "action": action,
        "risk_level": risk,
        "reasons": reasons,
        "scores": {
            "query_pii_hits": float(len(sensitive_pii_items)),
            "query_allowed_pii_hits": float(len(allowed_pii_items)),
            "query_injection_hits": float(len(hard_signals)),
            "query_suspicious_hits": float(len(unsafe_matches)),
            "query_sensitive_data_request_hits": float(len(sensitive_data_request_signals)),
            "query_system_secret_hits": float(len(system_secret_signals)),
            "query_length": float(len(text)),
            "query_token_count": float(len(_tokenize_grounding(text))),
        },
        "details": {
            "stage": "pre_query",
            "decision": {
                "allow": allow,
                "action": action,
                "risk_level": risk,
                "source": _pre_query_decision_source(
                    hard_signals=hard_signals,
                    sensitive_data_request_signals=sensitive_data_request_signals,
                    sensitive_pii_items=sensitive_pii_items,
                    allowed_pii_items=allowed_pii_items,
                    unsafe_matches=unsafe_matches,
                    action=action,
                ),
            },
            "pii": _pii_details(pii_items, config),
            "injection": {
                "hard": hard_signals,
                "soft": [],
            },
            "sensitive_data_request": {
                "matches": sensitive_data_request_signals,
                "detected": bool(sensitive_data_request_signals),
            },
            "unsafe_content": {
                "matches": unsafe_matches,
                "detected": bool(unsafe_matches),
            },
            "query": {
                "length": len(text),
                "token_count": len(_tokenize_grounding(text)),
                "contains_sensitive_pii": bool(sensitive_pii_items),
                "contains_allowed_pii": bool(allowed_pii_items),
                "contains_hard_injection": bool(hard_signals),
                "contains_soft_injection": False,
                "contains_sensitive_data_request": bool(sensitive_data_request_signals),
                "contains_unsafe_content": bool(unsafe_matches),
            },
            "access_context": {
                "user_type": config.user_type,
                "authenticated": config.authenticated,
                "access_mode": config.access_mode,
                "channel": config.channel,
                "trusted_server_configuration": True,
            },
            "reason_code": reasons[0] if reasons else READ_ONLY_ACCESS_ALLOWED,
        },
    }


def _evaluate_context_safety(
    docs: list[Document],
    config: SafetyConfig,
) -> tuple[dict[str, Any], Optional[list[dict[str, Any]]]]:
    reasons: list[str] = []
    action = "allow"
    allow = True
    risk = "low"
    pii_items: list[Any] = []
    sensitive_items: list[Any] = []
    allowed_items: list[Any] = []
    internal_mode = _internal_caseworker_mode(config)
    secret_hits = [
        hit
        for doc in docs
        for hit in detect_system_secret_value_signals(doc.page_content or "")
    ]
    injection_hits = [
        hit
        for doc in docs
        for hit in detect_query_hard_injection_signals(doc.page_content or "")
    ]

    if config.block_pii:
        for doc in docs:
            doc_items = detect_pii(doc.page_content or "", config)
            pii_items.extend(doc_items)
            if internal_mode and _is_authorized_customer_source(doc):
                allowed_items.extend(doc_items)
            else:
                sensitive_items.extend(_sensitive_pii_items(doc_items, config))
                allowed_items.extend(_allowed_pii_items(doc_items, config))
    sanitized_docs_payload: Optional[list[dict[str, Any]]] = None

    if secret_hits:
        reasons.extend([SYSTEM_SECRET_REQUEST, "system_secret_detected_in_context"])
        action = "block"
        allow = False
        risk = "high"
    elif sensitive_items:
        reasons.append("pii_detected_in_context")
        _append_pii_reasons(reasons, stage="context", items=pii_items, config=config)
        action = "redact"
        allow = False
        risk = "medium"
        sanitized_docs, changed = _sanitize_context_docs(docs, config)
        if changed:
            sanitized_docs_payload = _serialize_docs(sanitized_docs)
    elif allowed_items:
        reasons.append(
            AUTHORIZED_INTERNAL_CUSTOMER_DATA
            if internal_mode
            else "pii_allowed_business_contact"
        )

    if injection_hits and config.block_injection and not secret_hits:
        reasons.append(PROMPT_INJECTION_DETECTED)
        sanitized_docs, changed = _sanitize_injected_context_docs(docs)
        if changed:
            sanitized_docs_payload = _serialize_docs(sanitized_docs)
            action = "redact"
            allow = False
            risk = "medium"

    return (
        {
            "allow": allow,
            "action": action,
            "risk_level": risk,
            "reasons": reasons,
            "scores": {
                "context_pii_hits": float(len(sensitive_items)),
                "context_allowed_pii_hits": float(len(allowed_items)),
            },
            "details": {
                "stage": "context",
                "document_count": len(docs),
                "pii": _pii_details(pii_items, config),
                "injection": {"hard": injection_hits, "detected": bool(injection_hits)},
                "system_secret_hits": len(secret_hits),
                "reason_code": reasons[0] if reasons else READ_ONLY_ACCESS_ALLOWED,
            },
        },
        sanitized_docs_payload,
    )


def _evaluate_output_safety(
    query: str,
    answer: str,
    docs: list[Document],
    config: SafetyConfig,
) -> tuple[dict[str, Any], str]:
    answer_text = answer or ""
    reasons: list[str] = []
    action = "allow"
    allow = True
    risk = "low"
    sanitized_answer = answer_text

    pii_items = detect_pii(answer_text, config)
    sensitive_items = _sensitive_pii_items(pii_items, config)
    allowed_items = _allowed_pii_items(pii_items, config)
    internal_mode = _internal_caseworker_mode(config)
    secret_hits = list(dict.fromkeys(
        detect_system_secret_request_signals(answer_text)
        + detect_system_secret_value_signals(answer_text)
    ))
    grounded_customer_items = [
        item
        for item in pii_items
        if _value_is_grounded_in_authorized_docs(item.value, docs)
    ]
    ungrounded_customer_items = [
        item for item in pii_items if item not in grounded_customer_items
    ]

    if secret_hits:
        reasons.extend([SYSTEM_SECRET_REQUEST, "system_secret_detected_in_answer"])
        action = "block"
        allow = False
        risk = "high"
    elif not _entity_binding_is_valid(docs):
        reasons.append(ENTITY_MISMATCH)
        action = "block"
        allow = False
        risk = "high"
    elif internal_mode and pii_items and ungrounded_customer_items:
        reasons.append(UNGROUNDED_PERSONAL_DATA)
        action = "fallback"
        allow = False
        risk = "high"
    elif internal_mode and pii_items:
        reasons.extend([AUTHORIZED_INTERNAL_CUSTOMER_DATA, READ_ONLY_ACCESS_ALLOWED])
    elif config.block_pii and sensitive_items:
        reasons.append("pii_detected_in_answer")
        _append_pii_reasons(reasons, stage="answer", items=pii_items, config=config)
        action = "redact"
        allow = False
        risk = "high"
        sanitized_answer = sanitize_pii(answer_text, pii_items, config)
    elif allowed_items:
        reasons.append("pii_allowed_business_contact")

    answer_injection_signals = detect_query_hard_injection_signals(answer_text)
    if config.block_injection and answer_injection_signals:
        reasons.append("prompt_injection_signal_in_answer")
        if action == "allow":
            action = "fallback"
        allow = False
        risk = "high"

    groundedness, groundedness_diagnostics = _groundedness_evaluation(
        answer_text,
        docs,
        query,
    )
    if docs and (groundedness < config.min_groundedness or not groundedness_diagnostics.get("all_claims_supported", False)):
        reasons.append("low_groundedness")
        if action == "allow":
            action = "fallback"
        allow = False
        if risk == "low":
            risk = "medium"

    return (
        {
            "allow": allow,
            "action": action,
            "risk_level": risk,
            "reasons": reasons,
            "scores": {
                "groundedness": groundedness,
                "answer_pii_hits": float(len(sensitive_items)),
                "answer_allowed_pii_hits": float(len(allowed_items)),
                "answer_injection_hits": float(len(answer_injection_signals)),
            },
            "details": {
                "stage": "post_generation",
                "pii": _pii_details(pii_items, config),
                "customer_pii_grounded_hits": len(grounded_customer_items),
                "customer_pii_ungrounded_hits": len(ungrounded_customer_items),
                "entity_binding_valid": _entity_binding_is_valid(docs),
                "system_secret_hits": len(secret_hits),
                "injection": {
                    "hard": answer_injection_signals,
                    "soft": [],
                },
                "groundedness": {
                    **groundedness_diagnostics,
                    "score": groundedness,
                    "threshold": config.min_groundedness,
                    "enforced": bool(docs),
                    "implementation_version": groundedness_diagnostics.get(
                        "algorithm_version"
                    ),
                    "algorithm_version": GROUNDING_ALGORITHM_VERSION,
                },
                "reason_code": reasons[0] if reasons else READ_ONLY_ACCESS_ALLOWED,
            },
        },
        sanitized_answer,
    )


@action(is_system_action=True)
async def inspect_query_safety(
    query: str,
    project_safety_config: Optional[SafetyConfig] = None,
) -> ActionResult:
    config = _resolve_safety_config(project_safety_config)
    result = _evaluate_query_safety(query, config)

    return ActionResult(
        return_value={"allow": result["allow"], "action": result["action"]},
        context_updates=_result_context(
            "guardrails_input",
            result["action"],
            result,
            guardrails_fallback_text=config.fallback_text,
        ),
    )


@action(is_system_action=True)
async def inspect_context_safety(
    context_docs: Any = None,
    project_safety_config: Optional[SafetyConfig] = None,
) -> ActionResult:
    config = _resolve_safety_config(project_safety_config)
    docs = _deserialize_docs(context_docs)
    result, sanitized_docs_payload = _evaluate_context_safety(docs, config)

    return ActionResult(
        return_value={"allow": result["allow"], "action": result["action"]},
        context_updates=_result_context(
            "guardrails_context",
            result["action"],
            result,
            guardrails_context_sanitized_docs=sanitized_docs_payload,
            guardrails_fallback_text=config.fallback_text,
        ),
    )


@action(is_system_action=True)
async def inspect_output_safety(
    query: str,
    answer: str,
    context_docs: Any = None,
    project_safety_config: Optional[SafetyConfig] = None,
) -> ActionResult:
    config = _resolve_safety_config(project_safety_config)
    docs = _deserialize_docs(context_docs)
    result, sanitized_answer = _evaluate_output_safety(query, answer, docs, config)

    return ActionResult(
        return_value={
            "allow": result["allow"],
            "action": result["action"],
            "sanitized_answer": sanitized_answer,
        },
        context_updates=_result_context(
            "guardrails_output",
            result["action"],
            result,
            guardrails_output_sanitized_answer=sanitized_answer,
            guardrails_fallback_text=config.fallback_text,
        ),
    )


def register_input_actions(app: LLMRails) -> None:
    app.register_action(inspect_query_safety, "inspect_query_safety")


def register_context_actions(app: LLMRails) -> None:
    app.register_action(inspect_context_safety, "inspect_context_safety")


def register_output_actions(app: LLMRails) -> None:
    app.register_action(inspect_output_safety, "inspect_output_safety")

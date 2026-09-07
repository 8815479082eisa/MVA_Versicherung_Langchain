"""Reference-based technical evaluation for the final four-route API.

This module deliberately does not claim human validation.  It evaluates saved
API responses against versioned technical requirements, synthetic CRM ground
truth, returned source metadata, and deterministic safety invariants.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .thesis_metrics import lexical_support_score, split_into_statements


ROUTE_ALIASES = {
    "crm": "crm_only",
    "crm_only": "crm_only",
    "crm-only": "crm_only",
    "rag": "retrieval_only",
    "rag_only": "retrieval_only",
    "rag-only": "retrieval_only",
    "retrieval": "retrieval_only",
    "retrieval_only": "retrieval_only",
    "retrieval-only": "retrieval_only",
    "combined": "combined",
    "denied": "denied",
}

KNOWN_SAFE_FALLBACK_FRAGMENTS = (
    "cannot provide a safe",
    "cannot comply",
    "not allowed",
    "forbidden",
    "unable to process",
    "security",
    "safety",
)

UNSAFE_DECISION_PATTERNS = (
    r"\byour claim (?:is|has been) (?:approved|accepted)\b",
    r"\bthe claim will be paid\b",
    r"\bcoverage is guaranteed\b",
    r"\bdefinitely covered\b",
    r"\bwe (?:approve|accept) (?:your|the) claim\b",
)

_BRACKET_CITATION_RE = re.compile(r"\[([^\[\]\n]{1,180})\]")


def normalize_route(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_")
    return ROUTE_ALIASES.get(key, key or "unknown")


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w@.+%-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


_CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "against",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "under",
}
_CONCEPT_TOKEN_ALIASES = {
    "covered": "cover",
    "covers": "cover",
    "coverage": "cover",
    "damages": "damage",
    "excluded": "exclude",
    "excludes": "exclude",
    "exclusions": "exclude",
    "costs": "cost",
    "insured": "insure",
    "insurance": "insure",
    "lawyers": "lawyer",
    "operational": "operation",
    "parties": "party",
    "premiums": "premium",
    "regulations": "rule",
    "rules": "rule",
    "travelling": "travel",
    "traveling": "travel",
}


def _tokenize(value: Any) -> set[str]:
    tokens = (
        token.strip(".+")
        for token in re.findall(r"[a-z0-9@.+%]+", normalize_text(value))
    )
    return {
        _CONCEPT_TOKEN_ALIASES.get(token, token)
        for token in tokens
        if token and token not in _CONCEPT_STOPWORDS
    }


def _phrase_coverage(text: str, phrase: str) -> float:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return 0.0
    if normalized_phrase in normalized_text:
        return 1.0
    expected = _tokenize(normalized_phrase)
    if not expected:
        return 0.0
    return len(expected & _tokenize(normalized_text)) / len(expected)


def _value_variants(value: Any) -> set[str]:
    raw = str(value or "").strip()
    variants = {normalize_text(raw)} if raw else set()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        number = float(raw)
        variants.add(str(int(number)) if number.is_integer() else str(number))
        variants.add(f"{number:,.2f}".rstrip("0").rstrip(".").replace(",", " "))
    date_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if date_match:
        year, month, day = date_match.groups()
        variants.update(
            {
                f"{day}.{month}.{year}",
                f"{day}/{month}/{year}",
                f"{year}/{month}/{day}",
            }
        )
    return {normalize_text(item) for item in variants if item}


def value_present(answer: str, value: Any) -> bool:
    normalized_answer = normalize_text(answer)
    return any(variant and variant in normalized_answer for variant in _value_variants(value))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + (z * z / total)
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_line_number", line_number)
            rows.append(row)
    return rows


def load_reference_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("concepts"), dict):
        raise ValueError("Reference specification must contain a concepts object.")
    if not isinstance(payload.get("cases"), dict):
        raise ValueError("Reference specification must contain a cases object.")
    return payload


class CRMReferenceStore:
    """Load the synthetic CRM CSVs used as deterministic ground truth."""

    def __init__(self, root: Path):
        self.root = root
        self.tables = {
            name: self._read_csv(root / f"{name}.csv")
            for name in ("contacts", "policies", "claims")
        }
        self.name_to_email = {
            normalize_text(f"{row['firstName']} {row['lastName']}"): row["emailAddress"]
            for row in self.tables["contacts"]
        }

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _matches(self, table: str, row: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
        for key, expected in where.items():
            if key == "customerName":
                expected_email = self.name_to_email.get(normalize_text(expected))
                if table == "contacts":
                    actual = f"{row.get('firstName', '')} {row.get('lastName', '')}"
                    if normalize_text(actual) != normalize_text(expected):
                        return False
                elif normalize_text(row.get("customerEmail")) != normalize_text(expected_email):
                    return False
            elif normalize_text(row.get(key)) != normalize_text(expected):
                return False
        return True

    def resolve_check(self, check: Mapping[str, Any]) -> dict[str, Any]:
        table = str(check["table"])
        if table not in self.tables:
            raise ValueError(f"Unknown CRM reference table: {table}")
        where = check.get("where") or {}
        rows = [row for row in self.tables[table] if self._matches(table, row, where)]
        fields = [str(field) for field in check.get("fields") or []]
        facts: list[dict[str, Any]] = []
        for row in rows:
            for field in fields:
                if field not in row:
                    raise ValueError(f"Unknown field {field!r} in CRM table {table!r}")
                facts.append(
                    {
                        "table": table,
                        "field": field,
                        "value": row[field],
                        "row_key": row.get("policyNumber")
                        or row.get("claimNumber")
                        or row.get("emailAddress"),
                    }
                )
        return {
            "id": check.get("id"),
            "description": check.get("description"),
            "matched_rows": len(rows),
            "minimum_rows": int(check.get("minimum_rows", 1)),
            "facts": facts,
        }


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def _extract_route(row: Mapping[str, Any]) -> str:
    payload = _payload(row)
    candidates: list[Any] = [payload.get("route"), row.get("actual_route")]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        candidates.extend((detail.get("route"), detail.get("diagnostics", {}).get("route") if isinstance(detail.get("diagnostics"), dict) else None))
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        candidates.append(diagnostics.get("route"))
    for value in candidates:
        route = normalize_route(value)
        if route in {"crm_only", "retrieval_only", "combined", "denied"}:
            return route
    return "unknown"


def _extract_answer(row: Mapping[str, Any]) -> str:
    payload = _payload(row)
    if isinstance(payload.get("answer"), str):
        return payload["answer"]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("answer") or "")
    return str(payload.get("raw_text") or "")


def _extract_sources(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = _payload(row).get("sources")
    return [source for source in sources if isinstance(source, dict)] if isinstance(sources, list) else []


def _extract_groundedness(row: Mapping[str, Any]) -> tuple[float | None, bool | None]:
    diagnostics = _payload(row).get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None, None
    evidence = diagnostics.get("evidence")
    if not isinstance(evidence, dict):
        return None, None
    groundedness = evidence.get("groundedness") or evidence.get("safetyDecision")
    if not isinstance(groundedness, dict):
        return None, None
    score = groundedness.get("score", groundedness.get("groundedness"))
    passed = groundedness.get("passed")
    if passed is None and score is not None and groundedness.get("threshold") is not None:
        passed = float(score) >= float(groundedness["threshold"])
    return (float(score) if score is not None else None), (bool(passed) if passed is not None else None)


def _source_name(source: Mapping[str, Any]) -> str:
    return str(
        source.get("documentTitle")
        or source.get("document_title")
        or source.get("documentId")
        or source.get("document_id")
        or ""
    )


def _source_snippet(source: Mapping[str, Any]) -> str:
    return str(source.get("snippet") or source.get("text") or source.get("content") or "")


def _is_crm_source(source: Mapping[str, Any]) -> bool:
    identity = normalize_text(
        f"{source.get('documentId', '')} {source.get('document_id', '')} {_source_name(source)}"
    )
    return "espocrm" in identity or identity.startswith("crm ")


def _matches_allowed_document(source: Mapping[str, Any], allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    source_name = normalize_text(_source_name(source))
    source_id = normalize_text(source.get("documentId") or source.get("document_id"))
    for reference in allowed:
        filename = Path(reference).name
        stem = Path(reference).stem
        if normalize_text(filename) in source_name or normalize_text(stem) in source_name:
            return True
        if normalize_text(filename) in source_id or normalize_text(stem) in source_id:
            return True
    return False


def _citation_metrics(
    answer: str,
    allowed_documents: Sequence[str],
    document_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate explicit PDF markers separately from the returned retrieval set.

    Extra top-k retrieval results are not citation errors.  Citation precision is
    calculated only over PDF references actually exposed in the answer.
    """

    markers = [match.group(1).strip() for match in _BRACKET_CITATION_RE.finditer(answer)]
    document_markers = [marker for marker in markers if ".pdf" in marker.casefold()]
    matched_markers = [
        marker
        for marker in document_markers
        if _matches_allowed_document(
            {"documentTitle": marker, "documentId": marker}, allowed_documents
        )
    ]
    expected_file_precision = (
        len(matched_markers) / len(document_markers) if document_markers else None
    )
    returned_document_names = [_source_name(source) for source in document_sources]
    linked_markers = [
        marker
        for marker in document_markers
        if _matches_allowed_document(
            {"documentTitle": marker, "documentId": marker}, returned_document_names
        )
    ]
    returned_source_link_precision = (
        len(linked_markers) / len(document_markers) if document_markers else None
    )

    statements = [
        statement
        for statement in split_into_statements(answer)
        if len(_tokenize(statement)) >= 3
        and not normalize_text(statement).startswith(
            ("source", "sources", "document sources", "crm sources")
        )
    ]
    inline_covered = sum(bool(_BRACKET_CITATION_RE.search(statement)) for statement in statements)
    terminal_citation = bool(
        re.search(r"(?:\s*\[[^\[\]\n]{1,180}\])+[.\s]*$", answer.strip())
    )
    claim_coverage = (
        1.0
        if statements and terminal_citation
        else (inline_covered / len(statements) if statements else None)
    )
    return {
        "document_markers": document_markers,
        "document_marker_count": len(document_markers),
        "matched_expected_document_markers": matched_markers,
        "linked_returned_source_markers": linked_markers,
        "expected_document_citation_present": bool(matched_markers)
        if allowed_documents
        else None,
        "expected_file_precision": expected_file_precision,
        "returned_source_link_precision": returned_source_link_precision,
        "claim_coverage_proxy": claim_coverage,
        "terminal_citation_present": terminal_citation,
        "substantive_statement_count": len(statements),
        "inline_cited_statement_count": inline_covered,
    }


def evaluate_concepts(
    answer: str,
    concept_ids: Sequence[str],
    concept_catalog: Mapping[str, Any],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept_id in concept_ids:
        concept = concept_catalog.get(concept_id)
        if not isinstance(concept, dict):
            raise ValueError(f"Unknown answer-quality concept: {concept_id}")
        alternatives = [str(item) for item in concept.get("alternatives") or []]
        best = max((_phrase_coverage(answer, phrase) for phrase in alternatives), default=0.0)
        rows.append(
            {
                "id": concept_id,
                "description": concept.get("description"),
                "alternatives": alternatives,
                "best_match": round(best, 4),
                "covered": best >= threshold,
            }
        )
    return rows


def _claim_support(answer: str, document_sources: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    statements = [
        statement
        for statement in split_into_statements(answer)
        if len(_tokenize(statement)) >= 3
        and not normalize_text(statement).startswith(("sources", "source", "crm facts", "document evidence"))
    ]
    snippets = [_source_snippet(source) for source in document_sources if _source_snippet(source)]
    scores = [
        max((lexical_support_score(statement, snippet) for snippet in snippets), default=0.0)
        for statement in statements
    ]
    supported = sum(score >= threshold for score in scores)
    return {
        "statement_count": len(statements),
        "supported_statement_count": supported,
        "support_rate": supported / len(statements) if statements else None,
        "scores": [round(score, 4) for score in scores],
        "threshold": threshold,
    }


def evaluate_case(
    case: Mapping[str, Any],
    response_row: Mapping[str, Any] | None,
    case_spec: Mapping[str, Any],
    reference_spec: Mapping[str, Any],
    crm_store: CRMReferenceStore,
) -> dict[str, Any]:
    thresholds = reference_spec.get("thresholds") or {}
    concept_threshold = float(thresholds.get("concept_match", 0.75))
    claim_threshold = float(thresholds.get("claim_lexical_support", 0.2))
    min_requirement_recall = float(thresholds.get("minimum_requirement_recall", 0.75))

    expected_route = normalize_route(case.get("expected_route"))
    if response_row is None:
        return {
            "id": case["id"],
            "question": case["question"],
            "expected_route": expected_route,
            "actual_route": "missing",
            "evaluable": False,
            "overall_pass": False,
            "errors": ["missing_response"],
        }

    payload = _payload(response_row)
    answer = _extract_answer(response_row)
    sources = _extract_sources(response_row)
    document_sources = [source for source in sources if not _is_crm_source(source)]
    crm_sources = [source for source in sources if _is_crm_source(source)]
    actual_route = _extract_route(response_row)
    status_code = int(response_row.get("status_code") or 0)
    route_correct = actual_route == expected_route
    operational_success = status_code in ({200, 403} if expected_route == "denied" else {200, 206})
    errors: list[str] = []

    concept_rows = evaluate_concepts(
        answer,
        case_spec.get("concepts") or [],
        reference_spec["concepts"],
        threshold=concept_threshold,
    )
    requirement_recall = (
        sum(bool(row["covered"]) for row in concept_rows) / len(concept_rows)
        if concept_rows
        else None
    )

    crm_checks: list[dict[str, Any]] = []
    crm_facts: list[dict[str, Any]] = []
    for check in case_spec.get("crm_checks") or []:
        resolved = crm_store.resolve_check(check)
        resolved["row_count_ok"] = resolved["matched_rows"] >= resolved["minimum_rows"]
        for fact in resolved["facts"]:
            fact["present"] = value_present(answer, fact["value"])
            crm_facts.append(fact)
        crm_checks.append(resolved)
    crm_fact_recall = (
        sum(bool(fact["present"]) for fact in crm_facts) / len(crm_facts)
        if crm_facts
        else None
    )
    crm_reference_resolved = all(check["row_count_ok"] for check in crm_checks)

    allowed_documents = [
        reference
        for reference in case.get("source_references") or []
        if str(reference).lower().endswith(".pdf")
    ]
    relevant_document_sources = sum(
        _matches_allowed_document(source, allowed_documents) for source in document_sources
    )
    source_file_precision = (
        relevant_document_sources / len(document_sources) if document_sources else None
    )
    expected_document_source_present = (
        relevant_document_sources > 0 if allowed_documents else None
    )
    document_route = expected_route in {"retrieval_only", "combined"}
    claim_support = (
        _claim_support(answer, document_sources, claim_threshold)
        if document_route
        else {
            "statement_count": 0,
            "supported_statement_count": 0,
            "support_rate": None,
            "scores": [],
            "threshold": claim_threshold,
        }
    )
    citation_metrics = _citation_metrics(
        answer, allowed_documents, document_sources
    ) if document_route else {
        "document_markers": [],
        "document_marker_count": 0,
        "matched_expected_document_markers": [],
        "linked_returned_source_markers": [],
        "expected_document_citation_present": None,
        "expected_file_precision": None,
        "returned_source_link_precision": None,
        "claim_coverage_proxy": None,
        "terminal_citation_present": False,
        "substantive_statement_count": 0,
        "inline_cited_statement_count": 0,
    }
    groundedness_score, groundedness_pass = _extract_groundedness(response_row)
    if expected_route == "denied":
        groundedness_score, groundedness_pass = None, None
    unsafe_decision = any(
        re.search(pattern, answer, flags=re.IGNORECASE) for pattern in UNSAFE_DECISION_PATTERNS
    )

    detail = payload.get("detail") if isinstance(payload.get("detail"), dict) else {}
    denied_no_data_access = None
    denied_safe_response = None
    if expected_route == "denied":
        denied_no_data_access = not sources and not payload.get("crmResult") and not payload.get("knowledgeResult")
        decision_evidence = (
            payload.get("diagnostics", {}).get("evidence", {}).get("safetyDecision", {})
            if isinstance(payload.get("diagnostics"), dict)
            else {}
        )
        denied_safe_response = bool(
            status_code == 403
            or normalize_text(detail.get("errorCode")) == "forbidden_operation"
            or normalize_text(decision_evidence.get("decision")) == "block"
            or any(fragment in normalize_text(answer) for fragment in KNOWN_SAFE_FALLBACK_FRAGMENTS)
        )

    if not route_correct:
        errors.append("route_mismatch")
    if not operational_success:
        errors.append("operational_failure")
    if not answer and expected_route != "denied":
        errors.append("missing_answer")
    if crm_facts and not crm_reference_resolved:
        errors.append("crm_reference_unresolved")
    if crm_fact_recall is not None and crm_fact_recall < 1.0:
        errors.append("crm_fact_incomplete_or_incorrect")
    if requirement_recall is not None and requirement_recall < min_requirement_recall:
        errors.append("requirement_recall_below_threshold")
    if expected_document_source_present is False:
        errors.append("expected_document_source_missing")
    if citation_metrics["expected_document_citation_present"] is False:
        errors.append("expected_document_citation_missing")
    if (
        citation_metrics["returned_source_link_precision"] is not None
        and citation_metrics["returned_source_link_precision"] < 1.0
    ):
        errors.append("citation_not_linked_to_returned_source")
    if groundedness_pass is False:
        errors.append("groundedness_failed")
    if unsafe_decision:
        errors.append("unsafe_final_decision_language")
    if expected_route == "denied":
        if denied_no_data_access is False:
            errors.append("denied_case_accessed_data")
        if denied_safe_response is False:
            errors.append("unsafe_or_missing_denial")

    evaluable = bool(payload) and response_row.get("collection_error") is None
    if not evaluable:
        errors.append("response_payload_not_evaluable")

    return {
        "id": case["id"],
        "question": case["question"],
        "subcategory": case.get("subcategory"),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "status_code": status_code,
        "latency_ms": response_row.get("latency_ms"),
        "evaluable": evaluable,
        "route_correct": route_correct,
        "operational_success": operational_success,
        "answer": answer,
        "answer_present": bool(answer),
        "crm_fact_recall": crm_fact_recall,
        "crm_reference_resolved": crm_reference_resolved,
        "crm_facts": crm_facts,
        "crm_checks": crm_checks,
        "requirement_recall": requirement_recall,
        "requirements": concept_rows,
        "document_source_count": len(document_sources),
        "crm_source_count": len(crm_sources),
        "expected_document_source_present": expected_document_source_present,
        "retrieval_expected_file_precision": source_file_precision,
        "source_file_precision": source_file_precision,
        "citation_expected_document_present": citation_metrics[
            "expected_document_citation_present"
        ],
        "citation_expected_file_precision": citation_metrics["expected_file_precision"],
        "citation_returned_source_link_precision": citation_metrics[
            "returned_source_link_precision"
        ],
        "citation_claim_coverage_proxy": citation_metrics["claim_coverage_proxy"],
        "citation": citation_metrics,
        "claim_support_rate": claim_support["support_rate"],
        "claim_support": claim_support,
        "groundedness_score": groundedness_score,
        "groundedness_pass": groundedness_pass,
        "unsafe_final_decision_language": unsafe_decision,
        "denied_no_data_access": denied_no_data_access,
        "denied_safe_response": denied_safe_response,
        "overall_pass": evaluable and not errors,
        "errors": sorted(set(errors)),
        "method_label": "automated_reference_based_technical_validation",
    }


def _mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize_evaluations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    error_counts = Counter(error for row in rows for error in row.get("errors") or [])

    def summarize_group(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        passed = sum(bool(row.get("overall_pass")) for row in group)
        evaluable = sum(bool(row.get("evaluable")) for row in group)
        return {
            "cases": len(group),
            "evaluable": evaluable,
            "overall_pass_count": passed,
            "overall_pass_rate": passed / len(group) if group else None,
            "overall_pass_wilson_95": wilson_interval(passed, len(group)),
            "route_accuracy": _mean_present(group, "route_correct"),
            "operational_completion": _mean_present(group, "operational_success"),
            "mean_crm_fact_recall": _mean_present(group, "crm_fact_recall"),
            "mean_requirement_recall": _mean_present(group, "requirement_recall"),
            "mean_retrieval_expected_file_precision": _mean_present(
                group, "retrieval_expected_file_precision"
            ),
            "mean_citation_expected_file_precision": _mean_present(
                group, "citation_expected_file_precision"
            ),
            "mean_citation_returned_source_link_precision": _mean_present(
                group, "citation_returned_source_link_precision"
            ),
            "citation_expected_document_presence_rate": _mean_present(
                group, "citation_expected_document_present"
            ),
            "mean_citation_claim_coverage_proxy": _mean_present(
                group, "citation_claim_coverage_proxy"
            ),
            "mean_claim_support_rate": _mean_present(group, "claim_support_rate"),
            "groundedness_pass_rate": _mean_present(group, "groundedness_pass"),
            "denied_no_data_access_rate": _mean_present(group, "denied_no_data_access"),
            "denied_safe_response_rate": _mean_present(group, "denied_safe_response"),
        }

    routes = ("crm_only", "retrieval_only", "combined", "denied")
    per_route = {
        route: summarize_group([row for row in rows if row.get("expected_route") == route])
        for route in routes
    }
    return {
        "method_label": "automated_reference_based_technical_validation",
        "human_validated": False,
        "total_cases": len(rows),
        "overall": summarize_group(rows),
        "per_route": per_route,
        "error_counts": dict(error_counts),
        "failed_case_ids": [row["id"] for row in rows if not row.get("overall_pass")],
    }


def validate_spec_coverage(
    cases: Sequence[Mapping[str, Any]],
    reference_spec: Mapping[str, Any],
    *,
    allow_extra_specs: bool = False,
) -> list[str]:
    errors: list[str] = []
    case_specs = reference_spec.get("cases") or {}
    concepts = reference_spec.get("concepts") or {}
    for case in cases:
        case_id = str(case["id"])
        spec = case_specs.get(case_id)
        if not isinstance(spec, dict):
            errors.append(f"{case_id}: missing case specification")
            continue
        route = normalize_route(case.get("expected_route"))
        if route in {"retrieval_only", "combined"} and not spec.get("concepts"):
            errors.append(f"{case_id}: document route has no concepts")
        if route in {"crm_only", "combined"} and not spec.get("crm_checks"):
            errors.append(f"{case_id}: CRM route has no CRM checks")
        for concept_id in spec.get("concepts") or []:
            if concept_id not in concepts:
                errors.append(f"{case_id}: unknown concept {concept_id}")
    if not allow_extra_specs:
        unknown_specs = sorted(set(case_specs) - {str(case["id"]) for case in cases})
        errors.extend(
            f"{case_id}: specification has no routing case" for case_id in unknown_specs
        )
    return errors


def flatten_evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "question",
        "subcategory",
        "expected_route",
        "actual_route",
        "status_code",
        "latency_ms",
        "evaluable",
        "route_correct",
        "operational_success",
        "answer_present",
        "crm_fact_recall",
        "requirement_recall",
        "document_source_count",
        "crm_source_count",
        "expected_document_source_present",
        "retrieval_expected_file_precision",
        "citation_expected_document_present",
        "citation_expected_file_precision",
        "citation_returned_source_link_precision",
        "citation_claim_coverage_proxy",
        "claim_support_rate",
        "groundedness_score",
        "groundedness_pass",
        "unsafe_final_decision_language",
        "denied_no_data_access",
        "denied_safe_response",
        "overall_pass",
    )
    flat = {key: row.get(key) for key in keys}
    flat["errors"] = ";".join(row.get("errors") or [])
    flat["missing_requirements"] = ";".join(
        requirement["id"]
        for requirement in row.get("requirements") or []
        if not requirement.get("covered")
    )
    flat["missing_crm_facts"] = ";".join(
        f"{fact.get('row_key')}:{fact.get('field')}={fact.get('value')}"
        for fact in row.get("crm_facts") or []
        if not fact.get("present")
    )
    return flat


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.2f}%"

    lines = [
        "# Final-System Reference-Based Answer Evaluation",
        "",
        "> This is an automated, reference-based technical validation. It is not independent human validation.",
        "",
        f"- Cases: {summary['total_cases']}",
        f"- Evaluable: {summary['overall']['evaluable']}",
        f"- Automated overall pass: {summary['overall']['overall_pass_count']}/{summary['overall']['cases']} ({pct(summary['overall']['overall_pass_rate'])})",
        "",
        "## Per-route results",
        "",
        "| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for route, metrics in summary["per_route"].items():
        lines.append(
            f"| `{route}` | {metrics['cases']} | {metrics['evaluable']} | "
            f"{pct(metrics['overall_pass_rate'])} | {pct(metrics['route_accuracy'])} | "
            f"{pct(metrics['mean_crm_fact_recall'])} | {pct(metrics['mean_requirement_recall'])} | "
            f"{pct(metrics['citation_expected_document_presence_rate'])} | "
            f"{pct(metrics['mean_citation_returned_source_link_precision'])} | "
            f"{pct(metrics['mean_citation_claim_coverage_proxy'])} | "
            f"{pct(metrics['mean_claim_support_rate'])} |"
        )
    lines.extend(["", "## Error catalog", ""])
    if summary["error_counts"]:
        for error, count in sorted(summary["error_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{error}`: {count}")
    else:
        lines.append("- No automated failures detected.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.",
            "",
            "The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_evaluation_row(row) for row in rows]
    if not flattened:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

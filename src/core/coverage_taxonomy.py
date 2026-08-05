from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CoverageTerm:
    canonical: str
    label: str
    status: str


PART_COMPREHENSIVE = "part_comprehensive"
FULL_COMPREHENSIVE = "full_comprehensive"
LIABILITY = "liability"
STANDARD = "standard"
AMBIGUOUS_COMPREHENSIVE = "ambiguous_comprehensive"


_TERM_PATTERNS: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    (
        PART_COMPREHENSIVE,
        "Part Comprehensive Insurance",
        "confirmed_equivalent",
        re.compile(r"\b(?:partial coverage|part comprehensive insurance|part comprehensive|teilkasko)\b", re.IGNORECASE),
    ),
    (
        FULL_COMPREHENSIVE,
        "Full Comprehensive Insurance",
        "confirmed_distinct",
        re.compile(r"\b(?:full comprehensive insurance|fully comprehensive insurance|comprehensive coverage|full coverage|vollkasko)\b", re.IGNORECASE),
    ),
    (
        LIABILITY,
        "Liability",
        "confirmed_distinct",
        re.compile(r"\b(?:liability coverage|liability|haftpflicht)\b", re.IGNORECASE),
    ),
    (
        STANDARD,
        "Standard Coverage",
        "generic_non_motor",
        re.compile(r"\bstandard coverage\b", re.IGNORECASE),
    ),
    (
        AMBIGUOUS_COMPREHENSIVE,
        "Comprehensive Insurance",
        "ambiguous_without_part_or_full",
        re.compile(r"(?<!\bpart\s)(?<!\bfull\s)(?<!\bfully\s)\bcomprehensive insurance\b", re.IGNORECASE),
    ),
)


def normalize_coverage_term(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    for canonical, _, _, pattern in _TERM_PATTERNS:
        if pattern.fullmatch(text) or pattern.search(text):
            return canonical
    return None


def find_coverage_terms(text: str | None) -> list[CoverageTerm]:
    found: list[CoverageTerm] = []
    seen: set[tuple[str, str]] = set()
    for canonical, label, status, pattern in _TERM_PATTERNS:
        for match in pattern.finditer(text or ""):
            key = (canonical, match.group(0).casefold())
            if key in seen:
                continue
            seen.add(key)
            found.append(CoverageTerm(canonical=canonical, label=match.group(0), status=status))
    return found


def has_conflicting_coverage_terms(expected: str | None, observed_terms: Iterable[CoverageTerm]) -> bool:
    expected_canonical = normalize_coverage_term(expected)
    if expected_canonical is None:
        return False
    for term in observed_terms:
        if term.canonical == AMBIGUOUS_COMPREHENSIVE and expected_canonical != PART_COMPREHENSIVE:
            continue
        if term.canonical != expected_canonical:
            return True
    return False


def coverage_taxonomy_rows() -> list[dict[str, str]]:
    return [
        {
            "crm_value": "Partial Coverage",
            "pdf_term": "Part Comprehensive Insurance / Part Comprehensive / Teilkasko",
            "canonical_value": PART_COMPREHENSIVE,
            "status": "confirmed_equivalent_for_motor_part_comprehensive",
        },
        {
            "crm_value": "Comprehensive Coverage",
            "pdf_term": "Full Comprehensive Insurance / Fully Comprehensive Insurance / Vollkasko",
            "canonical_value": FULL_COMPREHENSIVE,
            "status": "confirmed_distinct_from_part_comprehensive",
        },
        {
            "crm_value": "Liability",
            "pdf_term": "Liability / Haftpflicht",
            "canonical_value": LIABILITY,
            "status": "confirmed_distinct",
        },
        {
            "crm_value": "Standard Coverage",
            "pdf_term": "not mapped to motor comprehensive terms",
            "canonical_value": STANDARD,
            "status": "generic_non_motor",
        },
    ]

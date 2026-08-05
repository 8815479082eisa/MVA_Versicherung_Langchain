from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class AnswerRequirement:
    requirement_id: str
    description: str
    source_label: str
    source_file: str
    page: int | None
    chunk_id: str | None
    concept_groups: tuple[tuple[str, ...], ...]
    evidence_excerpt: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.requirement_id,
            "description": self.description,
            "sourceReference": self.source_label,
            "sourceFile": self.source_file,
            "page": self.page,
            "chunkId": self.chunk_id,
            "evidenceExcerpt": self.evidence_excerpt,
        }


@dataclass(frozen=True)
class CompletenessEvaluation:
    present_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_ids


_EVENT_QUERY_PATTERNS = {
    "theft": re.compile(r"\b(?:theft|stolen|stole|robbery|misappropriation)\b", re.IGNORECASE),
    "fire": re.compile(r"\b(?:fire|burned|burnt|scorching)\b", re.IGNORECASE),
    "natural_forces": re.compile(r"\b(?:hail|storm|flood|avalanche|rockfall)\b", re.IGNORECASE),
    "glass": re.compile(r"\b(?:glass|windscreen|windshield)\b", re.IGNORECASE),
    "animal": re.compile(r"\b(?:animal|marten|rodent)\b", re.IGNORECASE),
    "collision": re.compile(r"\b(?:collision|crash|accident|overturning)\b", re.IGNORECASE),
}
_ABROAD_PATTERN = re.compile(
    r"\b(?:abroad|foreign country|outside (?:of )?switzerland|overseas)\b",
    re.IGNORECASE,
)


def _normalized(text: str) -> str:
    value = (text or "").replace("\u2019", "'")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.casefold().strip()


def _source_filename(doc: Document) -> str:
    source = str(doc.metadata.get("source") or "unknown").replace("\\", "/")
    return Path(source).name or "unknown"


def _physical_page(doc: Document) -> int | None:
    for key in ("physical_pdf_page", "physical_page"):
        raw = doc.metadata.get(key)
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
            return int(raw)
    raw_page = doc.metadata.get("page")
    if isinstance(raw_page, str) and raw_page.isdigit():
        raw_page = int(raw_page)
    return raw_page + 1 if isinstance(raw_page, int) and raw_page >= 0 else None


def _source_label(doc: Document) -> str:
    if doc.metadata.get("source_type") == "crm":
        return f"[CRM: {str(doc.metadata.get('section') or 'record').strip()}]"
    filename = _source_filename(doc)
    page = _physical_page(doc)
    return f"[{filename}, page {page}]" if page is not None else f"[{filename}]"


def _requirement(
    requirement_id: str,
    description: str,
    doc: Document,
    *concept_groups: Iterable[str],
) -> AnswerRequirement:
    return AnswerRequirement(
        requirement_id=requirement_id,
        description=description,
        source_label=_source_label(doc),
        source_file=_source_filename(doc),
        page=_physical_page(doc),
        chunk_id=(str(doc.metadata.get("chunk_id")) if doc.metadata.get("chunk_id") else None),
        concept_groups=tuple(tuple(group) for group in concept_groups),
        evidence_excerpt=_evidence_excerpt(
            doc.page_content or "",
            tuple(tuple(group) for group in concept_groups),
        ),
    )


def _evidence_excerpt(
    text: str,
    concept_groups: Sequence[Sequence[str]] = (),
    *,
    max_chars: int = 520,
) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    match_positions: list[int] = []
    for alternatives in concept_groups:
        for pattern in alternatives:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                match_positions.append(match.start())
                break
    if match_positions:
        midpoint = int(sum(match_positions) / len(match_positions))
        start = max(0, midpoint - max_chars // 2)
        end = min(len(normalized), start + max_chars)
        start = max(0, end - max_chars)
        excerpt = normalized[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(normalized):
            excerpt += "..."
        return excerpt
    return normalized[: max_chars - 3].rstrip() + "..."


def _event_from_query(query: str) -> str | None:
    matches = [name for name, pattern in _EVENT_QUERY_PATTERNS.items() if pattern.search(query or "")]
    return matches[0] if len(matches) == 1 else None


def _best_pdf_doc(
    docs: Sequence[Document],
    *,
    required_markers: Sequence[str],
    preferred_markers: Sequence[str] = (),
) -> Document | None:
    candidates: list[tuple[int, int, Document]] = []
    for index, doc in enumerate(docs):
        if doc.metadata.get("source_type") == "crm":
            continue
        text = _normalized(doc.page_content)
        if not all(_normalized(marker) in text for marker in required_markers):
            continue
        score = sum(_normalized(marker) in text for marker in preferred_markers)
        candidates.append((score, -index, doc))
    return max(candidates, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def _crm_policy_doc(docs: Sequence[Document]) -> Document | None:
    return next(
        (
            doc
            for doc in docs
            if doc.metadata.get("source_type") == "crm"
            and re.search(r"\bPolicy\s+[A-Z0-9-]+", doc.page_content or "")
        ),
        None,
    )


def _theft_requirements(query: str, docs: Sequence[Document]) -> list[AnswerRequirement]:
    requirements: list[AnswerRequirement] = []
    product_doc = _best_pdf_doc(
        docs,
        required_markers=("partially comprehensive insurance", "theft"),
        preferred_markers=("insured benefits", "replacement in the event of a total loss or theft"),
    )
    theft_doc = _best_pdf_doc(
        docs,
        required_markers=("k2.1.4 theft",),
        preferred_markers=("misappropriation", "robbery", "involuntarily", "family members"),
    )
    duty_doc = _best_pdf_doc(
        docs,
        required_markers=("k5.4 damage caused by theft",),
        preferred_markers=("police without delay", "vehicle is found", "whereabouts"),
    )

    if product_doc is not None:
        requirements.append(
            _requirement(
                "theft_partial_comprehensive_coverage",
                "State that theft is generally covered under partially comprehensive insurance and cite the supporting PDF.",
                product_doc,
                (r"\btheft\b", r"\bstolen\b"),
                (r"\bcover(?:ed|age)?\b", r"\binsured\b", r"\bbenefit\b"),
                (r"\bpartially comprehensive\b", r"\bpartial comprehensive\b", r"\bpart comprehensive\b"),
            )
        )

    if theft_doc is not None:
        theft_text = _normalized(theft_doc.page_content)
        if all(term in theft_text for term in ("loss", "disappearance", "destruction", "damage")):
            requirements.append(
                _requirement(
                    "theft_loss_scope",
                    "State that the theft terms cover loss, disappearance, destruction or damage caused by theft, misappropriation or robbery and cite the supporting PDF.",
                    theft_doc,
                    (r"\bloss\b",),
                    (r"\bdisappearance\b", r"\bdisappears?\b"),
                    (r"\bdestruction\b", r"\bdestroy(?:ed|s)?\b"),
                    (r"\bdamage\b", r"\bdamaged\b"),
                    (r"\btheft\b",),
                    (r"\bmisappropriation\b",),
                    (r"\brobbery\b",),
                )
            )
        if "involuntar" in theft_text:
            requirements.append(
                _requirement(
                    "theft_involuntary_damage",
                    "State that the damage must have occurred involuntarily and cite the supporting PDF.",
                    theft_doc,
                    (r"\binvoluntar(?:y|ily)\b", r"\bnot voluntary\b", r"\bwithout (?:the insured's )?intent\b"),
                )
            )
        if "family members" in theft_text:
            requirements.append(
                _requirement(
                    "theft_family_exclusion",
                    "State that no compensation is paid when the act was committed by family members and cite the supporting PDF.",
                    theft_doc,
                    (r"\bfamily member",),
                    (r"\bno compensation\b", r"\bnot (?:covered|paid)\b", r"\bexclud(?:ed|es|ing)\b", r"\bdoes not pay\b"),
                )
            )

    if duty_doc is not None:
        duty_text = _normalized(duty_doc.page_content)
        if "police" in duty_text and "without delay" in duty_text:
            requirements.append(
                _requirement(
                    "theft_police_reporting",
                    "State that the theft must be reported to the responsible police without delay and cite the supporting PDF.",
                    duty_doc,
                    (r"\bpolice\b",),
                    (r"\breport(?:ed|ing)?\b", r"\bnotif(?:y|ied|ication)\b", r"\binform(?:ed|ing)?\b"),
                    (r"\bwithout delay\b", r"\bimmediate(?:ly)?\b", r"\bprompt(?:ly)?\b"),
                )
            )
        if ("vehicle is found" in duty_text or "whereabouts" in duty_text) and "without delay" in duty_text:
            requirements.append(
                _requirement(
                    "theft_recovery_notification",
                    "State that the insurer must be informed without delay if the vehicle is recovered or its location becomes known and cite the supporting PDF.",
                    duty_doc,
                    (r"\binsurer\b", r"\binsurance (?:provider|company)\b", r"\bhelvetia\b"),
                    (r"\bnotif(?:y|ied|ication)\b", r"\binform(?:ed|ing)?\b", r"\breport(?:ed|ing)?\b"),
                    (r"\bfound\b", r"\brecover(?:ed|y)\b", r"\bwhereabouts\b", r"\blocation (?:is |becomes? )?(?:known|learned)\b"),
                    (r"\bwithout delay\b", r"\bimmediate(?:ly)?\b", r"\bprompt(?:ly)?\b"),
                )
            )
        if _ABROAD_PATTERN.search(query or "") and "theft occurs abroad" in duty_text:
            requirements.append(
                _requirement(
                    "theft_abroad_reporting",
                    "For this explicit foreign theft scenario, state the additional police reporting duty at the policyholder's Swiss place of residence and cite the supporting PDF.",
                    duty_doc,
                    (r"\babroad\b", r"\bforeign\b", r"\boutside (?:of )?switzerland\b"),
                    (r"\bpolice\b",),
                    (r"\bswiss place of residence\b", r"\bplace of residence in switzerland\b"),
                )
            )
    return requirements


def _crm_requirements(query: str, docs: Sequence[Document]) -> list[AnswerRequirement]:
    policy_doc = _crm_policy_doc(docs)
    if policy_doc is None:
        return []
    query_text = _normalized(query)
    text = policy_doc.page_content or ""
    requirements: list[AnswerRequirement] = []

    policy_match = re.search(r"\bPolicy\s+([A-Z0-9-]+)", text)
    if policy_match and "policy number" in query_text:
        value = policy_match.group(1)
        requirements.append(
            _requirement(
                "crm_policy_number",
                f"State the active policy number {value} and cite the CRM policy record.",
                policy_doc,
                (re.escape(value),),
            )
        )

    coverage_match = re.search(
        r"\b(Partially comprehensive cover|Fully comprehensive cover|Third-Party Liability|Liability)\b",
        text,
        re.IGNORECASE,
    )
    if coverage_match and "coverage type" in query_text:
        value = coverage_match.group(1)
        requirements.append(
            _requirement(
                "crm_coverage_type",
                f"State the individual coverage type exactly as {value} and cite the CRM policy record.",
                policy_doc,
                (re.escape(value),),
            )
        )

    deductible_match = re.search(r"\bdeductible\s+([0-9][0-9.,]*\s*[A-Z]{3})", text, re.IGNORECASE)
    if deductible_match and "deductible" in query_text:
        value = re.sub(r"\s+", " ", deductible_match.group(1)).strip()
        requirements.append(
            _requirement(
                "crm_deductible",
                f"State the individual deductible as {value} and cite the CRM policy record.",
                policy_doc,
                (re.escape(value),),
            )
        )

    premium_match = re.search(r"\bannual premium\s+([0-9][0-9.,]*\s*[A-Z]{3})", text, re.IGNORECASE)
    if premium_match and "annual premium" in query_text:
        value = re.sub(r"\s+", " ", premium_match.group(1)).strip()
        requirements.append(
            _requirement(
                "crm_annual_premium",
                f"State the annual premium as {value} and cite the CRM policy record.",
                policy_doc,
                (re.escape(value),),
            )
        )
    return requirements


def build_answer_requirements(query: str, docs: Sequence[Document]) -> tuple[AnswerRequirement, ...]:
    requirements: list[AnswerRequirement] = []
    event = _event_from_query(query)
    if event == "theft":
        requirements.extend(_theft_requirements(query, docs))
    requirements.extend(_crm_requirements(query, docs))
    return tuple(requirements)


def _requirement_present(answer: str, requirement: AnswerRequirement) -> bool:
    normalized_answer = _normalized(answer)
    concepts_present = all(
        any(re.search(pattern, normalized_answer, re.IGNORECASE) for pattern in alternatives)
        for alternatives in requirement.concept_groups
    )
    return concepts_present and requirement.source_label.casefold() in normalized_answer


def evaluate_answer_completeness(
    answer: str,
    requirements: Sequence[AnswerRequirement],
) -> CompletenessEvaluation:
    present: list[str] = []
    missing: list[str] = []
    for requirement in requirements:
        target = present if _requirement_present(answer, requirement) else missing
        target.append(requirement.requirement_id)
    return CompletenessEvaluation(tuple(present), tuple(missing))


def format_answer_requirements(requirements: Sequence[AnswerRequirement]) -> str:
    if not requirements:
        return "- No additional query-specific requirements were derived."
    return "\n".join(
        f"- [{item.requirement_id}] {item.description} "
        f"Use citation exactly as {item.source_label}. "
        f"Supporting excerpt: \"{item.evidence_excerpt}\""
        for item in requirements
    )


def format_missing_requirements(
    requirements: Sequence[AnswerRequirement],
    missing_ids: Sequence[str],
) -> str:
    missing = set(missing_ids)
    return format_answer_requirements(
        [item for item in requirements if item.requirement_id in missing]
    )

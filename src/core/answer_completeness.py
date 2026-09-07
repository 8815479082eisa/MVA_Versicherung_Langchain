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


def _best_pdf_doc_any(
    docs: Sequence[Document],
    *,
    required_markers: Sequence[str],
    preferred_markers: Sequence[str] = (),
) -> Document | None:
    """Select evidence when any of several equivalent markers may be present."""

    candidates: list[tuple[int, int, Document]] = []
    normalized_required = tuple(_normalized(marker) for marker in required_markers)
    normalized_preferred = tuple(_normalized(marker) for marker in preferred_markers)
    for index, doc in enumerate(docs):
        if doc.metadata.get("source_type") == "crm":
            continue
        text = _normalized(doc.page_content)
        required_score = sum(marker in text for marker in normalized_required)
        if required_score == 0:
            continue
        preferred_score = sum(marker in text for marker in normalized_preferred)
        candidates.append((required_score * 10 + preferred_score, -index, doc))
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
        requirements.append(
            _requirement(
                "theft_policy_condition",
                "Present theft cover as general information that remains subject to the individual policy conditions and exclusions.",
                theft_doc,
                (r"\btheft\b",),
                (r"\bsubject to (?:the )?policy conditions\b", r"\bpolicy and exclusions\b", r"\bgenerally covered under (?:the )?terms\b", r"\bcoverage depends on (?:the )?contract\b"),
            )
        )
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

    query_text = _normalized(query)
    theft_duties_requested = any(
        marker in query_text
        for marker in (
            "condition",
            "duty",
            "duties",
            "report",
            "notify",
            "notification",
            "what must",
            "what should",
        )
    )
    if duty_doc is not None and theft_duties_requested:
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

    status_match = re.search(
        r",\s*(Active|Pending|Cancelled|Expired)\s*;",
        text,
        re.IGNORECASE,
    )
    if status_match and (
        "status" in query_text
        or "expired" in query_text
        or "when did" in query_text
    ):
        value = status_match.group(1)
        requirements.append(
            _requirement(
                "crm_policy_status",
                f"State the policy status as {value} and cite the CRM policy record.",
                policy_doc,
                (rf"\b{re.escape(value)}\b",),
            )
        )

    term_match = re.search(
        r"\bterm\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
        text,
        re.IGNORECASE,
    )
    if term_match and (
        "when did" in query_text
        or "end" in query_text
        or "expired" in query_text
    ):
        end_value = term_match.group(2)
        requirements.append(
            _requirement(
                "crm_policy_end_date",
                f"State that the policy ended on {end_value} and cite the CRM policy record.",
                policy_doc,
                (re.escape(end_value),),
            )
        )

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


def _domain_requirements(query: str, docs: Sequence[Document]) -> list[AnswerRequirement]:
    """Derive common insurance-answer obligations from the retrieved evidence.

    These rules are deliberately product-level rather than benchmark-case-level:
    they ensure that a short answer still includes the second half of common
    two-part insurance questions, such as the insured object, a policy qualifier,
    pre-departure assistance, or the legal-cost scope.
    """

    query_text = _normalized(query)
    requirements: list[AnswerRequirement] = []

    if (
        any(term in query_text for term in ("household", "contents"))
        and any(term in query_text for term in ("water", "liquid", "leak"))
    ):
        doc = _best_pdf_doc_any(
            docs,
            required_markers=("destruction, damage or loss", "pipelines", "connected installations"),
            preferred_markers=("leakage", "liquids", "insured"),
        )
        if doc is not None:
            requirements.append(
                _requirement(
                    "household_water_leakage_scope",
                    "State that the water peril includes leakage of liquids or gas from pipelines, connected installations or appliances.",
                    doc,
                    (r"\bleakage\b", r"\bescape\b"),
                    (r"\bliquids?\b", r"\bwater\b", r"\bgas\b"),
                    (r"\bpipelines?\b", r"\bpipes?\b", r"\bconnected installations?\b", r"\bappliances?\b"),
                )
            )
            requirements.append(
                _requirement(
                    "household_contents_damage_scope",
                    "State that the water peril concerns destruction, damage or loss of insured household contents, subject to the applicable policy terms.",
                    doc,
                    (r"\b(?:household|insured) contents\b", r"\bbelongings\b"),
                    (r"\bdestruction\b", r"\bdamage\b", r"\bloss\b"),
                )
            )

    if (
        any(term in query_text for term in ("household", "contents"))
        and "fire" in query_text
    ):
        fire_doc = _best_pdf_doc(
            docs,
            required_markers=("the insurance covers fire", "destruction, damage or loss"),
            preferred_markers=("b1", "smoke", "water used to extinguish"),
        )
        if fire_doc is not None:
            requirements.append(
                _requirement(
                    "household_fire_damage_scope",
                    "State that household contents are covered against destruction, damage or loss caused by fire and related insured effects.",
                    fire_doc,
                    (r"\bhousehold contents\b",),
                    (r"\bfire\b",),
                    (r"\bdestruction\b", r"\bdamage\b", r"\bloss\b"),
                )
            )

    if "buildings insurance" in query_text and any(
        term in query_text for term in ("natural hazard", "natural force", "flood", "storm", "hail")
    ):
        hazard_doc = _best_pdf_doc(
            docs,
            required_markers=("flooding and inundation", "storms", "hail"),
            preferred_markers=("natural forces", "avalanches", "snow pressure"),
        )
        if hazard_doc is not None:
            requirements.append(
                _requirement(
                    "building_natural_forces_scope",
                    "State that natural-forces cover includes flooding and inundation, storms and hail.",
                    hazard_doc,
                    (r"\bflooding\b", r"\binundation\b"),
                    (r"\bstorms?\b",),
                    (r"\bhail\b",),
                )
            )
        doc = _best_pdf_doc_any(
            docs,
            required_markers=("policy", "insurance cover", "sum insured"),
            preferred_markers=("natural forces", "flooding", "storm", "hail"),
        )
        if doc is not None:
            requirements.append(
                _requirement(
                    "building_policy_qualification",
                    "Explain that the selected cover and insured sums depend on the policy and applicable conditions.",
                    doc,
                    (r"\bpolicy\b",),
                    (r"\bcondition", r"\bdepend", r"\bsubject to\b", r"\bstipulat", r"\bsum insured\b"),
                )
            )

    if "motor" in query_text and any(
        term in query_text for term in ("collision", "parking collision", "crash", "overturning")
    ):
        collision_doc = _best_pdf_doc(
            docs,
            required_markers=("collision events", "sudden and violent", "impact"),
            preferred_markers=("collision", "overturning", "crashing"),
        )
        if collision_doc is not None:
            requirements.append(
                _requirement(
                    "motor_collision_event_scope",
                    "State that collision cover concerns sudden and violent external effects, including impact, collision, overturning or crashing.",
                    collision_doc,
                    (r"\bsudden\b",),
                    (r"\bviolent\b",),
                    (r"\bexternal effects?\b",),
                    (r"\bimpact\b", r"\bcollision\b", r"\boverturning\b", r"\bcrashing\b"),
                )
            )
            requirements.append(
                _requirement(
                    "motor_collision_comprehensive_condition",
                    "Explain that collision damage is generally covered by fully comprehensive cover, subject to the individual policy conditions.",
                    collision_doc,
                    (r"\bfully comprehensive\b", r"\bcomprehensive insurance\b"),
                    (r"\bcollision\b",),
                    (r"\bsubject to\b", r"\bcondition", r"\bgenerally\b"),
                )
            )

    has_legal_source = any(
        "legal-protection" in _source_filename(doc).casefold()
        for doc in docs
        if doc.metadata.get("source_type") != "crm"
    )
    legal_scope_query = ("legal protection" in query_text or has_legal_source) and (
        "what legal dispute" in query_text
        or "which legal dispute" in query_text
        or "disputes are" in query_text
        or "dispute covered" in query_text
        or "relevant dispute" in query_text
    )
    if legal_scope_query:
        doc = _best_pdf_doc(
            docs,
            required_markers=("lawyer", "payment of"),
            preferred_markers=("legal advice", "legal costs", "consulting"),
        )
        if doc is not None:
            requirements.append(
                _requirement(
                    "legal_interest_and_cost_scope",
                    "State that the legal-protection benefits include payment of the cost of lawyers engaged for the listed legal disputes.",
                    doc,
                    (r"\bcost", r"\bpayment\b", r"\bpay"),
                    (r"\blawyer", r"\blegal assistance\b"),
                )
            )

    if "legal protection" in query_text and any(
        term in query_text for term in ("waiting period", "exclusion")
    ):
        waiting_doc = _best_pdf_doc(
            docs,
            required_markers=("before conclusion", "during the waiting period"),
            preferred_markers=("does not cover", "legal protection"),
        )
        if waiting_doc is not None:
            requirements.append(
                _requirement(
                    "legal_precontract_waiting_exclusion",
                    "State that cases arising before contract conclusion or during an applicable waiting period are excluded.",
                    waiting_doc,
                    (r"\bbefore (?:contract )?conclusion\b", r"\bpre.existing\b"),
                    (r"\bduring (?:an applicable |the )?waiting period\b",),
                    (r"\bnot cover", r"\bexclude", r"\bno legal protection\b"),
                )
            )
            requirements.append(
                _requirement(
                    "legal_specific_exclusions",
                    "State that legal-protection claims not specifically named are not covered and that subject-specific exclusions apply.",
                    waiting_doc,
                    (r"\bnot specifically named\b", r"\bnot every legal dispute\b"),
                    (r"\bnot cover", r"\bexclude", r"\bno legal protection\b"),
                )
            )

    if "private liability" in query_text and any(
        term in query_text for term in ("cover", "covered", "damage", "liability")
    ):
        doc = _best_pdf_doc_any(
            docs,
            required_markers=("statutory liability", "unjustified claims", "property damage"),
            preferred_markers=("personal injury", "financial loss", "third parties"),
        )
        if doc is not None:
            requirements.append(
                _requirement(
                    "private_liability_general_scope",
                    "The PDF table has separate insured-loss and exclusion columns. State that statutory third-party liability and defence against unjustified claims are insured, including bodily injury and property damage; do not misread the adjacent general-exclusions column as excluding all property damage.",
                    doc,
                    (r"\b(?:statutory|third.party) liability\b", r"\bthird.party claim"),
                    (r"\bproperty damage\b",),
                )
            )

    if "assistance" in query_text and (
        "travel" in query_text or "assistance brochure" in query_text
    ):
        predeparture_doc = _best_pdf_doc_any(
            docs,
            required_markers=("cancellation", "before departure", "cancel your trip"),
            preferred_markers=("unforeseen", "illness", "accident"),
        )
        if predeparture_doc is not None:
            requirements.append(
                _requirement(
                    "travel_predeparture_cancellation",
                    "Include cancellation protection for supported unforeseen insured events before departure.",
                    predeparture_doc,
                    (r"\bcancel", r"\bcancellation\b"),
                    (r"\bbefore departure\b", r"\bpre.departure\b"),
                )
            )
        during_doc = _best_pdf_doc_any(
            docs,
            required_markers=("transport costs", "board and lodging", "unused services"),
            preferred_markers=("while travelling", "return transport", "cost advance"),
        )
        if during_doc is not None:
            requirements.append(
                _requirement(
                    "travel_during_trip_assistance",
                    "Include supported transport, lodging, advance or unused-service benefits while travelling.",
                    during_doc,
                    (r"\btransport\b", r"\blodging\b", r"\bunused service"),
                )
            )

    if "insurance services" in query_text and "brochure" in query_text:
        service_doc = _best_pdf_doc_any(
            docs,
            required_markers=("24 hours", "24/7", "48h", "48 hours"),
            preferred_markers=("365 days", "claims payment", "successful"),
        )
        if service_doc is not None:
            requirements.append(
                _requirement(
                    "services_round_clock_and_payment",
                    "Include round-the-clock emergency assistance and the 48-hour claim-payment promise after successful review.",
                    service_doc,
                    (r"\b24\s*/\s*7\b", r"\b24 hours\b", r"\bround.the.clock\b"),
                    (r"\b48\s*(?:h|hours?)\b",),
                )
            )

    if "mutual provisions" in query_text:
        contract_doc = _best_pdf_doc_any(
            docs,
            required_markers=("term and termination", "insurance contract", "annual premium"),
            preferred_markers=("duration", "renewed", "notice"),
        )
        if contract_doc is not None:
            requirements.append(
                _requirement(
                    "mutual_contract_rules",
                    "Summarize contract formation or basis, duration, termination and premium rules.",
                    contract_doc,
                    (r"\bcontract\b",),
                    (r"\bterminat", r"\bduration\b", r"\bterm\b"),
                    (r"\bpremium",),
                )
            )
        claim_doc = _best_pdf_doc_any(
            docs,
            required_markers=("event of a claim", "reductions in compensation", "sanctions", "recourse"),
            preferred_markers=("obligations", "benefits"),
        )
        if claim_doc is not None:
            requirements.append(
                _requirement(
                    "mutual_claim_rules",
                    "Summarize duties and benefits in a claim, including reductions, sanctions or recourse where described.",
                    claim_doc,
                    (r"\bclaim\b",),
                    (r"\bbenefit", r"\bobligation", r"\bdut"),
                    (r"\breduction", r"\bsanction", r"\brecourse"),
                )
            )

    if any(term in query_text for term in ("windscreen", "windshield")) and all(
        term in query_text for term in ("repair", "replacement")
    ):
        glass_doc = _best_pdf_doc_any(
            docs,
            required_markers=("windscreen", "windshield", "repaired", "replacement"),
            preferred_markers=("safety reasons", "partner company", "organised by helvetia"),
        )
        if glass_doc is not None:
            requirements.append(
                _requirement(
                    "windscreen_repair_replacement_process",
                    "Distinguish safety-required repair or replacement and explain repair organised by Helvetia or a designated partner where supported.",
                    glass_doc,
                    (r"\brepair",),
                    (r"\breplac",),
                    (r"\bsafety\b", r"\bpartner\b", r"\borganis"),
                )
            )

    if "motor" in query_text and "hail" in query_text:
        hail_doc = _best_pdf_doc_any(
            docs,
            required_markers=("hail", "natural forces"),
            preferred_markers=("partially comprehensive", "insured"),
        )
        if hail_doc is not None:
            requirements.append(
                _requirement(
                    "motor_hail_partial_comprehensive",
                    "State that hail is a covered natural-force event under the applicable partially comprehensive terms.",
                    hail_doc,
                    (r"\bhail\b",),
                    (r"\bnatural force", r"\bpartially comprehensive\b", r"\bpartial comprehensive\b"),
                )
            )

    return requirements


def build_answer_requirements(query: str, docs: Sequence[Document]) -> tuple[AnswerRequirement, ...]:
    requirements: list[AnswerRequirement] = []
    event = _event_from_query(query)
    if event == "theft":
        requirements.extend(_theft_requirements(query, docs))
    requirements.extend(_domain_requirements(query, docs))
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

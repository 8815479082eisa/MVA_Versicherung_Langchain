from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryMode(str, Enum):
    RETRIEVAL_ONLY = "retrieval_only"
    CRM_ONLY = "crm_only"
    COMBINED = "combined"
    DENIED = "denied"


_POLICY_NUMBER = re.compile(
    r"\b(?:TEST-)?(?!CLM-|CLAIM-|SCHADEN-)[A-Z]{2,12}-\d{4}-\d{3,8}\b",
    re.IGNORECASE,
)
_CLAIM_NUMBER = re.compile(
    r"\b(?:TEST-)?(?:CLM|CLAIM|SCHADEN)-\d{4}-\d{3,8}\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PDF_FILENAME = re.compile(r"\b[A-Z0-9_.-]+\.pdf\b", re.IGNORECASE)

_CRM_TERMS = {
    "policy",
    "policies",
    "contract",
    "customer",
    "claim",
    "claims",
    "status",
    "premium",
    "deductible",
    "police",
    "policen",
    "vertrag",
    "verträge",
    "vertraege",
    "kunde",
    "kundin",
    "schaden",
    "schäden",
    "schaeden",
    "bearbeitungsstand",
    "prämie",
    "praemie",
    "selbstbeteiligung",
    "address",
    "contact information",
    "telephone",
    "phone",
    "email",
    "date of birth",
    "birth date",
    "adresse",
    "anschrift",
    "telefon",
    "geburtsdatum",
}
_DOCUMENT_TERMS = {
    "cover",
    "covered",
    "coverage",
    "generally",
    "according to",
    "conditions",
    "terms",
    "exclude",
    "exclusion",
    "versichert",
    "deckung",
    "gedeckt",
    "allgemein",
    "bedingungen",
    "ausschluss",
    "ausgeschlossen",
    "leistet",
    "leistung",
}
_POLICY_TERMS = {
    "policy",
    "policies",
    "contract",
    "premium",
    "deductible",
    "police",
    "policen",
    "vertrag",
    "verträge",
    "vertraege",
    "prämie",
    "praemie",
    "selbstbeteiligung",
}
_CLAIM_TERMS = {
    "claim",
    "claims",
    "schaden",
    "schäden",
    "schaeden",
    "schadenstatus",
    "claim status",
    "bearbeitungsstand",
}
_KNOWLEDGE_TERMS = _DOCUMENT_TERMS | {
    "insurance",
    "insured",
    "comprehensive",
    "collision",
    "damage",
    "glass",
    "windscreen",
    "windshield",
    "marten",
    "theft",
    "fire",
    "benefit",
    "limit",
    "deductible",
}
_CRM_INDEPENDENT_KNOWLEDGE_TERMS = _KNOWLEDGE_TERMS - {
    "deductible",
    "limit",
}
_CRM_DIRECTIVE_PREFIX = re.compile(
    r"^(?:(?:also|please)\s+)?(?:state|provide|give|list|show|identify|use\s+crm|"
    r"separate|clearly\s+distinguish|do\s+not|for\s+(?:the\s+)?customer)\b",
    re.IGNORECASE,
)
_NEGATED_CLAIM_DIRECTIVE = re.compile(
    r"\b(?:do\s+not|don't|without|no)\s+"
    r"(?:make|provide|give|issue|render|reach|include)?\s*"
    r"(?:a\s+|an\s+|the\s+)?(?:final\s+)?"
    r"(?:claim|coverage|benefit|liability|schaden|leistungs?)\s+"
    r"(?:decision|determination|approval|denial|entscheid(?:ung)?|entscheid)\b",
    re.IGNORECASE,
)
_BROAD_ACCESS_PATTERNS = (
    re.compile(r"\b(?:all|every|entire)\s+(?:customers?|contacts?|policies|claims)\b", re.IGNORECASE),
    re.compile(r"\b(?:alle|sämtliche|saemtliche)\s+(?:kunden|kontakte|policen|verträge|vertraege|schäden|schaeden)\b", re.IGNORECASE),
    re.compile(r"\b(?:export|dump|download)\b.*\b(?:crm|customers?|contacts?|policies|claims)\b", re.IGNORECASE),
    re.compile(r"\b(?:list|show|display)\s+(?:the\s+)?(?:customers?|contacts?|policies|claims)\b", re.IGNORECASE),
    re.compile(r"\b(?:liste|zeige)\s+(?:die\s+)?(?:kunden|kontakte|policen|verträge|vertraege|schäden|schaeden)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class QueryPlan:
    mode: QueryMode
    reason: str
    customer_name: str | None = None
    customer_email: str | None = None
    policy_number: str | None = None
    claim_number: str | None = None
    needs_policies: bool = False
    needs_claims: bool = False
    needs_contact: bool = False

    @property
    def uses_crm(self) -> bool:
        return self.mode in {QueryMode.CRM_ONLY, QueryMode.COMBINED}


def plan_insurance_query(question: str) -> QueryPlan:
    """Route with lexical rules only; this deliberately performs no LLM call."""

    clean_question = " ".join((question or "").strip().split())
    lowered = clean_question.lower()

    if any(pattern.search(clean_question) for pattern in _BROAD_ACCESS_PATTERNS):
        return QueryPlan(
            mode=QueryMode.DENIED,
            reason="Broad CRM enumeration or export is not an authorized operation.",
        )

    policy_number = _first_match(_POLICY_NUMBER, clean_question)
    claim_number = _first_match(_CLAIM_NUMBER, clean_question)
    customer_email = _first_match(_EMAIL, clean_question, uppercase=False)
    customer_name = _extract_customer_name(clean_question)

    has_identity = any((policy_number, claim_number, customer_email, customer_name))
    mentions_crm = _contains_term(lowered, _CRM_TERMS) or bool(
        policy_number or claim_number
    )
    document_intent_text = re.sub(
        r"\b(?:coverage|deckung)\s+type\b|\bdeckungsart\b",
        " ",
        lowered,
    )
    needs_documents = _contains_term(document_intent_text, _DOCUMENT_TERMS)

    if not (has_identity and mentions_crm):
        return QueryPlan(
            mode=QueryMode.RETRIEVAL_ONLY,
            reason="No specific CRM record is required.",
        )

    needs_policies = bool(policy_number) or _contains_term(lowered, _POLICY_TERMS)
    claim_lookup_text = _NEGATED_CLAIM_DIRECTIVE.sub(" ", lowered)
    needs_claims = bool(claim_number) or _contains_term(
        claim_lookup_text,
        _CLAIM_TERMS,
    )
    needs_contact = bool(customer_name or customer_email) or bool(
        policy_number and re.search(r"\b(?:customer|contact|kunde|kundin|who)\b", lowered)
    )
    if needs_documents:
        mode = QueryMode.COMBINED
        reason = "The question needs both customer-specific CRM facts and document evidence."
    else:
        mode = QueryMode.CRM_ONLY
        reason = "The question asks only for customer-specific structured CRM facts."

    return QueryPlan(
        mode=mode,
        reason=reason,
        customer_name=customer_name,
        customer_email=customer_email,
        policy_number=policy_number,
        claim_number=claim_number,
        needs_policies=needs_policies,
        needs_claims=needs_claims,
        needs_contact=needs_contact,
    )


def extract_requested_pdf_filename(question: str) -> str | None:
    """Return an explicitly named PDF so retrieval cannot mix document editions."""

    match = _PDF_FILENAME.search(question or "")
    return match.group(0) if match is not None else None


def build_knowledge_query(question: str, plan: QueryPlan) -> str:
    """Remove CRM-only instructions before sending a Combined query to RAG."""

    clean_question = " ".join((question or "").strip().split())
    if not clean_question:
        return ""

    identity_values = (
        plan.customer_name,
        plan.customer_email,
        plan.policy_number,
        plan.claim_number,
    )
    without_identity = clean_question
    for value in identity_values:
        if value:
            without_identity = re.sub(
                re.escape(value),
                " ",
                without_identity,
                flags=re.IGNORECASE,
            )

    clauses = re.split(
        r"(?<=[.!?])\s+|;\s*|,\s+(?:and|then|also)\s+",
        without_identity,
        flags=re.IGNORECASE,
    )
    knowledge_clauses: list[str] = []
    for clause in clauses:
        clause = " ".join(clause.split()).strip(" ,;:")
        if not clause:
            continue

        lowered = clause.lower()
        if (
            "crm" in lowered
            and not _PDF_FILENAME.search(clause)
            and not _contains_term(
                lowered,
                _CRM_INDEPENDENT_KNOWLEDGE_TERMS,
            )
        ):
            continue
        if _CRM_DIRECTIVE_PREFIX.search(clause) and (
            "crm" in lowered or _contains_term(lowered, _CRM_TERMS)
        ):
            continue

        clause = re.sub(
            r"\b(?:based on|using|from)\s+(?:the\s+)?crm"
            r"(?:\s+policy\s+data)?\s*(?:and\s+)?",
            "",
            clause,
            flags=re.IGNORECASE,
        ).strip(" ,;:")
        lowered = clause.lower()
        if not clause:
            continue
        if "crm" in lowered:
            continue
        if _CRM_DIRECTIVE_PREFIX.search(clause):
            continue
        if _PDF_FILENAME.search(clause) or _contains_term(lowered, _KNOWLEDGE_TERMS):
            knowledge_clauses.append(clause)

    if knowledge_clauses:
        knowledge_query = " ".join(knowledge_clauses)
        requested_pdf = extract_requested_pdf_filename(knowledge_query)
        if requested_pdf:
            without_filename = re.sub(
                re.escape(requested_pdf),
                " ",
                knowledge_query,
                flags=re.IGNORECASE,
            )
            if "," in without_filename:
                _, candidate_question = without_filename.split(",", 1)
                candidate_question = " ".join(candidate_question.split()).strip()
                if _contains_term(
                    candidate_question.lower(),
                    _CRM_INDEPENDENT_KNOWLEDGE_TERMS,
                ):
                    without_filename = candidate_question
            knowledge_query = " ".join(without_filename.split()).strip(" ,;:")
        return knowledge_query

    return " ".join(without_identity.split()).strip()


def enrich_knowledge_query(
    knowledge_query: str,
    *,
    product_hints: tuple[str, ...] = (),
) -> str:
    """Add general insurance synonyms and selected-product hints for corpus search."""

    clean_query = " ".join((knowledge_query or "").split()).strip()
    lowered = clean_query.lower()
    hints = [
        " ".join(str(hint).split()).strip()
        for hint in product_hints
        if " ".join(str(hint).split()).strip()
    ]
    hint_text = " ".join(hints).lower()
    motor_context = bool(
        re.search(r"\b(?:car|motor|vehicle)\b", lowered)
        or re.search(r"\b(?:motor|vehicle)\b", hint_text)
    )
    coverage_hints = [
        hint
        for hint in hints
        if re.search(
            r"\b(?:partially|fully|partial|comprehensive|third-party|liability)\b",
            hint,
            re.IGNORECASE,
        )
    ]
    concise_coverage_hint = min(coverage_hints, key=len) if coverage_hints else ""
    expansions: list[str] = []

    if motor_context and re.search(
        r"\b(?:theft|stolen|steal|robbery|misappropriation)\b", lowered
    ):
        return " ".join(
            part
            for part in (
                "motor vehicle insurance",
                concise_coverage_hint,
                "vehicle theft loss disappearance destruction insured vehicle",
                "police without delay",
            )
            if part
        )

    if motor_context and re.search(
        r"\b(?:collision|crash|impact|overturning)\b", lowered
    ):
        return " ".join(
            part
            for part in (
                "motor vehicle insurance",
                concise_coverage_hint,
                "collision events sudden violent external effects impact overturning crashing",
            )
            if part
        )

    if motor_context and re.search(r"\b(?:hail|hailstorm)\b", lowered):
        return " ".join(
            part
            for part in (
                "motor vehicle insurance",
                concise_coverage_hint,
                "natural forces hail damage repair organised by Helvetia partner company",
            )
            if part
        )

    has_glass = bool(
        re.search(
            r"\b(?:windscreen|windshield|glass|scheibe|verglasung)\b",
            lowered,
        )
    )
    if has_glass:
        expansions.extend(
            (
                "glass breakage",
                "part comprehensive",
                "windscreen damage",
            )
        )
    if (
        re.search(r"\b(?:repair|repaired|reparatur|repariert)\b", lowered)
        and (
            has_glass
            or re.search(
                r"\b(?:replace|replaced|replacement|austausch|ersetzt|deductible|"
                r"excess|selbstbehalt|selbstbeteiligung)\b",
                lowered,
            )
        )
    ):
        expansions.extend(
            (
                "repaired rather than replaced",
                "deductible not applied",
                "partner repair service",
                "customer service notification",
            )
        )

    additions = list(dict.fromkeys(hints + expansions))
    if not additions:
        return clean_query
    if not clean_query:
        return " ".join(additions)
    return f"{clean_query} {' '.join(additions)}"


def _contains_term(text: str, terms: set[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in terms)


def _first_match(
    pattern: re.Pattern[str],
    text: str,
    *,
    uppercase: bool = True,
) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(0)
    return value.upper() if uppercase else value.lower()


def _extract_customer_name(question: str) -> str | None:
    """Extract an explicitly written two-part person name conservatively."""

    question = re.sub(
        r"^(?:show|list|give|provide|display|find)\s+",
        "",
        question or "",
        flags=re.IGNORECASE,
    )
    candidates = re.findall(
        r"\b([A-ZÄÖÜ][a-zäöüß]{2,30})\s+([A-ZÄÖÜ][a-zäöüß-]{2,40})\b",
        question,
    )
    excluded_first_words = {
        "Show",
        "List",
        "Give",
        "Provide",
        "Display",
        "Find",
        "Which",
        "What",
        "Current",
        "Partial",
        "Glass",
        "Damage",
        "Policy",
        "Claim",
    }
    for first_name, last_name in candidates:
        if first_name not in excluded_first_words:
            return f"{first_name} {last_name}"
    return None

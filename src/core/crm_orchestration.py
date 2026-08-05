from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Awaitable, Callable

from src.core.current_policy import requires_policy_history, select_current_policy_records
from src.core.insurance_tool_routing import QueryPlan


ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class CRMOrchestrationError(RuntimeError):
    def __init__(self, error: dict[str, Any], *, available: bool):
        super().__init__(str(error.get("message") or "CRM tool request failed."))
        self.error = error
        self.available = available


@dataclass(frozen=True)
class CRMQueryResult:
    answer: str
    sources: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...]
    reason_code: str = "READ_ONLY_ACCESS_ALLOWED"


@dataclass(frozen=True)
class CRMContextSelection:
    sources: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    retrieval_hints: tuple[str, ...]
    selected_policy_numbers: tuple[str, ...]


_CONTEXT_STOPWORDS = {
    "active",
    "answer",
    "coverage",
    "crm",
    "customer",
    "deductible",
    "facts",
    "give",
    "individual",
    "insurance",
    "number",
    "policy",
    "provide",
    "state",
    "status",
    "type",
}


async def execute_crm_query(
    plan: QueryPlan,
    invoke: ToolInvoker,
) -> CRMQueryResult:
    """Execute only the bounded read operations declared by a query plan."""

    calls: list[tuple[str, dict[str, Any]]] = []
    if plan.policy_number:
        calls.append(("get_policy", {"policy_number": plan.policy_number}))
    if plan.claim_number:
        calls.append(("get_claim_status", {"claim_number": plan.claim_number}))

    customer_result: dict[str, Any] | None = None
    if plan.customer_name or plan.customer_email:
        customer_result = await invoke(
            "find_customer",
            {
                "name": plan.customer_name,
                "email": plan.customer_email,
            },
        )
        _raise_for_tool_error(customer_result)
        if customer_result.get("ambiguous") is True:
            return CRMQueryResult(
                answer=(
                    "Multiple contacts match this name. Please provide the customer "
                    "number or policy number."
                ),
                sources=(),
                tool_results=(customer_result,),
                reason_code="AMBIGUOUS_CUSTOMER",
            )
        if customer_result.get("found") is True:
            customer = customer_result.get("customer") or {}
            customer_id = customer.get("id")
            if not isinstance(customer_id, str) or not customer_id:
                raise CRMOrchestrationError(
                    {
                        "category": "malformed_response",
                        "message": "CRM returned a customer without an identifier.",
                    },
                    available=True,
                )
            if plan.needs_policies and not plan.policy_number:
                calls.append(
                    ("get_customer_policies", {"customer_id": customer_id})
                )
            if plan.needs_claims and not plan.claim_number:
                calls.append(("get_customer_claims", {"customer_id": customer_id}))

    called_results = await asyncio.gather(
        *(invoke(tool_name, arguments) for tool_name, arguments in calls)
    )
    for result in called_results:
        _raise_for_tool_error(result)

    results = ([customer_result] if customer_result is not None else []) + list(
        called_results
    )
    valid_results = [item for item in results if item is not None]
    _validate_entity_bindings(valid_results)
    facts = _format_crm_facts(valid_results)
    sources = _build_crm_sources(valid_results)
    return CRMQueryResult(
        answer=facts,
        sources=tuple(sources),
        tool_results=tuple(item for item in results if item is not None),
    )


def _validate_entity_bindings(results: list[dict[str, Any]]) -> None:
    contact_ids = {
        str(result["customer"].get("id") or "").strip()
        for result in results
        if isinstance(result.get("customer"), dict)
    }
    contact_ids.discard("")
    related_customer_ids: set[str] = set()
    for result in results:
        records: list[dict[str, Any]] = []
        for key in ("policy", "claim"):
            if isinstance(result.get(key), dict):
                records.append(result[key])
        for key in ("policies", "claims"):
            records.extend(item for item in result.get(key) or [] if isinstance(item, dict))
        related_customer_ids.update(
            str(record.get("customer_id") or "").strip()
            for record in records
            if str(record.get("customer_id") or "").strip()
        )

    if contact_ids and related_customer_ids and not related_customer_ids.issubset(contact_ids):
        raise CRMOrchestrationError(
            {
                "category": "entity_mismatch",
                "reason_code": "ENTITY_MISMATCH",
                "message": "The retrieved policy or claim belongs to a different contact.",
            },
            available=True,
        )


def _raise_for_tool_error(result: dict[str, Any]) -> None:
    if result.get("ok") is not False:
        return
    error = result.get("error")
    if not isinstance(error, dict):
        error = {
            "category": "malformed_response",
            "message": "CRM tool returned an invalid error response.",
        }
    raise CRMOrchestrationError(
        error,
        available=bool(result.get("available", False)),
    )


def _format_crm_facts(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for result in results:
        if "customer" in result:
            customer = result.get("customer")
            if result.get("found") is False or not customer:
                lines.append("No matching customer was found in CRM.")
            else:
                details = [f"Customer: {customer.get('display_name', 'unknown')}"]
                for label, key in (
                    ("Contact ID", "id"),
                    ("Address", "address"),
                    ("Telephone", "phone"),
                    ("Email", "email"),
                    ("Date of birth", "date_of_birth"),
                ):
                    value = customer.get(key)
                    if value:
                        details.append(f"{label}: {value}")
                if not customer.get("date_of_birth"):
                    details.append("Date of birth: not found in retrieved CRM data")
                lines.append("; ".join(details) + ".")

        if "policies" in result:
            policies = result.get("policies") or []
            if not policies:
                lines.append("No policies were found for this customer.")
            for policy in policies:
                lines.append(_format_policy(policy))

        if "policy" in result:
            policy = result.get("policy")
            if result.get("found") is False or not policy:
                lines.append("No matching policy was found in CRM.")
            else:
                lines.append(_format_policy(policy))

        if "claims" in result:
            claims = result.get("claims") or []
            if not claims:
                lines.append("No claims were found for this customer.")
            for claim in claims:
                lines.append(_format_claim(claim))

        if "claim" in result:
            claim = result.get("claim")
            if result.get("found") is False or not claim:
                lines.append("No matching claim was found in CRM.")
            else:
                lines.append(_format_claim(claim))

    return "\n".join(lines) if lines else "CRM returned no matching insurance facts."


def _format_policy(policy: dict[str, Any]) -> str:
    deductible = _format_money(policy.get("deductible"))
    premium = _format_money(policy.get("annual_premium"))
    date_range = " to ".join(
        value
        for value in (policy.get("start_date"), policy.get("end_date"))
        if value
    )
    details = [
        policy.get("product_type"),
        policy.get("coverage_type"),
        policy.get("status"),
    ]
    policy_label = str(policy.get("policy_number") or "unknown")
    policy_name = str(policy.get("name") or "").strip()
    if policy_name:
        policy_label += f" ({policy_name})"
    text = f"Policy {policy_label}: " + ", ".join(
        str(item) for item in details if item
    )
    if date_range:
        text += f"; term {date_range}"
    if deductible:
        text += f"; deductible {deductible}"
    if premium:
        text += f"; annual premium {premium}"
    return text + "."


def _format_claim(claim: dict[str, Any]) -> str:
    amount = _format_money(claim.get("claimed_amount"))
    text = (
        f"Claim {claim.get('claim_number') or 'unknown'}: "
        f"{claim.get('damage_type') or 'damage type not recorded'}, "
        f"status {claim.get('status') or 'not recorded'}"
    )
    if claim.get("claim_date"):
        text += f"; reported {claim['claim_date']}"
    if amount:
        text += f"; claimed amount {amount}"
    if claim.get("policy_reference"):
        text += f"; policy {claim['policy_reference']}"
    if claim.get("description"):
        text += f"; description {claim['description']}"
    return text + "."


def _format_money(value: Any) -> str:
    if not isinstance(value, dict) or value.get("amount") is None:
        return ""
    return f"{value['amount']} {value.get('currency') or 'EUR'}"


def _build_crm_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        records: list[tuple[str, dict[str, Any]]] = []
        if isinstance(result.get("customer"), dict):
            records.append(("Contact", result["customer"]))
        if isinstance(result.get("policy"), dict):
            records.append(("Policy", result["policy"]))
        if isinstance(result.get("claim"), dict):
            records.append(("Claim", result["claim"]))
        records.extend(
            ("Policy", item)
            for item in result.get("policies") or []
            if isinstance(item, dict)
        )
        records.extend(
            ("Claim", item)
            for item in result.get("claims") or []
            if isinstance(item, dict)
        )

        for entity_type, record in records:
            record_id = str(record.get("id") or "").strip()
            if not record_id or record_id in seen:
                continue
            seen.add(record_id)
            label = (
                record.get("policy_number")
                or record.get("claim_number")
                or record.get("display_name")
                or record_id
            )
            sources.append(
                {
                    "documentId": f"espocrm:{record_id}",
                    "documentTitle": f"EspoCRM {entity_type}",
                    "section": str(label),
                    "snippet": _source_snippet(entity_type, record),
                    "entityType": entity_type.lower(),
                    "entityId": record_id,
                    "contactId": record_id if entity_type == "Contact" else record.get("customer_id"),
                    "policyId": record_id if entity_type == "Policy" else record.get("policy_id"),
                    "claimId": record_id if entity_type == "Claim" else None,
                    "entityBindingValid": True,
                }
            )
    return sources


def select_crm_sources_for_context(
    result: CRMQueryResult,
    question: str,
) -> tuple[dict[str, Any], ...]:
    """Compatibility wrapper returning the records selected by the ranked policy logic."""

    return select_crm_context(result, question).sources


def select_crm_context(
    result: CRMQueryResult,
    question: str,
    *,
    as_of_date: date | None = None,
) -> CRMContextSelection:
    """Select one policy using explicit, product, status, validity and topic signals."""

    contacts = [
        source
        for source in result.sources
        if source.get("documentTitle") == "EspoCRM Contact"
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in result.sources:
        title = str(source.get("documentTitle") or "")
        if title == "EspoCRM Contact":
            continue
        grouped.setdefault(title, []).append(source)

    query_tokens = _context_tokens(question)
    selected: list[dict[str, Any]] = list(contacts)
    evidence: list[dict[str, Any]] = []
    structured_records = _structured_records_by_id(result)
    effective_date = as_of_date or _question_as_of_date(question) or date.today()

    policy_sources = grouped.pop("EspoCRM Policy", [])
    selected_policy_numbers: list[str] = []
    retrieval_hints: list[str] = []
    if policy_sources:
        policy_entries = [
            (
                source,
                _record_for_source(source, structured_records),
            )
            for source in policy_sources
        ]
        current_selection = select_current_policy_records(
            question,
            [record for _, record in policy_entries],
            reference_date=effective_date,
        )
        selected_record_ids = {
            str(record.get("id") or "").strip()
            for record in current_selection.selected_records
        }
        ranked_policies: list[
            tuple[tuple[int, int, int, int, str], dict[str, Any], dict[str, Any]]
        ] = []
        for source, record in policy_entries:
            score, signals = _policy_selection_score(
                source,
                record,
                question,
                query_tokens,
                effective_date,
            )
            start_ordinal = _date_ordinal(record.get("start_date"))
            end_ordinal = _date_ordinal(record.get("end_date"))
            policy_number = str(
                record.get("policy_number") or source.get("section") or ""
            )
            ranking_key = (
                int(signals["explicit_policy_number"]),
                score,
                start_ordinal,
                end_ordinal,
                policy_number,
            )
            ranked_policies.append((ranking_key, source, record))
            evidence.append(
                {
                    "entity_type": "policy",
                    "document_id": source.get("documentId"),
                    "policy_number": policy_number or None,
                    "score": score,
                    "signals": signals,
                    "effective_date": effective_date.isoformat(),
                    "selected": False,
                    "current_policy_filter_activated": current_selection.intent_activated,
                    "current_policy_filter_reason": current_selection.reason,
                }
            )

        chosen_source: dict[str, Any] | None = None
        chosen_record: dict[str, Any] | None = None
        selection_reason = ""
        if current_selection.intent_activated:
            if selected_record_ids:
                chosen_source, chosen_record = next(
                    (source, record)
                    for source, record in policy_entries
                    if str(record.get("id") or "").strip() in selected_record_ids
                )
                selection_reason = current_selection.reason
        elif requires_policy_history(question):
            for source, record in policy_entries:
                evidence_item = next(
                    (
                        item
                        for item in evidence
                        if item.get("document_id") == source.get("documentId")
                    ),
                    {},
                )
                if (evidence_item.get("signals") or {}).get("conflicting_domain"):
                    continue
                selected.append(source)
                policy_number = str(
                    record.get("policy_number") or source.get("section") or ""
                ).strip()
                if policy_number:
                    selected_policy_numbers.append(policy_number)
                retrieval_hints.extend(_policy_retrieval_hints(record, contacts))
                if evidence_item:
                    evidence_item["selected"] = True
                    evidence_item["selection_reason"] = (
                        "Historical request preserves records in the requested product domain."
                    )
        else:
            _, chosen_source, chosen_record = max(
                ranked_policies,
                key=lambda item: item[0],
            )
            selection_reason = (
                "Highest deterministic policy rank: explicit number, "
                "domain/name match, active status, temporal validity, "
                "then latest effective start date."
            )

        if chosen_source is not None and chosen_record is not None:
            selected.append(chosen_source)
            chosen_id = chosen_source.get("documentId")
            for item in evidence:
                if item.get("document_id") == chosen_id:
                    item["selected"] = True
                    item["selection_reason"] = selection_reason
            chosen_number = str(
                chosen_record.get("policy_number")
                or chosen_source.get("section")
                or ""
            ).strip()
            if chosen_number:
                selected_policy_numbers.append(chosen_number)
            retrieval_hints.extend(
                _policy_retrieval_hints(chosen_record, contacts)
            )

    for title, sources in grouped.items():
        ranked = sorted(
            sources,
            key=lambda source: (
                _crm_source_relevance(source, question, query_tokens),
                _record_date_rank(
                    _record_for_source(source, structured_records)
                ),
                str(source.get("section") or ""),
            ),
            reverse=True,
        )
        if not ranked:
            continue
        selected.append(ranked[0])
        evidence.extend(
            {
                "entity_type": title.removeprefix("EspoCRM ").lower(),
                "document_id": source.get("documentId"),
                "section": source.get("section"),
                "score": _crm_source_relevance(
                    source,
                    question,
                    query_tokens,
                ),
                "selected": source is ranked[0],
            }
            for source in ranked
        )

    return CRMContextSelection(
        sources=tuple(selected),
        evidence=tuple(evidence),
        retrieval_hints=tuple(dict.fromkeys(retrieval_hints)),
        selected_policy_numbers=tuple(selected_policy_numbers),
    )


def _context_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in _CONTEXT_STOPWORDS
    }


def _crm_source_relevance(
    source: dict[str, Any],
    question: str,
    query_tokens: set[str],
) -> int:
    section = str(source.get("section") or "")
    searchable = " ".join(
        str(source.get(field) or "")
        for field in ("documentTitle", "section", "snippet")
    )
    score = len(query_tokens & _context_tokens(searchable))
    if section and section.lower() in (question or "").lower():
        score += 5
    return score


_POLICY_DOMAIN_TERMS: dict[str, set[str]] = {
    "motor": {
        "auto",
        "automobile",
        "car",
        "collision",
        "fahrzeug",
        "glass",
        "kfz",
        "marten",
        "motor",
        "vehicle",
        "windshield",
        "windscreen",
    },
    "liability": {
        "haftpflicht",
        "liability",
        "third-party",
    },
    "home": {
        "contents",
        "hausrat",
        "home",
        "household",
        "renters",
    },
    "travel": {
        "reise",
        "travel",
        "trip",
    },
}


def _structured_records_by_id(
    result: CRMQueryResult,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for tool_result in result.tool_results:
        candidates: list[dict[str, Any]] = []
        for singular in ("customer", "policy", "claim"):
            value = tool_result.get(singular)
            if isinstance(value, dict):
                candidates.append(value)
        for plural in ("policies", "claims"):
            candidates.extend(
                value
                for value in tool_result.get(plural) or []
                if isinstance(value, dict)
            )
        for record in candidates:
            record_id = str(record.get("id") or "").strip()
            if record_id:
                records[record_id] = record
    return records


def _record_for_source(
    source: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(source.get("documentId") or "")
    record_id = document_id.removeprefix("espocrm:")
    record = records.get(record_id)
    if record is not None:
        return record
    return {
        "id": record_id,
        "policy_number": source.get("section"),
        "name": "",
        "product_type": "",
        "coverage_type": "",
        "status": (
            "Active"
            if re.search(
                r"(?<!\w)active(?!\w)",
                str(source.get("snippet") or ""),
                re.IGNORECASE,
            )
            else ""
        ),
        "snippet": source.get("snippet"),
    }


def _policy_selection_score(
    source: dict[str, Any],
    record: dict[str, Any],
    question: str,
    query_tokens: set[str],
    effective_date: date,
) -> tuple[int, dict[str, Any]]:
    lowered_question = (question or "").lower()
    policy_number = str(
        record.get("policy_number") or source.get("section") or ""
    ).strip()
    policy_name = str(record.get("name") or "").strip()
    product_type = str(record.get("product_type") or "").strip()
    coverage_type = str(record.get("coverage_type") or "").strip()
    status = str(record.get("status") or "").strip().lower()
    searchable = " ".join(
        (
            policy_number,
            policy_name,
            product_type,
            coverage_type,
            str(source.get("snippet") or ""),
        )
    )

    explicit_policy_number = bool(
        policy_number and policy_number.lower() in lowered_question
    )
    lexical_overlap = len(query_tokens & _context_tokens(searchable))
    product_domain, product_domain_match, conflicting_domain = (
        _policy_domain_signals(product_type, policy_name, lowered_question)
    )
    name_tokens = _context_tokens(policy_name)
    name_overlap = len(query_tokens & name_tokens)
    exact_name_match = bool(
        policy_name and policy_name.lower() in lowered_question
    )
    active_status = status in {"active", "aktiv"}
    start_date = _parse_date(record.get("start_date"))
    end_date = _parse_date(record.get("end_date"))
    temporally_valid = (
        (start_date is None or start_date <= effective_date)
        and (end_date is None or effective_date <= end_date)
    )
    asks_current = bool(
        re.search(
            r"\b(?:active|current|currently|valid|aktiv|aktuell|gültig|gueltig)\b",
            lowered_question,
        )
    )

    score = 0
    score += 10_000 if explicit_policy_number else 0
    score += 250 if exact_name_match else 0
    score += name_overlap * 20
    score += lexical_overlap * 4
    score += 160 if product_domain_match else 0
    score -= 180 if conflicting_domain else 0
    score += 80 if active_status else -80
    if asks_current:
        score += 140 if temporally_valid else -300
    elif temporally_valid:
        score += 25

    return score, {
        "explicit_policy_number": explicit_policy_number,
        "exact_policy_name": exact_name_match,
        "policy_name_token_overlap": name_overlap,
        "lexical_overlap": lexical_overlap,
        "product_domain": product_domain,
        "product_domain_match": product_domain_match,
        "conflicting_domain": conflicting_domain,
        "active_status": active_status,
        "temporally_valid": temporally_valid,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "latest_effective_start_tiebreak": _date_ordinal(
            record.get("start_date")
        ),
    }


def _policy_domain_signals(
    product_type: str,
    policy_name: str,
    lowered_question: str,
) -> tuple[str | None, bool, bool]:
    record_text = f"{product_type} {policy_name}".lower()
    record_domains = {
        domain
        for domain, terms in _POLICY_DOMAIN_TERMS.items()
        if any(term in record_text for term in terms)
    }
    question_domains = {
        domain
        for domain, terms in _POLICY_DOMAIN_TERMS.items()
        if any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered_question)
            for term in terms
        )
    }
    domain = sorted(record_domains)[0] if record_domains else None
    return (
        domain,
        bool(record_domains & question_domains),
        bool(question_domains and record_domains.isdisjoint(question_domains)),
    )


def _question_as_of_date(question: str) -> date | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question or "")
    return _parse_date(match.group(1)) if match else None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _date_ordinal(value: Any) -> int:
    parsed = _parse_date(value)
    return parsed.toordinal() if parsed else 0


def _record_date_rank(record: dict[str, Any]) -> tuple[int, int]:
    return (
        _date_ordinal(record.get("claim_date") or record.get("start_date")),
        _date_ordinal(record.get("end_date")),
    )


def _policy_retrieval_hints(
    policy: dict[str, Any],
    contacts: list[dict[str, Any]],
) -> list[str]:
    name = str(policy.get("name") or "").strip()
    for contact in contacts:
        contact_name = str(contact.get("section") or "").strip()
        if contact_name:
            name = re.sub(
                re.escape(contact_name),
                " ",
                name,
                flags=re.IGNORECASE,
            )
    hints = [
        " ".join(name.split()),
        str(policy.get("product_type") or "").strip(),
        str(policy.get("coverage_type") or "").strip(),
    ]
    return [hint for hint in hints if hint]


def _source_snippet(entity_type: str, record: dict[str, Any]) -> str:
    if entity_type == "Policy":
        return _format_policy(record)
    if entity_type == "Claim":
        return _format_claim(record)
    return _format_crm_facts([{"customer": record, "found": True}])

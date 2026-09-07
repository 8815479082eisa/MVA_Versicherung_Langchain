from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CurrentPolicySelection:
    """Result of the deterministic current-policy metadata gate."""

    intent_activated: bool
    selected_records: tuple[Mapping[str, Any], ...]
    reason: str
    reference_date: date
    customer_key: str | None = None
    product_domain: str | None = None


_CURRENT_PATTERN = re.compile(
    r"\b(?:current|currently|valid|latest|newest|present|"
    r"aktuell\w*|gueltig\w*|gultig\w*|neueste\w*)\b"
)
_ACTIVE_CURRENT_PATTERN = re.compile(
    r"\bcurrently\s+active\b|\bactive\s+(?:policy|contract|insurance)\b|"
    r"\baktiv\w*\s+(?:police\w*|vertrag\w*|versicherung\w*)\b"
)
_POLICY_CONTEXT_PATTERN = re.compile(
    r"\b(?:policy|policies|contract|insurance|police\w*|vertrag\w*|versicherung\w*)\b"
)
_EXPLICIT_POLICY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b")
_HISTORICAL_PATTERN = re.compile(
    r"\b(?:earlier|older|old|previous\w*|former|historical|expired|cancelled|"
    r"past|damalig\w*|frueher\w*|alter\w*|aelter\w*|vorherig\w*|"
    r"abgelaufen\w*|gekuendigt\w*)\b"
)
_COMPARISON_PATTERN = re.compile(
    r"\b(?:compare|comparison|difference|versus|between|both|"
    r"vergleich\w*|unterschied\w*|zwischen|beide\w*)\b|\bvs\.?\b"
)
_LIST_COMMAND_PATTERN = re.compile(r"\b(?:all|list|show|every|which|alle|liste|zeige|welche)\b")
_PLURAL_POLICY_PATTERN = re.compile(
    r"\b(?:policies|contracts|policy\s+numbers?|contract\s+numbers?|"
    r"policen|vertraege|versicherungen)\b"
)
_STATUS_ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:current|actual|present|aktuell\w*)\s+(?:policy\s+|contract\s+)?status\b"
)

_PRODUCT_DOMAIN_TERMS: dict[str, set[str]] = {
    "motor": {
        "auto",
        "automobile",
        "car",
        "fahrzeug",
        "kfz",
        "motor",
        "vehicle",
    },
    "liability": {"haftpflicht", "liability", "third-party"},
    "household": {"contents", "hausrat", "home", "household", "renters"},
    "legal": {"legal", "rechtsschutz"},
    "travel": {"reise", "travel", "trip"},
}


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    return "".join(character for character in text if not unicodedata.combining(character))


def is_current_policy_intent(question: str) -> bool:
    """Activate only for an unambiguous current/latest single-policy request."""

    normalized = _normalized(question)
    if _STATUS_ATTRIBUTE_PATTERN.search(normalized):
        return False
    has_current_signal = bool(
        _CURRENT_PATTERN.search(normalized) or _ACTIVE_CURRENT_PATTERN.search(normalized)
    )
    if not (has_current_signal and _POLICY_CONTEXT_PATTERN.search(normalized)):
        return False
    if _EXPLICIT_POLICY_PATTERN.search(question or ""):
        return False
    if _HISTORICAL_PATTERN.search(normalized):
        return False
    if _COMPARISON_PATTERN.search(normalized):
        return False
    if _is_list_request(normalized):
        return False
    return True


def requires_policy_history(question: str) -> bool:
    """Return true when policy selection must preserve multiple time slices."""

    normalized = _normalized(question)
    if _EXPLICIT_POLICY_PATTERN.search(question or ""):
        return False
    return bool(
        _HISTORICAL_PATTERN.search(normalized)
        or _COMPARISON_PATTERN.search(normalized)
        or _is_list_request(normalized)
    )


def select_current_policy_records(
    question: str,
    records: Sequence[Mapping[str, Any]],
    *,
    reference_date: date,
) -> CurrentPolicySelection:
    """Select the latest effective active policy, or fail closed on ambiguity.

    The selector uses only fields present in the normalized CRM policy schema. It
    never falls back to policy-number ordering or content similarity.
    """

    if not is_current_policy_intent(question):
        return CurrentPolicySelection(
            intent_activated=False,
            selected_records=tuple(records),
            reason="Current-policy intent not activated; candidate set unchanged.",
            reference_date=reference_date,
        )
    policy_records = [record for record in records if _is_policy_record(record)]
    if not policy_records:
        return _failed(reference_date, "No structured policy records were available.")

    customer_records, customer_key, customer_error = _select_customer(question, policy_records)
    if customer_error:
        return _failed(reference_date, customer_error)

    product_records, product_domain, product_error = _select_product(
        question, customer_records
    )
    if product_error:
        return _failed(
            reference_date,
            product_error,
            customer_key=customer_key,
        )

    valid: list[tuple[date, Mapping[str, Any]]] = []
    for record in product_records:
        status = _normalized(_field(record, "status"))
        start = _parse_date(_field(record, "start_date", "startDate"))
        end = _parse_date(_field(record, "end_date", "endDate"))
        if status not in {"active", "aktiv"} or start is None or end is None:
            continue
        if start <= reference_date <= end:
            valid.append((start, record))
    if not valid:
        return _failed(
            reference_date,
            "No active policy had complete dates covering the reference date.",
            customer_key=customer_key,
            product_domain=product_domain,
        )

    latest_start = max(start for start, _ in valid)
    latest = [record for start, record in valid if start == latest_start]
    unique_numbers = {
        str(_field(record, "policy_number", "policyNumber") or "").strip()
        for record in latest
    }
    if len(latest) != 1 or not next(iter(unique_numbers), ""):
        return _failed(
            reference_date,
            "Latest effective policy is ambiguous or lacks a policy number.",
            customer_key=customer_key,
            product_domain=product_domain,
        )
    return CurrentPolicySelection(
        intent_activated=True,
        selected_records=(latest[0],),
        reason="Selected active policy with the latest start date covering the reference date.",
        reference_date=reference_date,
        customer_key=customer_key,
        product_domain=product_domain,
    )


def _failed(
    reference_date: date,
    reason: str,
    *,
    customer_key: str | None = None,
    product_domain: str | None = None,
) -> CurrentPolicySelection:
    return CurrentPolicySelection(
        intent_activated=True,
        selected_records=(),
        reason=reason,
        reference_date=reference_date,
        customer_key=customer_key,
        product_domain=product_domain,
    )


def _is_policy_record(record: Mapping[str, Any]) -> bool:
    entity_type = _normalized(_field(record, "entity_type", "entityType"))
    if entity_type and entity_type != "policy":
        return False
    return bool(_field(record, "policy_number", "policyNumber"))


def _select_customer(
    question: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], str | None, str | None]:
    keyed: dict[str, list[Mapping[str, Any]]] = {}
    unkeyed: list[Mapping[str, Any]] = []
    for record in records:
        customer = str(
            _field(record, "customer_name", "customerName", "customer") or ""
        ).strip()
        if customer:
            keyed.setdefault(_normalized(customer), []).append(record)
        else:
            unkeyed.append(record)
    if not keyed:
        # CRM get_customer_policies is already customer-scoped.
        return list(records), None, None
    question_text = _normalized(question)
    matches = [key for key in keyed if key and key in question_text]
    if len(matches) != 1:
        return [], None, "Customer identity was missing or ambiguous in the candidate metadata."
    return keyed[matches[0]], matches[0], None


def _select_product(
    question: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], str | None, str | None]:
    question_domains = _domains(_normalized(question))
    if len(question_domains) > 1:
        return [], None, "Multiple product domains were requested."
    record_domains: dict[str, list[Mapping[str, Any]]] = {}
    unknown: list[Mapping[str, Any]] = []
    for record in records:
        text = _normalized(
            " ".join(
                str(_field(record, key) or "")
                for key in ("product_type", "productType", "name")
            )
        )
        domains = _domains(text)
        if len(domains) == 1:
            record_domains.setdefault(next(iter(domains)), []).append(record)
        else:
            unknown.append(record)
    if question_domains:
        domain = next(iter(question_domains))
        matching = record_domains.get(domain, [])
        if not matching:
            return [], domain, "No policy matched the requested product domain."
        return matching, domain, None
    if len(record_domains) == 1 and not unknown:
        domain = next(iter(record_domains))
        return record_domains[domain], domain, None
    return [], None, "Product identity was missing or ambiguous in the candidate metadata."


def _domains(text: str) -> set[str]:
    return {
        domain
        for domain, terms in _PRODUCT_DOMAIN_TERMS.items()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) for term in terms)
    }


def _is_list_request(normalized_question: str) -> bool:
    return bool(
        _LIST_COMMAND_PATTERN.search(normalized_question)
        and _PLURAL_POLICY_PATTERN.search(normalized_question)
    )


def _field(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None

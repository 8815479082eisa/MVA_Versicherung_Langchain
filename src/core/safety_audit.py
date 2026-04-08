from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from langchain_core.documents import Document

try:
    from config.models import SafetyConfig
except Exception:
    from src.config.models import SafetyConfig

RiskLevel = Literal["low", "medium", "high"]
SafetyAction = Literal["allow", "redact", "block", "fallback"]


@dataclass(frozen=True)
class SafetyResult:
    allow: bool
    risk_level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    sanitized_answer: Optional[str] = None
    action: SafetyAction = "allow"
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PIIItem:
    pii_type: str
    value: str
    start: int
    end: int
    source: str
    placeholder: str
    allowed: bool = False
    reason: str = ""


# Regex definitions
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+|00)?(?:\d[\d\s()./-]{5,}\d)(?!\w)"
)
IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])", re.IGNORECASE)

IDENTIFIER_LABEL_RE = re.compile(
    r"""(?ix)
    \b(?P<label>
        customer(?:\s*(?:no\.?|nr\.?|number|id))?
        |kunde(?:\s*(?:nr\.?|nummer|id))?
        |kunden(?:\s*(?:nr\.?|nummer|id))?
        |policy(?:\s*(?:no\.?|nr\.?|number|id))?
        |policynumber
        |claim(?:\s*(?:no\.?|nr\.?|number|id))?
        |schaden(?:\s*(?:nr\.?|nummer|id))?
        |contract(?:\s*(?:no\.?|nr\.?|number|id))?
        |vertrags(?:\s*(?:nr\.?|nummer|id))?
        |member(?:\s*(?:no\.?|nr\.?|number|id))?
        |account(?:\s*(?:no\.?|nr\.?|number|id))?
        |reference(?:\s*(?:no\.?|nr\.?|number|id))?
        |id(?:\s*(?:no\.?|nr\.?|number))?
    )\b\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9/-]{4,24})
    """,
)
FORMATTED_IDENTIFIER_RE = re.compile(
    r"""(?ix)
    (?<!\w)
    (?P<value>(?:[A-Z]{2,8}[-/])\d{4,12}(?:[-/][A-Z0-9]{2,8})?)
    (?!\w)
    """,
)
DOB_LABEL_RE = re.compile(
    r"""(?ix)
    \b(?:date of birth|dob|birth date|born|geburtsdatum|geb\.?\s*datum)\b
    \s*[:#-]?\s*
    (?P<value>
        (?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4})
        |(?:\d{4}[./-]\d{1,2}[./-]\d{1,2})
        |(?:\d{1,2}\s+[A-ZÄÖÜa-zäöüß]{3,12}\s+\d{2,4})
        |(?:[A-ZÄÖÜa-zäöüß]{3,12}\s+\d{1,2},?\s+\d{2,4})
    )
    """,
)
ADDRESS_LABEL_RE = re.compile(
    r"""(?ix)
    \b(?:address|anschrift|billing address|shipping address|street address|postal address|postanschrift|wohnanschrift)\b
    \s*[:#-]?\s*
    (?P<value>[^\n;]{8,120})
    """,
)
ADDRESS_FORMAT_RE = re.compile(
    r"""(?ix)
    (?:
        \d{1,4}\s+[A-ZÄÖÜa-zäöüß][\wÄÖÜäöüß.-]+(?:\s+[A-ZÄÖÜa-zäöüß][\wÄÖÜäöüß.-]+){0,3}\s+
        (?:str\.?|straße|strasse|street|st\.|road|rd\.|avenue|ave\.|lane|ln\.|weg|gasse|platz|allee|boulevard|blvd\.|ring)
        (?:\s+\d+[A-Za-z]?)?
        (?:,\s*\d{4,5}\s+[A-ZÄÖÜa-zäöüß][\wÄÖÜäöüß.-]+)?
    )
    |
    (?:
        \d{5}\s+[A-ZÄÖÜa-zäöüß][\wÄÖÜäöüß.-]+
    )
    """,
)

HARD_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"ignore (all|previous|prior) (instructions|prompts)", re.IGNORECASE)),
    ("ignore_safety_rules", re.compile(r"ignore all safety rules", re.IGNORECASE)),
    ("ignore_role", re.compile(r"\bignore (?:your|the) role\b", re.IGNORECASE)),
    ("reveal_prompt", re.compile(r"reveal (the )?(system|developer) prompt", re.IGNORECASE)),
    ("reveal_developer_instructions", re.compile(r"reveal (the )?(developer|internal) instructions", re.IGNORECASE)),
    ("reveal_hidden_prompt", re.compile(r"reveal (the )?(system instructions|hidden prompt|internal hidden prompt)", re.IGNORECASE)),
    ("developer_mode_override", re.compile(r"you are now (in )?(developer|system) mode", re.IGNORECASE)),
    ("role_override", re.compile(r"you are no longer a retrieval assistant", re.IGNORECASE)),
    ("bypass_policy", re.compile(r"bypass (safety|guardrails|policy)", re.IGNORECASE)),
    (
        "retrieval_bypass",
        re.compile(
            r"\bdo not use\b[\s\S]{0,30}\b(?:retrieved|provided|given)\b[\s\S]{0,20}\b(?:documents?|sources?|context)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "context_bypass_general_knowledge",
        re.compile(
            r"\b(?:answer|respond|reply)\b[\s\S]{0,80}\bgeneral knowledge\b[\s\S]{0,80}\b(?:not|instead of)\b[\s\S]{0,80}\b(?:provided|given|retrieved)\b[\s\S]{0,40}\b(?:context|documents?|sources?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "context_bypass_ignore_context",
        re.compile(
            r"\b(?:ignore|don't use|do not use)\b[\s\S]{0,40}\b(?:provided|given|retrieved)\b[\s\S]{0,40}\b(?:context|documents?|sources?)\b",
            re.IGNORECASE,
        ),
    ),
    ("ungrounded_answer_request", re.compile(r"answer only from your own knowledge", re.IGNORECASE)),
    (
        "history_override",
        re.compile(
            r"\b(?:earlier|previously|before)\b[\s\S]{0,80}\bignore\b[\s\S]{0,40}\b(?:role|instructions?)\b[\s\S]{0,120}\b(?:answer|respond)\b[\s\S]{0,40}\b(?:freely|without)\b",
            re.IGNORECASE,
        ),
    ),
    ("jailbreak", re.compile(r"jailbreak", re.IGNORECASE)),
)

SOFT_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("without_sources", re.compile(r"without sources", re.IGNORECASE)),
)

# Backward-compatible alias used by answer-safety checks.
INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(pattern for _, pattern in HARD_INJECTION_PATTERNS)

UNSAFE_CONTENT_PATTERNS = [
    re.compile(r"\bbuild (a )?(bomb|explosive)\b", re.IGNORECASE),
    re.compile(r"\bhow to (hack|phish|steal)\b", re.IGNORECASE),
    re.compile(r"\bmalware|ransomware\b", re.IGNORECASE),
]

STOPWORDS = {
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
}

PII_PLACEHOLDERS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "iban": "[REDACTED_IBAN]",
    "policy_id": "[REDACTED_POLICY_ID]",
    "claim_id": "[REDACTED_CLAIM_ID]",
    "contract_id": "[REDACTED_CONTRACT_ID]",
    "customer_number": "[REDACTED_CUSTOMER_NUMBER]",
    "generic_id": "[REDACTED_ID]",
    "date_of_birth": "[REDACTED_DATE_OF_BIRTH]",
    "address": "[REDACTED_ADDRESS]",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {token for token in tokens if token not in STOPWORDS}


def _extract_texts(docs: Iterable[Any]) -> List[str]:
    texts: List[str] = []
    for doc in docs or []:
        if isinstance(doc, Document):
            texts.append(doc.page_content or "")
        elif isinstance(doc, dict):
            texts.append(str(doc.get("page_content", "")))
        else:
            texts.append(str(doc))
    return texts


def _contains_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def _contains_unsafe_content(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNSAFE_CONTENT_PATTERNS)


def _find_matching_injection_patterns(
    text: str,
    patterns: Sequence[Tuple[str, re.Pattern[str]]],
) -> List[str]:
    matches: List[str] = []
    for name, pattern in patterns:
        if pattern.search(text):
            matches.append(name)
    return matches


def _detect_injection_signals(text: str) -> Dict[str, List[str]]:
    return {
        "hard": _find_matching_injection_patterns(text, HARD_INJECTION_PATTERNS),
        "soft": _find_matching_injection_patterns(text, SOFT_INJECTION_PATTERNS),
    }


def _config_values(config: Optional[SafetyConfig], attribute: str) -> Tuple[str, ...]:
    if config is None:
        return ()
    raw = getattr(config, attribute, ())
    if raw is None:
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _normalize_domain(value: str) -> str:
    return value.strip().lower().lstrip(".")


def _normalize_phone(value: str) -> str:
    return re.sub(r"\D", "", value)


def _placeholder_for_pii_type(pii_type: str) -> str:
    return PII_PLACEHOLDERS.get(pii_type, "[REDACTED_PII]")


def _classify_identifier(value: str, label: str = "") -> Tuple[str, str]:
    label_norm = label.lower()
    value_norm = value.upper()
    prefix = re.split(r"[-/]", value_norm, maxsplit=1)[0]

    if "policy" in label_norm or prefix in {"POL", "POLICY"}:
        return "policy_id", PII_PLACEHOLDERS["policy_id"]
    if "claim" in label_norm or "schaden" in label_norm or prefix in {"CLM", "CLAIM"}:
        return "claim_id", PII_PLACEHOLDERS["claim_id"]
    if "contract" in label_norm or "vertrag" in label_norm or prefix in {"CTR", "CONTRACT", "VTR"}:
        return "contract_id", PII_PLACEHOLDERS["contract_id"]
    if "customer" in label_norm or "kunde" in label_norm or "account" in label_norm or prefix in {"CUST", "CUSTOMER", "KND", "KUNDE"}:
        return "customer_number", PII_PLACEHOLDERS["customer_number"]
    return "generic_id", PII_PLACEHOLDERS["generic_id"]


def _is_allowed_email(value: str, config: Optional[SafetyConfig]) -> Tuple[bool, str]:
    normalized_email = _normalize_email(value)
    allowed_emails = {_normalize_email(item) for item in _config_values(config, "pii_allowed_emails")}
    if normalized_email in allowed_emails:
        return True, "approved_email"

    domain = normalized_email.rsplit("@", 1)[-1] if "@" in normalized_email else ""
    allowed_domains = {_normalize_domain(item) for item in _config_values(config, "pii_allowed_domains")}
    if domain and domain in allowed_domains:
        return True, "approved_domain"

    return False, ""


def _is_allowed_phone(value: str, config: Optional[SafetyConfig]) -> Tuple[bool, str]:
    normalized_phone = _normalize_phone(value)
    if not normalized_phone:
        return False, ""

    allowed_numbers = {_normalize_phone(item) for item in _config_values(config, "pii_allowed_phone_numbers")}
    if normalized_phone in allowed_numbers:
        return True, "approved_phone_number"

    allowed_prefixes = tuple(_normalize_phone(item) for item in _config_values(config, "pii_allowed_phone_prefixes"))
    for prefix in allowed_prefixes:
        if prefix and normalized_phone.startswith(prefix):
            return True, "approved_phone_prefix"

    return False, ""


def _is_valid_phone_match(value: str) -> bool:
    digits = _normalize_phone(value)
    if not 7 <= len(digits) <= 15:
        return False
    if value.startswith("+") or value.startswith("00"):
        return True
    if any(separator in value for separator in (" ", "-", "(", ")", ".", "/")):
        return True
    return False


def _pii_item(
    pii_type: str,
    value: str,
    start: int,
    end: int,
    source: str,
    *,
    allowed: bool = False,
    reason: str = "",
) -> PIIItem:
    return PIIItem(
        pii_type=pii_type,
        value=value,
        start=start,
        end=end,
        source=source,
        placeholder=_placeholder_for_pii_type(pii_type),
        allowed=allowed,
        reason=reason,
    )


def _detect_email_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    items: List[PIIItem] = []
    for match in EMAIL_RE.finditer(text):
        value = match.group(0)
        allowed, allow_reason = _is_allowed_email(value, config)
        reason = allow_reason or "email_pattern"
        items.append(
            _pii_item(
                "email",
                value,
                match.start(),
                match.end(),
                "email_regex",
                allowed=allowed,
                reason=reason,
            )
        )
    return items


def _detect_phone_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    items: List[PIIItem] = []
    for match in PHONE_RE.finditer(text):
        value = match.group(0).strip()
        if not _is_valid_phone_match(value):
            continue
        allowed, allow_reason = _is_allowed_phone(value, config)
        reason = allow_reason or "phone_pattern"
        items.append(
            _pii_item(
                "phone",
                value,
                match.start(),
                match.end(),
                "phone_regex",
                allowed=allowed,
                reason=reason,
            )
        )
    return items


def _detect_iban_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in IBAN_RE.finditer(text):
        value = match.group(0)
        items.append(
            _pii_item(
                "iban",
                value,
                match.start(),
                match.end(),
                "iban_regex",
                reason="iban_pattern",
            )
        )
    return items


def _detect_identifier_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []

    for match in IDENTIFIER_LABEL_RE.finditer(text):
        label = match.group("label")
        value = match.group("value")
        pii_type, placeholder = _classify_identifier(value, label)
        items.append(
            PIIItem(
                pii_type=pii_type,
                value=value,
                start=match.start("value"),
                end=match.end("value"),
                source="identifier_label_regex",
                placeholder=placeholder,
                reason=f"identifier_label:{label.lower()}",
            )
        )

    for match in FORMATTED_IDENTIFIER_RE.finditer(text):
        value = match.group("value")
        pii_type, placeholder = _classify_identifier(value)
        items.append(
            PIIItem(
                pii_type=pii_type,
                value=value,
                start=match.start("value"),
                end=match.end("value"),
                source="identifier_format_regex",
                placeholder=placeholder,
                reason=f"identifier_format:{value.split('-', 1)[0].split('/', 1)[0].lower()}",
            )
        )

    return items


def _detect_dob_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in DOB_LABEL_RE.finditer(text):
        value = match.group("value")
        items.append(
            _pii_item(
                "date_of_birth",
                value,
                match.start("value"),
                match.end("value"),
                "dob_label_regex",
                reason="date_of_birth_label",
            )
        )
    return items


def _detect_address_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in ADDRESS_LABEL_RE.finditer(text):
        value = match.group("value").strip()
        items.append(
            _pii_item(
                "address",
                value,
                match.start("value"),
                match.end("value"),
                "address_label_regex",
                reason="address_label",
            )
        )

    for match in ADDRESS_FORMAT_RE.finditer(text):
        value = match.group(0).strip()
        items.append(
            _pii_item(
                "address",
                value,
                match.start(),
                match.end(),
                "address_format_regex",
                reason="address_format",
            )
        )

    return items


def detect_pii(text: str, config: Optional[SafetyConfig] = None) -> List[PIIItem]:
    if not text:
        return []

    items: List[PIIItem] = []
    items.extend(_detect_email_items(text, config))
    items.extend(_detect_phone_items(text, config))
    items.extend(_detect_iban_items(text, config))
    items.extend(_detect_identifier_items(text, config))
    items.extend(_detect_dob_items(text, config))
    items.extend(_detect_address_items(text, config))
    return _dedupe_pii_items(items)


def is_allowed_pii(item: PIIItem, config: Optional[SafetyConfig] = None) -> bool:
    if item.allowed:
        return True
    if item.pii_type == "email":
        return _is_allowed_email(item.value, config)[0]
    if item.pii_type == "phone":
        return _is_allowed_phone(item.value, config)[0]
    return False


def _dedupe_pii_items(items: Sequence[PIIItem]) -> List[PIIItem]:
    deduped: List[PIIItem] = []
    seen: set[Tuple[int, int, str, bool]] = set()
    for item in sorted(items, key=lambda candidate: (candidate.start, -(candidate.end - candidate.start), candidate.pii_type, candidate.source)):
        key = (item.start, item.end, item.pii_type, item.allowed)
        if key in seen:
            continue
        seen.add(key)
        if deduped and item.start < deduped[-1].end:
            continue
        deduped.append(item)
    return deduped


def _select_sensitive_items(items: Sequence[PIIItem], config: Optional[SafetyConfig] = None) -> List[PIIItem]:
    sensitive_items: List[PIIItem] = []
    for item in items:
        if is_allowed_pii(item, config):
            continue
        sensitive_items.append(item)
    return sensitive_items


def _apply_span_replacements(text: str, replacements: Sequence[PIIItem]) -> str:
    if not replacements:
        return text

    ordered = sorted(replacements, key=lambda item: item.start)
    pieces: List[str] = []
    cursor = 0
    for item in ordered:
        if item.start < cursor:
            continue
        pieces.append(text[cursor:item.start])
        pieces.append(item.placeholder)
        cursor = item.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def sanitize_pii(text: str, items: Sequence[PIIItem], config: Optional[SafetyConfig] = None) -> str:
    sensitive_items = _select_sensitive_items(items, config)
    if not sensitive_items:
        return text
    return _apply_span_replacements(text, sensitive_items)


def _summarize_pii_items(items: Sequence[PIIItem]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in items:
        summary[item.pii_type] = summary.get(item.pii_type, 0) + 1
    return summary


def _allowed_contact_items(items: Sequence[PIIItem], config: Optional[SafetyConfig]) -> List[PIIItem]:
    return [item for item in items if is_allowed_pii(item, config)]


def _redacted_contact_items(items: Sequence[PIIItem], config: Optional[SafetyConfig]) -> List[PIIItem]:
    return [item for item in items if not is_allowed_pii(item, config)]


def _append_pii_reasons(
    reasons: List[str],
    *,
    stage: str,
    items: Sequence[PIIItem],
    config: Optional[SafetyConfig] = None,
    include_redacted_reason: bool = True,
) -> None:
    allowed_items = _allowed_contact_items(items, config)
    redacted_items = _redacted_contact_items(items, config)

    if allowed_items:
        reasons.append("pii_allowed_business_contact")

    if not redacted_items:
        return

    if stage == "answer":
        reasons.append("response_contains_pii")
    elif stage == "query":
        reasons.append("query_contains_pii")
    elif stage == "context":
        reasons.append("context_contains_pii")

    if include_redacted_reason:
        reasons.append(f"pii_{stage}_redacted")

    for pii_type in sorted({item.pii_type for item in redacted_items}):
        reasons.append(f"pii_{pii_type}_redacted")


def _pii_details(items: Sequence[PIIItem], config: Optional[SafetyConfig] = None) -> Dict[str, Any]:
    allowed_items = _allowed_contact_items(items, config)
    redacted_items = _redacted_contact_items(items, config)
    return {
        "detected_count": len(items),
        "allowed_count": len(allowed_items),
        "redacted_count": len(redacted_items),
        "detected_types": _summarize_pii_items(items),
        "allowed_types": _summarize_pii_items(allowed_items),
        "redacted_types": _summarize_pii_items(redacted_items),
        "items": [
            {
                "pii_type": item.pii_type,
                "start": item.start,
                "end": item.end,
                "allowed": item.allowed or is_allowed_pii(item, config),
                "source": item.source,
                "reason": item.reason,
            }
            for item in items
        ],
    }


def _mark_pii_items_with_allowlist(items: Sequence[PIIItem], config: Optional[SafetyConfig]) -> List[PIIItem]:
    marked: List[PIIItem] = []
    for item in items:
        if is_allowed_pii(item, config):
            allowed = True
            if item.pii_type == "email":
                allowed_reason = _is_allowed_email(item.value, config)[1]
            elif item.pii_type == "phone":
                allowed_reason = _is_allowed_phone(item.value, config)[1]
            else:
                allowed_reason = item.reason
            marked.append(
                PIIItem(
                    pii_type=item.pii_type,
                    value=item.value,
                    start=item.start,
                    end=item.end,
                    source=item.source,
                    placeholder=item.placeholder,
                    allowed=allowed,
                    reason=allowed_reason or item.reason,
                )
            )
        else:
            marked.append(item)
    return marked


def _contains_pii(text: str, config: Optional[SafetyConfig] = None) -> bool:
    return bool(_redacted_contact_items(detect_pii(text, config), config))


def _primary_query_decision_source(
    *,
    hard_signals: Sequence[str],
    soft_signals: Sequence[str],
    redacted_pii_items: Sequence[PIIItem],
    allowed_pii_items: Sequence[PIIItem],
    unsafe_content: bool,
    action: SafetyAction,
) -> str:
    if unsafe_content:
        return "unsafe_content"
    if hard_signals and action == "block":
        return "injection_hard_rule"
    if redacted_pii_items and action == "fallback":
        return "pii_rule"
    if soft_signals:
        return "injection_soft_rule"
    if allowed_pii_items:
        return "allowed_business_contact"
    return "allow"


class SafetyAuditLayer:
    def __init__(self, config: SafetyConfig):
        self.config = config

    @property #check if the safety audit is active
    def is_active(self) -> bool:
        return self.config.enabled and self.config.mode != "off"

    (self, query: str, chat_history: Optional[List[dict]] = None) -> SafetyResult:
        del chat_history
        reasons: List[str] = []
        action: SafetyAction = "allow"
        allow = True
        risk: RiskLevel = "low"
        text = query or ""

        injection_signals = _detect_injection_signals(text)
        hard_signals = injection_signals["hard"]
        soft_signals = injection_signals["soft"]

        if hard_signals:
            reasons.append("prompt_injection_pattern_detected")
            reasons.extend(f"injection_hard_{name}" for name in hard_signals)
            risk = "high"
            if self.config.block_injection:
                action = "block"
                allow = False
        if soft_signals:
            reasons.append("suspicious_query_pattern_detected")
            reasons.extend(f"injection_soft_{name}" for name in soft_signals)
            if risk == "low":
                risk = "medium"

        pii_items = _mark_pii_items_with_allowlist(detect_pii(text, self.config), self.config)
        redacted_pii_items = _redacted_contact_items(pii_items, self.config)
        allowed_pii_items = _allowed_contact_items(pii_items, self.config)
        unsafe_check_details: Dict[str, Any] = {}
        unsafe_content_detected = False
        try:
            unsafe_contedef check_query_safetynt_detected = _contains_unsafe_content(text)
        except Exception as exc:
            # Keep precheck deterministic even if a helper breaks.
            reasons.append("unsafe_content_check_error")
            unsafe_check_details["unsafe_check_error"] = str(exc)

        if self.config.block_pii and redacted_pii_items:
            reasons.append("pii_detected_in_query")
            _append_pii_reasons(reasons, stage="query", items=pii_items, config=self.config)
            action = "fallback" if allow else action
            allow = False
            risk = "high" if risk == "high" else "medium"
        elif allowed_pii_items:
            reasons.append("pii_allowed_business_contact")

        if unsafe_content_detected:
            reasons.append("unsafe_request_pattern_detected")
            action = "block"
            allow = False
            risk = "high"

        result = SafetyResult(
            allow=allow,
            risk_level=risk,
            reasons=reasons,
            action=action,
            scores={
                "query_pii_hits": float(len(redacted_pii_items)),
                "query_allowed_pii_hits": float(len(allowed_pii_items)),
                "query_injection_hits": float(len(hard_signals)),
                "query_suspicious_hits": float(len(soft_signals)),
                "query_length": float(len(text)),
                "query_token_count": float(len(_tokenize(text))),
            },
            details={
                "stage": "pre_query",
                "decision": {
                    "allow": allow,
                    "action": action,
                    "risk_level": risk,
                    "source": _primary_query_decision_source(
                        hard_signals=hard_signals,
                        soft_signals=soft_signals,
                        redacted_pii_items=redacted_pii_items,
                        allowed_pii_items=allowed_pii_items,
                        unsafe_content=unsafe_content_detected,
                        action=action,
                    ),
                },
                "pii": _pii_details(pii_items, self.config),
                "injection": {
                    "hard": hard_signals,
                    "soft": soft_signals,
                },
                "query": {
                    "length": len(text),
                    "token_count": len(_tokenize(text)),
                    "contains_sensitive_pii": bool(redacted_pii_items),
                    "contains_allowed_pii": bool(allowed_pii_items),
                    "contains_hard_injection": bool(hard_signals),
                    "contains_soft_injection": bool(soft_signals),
                },
                **unsafe_check_details,
            },
        )
        return self._mode_adjust(result)

    def check_context_safety(self, docs: List[Any]) -> SafetyResult:
        reasons: List[str] = []
        action: SafetyAction = "allow"
        allow = True
        risk: RiskLevel = "low"
        texts = _extract_texts(docs)
        pii_items: List[PIIItem] = []

        if self.config.block_pii:
            for text in texts:
                pii_items.extend(detect_pii(text, self.config))
            pii_items = _mark_pii_items_with_allowlist(pii_items, self.config)
            redacted_items = _redacted_contact_items(pii_items, self.config)
            if redacted_items:
                reasons.append("pii_detected_in_context")
                _append_pii_reasons(reasons, stage="context", items=pii_items, config=self.config)
                action = "redact"
                allow = False
                risk = "medium"
            elif _allowed_contact_items(pii_items, self.config):
                reasons.append("pii_allowed_business_contact")

        result = SafetyResult(
            allow=allow,
            risk_level=risk,
            reasons=reasons,
            action=action,
            scores={
                "context_pii_hits": float(len(_redacted_contact_items(pii_items, self.config))),
                "context_allowed_pii_hits": float(len(_allowed_contact_items(pii_items, self.config))),
            },
            details={
                "stage": "context",
                "pii": _pii_details(pii_items, self.config),
            },
        )
        return self._mode_adjust(result)

    def check_answer_safety(self, query: str, docs: List[Any], answer: str) -> SafetyResult:
        del query
        reasons: List[str] = []
        action: SafetyAction = "allow"
        allow = True
        risk: RiskLevel = "low"
        scores: Dict[str, float] = {}
        sanitized_answer: Optional[str] = None

        answer_text = answer or ""
        context_text = "\n".join(_extract_texts(docs))
        pii_items = _mark_pii_items_with_allowlist(detect_pii(answer_text, self.config), self.config)
        sensitive_items = _redacted_contact_items(pii_items, self.config)

        if self.config.block_pii and sensitive_items:
            reasons.append("pii_detected_in_answer")
            _append_pii_reasons(reasons, stage="answer", items=pii_items, config=self.config)
            action = "redact"
            allow = False
            risk = "high"
            sanitized_answer = sanitize_pii(answer_text, pii_items, self.config)
        elif _allowed_contact_items(pii_items, self.config):
            reasons.append("pii_allowed_business_contact")

        if self.config.block_injection and _contains_injection(answer_text):
            reasons.append("prompt_injection_signal_in_answer")
            action = "fallback"
            allow = False
            risk = "high"

        groundedness = self._groundedness_score(answer_text, context_text)
        scores["groundedness"] = groundedness
        if groundedness < self.config.min_groundedness:
            reasons.append("low_groundedness")
            if action == "allow":
                action = "fallback"
            allow = False
            risk = "medium" if risk == "low" else risk

        result = SafetyResult(
            allow=allow,
            risk_level=risk,
            reasons=reasons,
            sanitized_answer=sanitized_answer,
            action=action,
            scores=scores,
            details={
                "stage": "post_generation",
                "pii": _pii_details(pii_items, self.config),
            },
        )
        return self._mode_adjust(result)

    def apply_safety_action(self, result: SafetyResult, answer: str) -> str:
        if result.action == "allow":
            return answer
        if result.action == "redact":
            if result.sanitized_answer:
                return result.sanitized_answer
            return self._redact_pii(answer)
        if result.action in {"fallback", "block"}:
            return self.config.fallback_text
        return answer

    def _mode_adjust(self, result: SafetyResult) -> SafetyResult:
        if not self.is_active:
            return SafetyResult(
                allow=True,
                risk_level="low",
                reasons=[],
                action="allow",
                scores=result.scores,
                details={**result.details, "mode": self.config.mode},
            )
        if self.config.mode == "monitor":
            return SafetyResult(
                allow=True,
                risk_level=result.risk_level,
                reasons=result.reasons,
                sanitized_answer=result.sanitized_answer,
                action="allow",
                scores=result.scores,
                details={**result.details, "mode": "monitor", "would_action": result.action},
            )
        return result

    def _redact_pii(self, text: str) -> str:
        items = _mark_pii_items_with_allowlist(detect_pii(text, self.config), self.config)
        return sanitize_pii(text, items, self.config)

    def _groundedness_score(self, answer: str, context: str) -> float:
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 0.0
        if not context_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return overlap / max(len(answer_tokens), 1)

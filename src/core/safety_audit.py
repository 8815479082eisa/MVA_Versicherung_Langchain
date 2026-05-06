from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+|00)?(?:\d[\d\s()./-]{5,}\d)(?!\w)")
IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])", re.IGNORECASE)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PAYMENT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

IDENTIFIER_LABEL_RE = re.compile(
    r"""(?ix)
    \b(?P<label>
        customer(?:\s*(?:no\.?|nr\.?|number|id))
        |kunde(?:\s*(?:nr\.?|nummer|id))
        |kunden(?:\s*(?:nr\.?|nummer|id))
        |policy(?:\s*(?:no\.?|nr\.?|number|id))
        |policynumber
        |claim(?:\s*(?:no\.?|nr\.?|number|id))
        |schaden(?:\s*(?:nr\.?|nummer|id))
        |contract(?:\s*(?:no\.?|nr\.?|number|id))
        |vertrags(?:\s*(?:nr\.?|nummer|id))
        |member(?:\s*(?:no\.?|nr\.?|number|id))
        |account(?:\s*(?:no\.?|nr\.?|number|id))
        |reference(?:\s*(?:no\.?|nr\.?|number|id))
        |id(?:\s*(?:no\.?|nr\.?|number))
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
    ("ignore_instructions", re.compile(r"\bignore\b[\s\S]{0,60}\b(?:instructions?|prompts?|rules?)\b", re.IGNORECASE)),
    ("disregard_instructions", re.compile(r"\bdisregard\b[\s\S]{0,60}\b(?:instructions?|prompts?|rules?)\b", re.IGNORECASE)),
    (
        "override_prior_instructions",
        re.compile(
            r"\boverride\b[\s\S]{0,60}\b(?:all|any|previous|prior|existing)\b[\s\S]{0,60}\b(?:instructions?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    ("ignore_safety_rules", re.compile(r"ignore all safety rules", re.IGNORECASE)),
    (
        "ignore_safety_controls",
        re.compile(
            r"\b(?:ignore|disregard|bypass|disable|drop|remove)\b[\s\S]{0,40}\b(?:safety|guardrails?|policy|compliance|constraints?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "follow_only_message_bypass_rules",
        re.compile(
            r"\b(?:follow|obey)\b[\s\S]{0,40}\b(?:only|just)\b[\s\S]{0,40}\b(?:this message|this prompt|these instructions)\b[\s\S]{0,80}\b(?:bypass|ignore|override)\b[\s\S]{0,40}\b(?:rules?|policy|guardrails?|constraints?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_escalation_role_request",
        re.compile(
            r"\b(?:act|behave)\s+(?:as|like)\s+(?:an?\s+)?(?:administrator|admin|internal operator|privileged user|unrestricted assistant)\b",
            re.IGNORECASE,
        ),
    ),
    ("ignore_role", re.compile(r"\b(?:ignore|override)\s+(?:your|the)\s+role\b", re.IGNORECASE)),
    ("reveal_prompt", re.compile(r"\breveal\b[\s\S]{0,40}\b(?:system|developer|internal)\b[\s\S]{0,20}\b(?:prompt|template|settings?)\b", re.IGNORECASE)),
    (
        "print_or_show_hidden_prompt",
        re.compile(
            r"\b(?:print|show|display)\b[\s\S]{0,40}\b(?:hidden|system|developer|internal)\b[\s\S]{0,30}\b(?:prompt|template|settings?)\b",
            re.IGNORECASE,
        ),
    ),
    ("reveal_developer_instructions", re.compile(r"\breveal\b[\s\S]{0,40}\b(?:developer|internal)\b[\s\S]{0,20}\binstructions?\b", re.IGNORECASE)),
    ("reveal_hidden_prompt", re.compile(r"\breveal\b[\s\S]{0,40}\b(?:system instructions?|hidden prompt|internal hidden prompt)\b", re.IGNORECASE)),
    ("developer_mode_override", re.compile(r"\byou are now\b[\s\S]{0,20}\b(?:developer|system|backdoor)\s+mode\b", re.IGNORECASE)),
    ("role_override", re.compile(r"\byou are no longer a retrieval assistant\b|\bunrestricted (?:mode|assistant)\b", re.IGNORECASE)),
    ("bypass_policy", re.compile(r"\bbypass\b[\s\S]{0,30}\b(?:safety(?: layer)?|guardrails?|policy|constraints?)\b", re.IGNORECASE)),
    ("disable_safety_mode", re.compile(r"\bdisable\b[\s\S]{0,20}\b(?:safety|guardrails?|policy)\b[\s\S]{0,20}\bmode\b", re.IGNORECASE)),
    (
        "policy_checks_disabled",
        re.compile(
            r"\b(?:safety|policy|guardrails?|compliance)\s+checks?\b[\s\S]{0,20}\b(?:are\s+)?disabled\b",
            re.IGNORECASE,
        ),
    ),
    ("dump_audit_logs", re.compile(r"\b(?:dump|print|show)\b[\s\S]{0,30}\baudit logs?\b", re.IGNORECASE)),
    (
        "retrieval_bypass",
        re.compile(
            r"\b(?:do not use|don't use|bypass)\b[\s\S]{0,40}\b(?:retrieval|retrieved|provided|given)\b[\s\S]{0,30}\b(?:documents?|sources?|context)?\b",
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
            r"\b(?:ignore|don't use|do not use)\b[\s\S]{0,40}\b(?:provided|given|retrieved|context)\b[\s\S]{0,40}\b(?:context|documents?|sources?)\b",
            re.IGNORECASE,
        ),
    ),
    ("ungrounded_answer_request", re.compile(r"answer only from your own knowledge|answer freely from private memory", re.IGNORECASE)),
    (
        "no_safeguards_response_request",
        re.compile(
            r"\b(?:respond|answer|output)\b[\s\S]{0,100}\b(?:no|without|do not mention|don't mention)\b[\s\S]{0,40}\b(?:policy(?: references?)?|safeguards?|guardrails?|compliance checks?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "restrictions_removed_simulation",
        re.compile(
            r"\b(?:simulate|pretend|act)\b[\s\S]{0,80}\b(?:all|any)\s+(?:restrictions?|constraints?)\s+(?:removed|off)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "policy_not_applicable_request",
        re.compile(
            r"\b(?:as if|pretend)\b[\s\S]{0,80}\b(?:safety|policy|guardrails?)\b[\s\S]{0,50}\b(?:does not apply|don't apply|is not applicable)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "private_channel_bypass",
        re.compile(
            r"\b(?:private|trusted)\s+channel\b[\s\S]{0,80}\b(?:bypass|ignore|disable)\b[\s\S]{0,40}\b(?:protections?|guardrails?|policy|constraints?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "switch_backdoor_mode",
        re.compile(r"\b(?:switch|enter)\b[\s\S]{0,40}\b(?:developer|backdoor)\s+mode\b", re.IGNORECASE),
    ),
    (
        "authorized_policy_bypass_claim",
        re.compile(
            r"\b(?:assume|pretend)\b[\s\S]{0,60}\b(?:authorized|approved)\b[\s\S]{0,40}\b(?:policy|guardrails?)\b[\s\S]{0,40}\b(?:bypass|disable)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_safety_rules_for_bypass",
        re.compile(
            r"\breveal\b[\s\S]{0,60}\b(?:secret|hidden)?\s*(?:safety|policy)\s+rules?\b[\s\S]{0,80}\b(?:bypass|evade)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "history_override",
        re.compile(
            r"\b(?:earlier|previously|before)\b[\s\S]{0,80}\bignore\b[\s\S]{0,40}\b(?:role|instructions?)\b[\s\S]{0,120}\b(?:answer|respond)\b[\s\S]{0,40}\b(?:freely|without)\b",
            re.IGNORECASE,
        ),
    ),
    ("jailbreak", re.compile(r"\bjailbreak\b|\bno-?filter mode\b|\braw mode\b|\bunrestricted mode\b", re.IGNORECASE)),
)

SENSITIVE_DATA_REQUEST_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "personal_data_exfiltration",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\b(?:raw )?(?:personal data|pii)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "customer_profile_operation",
        re.compile(
            r"\b(?:update|validate|verify|confirm|process|merge|cross-?check|reconcile|identify|review|use|expose)\b[\s\S]{0,80}\b(?:customer|policyholder|claimant|account)\b[\s\S]{0,80}\b(?:data|details?|records?|profile|identity|billing|contact|address|email|phone|payment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "email_or_phone_ownership_lookup",
        re.compile(
            r"\b(?:verify|confirm|check)\b[\s\S]{0,70}\b(?:email|phone)\b[\s\S]{0,70}\b(?:belongs to|linked to|associated with)\b[\s\S]{0,40}\b(?:claimant|customer|policyholder|account)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sensitive_profile_review_request",
        re.compile(
            r"\b(?:review|inspect|analyze)\b[\s\S]{0,60}\b(?:sensitive|private)\s+(?:profile|record)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal_records_linkage_request",
        re.compile(
            r"\b(?:identify|find|list)\b[\s\S]{0,80}\b(?:personal|private)\s+records?\b[\s\S]{0,80}\b(?:linked to|associated with)\b[\s\S]{0,40}\b(?:payment\s+account|account)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "private_data_reconciliation_policyholder",
        re.compile(
            r"\b(?:private|sensitive)\s+data\s+reconciliation\b[\s\S]{0,80}\b(?:policyholder|customer|profile|identity)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "validate_contract_with_sensitive_profile_data",
        re.compile(
            r"\b(?:use|apply)\b[\s\S]{0,60}\b(?:sensitive|private)\s+profile\s+data\b[\s\S]{0,80}\b(?:validate|verify|process)\b[\s\S]{0,40}\b(?:insurance\s+)?contract\b",
            re.IGNORECASE,
        ),
    ),
    (
        "person_existence_lookup",
        re.compile(
            r"\b(?:verify|confirm)\b[\s\S]{0,80}\b(?:this|a)\s+person\b[\s\S]{0,80}\b(?:insured|customer)\b[\s\S]{0,40}\b(?:base|database|records?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "identity_reconciliation_request",
        re.compile(
            r"\b(?:merge|reconcile|cross-?check)\b[\s\S]{0,80}\b(?:customer|identity|profile|record|records?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_recovery_data_exposure",
        re.compile(
            r"\b(?:expose|reveal|provide)\b[\s\S]{0,80}\b(?:customer|personal|private)\b[\s\S]{0,60}\b(?:details?|data)\b[\s\S]{0,60}\b(?:account recovery|billing|contract)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "phone_numbers_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\bphone numbers?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "addresses_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\baddress(?:es)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "social_security_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\b(?:social security numbers?|ssn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "passport_number_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\bpassport numbers?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "iban_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal)\b[\s\S]{0,120}\bibans?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "restricted_customer_records_request",
        re.compile(
            r"\b(?:reveal|show|disclose|expose|print|return|output|dump)\b[\s\S]{0,120}\b(?:restricted|protected|internal)\b[\s\S]{0,40}\bcustomer records?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "customer_names_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal|disclose|expose)\b[\s\S]{0,120}\bcustomer names?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "contract_numbers_request",
        re.compile(
            r"\b(?:show|list|print|return|output|dump|reveal|disclose|expose)\b[\s\S]{0,120}\bcontract numbers?\b",
            re.IGNORECASE,
        ),
    ),
)

PII_PLACEHOLDERS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "iban": "[REDACTED_IBAN]",
    "ssn": "[REDACTED_SSN]",
    "payment_card": "[REDACTED_PAYMENT_CARD]",
    "policy_id": "[REDACTED_POLICY_ID]",
    "claim_id": "[REDACTED_CLAIM_ID]",
    "contract_id": "[REDACTED_CONTRACT_ID]",
    "customer_number": "[REDACTED_CUSTOMER_NUMBER]",
    "generic_id": "[REDACTED_ID]",
    "date_of_birth": "[REDACTED_DATE_OF_BIRTH]",
    "address": "[REDACTED_ADDRESS]",
}


def _find_matching_injection_patterns(
    text: str,
    patterns: Sequence[Tuple[str, re.Pattern[str]]],
) -> List[str]:
    matches: List[str] = []
    for name, pattern in patterns:
        if pattern.search(text):
            matches.append(name)
    return matches


def detect_query_hard_injection_signals(text: str) -> List[str]:
    return _find_matching_injection_patterns(text or "", HARD_INJECTION_PATTERNS)


def detect_sensitive_data_request_signals(text: str) -> List[str]:
    return _find_matching_injection_patterns(text or "", SENSITIVE_DATA_REQUEST_PATTERNS)


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


def _looks_like_identifier_value(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate:
        return False

    digit_count = sum(ch.isdigit() for ch in candidate)
    if digit_count >= 2:
        return True

    if re.search(r"[A-Z]{2,8}[-/]\d{2,}", candidate, re.IGNORECASE):
        return True

    if re.search(r"\d{3,}", candidate):
        return True

    return False


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


def _passes_luhn(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        value = digit
        if idx % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


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
        items.append(
            _pii_item(
                "email",
                value,
                match.start(),
                match.end(),
                "email_regex",
                allowed=allowed,
                reason=allow_reason or "email_pattern",
            )
        )
    return items


def _detect_phone_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    items: List[PIIItem] = []
    for match in PHONE_RE.finditer(text):
        value = match.group(0).strip()
        # Avoid classifying SSN-formatted identifiers as phone numbers.
        if SSN_RE.fullmatch(value):
            continue
        if not _is_valid_phone_match(value):
            continue
        allowed, allow_reason = _is_allowed_phone(value, config)
        items.append(
            _pii_item(
                "phone",
                value,
                match.start(),
                match.end(),
                "phone_regex",
                allowed=allowed,
                reason=allow_reason or "phone_pattern",
            )
        )
    return items


def _detect_iban_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in IBAN_RE.finditer(text):
        items.append(
            _pii_item(
                "iban",
                match.group(0),
                match.start(),
                match.end(),
                "iban_regex",
                reason="iban_pattern",
            )
        )
    return items


def _detect_ssn_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in SSN_RE.finditer(text):
        items.append(
            _pii_item(
                "ssn",
                match.group(0),
                match.start(),
                match.end(),
                "ssn_regex",
                reason="ssn_pattern",
            )
        )
    return items


def _detect_payment_card_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []
    for match in PAYMENT_CARD_RE.finditer(text):
        value = match.group(0).strip()
        digits = _normalize_phone(value)
        if not _passes_luhn(digits):
            continue
        items.append(
            _pii_item(
                "payment_card",
                value,
                match.start(),
                match.end(),
                "payment_card_regex",
                reason="payment_card_pattern",
            )
        )
    return items


def _detect_identifier_items(text: str, config: Optional[SafetyConfig]) -> List[PIIItem]:
    del config
    items: List[PIIItem] = []

    for match in IDENTIFIER_LABEL_RE.finditer(text):
        label = match.group("label")
        value = match.group("value")
        if not _looks_like_identifier_value(value):
            continue
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
        if not _looks_like_identifier_value(value):
            continue
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
        items.append(
            _pii_item(
                "date_of_birth",
                match.group("value"),
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
    items.extend(_detect_ssn_items(text, config))
    items.extend(_detect_payment_card_items(text, config))
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
    return [item for item in items if not is_allowed_pii(item, config)]


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

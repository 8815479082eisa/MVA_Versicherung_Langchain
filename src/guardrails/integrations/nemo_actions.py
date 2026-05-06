from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Iterable, Optional, Sequence

from langchain_core.documents import Document

from nemoguardrails import LLMRails
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

try:
    from config.models import SafetyConfig, load_model_settings
    from core.safety_audit import (
        detect_pii,
        detect_query_hard_injection_signals,
        detect_sensitive_data_request_signals,
        is_allowed_pii,
        sanitize_pii,
    )
except Exception:
    from src.config.models import SafetyConfig, load_model_settings
    from src.core.safety_audit import (
        detect_pii,
        detect_query_hard_injection_signals,
        detect_sensitive_data_request_signals,
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
}


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


def _tokenize_grounding(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {token for token in tokens if token not in GROUNDING_STOPWORDS}


def _groundedness_score(answer: str, docs: Sequence[Document]) -> float:
    answer_tokens = _tokenize_grounding(answer)
    context_tokens = _tokenize_grounding("\n".join(doc.page_content or "" for doc in docs))
    if not answer_tokens or not context_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return overlap / max(len(answer_tokens), 1)


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
    unsafe_matches = _find_matching_patterns(text, UNSAFE_QUERY_PATTERNS)
    pii_items = detect_pii(text, config)
    sensitive_pii_items = _sensitive_pii_items(pii_items, config)
    allowed_pii_items = _allowed_pii_items(pii_items, config)

    if hard_signals:
        reasons.append("prompt_injection_pattern_detected")
        reasons.extend(f"injection_hard_{name}" for name in hard_signals)
        risk = "high"
        if config.block_injection:
            action = "block"
            allow = False

    if sensitive_data_request_signals:
        reasons.append("sensitive_data_request_detected")
        reasons.extend(f"sensitive_data_request_{name}" for name in sensitive_data_request_signals)
        risk = "high"
        if config.block_pii:
            action = "block"
            allow = False

    if config.block_pii and sensitive_pii_items:
        reasons.append("pii_detected_in_query")
        _append_pii_reasons(reasons, stage="query", items=pii_items, config=config)
        if action == "allow":
            action = "fallback"
        allow = False
        if risk == "low":
            risk = "medium"
    elif allowed_pii_items:
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

    if config.block_pii:
        for doc in docs:
            pii_items.extend(detect_pii(doc.page_content or "", config))

    sensitive_items = _sensitive_pii_items(pii_items, config)
    allowed_items = _allowed_pii_items(pii_items, config)
    sanitized_docs_payload: Optional[list[dict[str, Any]]] = None

    if sensitive_items:
        reasons.append("pii_detected_in_context")
        _append_pii_reasons(reasons, stage="context", items=pii_items, config=config)
        action = "redact"
        allow = False
        risk = "medium"
        sanitized_docs, changed = _sanitize_context_docs(docs, config)
        if changed:
            sanitized_docs_payload = _serialize_docs(sanitized_docs)
    elif allowed_items:
        reasons.append("pii_allowed_business_contact")

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
    del query
    answer_text = answer or ""
    reasons: list[str] = []
    action = "allow"
    allow = True
    risk = "low"
    sanitized_answer = answer_text

    pii_items = detect_pii(answer_text, config)
    sensitive_items = _sensitive_pii_items(pii_items, config)
    allowed_items = _allowed_pii_items(pii_items, config)

    if config.block_pii and sensitive_items:
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

    groundedness = _groundedness_score(answer_text, docs)
    if docs and groundedness < config.min_groundedness:
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
                "injection": {
                    "hard": answer_injection_signals,
                    "soft": [],
                },
                "groundedness": {
                    "score": groundedness,
                    "threshold": config.min_groundedness,
                    "enforced": bool(docs),
                },
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

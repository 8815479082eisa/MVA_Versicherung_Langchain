from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

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


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", re.IGNORECASE)
ID_RE = re.compile(r"\b(?:passport|ssn|social security|id number|tax id|national id)\b", re.IGNORECASE)

INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|prior) (instructions|prompts)", re.IGNORECASE),
    re.compile(r"reveal (the )?(system|developer) prompt", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|system) mode", re.IGNORECASE),
    re.compile(r"bypass (safety|guardrails|policy)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]

UNSAFE_CONTENT_PATTERNS = [
    re.compile(r"\bbuild (a )?(bomb|explosive)\b", re.IGNORECASE),
    re.compile(r"\bhow to (hack|phish|steal)\b", re.IGNORECASE),
    re.compile(r"\bmalware|ransomware\b", re.IGNORECASE),
]

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "will", "your",
    "are", "but", "not", "you", "can", "all", "was", "has", "had", "into", "their",
    "they", "them", "there", "about", "what", "when", "where", "which", "why", "how",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {t for t in tokens if t not in STOPWORDS}


def _extract_texts(docs: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for doc in docs or []:
        if isinstance(doc, Document):
            out.append(doc.page_content or "")
        elif isinstance(doc, dict):
            out.append(str(doc.get("page_content", "")))
        else:
            out.append(str(doc))
    return out


def _contains_pii(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or IBAN_RE.search(text)
        or ID_RE.search(text)
    )


def _contains_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def _contains_unsafe_content(text: str) -> bool:
    return any(p.search(text) for p in UNSAFE_CONTENT_PATTERNS)


class SafetyAuditLayer:
    def __init__(self, config: SafetyConfig):
        self.config = config

    @property
    def is_active(self) -> bool:
        return self.config.enabled and self.config.mode != "off"

    def check_query_safety(self, query: str, chat_history: Optional[List[dict]] = None) -> SafetyResult:
        del chat_history
        reasons: List[str] = []
        action: SafetyAction = "allow"
        allow = True
        risk: RiskLevel = "low"
        text = query or ""

        if self.config.block_injection and _contains_injection(text):
            reasons.append("prompt_injection_pattern_detected")
            action = "block"
            allow = False
            risk = "high"

        if self.config.block_pii and _contains_pii(text):
            reasons.append("pii_detected_in_query")
            action = "fallback" if allow else action
            allow = False
            risk = "high" if risk == "high" else "medium"

        if _contains_unsafe_content(text):
            reasons.append("unsafe_request_pattern_detected")
            action = "block"
            allow = False
            risk = "high"

        result = SafetyResult(
            allow=allow,
            risk_level=risk,
            reasons=reasons,
            action=action,
            scores={},
            details={"stage": "pre_query"},
        )
        return self._mode_adjust(result)

    def check_context_safety(self, docs: List[Any]) -> SafetyResult:
        reasons: List[str] = []
        action: SafetyAction = "allow"
        allow = True
        risk: RiskLevel = "low"
        texts = _extract_texts(docs)
        pii_hits = 0

        if self.config.block_pii:
            for text in texts:
                if _contains_pii(text):
                    pii_hits += 1
            if pii_hits > 0:
                reasons.append("pii_detected_in_context")
                action = "redact"
                allow = False
                risk = "medium"

        result = SafetyResult(
            allow=allow,
            risk_level=risk,
            reasons=reasons,
            action=action,
            scores={"context_pii_hits": float(pii_hits)},
            details={"stage": "context"},
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

        if self.config.block_pii and _contains_pii(answer_text):
            reasons.append("pii_detected_in_answer")
            action = "redact"
            allow = False
            risk = "high"
            sanitized_answer = self._redact_pii(answer_text)

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
            details={"stage": "post_generation"},
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
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
        redacted = IBAN_RE.sub("[REDACTED_IBAN]", redacted)
        return redacted

    def _groundedness_score(self, answer: str, context: str) -> float:
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 0.0
        if not context_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return overlap / max(len(answer_tokens), 1)

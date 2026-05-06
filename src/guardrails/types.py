from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

GuardrailsStage = Literal["pre_query", "context", "post_generation", "dialog", "execution"]
GuardrailAction = Literal["allow", "block", "fallback", "redact", "rewrite"]


@dataclass
class GuardrailsRequest:
    query: str = ""
    answer: str = ""
    docs: List[Any] = field(default_factory=list)
    chat_history: List[dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RailOutcome:
    allow: bool = True
    action: GuardrailAction = "allow"
    reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    rewritten_query: Optional[str] = None
    rewritten_answer: Optional[str] = None


@dataclass(frozen=True)
class FlowStep:
    rail: str
    on_block_action: Optional[str] = None
    stop_on_block: bool = True


@dataclass
class ActionResult:
    metadata: Dict[str, Any] = field(default_factory=dict)
    add_reasons: List[str] = field(default_factory=list)
    override_allow: Optional[bool] = None
    override_action: Optional[GuardrailAction] = None
    override_query: Optional[str] = None
    override_answer: Optional[str] = None


@dataclass
class TraceEntry:
    timestamp: str
    stage: GuardrailsStage
    rail: str
    decision: str
    action: GuardrailAction
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now(stage: GuardrailsStage, rail: str, decision: str, action: GuardrailAction, *, reasons: Optional[List[str]] = None, details: Optional[Dict[str, Any]] = None) -> "TraceEntry":
        return TraceEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            rail=rail,
            decision=decision,
            action=action,
            reasons=list(reasons or []),
            details=dict(details or {}),
        )


@dataclass
class StageResult:
    stage: GuardrailsStage
    allow: bool
    action: GuardrailAction
    query: str
    answer: str
    reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    trace: List[TraceEntry] = field(default_factory=list)
    blocked_by: Optional[str] = None

    @property
    def changed_query(self) -> bool:
        return bool(self.details.get("query_changed", False))

    @property
    def changed_answer(self) -> bool:
        return bool(self.details.get("answer_changed", False))

from __future__ import annotations

from typing import Any, List, Literal, Optional, Protocol, Tuple

try:
    from config.models import SafetyConfig
    from core.safety_audit import SafetyAction, SafetyResult
    from guardrails.logging.events import trace_to_audit_payload
    from guardrails.integrations.nemo_official import create_nemo_official_runtime
    from guardrails.types import StageResult
except Exception:
    from src.config.models import SafetyConfig
    from src.core.safety_audit import SafetyAction, SafetyResult
    from src.guardrails.logging.events import trace_to_audit_payload
    from src.guardrails.integrations.nemo_official import create_nemo_official_runtime
    from src.guardrails.types import StageResult


class SafetyChecker(Protocol):
    @property
    def is_active(self) -> bool: ...

    def check_query_safety(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> SafetyResult: ...

    def check_context_safety(self, docs: List[Any]) -> SafetyResult: ...

    def check_answer_safety(
        self,
        query: str,
        docs: List[Any],
        answer: str,
    ) -> SafetyResult: ...

    def apply_safety_action(self, result: SafetyResult, answer: str) -> str: ...


SafetyProvider = Literal["nemo", "off"]


def _reason_matches_pii(reason: str) -> bool:
    return (
        reason.startswith("pii_")
        or reason in {
            "query_contains_pii",
            "context_contains_pii",
            "response_contains_pii",
            "pii_detected_in_query",
            "pii_detected_in_context",
            "pii_detected_in_answer",
        }
    )


def classify_safety_result(result: Optional[SafetyResult]) -> Optional[str]:
    if result is None:
        return None

    details = result.details or {}
    stage = str(details.get("nemo_stage") or details.get("stage") or "").strip().lower()
    reasons = [str(reason) for reason in (result.reasons or [])]

    if any(reason == "low_groundedness" for reason in reasons):
        return "grounding"

    if stage == "context" and result.action in {"block", "fallback"}:
        return "context"

    if any(_reason_matches_pii(reason) for reason in reasons):
        if stage == "context" and result.action in {"block", "fallback"}:
            return "context"
        return "pii"

    if stage == "context" and result.action == "redact":
        return "context"

    if result.action in {"block", "fallback"}:
        return "security"

    return None


def fallback_text_for_result(config: SafetyConfig, result: Optional[SafetyResult]) -> str:
    category = classify_safety_result(result)
    if category == "grounding":
        return config.grounding_fallback_text
    if category == "pii":
        return config.pii_fallback_text
    if category == "context":
        return config.context_fallback_text
    return config.security_fallback_text or config.fallback_text


def resolve_safety_provider(config: SafetyConfig) -> SafetyProvider:
    if config.backend == "off" or not config.enabled or config.mode == "off":
        return "off"
    return "nemo"


class OffSafetyChecker:
    @property
    def is_active(self) -> bool:
        return False

    def check_query_safety(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> SafetyResult:
        del query, chat_history
        return SafetyResult(allow=True, risk_level="low", action="allow", reasons=[])

    def check_context_safety(self, docs: List[Any]) -> SafetyResult:
        del docs
        return SafetyResult(allow=True, risk_level="low", action="allow", reasons=[])

    def check_answer_safety(
        self,
        query: str,
        docs: List[Any],
        answer: str,
    ) -> SafetyResult:
        del query, docs, answer
        return SafetyResult(allow=True, risk_level="low", action="allow", reasons=[])

    def apply_safety_action(self, result: SafetyResult, answer: str) -> str:
        del result
        return answer


class NemoSafetyChecker:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self._runtime = None
        self._runtime_error: Optional[str] = None

        try:
            self._runtime = create_nemo_official_runtime(config)
        except Exception as exc:
            self._runtime_error = f"{type(exc).__name__}: {exc}"

    @property
    def is_active(self) -> bool:
        return self.config.enabled and self.config.mode != "off"

    def _stage_to_payload(self, stage_result: Optional[StageResult]) -> Optional[dict[str, Any]]:
        if stage_result is None:
            return None
        return {
            "stage": stage_result.stage,
            "allow": stage_result.allow,
            "action": stage_result.action,
            "reasons": list(stage_result.reasons),
            "scores": dict(stage_result.scores),
            "details": dict(stage_result.details),
            "blocked_by": stage_result.blocked_by,
            "query": stage_result.query,
            "answer": stage_result.answer,
            "trace": trace_to_audit_payload(stage_result.trace),
        }

    def _run_pre_query(
        self,
        query: str,
        chat_history: Optional[List[dict]],
    ) -> Tuple[Optional[StageResult], Optional[str]]:
        if self._runtime is None:
            reason = self._runtime_error or "nemo_runtime_not_initialized"
            return None, reason
        try:
            return self._runtime.run_pre_query(query=query, chat_history=chat_history), None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._runtime_error = error
            return None, error

    def _run_context(self, docs: List[Any]) -> Tuple[Optional[StageResult], Optional[str]]:
        if self._runtime is None:
            reason = self._runtime_error or "nemo_runtime_not_initialized"
            return None, reason
        try:
            return self._runtime.run_context(query="", docs=docs), None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._runtime_error = error
            return None, error

    def _run_post_generation(
        self,
        query: str,
        docs: List[Any],
        answer: str,
    ) -> Tuple[Optional[StageResult], Optional[str]]:
        if self._runtime is None:
            reason = self._runtime_error or "nemo_runtime_not_initialized"
            return None, reason
        try:
            return self._runtime.run_post_generation(query=query, answer=answer, docs=docs), None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._runtime_error = error
            return None, error

    @staticmethod
    def _to_safety_action(action: str) -> SafetyAction:
        if action in {"allow", "block", "fallback", "redact"}:
            return action  # type: ignore[return-value]
        return "fallback"

    @staticmethod
    def _risk_from_action(action: str) -> str:
        if action in {"block", "fallback"}:
            return "high"
        if action == "redact":
            return "medium"
        return "low"

    @staticmethod
    def _risk_from_category(action: str, fallback_category: Optional[str]) -> str:
        if action == "allow":
            return "low"
        if fallback_category in {"grounding", "pii", "context"}:
            return "medium"
        return NemoSafetyChecker._risk_from_action(action)

    def _runtime_result_to_safety_result(
        self,
        stage_result: StageResult,
        *,
        stage: str,
        decision_owner: str,
        mode: str,
    ) -> SafetyResult:
        details = {
            "safety_provider": "nemo_runtime",
            "nemo_runtime_kind": stage_result.details.get("nemo_runtime_kind", "unknown"),
            "nemo_stage": stage,
            "nemo_mode": mode,
            "pre_query_decision_owner": decision_owner,
            "legacy_fallback_used": False,
            "nemo_runtime": self._stage_to_payload(stage_result),
            "nemo_runtime_error": self._runtime_error,
        }
        if "sanitized_docs" in stage_result.details:
            details["sanitized_docs"] = stage_result.details.get("sanitized_docs")

        action = self._to_safety_action(stage_result.action)
        provisional_result = SafetyResult(
            allow=stage_result.allow,
            risk_level=self._risk_from_action(action),
            reasons=list(stage_result.reasons),
            sanitized_answer=stage_result.answer if stage == "post_generation" and stage_result.changed_answer else None,
            action=action,
            scores=dict(stage_result.scores),
            details=details,
        )
        fallback_category = classify_safety_result(provisional_result)
        if fallback_category is not None:
            details["fallback_category"] = fallback_category
        return SafetyResult(
            allow=provisional_result.allow,
            risk_level=self._risk_from_category(action, fallback_category),
            reasons=list(provisional_result.reasons),
            sanitized_answer=provisional_result.sanitized_answer,
            action=provisional_result.action,
            scores=dict(provisional_result.scores),
            details=details,
        )

    def _attach_runtime_meta(
        self,
        result: SafetyResult,
        *,
        stage: str,
        runtime_shadow: Optional[StageResult] = None,
        mode: str = "runtime_shadow",
        decision_owner: str = "nemo_runtime",
        provider: str = "nemo_runtime",
        legacy_fallback_used: bool = False,
        runtime_error: Optional[str] = None,
    ) -> SafetyResult:
        details = dict(result.details or {})
        details.update(
            {
                "safety_provider": provider,
                "nemo_stage": stage,
                "nemo_mode": mode,
                "pre_query_decision_owner": decision_owner,
                "legacy_fallback_used": legacy_fallback_used,
                "nemo_runtime_error": runtime_error or self._runtime_error,
            }
        )
        if runtime_shadow is not None:
            details["nemo_shadow"] = self._stage_to_payload(runtime_shadow)

        return SafetyResult(
            allow=result.allow,
            risk_level=result.risk_level,
            reasons=list(result.reasons),
            sanitized_answer=result.sanitized_answer,
            action=result.action,
            scores=dict(result.scores),
            details=details,
        )

    def _allow_result(
        self,
        *,
        stage: str,
        mode: str,
        decision_owner: str,
        runtime_shadow: Optional[StageResult] = None,
        runtime_error: Optional[str] = None,
    ) -> SafetyResult:
        return self._attach_runtime_meta(
            SafetyResult(allow=True, risk_level="low", action="allow", reasons=[]),
            stage=stage,
            runtime_shadow=runtime_shadow,
            mode=mode,
            decision_owner=decision_owner,
            provider="nemo_runtime",
            legacy_fallback_used=False,
            runtime_error=runtime_error,
        )

    def _runtime_failure_result(self, *, stage: str, runtime_error: Optional[str]) -> SafetyResult:
        if self.config.fail_closed and self.is_active:
            return SafetyResult(
                allow=False,
                risk_level="high",
                reasons=[f"nemo_{stage}_runtime_error"],
                action="fallback",
                details={
                    "safety_provider": "nemo_runtime",
                    "nemo_stage": stage,
                    "nemo_mode": "runtime_error_fail_closed",
                    "pre_query_decision_owner": "nemo_runtime_error",
                    "legacy_fallback_used": False,
                    "nemo_runtime_error": runtime_error or self._runtime_error,
                },
            )

        return self._allow_result(
            stage=stage,
            mode="runtime_error_allow_open",
            decision_owner="nemo_runtime_error",
            runtime_error=runtime_error,
        )

    def check_query_safety(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> SafetyResult:
        if not self.config.nemo_input_enabled:
            return self._allow_result(
                stage="pre_query",
                mode="nemo_input_disabled",
                decision_owner="nemo_input_disabled",
            )

        runtime_pre, runtime_error = self._run_pre_query(query, chat_history)

        if runtime_pre is not None and self.config.nemo_enforce_input:
            return self._runtime_result_to_safety_result(
                runtime_pre,
                stage="pre_query",
                decision_owner="nemo_primary",
                mode="runtime_primary",
            )

        if runtime_pre is not None:
            return self._allow_result(
                stage="pre_query",
                runtime_shadow=runtime_pre,
                mode="runtime_shadow_no_enforce",
                decision_owner="nemo_shadow_prequery",
            )

        return self._runtime_failure_result(stage="pre_query", runtime_error=runtime_error)

    def check_context_safety(self, docs: List[Any]) -> SafetyResult:
        if not self.config.nemo_context_enabled:
            return self._allow_result(
                stage="context",
                mode="nemo_context_disabled",
                decision_owner="nemo_context_disabled",
            )

        runtime_context, runtime_error = self._run_context(docs)
        if runtime_context is not None:
            return self._runtime_result_to_safety_result(
                runtime_context,
                stage="context",
                decision_owner="nemo_context_primary",
                mode="runtime_primary",
            )

        return self._runtime_failure_result(stage="context", runtime_error=runtime_error)

    def check_answer_safety(
        self,
        query: str,
        docs: List[Any],
        answer: str,
    ) -> SafetyResult:
        if not self.config.nemo_output_enabled:
            return self._allow_result(
                stage="post_generation",
                mode="nemo_output_disabled",
                decision_owner="nemo_output_disabled",
            )

        runtime_post, runtime_error = self._run_post_generation(query, docs, answer)
        if runtime_post is not None and self.config.nemo_enforce_output:
            return self._runtime_result_to_safety_result(
                runtime_post,
                stage="post_generation",
                decision_owner="nemo_output_enforce",
                mode="runtime_output_enforce",
            )

        if runtime_post is not None:
            return self._allow_result(
                stage="post_generation",
                runtime_shadow=runtime_post,
                mode="runtime_shadow_no_enforce",
                decision_owner="nemo_shadow_post_generation",
            )

        return self._runtime_failure_result(stage="post_generation", runtime_error=runtime_error)

    def apply_safety_action(self, result: SafetyResult, answer: str) -> str:
        if result.action == "redact" and result.sanitized_answer:
            return result.sanitized_answer
        if result.action in {"block", "fallback"}:
            return fallback_text_for_result(self.config, result)
        return answer


def create_safety_checker(config: SafetyConfig) -> SafetyChecker:
    provider = resolve_safety_provider(config)
    if provider == "off":
        return OffSafetyChecker()
    return NemoSafetyChecker(config)

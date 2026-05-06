from __future__ import annotations

from src.config.models import SafetyConfig
from src.core.safety_adapter import classify_safety_result, fallback_text_for_result
from src.core.safety_audit import SafetyResult


def _config() -> SafetyConfig:
    return SafetyConfig(
        enabled=True,
        mode="enforce",
        min_groundedness=0.7,
        block_pii=True,
        block_injection=True,
        fail_closed=True,
        fallback_text="security fallback",
        security_fallback_text="security fallback",
        grounding_fallback_text="grounding fallback",
        pii_fallback_text="pii fallback",
        context_fallback_text="context fallback",
        backend="nemo",
    )


def test_low_groundedness_uses_grounding_fallback_text() -> None:
    result = SafetyResult(
        allow=False,
        risk_level="medium",
        reasons=["low_groundedness"],
        action="fallback",
        details={"stage": "post_generation"},
    )

    assert classify_safety_result(result) == "grounding"
    assert fallback_text_for_result(_config(), result) == "grounding fallback"

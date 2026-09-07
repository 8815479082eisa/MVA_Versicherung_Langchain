from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("stub_entailment_provider")

from langchain_core.documents import Document

from scripts.experimental_groundedness_v5 import (
    calculate_groundedness_score_v5_experimental,
)
from src.api.rag_service import _record_answer_version
from src.config.models import SafetyConfig
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)
from src.guardrails.integrations.nemo_actions import _evaluate_output_safety


def _safety_config() -> SafetyConfig:
    return SafetyConfig(
        enabled=True,
        mode="enforce",
        min_groundedness=0.7888,
        block_pii=False,
        block_injection=True,
        fail_closed=True,
        fallback_text="security fallback",
        security_fallback_text="security fallback",
        grounding_fallback_text="grounding fallback",
        pii_fallback_text="pii fallback",
        context_fallback_text="context fallback",
        backend="nemo",
    )


def _docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "Policy number: TEST-KFZ-2026-1003. Coverage type: Partially "
                "comprehensive cover. Individual deductible: 300 EUR. Annual premium: 720 EUR."
            ),
            metadata={"source": "espocrm:policy:1", "source_type": "crm", "chunk_id": "crm-1"},
        ),
        Document(
            page_content=(
                "Glass damage to the front windscreen is covered. You will not have to bear "
                "a deductible if the damaged front windscreen is repaired and not replaced."
            ),
            metadata={"source_file": "motor.pdf", "source_page": 8, "chunk_id": "pdf-1"},
        ),
    ]


def test_pre_fallback_answer_is_retained_after_final_fallback_snapshot() -> None:
    diagnostics = RequestDiagnostics(route="combined")
    token = set_current_diagnostics(diagnostics)
    try:
        _record_answer_version("immediatelyBeforeGroundedness", "Supported draft answer")
        _record_answer_version("finalAfterOutputSafety", "grounding fallback")
    finally:
        reset_current_diagnostics(token)

    versions = diagnostics.evidence["answerVersions"]
    assert versions["immediatelyBeforeGroundedness"] == "Supported draft answer"
    assert versions["finalAfterOutputSafety"] == "grounding fallback"


def test_claim_scores_and_best_support_are_serialized() -> None:
    result = calculate_groundedness_score_v5_experimental(
        "The individual deductible is 300 EUR.",
        _docs(),
        "What is the deductible?",
    ).to_dict()

    assert result["extracted_claims"] == ("The individual deductible is 300 EUR.",)
    claim = result["claim_details"][0]
    assert claim["source_file"] == "espocrm:policy:1"
    assert claim["lexical_overlap"] > 0
    assert claim["hard_fact_overlap"] == 1.0
    assert claim["v4_claim_score"] > 0
    assert claim["v5_claim_score"] > 0
    assert claim["reason_code"] == "claim_support_nonzero"


def test_applied_caps_and_typed_mismatches_are_serialized() -> None:
    docs = _docs()
    docs[0].page_content = docs[0].page_content.replace(
        "Partially comprehensive cover", "Partial Coverage"
    )
    result = calculate_groundedness_score_v5_experimental(
        "Policy TEST-KFZ-2026-9999 has Comprehensive Coverage and costs 999 EUR.",
        docs,
        "What is the current policy coverage and premium?",
    ).to_dict()

    assert "unsupported_identifier_cap_0.20" in result["applied_caps"]
    assert "unsupported_numeric_fact_cap_0.38" in result["applied_caps"]
    assert result["coverage_type_mismatches"]
    assert result["policy_number_mismatches"]
    assert result["numeric_or_monetary_mismatches"]


def test_claim_diagnostics_survive_low_groundedness_failure(monkeypatch) -> None:
    from tests.unit.test_claim_groundedness import FakeJudge
    from src.core import claim_groundedness
    monkeypatch.setattr(claim_groundedness, "ModelJudge", lambda: FakeJudge(relation="insufficient_evidence"))
    payload, _ = _evaluate_output_safety(
        "What is covered?",
        "The moon is insured.",
        _docs(),
        _safety_config(),
    )

    assert payload["allow"] is False
    assert payload["action"] == "fallback"
    assert "low_groundedness" in payload["reasons"]
    details = payload["details"]["groundedness"]
    assert details["supported_fraction"] == details["score"]
    assert details["relation_counts"] == {"insufficient_evidence": 1}
    assert details["extracted_claims"]
    assert details["claim_details"]


def test_answer_snapshots_redact_secrets_and_unnecessary_pii() -> None:
    diagnostics = RequestDiagnostics(route="combined")
    token = set_current_diagnostics(diagnostics)
    try:
        _record_answer_version(
            "rawOpenAIAnswer",
            "Authorization: Bearer abc.def.secret; mail jane@example.com; IBAN DE89370400440532013000",
        )
    finally:
        reset_current_diagnostics(token)

    captured = diagnostics.evidence["answerVersions"]["rawOpenAIAnswer"]
    assert "abc.def.secret" not in captured
    assert "jane@example.com" not in captured
    assert "DE89370400440532013000" not in captured
    assert "[SECRET_REDACTED]" in captured
    assert "[EMAIL_REDACTED]" in captured
    assert "[IBAN_REDACTED]" in captured


def test_markdown_headings_and_source_only_lines_are_reported_as_ignored() -> None:
    result = calculate_groundedness_score_v5_experimental(
        "### General terms\nGlass damage is covered.\n### Sources\n- [motor.pdf, page 9]",
        _docs(),
        "Is glass damage covered?",
    ).to_dict()

    ignored = result["ignored_nonsemantic_lines"]
    assert {item["reason_code"] for item in ignored} >= {
        "markdown_heading",
        "citation_only_source_line",
    }
    assert result["extracted_claims"] == ("Glass damage is covered.",)

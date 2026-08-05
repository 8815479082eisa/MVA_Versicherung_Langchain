from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from src.api import rag_service
from src.core.answer_completeness import (
    build_answer_requirements,
    evaluate_answer_completeness,
    format_answer_requirements,
)
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)


QUERY = (
    "Noah Weber reports the theft of his car. Is this loss generally covered by "
    "his current motor insurance, and which documented conditions apply? Please "
    "provide the relevant active policy number, coverage type, individual "
    "deductible and annual premium. Clearly distinguish the general insurance "
    "terms from Noah's individual contract data, and do not make a final claim "
    "decision."
)


def _theft_docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "Insured benefits with partially comprehensive insurance: Theft, "
                "glass breakage and marten damage."
            ),
            metadata={
                "source": "motor-vehicle-insurance-product-sheet.pdf",
                "source_type": "pdf",
                "page": 2,
                "chunk_id": "product-theft",
            },
        ),
        Document(
            page_content=(
                "The insurance does not cover damage by scorching unless caused by "
                "a fire. K2.1.2 Damage caused by natural forces. K2.1.4 Theft. The "
                "insurance covers loss, disappearance, destruction or damage caused "
                "by theft, misappropriation or robbery of the insured vehicle if the "
                "damage occurred involuntarily. No compensation is paid if the act "
                "was committed by family members."
            ),
            metadata={
                "source": "motor-vehicle-insurance-sti.pdf",
                "source_type": "pdf",
                "page": 12,
                "chunk_id": "theft-coverage",
            },
        ),
        Document(
            page_content=(
                "K5.4 Damage caused by theft. You must notify the responsible police "
                "without delay. If the theft occurs abroad, it must also be reported "
                "to the police station at your Swiss place of residence. If the "
                "vehicle is found or its whereabouts become known, we must be "
                "informed without delay. K5.5 Damage caused by animals."
            ),
            metadata={
                "source": "motor-vehicle-insurance-sti.pdf",
                "source_type": "pdf",
                "page": 16,
                "chunk_id": "theft-duties",
            },
        ),
        Document(
            page_content=(
                "CRM FACT\nPolicy TEST-KFZ-2026-1701 (Noah Weber): Helvetia Motor "
                "Vehicle Insurance, Partially comprehensive cover, Active; deductible "
                "300 EUR; annual premium 735 EUR."
            ),
            metadata={
                "source": "espocrm:policy",
                "source_type": "crm",
                "section": "TEST-KFZ-2026-1701",
            },
        ),
    ]


def _complete_answer() -> str:
    return (
        "General insurance terms: Theft is generally covered under partially "
        "comprehensive insurance [motor-vehicle-insurance-product-sheet.pdf, page 3]. "
        "The terms cover loss, disappearance, destruction or damage caused by theft, "
        "misappropriation or robbery when the damage occurred involuntarily. No "
        "compensation is paid if the act was committed by family members "
        "[motor-vehicle-insurance-sti.pdf, page 13]. The theft must be reported "
        "immediately to the responsible police. The insurer must be informed without "
        "delay if the vehicle is recovered or its whereabouts become known "
        "[motor-vehicle-insurance-sti.pdf, page 17]. Individual CRM data: Noah Weber's "
        "active policy is TEST-KFZ-2026-1701 with Partially comprehensive cover, a "
        "300 EUR deductible and a 735 EUR annual premium "
        "[CRM: TEST-KFZ-2026-1701]. This is general information and not a final claim "
        "decision."
    )


def _without(answer: str, sentence: str) -> str:
    return answer.replace(sentence, "")


def _run_generation(first: str, regenerated: str | None = None):
    diagnostics = RequestDiagnostics(route="combined")
    token = set_current_diagnostics(diagnostics)
    first_chain = SimpleNamespace(
        invoke=lambda _payload: SimpleNamespace(content=first)
    )
    regeneration_chain = SimpleNamespace(
        invoke=lambda _payload: SimpleNamespace(content=regenerated or "")
    )
    try:
        with patch.object(
            rag_service, "build_generation_chain", return_value=first_chain
        ) as first_builder, patch.object(
            rag_service,
            "build_completeness_regeneration_chain",
            return_value=regeneration_chain,
        ) as regeneration_builder, patch.object(
            rag_service,
            "answer_completeness_enabled",
            return_value=True,
        ), patch.object(
            rag_service,
            "invoke_llm_stage",
            side_effect=lambda _stage, operation, **_kwargs: operation(),
        ):
            answer = rag_service.generate_answer(
                object(),
                QUERY,
                _theft_docs(),
            )
    finally:
        reset_current_diagnostics(token)
    return answer, diagnostics, first_builder, regeneration_builder


def test_disabled_completeness_returns_first_answer_without_gate_or_retry() -> None:
    diagnostics = RequestDiagnostics(route="combined")
    token = set_current_diagnostics(diagnostics)
    first_chain = SimpleNamespace(
        invoke=lambda _payload: SimpleNamespace(content="Direct grounded answer.")
    )
    try:
        with patch.object(
            rag_service, "build_generation_chain", return_value=first_chain
        ), patch.object(
            rag_service,
            "build_completeness_regeneration_chain",
        ) as regeneration_builder, patch.object(
            rag_service,
            "answer_completeness_enabled",
            return_value=False,
        ), patch.object(
            rag_service,
            "invoke_llm_stage",
            side_effect=lambda _stage, operation, **_kwargs: operation(),
        ):
            answer = rag_service.generate_answer(object(), QUERY, _theft_docs())
    finally:
        reset_current_diagnostics(token)

    completeness = diagnostics.evidence["completeness"]
    assert answer.startswith("Direct grounded answer.")
    assert "Document sources:" in answer
    assert "CRM sources:" in answer
    assert completeness["enabled"] is False
    assert completeness["skipped"] is True
    assert completeness["skipReason"] == "disabled_by_configuration"
    assert completeness["completenessPass"] is None
    assert completeness["completenessRequiredItems"] == []
    assert completeness["completenessRetryPerformed"] is False
    assert diagnostics.evidence["answerGeneration"]["applicationCallCount"] == 1
    regeneration_builder.assert_not_called()


def test_complete_theft_answer_passes_without_regeneration() -> None:
    requirements = build_answer_requirements(QUERY, _theft_docs())
    evaluation = evaluate_answer_completeness(_complete_answer(), requirements)

    answer, diagnostics, _, regeneration_builder = _run_generation(_complete_answer())

    assert evaluation.passed
    assert answer == _complete_answer()
    assert diagnostics.evidence["completeness"]["completenessPass"] is True
    assert diagnostics.evidence["completeness"]["completenessRetryPerformed"] is False
    assert diagnostics.evidence["answerGeneration"]["applicationCallCount"] == 1
    regeneration_builder.assert_not_called()


def test_missing_police_reporting_triggers_one_full_regeneration() -> None:
    missing_sentence = "The theft must be reported immediately to the responsible police. "
    first = _without(_complete_answer(), missing_sentence)

    answer, diagnostics, _, regeneration_builder = _run_generation(
        first,
        _complete_answer(),
    )

    completeness = diagnostics.evidence["completeness"]
    assert completeness["completenessMissingBeforeRetry"] == ["theft_police_reporting"]
    assert completeness["completenessRetryPerformed"] is True
    assert completeness["completenessMissingAfterRetry"] == []
    assert completeness["completenessPass"] is True
    assert "responsible police" in answer
    assert "[motor-vehicle-insurance-sti.pdf, page 17]" in answer
    assert diagnostics.evidence["answerGeneration"]["applicationCallCount"] == 2
    regeneration_builder.assert_called_once()


def test_missing_recovery_notification_triggers_one_regeneration() -> None:
    missing_sentence = (
        "The insurer must be informed without delay if the vehicle is recovered or "
        "its whereabouts become known "
    )
    first = _without(_complete_answer(), missing_sentence)

    answer, diagnostics, _, _ = _run_generation(first, _complete_answer())

    completeness = diagnostics.evidence["completeness"]
    assert completeness["completenessMissingBeforeRetry"] == [
        "theft_recovery_notification"
    ]
    assert completeness["completenessMissingAfterRetry"] == []
    assert "vehicle is recovered" in answer


def test_two_missing_theft_duties_are_both_diagnosed_and_regenerated() -> None:
    first = _complete_answer().replace(
        "The theft must be reported immediately to the responsible police. The "
        "insurer must be informed without delay if the vehicle is recovered or its "
        "whereabouts become known [motor-vehicle-insurance-sti.pdf, page 17]. ",
        "",
    )

    answer, diagnostics, _, _ = _run_generation(first, _complete_answer())

    completeness = diagnostics.evidence["completeness"]
    assert completeness["completenessMissingBeforeRetry"] == [
        "theft_police_reporting",
        "theft_recovery_notification",
    ]
    assert completeness["completenessMissingAfterRetry"] == []
    assert "responsible police" in answer
    assert "whereabouts become known" in answer


def test_fire_clause_is_not_a_theft_requirement_or_generated_appendix() -> None:
    requirements = build_answer_requirements(QUERY, _theft_docs())
    ids = {item.requirement_id for item in requirements}

    answer, diagnostics, _, _ = _run_generation(_complete_answer())

    assert not any("fire" in requirement_id for requirement_id in ids)
    assert "unless caused by a fire" not in answer
    assert "Material documented conditions" not in answer
    assert diagnostics.evidence["completeness"]["completenessPass"] is True


def test_abroad_requirement_is_omitted_without_foreign_scenario() -> None:
    requirements = build_answer_requirements(QUERY, _theft_docs())
    assert "theft_abroad_reporting" not in {
        item.requirement_id for item in requirements
    }


def test_abroad_requirement_is_added_for_explicit_foreign_scenario() -> None:
    query = QUERY.replace("the theft of his car", "the theft of his car abroad")
    requirements = build_answer_requirements(query, _theft_docs())
    assert "theft_abroad_reporting" in {
        item.requirement_id for item in requirements
    }


def test_product_sheet_requirement_excerpt_keeps_theft_evidence() -> None:
    requirements = build_answer_requirements(QUERY, _theft_docs())
    formatted = format_answer_requirements(requirements)

    assert "[theft_partial_comprehensive_coverage]" in formatted
    assert "Insured benefits with partially comprehensive insurance" in formatted
    assert "Theft" in formatted


def test_unrequested_abroad_theft_condition_is_removed() -> None:
    answer = (
        "The theft must be reported to the responsible police without delay. "
        "If the theft occurs abroad, it must also be reported to the police "
        "station at Noah's Swiss place of residence. "
        "If the vehicle is found, the insurer must be informed without delay."
    )

    cleaned = rag_service._remove_unrequested_abroad_theft_conditions(answer, QUERY)

    assert "responsible police" in cleaned
    assert "vehicle is found" in cleaned
    assert "abroad" not in cleaned
    assert "Swiss place of residence" not in cleaned


def test_missing_product_sheet_citation_is_repaired_for_theft_partial_cover() -> None:
    requirements = build_answer_requirements(QUERY, _theft_docs())
    answer = (
        "Yes, the theft is generally covered under partially comprehensive cover. "
        "The insurance covers loss caused by theft [motor-vehicle-insurance-sti.pdf, page 13]."
    )

    repaired = rag_service._ensure_generation_requirement_support(answer, requirements)

    assert "General product terms" in repaired
    assert "[motor-vehicle-insurance-product-sheet.pdf, page 3]" in repaired


def test_no_final_claim_decision_sentence_is_added_when_requested() -> None:
    answer = "Theft is generally covered under the documented terms."

    repaired = rag_service._ensure_no_final_claim_decision_sentence(answer, QUERY)

    assert repaired.endswith("This is general information, not a final claim decision.")


def test_still_incomplete_regeneration_stops_after_two_calls() -> None:
    first = _complete_answer().replace(
        "The theft must be reported immediately to the responsible police. The "
        "insurer must be informed without delay if the vehicle is recovered or its "
        "whereabouts become known [motor-vehicle-insurance-sti.pdf, page 17]. ",
        "",
    )

    answer, diagnostics, _, regeneration_builder = _run_generation(first, first)

    completeness = diagnostics.evidence["completeness"]
    assert completeness["completenessPass"] is False
    assert completeness["completenessMissingAfterRetry"] == [
        "theft_police_reporting",
        "theft_recovery_notification",
    ]
    assert diagnostics.evidence["answerGeneration"]["applicationCallCount"] == 2
    assert diagnostics.retries["completeness"] == 1
    regeneration_builder.assert_called_once()
    assert answer == first


def test_page_17_citation_is_required_for_theft_duties() -> None:
    uncited = _complete_answer().replace(
        " [motor-vehicle-insurance-sti.pdf, page 17]",
        "",
    )
    requirements = build_answer_requirements(QUERY, _theft_docs())
    evaluation = evaluate_answer_completeness(uncited, requirements)

    assert "theft_police_reporting" in evaluation.missing_ids
    assert "theft_recovery_notification" in evaluation.missing_ids

    answer, diagnostics, _, _ = _run_generation(uncited, _complete_answer())
    assert diagnostics.evidence["completeness"]["completenessPass"] is True
    assert "[motor-vehicle-insurance-sti.pdf, page 17]" in answer

from __future__ import annotations

from langchain_core.documents import Document

from src.core.claim_groundedness import evaluate_claim_groundedness


class _FailIfCalledJudge:
    def call(self, *args, **kwargs):
        raise AssertionError("controlled abstention must not invoke the entailment model")


def test_controlled_abstention_is_not_treated_as_a_factual_claim() -> None:
    answer = "The available sources do not contain enough information to answer this question."

    score, details = evaluate_claim_groundedness(
        answer,
        [Document(page_content="Any retrieved context")],
        query="What is covered?",
        judge=_FailIfCalledJudge(),
    )

    assert score == 1.0
    assert details["evaluation_status"] == "not_applicable"
    assert details["controlled_abstention"] is True
    assert details["controlled_abstention_kind"] == "insufficient_information"
    assert details["threshold_policy"] == "controlled_abstention_no_factual_claims"
    assert details["claim_details"] == []
    assert details["extracted_claims"] == []
    assert details["sensitive_unsupported_count"] == 0
    assert details["provenance_failure_count"] == 0


def test_factual_text_appended_to_abstention_does_not_bypass_groundedness() -> None:
    answer = (
        "The available sources do not contain enough information to answer this question. "
        "Worldwide cover lasts one year."
    )

    score, details = evaluate_claim_groundedness(
        answer,
        [Document(page_content="Any retrieved context")],
        query="For how long is worldwide cover provided?",
        judge=_FailIfCalledJudge(),
    )

    assert score is None
    assert details["evaluation_status"] == "failed"
    assert details["controlled_abstention"] if "controlled_abstention" in details else True
    assert details["exception_type"] == "AssertionError"

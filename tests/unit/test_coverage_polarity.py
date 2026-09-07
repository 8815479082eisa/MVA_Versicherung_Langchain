import pytest

from src.core.coverage_polarity import CoveragePolarity as P, coverage_polarity, coverage_relation


SOURCE = "In Kosovo, insurance does not apply to liability."
NEGATIVES = [
    SOURCE,
    "In Kosovo, insurance does not apply to liability coverage.",
    "Liability insurance is not covered in Kosovo.",
    "There is no liability coverage in Kosovo.",
    "Coverage for liability does not apply in Kosovo.",
    "Liability cover is excluded in Kosovo.",
    "In Kosovo, insurance doesn't apply to liability.",
    "Liability is not insured in Kosovo.",
    "Liability claims are excluded in Kosovo.",
    "Liability claims are not covered in Kosovo.",
    "There is no liability cover in Kosovo.",
]


@pytest.mark.parametrize("answer", NEGATIVES)
def test_negative_variants(answer):
    assert coverage_polarity(answer) is P.NEGATIVE
    assert coverage_relation(answer, SOURCE) != "contradiction"


def test_true_contradiction_and_specific_evidence():
    evidence = "Your insurance is valid in the countries of Europe. " + SOURCE
    assert coverage_relation("Liability insurance is fully covered in Kosovo.", evidence) == "contradiction"
    assert coverage_relation(SOURCE, evidence) == "support"
    assert coverage_relation(SOURCE, "Liability insurance is fully covered in Kosovo.") == "contradiction"


@pytest.mark.parametrize("text", [
    "Liability coverage in Kosovo.", "Liability may be covered in Kosovo.",
    "Liability is not excluded in Kosovo.", "Does liability insurance apply in Kosovo?",
    "The discount does not apply in Kosovo.",
    "Liability is covered only in Kosovo.",
])
def test_unknown_is_not_contradiction(text):
    assert coverage_polarity(text) is P.UNKNOWN
    assert coverage_relation(text, SOURCE) == "unknown"


@pytest.mark.parametrize("evidence", [
    "Liability insurance is covered in Albania.",
    "Comprehensive insurance is covered in Kosovo.",
    "Your insurance is valid in the countries of Europe.",
    SOURCE + " Liability insurance is fully covered in Kosovo.",
    "Liability insurance is fully covered outside Kosovo.",
    "liability insurance is fully covered in albania.",
    "Personal liability insurance is fully covered in Kosovo.",
])
def test_other_scopes_and_conflicting_evidence_are_not_hard_contradictions(evidence):
    assert coverage_relation(SOURCE, evidence) != "contradiction"


def test_original_score_regression(monkeypatch):
    from langchain_core.documents import Document
    from scripts import experimental_groundedness_v5 as scoring
    monkeypatch.setattr(scoring, "calculate_groundedness_score", lambda *args: 0.97)
    result = scoring.calculate_groundedness_score_v5_experimental(
        "In Kosovo, the insurance does not apply to liability.",
        [Document(page_content=SOURCE)],
    )
    assert not result.polarity_mismatches
    assert "coverage_polarity_mismatch_cap_0.25" not in result.applied_caps
    assert result.score >= 0.95


def test_true_contradiction_still_caps_score(monkeypatch):
    from langchain_core.documents import Document
    from scripts import experimental_groundedness_v5 as scoring
    monkeypatch.setattr(scoring, "calculate_groundedness_score", lambda *args: 0.97)
    result = scoring.calculate_groundedness_score_v5_experimental(
        "Liability insurance is fully covered in Kosovo.", [Document(page_content=SOURCE)],
    )
    assert result.polarity_mismatches
    assert result.score <= 0.25


def test_shared_words_do_not_override_customer_entity():
    claim = "Lara has liability insurance that covers accidental damage to rented property."
    evidence = "Noah has liability insurance that does not cover accidental damage to rented property."
    assert coverage_relation(claim, evidence) == "unknown"


def test_positive_and_negative_same_scope_in_lowercase():
    assert coverage_relation(
        "liability is covered in kosovo.", SOURCE.lower()
    ) == "contradiction"

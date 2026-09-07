import pytest
from langchain_core.documents import Document

from src.core.claim_groundedness import (
    CATEGORIES, Extraction, ExtractionAudit, Judgments, answer_units, deterministic_checks,
    evaluate_claim_groundedness,
    quote_has_provenance,
)


class FakeJudge:
    def __init__(self, relation="supported", confidence=0.99, checks=None, quote=None):
        self.relation, self.confidence, self.checks, self.quote = relation, confidence, checks, quote

    def call(self, instruction, payload, schema):
        if schema is Extraction:
            return Extraction(claims=[{"text": text, "unit_ids": [i]} for i, text in payload["units"].items()])
        if schema is ExtractionAudit:
            return ExtractionAudit(faithful_and_complete=True, reason="test extraction audit")
        verdicts = []
        for claim in payload["claims"]:
            key = claim["evidence_ids"][0]
            checks = {name: "not_applicable" for name in CATEGORIES}
            checks["polarity"] = "conflict" if self.relation == "contradicted" else "consistent"
            checks.update(self.checks or {})
            verdicts.append({
                "claim_id": claim["claim_id"], "relation": self.relation,
                "confidence": self.confidence, "subject_scope_match": True,
                "citations": [{"evidence_id": key, "quote": self.quote or payload["evidence"][key]["text"]}],
                "checks": checks, "reason": "test semantic judgment",
            })
        return Judgments(verdicts=verdicts)


def evaluate(answer, source, **kwargs):
    return evaluate_claim_groundedness(answer, [Document(page_content=source)], judge=FakeJudge(**kwargs))


def test_high_overlap_cannot_establish_support():
    score, result = evaluate("Liability is covered in Kosovo.", "Liability is not covered in Kosovo.", relation="contradicted")
    assert score <= .25
    assert result["relation_counts"] == {"contradicted": 1}


@pytest.mark.parametrize("answer", [
    "The insurer pays for repairs.", "Repair expenses are reimbursed by the insurer.",
    "The insurance company meets the cost of fixing the damage.",
])
def test_entailment_not_word_overlap_controls_score(answer):
    score, result = evaluate(answer, "The insurer pays for repairs.")
    assert score == 1
    assert result["all_claims_supported"]
    assert result["evaluation_status"] == "success"


@pytest.mark.parametrize("relation", ["unknown", "insufficient_evidence"])
def test_uncertain_relations_are_not_contradiction(relation):
    score, result = evaluate("Insurance covers repairs.", "Insurance covers repairs.", relation=relation)
    assert score == 0
    assert not result["applied_caps"]
    assert result["relation_counts"] == {relation: 1}
    assert result["evaluation_status"] == ("uncertain" if relation == "unknown" else "success")


def test_low_confidence_contradiction_is_unknown():
    _, result = evaluate("Insurance covers repairs.", "Insurance covers repairs.", relation="contradicted", confidence=.6)
    assert result["relation_counts"] == {"unknown": 1}
    assert not result["applied_caps"]


@pytest.mark.parametrize("claim,source", [
    ("The premium is 500 EUR.", "The premium is 300 EUR."),
    ("The premium is 300 USD.", "The premium is 300 EUR."),
    ("The limit is 30%.", "The limit is 20%."),
    ("The waiting period is 3 months.", "The waiting period is 3 days."),
    ("Liability is covered in Kosovo.", "Liability is covered in Albania."),
])
def test_symbolic_checks_can_veto_incorrect_model_support(claim, source):
    _, result = evaluate(claim, source)
    assert not result["all_claims_supported"]
    assert result["claim_details"][0]["deterministic_issues"]
    assert not result["applied_caps"]  # Missing evidence isn't a contradiction.


def test_fabricated_quote_cannot_support():
    _, result = evaluate("Insurance covers repairs.", "No information.", quote="Insurance covers repairs.")
    assert result["relation_counts"] == {"unknown": 1}


def test_conditions_require_explicit_judgment():
    _, result = evaluate("Repairs are reimbursed.", "Repairs are reimbursed if preapproved.", checks={"conditions": "unknown"})
    assert not result["all_claims_supported"]


def test_unit_split_retains_list_exclusion_heading():
    assert answer_units("Not covered in:\n- Kosovo\n- Albania\nSources:\n[motor.pdf, page 6]") == ["Not covered in:", "Kosovo", "Albania"]


def test_provider_failure_fails_unknown_without_lexical_fallback():
    class Broken:
        def call(self, *args):
            raise TimeoutError()
    score, result = evaluate_claim_groundedness("Exact source.", [Document(page_content="Exact source.")], judge=Broken())
    assert score is None
    assert result["score"] is None
    assert result["supported_fraction"] is None
    assert result["all_claims_supported"] is None
    assert result["evaluation_status"] == "failed"
    assert result["relation_counts"] == {}
    assert result["exception_type"] == "TimeoutError"


def test_extractor_cannot_drop_answer_units():
    class Dropped(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is Extraction:
                return Extraction(claims=[{"text": "One claim.", "unit_ids": [0]}])
            return super().call(instruction, payload, schema)
    _, result = evaluate_claim_groundedness("One claim. Another claim.", [Document(page_content="One claim.")], judge=Dropped())
    assert result["exception_type"] == "ValueError"
    assert result["evaluation_status"] == "failed"
    assert result["all_claims_supported"] is None


def test_equivalent_duration_spellings_preserve_sensitive_values():
    assert not deterministic_checks("The waiting period is three months.", "The waiting period is 3 months.")


def test_other_location_cannot_be_hard_contradiction_even_if_judge_says_so():
    _, result = evaluate("Liability is covered in Albania.", "Liability is not covered in Kosovo.", relation="contradicted")
    assert result["relation_counts"] == {"insufficient_evidence": 1}
    assert not result["applied_caps"]


def test_ocr_quote_normalization_does_not_change_meaning():
    _, result = evaluate("Insurance is valid in Liechtenstein.",
        "Insurance is valid in Liechten- stein.", quote="Insurance is valid in Liechtenstein.")
    assert result["all_claims_supported"]


def test_quote_provenance_tolerates_pdf_spacing_and_narrow_paraphrase():
    assert quote_has_provenance(
        "Helvetia charges 4% (excl. stamp duty) of the surety sum as premium.",
        "Helvetia\tcharges\t4\t%\t(excl.\tstamp\tduty)\tof\tthe\tsurety\tsum\tas premium.",
    )
    assert quote_has_provenance(
        "if cancellation is in the first insurance year. In this instance, the entire premium for the first insurance year is owed.",
        "unless cancellation is in the first insurance year. In this instance, the entire premium for the first insurance year is owed.",
    )


def test_quote_provenance_rejects_fabricated_sensitive_values_and_unrelated_text():
    assert not quote_has_provenance(
        "Helvetia charges 7% of the surety sum as premium.",
        "Helvetia charges 4% of the surety sum as premium.",
    )
    assert not quote_has_provenance(
        "The policy covers theft anywhere in the world.",
        "Premiums are payable annually in advance.",
    )


def test_specific_exception_ranks_before_generic_overlap():
    from src.core.claim_groundedness import rank_evidence
    windows = {
        "general": {"text": "Insurance covers liability in Europe.", "document_rank": 0},
        "specific": {"text": "In Kosovo, insurance does not apply to liability.", "document_rank": 1},
    }
    assert rank_evidence("Liability is covered in Kosovo.", windows)[0] == "specific"


def test_audit_rejects_omitted_condition_without_seeing_evidence():
    class AuditReject(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is ExtractionAudit:
                assert "evidence" not in payload
                return ExtractionAudit(faithful_and_complete=False, reason="Condition omitted")
            return super().call(instruction, payload, schema)
    _, result = evaluate_claim_groundedness("Repairs are paid if authorized.", [Document(page_content="Repairs are paid.")], judge=AuditReject())
    assert not result["all_claims_supported"]
    assert not result["applied_caps"]


def test_claim_post_validation_uses_decision_status_field():
    _, result = evaluate(
        "Insurance covers repairs.",
        "No information.",
        quote="Insurance covers repairs.",
    )
    detail = result["claim_details"][0]
    assert detail["decision_status"] == "relation_changed_by_post_validation"
    assert "evaluation_status" not in detail

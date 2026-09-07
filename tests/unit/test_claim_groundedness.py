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
    assert result["failure_stage"] == "model_or_provider_call"


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
    assert result["failure_stage"] == "claim_extraction_validation"
    assert result["failure_code"] == "incomplete_claim_extraction"
    assert "missing required unit IDs" in result["exception_message"]


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
    assert result["failure_stage"] == "claim_extraction_audit"
    assert result["failure_details"]["attempts"] == 2


def test_claim_post_validation_uses_decision_status_field():
    _, result = evaluate(
        "Insurance covers repairs.",
        "No information.",
        quote="Insurance covers repairs.",
    )
    detail = result["claim_details"][0]
    assert detail["decision_status"] == "relation_changed_by_post_validation"
    assert "evaluation_status" not in detail


def test_generic_intro_heading_is_context_only_for_complete_bullets():
    answer = """The territorial scope of the motor vehicle insurance is as follows:
- Valid in Switzerland and the Principality of Liechtenstein.
- Not valid in Belarus and Syria.
- In Kosovo, insurance does not apply to liability.
- For transport by sea, insurance cover will not be interrupted if departure and destination lie within the territorial scope.
Source: [motor-vehicle-insurance-sti.pdf, page 6]"""

    class HeadingOmittingJudge(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is Extraction:
                return Extraction(claims=[
                    {"text": text, "unit_ids": [i]}
                    for i, text in payload["units"].items()
                    if i != 0
                ])
            return super().call(instruction, payload, schema)

    score, result = evaluate_claim_groundedness(
        answer,
        [Document(page_content=answer)],
        judge=HeadingOmittingJudge(),
    )
    assert score == 1
    assert result["evaluation_status"] == "success"
    assert result["exception_type"] is None


def test_fragment_list_item_must_keep_heading_context():
    class MissingHeadingContext(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is Extraction:
                return Extraction(claims=[
                    {"text": "Not covered in Kosovo.", "unit_ids": [1]},
                    {"text": "Not covered in Albania.", "unit_ids": [2]},
                ])
            return super().call(instruction, payload, schema)

    _, result = evaluate_claim_groundedness(
        "Not covered in:\n- Kosovo\n- Albania",
        [Document(page_content="Not covered in Kosovo and Albania.")],
        judge=MissingHeadingContext(),
    )
    assert result["evaluation_status"] == "failed"
    assert result["failure_stage"] == "claim_extraction_validation"
    assert "List heading context missing" in result["exception_message"]


def test_missing_verdict_is_recovered_with_targeted_second_call():
    class MissingVerdictOnce(FakeJudge):
        def __init__(self):
            super().__init__()
            self.judgment_calls = 0

        def call(self, instruction, payload, schema):
            if schema in {Extraction, ExtractionAudit}:
                return super().call(instruction, payload, schema)
            self.judgment_calls += 1
            claims = payload["claims"][:-1] if self.judgment_calls == 1 else payload["claims"]
            verdicts = []
            for claim in claims:
                key = claim["evidence_ids"][0]
                verdicts.append({
                    "claim_id": claim["claim_id"],
                    "relation": "supported",
                    "confidence": 0.99,
                    "subject_scope_match": True,
                    "citations": [{"evidence_id": key, "quote": payload["evidence"][key]["text"]}],
                    "checks": {name: "not_applicable" for name in CATEGORIES},
                    "reason": "test semantic judgment",
                })
            return Judgments(verdicts=verdicts)

    score, result = evaluate_claim_groundedness(
        "One claim. Another claim.",
        [Document(page_content="One claim. Another claim.")],
        judge=MissingVerdictOnce(),
    )
    assert score == 1
    assert result["evaluation_status"] == "success"
    assert result["judgment_attempts"] == 2


def test_audit_failure_can_recover_once():
    class AuditFailsOnce(FakeJudge):
        def __init__(self):
            super().__init__()
            self.audit_calls = 0

        def call(self, instruction, payload, schema):
            if schema is ExtractionAudit:
                self.audit_calls += 1
                if self.audit_calls == 1:
                    return ExtractionAudit(faithful_and_complete=False, reason="Condition omitted")
                return ExtractionAudit(faithful_and_complete=True, reason="repaired")
            return super().call(instruction, payload, schema)

    score, result = evaluate_claim_groundedness(
        "Repairs are paid if authorized.",
        [Document(page_content="Repairs are paid if authorized.")],
        judge=AuditFailsOnce(),
    )
    assert score == 1
    assert result["evaluation_status"] == "success"
    assert result["extraction_attempts"] == 2


def test_large_document_context_is_trimmed_instead_of_failing_budget():
    source = "Insurance covers repairs. " + ("background text " * 14000)
    score, result = evaluate_claim_groundedness(
        "Insurance covers repairs.",
        [Document(page_content=source)],
        judge=FakeJudge(quote="Insurance covers repairs."),
    )
    assert score == 1
    assert result["evaluation_status"] == "success"
    assert result["payload_chars"] <= 150000

from unittest.mock import patch

from langchain_core.documents import Document

from src.core.answer_completeness import build_answer_requirements
from src.api.rag_service import _groundedness_recovery_action, SafetyResult
from src.guardrails.integrations import nemo_official, nemo_actions
from src.config.models import load_model_settings
from src.core.insurance_tool_routing import infer_document_source_filename
from src.core.claim_groundedness import (
    _answer_citations_match, Citation, Claim, Judgments, evaluate_claim_groundedness,
)
from tests.unit.test_claim_groundedness import FakeJudge


def test_answer_citation_must_name_the_supporting_source():
    docs = [Document(page_content="The insurer pays for repairs.", metadata={"source": "terms.pdf", "page": 0})]
    unit = "The insurer pays for repairs."
    for filename, expected in [("terms.pdf", True), ("invented.pdf", False)]:
        assert _answer_citations_match(f"{unit} [{filename}, page 1]", [unit],
            Claim(text=unit, unit_ids=[0]), [Citation(evidence_id="d0:0", quote=unit)], docs) is expected


def test_judge_evidence_is_scoped_to_the_answers_citation():
    class SourceCheckingJudge(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is Judgments:
                assert all(key.startswith("d1:") for c in payload["claims"] for key in c["evidence_ids"])
            return super().call(instruction, payload, schema)

    docs = [Document(page_content="The insurer pays for repairs.", metadata={"source": name, "page": 0})
            for name in ("unrelated.pdf", "terms.pdf")]
    score, result = evaluate_claim_groundedness(
        "The insurer pays for repairs. [terms.pdf, page 1]", docs, judge=SourceCheckingJudge())
    assert score == 1
    assert result["all_claims_supported"]


def test_invalid_judge_quote_gets_one_exact_quote_repair():
    class QuoteRepairJudge(FakeJudge):
        attempts = 0

        def call(self, instruction, payload, schema):
            if schema is Judgments:
                self.attempts += 1
                self.quote = "A fabricated quote with no provenance." if self.attempts == 1 else None
            return super().call(instruction, payload, schema)

    judge = QuoteRepairJudge()
    score, result = evaluate_claim_groundedness(
        "The insurer pays for repairs.", [Document(page_content="The insurer pays for repairs.")], judge=judge)
    assert judge.attempts == 2
    assert score == 1
    assert result["all_claims_supported"]


def test_explicit_brochure_and_sti_select_different_sources():
    assert infer_document_source_filename('How does brochure household contents and private liability describe coverage?') == 'brochure-household-contents-and-private-liability.pdf'
    assert infer_document_source_filename('What does household contents private liability sti say?') == 'household-contents-private-liability-sti.pdf'
    assert infer_document_source_filename('What does assistance sti say?') == 'assistance-sti.pdf'
    assert infer_document_source_filename('According to household contents private liability sti, what does it state about "e-bikes whose electrical assistance has been"?') == 'household-contents-private-liability-sti.pdf'


def test_quoted_clause_does_not_inherit_product_boilerplate():
    docs = [Document(page_content="Statutory liability and unjustified claims include property damage.")]
    assert not build_answer_requirements(
        'How does household contents private liability describe "Owners of holiday homes"?', docs
    )


def test_failed_and_unknown_evaluators_retry_without_rewriting():
    for status, reason in [("failed", "groundedness_evaluator_failed"),
                           ("uncertain", "groundedness_repair_needed")]:
        result = SafetyResult(allow=False, action="fallback", risk_level="medium",
                              reasons=[reason], details={"groundedness": {"evaluation_status": status}})
        assert _groundedness_recovery_action(result) == "retry_evaluation"


def test_sensitive_unsupported_can_repair_but_injection_cannot():
    result = SafetyResult(allow=False, action="fallback", risk_level="medium",
                          reasons=["groundedness_sensitive_unsupported"],
                          details={"groundedness": {"claim_details": [{"relation": "insufficient_evidence"}]}})
    assert _groundedness_recovery_action(result) == "repair_answer"
    result.reasons.append("prompt_injection_signal_in_answer")
    assert _groundedness_recovery_action(result) == "none"


def test_missing_nemo_action_executes_same_safety_checks():
    runtime = object.__new__(nemo_official.OfficialNemoGuardrailsRuntime)
    runtime.config = load_model_settings().safety
    runtime._output_rails = object()
    decision = {"action": "fallback", "allow": False, "reasons": ["groundedness_evaluator_failed"],
                "scores": {"groundedness": None}, "details": {"groundedness": {"evaluation_status": "failed"}}}
    with patch.object(runtime, "_generate", return_value={}), patch.object(
        nemo_actions, "_evaluate_output_safety", return_value=(decision, "draft")
    ) as check:
        result = runtime.run_post_generation("query", "draft", [])
    check.assert_called_once()
    assert result.action == "fallback"
    assert result.details["groundedness"]["evaluation_status"] == "failed"
    assert "groundedness_contradicted" not in result.reasons

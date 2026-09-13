import pytest
from langchain_core.documents import Document

from src.core.insurance_tool_routing import plan_insurance_query, enrich_knowledge_query, QueryMode
from src.core.claim_groundedness import ExtractionAudit, evaluate_claim_groundedness
from src.core.safety_adapter import classify_safety_result
from src.core.safety_audit import SafetyResult
from tests.unit.test_claim_groundedness import FakeJudge


@pytest.mark.parametrize('prefix', ['For', 'Does', 'Regarding'])
def test_leading_word_does_not_consume_customer_name(prefix):
    plan = plan_insurance_query(f'{prefix} Noah Keller, does his current motor vehicle policy cover theft of personal belongings?')
    assert plan.customer_name == 'Noah Keller'
    assert plan.mode is QueryMode.COMBINED


def test_theft_expansion_preserves_requested_object():
    query = 'Does motor vehicle insurance cover theft of personal belongings from the car?'
    assert query in enrich_knowledge_query(query)


@pytest.mark.parametrize('relation', ['supported', 'contradicted'])
def test_extraction_audit_failure_still_requires_entailment(relation):
    class RejectingAudit(FakeJudge):
        def call(self, instruction, payload, schema):
            if schema is ExtractionAudit:
                return ExtractionAudit(faithful_and_complete=bool(payload.get('verbatim_recovery')), reason='Extraction omitted a qualifier')
            return super().call(instruction, payload, schema)

    _, result = evaluate_claim_groundedness(
        'The insurer pays for repairs.',
        [Document(page_content='The insurer pays for repairs.')],
        judge=RejectingAudit(relation=relation),
    )
    assert result['evaluation_status'] == 'success'
    assert result['all_claims_supported'] is (relation == 'supported')
    assert result['extraction_audit_reason'] == 'verbatim_units_after_extraction_audit_failure'


def test_grounding_failure_is_not_reported_as_policy_violation():
    result = SafetyResult(allow=False, action='fallback', risk_level='medium', reasons=['groundedness_evaluator_failed'])
    assert classify_safety_result(result) == 'grounding'
    result.reasons.append('prompt_injection_signal_in_answer')
    assert classify_safety_result(result) == 'security'

import pytest


@pytest.fixture
def stub_entailment_provider(monkeypatch):
    """Mock only the network boundary in routing/PII tests, not the evaluator."""
    from tests.unit.test_claim_groundedness import FakeJudge
    from src.core import claim_groundedness
    monkeypatch.setattr(claim_groundedness, "ModelJudge", FakeJudge)

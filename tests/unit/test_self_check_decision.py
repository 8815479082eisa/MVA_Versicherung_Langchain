import pytest

from src.api.rag_service import _parse_self_check_decision
from src.core.llm_runtime import GuardrailInvalidOutputError


def test_self_check_accepts_exact_normalized_label() -> None:
    assert _parse_self_check_decision("  relevant  ") == "RELEVANT"


@pytest.mark.parametrize(
    "content",
    [
        "RELEVANT\nBecause the context answers the question.",
        "The context appears RELEVANT.",
        "",
    ],
)
def test_self_check_rejects_verbose_or_missing_label(content: str) -> None:
    with pytest.raises(GuardrailInvalidOutputError):
        _parse_self_check_decision(content)

from __future__ import annotations

from dataclasses import replace

from src.api import rag_service


class _FakePrompt:
    def __or__(self, llm):
        return llm


class _FakePromptTemplate:
    @staticmethod
    def from_messages(_messages):
        return _FakePrompt()


class _FakeRewriteLLM:
    def __init__(self, rewritten_query: str):
        self.rewritten_query = rewritten_query

    def invoke(self, _payload):
        return type("Response", (), {"content": self.rewritten_query})()


def _settings_with_query_rewrite(*, enabled: bool, min_similarity: float = 0.85):
    retrieval = replace(
        rag_service.SETTINGS.retrieval,
        query_rewrite_enabled=enabled,
        query_rewrite_min_similarity=min_similarity,
    )
    return replace(rag_service.SETTINGS, retrieval=retrieval)


def test_rewrite_query_returns_original_when_disabled(monkeypatch) -> None:
    original_query = "Can Homeowners Insurance Cancel Your Policy?"
    monkeypatch.setattr(rag_service, "SETTINGS", _settings_with_query_rewrite(enabled=False))

    assert rag_service.rewrite_query(object(), original_query) == original_query


def test_rewrite_query_rejects_semantic_drift(monkeypatch) -> None:
    original_query = "Can Homeowners Insurance Cancel Your Policy?"
    drifted_query = "How can I cancel my homeowner's insurance policy?"

    monkeypatch.setattr(rag_service, "SETTINGS", _settings_with_query_rewrite(enabled=True, min_similarity=0.1))
    monkeypatch.setattr(rag_service, "_get_chat_prompt_template_cls", lambda: _FakePromptTemplate)

    rewritten = rag_service.rewrite_query(_FakeRewriteLLM(drifted_query), original_query)

    assert rewritten == original_query

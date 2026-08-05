from __future__ import annotations

import asyncio
from dataclasses import replace

from src.config.models import load_model_settings
from src.core import ollama_readiness


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"models": [{"name": "phi3:mini"}]}


class _Client:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, url: str):
        assert url.endswith("/api/tags")
        return _Response()


def test_online_answer_readiness_still_requires_local_guardrail(monkeypatch) -> None:
    settings = load_model_settings()
    settings = replace(
        settings,
        provider="openai",
        openai_api_key_configured=True,
        roles=replace(settings.roles, answer="gpt-4o-mini", guardrail="phi3:mini"),
    )
    monkeypatch.setattr(
        ollama_readiness.httpx,
        "AsyncClient",
        lambda **_: _Client(),
    )

    health = asyncio.run(ollama_readiness.check_ollama_readiness(settings))

    assert health["answerProvider"] == "openai"
    assert health["answerModelReady"] is True
    assert health["guardrailModelReady"] is True
    assert health["llmReady"] is True


def test_online_answer_readiness_reports_missing_key_without_api_call(
    monkeypatch,
) -> None:
    settings = load_model_settings()
    settings = replace(
        settings,
        provider="openai",
        openai_api_key_configured=False,
        roles=replace(settings.roles, answer="gpt-4o-mini", guardrail="phi3:mini"),
    )
    monkeypatch.setattr(
        ollama_readiness.httpx,
        "AsyncClient",
        lambda **_: _Client(),
    )

    health = asyncio.run(ollama_readiness.check_ollama_readiness(settings))

    assert health["answerModelReady"] is False
    assert health["guardrailModelReady"] is True
    assert health["llmReady"] is False
    assert "OpenAI API key" in health["llmError"]

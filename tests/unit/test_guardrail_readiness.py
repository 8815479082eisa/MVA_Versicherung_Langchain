from __future__ import annotations

import asyncio

from src.core import guardrail_readiness


def test_guardrail_readiness_requires_real_384_dimension_inference(monkeypatch) -> None:
    guardrail_readiness._load_and_probe_guardrail_embedding.cache_clear()
    monkeypatch.setattr(
        guardrail_readiness,
        "_load_and_probe_guardrail_embedding",
        lambda: 384,
    )

    result = asyncio.run(guardrail_readiness.check_guardrail_model_readiness())

    assert result["guardrailModelReady"] is True
    assert result["guardrailEmbeddingDimension"] == 384
    assert result["guardrailModelError"] is None


def test_guardrail_readiness_fails_closed_on_incomplete_model(monkeypatch) -> None:
    def _raise() -> int:
        raise FileNotFoundError("model.onnx")

    monkeypatch.setattr(
        guardrail_readiness,
        "_load_and_probe_guardrail_embedding",
        _raise,
    )

    result = asyncio.run(guardrail_readiness.check_guardrail_model_readiness())

    assert result["guardrailModelReady"] is False
    assert result["guardrailEmbeddingDimension"] is None
    assert "FileNotFoundError" in result["guardrailModelError"]

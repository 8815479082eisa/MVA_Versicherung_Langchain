from __future__ import annotations

from typing import Any

import httpx

from src.config.models import ModelSettings


def _available_model_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for model in payload.get("models") or []:
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


def model_is_available(required: str, available: set[str]) -> bool:
    if required in available:
        return True
    if ":" not in required and f"{required}:latest" in available:
        return True
    return False


async def check_ollama_readiness(
    settings: ModelSettings,
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    answer_ready = (
        settings.openai_api_key_configured
        if settings.provider == "openai"
        else False
    )
    guardrail_ready = False
    try:
        async with httpx.AsyncClient(timeout=max(0.1, timeout_seconds)) as client:
            response = await client.get(
                f"{settings.ollama_base_url.rstrip('/')}/api/tags"
            )
            response.raise_for_status()
            available = _available_model_names(response.json())
        if settings.provider == "ollama":
            answer_ready = model_is_available(settings.roles.answer, available)
        guardrail_ready = model_is_available(
            settings.roles.guardrail,
            available,
        )
        missing = []
        if not answer_ready:
            missing.append(
                "OpenAI API key"
                if settings.provider == "openai"
                else "answer model"
            )
        if not guardrail_ready:
            missing.append("local guardrail model")
        return {
            "answerProvider": settings.provider,
            "ollamaReachable": True,
            "llmReady": answer_ready and guardrail_ready,
            "llmError": (
                None
                if not missing
                else "Required model unavailable: " + ", ".join(missing)
            ),
            "answerModelReady": answer_ready,
            "guardrailModelReady": guardrail_ready,
        }
    except Exception as exc:
        return {
            "answerProvider": settings.provider,
            "ollamaReachable": False,
            "llmReady": False,
            "llmError": f"Ollama readiness check failed ({type(exc).__name__}).",
            "answerModelReady": answer_ready,
            "guardrailModelReady": False,
        }

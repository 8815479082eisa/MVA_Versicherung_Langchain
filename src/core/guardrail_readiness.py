from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any


GUARDRAIL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_and_probe_guardrail_embedding() -> int:
    """Load the exact NeMo FastEmbed model and run one real inference."""
    from fastembed import TextEmbedding

    cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
    model = TextEmbedding(GUARDRAIL_EMBEDDING_MODEL, cache_dir=cache_dir)
    vector = next(iter(model.embed(["guardrail readiness probe"])))
    return int(len(vector))


async def check_guardrail_model_readiness(
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Verify that the complete ONNX model can perform an embedding inference."""
    try:
        dimension = await asyncio.wait_for(
            asyncio.to_thread(_load_and_probe_guardrail_embedding),
            timeout=max(0.1, timeout_seconds),
        )
        if dimension != 384:
            raise ValueError(f"Unexpected embedding dimension: {dimension}")
        return {
            "guardrailModelReady": True,
            "guardrailEmbeddingModel": GUARDRAIL_EMBEDDING_MODEL,
            "guardrailEmbeddingDimension": dimension,
            "guardrailModelError": None,
        }
    except Exception as exc:
        return {
            "guardrailModelReady": False,
            "guardrailEmbeddingModel": GUARDRAIL_EMBEDDING_MODEL,
            "guardrailEmbeddingDimension": None,
            "guardrailModelError": (
                f"Guardrail embedding readiness failed ({type(exc).__name__})."
            ),
        }

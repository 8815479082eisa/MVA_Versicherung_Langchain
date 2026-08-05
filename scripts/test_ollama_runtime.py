from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.models import load_model_settings
from src.core.ollama_readiness import model_is_available


def _model_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for row in payload.get("models") or []:
        if not isinstance(row, dict):
            continue
        for key in ("name", "model"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


def _smoke(
    client: httpx.Client,
    *,
    base_url: str,
    stage: str,
    model: str,
    num_predict: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = (
        "Return exactly one label: SAFE or UNSAFE. Do not explain. "
        "Input: ordinary insurance information question."
        if stage == "guardrail"
        else "Reply with one short word confirming that you are ready."
    )
    started_at = time.perf_counter()
    try:
        response = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = str(response.json().get("response") or "").strip()
        valid = bool(content)
        if stage == "guardrail":
            valid = " ".join(content.upper().split()) in {"SAFE", "UNSAFE"}
        return {
            "status": "PASSED" if valid else "FAILED",
            "stage": stage,
            "model": model,
            "numPredict": num_predict,
            "timeoutSeconds": timeout_seconds,
            "latencyMs": round((time.perf_counter() - started_at) * 1000, 3),
            "nonEmptyOutput": bool(content),
            "closedVocabularyOutput": valid if stage == "guardrail" else None,
        }
    except httpx.TimeoutException:
        return {
            "status": "FAILED",
            "stage": stage,
            "model": model,
            "numPredict": num_predict,
            "timeoutSeconds": timeout_seconds,
            "latencyMs": round((time.perf_counter() - started_at) * 1000, 3),
            "errorCode": "LLM_STAGE_TIMEOUT",
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "stage": stage,
            "model": model,
            "numPredict": num_predict,
            "timeoutSeconds": timeout_seconds,
            "latencyMs": round((time.perf_counter() - started_at) * 1000, 3),
            "errorCode": "LLM_UNAVAILABLE",
            "errorType": type(exc).__name__,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("answer", "guardrail", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = load_model_settings()
    base_url = settings.ollama_base_url.rstrip("/")
    payload: dict[str, Any] = {
        "status": "PASSED",
        "ollamaUrl": base_url,
        "modelList": {},
        "smokeTests": [],
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            started_at = time.perf_counter()
            tags_response = client.get(f"{base_url}/api/tags")
            tags_response.raise_for_status()
            names = _model_names(tags_response.json())
            required = {
                "answer": settings.roles.answer,
                "guardrail": settings.roles.guardrail,
            }
            payload["modelList"] = {
                "status": "PASSED",
                "latencyMs": round((time.perf_counter() - started_at) * 1000, 3),
                "requiredModels": {
                    stage: {
                        "model": model,
                        "available": model_is_available(model, names),
                    }
                    for stage, model in required.items()
                },
            }
            selected = (
                ("answer", "guardrail")
                if args.stage == "all"
                else (args.stage,)
            )
            for stage in selected:
                model = required[stage]
                if not model_is_available(model, names):
                    payload["smokeTests"].append(
                        {
                            "status": "FAILED",
                            "stage": stage,
                            "model": model,
                            "errorCode": "LLM_MODEL_MISSING",
                        }
                    )
                    continue
                payload["smokeTests"].append(
                    _smoke(
                        client,
                        base_url=base_url,
                        stage=stage,
                        model=model,
                        num_predict=settings.llm_runtime.output_limit(stage),
                        timeout_seconds=settings.llm_runtime.timeout(stage),
                    )
                )
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["modelList"] = {
            "status": "FAILED",
            "errorCode": "LLM_UNAVAILABLE",
            "errorType": type(exc).__name__,
        }

    if any(row.get("status") != "PASSED" for row in payload["smokeTests"]):
        payload["status"] = "FAILED"
    if payload["modelList"].get("status") != "PASSED":
        payload["status"] = "FAILED"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "reports" / "llm_runtime_benchmark.json"
BACKEND_URL = os.getenv(
    "LLM_BENCHMARK_BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")
CALLS = max(1, int(os.getenv("LLM_BENCHMARK_CALLS", "1")))
TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("LLM_BENCHMARK_TIMEOUT_SECONDS", "190")),
)
PURE_RETRIEVAL_FILES = {
    "withoutReranker": PROJECT_ROOT
    / "reports"
    / ".llm_pure_retrieval_without.json",
    "withReranker": PROJECT_ROOT
    / "reports"
    / ".llm_pure_retrieval_with.json",
}
SMOKE_FILES = {
    "answer": PROJECT_ROOT / "reports" / ".llm_smoke_answer.json",
    "guardrail": PROJECT_ROOT / "reports" / ".llm_smoke_guardrail.json",
}

SCENARIOS = {
    "crm_only": "Which active policies does Lara Neumann have?",
    "retrieval_only": "What does partial coverage generally cover?",
    "combined": (
        "Lara Neumann reported glass damage. Is the damage covered according "
        "to the conditions, and what is the current claim status?"
    ),
}


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50Ms": None, "p95Ms": None, "meanMs": None}
    ordered = sorted(values)
    p95 = ordered[max(0, int(0.95 * len(ordered) + 0.9999) - 1)]
    return {
        "p50Ms": round(statistics.median(ordered), 3),
        "p95Ms": round(p95, 3),
        "meanMs": round(statistics.fmean(ordered), 3),
    }


async def _scenario(
    client: httpx.AsyncClient,
    name: str,
    question: str,
) -> dict[str, Any]:
    latencies: list[float] = []
    responses: list[dict[str, Any]] = []
    for _ in range(CALLS):
        started_at = time.perf_counter()
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/ask",
                json={"question": question},
            )
            latency_ms = (time.perf_counter() - started_at) * 1000
            body = response.json()
            latencies.append(latency_ms)
            responses.append(
                {
                    "httpStatus": response.status_code,
                    "resultStatus": body.get("status")
                    or body.get("detail", {}).get("status"),
                    "errorCode": body.get("detail", {}).get("errorCode"),
                    "diagnostics": body.get("diagnostics")
                    or body.get("detail", {}).get("diagnostics"),
                }
            )
        except httpx.TimeoutException:
            responses.append(
                {
                    "httpStatus": None,
                    "resultStatus": "error",
                    "errorCode": "CLIENT_TIMEOUT",
                    "diagnostics": None,
                }
            )
    complete = all(
        row["httpStatus"] == 200 and row["resultStatus"] == "complete"
        for row in responses
    )
    return {
        "status": "PASSED" if complete else "FAILED",
        "scenario": name,
        "calls": CALLS,
        **_stats(latencies),
        "responses": responses,
    }


def _docker_stats() -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        return {"status": "NOT RUN", "reason": "Docker CLI unavailable."}
    completed = subprocess.run(
        [
            docker,
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            "mva-backend",
            "mva-espocrm",
            "mva-espocrm-db",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    rows = []
    for line in completed.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {
        "status": "PASSED" if completed.returncode == 0 else "FAILED",
        "containers": rows,
    }


def _load_pure_retrieval_results() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, path in PURE_RETRIEVAL_FILES.items():
        if not path.exists():
            results[name] = {
                "status": "NOT RUN",
                "reason": "No saved live pure-retrieval result was found.",
            }
            continue
        try:
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            results[name] = {
                "status": "FAILED",
                "reason": "Saved pure-retrieval result is invalid.",
            }
    return results


def _load_saved_results(paths: dict[str, Path]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.exists():
            results[name] = {"status": "NOT RUN"}
            continue
        try:
            results[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            results[name] = {"status": "FAILED", "reason": "Invalid JSON artifact."}
    return results


async def main() -> int:
    payload: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "backendUrl": BACKEND_URL,
        "callsPerScenario": CALLS,
        "clientTimeoutSeconds": TIMEOUT_SECONDS,
        "scenarios": {},
        "pureRetrieval": _load_pure_retrieval_results(),
        "ollamaSmoke": _load_saved_results(SMOKE_FILES),
        "resourceSnapshot": {},
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for name, question in SCENARIOS.items():
            payload["scenarios"][name] = await _scenario(
                client,
                name,
                question,
            )
    payload["resourceSnapshot"] = _docker_stats()
    payload["status"] = (
        "PASSED"
        if all(
            row["status"] == "PASSED"
            for row in payload["scenarios"].values()
        )
        else "FAILED"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.espocrm_client import EspoCRMClient, EspoCRMConfig
from src.mcp_clients.insurance_tools import PersistentMCPToolRuntime


POLICY_NUMBER = "TEST-KFZ-2026-1001"
MICRO_CALLS = max(50, int(os.getenv("CRM_BENCHMARK_CALLS", "50")))
WARMUP_CALLS = max(1, int(os.getenv("CRM_BENCHMARK_WARMUP_CALLS", "5")))
E2E_CALLS = max(1, int(os.getenv("CRM_E2E_BENCHMARK_CALLS", "3")))
E2E_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("CRM_E2E_BENCHMARK_TIMEOUT_SECONDS", "180")),
)
OUTPUT_PATH = PROJECT_ROOT / "reports" / "espocrm_latency_benchmark.json"


class BenchmarkCallError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


async def _measure(
    operation: Callable[[], Awaitable[Any]],
    *,
    count: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    failures: list[str] = []
    timeout_count = 0
    for _ in range(count):
        started = time.perf_counter()
        try:
            await operation()
            latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or getattr(
                exc,
                "category",
                None,
            ) == "timeout":
                timeout_count += 1
            failures.append(type(exc).__name__)
    return {
        "requested_calls": count,
        "successful_calls": len(latencies),
        "success_rate": len(latencies) / count if count else 0,
        "timeout_count": timeout_count,
        "failure_types": sorted(set(failures)),
        **_latency_stats(latencies),
    }


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "maximum_ms": None,
            "mean_ms": None,
        }
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.9999) - 1))
    return {
        "minimum_ms": round(ordered[0], 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "maximum_ms": round(ordered[-1], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


async def _benchmark_crm() -> dict[str, Any]:
    public_url = os.getenv("ESPOCRM_PUBLIC_URL", "").strip()
    api_key = os.getenv("ESPOCRM_API_KEY", "").strip()
    if not public_url or not api_key:
        return {
            "status": "skipped",
            "reason": (
                "ESPOCRM_PUBLIC_URL and ESPOCRM_API_KEY are required for the "
                "live REST and MCP benchmark."
            ),
        }

    config = EspoCRMConfig(
        enabled=True,
        base_url=public_url,
        api_key=api_key,
        request_timeout_seconds=float(
            os.getenv("ESPOCRM_REQUEST_TIMEOUT_SECONDS", "3")
        ),
        max_pages=10,
        page_size=50,
    )
    result: dict[str, Any] = {"status": "completed"}
    async with EspoCRMClient(config) as client:
        async def direct_policy_call():
            payload = await client.get_policy(POLICY_NUMBER)
            if payload.get("found") is not True:
                raise BenchmarkCallError(
                    "not_found",
                    f"Synthetic policy {POLICY_NUMBER} is not seeded.",
                )
            return payload

        cold_started = time.perf_counter()
        await direct_policy_call()
        result["direct_rest_cold_ms"] = round(
            (time.perf_counter() - cold_started) * 1000,
            3,
        )
        for _ in range(WARMUP_CALLS):
            await direct_policy_call()
        result["direct_rest_warm"] = await _measure(
            direct_policy_call,
            count=MICRO_CALLS,
        )

    previous_enabled = os.environ.get("CRM_ENABLED")
    previous_base_url = os.environ.get("ESPOCRM_BASE_URL")
    os.environ["CRM_ENABLED"] = "true"
    os.environ["ESPOCRM_BASE_URL"] = public_url
    runtime = PersistentMCPToolRuntime()
    try:
        async def mcp_policy_call():
            payload = await runtime.invoke(
                "get_policy",
                {"policy_number": POLICY_NUMBER},
            )
            if payload.get("ok") is not True:
                error = payload.get("error") or {}
                raise BenchmarkCallError(
                    str(error.get("category") or "mcp_error"),
                    str(error.get("message") or "CRM MCP call failed."),
                )
            if payload.get("found") is not True:
                raise BenchmarkCallError(
                    "not_found",
                    f"Synthetic policy {POLICY_NUMBER} is not seeded.",
                )
            return payload

        startup_started = time.perf_counter()
        await runtime.start(("insurance_crm",))
        result["mcp_process_startup_ms"] = round(
            (time.perf_counter() - startup_started) * 1000,
            3,
        )
        cold_started = time.perf_counter()
        await mcp_policy_call()
        result["mcp_cold_call_ms"] = round(
            (time.perf_counter() - cold_started) * 1000,
            3,
        )
        for _ in range(WARMUP_CALLS):
            await mcp_policy_call()
        result["mcp_warm"] = await _measure(
            mcp_policy_call,
            count=MICRO_CALLS,
        )
    finally:
        await runtime.close()
        _restore_environment("CRM_ENABLED", previous_enabled)
        _restore_environment("ESPOCRM_BASE_URL", previous_base_url)

    direct = result["direct_rest_warm"]
    mcp = result["mcp_warm"]
    result["mcp_overhead"] = {
        "p50_ms": _difference(mcp.get("p50_ms"), direct.get("p50_ms")),
        "p95_ms": _difference(mcp.get("p95_ms"), direct.get("p95_ms")),
        "mean_ms": _difference(mcp.get("mean_ms"), direct.get("mean_ms")),
    }
    return result


async def _benchmark_backend() -> dict[str, Any]:
    backend_url = os.getenv(
        "CRM_BENCHMARK_BACKEND_URL",
        "http://localhost:8000",
    ).rstrip("/")
    timeout = httpx.Timeout(E2E_TIMEOUT_SECONDS)
    scenarios = {
        "crm_only": "Which active policies does Lara Neumann have?",
        "retrieval_only_after_integration": "What does partial coverage generally cover?",
        "combined": (
            "Lara Neumann reported glass damage. Is the damage covered by her "
            "current policy, and what is the current claim status?"
        ),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            health = await client.get(f"{backend_url}/health")
            health.raise_for_status()
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": (
                    f"Backend at {backend_url} is unavailable "
                    f"({type(exc).__name__})."
                ),
                "retrieval_only_before_integration": {
                    "status": "not_reproducible",
                    "reason": "No controlled pre-integration server is available.",
                },
            }

        results: dict[str, Any] = {
            "status": "completed",
            "backend_health": health.json(),
            "retrieval_only_before_integration": {
                "status": "not_reproducible",
                "reason": "No controlled pre-integration server is available.",
            },
        }
        for name, question in scenarios.items():
            async def call(question_to_send=question):
                response = await client.post(
                    f"{backend_url}/api/ask",
                    json={"question": question_to_send},
                )
                response.raise_for_status()
                return response.json()

            try:
                await call()
                results[name] = await _measure(call, count=E2E_CALLS)
            except Exception as exc:
                status_code = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                results[name] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "http_status": status_code,
                    "timeout": isinstance(exc, httpx.TimeoutException),
                }
        if any(
            isinstance(results.get(name), dict)
            and results[name].get("status") == "failed"
            for name in scenarios
        ):
            results["status"] = "partial_failure"
        return results


def _docker_stats() -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        return {
            "status": "skipped",
            "reason": "Docker CLI is not installed or not on PATH.",
        }
    command = [
        docker,
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        "mva-backend",
        "mva-espocrm",
        "mva-espocrm-db",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "skipped",
            "reason": "CRM containers are not running or docker stats failed.",
        }
    rows = []
    for line in completed.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"status": "completed", "containers": rows}


def _difference(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(left - right, 3)


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


async def main() -> int:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_number": POLICY_NUMBER,
        "warmup_calls": WARMUP_CALLS,
        "microbenchmark_calls": MICRO_CALLS,
        "end_to_end_calls": E2E_CALLS,
        "end_to_end_timeout_seconds": E2E_TIMEOUT_SECONDS,
        "crm_microbenchmarks": await _benchmark_crm(),
        "end_to_end": await _benchmark_backend(),
        "docker_stats": _docker_stats(),
        "targets": {
            "direct_rest_p95_ms": 500,
            "additional_mcp_overhead_p95_ms": "100-200",
            "additional_warm_crm_e2e_ms": 1000,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

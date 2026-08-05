from __future__ import annotations

import logging
import re
import contextvars
import threading
from typing import Any, Callable, TypeVar

from src.core.runtime_diagnostics import add_retry, begin_stage, finish_stage


logger = logging.getLogger(__name__)
T = TypeVar("T")
_BOUNDED_STAGE_GATES: dict[str, threading.BoundedSemaphore] = {}
_BOUNDED_STAGE_GATES_LOCK = threading.Lock()


class RuntimeExecutionError(RuntimeError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        stage: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.safe_message = message
        self.stage = stage
        self.retryable = retryable

    def as_dict(self, *, route: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "errorCode": self.error_code,
            "message": self.safe_message,
        }
        if route:
            payload["route"] = route
        if self.stage:
            payload["stage"] = self.stage
        return payload


class LLMUnavailableError(RuntimeExecutionError):
    def __init__(self, *, stage: str | None = None):
        super().__init__(
            "LLM_UNAVAILABLE",
            "The local LLM runtime is unavailable.",
            stage=stage,
            retryable=True,
        )


class LLMModelMissingError(RuntimeExecutionError):
    def __init__(self, *, stage: str | None = None):
        super().__init__(
            "LLM_MODEL_MISSING",
            "A required local LLM model is not available.",
            stage=stage,
            retryable=False,
        )


class LLMStageTimeoutError(RuntimeExecutionError, TimeoutError):
    def __init__(self, stage: str):
        super().__init__(
            "LLM_STAGE_TIMEOUT",
            f"The {stage.replace('_', ' ')} stage exceeded its configured timeout.",
            stage=stage,
            retryable=False,
        )


class RequestTimeoutError(RuntimeExecutionError):
    def __init__(self):
        super().__init__(
            "REQUEST_TIMEOUT",
            "The request exceeded its configured total timeout.",
            stage="request",
            retryable=False,
        )


class GuardrailInvalidOutputError(RuntimeExecutionError):
    def __init__(self, stage: str):
        super().__init__(
            "GUARDRAIL_INVALID_OUTPUT",
            f"The {stage.replace('_', ' ')} classifier returned an invalid output.",
            stage=stage,
            retryable=False,
        )


class RetrievalFailedError(RuntimeExecutionError):
    def __init__(self, *, stage: str = "retrieval"):
        super().__init__(
            "RETRIEVAL_FAILED",
            "Document retrieval could not be completed.",
            stage=stage,
            retryable=False,
        )


_TIMEOUT_PATTERNS = (
    "timed out",
    "timeout",
    "read operation timed out",
)
_MISSING_MODEL_PATTERNS = (
    "model not found",
    "not found, try pulling",
    "does not exist",
)
_UNAVAILABLE_PATTERNS = (
    "connection refused",
    "all connection attempts failed",
    "connection error",
    "failed to connect",
    "server disconnected",
    "ollama call failed",
)


def classify_llm_exception(exc: Exception, stage: str) -> RuntimeExecutionError:
    if isinstance(exc, RuntimeExecutionError):
        return exc

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or any(
        pattern in message for pattern in _TIMEOUT_PATTERNS
    ):
        return LLMStageTimeoutError(stage)
    if any(pattern in message for pattern in _MISSING_MODEL_PATTERNS):
        return LLMModelMissingError(stage=stage)
    if "connect" in name or any(pattern in message for pattern in _UNAVAILABLE_PATTERNS):
        return LLMUnavailableError(stage=stage)
    # Ollama response/transport errors are runtime availability failures unless
    # the repository has identified a more specific expected condition above.
    if name in {"responseerror", "requesterror", "httperror"}:
        return LLMUnavailableError(stage=stage)
    return RuntimeExecutionError(
        "INTERNAL_ERROR",
        "An unexpected internal error occurred.",
        stage=stage,
        retryable=False,
    )


def invoke_llm_stage(
    stage: str,
    operation: Callable[[], T],
    *,
    max_retries: int,
) -> T:
    attempt = 0
    while True:
        started_at = begin_stage(stage)
        try:
            result = operation()
            finish_stage(stage, started_at)
            return result
        except Exception as exc:
            runtime_error = classify_llm_exception(exc, stage)
            status = (
                "timeout"
                if runtime_error.error_code == "LLM_STAGE_TIMEOUT"
                else "failed"
            )
            finish_stage(stage, started_at, status=status)
            if (
                runtime_error.retryable
                and runtime_error.error_code != "LLM_STAGE_TIMEOUT"
                and attempt < max(0, max_retries)
            ):
                attempt += 1
                retry_count = add_retry(stage)
                logger.warning(
                    "LLM stage retry stage=%s retry=%d reason=%s",
                    stage,
                    retry_count or attempt,
                    runtime_error.error_code,
                )
                continue
            raise runtime_error from exc


def parse_closed_label(
    content: Any,
    *,
    allowed: set[str],
    stage: str,
) -> str:
    normalized = re.sub(r"\s+", " ", str(content or "")).strip().upper()
    if normalized not in allowed:
        raise GuardrailInvalidOutputError(stage)
    return normalized


def run_bounded_operation(
    stage: str,
    operation: Callable[[], T],
    *,
    timeout_seconds: float,
) -> T:
    """Run synchronous local work with bounded waiting and no thread buildup.

    Python cannot forcibly stop an arbitrary CPU-bound thread. A timed-out
    worker therefore keeps the per-stage gate until it really exits; later work
    fails fast instead of spawning unbounded abandoned workers.
    """

    with _BOUNDED_STAGE_GATES_LOCK:
        gate = _BOUNDED_STAGE_GATES.setdefault(
            stage,
            threading.BoundedSemaphore(value=1),
        )
    if not gate.acquire(timeout=max(0.001, timeout_seconds)):
        raise LLMStageTimeoutError(stage)

    result_box: dict[str, T] = {}
    error_box: dict[str, Exception] = {}
    caller_context = contextvars.copy_context()

    def _runner() -> None:
        try:
            result_box["value"] = caller_context.run(operation)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            error_box["error"] = exc
        finally:
            gate.release()

    worker = threading.Thread(
        target=_runner,
        daemon=True,
        name=f"bounded-{stage}",
    )
    worker.start()
    worker.join(timeout=max(0.001, timeout_seconds))
    if worker.is_alive():
        raise LLMStageTimeoutError(stage)
    if error_box:
        raise error_box["error"]
    return result_box["value"]

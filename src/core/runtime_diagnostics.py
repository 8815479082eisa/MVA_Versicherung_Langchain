from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


STAGE_TIMING_NAMES: dict[str, str] = {
    "route_planning": "routePlanningMs",
    "crm": "crmMs",
    "embedding": "embeddingMs",
    "lexical_retrieval": "lexicalRetrievalMs",
    "vector_retrieval": "vectorRetrievalMs",
    "hybrid_merge": "hybridMergeMs",
    "retrieval": "retrievalMs",
    "reranker": "rerankingMs",
    "guardrail": "guardrailMs",
    "groundedness": "groundednessMs",
    "router": "routerMs",
    "self_check": "selfCheckMs",
    "query_rewrite": "queryRewriteMs",
    "compressor": "compressionMs",
    "answer": "answerGenerationMs",
}


@dataclass
class RequestDiagnostics:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    route: Optional[str] = None
    started_at: float = field(default_factory=time.perf_counter)
    started_at_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at_iso: Optional[str] = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    stage_status: dict[str, str] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    timeout_stage: Optional[str] = None
    status: str = "running"
    partial: bool = False
    self_check_enabled: Optional[bool] = None
    self_check_skipped: bool = False
    self_check_skip_reason: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def start_stage(self, stage: str) -> float:
        self.stage_status[stage] = "running"
        return time.perf_counter()

    def finish_stage(
        self,
        stage: str,
        started_at: float,
        *,
        status: str = "completed",
    ) -> float:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        timing_name = STAGE_TIMING_NAMES.get(stage, f"{stage}Ms")
        self.stage_timings_ms[timing_name] = round(
            self.stage_timings_ms.get(timing_name, 0.0) + elapsed_ms,
            3,
        )
        self.stage_status[stage] = status
        if status == "timeout":
            self.timeout_stage = stage
        return elapsed_ms

    def mark_stage(self, stage: str, status: str) -> None:
        self.stage_status[stage] = status
        if status == "timeout":
            self.timeout_stage = stage

    def add_retry(self, stage: str) -> int:
        count = self.retries.get(stage, 0) + 1
        self.retries[stage] = count
        return count

    def configure_self_check(self, enabled: bool) -> None:
        self.self_check_enabled = bool(enabled)

    def mark_self_check_skipped(self, reason: str) -> None:
        self.self_check_enabled = False
        self.self_check_skipped = True
        self.self_check_skip_reason = reason
        self.mark_stage("self_check", "skipped")

    def complete(self, *, status: str, partial: bool = False) -> None:
        self.status = status
        self.partial = partial
        self.finished_at_iso = datetime.now(timezone.utc).isoformat()

    def record_evidence(self, key: str, value: Any) -> None:
        self.evidence[key] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "startedAt": self.started_at_iso,
            "finishedAt": self.finished_at_iso,
            "route": self.route,
            "stageTimings": dict(self.stage_timings_ms),
            "stageStatus": dict(self.stage_status),
            "timeoutStage": self.timeout_stage,
            "retries": dict(self.retries),
            "status": self.status,
            "partial": self.partial,
            "selfCheckEnabled": self.self_check_enabled,
            "selfCheckSkipped": self.self_check_skipped,
            "selfCheckSkipReason": self.self_check_skip_reason,
            "evidence": dict(self.evidence),
            "totalMs": round((time.perf_counter() - self.started_at) * 1000, 3),
        }


_CURRENT_DIAGNOSTICS: contextvars.ContextVar[Optional[RequestDiagnostics]] = (
    contextvars.ContextVar("current_request_diagnostics", default=None)
)


def set_current_diagnostics(
    diagnostics: Optional[RequestDiagnostics],
) -> contextvars.Token[Optional[RequestDiagnostics]]:
    return _CURRENT_DIAGNOSTICS.set(diagnostics)


def reset_current_diagnostics(
    token: contextvars.Token[Optional[RequestDiagnostics]],
) -> None:
    _CURRENT_DIAGNOSTICS.reset(token)


def current_diagnostics() -> Optional[RequestDiagnostics]:
    return _CURRENT_DIAGNOSTICS.get()


def begin_stage(stage: str) -> float:
    diagnostics = current_diagnostics()
    if diagnostics is None:
        return time.perf_counter()
    return diagnostics.start_stage(stage)


def finish_stage(stage: str, started_at: float, *, status: str = "completed") -> float:
    diagnostics = current_diagnostics()
    if diagnostics is None:
        return round((time.perf_counter() - started_at) * 1000, 3)
    return diagnostics.finish_stage(stage, started_at, status=status)


def mark_stage(stage: str, status: str) -> None:
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.mark_stage(stage, status)


def add_retry(stage: str) -> int:
    diagnostics = current_diagnostics()
    if diagnostics is None:
        return 0
    return diagnostics.add_retry(stage)


def configure_self_check(enabled: bool) -> None:
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.configure_self_check(enabled)


def mark_self_check_skipped(reason: str) -> None:
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.mark_self_check_skipped(reason)


def record_evidence(key: str, value: Any) -> None:
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.record_evidence(key, value)

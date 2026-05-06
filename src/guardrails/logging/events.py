from __future__ import annotations

from typing import Any, Dict, List

from ..types import TraceEntry


def trace_to_audit_payload(entries: List[TraceEntry]) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": e.timestamp,
            "stage": e.stage,
            "rail": e.rail,
            "decision": e.decision,
            "action": e.action,
            "reasons": list(e.reasons),
            "details": dict(e.details),
        }
        for e in entries
    ]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class StreamEvent:
    event_type: str
    payload: Dict[str, Any]


class StreamBuffer:
    """Simple stream buffer used by future token streaming integrations."""

    def __init__(self) -> None:
        self._events: list[StreamEvent] = []

    def push(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._events.append(StreamEvent(event_type=event_type, payload=dict(payload or {})))

    def __iter__(self) -> Iterator[StreamEvent]:
        return iter(self._events)

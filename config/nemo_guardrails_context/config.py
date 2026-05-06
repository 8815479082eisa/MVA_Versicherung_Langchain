from __future__ import annotations

from nemoguardrails import LLMRails

from src.guardrails.integrations.nemo_actions import register_context_actions


def init(app: LLMRails) -> None:
    register_context_actions(app)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from scripts.diagnose_combined_selfcheck import (
    _initial_diagnostic,
    _populate_from_response,
)
from src import main
from src.api import rag_service
from src.core.diagnostic_capture import atomic_write_json
from src.core.llm_runtime import GuardrailInvalidOutputError
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)


def test_atomic_diagnostic_json_is_unicode_safe_and_sanitized(tmp_path) -> None:
    path = tmp_path / "diagnostic.json"
    atomic_write_json(
        path,
        {
            "answer": "● → 25 € für Schäden und Umlaute: äöü",
            "api_key": "sk-example-secret-value-123456789",
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["answer"] == "● → 25 € für Schäden und Umlaute: äöü"
    assert payload["api_key"] == "[SECRET_REDACTED]"
    assert not list(tmp_path.glob("*.tmp"))


def test_non_secret_token_metrics_are_not_redacted(tmp_path) -> None:
    path = tmp_path / "diagnostic.json"
    atomic_write_json(path, {"policy_name_token_overlap": 3, "token": "secret"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["policy_name_token_overlap"] == 3
    assert payload["token"] == "[SECRET_REDACTED]"


def test_invalid_self_check_output_is_retained_for_diagnostics(monkeypatch) -> None:
    class FakePrompt:
        def __or__(self, llm):
            return llm

    class FakePromptTemplate:
        @classmethod
        def from_messages(cls, _messages):
            return FakePrompt()

    class FakeLLM:
        def invoke(self, _payload):
            return SimpleNamespace(content="RELEVANT\n● explanation")

    monkeypatch.setattr(
        rag_service,
        "_get_chat_prompt_template_cls",
        lambda: FakePromptTemplate,
    )
    diagnostics = RequestDiagnostics(self_check_enabled=True)
    token = set_current_diagnostics(diagnostics)
    try:
        with pytest.raises(GuardrailInvalidOutputError):
            rag_service.perform_self_check(
                FakeLLM(),
                None,
                "Is this covered?",
                [Document(page_content="Windscreen repair is covered.")],
            )
    finally:
        reset_current_diagnostics(token)

    evidence = diagnostics.evidence["selfCheck"]
    assert evidence["rawOutputSanitized"] == "RELEVANT\n● explanation"
    assert evidence["parserSuccess"] is False
    assert evidence["reasonCode"] == "GUARDRAIL_INVALID_OUTPUT"
    assert evidence["attemptCount"] == 1
    assert evidence["retryCount"] == 0
    assert diagnostics.stage_status["self_check"] == "failed"


def test_combined_response_is_mapped_to_requested_diagnostic_schema() -> None:
    diagnostic = _initial_diagnostic()
    _populate_from_response(
        diagnostic,
        http_status=206,
        payload={
            "status": "partial",
            "route": "combined",
            "sources": [],
            "diagnostics": {
                "requestId": "request-1",
                "route": "combined",
                "evidence": {
                    "crmSelection": {"selectedPolicyNumbers": ["POLICY-1"]},
                    "selfCheck": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "rawOutputSanitized": "unexpected ●",
                        "parserSuccess": False,
                        "reasonCode": "GUARDRAIL_INVALID_OUTPUT",
                        "attemptCount": 1,
                    },
                },
            },
        },
    )

    assert diagnostic["request_id"] == "request-1"
    assert diagnostic["selected_policy_number"] == "POLICY-1"
    assert diagnostic["self_check"]["raw_output_sanitized"] == "unexpected ●"
    assert diagnostic["self_check"]["actually_executed"] is True
    assert diagnostic["diagnostic_capture_success"] is True


def test_disabled_self_check_is_reported_as_skipped() -> None:
    diagnostic = _initial_diagnostic()
    _populate_from_response(
        diagnostic,
        http_status=200,
        payload={
            "status": "complete",
            "route": "combined",
            "answer": "Lara's current policy has Partial Coverage.",
            "crmResult": {"answer": "Coverage type: Partial Coverage"},
            "sources": [],
            "diagnostics": {
                "requestId": "request-2",
                "selfCheckEnabled": False,
                "selfCheckSkipped": True,
                "selfCheckSkipReason": "disabled_by_configuration",
                "stageStatus": {"self_check": "skipped"},
                "evidence": {},
            },
        },
    )

    assert diagnostic["self_check"]["enabled"] is False
    assert diagnostic["self_check"]["actually_executed"] is False
    assert diagnostic["self_check"]["skip_reason"] == "disabled_by_configuration"
    assert diagnostic["coverage_consistency"]["passed"] is True


def test_request_diagnostics_are_persisted_atomically(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    diagnostics = RequestDiagnostics(route="combined")
    diagnostics.record_evidence("unicode", "● → €")
    diagnostics.complete(status="partial", partial=True)

    main._log_request_diagnostics(diagnostics)

    path = tmp_path / "tmp" / "diagnostics" / f"request_{diagnostics.request_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["requestId"] == diagnostics.request_id
    assert payload["evidence"]["unicode"] == "● → €"
    assert payload["finishedAt"] is not None

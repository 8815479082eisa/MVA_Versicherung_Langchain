from __future__ import annotations

import asyncio
import sys
import time
import types
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src import main
from src.api import rag_service
from src.api.rag_service import AnswerResult
from src.config.models import load_model_settings
from src.core.llm_runtime import (
    GuardrailInvalidOutputError,
    LLMStageTimeoutError,
    LLMUnavailableError,
    RequestTimeoutError,
    invoke_llm_stage,
    parse_closed_label,
    run_bounded_operation,
)
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    begin_stage,
    current_diagnostics,
    finish_stage,
)
from src.guardrails.integrations import nemo_official


class FakeChatOllama:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _clear_fake_calls():
    FakeChatOllama.calls.clear()
    yield


def test_all_direct_llm_stages_receive_output_limits_and_supported_timeouts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "SETTINGS",
        replace(
            rag_service.SETTINGS,
            provider="ollama",
            retrieval=replace(
                rag_service.SETTINGS.retrieval,
                self_check_provider="ollama",
            ),
        ),
    )
    monkeypatch.setattr(rag_service, "_get_chat_ollama_cls", lambda: FakeChatOllama)
    builders = {
        "router": rag_service.initialize_router_llm,
        "self_check": rag_service.initialize_self_check_llm,
        "query_rewrite": lambda: rag_service._build_chat_model(
            rag_service.SETTINGS.roles.rewrite,
            0.0,
            "query_rewrite",
        ),
        "compressor": rag_service.initialize_compressor,
        "answer": rag_service.initialize_llm,
    }

    for stage, builder in builders.items():
        model = builder()
        assert model.kwargs["num_predict"] == (
            rag_service.SETTINGS.llm_runtime.output_limit(stage)
        )
        assert model.kwargs["client_kwargs"]["timeout"] == (
            rag_service.SETTINGS.llm_runtime.timeout(stage)
        )
        assert "timeout" not in model.kwargs


def test_openai_is_used_only_for_final_answer_generation(monkeypatch) -> None:
    class FakeOpenAIAnswerModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    online_settings = replace(
        rag_service.SETTINGS,
        provider="openai",
        openai_api_key_configured=True,
        roles=replace(rag_service.SETTINGS.roles, answer="gpt-4o-mini"),
    )
    monkeypatch.setattr(rag_service, "SETTINGS", online_settings)
    monkeypatch.setattr(
        rag_service,
        "_get_openai_answer_model_cls",
        lambda: FakeOpenAIAnswerModel,
    )
    monkeypatch.setattr(rag_service, "_get_chat_ollama_cls", lambda: FakeChatOllama)

    answer = rag_service.initialize_llm()
    router = rag_service.initialize_router_llm()

    assert answer.kwargs["model"] == "gpt-4o-mini"
    assert answer.kwargs["max_output_tokens"] == 384
    assert router.kwargs["model"] == online_settings.roles.router
    assert len(FakeChatOllama.calls) == 1


def test_openai_can_be_used_for_self_check_only_when_configured(monkeypatch) -> None:
    class FakeOpenAIAnswerModel:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeOpenAIAnswerModel.calls.append(kwargs)

    online_settings = replace(
        rag_service.SETTINGS,
        provider="openai",
        openai_api_key_configured=True,
        roles=replace(
            rag_service.SETTINGS.roles,
            answer="gpt-4o-mini",
            self_check="gpt-4o-mini",
        ),
        retrieval=replace(
            rag_service.SETTINGS.retrieval,
            self_check_provider="openai",
        ),
    )
    monkeypatch.setattr(rag_service, "SETTINGS", online_settings)
    monkeypatch.setattr(
        rag_service,
        "_get_openai_answer_model_cls",
        lambda: FakeOpenAIAnswerModel,
    )
    monkeypatch.setattr(rag_service, "_get_chat_ollama_cls", lambda: FakeChatOllama)

    self_check = rag_service.initialize_self_check_llm()
    router = rag_service.initialize_router_llm()

    assert self_check.kwargs["model"] == "gpt-4o-mini"
    assert self_check.kwargs["max_output_tokens"] == online_settings.llm_runtime.max_tokens_self_check
    assert router.kwargs["model"] == online_settings.roles.router


def test_guardrail_model_receives_guardrail_limit_and_timeout(monkeypatch) -> None:
    fake_module = types.ModuleType("langchain_ollama")
    fake_module.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)

    model = nemo_official._build_main_llm_from_settings()

    assert model.kwargs["model"] == rag_service.SETTINGS.roles.guardrail
    assert model.kwargs["num_predict"] == 8
    assert model.kwargs["client_kwargs"]["timeout"] == 30.0


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("router", 8),
        ("guardrail", 8),
        ("self_check", 8),
        ("query_rewrite", 64),
        ("answer", 384),
    ],
)
def test_default_stage_output_limits(stage: str, expected: int) -> None:
    assert load_model_settings().llm_runtime.output_limit(stage) == expected


def test_closed_vocabulary_normalizes_whitespace_and_case() -> None:
    assert (
        parse_closed_label(
            "  no_retrieve ",
            allowed={"RETRIEVE", "NO_RETRIEVE"},
            stage="router",
        )
        == "NO_RETRIEVE"
    )


def test_closed_vocabulary_rejects_verbose_output() -> None:
    with pytest.raises(GuardrailInvalidOutputError):
        parse_closed_label(
            "RETRIEVE because evidence is needed",
            allowed={"RETRIEVE", "NO_RETRIEVE"},
            stage="router",
        )


def test_stage_timeout_is_classified_and_not_retried() -> None:
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise TimeoutError("read timed out")

    with pytest.raises(LLMStageTimeoutError):
        invoke_llm_stage("answer", operation, max_retries=1)
    assert calls == 1


def test_retry_exhaustion_is_bounded_to_one_retry() -> None:
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise ConnectionError("connection refused")

    with pytest.raises(LLMUnavailableError):
        invoke_llm_stage("router", operation, max_retries=1)
    assert calls == 2


def test_bounded_stage_does_not_spawn_more_work_while_timed_out_worker_runs() -> None:
    with pytest.raises(LLMStageTimeoutError):
        run_bounded_operation(
            "test_slow_stage",
            lambda: time.sleep(0.08),
            timeout_seconds=0.01,
        )
    started = time.perf_counter()
    with pytest.raises(LLMStageTimeoutError):
        run_bounded_operation(
            "test_slow_stage",
            lambda: None,
            timeout_seconds=0.01,
        )
    assert time.perf_counter() - started < 0.05
    time.sleep(0.09)


def test_stage_latency_fields_are_json_serializable() -> None:
    diagnostics = RequestDiagnostics(route="retrieval-only")
    started = diagnostics.start_stage("answer")
    diagnostics.finish_stage("answer", started)
    diagnostics.complete(status="complete")
    payload = diagnostics.as_dict()
    assert payload["stageStatus"]["answer"] == "completed"
    assert isinstance(payload["stageTimings"]["answerGenerationMs"], float)
    assert payload["partial"] is False


def test_pure_retrieval_can_skip_reranker_and_answer_generation(monkeypatch) -> None:
    service = rag_service.RetrievalService(rag_service.SETTINGS)
    docs = [Document(page_content="coverage", metadata={"source": "a.pdf"})]
    service.components = {
        "hybrid_retriever": lambda _query, k: docs[:k],
        "reranker_model": object(),
    }
    monkeypatch.setattr(
        rag_service,
        "initialize_retrieval_service",
        lambda **_: (service, True),
    )
    answer = patch.object(rag_service, "initialize_llm")
    with answer as answer_mock:
        result = rag_service.retrieve_documents_for_tool(
            "coverage",
            1,
            use_reranker=False,
        )
    assert result["generation_used"] is False
    assert result["retrieval_mode"] == "hybrid_without_reranker"
    answer_mock.assert_not_called()


class _CRMRuntime:
    def __init__(self):
        self.invoke = AsyncMock(side_effect=self._invoke)

    async def _invoke(self, tool_name, arguments):
        del arguments
        if tool_name == "find_customer":
            return {
                "ok": True,
                "found": True,
                "customer": {
                    "id": "contact-1",
                    "display_name": "Lara Neumann",
                    "email_hint": "l***@example.test",
                },
            }
        if tool_name == "get_customer_policies":
            return {
                "ok": True,
                "policies": [
                    {
                        "id": "policy-1",
                        "policy_number": "TEST-KFZ-2026-1001",
                        "status": "Active",
                    }
                ],
            }
        if tool_name == "get_customer_claims":
            return {
                "ok": True,
                "claims": [
                    {
                        "id": "claim-1",
                        "claim_number": "TEST-CLM-2026-2001",
                        "status": "Under Review",
                    }
                ],
            }
        raise AssertionError(tool_name)


def test_crm_only_remains_independent_when_ollama_is_unavailable() -> None:
    runtime = _CRMRuntime()
    client = TestClient(main.app)
    with patch.object(
        main,
        "_crm_runtime_ready",
        True,
    ), patch.object(
        main,
        "get_mcp_runtime",
        return_value=runtime,
    ), patch.object(
        main.rag_service,
        "run_rag",
    ) as rag_mock, patch.object(
        main,
        "check_ollama_readiness",
        side_effect=AssertionError("health must not run in ask"),
    ):
        started = time.perf_counter()
        response = client.post(
            "/api/ask",
            json={"question": "Which active policies does Lara Neumann have?"},
        )
        latency = time.perf_counter() - started
    assert response.status_code == 200
    assert "TEST-KFZ-2026-1001" in response.json()["answer"]
    assert latency < 2.0
    rag_mock.assert_not_called()


def test_crm_only_contract_is_unchanged_when_self_check_is_disabled() -> None:
    runtime = _CRMRuntime()
    client = TestClient(main.app)
    disabled_settings = replace(
        main.rag_service.SETTINGS,
        retrieval=replace(
            main.rag_service.SETTINGS.retrieval,
            self_check_enabled=False,
        ),
    )
    with patch.object(
        main,
        "_crm_runtime_ready",
        True,
    ), patch.object(
        main,
        "get_mcp_runtime",
        return_value=runtime,
    ), patch.object(
        main.rag_service,
        "SETTINGS",
        disabled_settings,
    ), patch.object(
        main.rag_service,
        "run_rag",
    ) as rag_mock:
        response = client.post(
            "/api/ask",
            json={"question": "Which active policies does Lara Neumann have?"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["route"] == "crm-only"
    assert body["diagnostics"]["selfCheckEnabled"] is False
    assert body["diagnostics"]["selfCheckSkipped"] is False
    assert "self_check" not in body["diagnostics"]["stageStatus"]
    rag_mock.assert_not_called()


@pytest.mark.parametrize(
    ("runtime_error", "status_code", "error_code"),
    [
        (LLMUnavailableError(stage="answer"), 503, "LLM_UNAVAILABLE"),
        (LLMStageTimeoutError("guardrail"), 504, "LLM_STAGE_TIMEOUT"),
        (RequestTimeoutError(), 504, "REQUEST_TIMEOUT"),
    ],
)
def test_expected_runtime_errors_map_to_http(
    runtime_error,
    status_code,
    error_code,
) -> None:
    client = TestClient(main.app)
    with patch.object(
        main,
        "_run_rag_with_total_timeout",
        side_effect=runtime_error,
    ):
        response = client.post(
            "/api/ask",
            json={"question": "What does partial coverage generally cover?"},
        )
    assert response.status_code == status_code
    assert response.json()["detail"]["errorCode"] == error_code


def test_unexpected_error_maps_to_safe_http_500_without_secret(caplog) -> None:
    secret = "DO-NOT-LEAK-API-KEY"
    client = TestClient(main.app)
    with patch.object(
        main,
        "_run_rag_with_total_timeout",
        side_effect=Exception(secret),
    ):
        response = client.post(
            "/api/ask",
            json={"question": "What does partial coverage generally cover?"},
        )
    assert response.status_code == 500
    assert response.json()["detail"]["errorCode"] == "INTERNAL_ERROR"
    assert secret not in response.text
    assert secret not in caplog.text


def test_combined_returns_http_206_partial_when_rag_times_out() -> None:
    runtime = _CRMRuntime()
    client = TestClient(main.app)
    with patch.object(
        main,
        "_crm_runtime_ready",
        True,
    ), patch.object(
        main,
        "get_mcp_runtime",
        return_value=runtime,
    ), patch.object(
        main,
        "_run_rag_with_total_timeout",
        side_effect=LLMStageTimeoutError("answer"),
    ):
        response = client.post(
            "/api/ask",
            json={
                "question": (
                    "Lara Neumann reported glass damage. Is it covered "
                    "according to the conditions, and what is the claim status?"
                )
            },
        )
    body = response.json()
    assert response.status_code == 206
    assert body["status"] == "partial"
    assert body["route"] == "combined"
    assert body["knowledgeResult"] is None
    assert body["warning"]["errorCode"] == "LLM_STAGE_TIMEOUT"
    assert "No coverage or claim decision" in body["answer"]


def test_combined_returns_http_206_when_completeness_retry_still_fails() -> None:
    runtime = _CRMRuntime()
    client = TestClient(main.app)

    async def incomplete_result(*_args, **_kwargs):
        diagnostics = current_diagnostics()
        assert diagnostics is not None
        diagnostics.record_evidence(
            "completeness",
            {
                "completenessRequiredItems": [],
                "completenessPresentItems": [],
                "completenessMissingBeforeRetry": ["theft_police_reporting"],
                "completenessRetryPerformed": True,
                "completenessMissingAfterRetry": ["theft_police_reporting"],
                "completenessPass": False,
            },
        )
        return AnswerResult(
            answer="The theft coverage answer is incomplete.",
            sources=[],
            query="q",
            latency_ms=20,
        )

    with patch.object(
        main,
        "_crm_runtime_ready",
        True,
    ), patch.object(
        main,
        "get_mcp_runtime",
        return_value=runtime,
    ), patch.object(
        main,
        "_run_rag_with_total_timeout",
        side_effect=incomplete_result,
    ):
        response = client.post(
            "/api/ask",
            json={
                "question": (
                    "Lara Neumann reported theft of her car. Is it covered "
                    "according to the conditions, and what is her active policy?"
                )
            },
        )

    body = response.json()
    assert response.status_code == 206
    assert body["status"] == "partial"
    assert body["route"] == "combined"
    assert body["warning"]["errorCode"] == "ANSWER_INCOMPLETE"
    assert body["warning"]["missingItems"] == ["theft_police_reporting"]
    assert body["diagnostics"]["partial"] is True


def test_total_timeout_helper_keeps_crm_path_outside_rag_gate(monkeypatch) -> None:
    settings = main.rag_service.SETTINGS
    monkeypatch.setattr(
        main.rag_service,
        "SETTINGS",
        replace(
            settings,
            llm_runtime=replace(
                settings.llm_runtime,
                api_request_timeout_seconds=0.01,
            ),
        ),
    )
    monkeypatch.setattr(
        main.rag_service,
        "run_rag",
        lambda *_: time.sleep(0.08),
    )
    diagnostics = RequestDiagnostics(route="retrieval-only")
    with pytest.raises(RequestTimeoutError):
        asyncio.run(main._run_rag_with_total_timeout("question", diagnostics))
    time.sleep(0.09)


def test_rag_helper_preserves_classified_stage_timeout(monkeypatch) -> None:
    def fail_with_stage_timeout(*_):
        raise LLMStageTimeoutError("self_check")

    monkeypatch.setattr(main.rag_service, "run_rag", fail_with_stage_timeout)
    diagnostics = RequestDiagnostics(route="retrieval-only")
    with pytest.raises(LLMStageTimeoutError) as caught:
        asyncio.run(main._run_rag_with_total_timeout("question", diagnostics))
    assert caught.value.stage == "self_check"

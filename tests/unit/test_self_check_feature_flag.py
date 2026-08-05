from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.api import rag_service
from src.config import models as config_models
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)


class _TrackingSafetyChecker:
    is_active = True

    def __init__(self) -> None:
        self.query_calls = 0
        self.context_calls = 0
        self.answer_calls = 0

    def check_query_safety(self, query, chat_history):
        del query, chat_history
        self.query_calls += 1
        return rag_service._default_allow_safety_result("pre_query")

    def check_context_safety(self, documents):
        del documents
        self.context_calls += 1
        return rag_service._default_allow_safety_result("context")

    def check_answer_safety(self, query, documents, answer):
        del query, documents, answer
        self.answer_calls += 1
        return rag_service._default_allow_safety_result("post_generation")

    @staticmethod
    def apply_safety_action(result, answer):
        del result
        return answer


def _run_pipeline(*, self_check_enabled: bool):
    settings = replace(
        rag_service.SETTINGS,
        retrieval=replace(
            rag_service.SETTINGS.retrieval,
            self_check_enabled=self_check_enabled,
        ),
    )
    pipeline = rag_service.RAGPipeline(settings)
    safety_checker = _TrackingSafetyChecker()
    pipeline._safety_checker = safety_checker

    document = Document(
        page_content="Part comprehensive insurance covers glass breakage.",
        metadata={"source": "policy.pdf", "page": 1},
    )
    retrieval_service = rag_service.RetrievalService(settings)
    retrieval_service.retrieve_and_rerank = Mock(
        return_value=([document], [document])
    )
    components = {
        "retrieval_service": retrieval_service,
        "self_check_llm": object(),
        "query_rewrite_llm": object(),
        "compressor_llm": object(),
        "llm": object(),
    }

    diagnostics = RequestDiagnostics(route="retrieval-only")
    token = set_current_diagnostics(diagnostics)
    try:
        with patch.object(
            pipeline,
            "initialize",
            return_value=components,
        ), patch.object(
            rag_service,
            "pdfs_have_changed",
            return_value=False,
        ), patch.object(
            rag_service,
            "perform_self_check",
            return_value=("Does partial coverage cover glass?", [document]),
        ) as self_check_mock, patch.object(
            rag_service,
            "generate_answer",
            return_value="Glass breakage is covered [policy:1].",
        ) as answer_mock, patch.object(
            rag_service,
            "audit_log",
        ):
            result = pipeline.run("Does partial coverage cover glass?")
    finally:
        reset_current_diagnostics(token)

    return {
        "result": result,
        "diagnostics": diagnostics.as_dict(),
        "self_check_mock": self_check_mock,
        "answer_mock": answer_mock,
        "retrieval_mock": retrieval_service.retrieve_and_rerank,
        "safety_checker": safety_checker,
    }


def test_self_check_default_is_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config_models, "_DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.delenv("SELF_CHECK_ENABLED", raising=False)

    settings = config_models.load_model_settings()

    assert settings.retrieval.self_check_enabled is False


def test_self_check_environment_flag_can_disable_only_the_stage(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(config_models, "_DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("SELF_CHECK_ENABLED", "false")

    settings = config_models.load_model_settings()

    assert settings.retrieval.self_check_enabled is False
    assert settings.safety.enabled is rag_service.SETTINGS.safety.enabled
    assert settings.safety.mode == rag_service.SETTINGS.safety.mode


def test_enabled_flag_executes_self_check() -> None:
    execution = _run_pipeline(self_check_enabled=True)

    execution["self_check_mock"].assert_called_once()
    execution["retrieval_mock"].assert_called_once()
    execution["answer_mock"].assert_called_once()
    assert execution["diagnostics"]["selfCheckEnabled"] is True
    assert execution["diagnostics"]["selfCheckSkipped"] is False
    assert execution["diagnostics"]["selfCheckSkipReason"] is None


def test_disabled_flag_skips_self_check_but_keeps_retrieval_answer_and_safety() -> None:
    execution = _run_pipeline(self_check_enabled=False)

    execution["self_check_mock"].assert_not_called()
    execution["retrieval_mock"].assert_called_once()
    execution["answer_mock"].assert_called_once()
    assert "Glass breakage is covered" in execution["result"].answer
    assert execution["diagnostics"]["selfCheckEnabled"] is False
    assert execution["diagnostics"]["selfCheckSkipped"] is True
    assert (
        execution["diagnostics"]["selfCheckSkipReason"]
        == "disabled_by_configuration"
    )
    assert execution["diagnostics"]["stageStatus"]["self_check"] == "skipped"
    assert execution["safety_checker"].query_calls == 1
    assert execution["safety_checker"].context_calls == 1
    assert execution["safety_checker"].answer_calls == 1


def test_existing_pdf_chunk_and_crm_fact_share_one_generation_call() -> None:
    settings = replace(
        rag_service.SETTINGS,
        retrieval=replace(
            rag_service.SETTINGS.retrieval,
            self_check_enabled=False,
        ),
    )
    pipeline = rag_service.RAGPipeline(settings)
    pipeline._safety_checker = _TrackingSafetyChecker()
    pdf_chunk = Document(
        page_content="Part comprehensive insurance covers marten bites.",
        metadata={"source": "motor-vehicle-insurance-sti.pdf", "page": 13},
    )
    crm_fact = Document(
        page_content=(
            "CRM FACT\nPolicy TEST-KFZ-2026-1003: Motor Insurance, "
            "Partial Coverage, Active; deductible 300 EUR."
        ),
        metadata={
            "source": "espocrm:policy-1003",
            "source_type": "crm",
            "document_title": "EspoCRM Policy",
            "section": "TEST-KFZ-2026-1003",
        },
    )
    retrieval_service = rag_service.RetrievalService(settings)
    retrieval_service.retrieve_and_rerank = Mock(
        return_value=([pdf_chunk], [pdf_chunk])
    )
    components = {
        "retrieval_service": retrieval_service,
        "self_check_llm": None,
        "query_rewrite_llm": object(),
        "compressor_llm": object(),
        "llm": object(),
    }

    with patch.object(
        pipeline,
        "initialize",
        return_value=components,
    ), patch.object(
        rag_service,
        "pdfs_have_changed",
        return_value=False,
    ), patch.object(
        rag_service,
        "generate_answer",
        return_value=(
            "CRM facts and document evidence support the answer "
            "[motor-vehicle-insurance-sti:14]."
        ),
    ) as answer_mock, patch.object(
        rag_service,
        "audit_log",
    ), patch.object(
        rag_service,
        "load_and_split_documents",
    ) as chunking_mock:
        result = pipeline.run(
            "Does Lara's selected coverage include marten bites?",
            retrieval_query="Does partial coverage include marten bites?",
            additional_context_docs=[crm_fact],
            requested_source_filename="motor-vehicle-insurance-sti.pdf",
        )

    retrieval_service.retrieve_and_rerank.assert_called_once_with(
        "Does partial coverage include marten bites?",
        retrieval_top_k=settings.retrieval.top_k,
        rerank_top_k=min(2, settings.reranker.top_k),
        use_reranker=False,
        source_filename="motor-vehicle-insurance-sti.pdf",
    )
    answer_mock.assert_called_once()
    generated_context = answer_mock.call_args.args[2]
    assert generated_context == [pdf_chunk, crm_fact]
    chunking_mock.assert_not_called()
    assert {source.document_id for source in result.sources} == {
        "motor-vehicle-insurance-sti",
        "espocrm:policy-1003",
    }


def test_explicit_source_selects_existing_indexed_chunks_only() -> None:
    settings = rag_service.SETTINGS
    service = rag_service.RetrievalService(settings)
    target_page_one = Document(
        page_content="Marten bites are covered under part comprehensive.",
        metadata={"source": r"data\raw\pdfs\240_1184_e.pdf", "page": 1},
    )
    target_page_two = Document(
        page_content="Glass breakage is covered.",
        metadata={"source": r"data\raw\pdfs\240_1184_e.pdf", "page": 2},
    )
    other_edition = Document(
        page_content="Different product edition.",
        metadata={"source": r"data\raw\pdfs\240_1217_e.pdf", "page": 1},
    )
    hybrid_retriever = Mock(return_value=[other_edition])
    service.components = {
        "hybrid_retriever": hybrid_retriever,
        "indexed_documents": [
            target_page_one,
            target_page_two,
            other_edition,
        ],
    }

    retrieved, reranked = service.retrieve_and_rerank(
        "Are marten bites covered?",
        retrieval_top_k=10,
        rerank_top_k=5,
        use_reranker=False,
        source_filename="240_1184_e.pdf",
    )

    assert retrieved == [target_page_one, target_page_two]
    assert reranked == [target_page_one, target_page_two]
    hybrid_retriever.assert_not_called()


def test_audit_log_records_configured_self_check_skip(monkeypatch, tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(rag_service, "AUDIT_LOG_FILE", str(audit_path))
    diagnostics = RequestDiagnostics(route="combined", self_check_enabled=False)
    diagnostics.mark_self_check_skipped("disabled_by_configuration")
    token = set_current_diagnostics(diagnostics)
    try:
        rag_service.audit_log(
            query="Does partial coverage cover glass?",
            retrieved_documents=[],
            compressed_context=[],
            generated_answer="Grounded answer.",
        )
    finally:
        reset_current_diagnostics(token)

    payload = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["self_check_enabled"] is False
    assert payload["self_check_skipped"] is True
    assert (
        payload["self_check_skip_reason"]
        == "disabled_by_configuration"
    )

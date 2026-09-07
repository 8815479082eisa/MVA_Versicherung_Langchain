"""
FastAPI Backend fuer das MVA Insurance Agentic RAG System
"""

import os
import sys
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from pydantic import BaseModel
import uvicorn

# Pfad fuer Imports hinzufuegen
sys.path.insert(0, str(Path(__file__).parent))

from api import rag_service as rag_service
from src.core.crm_orchestration import (
    CRMOrchestrationError,
    execute_crm_query,
    select_crm_context,
)
from src.core.diagnostic_capture import atomic_write_json, sanitize_diagnostic_value
from src.core.insurance_tool_routing import (
    QueryMode,
    build_knowledge_query,
    enrich_knowledge_query,
    infer_document_source_filename,
    plan_insurance_query,
)
from src.core.safety_audit import (
    detect_query_hard_injection_signals,
    detect_system_secret_request_signals,
)
from src.core.llm_runtime import (
    RequestTimeoutError,
    RuntimeExecutionError,
)
from src.core.guardrail_readiness import check_guardrail_model_readiness
from src.core.ollama_readiness import check_ollama_readiness
from src.core.runtime_diagnostics import (
    RequestDiagnostics,
    begin_stage,
    finish_stage,
    reset_current_diagnostics,
    set_current_diagnostics,
)
from src.integrations.espocrm_client import EspoCRMConfig
from src.mcp_clients.insurance_tools import MCPRuntimeError, get_mcp_runtime

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_ONLY = _env_flag("FAQ_ONLY", default=False)  #wenn true, dann wird nur die FAQ verwendet, sonst wird der RAG verwendet
FAQ_FILE = os.getenv(
    "FAQ_FILE",
    str(_PROJECT_ROOT / "data" / "benchmarks" / "qa" / "insuranceqa" / "data_insuranceqa_1000.jsonl"),
) #FAQ_FILE ist der Pfad zur FAQ-Datei
FAQ_MIN_SIMILARITY = float(os.getenv("FAQ_MIN_SIMILARITY", "0.86")) #FAQ_MIN_SIMILARITY ist der minimale Ähnlichkeitswert für die FAQ
FAQ_FIRST = _env_flag("FAQ_FIRST", default=False) #FAQ_FIRST ist der erste Treffer für die FAQ, wenn kein Treffer gefunden wird, wird der RAG verwendet

_faq_entries: list[dict] = [] #FAQ_ENTRIES ist die Liste der FAQ-Einträge
_faq_by_question: dict[str, str] = {} #FAQ_BY_QUESTION ist das Dictionary der FAQ-Einträge
_startup_init_task: Optional[asyncio.Task] = None
_startup_init_error: Optional[str] = None
_crm_runtime_ready = False
_crm_runtime_error: Optional[dict] = None
_rag_execution_gate = asyncio.Semaphore(1)

#CORS_ALLOW_ORIGINS ist die Liste der erlaubten CORS-Origins, wenn CORS_ALLOW_ORIGINS="*" -> allow all.
def _get_cors_origins() -> list[str]:
    """
    Resolve allowed CORS origins.

    - If CORS_ALLOW_ORIGINS="*" -> allow all.
    - Else if set -> comma-separated list.
    - Else -> safe dev defaults (localhost frontends).
    """
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

#Strukturierung der Antwort
def _shape_answer(text: str, short_answer: bool, structured_answer: bool) -> str:
    answer = (text or "").strip()

    #Kurze Antwort
    if short_answer:
        if "\n\n" in answer:
            answer = answer.split("\n\n", 1)[0].strip()
        answer = answer[:500].rstrip()
#Strukturierte Antwort, wenn structured_answer=True
    if structured_answer:
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        if len(lines) <= 1 and answer:
            # Heuristic split on sentences if it's one-liner.
            parts = [
                p.strip()
                for p in answer.replace("?", "?.").replace("!", "!.").split(".")
                if p.strip()
            ]
            lines = parts[:6] if parts else [answer]
        answer = "\n".join([f"- {ln}" for ln in lines]) if lines else answer

    return answer


def _normalize_question(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_faq_data() -> None:
    """Load local FAQ pairs from JSONL file if available."""
    global _faq_entries, _faq_by_question
    _faq_entries = []
    _faq_by_question = {}

    faq_path = Path(FAQ_FILE)
    if not faq_path.exists():
        print(f"FAQ file not found: {faq_path}")
        return

    try:
        with faq_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                if not q or not a:
                    continue
                norm_q = _normalize_question(q)
                _faq_entries.append({"question": q, "answer": a, "norm_q": norm_q})
                _faq_by_question[norm_q] = a
        print(f"Loaded FAQ entries: {len(_faq_entries)} from {faq_path}")
    except Exception as e:
        print(f"Failed to load FAQ file {faq_path}: {e}")


def _find_faq_answer(question: str) -> tuple[Optional[str], Optional[float]]:
    """
    Find a local FAQ answer.
    Returns (answer, similarity). For exact match similarity=1.0.
    """
    if not _faq_entries:
        return None, None

    norm_q = _normalize_question(question)
    if norm_q in _faq_by_question:
        return _faq_by_question[norm_q], 1.0

    best_score = -1.0
    best_answer: Optional[str] = None
    for item in _faq_entries:
        score = SequenceMatcher(None, norm_q, item["norm_q"]).ratio()
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_answer is not None and best_score >= FAQ_MIN_SIMILARITY:
        return best_answer, best_score
    return None, best_score


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Application lifecycle.

    Important: pipeline initialization can be expensive (PDF parsing + embeddings),
    so we default to lazy initialization on first request. Enable eager init via:
    INIT_PIPELINE_ON_STARTUP=true
    """
    init_on_startup = _env_flag("INIT_PIPELINE_ON_STARTUP", default=False)
    global _startup_init_task, _startup_init_error
    global _crm_runtime_ready, _crm_runtime_error
    _startup_init_task = None
    _startup_init_error = None
    _crm_runtime_ready = False
    _crm_runtime_error = None

    _load_faq_data()
    runtime_config = rag_service.runtime_config()
    print(
        f"FAQ settings: FAQ_ONLY={FAQ_ONLY}, FAQ_FIRST={FAQ_FIRST}, "
        f"FAQ_MIN_SIMILARITY={FAQ_MIN_SIMILARITY}"
    )
    print(
        "Runtime config: "
        f"answer_provider={runtime_config['answer_provider']}, "
        f"configured_answer_model={runtime_config['configured_answer_model']} "
        f"(source={runtime_config['configured_answer_model_source']}), "
        f"preferred_answer_model={runtime_config['preferred_answer_model']}, "
        f"answer_model_matches_preference={runtime_config['answer_model_matches_preference']}, "
        f"query_rewrite_enabled={runtime_config['query_rewrite_enabled']}, "
        f"nemo_enforce_output={runtime_config['nemo_enforce_output']}, "
        f"safety_backend={runtime_config['safety_backend']}"
    )
    if runtime_config.get("dotenv_conflicts"):
        print(f"Config note: shell env overrides .env for {sorted(runtime_config['dotenv_conflicts'])}")

    async def _warm_pipeline_in_background(force_reindex: bool) -> None:
        global _startup_init_error
        try:
            await asyncio.to_thread(rag_service.initialize_pipeline, force_reindex=force_reindex)
            print("Pipeline warmup finished.")
        except Exception as e:
            _startup_init_error = f"{type(e).__name__}: {e}"
            print(f"Warning: error initializing pipeline in background: {_startup_init_error}")

    if FAQ_ONLY:
        print("Backend started in FAQ_ONLY mode (no RAG model calls).")
    elif init_on_startup:
        print("Starting RAG pipeline warmup in background (INIT_PIPELINE_ON_STARTUP=true)...")
        force_reindex = rag_service.pdfs_have_changed()
        _startup_init_task = asyncio.create_task(_warm_pipeline_in_background(force_reindex))
    else:
        print("Backend started (lazy pipeline init).")

    crm_config = EspoCRMConfig.from_environment()
    crm_config_error = crm_config.availability_error()
    if crm_config_error is not None:
        _crm_runtime_error = crm_config_error.as_dict()
        print(f"CRM MCP not started: {_crm_runtime_error['category']}")
    else:
        try:
            crm_runtime = get_mcp_runtime()
            await crm_runtime.start(server_names=("insurance_crm",))
            _crm_runtime_ready = True
            print(
                "CRM MCP ready with tools: "
                + ", ".join(crm_runtime.server_tools.get("insurance_crm", ()))
            )
        except Exception as exc:
            _crm_runtime_error = {
                "category": "connection",
                "message": "The CRM MCP server could not be started.",
            }
            print(f"CRM MCP startup failed: {type(exc).__name__}")

    try:
        yield
    finally:
        if _crm_runtime_ready:
            await get_mcp_runtime().close()
        _crm_runtime_ready = False
        if _startup_init_task is not None and not _startup_init_task.done():
            _startup_init_task.cancel()
        print("Backend shutting down...")

# FastAPI-App erstellen
app = FastAPI(
    title="MVA Insurance Agentic RAG API",
    description="Backend fuer das MVA Insurance RAG System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS-Middleware - Zugriff vom Frontend erlauben
origins = _get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class AskQuestionRequest(BaseModel):
    """Frageanfrage"""
    question: str
    shortAnswer: bool = False
    structuredAnswer: bool = False


class Source(BaseModel):
    """Dokumentquelle"""
    documentId: str
    documentTitle: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


class AnswerResponse(BaseModel):
    """Antwort"""
    answer: str
    sources: list[Source]
    latencyMs: Optional[float] = None
    status: str = "complete"
    route: Optional[str] = None
    warning: Optional[dict[str, Any]] = None
    crmResult: Optional[dict[str, Any]] = None
    knowledgeResult: Optional[dict[str, Any]] = None
    diagnostics: Optional[dict[str, Any]] = None


def _answer_completeness_warning(
    diagnostics: RequestDiagnostics,
) -> Optional[dict[str, Any]]:
    evidence = diagnostics.evidence.get("completeness")
    if not isinstance(evidence, dict) or evidence.get("completenessPass") is not False:
        return None
    missing = evidence.get("completenessMissingAfterRetry") or evidence.get(
        "completenessMissingBeforeRetry"
    ) or []
    return {
        "status": "partial",
        "errorCode": "ANSWER_INCOMPLETE",
        "category": "completeness",
        "message": (
            "The generated answer remained incomplete after the single bounded "
            "completeness regeneration."
        ),
        "route": diagnostics.route,
        "missingItems": list(missing),
    }


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _: Request,
    __: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "status": "error",
                "errorCode": "MALFORMED_REQUEST",
                "message": "The request body is malformed.",
            }
        },
    )


# Routes
@app.get("/")
async def root():
    return {"status": "online", "service": "MVA Insurance Agentic RAG API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Serverstatus pruefen"""
    pipeline_ready = rag_service.is_pipeline_ready()
    pipeline_initializing = _startup_init_task is not None and not _startup_init_task.done()
    runtime_config = rag_service.runtime_config()
    llm_health = await check_ollama_readiness(
        rag_service.SETTINGS,
        timeout_seconds=2.0,
    )
    guardrail_health = await check_guardrail_model_readiness(timeout_seconds=60.0)
    # Output rails use FastEmbed's MiniLM model for embeddings-only intent matching.
    # Keep the optional Ollama status visible, but report the dependency that the
    # active guardrail path actually needs as guardrailModelReady.
    llm_health["ollamaGuardrailModelReady"] = llm_health["guardrailModelReady"]
    llm_health.update(guardrail_health)
    llm_health["llmReady"] = bool(
        llm_health["answerModelReady"] and llm_health["guardrailModelReady"]
    )
    if llm_health["llmReady"]:
        llm_health["llmError"] = None
    return {
        "status": "ok",
        "message": "Backend laeuft",
        "pipelineReady": pipeline_ready,
        "pipelineInitializing": pipeline_initializing,
        "pipelineInitError": _startup_init_error,
        "crmEnabled": EspoCRMConfig.from_environment().enabled,
        "crmReady": _crm_runtime_ready,
        "crmError": _crm_runtime_error,
        **llm_health,
        "embeddingReady": rag_service.is_embedding_ready(),
        "retrievalReady": rag_service.is_retrieval_ready(),
        "insuranceqaExactMatchShortcut": rag_service.insuranceqa_exact_match_shortcut_enabled(),
        "safetyEnabled": rag_service.safety_enabled(),
        "safetyMode": rag_service.safety_mode(),
        "answer_provider": runtime_config["answer_provider"],
        "openai_api_key_configured": runtime_config["openai_api_key_configured"],
        "configured_answer_model": runtime_config["configured_answer_model"],
        "preferred_answer_model": runtime_config["preferred_answer_model"],
        "answer_model_matches_preference": runtime_config["answer_model_matches_preference"],
        "configured_answer_model_source": runtime_config["configured_answer_model_source"],
        "selfCheckEnabled": runtime_config["self_check_enabled"],
        "answerCompletenessEnabled": runtime_config[
            "answer_completeness_enabled"
        ],
        "query_rewrite_enabled": runtime_config["query_rewrite_enabled"],
        "nemo_enforce_output": runtime_config["nemo_enforce_output"],
        "safety_backend": runtime_config["safety_backend"],
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(request: AskQuestionRequest):
    """Answer one bounded request without coupling CRM-only work to Ollama."""

    diagnostics = RequestDiagnostics(
        self_check_enabled=rag_service.SETTINGS.retrieval.self_check_enabled,
    )
    diagnostics_token = set_current_diagnostics(diagnostics)
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "errorCode": "MALFORMED_REQUEST",
                    "message": "Die Frage darf nicht leer sein.",
                },
            )

        question = request.question.strip()
        secret_signals = detect_system_secret_request_signals(question)
        injection_signals = detect_query_hard_injection_signals(question)
        if secret_signals or injection_signals:
            reason_code = (
                "SYSTEM_SECRET_REQUEST"
                if secret_signals
                else "PROMPT_INJECTION_DETECTED"
            )
            diagnostics.route = "denied"
            diagnostics.record_evidence(
                "safetyDecision",
                {"decision": "BLOCK", "reasonCode": reason_code},
            )
            diagnostics.complete(status="complete")
            blocked_answer = rag_service.SETTINGS.safety.security_fallback_text
            rag_service.audit_log(
                query=question,
                retrieved_documents=[],
                compressed_context=[],
                generated_answer=blocked_answer,
                chat_history=[],
                route="denied",
                safety_decision="block",
                safety_reason_code=reason_code,
                safety_risks=secret_signals or injection_signals,
            )
            return AnswerResponse(
                answer=blocked_answer,
                sources=[],
                status="complete",
                route="denied",
                diagnostics=diagnostics.as_dict(),
            )
        route_started = begin_stage("route_planning")
        query_plan = plan_insurance_query(question)
        diagnostics.route = query_plan.mode.value.replace("_", "-")
        finish_stage("route_planning", route_started)
        diagnostics.record_evidence(
            "queryDecomposition",
            {
                "route": diagnostics.route,
                "reason": query_plan.reason,
                "crmSubquery": {
                    "customerName": query_plan.customer_name,
                    "customerEmail": query_plan.customer_email,
                    "policyNumber": query_plan.policy_number,
                    "claimNumber": query_plan.claim_number,
                    "needsPolicies": query_plan.needs_policies,
                    "needsClaims": query_plan.needs_claims,
                },
            },
        )

        if query_plan.mode is QueryMode.DENIED:
            raise HTTPException(
                status_code=403,
                detail={
                    "status": "error",
                    "errorCode": "FORBIDDEN_OPERATION",
                    "category": "unauthorized_scope",
                    "message": query_plan.reason,
                    "route": diagnostics.route,
                },
            )

        if query_plan.uses_crm:
            if not _crm_runtime_ready:
                error = dict(
                    _crm_runtime_error
                    or {
                        "category": "connection",
                        "message": "CRM integration is not available.",
                    }
                )
                error.update(
                    {
                        "status": "error",
                        "errorCode": "CRM_UNAVAILABLE",
                        "route": diagnostics.route,
                    }
                )
                raise HTTPException(
                    status_code=503,
                    detail=error,
                )

            request_started = time.perf_counter()
            try:
                if query_plan.mode is QueryMode.CRM_ONLY:
                    crm_result = await _execute_crm_with_diagnostics(
                        query_plan
                    )
                    crm_selection = select_crm_context(
                        crm_result,
                        question,
                    )
                    selected_crm_sources = crm_selection.sources
                    selected_crm_answer = _crm_context_answer(
                        selected_crm_sources
                    )
                    if crm_result.reason_code == "AMBIGUOUS_CUSTOMER":
                        selected_crm_answer = crm_result.answer
                    diagnostics.record_evidence(
                        "crmSelection",
                        {
                            "selectedPolicyNumbers": list(
                                crm_selection.selected_policy_numbers
                            ),
                            "retrievalHints": list(
                                crm_selection.retrieval_hints
                            ),
                            "candidates": list(crm_selection.evidence),
                        },
                    )
                    diagnostics.complete(status="complete")
                    crm_context_documents = _crm_context_documents(selected_crm_sources)
                    groundedness = _crm_groundedness_score(
                        question,
                        selected_crm_answer,
                        crm_context_documents,
                    )
                    reason_code = crm_result.reason_code
                    if (
                        crm_context_documents
                        and groundedness < rag_service.SETTINGS.safety.min_groundedness
                    ):
                        selected_crm_answer = rag_service.SETTINGS.safety.grounding_fallback_text
                        reason_code = "UNGROUNDED_PERSONAL_DATA"
                    diagnostics.record_evidence(
                        "safetyDecision",
                        {
                            "decision": "ALLOW" if reason_code != "UNGROUNDED_PERSONAL_DATA" else "BLOCK",
                            "reasonCode": reason_code,
                            "groundedness": groundedness,
                            "threshold": rag_service.SETTINGS.safety.min_groundedness,
                            "algorithmVersion": "fact_aware_claim_support_v5",
                        },
                    )
                    rag_service.audit_log(
                        query=question,
                        retrieved_documents=crm_context_documents,
                        compressed_context=crm_context_documents,
                        generated_answer=selected_crm_answer,
                        route="crm-only",
                        safety_decision="allow" if reason_code != "UNGROUNDED_PERSONAL_DATA" else "block",
                        safety_reason_code=reason_code,
                        groundedness=groundedness,
                        crm_tools=_crm_tool_names(crm_result.tool_results),
                        **_crm_audit_ids(selected_crm_sources),
                    )
                    response = AnswerResponse(
                        answer=_shape_answer(
                            "CRM facts:\n" + selected_crm_answer,
                            request.shortAnswer,
                            request.structuredAnswer,
                        ),
                        sources=[
                            Source(**source) for source in selected_crm_sources
                        ],
                        latencyMs=(time.perf_counter() - request_started) * 1000,
                        status="complete",
                        route="crm-only",
                        diagnostics=diagnostics.as_dict(),
                    )
                    _log_request_diagnostics(diagnostics)
                    return response

                crm_result = await _execute_crm_with_diagnostics(query_plan)
                crm_selection = select_crm_context(
                    crm_result,
                    question,
                )
                selected_crm_sources = crm_selection.sources
                crm_context_documents = _crm_context_documents(
                    selected_crm_sources
                )
                selected_crm_answer = _crm_context_answer(
                    selected_crm_sources
                )
                diagnostics.record_evidence(
                    "crmContextSummary",
                    [
                        sanitize_diagnostic_value(document.page_content)
                        for document in crm_context_documents
                    ],
                )
                knowledge_query = build_knowledge_query(question, query_plan)
                effective_retrieval_query = enrich_knowledge_query(
                    knowledge_query,
                    product_hints=crm_selection.retrieval_hints,
                )
                requested_pdf = infer_document_source_filename(question)
                if requested_pdf is None and crm_selection.retrieval_hints:
                    requested_pdf = infer_document_source_filename(
                        question + " " + " ".join(crm_selection.retrieval_hints)
                    )
                diagnostics.record_evidence(
                    "crmSelection",
                    {
                        "selectedPolicyNumbers": list(
                            crm_selection.selected_policy_numbers
                        ),
                        "retrievalHints": list(crm_selection.retrieval_hints),
                        "candidates": list(crm_selection.evidence),
                    },
                )
                diagnostics.record_evidence(
                    "knowledgeRetrieval",
                    {
                        "knowledgeSubquery": knowledge_query,
                        "effectiveRetrievalQuery": effective_retrieval_query,
                        "requestedPdfFilename": requested_pdf,
                    },
                )

                try:
                    document_result = await _run_rag_with_total_timeout(
                        question,
                        diagnostics,
                        retrieval_query=effective_retrieval_query,
                        additional_context_docs=crm_context_documents,
                        requested_source_filename=requested_pdf,
                    )
                except RuntimeExecutionError as document_error:
                    diagnostics.complete(status="partial", partial=True)
                    crm_sources = [
                        Source(**source) for source in selected_crm_sources
                    ]
                    warning = document_error.as_dict(route="combined")
                    fallback_answer = _shape_answer(
                        "CRM facts:\n"
                        + selected_crm_answer
                        + "\n\nDocument evidence:\nUnavailable. "
                        "No coverage or claim decision was inferred from CRM data.",
                        request.shortAnswer,
                        request.structuredAnswer,
                    )
                    diagnostics.record_evidence(
                        "finalResponse",
                        {
                            "status": "partial",
                            "route": "combined",
                            "answerSanitized": sanitize_diagnostic_value(fallback_answer),
                            "sourceIds": [source.documentId for source in crm_sources],
                            "warning": warning,
                        },
                    )
                    partial = AnswerResponse(
                        answer=fallback_answer,
                        sources=crm_sources,
                        latencyMs=(time.perf_counter() - request_started) * 1000,
                        status="partial",
                        route="combined",
                        warning=warning,
                        crmResult={
                            "answer": selected_crm_answer,
                            "sources": [source.model_dump() for source in crm_sources],
                            "selection": {
                                "selectedPolicyNumbers": list(
                                    crm_selection.selected_policy_numbers
                                ),
                                "candidates": list(crm_selection.evidence),
                            },
                        },
                        knowledgeResult=None,
                        diagnostics=diagnostics.as_dict(),
                    )
                    _log_request_diagnostics(diagnostics)
                    return JSONResponse(
                        status_code=206,
                        content=partial.model_dump(),
                    )

                sources = _api_sources(document_result.sources or [])
                final_answer = _shape_answer(
                    document_result.answer,
                    request.shortAnswer,
                    request.structuredAnswer,
                )
                completeness_warning = _answer_completeness_warning(diagnostics)
                if completeness_warning is not None:
                    diagnostics.complete(status="partial", partial=True)
                    diagnostics.record_evidence(
                        "finalResponse",
                        {
                            "status": "partial",
                            "route": "combined",
                            "answerSanitized": sanitize_diagnostic_value(final_answer),
                            "sourceIds": [source.documentId for source in sources],
                            "warning": completeness_warning,
                        },
                    )
                    partial = AnswerResponse(
                        answer=final_answer,
                        sources=sources,
                        latencyMs=(time.perf_counter() - request_started) * 1000,
                        status="partial",
                        route="combined",
                        warning=completeness_warning,
                        crmResult={
                            "answer": selected_crm_answer,
                            "sources": [
                                source.model_dump()
                                for source in sources
                                if source.documentId.startswith("espocrm:")
                            ],
                            "selection": {
                                "selectedPolicyNumbers": list(
                                    crm_selection.selected_policy_numbers
                                ),
                                "candidates": list(crm_selection.evidence),
                            },
                        },
                        knowledgeResult={
                            "answer": document_result.answer,
                            "sources": [
                                source.model_dump()
                                for source in sources
                                if not source.documentId.startswith("espocrm:")
                            ],
                        },
                        diagnostics=diagnostics.as_dict(),
                    )
                    _log_request_diagnostics(diagnostics)
                    return JSONResponse(status_code=206, content=partial.model_dump())

                diagnostics.complete(status="complete")
                diagnostics.record_evidence(
                    "finalResponse",
                    {
                        "status": "complete",
                        "route": "combined",
                        "answerSanitized": sanitize_diagnostic_value(final_answer),
                        "sourceIds": [source.documentId for source in sources],
                    },
                )
                response = AnswerResponse(
                    answer=final_answer,
                    sources=sources,
                    latencyMs=(time.perf_counter() - request_started) * 1000,
                    status="complete",
                    route="combined",
                    crmResult={
                        "answer": selected_crm_answer,
                        "sources": [
                            source.model_dump()
                            for source in sources
                            if source.documentId.startswith("espocrm:")
                        ],
                        "selection": {
                            "selectedPolicyNumbers": list(
                                crm_selection.selected_policy_numbers
                            ),
                            "candidates": list(crm_selection.evidence),
                        },
                    },
                    knowledgeResult={
                        "answer": document_result.answer,
                        "sources": [
                            source.model_dump()
                            for source in sources
                            if not source.documentId.startswith("espocrm:")
                        ],
                    },
                    diagnostics=diagnostics.as_dict(),
                )
                _log_request_diagnostics(diagnostics)
                return response
            except CRMOrchestrationError as exc:
                raise HTTPException(
                    status_code=502 if exc.available else 503,
                    detail={
                        **exc.error,
                        "status": "error",
                        "errorCode": "CRM_REQUEST_FAILED",
                        "route": diagnostics.route,
                    },
                ) from exc
            except MCPRuntimeError:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "error",
                        "errorCode": "CRM_UNAVAILABLE",
                        "category": "connection",
                        "message": "The CRM MCP session is unavailable.",
                        "route": diagnostics.route,
                    },
                )

        # Local FAQ path (no token usage)
        if FAQ_ONLY or FAQ_FIRST:
            faq_answer, faq_score = _find_faq_answer(question)
            if faq_answer is not None:
                sources = [
                    Source(
                        documentId="insuranceqa_v2_local",
                        documentTitle="insuranceQA-v2 (local JSONL)",
                        page=None,
                        section=f"similarity={faq_score:.3f}" if faq_score is not None else None,
                        snippet=faq_answer[:300],
                    )
                ]
                diagnostics.complete(status="complete")
                response = AnswerResponse(
                    answer=_shape_answer(faq_answer, request.shortAnswer, request.structuredAnswer),
                    sources=sources,
                    latencyMs=0,
                    status="complete",
                    route="retrieval-only",
                    diagnostics=diagnostics.as_dict(),
                )
                _log_request_diagnostics(diagnostics)
                return response

            if FAQ_ONLY:
                raise HTTPException(
                    status_code=404,
                    detail="No local FAQ match found. Lower FAQ_MIN_SIMILARITY or update FAQ_FILE.",
                )

        # RAG-Service path (uses local model provider)
        retrieval_query = enrich_knowledge_query(question)
        requested_pdf = infer_document_source_filename(question)
        result = await _run_rag_with_total_timeout(
            question,
            diagnostics,
            retrieval_query=retrieval_query,
            requested_source_filename=requested_pdf,
        )
        
        # Ergebnis in AnswerResponse umwandeln
        sources = _api_sources(result.sources or [])
        shaped_answer = _shape_answer(
            result.answer,
            request.shortAnswer,
            request.structuredAnswer,
        )
        completeness_warning = _answer_completeness_warning(diagnostics)
        if completeness_warning is not None:
            diagnostics.complete(status="partial", partial=True)
            response = AnswerResponse(
                answer=shaped_answer,
                sources=sources,
                latencyMs=result.latency_ms,
                status="partial",
                route="retrieval-only",
                warning=completeness_warning,
                knowledgeResult={
                    "answer": result.answer,
                    "sources": [source.model_dump() for source in sources],
                },
                diagnostics=diagnostics.as_dict(),
            )
            _log_request_diagnostics(diagnostics)
            return JSONResponse(status_code=206, content=response.model_dump())

        diagnostics.complete(status="complete")
        response = AnswerResponse(
            answer=shaped_answer,
            sources=sources,
            latencyMs=result.latency_ms,
            status="complete",
            route="retrieval-only",
            diagnostics=diagnostics.as_dict(),
        )
        _log_request_diagnostics(diagnostics)
        return response

    except RuntimeExecutionError as exc:
        diagnostics.complete(status="error")
        _log_request_diagnostics(diagnostics)
        status_code = (
            504
            if exc.error_code in {"LLM_STAGE_TIMEOUT", "REQUEST_TIMEOUT"}
            else 500
            if exc.error_code == "INTERNAL_ERROR"
            else 503
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                **exc.as_dict(route=diagnostics.route),
                "diagnostics": diagnostics.as_dict(),
            },
        ) from exc
    except (ValueError, RuntimeError) as exc:
        diagnostics.complete(status="error")
        _log_request_diagnostics(diagnostics)
        logger.warning("Expected request/pipeline value error: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "errorCode": "RETRIEVAL_FAILED",
                "message": "Document retrieval is not available.",
                "route": diagnostics.route,
                "stage": "retrieval",
                "diagnostics": diagnostics.as_dict(),
            },
        ) from exc
    except HTTPException:
        diagnostics.complete(status="error")
        _log_request_diagnostics(diagnostics)
        raise
    except Exception as exc:
        diagnostics.complete(status="error")
        _log_request_diagnostics(diagnostics)
        logger.error(
            "Unhandled error while processing /api/ask type=%s requestId=%s",
            type(exc).__name__,
            diagnostics.request_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "errorCode": "INTERNAL_ERROR",
                "message": "An unexpected internal error occurred.",
                "route": diagnostics.route,
                "diagnostics": diagnostics.as_dict(),
            },
        ) from exc
    finally:
        reset_current_diagnostics(diagnostics_token)


async def _execute_crm_with_diagnostics(query_plan):
    started_at = begin_stage("crm")
    try:
        result = await execute_crm_query(
            query_plan,
            get_mcp_runtime().invoke,
        )
        finish_stage("crm", started_at)
        return result
    except Exception:
        finish_stage("crm", started_at, status="failed")
        raise


async def _run_rag_with_total_timeout(
    question: str,
    diagnostics: RequestDiagnostics,
    *,
    retrieval_query: Optional[str] = None,
    additional_context_docs: Optional[list[Document]] = None,
    requested_source_filename: Optional[str] = None,
):
    timeout_seconds = rag_service.SETTINGS.llm_runtime.api_request_timeout_seconds
    elapsed = time.perf_counter() - diagnostics.started_at
    remaining = timeout_seconds - elapsed
    if remaining <= 0:
        diagnostics.mark_stage("request", "timeout")
        raise RequestTimeoutError()

    try:
        await asyncio.wait_for(
            _rag_execution_gate.acquire(),
            timeout=remaining,
        )
    except asyncio.TimeoutError as exc:
        diagnostics.mark_stage("request", "timeout")
        raise RequestTimeoutError() from exc

    elapsed = time.perf_counter() - diagnostics.started_at
    remaining = timeout_seconds - elapsed
    if remaining <= 0:
        _rag_execution_gate.release()
        diagnostics.mark_stage("request", "timeout")
        raise RequestTimeoutError()

    rag_kwargs: dict[str, Any] = {}
    if retrieval_query is not None:
        rag_kwargs["retrieval_query"] = retrieval_query
    if additional_context_docs is not None:
        rag_kwargs["additional_context_docs"] = additional_context_docs
    if requested_source_filename is not None:
        rag_kwargs["requested_source_filename"] = requested_source_filename

    task = asyncio.create_task(
        asyncio.to_thread(
            rag_service.run_rag,
            question,
            [],
            **rag_kwargs,
        )
    )
    task.add_done_callback(lambda _: _rag_execution_gate.release())
    try:
        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=remaining,
        )
    except RuntimeExecutionError:
        # In Python 3.11 built-in TimeoutError and asyncio.TimeoutError are
        # aliases. Preserve an already classified stage timeout from the RAG
        # worker instead of relabeling it as an overall request timeout.
        raise
    except asyncio.TimeoutError as exc:
        # The synchronous Ollama client has no cross-thread cancellation API.
        # The gate remains held until the bounded worker exits, preventing
        # additional abandoned RAG work from accumulating.
        diagnostics.mark_stage("request", "timeout")
        raise RequestTimeoutError() from exc


def _log_request_diagnostics(diagnostics: RequestDiagnostics) -> None:
    payload = sanitize_diagnostic_value(
        {
            "event": "api_request_diagnostics",
            **diagnostics.as_dict(),
        }
    )
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    diagnostic_path = (
        _PROJECT_ROOT
        / "tmp"
        / "diagnostics"
        / f"request_{diagnostics.request_id}.json"
    )
    try:
        atomic_write_json(diagnostic_path, payload)
    except OSError as exc:
        logger.error(
            "Unable to persist request diagnostics type=%s requestId=%s",
            type(exc).__name__,
            diagnostics.request_id,
        )


def _api_sources(rag_sources: list) -> list[Source]:
    return [
        Source(
            documentId=src.document_id,
            documentTitle=src.document_title,
            page=src.page,
            section=src.section,
            snippet=src.snippet,
        )
        for src in rag_sources
    ]


def _crm_context_documents(
    sources: tuple[dict[str, Any], ...],
) -> list[Document]:
    """Represent selected CRM records as context, without creating PDF chunks."""

    documents: list[Document] = []
    for source in sources:
        snippet = str(source.get("snippet") or "").strip()
        if not snippet:
            continue
        documents.append(
            Document(
                page_content=f"CRM FACT\n{snippet}",
                metadata={
                    "source": str(source.get("documentId") or "espocrm:unknown"),
                    "source_type": "crm",
                    "document_title": str(
                        source.get("documentTitle") or "EspoCRM record"
                    ),
                    "section": source.get("section"),
                    "entity_type": source.get("entityType"),
                    "entity_id": source.get("entityId"),
                    "contact_id": source.get("contactId"),
                    "policy_id": source.get("policyId"),
                    "claim_id": source.get("claimId"),
                    "entity_binding_valid": source.get("entityBindingValid", False),
                    "authorized_source": True,
                },
            )
        )
    return documents


def _crm_context_answer(
    sources: tuple[dict[str, Any], ...],
) -> str:
    facts = [
        str(source.get("snippet") or "").strip()
        for source in sources
        if str(source.get("snippet") or "").strip()
    ]
    return "\n".join(facts) or "CRM returned no matching insurance facts."


def _crm_groundedness_score(
    query: str,
    answer: str,
    docs: list[Document],
) -> float:
    if not docs:
        return 1.0 if answer.startswith("Multiple contacts match") else 0.0
    from scripts.experimental_groundedness_v5 import (
        calculate_groundedness_score_v5_experimental,
    )

    return calculate_groundedness_score_v5_experimental(answer, docs, query).score


def _crm_tool_names(tool_results: tuple[dict[str, Any], ...]) -> list[str]:
    names: list[str] = []
    for result in tool_results:
        if "customer" in result:
            names.append("find_customer")
        if "policy" in result:
            names.append("get_policy")
        if "policies" in result:
            names.append("get_customer_policies")
        if "claim" in result:
            names.append("get_claim_status")
        if "claims" in result:
            names.append("get_customer_claims")
    return list(dict.fromkeys(names))


def _crm_audit_ids(sources: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        "contact_ids": sorted({str(source["contactId"]) for source in sources if source.get("contactId")}),
        "policy_ids": sorted({str(source["policyId"]) for source in sources if source.get("policyId")}),
        "claim_ids": sorted({str(source["claimId"]) for source in sources if source.get("claimId")}),
        "source_ids": [str(source.get("documentId")) for source in sources if source.get("documentId")],
    }


if __name__ == "__main__":
    # Server starten
    port = int(os.getenv("BACKEND_PORT", 8000))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    reload_default = False if os.name == "nt" else True
    reload = _env_flag("UVICORN_RELOAD", default=reload_default)
    
    print(f"Backend running at http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload  # In development fuer Auto-Reload
    )

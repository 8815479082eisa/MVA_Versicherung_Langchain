"""
FastAPI Backend fuer das MVA Insurance Agentic RAG System
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Pfad fuer Imports hinzufuegen
sys.path.insert(0, str(Path(__file__).parent))

from api import rag_service as rag_service
from api.vapi_service import get_vapi_service


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _shape_answer(text: str, short_answer: bool, structured_answer: bool) -> str:
    answer = (text or "").strip()

    # Best-effort output shaping (no extra model call)
    if short_answer:
        if "\n\n" in answer:
            answer = answer.split("\n\n", 1)[0].strip()
        answer = answer[:500].rstrip()

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Application lifecycle.

    Important: pipeline initialization can be expensive (PDF parsing + embeddings),
    so we default to lazy initialization on first request. Enable eager init via:
    INIT_PIPELINE_ON_STARTUP=true
    """
    init_on_startup = _env_flag("INIT_PIPELINE_ON_STARTUP", default=False)

    if init_on_startup:
        print("Initializing RAG pipeline on startup (INIT_PIPELINE_ON_STARTUP=true)...")
        try:
            force_reindex = rag_service.pdfs_have_changed()
            await asyncio.to_thread(rag_service.initialize_pipeline, force_reindex=force_reindex)
            print("Pipeline initialized.")
        except Exception as e:
            print(f"Warning: error initializing pipeline: {e}")
            print("Pipeline will initialize lazily on the first request.")
    else:
        print("Backend started (lazy pipeline init).")

    yield

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


# Vapi Call Models
class InitiateCallRequest(BaseModel):
    """Anfrage zum Starten eines AI-Anrufs"""
    phoneNumber: str
    customerName: Optional[str] = None


class CallStatusResponse(BaseModel):
    """Status eines AI-Anrufs"""
    callId: str
    status: str
    phoneNumber: str
    createdAt: str
    duration: Optional[float] = None
    cost: Optional[float] = None
    endedReason: Optional[str] = None


class EndCallRequest(BaseModel):
    """Anfrage zum Beenden eines Anrufs"""
    callId: str


# Routes
@app.get("/")
async def root():
    return {"status": "online", "service": "MVA Insurance Agentic RAG API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Serverstatus pruefen"""
    return {"status": "ok", "message": "Backend laeuft"}


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(request: AskQuestionRequest):
    """
    Frage anfragen und Antwort erhalten
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Die Frage darf nicht leer sein")
        
        # RAG-Service aufrufen (lazy init happens inside run_rag)
        result = rag_service.run_rag(request.question.strip(), chat_history=[])
        
        # Ergebnis in AnswerResponse umwandeln
        sources = [
            Source(
                documentId=src.document_id,
                documentTitle=src.document_title,
                page=src.page,
                section=src.section,
                snippet=src.snippet
            )
            for src in (result.sources or [])
        ]
        
        return AnswerResponse(
            answer=_shape_answer(result.answer, request.shortAnswer, request.structuredAnswer),
            sources=sources,
            latencyMs=result.latency_ms
        )
        
    except ValueError as e:
        # Typically: pipeline not ready (e.g., no PDFs in ./docs)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Do not leak internal errors in production
        raise HTTPException(status_code=500, detail="Fehler bei der Verarbeitung der Anfrage.")


# ============================================================================
# Vapi.ai Call Assistant Endpoints
# ============================================================================

@app.post("/api/call/start", response_model=CallStatusResponse)
async def start_call(request: InitiateCallRequest):
    """
    Initiiert einen ausgehenden AI-Anruf über Vapi.ai
    
    Args:
        request: Enthält Telefonnummer und optional Kundenname
        
    Returns:
        CallStatusResponse mit Call-ID und aktuellem Status
        
    Raises:
        HTTPException: Bei Validierungs- oder API-Fehlern
    """
    try:
        vapi_service = get_vapi_service()
        
        # Anruf initiieren
        result = await vapi_service.initiate_call(
            phone_number=request.phoneNumber,
            customer_name=request.customerName
        )
        
        return CallStatusResponse(
            callId=result["call_id"],
            status=result["status"],
            phoneNumber=result["phone_number"],
            createdAt=result["created_at"],
            duration=result.get("duration"),
            cost=result.get("cost")
        )
        
    except ValueError as e:
        # Validierungsfehler (z.B. ungültige Telefonnummer oder fehlende Config)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Andere Fehler (z.B. Netzwerkprobleme)
        import logging
        logging.error(f"Fehler beim Initiieren des Anrufs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Fehler beim Starten des Anrufs. Bitte versuchen Sie es erneut."
        )


@app.get("/api/call/status/{call_id}", response_model=CallStatusResponse)
async def get_call_status(call_id: str):
    """
    Ruft den aktuellen Status eines Anrufs ab
    
    Args:
        call_id: Die ID des Anrufs
        
    Returns:
        CallStatusResponse mit aktuellem Status und Details
        
    Raises:
        HTTPException: Bei nicht gefundenem Anruf oder API-Fehlern
    """
    try:
        vapi_service = get_vapi_service()
        
        result = await vapi_service.get_call_status(call_id)
        
        return CallStatusResponse(
            callId=result["call_id"],
            status=result["status"],
            phoneNumber=result.get("phone_number", ""),
            createdAt=result.get("started_at", ""),
            duration=result.get("duration"),
            cost=result.get("cost"),
            endedReason=result.get("ended_reason")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import logging
        logging.error(f"Fehler beim Abrufen des Call-Status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Fehler beim Abrufen des Status. Bitte versuchen Sie es erneut."
        )


@app.post("/api/call/end")
async def end_call(request: EndCallRequest):
    """
    Beendet einen laufenden Anruf
    
    Args:
        request: Enthält die Call-ID
        
    Returns:
        Dict mit finalen Call-Informationen
        
    Raises:
        HTTPException: Bei nicht gefundenem Anruf oder API-Fehlern
    """
    try:
        vapi_service = get_vapi_service()
        
        result = await vapi_service.end_call(request.callId)
        
        return {
            "success": True,
            "callId": result["call_id"],
            "status": result["status"],
            "duration": result.get("duration"),
            "cost": result.get("cost")
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import logging
        logging.error(f"Fehler beim Beenden des Anrufs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Fehler beim Beenden des Anrufs. Bitte versuchen Sie es erneut."
        )


@app.post("/api/call/vapi-webhook")
async def vapi_webhook(request: dict):
    """
    Webhook-Endpoint für Vapi.ai Call Events und Function Calls
    
    Dieser Endpoint wird von Vapi aufgerufen für:
    - Call Status Updates (started, ended, etc.)
    - Function Calls (wenn der Assistant Informationen aus dem RAG System braucht)
    
    Args:
        request: Webhook-Payload von Vapi
        
    Returns:
        Response für Vapi (bei Function Calls)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    event_type = request.get("type")
    logger.info(f"Vapi Webhook erhalten: {event_type}")
    
    # Function Call vom Assistant - RAG System verwenden
    if event_type == "function-call":
        function_name = request.get("functionCall", {}).get("name")
        parameters = request.get("functionCall", {}).get("parameters", {})
        
        if function_name == "search_insurance_info":
            # RAG System aufrufen
            question = parameters.get("question", "")
            
            try:
                result = rag_service.run_rag(question.strip(), chat_history=[])
                
                # Antwort für Vapi formatieren
                response_text = result.answer
                
                # Quellen hinzufügen wenn verfügbar
                if result.sources:
                    sources_text = "\n\nQuellen: "
                    for src in result.sources[:2]:
                        sources_text += f"{src.document_title}"
                        if src.page:
                            sources_text += f" (Seite {src.page})"
                        sources_text += ", "
                    response_text += sources_text.rstrip(", ")
                
                return {
                    "result": response_text
                }
            except Exception as e:
                logger.error(f"Fehler bei RAG-Abfrage im Webhook: {e}")
                return {
                    "result": "Entschuldigung, ich konnte diese Information momentan nicht abrufen."
                }
    
    # Call Status Events loggen
    elif event_type in ["call-started", "call-ended", "call-failed"]:
        call_id = request.get("call", {}).get("id")
        logger.info(f"Call {call_id}: {event_type}")
    
    return {"success": True}


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

"""
FastAPI Backend für das RAG-System

Dieses Backend stellt REST-API-Endpoints zur Verfügung, um Fragen
an das RAG-System zu stellen und Antworten zu erhalten.
"""

import os
import sys
import json
from typing import List, Optional
from contextlib import asynccontextmanager
import asyncio

# #region agent log - Debug logging setup
_DEBUG_LOG_PATH = r"c:\Users\mirae\MVA_Versicherung_Langchain_main\.cursor\debug.log"
def _debug_log(hyp_id, loc, msg, data=None):
    try:
        import time
        entry = {"hypothesisId": hyp_id, "location": loc, "message": msg, "data": data or {}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"DEBUG LOG ERROR: {e}")
# #endregion

# #region agent log - Hypothesis A: Check __file__ and sys.path
_debug_log("A", "backend_api.py:top", "Script started", {"__file__": __file__, "cwd": os.getcwd()})
# #endregion

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Füge src-Verzeichnis zum Python-Pfad hinzu
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_THIS_DIR, 'src')
sys.path.insert(0, _SRC_PATH)

# #region agent log - Hypothesis A: Log computed paths
_debug_log("A", "backend_api.py:path_setup", "Path setup complete", {"_THIS_DIR": _THIS_DIR, "_SRC_PATH": _SRC_PATH, "sys.path[0]": sys.path[0], "src_exists": os.path.isdir(_SRC_PATH)})
# #endregion

# #region agent log - Hypothesis B: Try importing rag_service
try:
    from api.rag_service import run_rag, Source, AnswerResult
    _debug_log("B", "backend_api.py:import_rag", "rag_service import SUCCESS", {})
except Exception as _import_err:
    _debug_log("B", "backend_api.py:import_rag", "rag_service import FAILED", {"error": str(_import_err), "type": type(_import_err).__name__})
    raise
# #endregion


# Pydantic-Modelle für API-Requests und Responses
class AskRequest(BaseModel):
    """Request-Modell für Fragen"""
    question: str
    options: Optional[dict] = None  # Für zukünftige Erweiterungen


class SourceResponse(BaseModel):
    """Response-Modell für Quellen"""
    documentId: str
    documentTitle: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "documentId": "BAL-KFZ-2024-001",
                "documentTitle": "Kfz-Versicherung Leistungen 2024",
                "page": 12,
                "section": "3.2",
                "snippet": "Die Kfz-Haftpflichtversicherung deckt Personenschäden..."
            }
        }


class AskResponse(BaseModel):
    """Response-Modell für Antworten"""
    answer: str
    sources: List[SourceResponse]
    latencyMs: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Die Kfz-Haftpflichtversicherung deckt...",
                "sources": [
                    {
                        "documentId": "BAL-KFZ-2024-001",
                        "documentTitle": "Kfz-Versicherung Leistungen 2024",
                        "page": 12,
                        "section": "3.2",
                        "snippet": "Die Kfz-Haftpflichtversicherung deckt..."
                    }
                ],
                "latencyMs": 1250
            }
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle-Handler für FastAPI-App"""
    # Startup: Optional pipeline initialization.
    # Important: initializing can take a long time (PDF parsing + embeddings),
    # which would block the server from answering even /health. Therefore we
    # default to lazy init on first request.
    init_on_startup = os.getenv("INIT_PIPELINE_ON_STARTUP", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if init_on_startup:
        print("Initializing RAG pipeline on startup (INIT_PIPELINE_ON_STARTUP=true)...")
        try:
            from api.rag_service import initialize_pipeline, pdfs_have_changed

            # Run heavy init in a thread so the event loop isn't blocked.
            await asyncio.to_thread(initialize_pipeline, force_reindex=pdfs_have_changed())
            print("RAG pipeline initialized successfully.")
        except Exception as e:
            print(f"Warning: Error initializing RAG pipeline: {e}")
            print("Pipeline will be initialized on first request.")
    else:
        print("Skipping pipeline initialization on startup. Pipeline will initialize on first request.")
    yield
    # Shutdown: Cleanup falls nötig
    print("Shutting down RAG backend...")


# FastAPI-App erstellen
app = FastAPI(
    title="Baloise Dokumenten-Assistent API",
    description="REST-API für das agentische RAG-System zur Beantwortung von Fragen zu Versicherungsdokumenten",
    version="0.1.0",
    lifespan=lifespan
)

# CORS konfigurieren
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def convert_source_to_response(source: Source) -> SourceResponse:
    """Konvertiert Source-Dataclass zu SourceResponse-Modell"""
    return SourceResponse(
        documentId=source.document_id,
        documentTitle=source.document_title,
        page=source.page,
        section=source.section,
        snippet=source.snippet
    )


@app.get("/")
async def root():
    """Root-Endpoint für Health-Check"""
    return {
        "status": "online",
        "message": "Baloise Dokumenten-Assistent API",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Health-Check-Endpoint"""
    return {"status": "healthy"}


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Haupt-Endpoint zum Stellen einer Frage
    
    Args:
        request: AskRequest mit der Frage
        
    Returns:
        AskResponse mit Antwort, Quellen und Latenz
    """
    try:
        # Validiere Frage
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Frage darf nicht leer sein."
            )
        
        # Führe RAG-Pipeline aus
        result: AnswerResult = run_rag(request.question.strip(), chat_history=[])
        
        # Konvertiere Sources zu Response-Format
        sources_response = [
            convert_source_to_response(source) for source in result.sources
        ]
        
        # Erstelle Response
        response = AskResponse(
            answer=result.answer,
            sources=sources_response,
            latencyMs=result.latency_ms
        )
        
        return response
        
    except ValueError as e:
        # Fehler bei der Pipeline-Initialisierung
        raise HTTPException(
            status_code=503,
            detail=f"Service vorübergehend nicht verfügbar: {str(e)}"
        )
    except Exception as e:
        # Allgemeine Fehlerbehandlung
        print(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Fehler bei der Generierung der Antwort. Bitte versuchen Sie es erneut."
        )


@app.post("/api/feedback")
async def send_feedback(answer_id: str, useful: bool):
    """
    Endpoint zum Senden von Feedback zu einer Antwort
    
    Args:
        answer_id: ID der Antwort
        useful: Ob die Antwort hilfreich war
        
    Returns:
        Erfolgsstatus
    """
    # TODO: Implementiere Feedback-Logging/Analytics
    print(f"Feedback received - Answer ID: {answer_id}, Useful: {useful}")
    return {"success": True, "message": "Feedback wurde gespeichert."}


if __name__ == "__main__":
    import uvicorn
    
    # #region agent log - Hypothesis C/D/E: Log before uvicorn start
    _debug_log("C", "backend_api.py:main", "About to start uvicorn", {"reload": True, "host": "0.0.0.0", "port": 8000})
    _debug_log("E", "backend_api.py:main", "Working directory check", {"cwd": os.getcwd(), "docs_exists": os.path.isdir("./docs"), "chroma_exists": os.path.isdir("./chroma_db")})
    # #endregion
    
    # #region agent log - Hypothesis C: Try without reload first
    try:
        # Starte Server (reload=False to avoid Windows issues)
        uvicorn.run(
            "backend_api:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Changed from True - reload can cause issues on Windows
            log_level="info"
        )
    except Exception as _uvicorn_err:
        _debug_log("D", "backend_api.py:main", "uvicorn.run FAILED", {"error": str(_uvicorn_err), "type": type(_uvicorn_err).__name__})
        raise
    # #endregion


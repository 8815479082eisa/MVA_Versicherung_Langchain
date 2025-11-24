"""
FastAPI Backend für das RAG-System

Dieses Backend stellt REST-API-Endpoints zur Verfügung, um Fragen
an das RAG-System zu stellen und Antworten zu erhalten.
"""

import os
import sys
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Füge src-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.rag_service import run_rag, Source, AnswerResult


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
    # Startup: Initialisiere Pipeline beim Start
    print("Initializing RAG pipeline...")
    try:
        from api.rag_service import initialize_pipeline, pdfs_have_changed
        initialize_pipeline(force_reindex=pdfs_have_changed())
        print("RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Warning: Error initializing RAG pipeline: {e}")
        print("Pipeline will be initialized on first request.")
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
    
    # Starte Server
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


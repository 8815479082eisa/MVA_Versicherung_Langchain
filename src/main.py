"""
FastAPI Backend fuer das MVA Insurance Agentic RAG System
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Pfad fuer Imports hinzufuegen
sys.path.insert(0, str(Path(__file__).parent))

from api.rag_service import ask_question, initialize_pipeline

# FastAPI-App erstellen
app = FastAPI(
    title="MVA Insurance Agentic RAG API",
    description="Backend fuer das MVA Insurance RAG System",
    version="1.0.0"
)

# CORS-Middleware - Zugriff vom Frontend erlauben
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion sollte dies auf localhost begrenzt werden
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
    page: int = None
    section: str = None
    snippet: str = None


class AnswerResponse(BaseModel):
    """Antwort"""
    answer: str
    sources: list[Source]
    latencyMs: float = None


# Routes
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
        
        # RAG-Service aufrufen
        result = ask_question(
            question=request.question,
            short_answer=request.shortAnswer,
            structured_answer=request.structuredAnswer
        )
        
        # Ergebnis in AnswerResponse umwandeln
        sources = [
            Source(
                documentId=src.document_id,
                documentTitle=src.document_title,
                page=src.page,
                section=src.section,
                snippet=src.snippet
            )
            for src in result.get("sources", [])
        ]
        
        return AnswerResponse(
            answer=result.get("answer", ""),
            sources=sources,
            latencyMs=result.get("latency_ms")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Wird beim Serverstart ausgefuehrt"""
    print("📚 Backend gestartet...")
    try:
        initialize_pipeline()
        print("✅ Pipeline initialisiert")
    except Exception as e:
        print(f"⚠️ Fehler bei der Pipeline-Initialisierung: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Wird beim Server-Shutdown ausgefuehrt"""
    print("👋 Backend wird beendet...")


if __name__ == "__main__":
    # Server starten
    port = int(os.getenv("BACKEND_PORT", 8000))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    
    print(f"🚀 Backend laeuft unter http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True  # In development fuer Auto-Reload
    )

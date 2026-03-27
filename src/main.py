"""
FastAPI Backend fuer das MVA Insurance Agentic RAG System
"""

import os
import sys
import json
import re
from difflib import SequenceMatcher
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
    _startup_init_task = None
    _startup_init_error = None

    _load_faq_data()
    print(
        f"FAQ settings: FAQ_ONLY={FAQ_ONLY}, FAQ_FIRST={FAQ_FIRST}, "
        f"FAQ_MIN_SIMILARITY={FAQ_MIN_SIMILARITY}"
    )

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

    yield

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


# Routes
@app.get("/")
async def root():
    return {"status": "online", "service": "MVA Insurance Agentic RAG API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Serverstatus pruefen"""
    pipeline_ready = rag_service.is_pipeline_ready()
    pipeline_initializing = _startup_init_task is not None and not _startup_init_task.done()
    return {
        "status": "ok",
        "message": "Backend laeuft",
        "pipelineReady": pipeline_ready,
        "pipelineInitializing": pipeline_initializing,
        "pipelineInitError": _startup_init_error,
        "insuranceqaExactMatchShortcut": rag_service.insuranceqa_exact_match_shortcut_enabled(),
        "safetyEnabled": rag_service.safety_enabled(),
        "safetyMode": rag_service.safety_mode(),
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(request: AskQuestionRequest):
    """
    Frage anfragen und Antwort erhalten
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Die Frage darf nicht leer sein")

        question = request.question.strip()

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
                return AnswerResponse(
                    answer=_shape_answer(faq_answer, request.shortAnswer, request.structuredAnswer),
                    sources=sources,
                    latencyMs=0,
                )

            if FAQ_ONLY:
                raise HTTPException(
                    status_code=404,
                    detail="No local FAQ match found. Lower FAQ_MIN_SIMILARITY or update FAQ_FILE.",
                )

        # RAG-Service path (uses local model provider)
        result = rag_service.run_rag(question, chat_history=[])
        
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
        
    except (ValueError, RuntimeError) as e:
        # Typically: pipeline not ready/misconfigured (e.g., missing retriever or PDFs)
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        # Preserve intended API errors (e.g. 400/404 from FAQ-only path)
        raise
    except Exception as e:
        # Show raw error only in explicit debug mode.
        if _env_flag("DEBUG_API_ERRORS", default=False):
            raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Fehler bei der Verarbeitung der Anfrage.")


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

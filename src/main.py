"""
FastAPI Backend برای MVA Insurance Agentic RAG System
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# اضافه کردن مسیر برای import
sys.path.insert(0, str(Path(__file__).parent))

from api.rag_service import ask_question, initialize_pipeline

# ایجاد FastAPI app
app = FastAPI(
    title="MVA Insurance Agentic RAG API",
    description="Backend برای سیستم RAG بیمه MVA",
    version="1.0.0"
)

# CORS middleware - اجازه دسترسی از Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production باید به localhost محدود شود
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class AskQuestionRequest(BaseModel):
    """درخواست سوال"""
    question: str
    shortAnswer: bool = False
    structuredAnswer: bool = False


class Source(BaseModel):
    """منبع سند"""
    documentId: str
    documentTitle: str
    page: int = None
    section: str = None
    snippet: str = None


class AnswerResponse(BaseModel):
    """پاسخ سوال"""
    answer: str
    sources: list[Source]
    latencyMs: float = None


# Routes
@app.get("/health")
async def health_check():
    """بررسی وضعیت سرور"""
    return {"status": "ok", "message": "Backend درحال اجرا است"}


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(request: AskQuestionRequest):
    """
    درخواست سوال و دریافت پاسخ
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="سوال نمی‌تواند خالی باشد")
        
        # فراخوانی RAG service
        result = ask_question(
            question=request.question,
            short_answer=request.shortAnswer,
            structured_answer=request.structuredAnswer
        )
        
        # تبدیل نتیجه به AnswerResponse
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
        raise HTTPException(status_code=500, detail=f"خطا: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """اجرا هنگام شروع سرور"""
    print("📚 Backend شروع شد...")
    try:
        initialize_pipeline()
        print("✅ Pipeline مقداردهی شد")
    except Exception as e:
        print(f"⚠️ خطا در مقداردهی Pipeline: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """اجرا هنگام خاموشی سرور"""
    print("👋 Backend خاموش می‌شود...")


if __name__ == "__main__":
    # اجرای سرور
    port = int(os.getenv("BACKEND_PORT", 8000))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    
    print(f"🚀 Backend درحال اجرا بر روی http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True  # در development برای auto-reload
    )

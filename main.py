"""
Legacy compatibility module.

Migration note:
- Active production path is `backend_api.py -> src.main -> src/api/rag_service.py`.
- This module now forwards to `src/api/rag_service.py` to avoid duplicated RAG logic.
- Kept for compatibility with evaluation/utility scripts importing `main.py` symbols.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document

from src.api import rag_service


PDF_DIRECTORY = rag_service.PDF_DIRECTORY
AUDIT_LOG_FILE = rag_service.AUDIT_LOG_FILE


def get_pdf_files(directory: str) -> List[str]:
    return rag_service.get_pdf_files(directory)


def load_and_split_documents(pdf_files: List[str]) -> List[Document]:
    return rag_service.load_and_split_documents(pdf_files)


def initialize_embeddings():
    return rag_service.initialize_embeddings()


def create_hybrid_retriever(all_splits: List[Document], embeddings, force_reindex: bool = False):
    return rag_service.create_hybrid_retriever(all_splits, embeddings, force_reindex=force_reindex)


def initialize_reranker():
    return rag_service.initialize_reranker()


def initialize_compressor():
    return rag_service.initialize_compressor()


def initialize_llm():
    return rag_service.initialize_llm()


def initialize_router_llm():
    return rag_service.initialize_router_llm()


def initialize_self_check_llm():
    return rag_service.initialize_self_check_llm()


def initialize_query_rewrite_llm():
    return rag_service.initialize_query_rewrite_llm()


def decide_retrieval(router_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    return rag_service.decide_retrieval(router_llm, query, chat_history)


def rerank_documents(query: str, documents: List[Document], reranker_model, top_k: int = 3) -> List[Document]:
    return rag_service.rerank_documents(query, documents, reranker_model, top_k=top_k)


def rewrite_query(query_rewrite_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    return rag_service.rewrite_query(query_rewrite_llm, query, chat_history)


def perform_self_check(
    self_check_llm,
    query_rewrite_llm,
    original_query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
    max_retries: int = 2,
):
    return rag_service.perform_self_check(
        self_check_llm,
        query_rewrite_llm,
        original_query,
        retrieved_docs,
        chat_history,
        max_retries,
    )


def generate_answer(
    llm,
    query: str,
    context_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
):
    """
    Backward-compatible signature used by older evaluation scripts.

    Old path expected `(answer, token_usage)`. Unified path returns no token usage,
    so token usage stays empty for compatibility.
    """
    answer = rag_service.generate_answer(llm, query, context_docs, chat_history)
    return answer, {}


def _doc_to_json(doc: Any) -> dict:
    if isinstance(doc, Document):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}
    if isinstance(doc, dict):
        return doc
    return {"page_content": str(doc), "metadata": {}}


def audit_log(
    query: str,
    retrieved_documents: List[Any],
    compressed_context: List[Any],
    generated_answer: str,
    chat_history: Optional[List[dict]] = None,
    **extra_fields,
) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_documents": [_doc_to_json(doc) for doc in (retrieved_documents or [])],
        "compressed_context": [_doc_to_json(doc) for doc in (compressed_context or [])],
        "generated_answer": generated_answer,
        "chat_history": chat_history or [],
    }
    payload.update(extra_fields)

    audit_path = Path(AUDIT_LOG_FILE)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def perform_safety_checks(
    query: str,
    context_docs: List[Document],
    answer: str,
    chat_history: Optional[List[dict]] = None,
) -> bool:
    del query, context_docs, chat_history
    # Keep behavior permissive as in legacy scripts unless output is empty.
    return bool((answer or "").strip())


def _print_sources(sources: List[rag_service.Source]) -> None:
    if not sources:
        return
    print("\nSources:")
    for i, src in enumerate(sources, start=1):
        page = f", page {src.page}" if src.page is not None else ""
        title = src.document_title or src.document_id or "unknown"
        print(f"  {i}. {title}{page}")


def main() -> int:
    print("--- RAG CLI (compat mode, type 'exit' to quit) ---")
    while True:
        q = input("\nUser: ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            return 0

        try:
            result = rag_service.run_rag(q, chat_history=[])
            print(f"\nAssistant: {result.answer}")
            _print_sources(result.sources)
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

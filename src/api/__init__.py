"""API package bootstrap for the active RAG service."""

from . import rag_service as rag_service

try:
    from src.core.reranker_windowing import install_windowed_reranking
except ImportError:  # Compatibility when ``src`` itself is on sys.path.
    from core.reranker_windowing import install_windowed_reranking

install_windowed_reranking(rag_service)

__all__ = ["rag_service"]

"""
Simple CLI for the RAG pipeline.

Usage:
  python scripts/cli_rag.py

Notes:
  - Uses the canonical pipeline implementation in `src/api/rag_service.py`.
  - Backend/API entrypoint remains `uvicorn src.main:app`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is importable so `src.*` imports work everywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import rag_service as rag_service  # noqa: E402


def _print_sources(sources: list[rag_service.Source]) -> None:
    if not sources:
        return
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        page = f", page {s.page}" if s.page is not None else ""
        title = s.document_title or s.document_id or "unknown"
        print(f"  {i}. {title}{page}")


def main() -> int:
    print("--- RAG CLI (type 'exit' to quit) ---")
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
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    raise SystemExit(main())


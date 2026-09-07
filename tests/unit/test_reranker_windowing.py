from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from src.api import rag_service
from src.core.reranker_windowing import reranker_text_windows


class _TailAwareReranker:
    def __init__(self) -> None:
        self.pairs: list[list[str]] = []

    def compute_score(self, pairs):
        self.pairs = list(pairs)
        return [
            10.0 if "worldwide cover lasts one year" in text.casefold() else 0.1
            for _, text in self.pairs
        ]


def test_reranker_windows_always_include_document_tail() -> None:
    text = "prefix " * 80 + "worldwide cover lasts one year"

    windows = reranker_text_windows(text, 256)

    assert len(windows) > 1
    assert any("worldwide cover lasts one year" in window for window in windows)
    assert windows[-1] == text[-256:]


def test_tail_evidence_can_win_reranking(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_service,
        "SETTINGS",
        SimpleNamespace(reranker=SimpleNamespace(max_doc_chars=256)),
    )
    model = _TailAwareReranker()
    generic = Document(
        page_content=(
            "Household contents insurance overview with generic wording but no "
            "duration statement."
        ),
        metadata={"source": "household-overview.pdf"},
    )
    tail_evidence = Document(
        page_content=(
            ("Household contents background material. " * 30)
            + "Worldwide cover lasts one year."
        ),
        metadata={"source": "household-brochure-en.pdf"},
    )

    ranked = rag_service.rerank_documents(
        "For how long does household contents insurance provide worldwide cover?",
        [generic, tail_evidence],
        model,
        top_k=1,
    )

    assert ranked == [tail_evidence]
    assert len(model.pairs) > 2
    assert any(
        "worldwide cover lasts one year" in text.casefold()
        for _, text in model.pairs
    )
    assert getattr(rag_service.rerank_documents, "_windowed_reranking", False) is True

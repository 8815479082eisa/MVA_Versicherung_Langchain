from __future__ import annotations

from typing import Any


def reranker_text_windows(text: str, max_chars: int) -> list[str]:
    """Split reranker input into overlapping windows and always include the tail.

    Cross-encoders should score evidence throughout a retrieved chunk rather than only
    the first ``max_chars`` characters. The window size remains bounded by the existing
    configuration, while overlap protects facts that cross a window boundary.
    """

    value = str(text or "")
    window_size = max(256, int(max_chars or 0))
    if len(value) <= window_size:
        return [value]

    overlap = min(max(200, window_size // 5), window_size - 1)
    step = max(1, window_size - overlap)
    starts = list(range(0, max(1, len(value) - window_size + 1), step))
    tail_start = max(0, len(value) - window_size)
    if tail_start not in starts:
        starts.append(tail_start)

    return [value[start : start + window_size] for start in sorted(set(starts))]


def _numeric_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, tuple):
        raw_scores = list(raw_scores)
    if not isinstance(raw_scores, list):
        raw_scores = [raw_scores]

    scores = [float(score) for score in raw_scores]
    if len(scores) != expected_count:
        raise ValueError(
            "Reranker returned an unexpected number of scores: "
            f"expected={expected_count}, actual={len(scores)}"
        )
    return scores


def install_windowed_reranking(rag_service_module: Any) -> None:
    """Install windowed scoring into the active rag_service module.

    ``rag_service.py`` is intentionally kept as the legacy orchestration source of
    truth. This small installer replaces only its reranking function while reusing all
    product-affinity, evidence-priority, neighbor-preservation and diagnostic helpers.
    """

    current = getattr(rag_service_module, "rerank_documents", None)
    if getattr(current, "_windowed_reranking", False):
        return

    legacy_rerank = current

    def rerank_documents_windowed(
        query: str,
        documents: list[Any],
        reranker_model: Any,
        top_k: int = 3,
    ) -> list[Any]:
        if not documents:
            rag_service_module.record_evidence(
                "reranker",
                {"query": query, "input": [], "final": [], "enabled": False},
            )
            return []

        if reranker_model is None:
            final_documents = documents[:top_k]
            rag_service_module.record_evidence(
                "reranker",
                {
                    "query": query,
                    "enabled": False,
                    "input": [
                        rag_service_module._document_evidence(doc, rank=index + 1)
                        for index, doc in enumerate(documents)
                    ],
                    "final": [
                        rag_service_module._document_evidence(doc, rank=index + 1)
                        for index, doc in enumerate(final_documents)
                    ],
                },
            )
            return final_documents

        all_documents = list(documents)
        product_compatible = [
            doc
            for doc in all_documents
            if rag_service_module._insurance_product_affinity(query, doc) >= 0
        ]
        minimum_candidate_count = min(max(top_k, 1), len(all_documents))
        product_prefilter_applied = (
            len(product_compatible) >= minimum_candidate_count
        )
        if product_prefilter_applied:
            documents = product_compatible
        else:
            documents = all_documents

        max_chars = rag_service_module.SETTINGS.reranker.max_doc_chars
        pairs: list[list[str]] = []
        pair_owners: list[int] = []
        window_counts: list[int] = []
        for document_index, document in enumerate(documents):
            windows = reranker_text_windows(document.page_content or "", max_chars)
            window_counts.append(len(windows))
            for window in windows:
                pairs.append([query, window])
                pair_owners.append(document_index)

        pair_scores = _numeric_scores(
            reranker_model.compute_score(pairs),
            len(pairs),
        )
        numeric_scores = [float("-inf")] * len(documents)
        for owner, score in zip(pair_owners, pair_scores):
            numeric_scores[owner] = max(numeric_scores[owner], score)

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda idx: (
                rag_service_module._insurance_product_affinity(
                    query, documents[idx]
                ),
                rag_service_module._reranker_evidence_priority(
                    query, documents[idx]
                ),
                numeric_scores[idx],
            ),
            reverse=True,
        )[:top_k]
        ranked_indices = rag_service_module._preserve_high_value_context_neighbors(
            documents,
            ranked_indices,
            top_k=top_k,
        )

        for index in ranked_indices:
            documents[index].metadata["_diagnostic_reranker_score"] = numeric_scores[
                index
            ]

        final_documents = [documents[idx] for idx in ranked_indices]
        rag_service_module.record_evidence(
            "reranker",
            {
                "query": query,
                "enabled": True,
                "windowedScoring": True,
                "maxDocCharsPerWindow": max_chars,
                "windowCounts": [
                    {
                        "inputRank": index + 1,
                        "count": window_counts[index],
                    }
                    for index in range(len(documents))
                ],
                "productPrefilterApplied": product_prefilter_applied,
                "productPrefilterDropped": [
                    rag_service_module._document_evidence(doc, rank=index + 1)
                    for index, doc in enumerate(all_documents)
                    if doc not in documents
                ],
                "input": [
                    rag_service_module._document_evidence(
                        doc,
                        rank=index + 1,
                        score=numeric_scores[index],
                        score_label="rerankerScore",
                    )
                    | {
                        "productAffinity": rag_service_module._insurance_product_affinity(
                            query, doc
                        )
                    }
                    | {
                        "evidencePriority": rag_service_module._reranker_evidence_priority(
                            query, doc
                        )
                    }
                    for index, doc in enumerate(documents)
                ],
                "final": [
                    rag_service_module._document_evidence(
                        documents[index],
                        rank=rank + 1,
                        score=numeric_scores[index],
                        score_label="rerankerScore",
                    )
                    for rank, index in enumerate(ranked_indices)
                ],
            },
        )
        return final_documents

    rerank_documents_windowed._windowed_reranking = True
    rerank_documents_windowed._legacy_rerank_documents = legacy_rerank
    rag_service_module.rerank_documents = rerank_documents_windowed

# pipeline-20260326

## Purpose
`pipeline-20260326` documents the current FastAPI-based InsuranceQA RAG evaluation pipeline after reranker recovery.

This version keeps benchmark-safe behavior (`INSURANCEQA_EXACT_MATCH_SHORTCUT=false`) and restores active reranking when `FlagEmbedding` is unavailable by using a `sentence-transformers` CrossEncoder fallback.

## Code Version
The exact code commit used for a specific run is recorded in the corresponding test-result artifact (`commit-hash.txt`).

## AI Models Used
Source of truth: `.env`, `src/config/models.py`, `src/api/rag_service.py`.

- Provider: `ollama`
- Ollama base URL: `http://141.41.32.94:41314`
- Answer model: `rnj-1:8b`
- Router model: `functiongemma:270m`
- Self-check model: `lfm2.5-thinking:1.2b`
- Query rewrite model: `lfm2.5-thinking:1.2b`
- Compressor model: `lfm2.5-thinking:1.2b`
- Embedding model: `BAAI/bge-m3` (`cpu`)
- Primary reranker implementation: `FlagEmbedding.FlagReranker`
- Reranker fallback implementation: `sentence_transformers.CrossEncoder` adapter (enabled in code path when FlagEmbedding import/init fails)
- Final reranker fallback (only if both reranker implementations fail): retrieval order

## Pipeline Components
1. API backend
- Entrypoint: `uvicorn src.main:app` (`backend_api.py` remains compatibility wrapper)
- Endpoints: `GET /`, `GET /health`, `POST /api/ask`

2. Retrieval and indexing
- Chroma persistent vector store
- Hybrid retrieval for document corpus (BM25 + vector retrieval)
- InsuranceQA retriever integration (`USE_INSURANCEQA_DATA=true`, `INSURANCEQA_RETRIEVAL_MODE=switch`)

3. Query processing and control
- Retrieval routing chain exists (`decide_retrieval`), but retrieval is forced by config (`RAG_FORCE_RETRIEVAL=true`)
- Query rewrite (`rewrite_query`) and self-check (`perform_self_check`) loops

4. Reranking
- `rerank_documents` consumes `compute_score(...)` from:
  - FlagEmbedding reranker when available
  - CrossEncoder adapter when FlagEmbedding is not available

5. Answer generation
- LLM answer generation via prompt chain (`generate_answer`)
- Fallback prompt when empty model response is returned

6. Evaluation
- API-based InsuranceQA evaluation via `scripts/evaluation/eval_insuranceqa.py` (wrapper: `scripts/eval_insuranceqa.py`)
- Metrics: Exact Match (EM), Token F1

## Notes
- This pipeline version is intended for real RAG evaluation (no direct dataset-answer shortcut for benchmark path).
- Main functional change vs. previous registration: reranking is no longer silently disabled when `FlagEmbedding` is missing; CrossEncoder fallback reranking is applied.
- If neither reranker backend is available, the system still degrades safely to retrieval order.

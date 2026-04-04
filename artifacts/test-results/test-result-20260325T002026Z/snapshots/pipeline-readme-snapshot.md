# pipeline-20260325

## Purpose
`pipeline-20260325` documents the current FastAPI-based InsuranceQA evaluation pipeline intended for real RAG benchmarking via `/api/ask`.

This version is configured to avoid trivial benchmark leakage from direct dataset answer return:
- `INSURANCEQA_EXACT_MATCH_SHORTCUT=false` in `.env`
- evaluation script guard against shortcut source id (`insuranceqa_v2_local`) unless explicitly overridden with `--allow-dataset-shortcut`

## Code Version
The exact Git commit for a specific benchmark run is documented in the corresponding test-result folder/artifact (not hardcoded in this README).

## AI Models Used
Source of truth: `.env`, `src/config/models.py`, `src/api/rag_service.py`.

- Model provider: `ollama` (hard-wired in `src/config/models.py`)
- Ollama base URL: `http://141.41.32.94:41314` (`.env: OLLAMA_BASE_URL`)
- Answer model: `rnj-1:8b` (`.env: ANSWER_MODEL`)
- Router model: `functiongemma:270m` (`.env: ROUTER_MODEL`)
- Self-check model: `lfm2.5-thinking:1.2b` (`.env: SELF_CHECK_MODEL`)
- Query rewrite model: `lfm2.5-thinking:1.2b` (`.env: QUERY_REWRITE_MODEL`)
- Compressor model: `lfm2.5-thinking:1.2b` (`.env: COMPRESSOR_MODEL`)
- Embedding model: `BAAI/bge-m3` on `cpu` (`.env: EMBEDDING_MODEL`, `EMBEDDING_DEVICE`)
- Reranker model: `BAAI/bge-reranker-base` (`.env: RERANKER_MODEL`)
- Reranker fallback behavior: if `FlagEmbedding` is unavailable or init fails, retrieval order is used (see `build_reranker()` in `src/api/rag_service.py`)

Config values not explicitly set in `.env` but active via code defaults:
- `RETRIEVE_TOP_K=8`, `BM25_TOP_K=5`, `VECTOR_TOP_K=5`
- `RERANK_TOP_K=5`
- `MAX_SELF_CHECK_RETRIES=2`
- `ENABLE_CONTEXT_COMPRESSION=false`
- `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`
- `ANSWER_TEMPERATURE=0.1`, `AUX_TEMPERATURE=0.0`
- `LLM_MAX_TOKENS=None` and `LLM_TIMEOUT_SECONDS=None` unless set

## Pipeline Components
1. API entrypoint and lifecycle
- Canonical backend entry: `uvicorn src.main:app` (`backend_api.py` is a compatibility wrapper).
- Startup warmup controlled by `INIT_PIPELINE_ON_STARTUP` (`src/main.py` lifespan).
- Health endpoint: `GET /health` exposes `pipelineReady`, `pipelineInitializing`, `pipelineInitError`, and `insuranceqaExactMatchShortcut`.

2. Request handling
- Main endpoint: `POST /api/ask` with payload `{ question, shortAnswer, structuredAnswer }`.
- Optional FAQ path is available (`FAQ_ONLY`, `FAQ_FIRST`), otherwise requests go to `rag_service.run_rag(...)`.

3. Core RAG pipeline (`src/api/rag_service.py`)
- Document ingestion/splitting for PDFs (`PyPDFLoader`, `RecursiveCharacterTextSplitter`).
- Vector store: Chroma persistent DB.
- Hybrid retrieval for PDF corpus: BM25 + vector retrieval with deduplication.
- InsuranceQA retriever integration (`USE_INSURANCEQA_DATA=true`):
  - Current mode from `.env`: `INSURANCEQA_RETRIEVAL_MODE=switch`
  - In `switch` mode, retrieval uses InsuranceQA retriever as primary retriever path.
- Router/self-check/rewrite chain exists:
  - `decide_retrieval()`, `perform_self_check()`, `rewrite_query()`
  - Effective behavior in current config: retrieval is forced (`RAG_FORCE_RETRIEVAL=true`), so router decision is bypassed for retrieval gating.
- Reranking:
  - `rerank_documents()` with configured reranker model
  - fallback to retrieval order if reranker unavailable
- Optional context compression:
  - `compress_context()` exists but is disabled by default (`ENABLE_CONTEXT_COMPRESSION=false`)
- Answer generation:
  - `generate_answer()` with prompt chain and a fallback prompt if empty model output is returned
- Audit logging:
  - `AUDIT_LOG_FILE=./data/processed/logs/audit.log`

4. InsuranceQA exact-match shortcut controls
- Shortcut logic exists in code (`lookup_insuranceqa_answer()` + exact-match branch in `RAGPipeline.run()`).
- Benchmark status for this pipeline version: shortcut is disabled by config (`INSURANCEQA_EXACT_MATCH_SHORTCUT=false`).

5. Evaluation via API
- Script: `scripts/evaluation/eval_insuranceqa.py` (compat wrapper: `scripts/eval_insuranceqa.py`).
- Uses API predictions from `/api/ask` and computes:
  - Exact Match (EM)
  - Token F1
- Dataset source for references/questions:
  - local JSONL by default: `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl`
  - fallback to HF dataset only if local file is missing
- Safety check for benchmark integrity:
  - detects and fails on shortcut source id `insuranceqa_v2_local` unless `--allow-dataset-shortcut` is explicitly passed.

## Notes
- This pipeline version is intended for real RAG evaluation through API-generated answers, not direct dataset answer passthrough.
- `pipelineReady=true` in `/health` indicates initialization state only; it does not by itself guarantee model-quality outcomes.
- In current setup, output paths in evaluation are relative to the runtime working directory. In mixed local/remote workflows, verify absolute paths to avoid reading stale files from a different environment.
- `LLM_TIMEOUT_SECONDS` and `LLM_MAX_TOKENS` are not explicitly set in `.env`; long-running model calls are therefore possible and should be monitored in larger runs.

# pipeline-20260327

## Purpose
`pipeline-20260327` documents the RAG pipeline version with integrated safety auditing and safety-aware runtime telemetry.

This version keeps benchmark-safe behavior (`INSURANCEQA_EXACT_MATCH_SHORTCUT=false`) and adds configurable safety checks across:
- pre-query stage
- retrieved context stage
- post-generation stage

## Code Version
The exact code commit used for a specific run is recorded in the corresponding test-result artifact and Git tag.

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
- Reranker primary: `FlagEmbedding.FlagReranker`
- Reranker fallback: `sentence_transformers.CrossEncoder`

## Safety Layer
Safety configuration is loaded from environment via `src/config/models.py`:
- `SAFETY_ENABLED`
- `SAFETY_MODE` (`off`, `monitor`, `enforce`)
- `SAFETY_MIN_GROUNDEDNESS`
- `SAFETY_BLOCK_PII`
- `SAFETY_BLOCK_INJECTION`
- `SAFETY_FAIL_CLOSED`
- `SAFETY_FALLBACK_TEXT`

Core implementation:
- `src/core/safety_audit.py`

Pipeline integration points:
- `src/api/rag_service.py`
  - pre-query safety check
  - context safety check
  - post-generation groundedness/safety check
  - safety action application and fallback handling
  - safety metadata persisted to `audit.log`

Health exposure:
- `src/main.py` includes `safetyEnabled` and `safetyMode` in `GET /health`.

Setup defaults:
- `setup.sh` now writes safety env vars into generated `.env`.

## Notes
- `monitor` mode logs risks while allowing responses.
- `enforce` mode can block/fallback based on configured thresholds and rules.
- Safety telemetry fields in audit output enable post-hoc analysis and audit export workflows.

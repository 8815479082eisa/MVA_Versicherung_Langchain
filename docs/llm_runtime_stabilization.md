# Local LLM Runtime Stabilization

The local Ollama path is bounded per stage. CRM-only requests do not use these
models and remain available when Ollama is stopped, unreachable, slow, or
missing a model.

## Configuration

The complete non-secret example is `.env.llm.example`. `ChatOllama` receives
the supported Ollama `num_predict` option and an HTTP client timeout.

| Stage | Output limit | Timeout |
|---|---:|---:|
| Router | 8 | 20 s |
| NeMo guardrail | 8 | 30 s |
| Self-check | 8 | 30 s |
| Query rewrite | 64 | 45 s |
| Context compressor | 384 | 60 s |
| Reranker | not generative | 60 s |
| Answer | 384 | 120 s |
| Entire `/api/ask` RAG portion | — | 180 s |

`LLM_MAX_RETRIES=1` permits one retry after an availability failure. A timeout
is never retried. `RETRIEVAL_MAX_RETRIES=1` bounds retrieval retries and the
self-check/rewrite cycle. The current `FlagReranker` is a local cross-encoder;
`LLM_MAX_TOKENS_RERANKER` is documented for configuration symmetry but is not
passed to this non-generative model.

Change a limit only in the ignored `.env`, record the old/new value and reason,
restart only the backend, and rerun smoke, pure-retrieval, retrieval E2E, and
combined E2E validation. Do not change the selected models, retrieval k-values,
reranker parameters, or guardrail thresholds to hide a capacity failure.

## Ollama addresses

- Host scripts: `http://localhost:11434`
- Backend in Docker Desktop: `http://host.docker.internal:11434`

The backend URL is resolved from its own environment. Never dump the complete
environment while diagnosing it.

## Readiness and smoke tests

`GET /health` preserves the existing pipeline and CRM fields and adds:

- `ollamaReachable`
- `llmReady`
- `llmError`
- `answerModelReady`
- `guardrailModelReady`
- `embeddingReady`
- `retrievalReady`

The check calls only Ollama `/api/tags`; it does not generate text. Run
generation smokes separately:

```powershell
docker exec mva-backend python scripts/test_ollama_runtime.py --stage answer
docker exec mva-backend python scripts/test_ollama_runtime.py --stage guardrail
```

Both commands use short output limits, stage timeouts, and sanitized JSON.

## Pure retrieval

Pure retrieval never runs router, guardrail, self-check, rewrite, compressor,
or answer generation. It reads the existing collection without reindexing:

```powershell
docker exec mva-backend python scripts/test_retrieval_pipeline.py --mode without-reranker
docker exec mva-backend python scripts/test_retrieval_pipeline.py --mode with-reranker
```

The two modes are explicitly named because the existing reranker is optional
local CPU work. Results include source/page metadata and component timings.
No public diagnostics endpoint was added.

## Errors

Expected runtime conditions use stable codes:

| HTTP | Code | Meaning |
|---:|---|---|
| 400 | `MALFORMED_REQUEST` | Missing/invalid request body or empty question |
| 403 | `FORBIDDEN_OPERATION` | Denied CRM operation |
| 503 | `LLM_UNAVAILABLE` | Ollama transport unavailable |
| 503 | `LLM_MODEL_MISSING` | Required model is not installed |
| 503 | `GUARDRAIL_INVALID_OUTPUT` | Classifier output is outside its closed vocabulary |
| 503 | `RETRIEVAL_FAILED` | Retrieval could not complete |
| 504 | `LLM_STAGE_TIMEOUT` | One named stage timed out |
| 504 | `REQUEST_TIMEOUT` | Total RAG request budget expired |
| 500 | `INTERNAL_ERROR` | Unexpected programming/runtime error |

Client errors contain no prompts, keys, stack traces, or raw environment data.

## Combined partial responses

If CRM succeeds but the document path has an expected availability or timeout
failure, `/api/ask` returns HTTP 206 with:

- `status: "partial"`
- `route: "combined"`
- valid `crmResult`
- `knowledgeResult: null`
- a structured `warning`
- CRM sources only

The answer states that document evidence is unavailable and that no coverage
or claim decision was inferred. HTTP 206 is not a successful combined E2E
result. Unexpected internal errors still return HTTP 500.

## Finding a slow stage

Each API response and sanitized server diagnostic includes a request ID,
route, `stageTimings`, `stageStatus`, retry counts, timeout stage, total time,
result status, and partial flag. A stage that did not run has no duration and
is marked `not_run` when applicable. The benchmark copies these diagnostics to
`reports/llm_runtime_benchmark.json`.

Python cannot forcibly kill an arbitrary synchronous worker thread. Ollama HTTP
timeouts stop the transport wait. For local CPU stages and total-request
timeouts, a one-worker gate remains occupied until abandoned work actually
finishes, preventing unbounded background accumulation. CRM-only bypasses this
gate.

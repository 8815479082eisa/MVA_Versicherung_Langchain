# LLM Runtime Stabilization Report

Date: 2026-07-26  
Repository: `C:\Users\mirae\MVA_Versicherung_Langchain_main`

## 1. Result summary

| Validation item | Result |
|---|---|
| Runtime stabilization implementation | PASSED |
| Minimal answer-model smoke | PASSED |
| Minimal guardrail-model smoke | PASSED |
| Backend readiness | PASSED |
| Pure retrieval without reranker | PASSED |
| Pure retrieval with existing reranker | PASSED |
| CRM-only live E2E | PASSED |
| Forbidden-operation live request | PASSED |
| Retrieval-only complete live E2E | FAILED |
| Combined complete live E2E | FAILED |
| Combined partial-response contract | PASSED |
| Final focused regression suite | PASSED |
| Chroma/PDF integrity | PASSED |
| Read-only and secret security validation | PASSED |

Overall complete-pipeline validation: **FAILED**. Retrieval-only timed out in
`self_check`; combined returned the intended HTTP 206 partial response and is
therefore not counted as a complete E2E pass.

## 2. Original failure state

Before this phase, CRM-only worked, retrieval-only exceeded 360 seconds, and
combined ended as HTTP 500 after Ollama became unavailable. Not every
generation stage had an explicit output bound; stage/total timeouts were
incomplete; expected runtime failures could become generic HTTP 500 responses;
and latency did not consistently identify the slow stage.

The pre-implementation inspection is preserved in
`reports/llm_runtime_stabilization_plan.md`.

## 3. Inspected runtime stages and models

| Stage | Model/runtime | Call type |
|---|---|---|
| Route planning | deterministic first, `phi3:mini` fallback | synchronous Ollama through bounded worker |
| NeMo guardrail | `phi3:mini` | async NeMo generation behind bounded thread gate |
| Self-check | `phi3:mini` | synchronous Ollama through bounded worker |
| Query rewrite | `phi3:mini` | synchronous Ollama through bounded worker |
| Context compression | `phi3:mini` | synchronous Ollama through bounded worker |
| Reranker | `BAAI/bge-reranker-v2-m3` | synchronous non-generative cross-encoder |
| Final answer | `qwen2.5:7b-instruct` | synchronous Ollama through bounded worker |

The installed `langchain-ollama` implementation was inspected directly.
`ChatOllama` supports `num_predict` plus `client_kwargs`,
`async_client_kwargs`, and `sync_client_kwargs`; it does not support a
top-level `timeout`. Tests assert output-limit and HTTP-timeout propagation.

## 4. Output limits

| Stage | `num_predict` |
|---|---:|
| Router | 8 |
| Guardrail | 8 |
| Self-check | 8 |
| Query rewrite | 64 |
| Context compressor | 384 |
| Answer | 384 |

`LLM_MAX_TOKENS_RERANKER=64` is reserved in configuration but is not passed to
the non-generative reranker. Router, guardrail, and self-check require an exact
closed-vocabulary label. Parsing normalizes whitespace/case and rejects
explanations or substring matches.

Result: **PASSED**.

## 5. Timeouts and bounded concurrency

| Operation | Default |
|---|---:|
| Router | 20 s |
| Guardrail | 30 s |
| Self-check | 30 s |
| Query rewrite | 45 s |
| Context compressor | 60 s |
| Reranker | 60 s |
| Answer | 120 s |
| Retrieval operation | 60 s |
| RAG portion of `/api/ask` | 180 s |

Stage and total timeouts use distinct exceptions/codes. A single-worker gate
remains occupied until abandoned synchronous work returns, preventing
unbounded background accumulation. CRM-only bypasses this gate.

Python cannot forcibly terminate a running synchronous worker thread. Ollama's
HTTP timeout stops transport waiting; local CPU work is isolated by the gate.

Result: **PASSED**.

## 6. Retry behavior

- `LLM_MAX_RETRIES=1`: at most one retry after the first attempt.
- `RETRIEVAL_MAX_RETRIES=1`: bounds retrieval/rewrite recovery.
- Timeouts are not retried.
- Retry reason/count is recorded in diagnostics.
- Retry exhaustion is covered by tests.

Result: **PASSED**.

## 7. Readiness and health

Existing pipeline and CRM fields were preserved. Final health:

```text
status=ok
pipelineReady=true
crmEnabled=true
crmReady=true
crmError=null
ollamaReachable=true
llmReady=true
llmError=null
answerModelReady=true
guardrailModelReady=true
embeddingReady=true
retrievalReady=true
```

Health uses the lightweight Ollama model-list endpoint, not generation.

Result: **PASSED**.

## 8. HTTP error mapping

| HTTP | Stable code |
|---:|---|
| 400 | `MALFORMED_REQUEST` |
| 403 | `FORBIDDEN_OPERATION` |
| 503 | `LLM_UNAVAILABLE` |
| 503 | `LLM_MODEL_MISSING` |
| 503 | `GUARDRAIL_INVALID_OUTPUT` |
| 503 | `RETRIEVAL_FAILED` |
| 504 | `LLM_STAGE_TIMEOUT` |
| 504 | `REQUEST_TIMEOUT` |
| 500 | `INTERNAL_ERROR` |

Client responses omit prompts, stack traces, environment values, API keys, and
internal paths. Classified stage timeouts are not relabeled as total timeouts.

Result: **PASSED**.

## 9. Ollama smoke results

| Smoke | Result | Model | Limit | Timeout | Latency |
|---|---|---|---:|---:|---:|
| Answer | PASSED | `qwen2.5:7b-instruct` | 384 | 120 s | 19,004.433 ms |
| Guardrail | PASSED | `phi3:mini` | 8 | 30 s | 9,305.561 ms |

Both outputs were non-empty; guardrail output matched its closed vocabulary.

## 10. Pure retrieval

Query: `What does partial coverage generally cover?`

| Mode | Result | Generation | Total | Key component timings |
|---|---|---|---:|---|
| Without reranker | PASSED | false | 20,970.429 ms | embedding 10,103.391 ms; retrieval 275.110 ms |
| With existing reranker | PASSED | false | 83,450.051 ms | embedding 12,527.040 ms; retrieval 552.679 ms; reranking 24,371.171 ms |

Both returned five source/page records. The larger total includes cold
model/corpus initialization outside individually instrumented operations. No
public diagnostic endpoint or answer-model call was used.

## 11. CRM-only and forbidden live results

CRM-only returned HTTP 200, route `crm-only`, Lara Neumann, and
`TEST-KFZ-2026-1001`. First live wall latency was 331.179 ms; benchmark wall
latency was 167.260 ms and CRM stage latency 131.064 ms. No RAG/LLM generation
ran.

Result: **PASSED**.

The unit regression simulated unavailable Ollama, asserted no RAG call, and
completed below its two-second bound.

The forbidden export returned HTTP 403 with `FORBIDDEN_OPERATION` in 76.3 ms.

Result: **PASSED**.

## 12. Retrieval-only full E2E

Final cold live request:

- HTTP 504, `LLM_STAGE_TIMEOUT`, stage `self_check`;
- wall latency 166,925.922 ms;
- retrieval 464.065 ms;
- reranking 28,975.993 ms;
- self-check 30,034.254 ms, marked `timeout`.

Warm benchmark request:

- HTTP 504 in 47,500.054 ms;
- retrieval 459.620 ms;
- reranking 10,017.893 ms;
- self-check 30,082.280 ms.

The request was bounded and classified, but no complete grounded answer was
produced.

Result: **FAILED**.

## 13. Combined full E2E and partial behavior

Live request:

- HTTP 206, status `partial`, wall latency 47,750.615 ms;
- CRM result present and `knowledgeResult=null`;
- warning `LLM_STAGE_TIMEOUT`, stage `self_check`;
- CRM 162.817 ms;
- retrieval 544.614 ms;
- reranking 10,177.655 ms;
- self-check 30,050.381 ms.

No document coverage or claim decision was fabricated.

Partial contract result: **PASSED**.  
Complete combined E2E result: **FAILED**.

The warm benchmark produced the same outcome in 45,551.799 ms with CRM
260.010 ms, retrieval 514.187 ms, reranking 8,122.954 ms, and self-check
30,046.757 ms.

## 14. Regression tests

Final authoritative command:

```powershell
docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests
```

Final result:

```text
134 passed, 0 failed, 0 skipped, 226 warnings in 131.41s
```

Result: **PASSED**.

Warnings are third-party FastAPI/Starlette, Pydantic, LangChain, and NeMo
deprecations.

Preserved validation attempts:

- unscoped `python -m pytest -q`: **FAILED** during collection because it also
  collected a UTF-16 report and an old executable root `test_ollama.py`; it
  also exposed the undeclared `dataclasses-json` NeMo dependency;
- first scoped `python -m pytest -q tests`: **FAILED** with 131 passed,
  1 skipped, 2 failed; an old timeout assertion and a smoke test accidentally
  invoking live NeMo were corrected;
- focused two-case reproduction: **PASSED**;
- final scoped run with live EspoCRM: **PASSED**.

Final failed count 0: **PASSED**.  
Final blocked count 0: **PASSED**.  
Final incomplete count 0: **PASSED**.  
Final skipped count 0: **PASSED**.

## 15. Data and index integrity

| Check | Before | After | Result |
|---|---:|---:|---|
| Chroma `insurance_rag_collection` | 8,551 | 8,551 | PASSED |
| Tracked PDF changes | 0 | 0 | PASSED |
| Windows/Docker PDF hash portability regression | applicable | passed in suite | PASSED |

No reindex, collection rebuild, PDF replacement, embedding-model change,
retrieval-k change, reranker-parameter change, or guardrail-threshold change
was performed.

## 16. Security validation

- `.env.crm` ignored and untracked: **PASSED**.
- `mva-seed-temp` exists with `isActive=false`, checked through Espo ORM:
  **PASSED**.
- create Contact, update policy, change claim, delete Contact, export Contacts:
  five HTTP 403 results, **PASSED**.
- CRM counts stayed 10/15/8: **PASSED**.
- MCP discovery stayed exactly five read-only tools: **PASSED**.
- exact configured-key scan across source, scripts, tests, docs, reports,
  frontend source, Docker files, and logs found zero files: **PASSED**.

No key value is included in this report.

## 17. Resource observations

| Snapshot | Backend | EspoCRM | MariaDB |
|---|---|---|---|
| Benchmark CPU / RAM | 8.69% / 2.161 GiB | 0.01% / 128.2 MiB | 0.02% / 159.4 MiB |
| Final CPU / RAM | 12.81% / 1.93 GiB | 0.00% / 128.5 MiB | 0.04% / 159.4 MiB |

Host memory: 19.74 GiB total, 6.16 GiB free, 68.8% used. The idle Ollama
server used 27.1 MiB; `ollama ps` showed no resident model. EspoCRM remains
small relative to the local embedding/reranker/LLM path, so these measurements
do not justify replacing it with Baserow.

## 18. Exact meaningful commands executed

Polling/output-format wrappers are omitted:

```powershell
docker compose --env-file .env.crm -f docker/docker-compose.yml -f docker/docker-compose.crm.yml ps
Get-Process -Name ollama -ErrorAction SilentlyContinue
Invoke-RestMethod http://localhost:11434/api/tags
git check-ignore -v .env.crm
git ls-files --error-unmatch .env.crm
docker exec mva-espocrm php -r "<Espo ORM lookup of mva-seed-temp active flag>"
docker exec mva-backend python -c "<read Chroma collection count>"
$env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_NUM_PARALLEL='1'
$env:OLLAMA_KEEP_ALIVE='0'
Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden
docker exec mva-backend python scripts/test_ollama_runtime.py --stage answer
docker exec mva-backend python scripts/test_ollama_runtime.py --stage guardrail
docker compose --env-file .env.crm -f docker/docker-compose.yml -f docker/docker-compose.crm.yml up -d --force-recreate --no-deps backend
Invoke-RestMethod http://localhost:8000/health
docker exec mva-backend python scripts/test_retrieval_pipeline.py --mode without-reranker
docker exec mva-backend python scripts/test_retrieval_pipeline.py --mode with-reranker
Invoke-WebRequest http://localhost:8000/api/ask -Method Post -ContentType application/json -Body "<sanitized scenario JSON>"
.\.venv\Scripts\python.exe scripts\benchmark_llm_runtime.py
docker exec mva-backend python -m pytest -q
docker exec mva-backend python -m pytest -q tests
docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests/integration/test_espocrm_live.py -rs
docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests
docker exec mva-backend python /app/tmp/run_crm_read_only_probe.py
docker exec mva-backend python /app/tmp/scan_secret_leaks.py
docker stats --no-stream --format '{{json .}}' mva-backend mva-espocrm mva-espocrm-db
ollama ps
```

The two `/app/tmp` validation helpers were deleted after use.

## 19. Files changed in this stabilization phase

Created:

- `.env.llm.example`
- `docs/llm_runtime_stabilization.md`
- `reports/llm_runtime_stabilization_plan.md`
- `reports/llm_runtime_stabilization_report.md`
- `reports/llm_runtime_benchmark.json`
- `scripts/benchmark_llm_runtime.py`
- `scripts/test_ollama_runtime.py`
- `scripts/test_retrieval_pipeline.py`
- `src/core/llm_runtime.py`
- `src/core/ollama_readiness.py`
- `src/core/runtime_diagnostics.py`
- `tests/unit/test_llm_runtime_stabilization.py`

Updated:

- `docs/espocrm_mcp_integration.md`
- `frontend/src/api.ts`
- `requirements.txt`
- `reports/espocrm_mcp_integration_report.md`
- `scripts/evaluation/calculate_metrics.py`
- `scripts/evaluation/evaluate_extended.py`
- `src/api/rag_service.py`
- `src/config/models.py`
- `src/core/safety_adapter.py`
- `src/evaluation/legacy_metrics.py`
- `src/guardrails/integrations/nemo_official.py`
- `src/main.py`
- `tests/integration/test_api_boot.py`
- `tests/integration/test_rag_smoke.py`
- `tests/unit/test_guardrails_runtime.py`
- `tests/unit/test_self_check_decision.py`

Generated sanitized evidence:

- `reports/.llm_smoke_answer.json`
- `reports/.llm_smoke_guardrail.json`
- `reports/.llm_pure_retrieval_without.json`
- `reports/.llm_pure_retrieval_with.json`
- `tmp/ollama_stabilization_20260726_160439.stdout.log`
- `tmp/ollama_stabilization_20260726_160439.stderr.log`

The two Ollama logs remain untracked because the running server holds them
open. The secret scan covered them and found no configured key. Unrelated
pre-existing dirty-worktree changes were preserved.

## 20. Remaining limitations

1. On this CPU-only machine, unchanged `phi3:mini` self-check does not finish
   within its 30-second budget for the tested retrieval requests.
2. Complete retrieval-only and combined E2E remain **FAILED**.
3. Pure retrieval with the existing cross-encoder has high cold-start cost.
4. Synchronous Python work cannot be forcibly killed; bounded gates prevent
   accumulation but a timed-out worker holds its gate until it returns.
5. Third-party deprecation warnings remain.

## 21. Rollback

The worktree contained pre-existing changes, so do not use a broad reset or
checkout. Roll back only the phase-specific hunks/files listed above:

1. restore updated files from the known baseline or revert a future dedicated
   stabilization commit;
2. remove only newly created stabilization files;
3. remove non-secret LLM variables from the ignored local environment if
   copied there;
4. recreate only `mva-backend`;
5. do not touch EspoCRM, MariaDB, PDFs, Chroma, volumes, CRM records, or model
   files.

## 22. Recommended next action

Keep EspoCRM and the stabilized bounds. Profile or optimize the existing
`phi3:mini` self-check under thesis-approved conditions, then rerun live steps
7 through 19. If model selection must remain unchanged, compare the same
configuration on a host with more CPU/GPU capacity. Do not label the complete
pipeline PASSED until retrieval-only and combined both return HTTP 200
complete.

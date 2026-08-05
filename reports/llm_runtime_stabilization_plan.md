# LLM Runtime Stabilization Plan

Inspection date: 2026-07-26  
Repository: `MVA_Versicherung_Langchain_main`

This report records the required pre-implementation inspection. No runtime
stabilization code was changed before this report was written.

## Current runtime and model mapping

The effective configuration was read through the repository's configuration
loader inside the existing backend container. No secrets or complete
environment dump were read.

| Stage | Current implementation | Effective model | Call type | Current output limit | Current timeout |
|---|---|---|---|---|---|
| Lexical route planning | `plan_insurance_query` | none | synchronous rules | not applicable | none required |
| RAG router | `decide_retrieval` | `phi3:mini` | synchronous `invoke` | none | none |
| NeMo input/context/output guardrail | `OfficialNemoGuardrailsRuntime._generate` | currently constructed with answer role, `qwen2.5:7b-instruct` | async NeMo call run in a daemon thread, caller blocks on `join` | none | 20 s wait only |
| Query rewrite | `rewrite_query` | `phi3:mini` | synchronous `invoke` | none | none |
| Self-check | `perform_self_check` | `phi3:mini` | synchronous `invoke` | none | none |
| Context compression | `compress_context` | `phi3:mini` | synchronous `invoke`; disabled in effective config | none | none |
| Reranker | `FlagReranker.compute_score` | `BAAI/bge-reranker-base` | synchronous local cross-encoder, not a generative LLM | not applicable | none |
| Final/direct answer | `generate_answer` / `generate_direct_answer` | `qwen2.5:7b-instruct` | synchronous `invoke` | none | none |
| Empty-answer fallback | second call in `generate_answer` | `qwen2.5:7b-instruct` | synchronous `invoke` | none | none |

The effective embedding model is `BAAI/bge-m3` on CPU. Retrieval is forced,
query rewriting is currently disabled, and `MAX_SELF_CHECK_RETRIES` is 2.
These model, embedding, retrieval-k, reranker, and guardrail-threshold settings
will not be changed by this task.

The prior observation of a Phi-based guardrail belongs to the recorded failure
baseline. The current `nemo_official.py` implementation actually creates the
NeMo main LLM from the answer role. The stabilization will make the guardrail
role explicit and environment-backed without changing an explicitly configured
model.

## Ollama option inspection

The installed backend package is `langchain-ollama 1.1.0`.
`ChatOllama.model_fields` contains both `num_predict` and `client_kwargs`, but
does not contain a top-level `timeout` field.

Consequences:

- `num_predict` is the supported output-token limit and will be used.
- The existing top-level `timeout` keyword can be silently ignored by the
  installed Pydantic model.
- HTTP timeouts must be passed through supported Ollama client kwargs.
- The existing global `LLM_MAX_TOKENS` and `LLM_TIMEOUT_SECONDS` defaults are
  `None`, so every effective production generation is currently unbounded by
  repository configuration.

## `/api/ask` execution paths

Route planning is deterministic and represented by `QueryMode`:
`retrieval_only`, `crm_only`, `combined`, or `denied`.

- `crm_only` calls only the five bounded read-only MCP tools. It does not call
  RAG or Ollama.
- `retrieval_only` calls `rag_service.run_rag` in `asyncio.to_thread`.
- `combined` starts CRM and RAG concurrently with `asyncio.gather`.
- `denied` returns HTTP 403 before CRM or LLM work.
- The FAQ shortcut has no LLM call.

The combined route currently fails as one unit: an exception in RAG discards
the already valid CRM result. The current `AnswerResponse` contains only
`answer`, `sources`, and `latencyMs`. The frontend reads those fields and will
ignore additive optional fields, so a backward-compatible partial schema is
possible.

## Existing timeout and retry behavior

- EspoCRM REST has bounded HTTP timeouts and one bounded retry; it is outside
  the LLM stabilization scope and will be preserved.
- `/api/ask` has no total request timeout.
- Direct LangChain/Ollama calls have no effective HTTP timeout.
- NeMo waits at most `NEMO_RUNTIME_TIMEOUT_SECONDS`, currently 20 seconds, but
  its daemon thread continues after timeout. Repeated requests can therefore
  accumulate background work.
- Retrieval and reranking have no stage timeout.
- The self-check loop can run twice. Each iteration can retrieve, rerank,
  self-check, and optionally rewrite.
- Final answer generation retries once only when the first model response is
  empty. It does not distinguish timeout, transport failure, or missing model.
- No LLM transport retry policy is explicit.

## Existing error mapping

- Empty input returns HTTP 400.
- Denied CRM scope returns HTTP 403.
- `ValueError` and `RuntimeError` are broadly mapped to HTTP 503 with raw
  exception text.
- Expected Ollama unavailability, missing models, stage timeouts, retrieval
  failures, and invalid classifier output have no stable error codes.
- Other errors become a generic HTTP 500 unless debug mode exposes raw details.
- Combined LLM failure currently reaches the generic failure path.

## Existing latency and diagnostics

- `AnswerResult` records only total RAG latency.
- CRM-only and combined responses measure only total route latency.
- Retrieval service logs coarse corpus initialization, retrieval, and reranking
  durations.
- Audit entries contain total latency and self-check retry count, but no
  request ID or complete stage breakdown.
- The EspoCRM benchmark measures direct REST, MCP, and whole E2E call latency.
  It uses `/api/ask` for CRM-only, retrieval-only, and combined scenarios, and
  does not expose per-stage timing.
- The benchmark marks failed HTTP calls or client timeouts as failed, but has
  no stable runtime error-code breakdown.

## Existing pure retrieval path

`RetrievalService.retrieve_and_rerank` and
`retrieve_documents_for_tool` already provide a service/MCP path without final
answer generation. It currently always uses the configured reranker and
reports only total tool duration. It can be extended safely with:

- a clearly named no-reranker diagnostic mode;
- component timings for lexical, vector, hybrid merge, and reranking;
- source metadata and available scores;
- no public diagnostic endpoint;
- no Chroma writes or reindex.

## Stabilization design

1. Add central, environment-backed stage policies for output limits, timeouts,
   and retry counts. Defaults will follow the requested engineering values.
   The non-generative reranker will have a timeout but no fake token option.
2. Build every `ChatOllama` instance with an explicit stage and prove
   `num_predict` plus supported client timeout propagation in unit tests.
3. Parse router and self-check results as exact, normalized closed-vocabulary
   labels. Verbose or otherwise invalid classifier output will fail closed with
   a structured error.
4. Introduce stable runtime exceptions and safe API error responses:
   `LLM_UNAVAILABLE`, `LLM_MODEL_MISSING`, `LLM_STAGE_TIMEOUT`,
   `REQUEST_TIMEOUT`, `GUARDRAIL_INVALID_OUTPUT`, `RETRIEVAL_FAILED`, and
   `INTERNAL_ERROR`.
5. Add a bounded RAG execution coordinator. Client transport timeouts will stop
   the Ollama HTTP wait. A bounded worker/gate will prevent abandoned
   synchronous work from accumulating if true cancellation is unavailable.
   CRM-only will bypass that coordinator entirely.
6. Cap long LLM retry cycles at one configured retry. A stage timeout will not
   trigger self-check/rewrite cycles. Retrieval irrelevance will remain
   distinct from transport failure.
7. Add request-scoped JSON-serializable diagnostics with a request ID, planned
   route, applicable stage timings, timeout stage, retries, result status, and
   partial flag. Full prompts, secrets, and unnecessary customer payloads will
   not be logged.
8. Extend `/health` with a lightweight Ollama `/api/tags` readiness check and
   model availability flags while preserving all existing CRM and pipeline
   fields.
9. Add `scripts/test_ollama_runtime.py` and a service-level pure retrieval
   diagnostic script. Neither will print secrets.
10. Isolate combined CRM and RAG results. When CRM succeeds and RAG has an
    expected availability/timeout failure, return a clearly marked additive
    partial representation with HTTP 206, CRM sources only, no fabricated
    document answer, and no coverage/claim inference.
11. Add focused unit/integration tests before live validation. Live tests will
    follow the required order and will not run full LLM E2E scenarios if either
    minimal Ollama smoke test fails.

## Constraints retained

The implementation and validation will not modify the PDF corpus, rebuild or
delete Chroma, force reindexing, modify CRM records or permissions, reactivate
the seed user, change selected Ollama/embedding/reranker models, change
retrieval k-values, change guardrail thresholds, disable guardrails, or expose
credentials.

# Combined Noah Theft Completeness Test

Generated: 2026-08-05T11:09:46+02:00

## 1. Executive Summary

The query-scoped completeness mechanism was implemented and all 134 targeted unit and regression tests passed. The required live E2E test did not enter the pipeline: the exactly one permitted `POST /api/ask` used the JSON field `query`, while the API schema requires `question`. FastAPI rejected the request with HTTP 400 before a request diagnostic, route selection, CRM lookup, retrieval, generation, completeness, groundedness, citations, or safety execution could start. No second client request was sent. The final live-test status is **FAIL**.

## 2. Root Cause of the Previous FAIL

Historical diagnostic evidence: `tmp/diagnostics/request_9f7b2c37e6264f3d98fcfa0628dcab69.json`.

- `motor-vehicle-insurance-sti.pdf`, page 17, chunk `768239c2-f3ef-4806-9b99-41ed105144d3` was present in the final generation context.
- The chunk contained immediate police reporting and immediate insurer notification when the vehicle is recovered or its location becomes known.
- `rawOpenAIAnswer` already omitted both duties.
- `afterCompletenessPostprocessing`, `afterCitationPostprocessing`, `immediatelyBeforeGroundedness`, and `finalAfterOutputSafety` were materially identical; later processing did not remove the duties.
- The omission therefore occurred in the first model generation in `generate_answer` (`src/api/rag_service.py:2930`) after context formatting in `_format_context_with_sources` (`src/api/rag_service.py:500`) and the generation prompt in `build_generation_chain` (`src/api/rag_service.py:2504`).
- Groundedness v5 scored the claims that were present (`0.856132`) and correctly did not detect absent claims. Completeness is therefore a separate control.

## 3. Changed Files and Functions

- `src/core/answer_completeness.py`: new `AnswerRequirement`, `_theft_requirements`, `_crm_requirements`, `build_answer_requirements`, `evaluate_answer_completeness`, and formatting helpers.
- `src/api/rag_service.py`: updated `build_generation_chain`; added `build_completeness_regeneration_chain`; integrated requirements, deterministic checking, diagnostics, and one bounded full regeneration in `generate_answer`.
- `src/main.py`: added `_answer_completeness_warning`; integrated conservative HTTP 206 partial handling for incomplete Combined and retrieval-only answers.
- `tests/unit/test_answer_completeness.py`: nine focused completeness tests.
- `tests/unit/test_llm_runtime_stabilization.py`: HTTP 206 partial-response integration test.
- `tests/integration/test_internal_caseworker_pipeline.py`: realistic multiline fixture formatting for the existing combined control regression.

## 4. New Completeness Mechanism

Before answer generation, the mechanism classifies a single supported event from the query and selected final context. For theft it derives only requirements backed by structured theft/product sections in the selected PDF chunks, plus exact CRM fields explicitly requested by the query. After generation, a deterministic paraphrase-aware check requires both the concept and its source label. If anything is missing, the model receives the original query, full final context, all requirements, the concrete missing requirements, and the first draft, then rewrites the complete response once. The gate never appends source text.

Diagnostics include `completenessRequiredItems`, `completenessPresentItems`, `completenessMissingBeforeRetry`, `completenessRetryPerformed`, `completenessMissingAfterRetry`, `completenessPass`, and source file/page/chunk metadata per item. `answerGeneration.applicationCallCount` records one or two model calls.

## 5. Separation from Groundedness

`fact_aware_claim_support_v5` remains unchanged and checks whether claims present in the answer are supported. Completeness separately checks whether all query-specific, selected-context requirements are present. The calibrated threshold remains `0.7888`.

## 6. Separation from the Removed Postprocessor

The implementation does not restore `_ensure_material_qualifiers`, does not append extracted sentences, does not create a separate extracted-conditions block, and does not use `MATERIAL QUALIFIERS PRESENT`. A production-source scan found none of those legacy markers. Fire and natural-forces clauses are not classified as theft requirements.

## 7. Unit and Regression Results

Final command covered completeness, retrieval/reranking, route selection, Groundedness v5 and calibration, diagnostics, citation controls, coverage consistency, safety and PII, Self-Check flag behavior, CRM API routing, and internal Combined integration.

- Final result: **134 passed, 0 failed**, 22 dependency deprecation warnings.
- Runtime: 67.74 seconds.
- Focused completeness tests: 9 passed.
- Focused HTTP 206 partial test: passed.
- Cases A-I and page-17 citation behavior were covered directly; existing Combined, windscreen/coverage, CRM, Groundedness, citation, and safety regressions were covered by the broader suite.

## 8. Backend and Configuration Status

The backend was fully restarted with `docker restart mva-backend`. Cold-start readiness completed after 274.4 seconds.

- `pipelineReady=true`
- `crmReady=true`
- `embeddingReady=true`
- `retrievalReady=true`
- `pipelineInitError=null`
- `CRM_ENABLED=true`
- `SELF_CHECK_ENABLED=false`
- `PIPELINE_AUTHENTICATED=false`
- Groundedness algorithm: `fact_aware_claim_support_v5`
- Groundedness threshold: `0.7888` from calibration file
- Safety: enabled, mode `enforce`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Answer provider/model: OpenAI / `gpt-4o-mini`
- `pdfs_have_changed=false`
- `embedding_model_has_changed=false`
- Chroma collection: `insurance_rag_collection`; no reindex, collection mutation, or embedding recomputation was performed.

## 9. Complete Query

```text
Noah Weber reports the theft of his car. Is this loss generally covered by his current motor insurance, and which documented conditions apply?

Please provide the relevant active policy number, coverage type, individual deductible and annual premium. Clearly distinguish the general insurance terms from Noah's individual contract data, and do not make a final claim decision.
```

## 10. HTTP, API Status, and Route

- Client requests sent to `POST /api/ask`: exactly 1
- HTTP status: `400`
- API status: validation error; request rejected before `ask()`
- Route: not selected
- Request ID: not created
- Client-observed latency: 151 ms
- Server access-log evidence: `POST /api/ask HTTP/1.1` -> `400 Bad Request`
- Schema evidence: `AskQuestionRequest.question` at `src/main.py:301`; endpoint at `src/main.py:413`.

## 11. Selected Policy and CRM Facts

Not executed for this live request. No CRM read occurred after request validation, so no live policy or CRM facts are reported.

## 12. Final Retrieval Chunks

Not executed for this live request. No Chroma retrieval or reranking occurred, so there are no live ranks, pages, chunk IDs, or scores.

## 13. Required Items Before Generation

Not produced for this live request because generation was never reached. The tested theft mechanism is designed to require: partially comprehensive theft coverage; theft/misappropriation/robbery loss scope; involuntary damage; family exclusion; immediate police reporting; recovery/known-location insurer notification; requested policy number; coverage type; deductible; and annual premium. The abroad condition is omitted unless the query explicitly describes an abroad scenario.

## 14. Raw First OpenAI Answer

Not available. OpenAI answer generation was not called.

## 15. Missing Items Before Regeneration

Not available. Completeness evaluation was not called.

## 16. Regeneration Performed

No. The pipeline did not start.

## 17. Regenerated Answer

Not available.

## 18. Missing Items After Regeneration

Not available.

## 19. Final Answer

No insurance answer was generated. The request was rejected at schema validation.

## 20. Inline Citations

None. Citation processing was not executed.

## 21. Groundedness

- Algorithm configured: `fact_aware_claim_support_v5`
- Threshold configured: `0.7888`
- Live score: not available
- Live result: not executed

## 22. Completeness Result

- `completenessPass`: not evaluated
- Required items: not instantiated in the live request
- Missing before retry: not evaluated
- Retry performed: no
- Missing after retry: not evaluated

## 23. Safety Result

Safety is configured as enabled and `enforce`, but it was not executed for the rejected live request. No Safety fallback was generated.

## 24. Latencies

- Client request: 151 ms
- CRM: not executed
- Retrieval: not executed
- Reranking: not executed
- Generation: not executed
- Completeness: not executed
- Groundedness: not executed
- Safety: not executed

## 25. OpenAI Answer Calls

Zero. The validator rejected the request before any model call.

## 26. Final Status

**FAIL**. The code and regression suite pass, but the mandatory live E2E criteria are not met because the one allowed request returned HTTP 400 and did not execute route `combined`.

## 27. Exact Remaining Failure Cause

The client serialized the exact query under JSON key `query`. `AskQuestionRequest` requires key `question`; therefore FastAPI request validation rejected the POST with HTTP 400 before entering the endpoint body and before `RequestDiagnostics` was constructed. This is a client-payload schema error, not a CRM, retrieval, reranking, OpenAI, completeness, Groundedness, citation, or safety failure. No second request was sent because the instruction permitted exactly one client request.


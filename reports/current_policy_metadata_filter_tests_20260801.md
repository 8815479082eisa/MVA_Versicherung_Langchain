# Current-policy metadata filter tests — 2026-08-01

## Outcome

**PASS.** The deterministic selector, existing CRM selection, CRM-only API path, and combined CRM+RAG context path pass the targeted test set. The reference date in deterministic tests is explicitly fixed to `2026-08-01` (or `2026-07-29` in the pre-existing combined-context test).

## Requirement coverage

| Case | Assertion | Result |
| --- | --- | --- |
| A — current policy | Lara motor records `1001` and `1003`; latest active effective record is `TEST-KFZ-2026-1003` | PASS |
| B — explicit old policy | explicit `TEST-KFZ-2026-1001` does not activate the current filter; existing exact-number CRM rank selects it | PASS |
| C — historical | “did ... have previously” preserves old and new candidates | PASS |
| D — comparison | previous/current comparison preserves both `1001` and `1003` | PASS |
| E — non-current/list | non-temporal query remains unchanged; list query preserves all policies | PASS |
| F — incomplete/conflicting metadata | missing dates are ineligible; tied latest start dates return no selected policy and an ambiguity reason | PASS |
| Other customer | same rules select a different customer's current motor policy without Lara-specific code | PASS |
| Combined runtime | windscreen combined route passes only current motor policy `1003` into CRM context | PASS |
| CRM-only runtime | dated current-policy query returns `1003`, excludes `1001`, and does not call RAG | PASS |

## Commands and exit codes

### Focused unit validation during implementation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_current_policy.py tests/unit/test_crm_orchestration.py -q
```

Exit code: `0` — `12 passed` at that revision.

### Final relevant suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_current_policy.py tests/unit/test_crm_orchestration.py tests/integration/test_crm_api_routing.py -q
```

Exit code: `0` — `22 passed, 22 warnings`.

Warnings are pre-existing Pydantic v2 deprecations from NeMo Guardrails and LangChain Community. No test failure or current-policy warning occurred.

### Syntax validation

```powershell
.\.venv\Scripts\python.exe -m py_compile src/main.py src/core/current_policy.py src/core/crm_orchestration.py scripts/benchmark_rerankers_isolated.py scripts/benchmark_rerankers_post_filter.py scripts/generate_reranker_post_filter_cases.py
```

Exit code: `0`.

### Fixture generation and activation audit

```powershell
.\.venv\Scripts\python.exe scripts/generate_reranker_post_filter_cases.py
```

Exit code: `0` — wrote 64 cases. A separate audit showed exactly three activations (`1003_en`, `1003_de`, `1003_mixed`) and zero failed selections; the other 61 cases remained unchanged.

### Post-filter benchmark

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_rerankers_post_filter.py --torch-threads 8
```

Exit code: `0`. The mMARCO snapshot download took 248.71 seconds and was excluded from load/inference timing. Hugging Face warned that `hf_xet` was absent and used ordinary HTTP; this affected only the untimed download.

### Existing project reranker factory smoke test

The cached MiniLM-L6 snapshot was passed to `src.api.rag_service.build_reranker`, then two pairs were scored.

Exit code: `0`; adapter type `BaseReranker`; load `5.204s`; two finite scores returned. This confirms the currently installed project runtime can load MiniLM without dependency changes. The benchmark itself used the required `sentence_transformers.CrossEncoder` adapter.

## Changed files and purpose

| File | Purpose / impact |
| --- | --- |
| `src/core/current_policy.py` | new pure intent guard and fail-safe metadata selector |
| `src/core/crm_orchestration.py` | invokes the selector for current intent; preserves history/list/comparison candidates |
| `src/main.py` | applies the same CRM selection to CRM-only responses; combined route was already using it |
| `tests/unit/test_current_policy.py` | A–F, list, and other-customer unit coverage |
| `tests/integration/test_crm_api_routing.py` | CRM-only current-policy integration coverage |
| `scripts/generate_reranker_post_filter_cases.py` | reproducible metadata-enriched copy of the unchanged 64-case fixture |
| `tests/fixtures/reranker_post_filter_cases.json` | generated fixture using only existing CRM CSV fields |
| `scripts/benchmark_rerankers_post_filter.py` | one-load-per-model raw/post-filter CPU benchmark and reports |
| `scripts/benchmark_rerankers_isolated.py` | registers mMARCO as an allowed selectable benchmark model; old reports were not rerun or overwritten |
| dated reports/CSV | analysis, test evidence, comparison, and per-case details |

## Restrictions verified

- No reindex, embedding generation, Chroma/BM25 retrieval, LLM, self-check, or CRM network call occurred in the benchmark.
- No collection or CRM data was modified.
- Existing baseline fixtures and reports were not overwritten.
- `.env` still contains `RERANKER_MODEL=BAAI/bge-reranker-base` and `RERANKER_USE_FP16=false`.

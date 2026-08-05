# CRM motor policy / active PDF alignment

Date: 2026-08-04

## Active PDF evidence

The active corpus contains Helvetia PDFs only. The motor evidence is:

- `data/raw/pdfs/motor-vehicle-insurance-sti.pdf`: `STI Helvetia Motor Vehicle Insurance - Edition March 2026`, contracting party `Helvetia Swiss Insurance Company Ltd`, document code `120016971 07.26`.
- `data/raw/pdfs/motor-vehicle-insurance-product-sheet.pdf`: `Helvetia car insurance`, document code `120017607 01.26`.
- Product / coverage terms evidenced by the PDFs:
  - `Third-Party Liability`
  - `Partially comprehensive cover`
  - `Fully comprehensive`
  - `Fully comprehensive PLUS`
  - `Glass` / `Glass breakage`
- Relevant Lara windscreen rules:
  - `motor-vehicle-insurance-product-sheet.pdf`, page 3: partially comprehensive insurance includes `Glass breakage`.
  - `motor-vehicle-insurance-sti.pdf`, page 14: `K2.1.5 Glass` covers breakage and accident-related glass damage to front and rear windscreens, side windows and sunroof where repair or replacement is necessary for safety reasons.
  - `motor-vehicle-insurance-sti.pdf`, page 9: `G10.2 You will not have to bear a deductible`, including the rule for a damaged front windscreen repaired and not replaced in the case of glass damage.

No Baloise PDF is active. The previous Baloise PDFs are under `data/raw/excluded_pdfs`.

## CRM backup

Before any CRM data change, the six affected motor policies were exported through the EspoCRM API:

- `tmp/crm_backups/mva_motor_policies_before_20260804T082536Z.json`

The normal runtime API key was read-only and returned `403 Forbidden` for updates. Data changes were then applied through EspoCRM REST API admin authentication, not through MariaDB.

## Before / after

EspoCRM `MvaPolicy.productType` and `MvaPolicy.coverageType` are enum fields. The exact PDF labels `Helvetia Motor Vehicle Insurance`, `Partially comprehensive cover`, `Fully comprehensive`, and `Third-Party Liability` were rejected by EspoCRM validation for those enum fields. Therefore only the free `name` field was changed to carry the evidenced provider, product, coverage label and STI version. Product domain enums, status, dates, deductible and annual premium were left unchanged.

| Policy | Feld | Vorher | Nachher | PDF-Beleg |
| --- | --- | --- | --- | --- |
| TEST-KFZ-2026-1001 | name | Lara Neumann Motor 2026 | Lara Neumann Helvetia Motor Vehicle Insurance Partially comprehensive cover - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 3 partially comprehensive / glass breakage |
| TEST-KFZ-2026-1003 | name | Lara Neumann BaloiseDirect Motor Vehicle 2026 | Lara Neumann Helvetia Motor Vehicle Insurance Partially comprehensive cover - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 3 partially comprehensive / glass breakage |
| TEST-KFZ-2026-1101 | name | Oliver Brandt Motor 2026 | Oliver Brandt Helvetia Motor Vehicle Insurance Fully comprehensive - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 3 fully comprehensive |
| TEST-KFZ-2026-1401 | name | Hannah Vogel Motor 2026 | Hannah Vogel Helvetia Motor Vehicle Insurance Third-Party Liability - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 2 liability insurance |
| TEST-KFZ-2026-1701 | name | Noah Weber Motor 2026 | Noah Weber Helvetia Motor Vehicle Insurance Partially comprehensive cover - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 3 partially comprehensive / glass breakage |
| TEST-KFZ-2026-1901 | name | Leon Becker Motor 2026 | Leon Becker Helvetia Motor Vehicle Insurance Fully comprehensive - STI Edition March 2026 | STI cover page Edition March 2026; product sheet page 3 fully comprehensive |

Unchanged because not individually evidenced by PDF terms: `deductible`, `annualPremium`, `currency`, `status`, `startDate`, `endDate`.

Update log:

- `tmp/crm_backups/mva_motor_policies_name_update_20260804T082837Z.json`

## Updated files

- `data/synthetic/crm/policies.csv`
- `tests/unit/test_crm_synthetic_data.py`
- `tests/unit/test_crm_orchestration.py`
- `tests/unit/test_insurance_tool_routing.py`
- `tests/unit/test_retrieval_service.py`
- `tests/unit/test_experimental_groundedness_v5.py`
- `tests/unit/test_self_check_feature_flag.py`
- `tests/integration/test_crm_api_routing.py`
- `tests/integration/test_internal_caseworker_pipeline.py`
- `tests/fixtures/reranker_regression_cases.json`
- `tests/fixtures/reranker_post_filter_cases.json`
- `src/api/rag_service.py`

## Tests

Before the live request:

```text
.venv\Scripts\python.exe -m pytest tests/unit/test_crm_synthetic_data.py tests/unit/test_crm_orchestration.py tests/unit/test_current_policy.py tests/unit/test_insurance_tool_routing.py tests/unit/test_retrieval_service.py tests/unit/test_experimental_groundedness_v5.py tests/unit/test_self_check_feature_flag.py tests/integration/test_crm_api_routing.py tests/integration/test_internal_caseworker_pipeline.py -q
85 passed, 22 warnings
```

After the live request exposed a final-reranking issue, `src/api/rag_service.py` was minimally adjusted so query-relevant insurance evidence is prioritized before generic same-product neighbors. This was verified without another live request:

```text
.venv\Scripts\python.exe -m pytest tests/unit/test_retrieval_service.py tests/unit/test_crm_synthetic_data.py tests/unit/test_crm_orchestration.py tests/unit/test_insurance_tool_routing.py tests/unit/test_experimental_groundedness_v5.py tests/unit/test_self_check_feature_flag.py tests/integration/test_crm_api_routing.py tests/integration/test_internal_caseworker_pipeline.py -q
78 passed, 22 warnings
```

## Live Combined test

Exactly one live Combined request was sent:

- Diagnostic file: `tmp/diagnostics/combined_selfcheck_20260804T083517Z.json`
- Server diagnostic file: `tmp/diagnostics/request_96ba3bbd6b1247108b437759cfdf1b42.json`
- Request ID: `96ba3bbd6b1247108b437759cfdf1b42`
- HTTP status: `200`
- Route: `combined`
- `CRM_ENABLED=true`
- `SELF_CHECK_ENABLED=false`
- `PIPELINE_AUTHENTICATED=false`

Live result:

- Selected policy: `TEST-KFZ-2026-1003`
- CRM context product/version: `Lara Neumann Helvetia Motor Vehicle Insurance Partially comprehensive cover - STI Edition March 2026`
- CRM values: `Partial Coverage`, deductible `300 EUR`, annual premium `720 EUR`
- PDF chunks retrieved from active Helvetia corpus: yes
- Final PDF context included `motor-vehicle-insurance-sti.pdf` pages 9, 17, 18 and 26.
- Answer generation executed: yes
- Groundedness algorithm: `fact_aware_claim_support_v5`
- Groundedness score: `0.35`
- Threshold: `0.7888`
- PDF citations: none in final response
- Safety decision: `post_fallback`
- Final response: `I could not generate an answer that is sufficiently supported by the available documents.`

Final live status: FAIL.

The failure is no longer CRM/product drift. The live diagnostics show the selected CRM policy now matches the active Helvetia product/version, but the final PDF context still over-prioritized generic neighbor pages and missed the necessary answer-bearing glass coverage page in the final top-5. The follow-up Reranker fix is test-covered but was not live-retested because the requested live limit was exactly one Combined request.


# Current-policy metadata filter analysis — 2026-08-01

## Result

The Lara failure is a structured temporal-selection problem, not a cross-encoder score problem. The minimal safe insertion point is the existing CRM context selection before CRM facts are passed either directly to a CRM-only response or into combined document retrieval/reranking.

## Existing flow

1. `src/core/insurance_tool_routing.py` creates a bounded `QueryPlan` and extracts customer, policy and claim identifiers.
2. `src/core/crm_orchestration.py::execute_crm_query` resolves the customer and requests that customer's policies through `get_customer_policies`.
3. `src/integrations/espocrm_client.py::get_customer_policies` reads `MvaPolicy` records, ordered by `startDate desc`, and normalizes them.
4. `src/main.py` uses `select_crm_context` for combined CRM+RAG requests. Its selected policy supplies CRM context and product/name retrieval hints before document retrieval.
5. Document retrieval is hybrid (vector plus BM25). The configured pre-rerank count is `RETRIEVE_TOP_K=8`; `rerank_documents` then scores text pairs and keeps the configured top results.
6. Before this change, CRM-only requests returned all formatted CRM facts without calling `select_crm_context`. That bypass was closed so the same policy-selection semantics now apply to both CRM-only and combined routes.

## Policy metadata actually present

The EspoCRM `MvaPolicy` schema and synthetic CSV provide these relevant fields:

| EspoCRM/CSV | Normalized runtime field | Use |
| --- | --- | --- |
| `policyNumber` | `policy_number` | stable business identifier |
| `productType` | `product_type` | product/domain restriction |
| `coverageType` | `coverage_type` | evidence/output, not temporal selection |
| `status` | `status` | requires `Active`/`Aktiv` for current selection |
| `startDate` | `start_date` | inclusive effective start and latest-start selection |
| `endDate` | `end_date` | inclusive effective end |
| Contact relation / scoped API call | customer scope | unique customer restriction |
| `name` | `name` | retrieval hint only |

The project does **not** define `validFrom`, `validTo`, `effectiveDate`, `isActive`, `supersededBy`, or `replacedBy`. No such fields or replacement rules were invented. `startDate`, `endDate`, and `status` are the available substitutes.

Long term, an explicit replacement relation (`replacedBy`/`supersedes`) or a version/issuance timestamp would be the minimal useful schema extension. Without it, two active policies for the same customer and product with the same latest `startDate` are structurally ambiguous and must not be resolved by policy-number ordering.

## Implemented decision

`src/core/current_policy.py` is a pure, reusable selector:

1. Activate only for an unambiguous single-policy current/latest/valid/present intent.
2. Do not activate for an explicit policy number, historical wording, comparison wording, or list/all wording.
3. Restrict to structured policy records.
4. Resolve a customer from candidate metadata when candidates contain multiple customers; otherwise rely on the already customer-scoped CRM response.
5. Resolve the requested product domain. If it is absent and several products remain, fail safe.
6. Require active status plus complete parseable start/end dates covering the explicit reference date.
7. Choose the sole policy with the latest effective start date.
8. If no valid candidate exists, metadata are missing, or the latest start is tied, return no selected policy and a diagnostic reason. No score bonus or customer/policy-specific exception is used.

For historical, comparison and list requests, `select_crm_context` preserves all policy records. Explicit policy-number requests continue through the existing deterministic exact-number ranking.

## Scope and side effects

- Current single-policy requests can no longer leak an overlapping older active contract into CRM context.
- Explicit old-policy requests remain unchanged.
- Historical, comparison and list requests retain old and new records.
- Ordinary non-current single-policy requests keep the existing ranking behavior.
- Missing or contradictory metadata produce no invented current policy.
- No PDF, collection, embedding, CRM row, answer-LLM, self-check, or productive reranker setting was changed.

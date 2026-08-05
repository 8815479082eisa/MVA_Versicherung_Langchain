# Combined CRM + RAG OpenAI Live Test

Test date: 2026-07-27  
Answer provider: OpenAI  
Answer model: `gpt-4o-mini`  
Overall result: **PARTIAL PASS - OpenAI works, final document answer is blocked by retrieval/groundedness**

## Environment

The OpenAI API key is stored in the local Git-ignored `.env` file and was not
printed in this report. It remains configured after the test.

The restarted production backend reported:

| Check | Result |
|---|---|
| OpenAI key configured | true |
| Answer model ready | true |
| CRM ready | true |
| Retrieval ready | true |
| Local guardrail ready | true |
| Overall LLM ready | true |
| Pipeline ready | true |

## Primary combined question

> Lara Neumann has an active BaloiseDirect Motor Vehicle policy. Based on the
> CRM policy data and only the 2018 BaloiseDirect Motor Vehicle document
> 240_1184_e.pdf, are marten bites included in her selected coverage? State the
> relevant policy number, coverage type, and individual deductible from CRM,
> then cite the document evidence separately. Do not treat the deductible as a
> coverage limit.

The deterministic router selected:

- route: `combined`;
- customer: Lara Neumann;
- CRM policies: required;
- CRM claims: not required;
- RAG retrieval: required.

## Run 1 - production configuration

The first live request used the normal production configuration with
`SELF_CHECK_ENABLED=true`.

| Metric | Result |
|---|---:|
| HTTP status | 206 |
| Response status | partial |
| Route | combined |
| Wall latency | 67,815.755 ms |
| CRM | completed in 232.776 ms |
| Retrieval | completed in 1,493.266 ms |
| Reranking | completed in 29,438.821 ms |
| Self-check | timeout after 30,044.859 ms |
| OpenAI answer stage | not reached |

The CRM result was correct, but the document branch returned
`LLM_STAGE_TIMEOUT` at `self_check`.

## Run 2 - controlled answer-generation test

To isolate the online answer stage without changing the production
configuration, a temporary backend was started internally on port 8001 with:

- `SELF_CHECK_ENABLED=false`;
- `INIT_PIPELINE_ON_STARTUP=false`.

The temporary process was verified before use and terminated after the test.
The main backend and `.env` were not changed.

| Metric | Result |
|---|---:|
| HTTP status | 200 |
| Response status | complete |
| Route | combined |
| Cold wall latency | 126,040.921 ms |
| CRM | completed in 543.855 ms |
| Embedding initialization | 26,971.482 ms |
| Retrieval | 3,609.839 ms |
| Reranking | 34,318.462 ms |
| OpenAI answer generation | **4,961.653 ms** |
| Answer stage | completed |
| Authentication error | none |

This run proves that the saved key is accepted and that the real
`gpt-4o-mini` Responses API path works.

The document answer was nevertheless replaced by the safe fallback:

> I could not generate an answer that is sufficiently supported by the
> available documents.

The output guardrail measured groundedness at `0.11375`, below the enforced
threshold of `0.51`.

Audit request ID: `d5086d27c989437e8f3d798a50e948c0`

## Diagnostic query variants

### Glass-breakage variant

A shorter combined question about windshield glass breakage also reached
OpenAI:

| Metric | Result |
|---|---:|
| HTTP status | 200 |
| Route | combined |
| Wall latency | 22,836.335 ms |
| Retrieval | 1,551.424 ms |
| Reranking | 12,067.033 ms |
| OpenAI answer generation | **2,663.430 ms** |
| Groundedness | `0.112212` |

The audit record showed that `240_1184_e.pdf` page 2 was not present in the
final reranked context. The retrieved old-document chunk was only its title
page. The context therefore did not contain the required Baloise glass rule,
and the groundedness fallback was correct.

Audit request ID: `b5c517b8c3d24cbb8fd8d0c344f9026c`

### Focused marten-bite variant

A focused query placed the document question before the CRM clause and used a
lowercase product title to avoid the name parser interpreting `Motor Vehicle`
as a person's name.

| Metric | Result |
|---|---:|
| HTTP status | 200 |
| Route | combined |
| Wall latency | 22,255.546 ms |
| Retrieval | 1,248.183 ms |
| Reranking | 11,628.186 ms |
| OpenAI answer generation | **3,077.215 ms** |

The answer stage completed, but the final document answer was again replaced
by the groundedness fallback.

Audit request ID: `16c2e453d4bc47ff8660e4c89026692c`

### Email identity variant

Using `lara.neumann@example.test` made CRM routing unambiguous, but the input
guardrail classified the email as PII and stopped the document branch before
retrieval. No OpenAI call was made for this variant.

Audit request ID: `68729bcf88204e63b08c07d880b581be`

## CRM result

The CRM branch consistently returned the correct BaloiseDirect policy:

| Field | Value |
|---|---|
| Customer | Lara Neumann |
| Relevant policy | `TEST-KFZ-2026-1003` |
| Product | Motor Insurance |
| Coverage | Partial Coverage |
| Status | Active |
| Deductible | 300 EUR |
| Annual premium | 720 EUR |

The CRM orchestrator also returned Lara's unrelated liability policy and older
motor policy. This is correct raw retrieval but adds noise to the combined
answer.

## Evaluation

| Component | Result |
|---|---|
| API key persistence | PASS |
| OpenAI authentication | PASS |
| Real online answer call | PASS |
| `gpt-4o-mini` configuration | PASS |
| Combined routing | PASS |
| CRM MCP retrieval | PASS |
| Isolated RAG retrieval | PASS |
| Production self-check | FAIL - timeout |
| Correct evidence in combined reranked context | FAIL |
| Grounded final document answer | FAIL safely |
| Full semantic CRM + RAG answer | **NOT YET PASS** |

## Root cause

The remaining problem is not the OpenAI connection.

The combined user question is sent almost unchanged to the document retriever.
Customer identity, CRM field requests, product names, filenames and the
knowledge question compete in the same retrieval query. In addition, both the
2018 `240_1184_e.pdf` and newer `240_1217_e.pdf` editions are present in the
same Chroma collection.

This causes the reranker to omit the exact evidence page even though an
isolated document-only retrieval test can find it. The groundedness guardrail
then correctly rejects the unsupported generated answer.

## Recommended fix before the next test

1. Split a Combined question into:
   - a CRM subquery;
   - a clean knowledge subquery.
2. Pass only the knowledge subquery to RAG.
3. Apply a source/version metadata filter when a document is explicitly named.
4. Select the relevant BaloiseDirect policy instead of returning every policy.
5. Revisit the 30-second local self-check timeout after measuring the cleaned
   retrieval path.
6. Repeat the same test with output guardrails still enforced.

## Related artifacts

- `reports/combined_crm_rag_marten_test_20260727_134543.md`
- `reports/combined_crm_rag_marten_test_20260727_134543_retrieval_exact.json`
- `data/processed/logs/audit.log`

# Combined CRM + RAG Test: BaloiseDirect Marten Bite

Test date: 2026-07-27  
Overall status: **BLOCKED - full answer generation was not executed**

## Objective

Verify a customer-specific question that requires both:

1. EspoCRM retrieval for Lara Neumann's selected BaloiseDirect policy.
2. RAG retrieval from the BaloiseDirect Motor Vehicle terms.
3. Online OpenAI answer generation from the combined evidence.

## Test question

> Lara Neumann has an active BaloiseDirect Motor Vehicle policy. Based on the
> CRM policy data and only the 2018 BaloiseDirect Motor Vehicle document
> 240_1184_e.pdf, are marten bites included in her selected coverage? State the
> relevant policy number, coverage type, and individual deductible from CRM,
> then cite the document evidence separately. Do not treat the deductible as a
> coverage limit.

## Expected routing

The deterministic query planner returned:

| Field | Result |
|---|---|
| Route | `combined` |
| Customer | Lara Neumann |
| CRM required | yes |
| Policy retrieval required | yes |
| Claim retrieval required | no |
| Document retrieval required | yes |

Result: **PASS**

## CRM verification

The CRM MCP server discovered all five approved read-only tools and successfully
retrieved Lara Neumann and her policies.

The relevant policy is:

| Field | CRM value |
|---|---|
| Policy name | Lara Neumann BaloiseDirect Motor Vehicle 2026 |
| Policy number | `TEST-KFZ-2026-1003` |
| Product | Motor Insurance |
| Coverage type | Partial Coverage |
| Status | Active |
| Term | 2026-07-01 to 2027-06-30 |
| Individual deductible | 300 EUR |
| Annual premium | 720 EUR |

Lara also has the older motor policy `TEST-KFZ-2026-1001` with a 150 EUR
deductible. It is not the policy named BaloiseDirect Motor Vehicle and must not
be selected as the answer to this question.

Result: **PASS**

## PDF ground truth

The visually inspected page 2 of `240_1184_e.pdf` is the Product Information
page of the 2018 BaloiseDirect Motor Vehicle document.

Under Part comprehensive insurance, it explicitly lists damage caused by
`marten bites` among the covered causes. The same page explains that the
selected insurance cover and individual contract details are found in the
customer's policy.

The supported combined conclusion is therefore:

- The document supports marten-bite coverage under Part comprehensive.
- CRM shows that Lara's relevant BaloiseDirect policy has Partial Coverage.
- CRM supplies the individual deductible of 300 EUR.
- The 300 EUR value is a deductible, not a coverage or payment limit.
- This evidence does not itself constitute a final claim decision.

Result: **PASS**

## RAG retrieval test

Exact-document retrieval query:

> According to the 2018 edition BaloiseDirect Motor Vehicle document
> 240_1184_e.pdf, does part comprehensive insurance cover damage to the insured
> vehicle caused by marten bites, and what rule applies to the agreed deductible
> for each claim?

| Metric | Result |
|---|---:|
| Status | PASS |
| Generation used | false |
| Retrieved chunks | 8 |
| Reranked chunks | 5 |
| Wall latency | 50,490.685 ms |
| Target evidence | `240_1184_e.pdf`, page 2 |
| Target evidence rerank | rank 2 |

The required old-document passage was retrieved. However, the newer
`240_1217_e.pdf` document was ranked first even though the question named the
2018 file. Both editions are present in the same collection. For strict
version-specific questions, a source/version metadata filter should be added
before production use.

Result: **PASS with version-selection warning**

## Live combined API attempt

The exact test question was sent to `POST /api/ask`.

| Field | Actual result |
|---|---|
| HTTP status | 503 |
| Wall latency | 8,100.365 ms |
| Route | `combined` |
| Error code | `RETRIEVAL_FAILED` |
| Reported stage | `retrieval` |
| CRM stage | completed in 279.895 ms |
| Local guardrail stage | completed in 61.517 ms |
| Answer generation | not executed |

The health endpoint confirmed the root cause:

| Runtime field | Value |
|---|---|
| Answer provider | OpenAI |
| Answer model | `gpt-4o-mini` |
| OpenAI API key configured | false |
| CRM ready | true |
| Retrieval ready | true |
| Local guardrail ready | true |
| Overall LLM ready | false |
| Pipeline initialization error | `OPENAI_API_KEY` is not configured |

Result: **BLOCKED**

## Final assessment

The test proves that:

- the question is correctly routed to the Combined CRM + RAG path;
- Lara Neumann and the relevant BaloiseDirect policy are available through the
  read-only CRM MCP tools;
- the required BaloiseDirect marten-bite passage is available through RAG;
- local safety processing runs successfully.

It does **not** prove full end-to-end answer generation because no valid rotated
OpenAI API key is configured. The test must not be reported as a full PASS.

## Required rerun

1. Revoke the API key that was previously pasted into chat.
2. Create a new OpenAI API key.
3. Add it locally as `OPENAI_API_KEY` in the ignored `.env` file.
4. Restart `mva-backend`.
5. Rerun the exact same question and require:
   - HTTP 200;
   - route `combined`;
   - CRM source for `TEST-KFZ-2026-1003`;
   - document source `240_1184_e.pdf`, page 2;
   - explicit separation of CRM facts and document evidence;
   - no statement that 300 EUR is a coverage limit.

## Artifacts

- `reports/combined_crm_rag_marten_test_20260727_134543_retrieval.json`
- `reports/combined_crm_rag_marten_test_20260727_134543_retrieval_exact.json`
- `tmp/pdfs/240_1184_e_page_2.png`

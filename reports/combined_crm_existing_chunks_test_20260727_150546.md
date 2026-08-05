# Combined CRM + Existing-Chunks Live Test

Date: 2026-07-27  
Endpoint: `POST http://localhost:8000/api/ask`  
Route: `combined`  
Answer provider: OpenAI  
Answer model: `gpt-4o-mini`

## Corrected target architecture

No PDF was re-read, split, or re-indexed during this request.

1. The deterministic router identified a Combined CRM + knowledge request.
2. CRM retrieval found Lara Neumann and her policies.
3. The context selector retained the Lara Neumann contact and the query-relevant BaloiseDirect policy.
4. The document retrieval query was reduced to:

   `are marten bites included in her selected coverage?`

5. The explicitly requested source `240_1184_e.pdf` was used as a source constraint.
6. The two best matching chunks were selected from the already persisted ChromaDB chunks for that PDF.
7. These two PDF chunks and the two selected CRM fact records were passed together to one OpenAI answer-generation call.
8. The generated answer passed the post-generation groundedness and safety checks.

## Test question

> Lara Neumann has an active BaloiseDirect Motor Vehicle policy. Based on the CRM policy data and only the 2018 BaloiseDirect Motor Vehicle document 240_1184_e.pdf, are marten bites included in her selected coverage? State the relevant policy number, coverage type, and individual deductible from CRM, then cite the document evidence separately. Do not treat the deductible as a coverage limit.

## Selected CRM context

- Contact: Lara Neumann
- Policy number: `TEST-KFZ-2026-1003`
- Policy name: Lara Neumann BaloiseDirect Motor Vehicle 2026
- Product: Motor Insurance
- Coverage type: Partial Coverage
- Status: Active
- Term: 2026-07-01 to 2027-06-30
- Individual deductible: 300 EUR
- Annual premium: 720 EUR

Unrelated policies for Lara Neumann were not included in the generation context.

## Selected existing PDF chunks

Exactly two already-indexed chunks from `240_1184_e.pdf` were passed to generation:

- Chunk metadata page `5`: contains the detailed part-comprehensive item for marten bites, including consequential damage.
- Chunk metadata page `1`: contains the product summary that lists marten bites among the insured causes of damage.

The audit record confirms `hasMarten=true` for both chunks.

## Generated answer

> Lara Neumann has an active BaloiseDirect Motor Vehicle policy with the following details: Policy number TEST-KFZ-2026-1003, Coverage type: Partial Coverage, and Individual deductible: 300 EUR.
>
> Marten bites are included in her selected coverage as part of the part comprehensive insurance, which indemnifies for damage caused by marten bites [data\raw\pdfs\240_1184_e:1].

## Runtime result

| Metric | Result |
|---|---:|
| API status | `complete` |
| Route | `combined` |
| Warning | none |
| End-to-end client time | 12.24 s |
| Backend total time | 12.10 s |
| CRM stage | 105.68 ms |
| Existing-chunk selection | 57.88 ms |
| Answer generation | 4.63 s |
| Groundedness score | 0.652619 |
| Final safety decision | `allow` |

## Self-check state

- `selfCheckEnabled=false`
- `selfCheckSkipped=true`
- `selfCheckSkipReason=disabled_by_configuration`
- The value is persisted as `SELF_CHECK_ENABLED=false`.
- The application default is also `false`; tests do not switch it back to `true`.

## Latency optimization

An intermediate successful run took 42.49 seconds because the CPU neural reranker consumed 28.22 seconds. For an explicitly named PDF, the source scope and deterministic lexical chunk selection already provide a bounded candidate set. The redundant neural reranker is therefore skipped for this path.

The optimized request completed in 12.24 seconds, about 71% faster, while retaining both chunks that contain the relevant marten evidence.

## Verification

- Python compilation: passed
- Full unit suite: `142 passed`
- Combined integration tests: passed
- Test assertion confirms `load_and_split_documents` is not called in this Combined request path.
- Test assertion confirms existing PDF chunks and CRM facts share one `generate_answer` call.
- Live health check confirms CRM, retrieval, and LLM are ready with Self-Check disabled.

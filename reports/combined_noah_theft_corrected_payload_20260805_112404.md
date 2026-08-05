# Combined Noah Theft Live E2E - Corrected Client Payload

Generated: 2026-08-05T11:24:04+02:00

## Executive Summary

The client payload error was corrected in the one-off PowerShell test command: the exact question was serialized under the top-level key `question`. No production code or configuration was changed. Exactly one new `POST /api/ask` request was sent, with no client retry.

The request entered the endpoint and selected route `combined`. CRM policy selection and retrieval were correct, including the page-17 theft-duty chunk. The run nevertheless ended **FAIL** with HTTP 206 because:

1. Completeness still considered `theft_partial_comprehensive_coverage` missing after the single regeneration. The answer stated partially comprehensive coverage but omitted the required product-sheet citation `[motor-vehicle-insurance-product-sheet.pdf, page 3]`.
2. Groundedness v5 produced `0.38`, below `0.7888`, after applying `unsupported_numeric_fact_cap_0.38` for `unsupported_atomic_facts=["number:3"]`. The generated answer used an inline numbered list item `3.`; diagnostics did not identify any unsupported semantic insurance claim.
3. Safety therefore returned the configured low-groundedness fallback. The final client answer contained no citations.

## Client Payload Correction

The previous execution used an ad hoc PowerShell command rather than a persisted client script. The command was corrected from `{ "query": ... }` to `{ "question": ... }`.

Local preflight evidence:

- FastAPI schema: `AskQuestionRequest.question` in `src/main.py:301-303`.
- Serialized top-level keys: `["question"]`.
- `hasQuestion=true`; `hasQuery=false`.
- Serialized question exactly matched the required text.
- Configuration changed: **No**.

## Runtime Configuration

| Setting | Effective value |
|---|---|
| CRM | enabled and ready |
| Self-Check | disabled |
| Pipeline authenticated | false |
| Groundedness | `fact_aware_claim_support_v5` |
| Threshold | `0.7888`, calibration file |
| Safety | enabled, `enforce` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer provider/model | OpenAI / `gpt-4o-mini` |
| PDF fingerprint changed | false |
| Embedding model changed | false |

No reindex, collection mutation, embedding recomputation, CRM reseeding, PDF change, or Chroma change was performed.

## Exact Question

```text
Noah Weber reports the theft of his car. Is this loss generally covered by his current motor insurance, and which documented conditions apply?

Please provide the relevant active policy number, coverage type, individual deductible and annual premium. Clearly distinguish the general insurance terms from Noah's individual contract data, and do not make a final claim decision.
```

## A. HTTP and Routing

| Field | Result |
|---|---|
| HTTP status | `206` |
| API status | `partial` |
| Request ID | `05027d5c31d34e9282fdc245044d9107` |
| Route | `combined` |
| Pipeline entered | Yes |
| Client request count | 1 |
| Client retry | No |
| Server timeout stage | none |
| Self-Check | skipped, `disabled_by_configuration` |

All route, CRM, retrieval, reranker, answer, and Groundedness stages reached `completed` status.

## B. CRM and Policy Selection

Selected policy: `TEST-KFZ-2026-1701`.

| Fact | Value | Evidence |
|---|---|---|
| Customer | Noah Weber | CRM contact `espocrm:6a65e0b8321321652` |
| Policy | `TEST-KFZ-2026-1701` | CRM policy `espocrm:6a65e0b9a9eb98639` |
| Product | Helvetia Motor Vehicle Insurance, STI Edition March 2026 | same CRM policy |
| Coverage | Partially comprehensive cover | same CRM policy |
| Status | Active | same CRM policy |
| Term | 2026-05-01 to 2027-04-30 | same CRM policy |
| Deductible | 300 EUR | same CRM policy |
| Annual premium | 735 EUR | same CRM policy |

The CRM record explicitly associates the policy with Noah Weber. `TEST-RS-2026-1702`, an active Legal Protection policy, was considered but not selected because it conflicted with the motor domain. No inactive policy was selected or present in the final candidate list.

## C. Retrieval and Reranking

| Rank | File | Page | Chunk ID | Reranker score | Content |
|---:|---|---:|---|---:|---|
| 1 | `motor-vehicle-insurance-sti.pdf` | 13 | `dfa04e4f-b781-40d9-baa2-eac6f7a2ed58` | 2.59433007 | K2.1.4: theft/misappropriation/robbery scope, involuntary damage, family exclusion |
| 2 | `motor-vehicle-insurance-sti.pdf` | 13 | `f78066a2-fb42-404e-b1bb-1bd7ee8b8312` | -1.96668172 | Neighboring fire/natural-forces text; not used as a completeness requirement |
| 3 | `motor-vehicle-insurance-sti.pdf` | 17 | `768239c2-f3ef-4806-9b99-41ed105144d3` | 2.17699337 | K5.4: immediate police report and immediate recovery/known-location notification |
| 4 | `motor-vehicle-insurance-product-sheet.pdf` | 3 | `f69012fa-1c36-4fbb-9e71-f92bb1188b11` | 4.4769516 | Theft listed as a partially comprehensive insured benefit |
| 5 | `motor-vehicle-insurance-sti.pdf` | 16 | `7c0096a6-e783-4c1a-976e-0360b31d8ed3` | 0.17321973 | Comprehensive-insurance exclusions/operation damage neighbor |

All five PDF chunks plus the Noah contact and selected policy CRM facts reached the final generation context. The page-17 duties chunk was present at final rank 3.

## D. Completeness Requirements

| Requirement ID | Concept | Required source | File/page/chunk | Type |
|---|---|---|---|---|
| `theft_partial_comprehensive_coverage` | Theft generally covered under partially comprehensive insurance | `[motor-vehicle-insurance-product-sheet.pdf, page 3]` | product sheet, p.3, `f69012fa-1c36-4fbb-9e71-f92bb1188b11` | General terms |
| `theft_loss_scope` | Loss/disappearance/destruction/damage through theft, misappropriation or robbery | `[motor-vehicle-insurance-sti.pdf, page 13]` | STI, p.13, `dfa04e4f-b781-40d9-baa2-eac6f7a2ed58` | General terms |
| `theft_involuntary_damage` | Damage must occur involuntarily | `[motor-vehicle-insurance-sti.pdf, page 13]` | STI, p.13, same chunk | General terms |
| `theft_family_exclusion` | No compensation for acts by family members | `[motor-vehicle-insurance-sti.pdf, page 13]` | STI, p.13, same chunk | General terms |
| `theft_police_reporting` | Notify responsible police without delay | `[motor-vehicle-insurance-sti.pdf, page 17]` | STI, p.17, `768239c2-f3ef-4806-9b99-41ed105144d3` | General terms |
| `theft_recovery_notification` | Inform insurer immediately on recovery/known location | `[motor-vehicle-insurance-sti.pdf, page 17]` | STI, p.17, same chunk | General terms |
| `crm_policy_number` | `TEST-KFZ-2026-1701` | `[CRM: TEST-KFZ-2026-1701]` | `espocrm:6a65e0b9a9eb98639` | Individual CRM data |
| `crm_coverage_type` | Partially comprehensive cover | same CRM citation | same CRM record | Individual CRM data |
| `crm_deductible` | 300 EUR | same CRM citation | same CRM record | Individual CRM data |
| `crm_annual_premium` | 735 EUR | same CRM citation | same CRM record | Individual CRM data |

No abroad requirement was instantiated. Fire and natural-forces content was not classified as a theft requirement.

## E. Generation and Regeneration

### Raw first OpenAI answer

```text
Yes, the loss of Noah Weber's car due to theft is generally covered under his current motor insurance, which is a partially comprehensive cover. The relevant conditions that apply are as follows:

1. **Coverage for Theft**: The insurance covers loss, disappearance, destruction, or damage caused by theft, misappropriation, or robbery, provided that the damage occurred involuntarily. However, no compensation is paid if the act was committed by family members.
   - Evidence: [motor-vehicle-insurance-sti.pdf, page 13]

2. **Police Reporting**: The theft must be reported to the responsible police without delay. If the theft occurs abroad, it must also be reported to the police station at his Swiss place of residence.
   - Evidence: [motor-vehicle-insurance-sti.pdf, page 17]

3. **Notification of Recovery**: If the vehicle is found or if anything is learned regarding its whereabouts, the insurer must be informed without delay.
   - Evidence: [motor-vehicle-insurance-sti.pdf, page 17]

Now, regarding Noah's individual contract data:

- **Active Policy Number**: TEST-KFZ-2026-1701
  - Evidence: [CRM: TEST-KFZ-2026-1701]

- **Coverage Type**: Partially comprehensive cover
  - Evidence: [CRM: TEST-KFZ-2026-1701]

- **Individual Deductible**: 300 EUR
  - Evidence: [CRM: TEST-KFZ-2026-1701]

- **Annual Premium**: 735 EUR
  - Evidence: [CRM: TEST-KFZ-2026-1701]

This information provides a general overview of the coverage for theft under Noah's policy, along with his specific contract details. Please note that this is not a final claim decision.
```

Present after first answer: all requirements except `theft_partial_comprehensive_coverage`.

Missing before retry: `theft_partial_comprehensive_coverage`. The general coverage sentence was present, but the required product-sheet page-3 source label was absent.

Completeness retry performed: **Yes**. The regenerated answer was textually identical to the first answer and still omitted `[motor-vehicle-insurance-product-sheet.pdf, page 3]`.

- Answer-generation calls: 2, the configured maximum.
- Missing after retry: `theft_partial_comprehensive_coverage`.
- `completenessPass=false`.

## F. Final Answer

Exact answer returned to the client:

```text
I could not generate an answer that is sufficiently supported by the available documents.
```

The pre-fallback answer correctly separated general terms from CRM facts, included the correct policy, coverage, deductible and premium, and avoided a final claim decision. It included the police and recovery duties. It did not contain fire/natural-forces clauses, a separate extracted-conditions block, or legacy material-qualifier output. However, it unnecessarily included the abroad reporting sentence even though the query did not describe an abroad scenario. The final fallback contained none of the requested facts.

## G. Citations

Pre-fallback citations:

- `[motor-vehicle-insurance-sti.pdf, page 13]`: supported theft scope, involuntary condition, family exclusion.
- `[motor-vehicle-insurance-sti.pdf, page 17]` twice: supported police reporting and recovery/known-location notification.
- `[CRM: TEST-KFZ-2026-1701]` four times: supported policy number, coverage, deductible, premium.

Missing pre-fallback citation: `[motor-vehicle-insurance-product-sheet.pdf, page 3]` for partially comprehensive theft coverage.

Final client answer citations: none. `pdfCitations=[]`. Therefore final Citation Presence is **FAIL**. A named Citation Support result was not emitted after fallback; Groundedness reported `citation_mismatches=[]` for the pre-fallback answer, so the citations that were present did not point to conflicting sources. Page 17 was appropriately cited before fallback.

## H. Groundedness

- Algorithm: `fact_aware_claim_support_v5`
- Implementation: `fact_aware_claim_support_v5_experimental`
- Threshold: `0.7888` from calibration file
- Base v4 score: `0.75744`
- Minimum claim support: `0.8`
- Average claim support: `0.886294`
- Final score: `0.38`
- Result: **FAIL**
- Cap: `unsupported_numeric_fact_cap_0.38`
- Unsupported atomic facts: `number:3`
- Semantic unsupported facts: none listed
- Citation, policy-number, coverage-type, structured, and polarity mismatches: none

Claims considered: general theft coverage; theft/misappropriation/robbery and involuntary damage; family exclusion; police reporting; abroad reporting; recovery notification; policy number; coverage type; deductible; annual premium.

The diagnostics establish that `number:3` triggered the numeric cap. The generated answer contains the numbered list marker `3.` on the recovery-notification heading; treating that marker as the unsupported fact is consistent with the evidence, but the diagnostic does not itself store a character-level provenance link. Groundedness remains separate from Completeness.

## I. Safety and Consistency

- Coverage consistency: no coverage-type mismatch and no policy-number mismatch; **PASS** on those recorded checks.
- Numeric consistency: `number:3` mismatch recorded; deductible and premium claim details matched CRM.
- Safety pre-action: `allow`.
- Safety context action: `redact`.
- PII: CRM email redacted; allowed business contact recognized; no raw email exposed in the final answer.
- Safety post-action: `fallback`.
- Safety decision: `post_fallback`.
- Safety fallback: **Yes**, reason `low_groundedness`.
- HTTP 206 handling: **Yes**. API warning `ANSWER_INCOMPLETE`, missing item `theft_partial_comprehensive_coverage`.

## J. Latencies

| Stage | Latency |
|---|---:|
| Route planning | 3.382 ms |
| CRM | 700.042 ms |
| Guardrail/safety pre-context processing | 2893.237 ms |
| Lexical retrieval | 28.130 ms |
| Vector retrieval | 2609.921 ms |
| Hybrid merge | 261.050 ms |
| Total retrieval | 2900.596 ms |
| Reranking | 10831.538 ms |
| Answer generation aggregate, both calls | 15017.375 ms |
| First generation separately | not captured |
| Completeness evaluation separately | not captured |
| Regeneration separately | not captured |
| Groundedness | 2764.747 ms |
| Safety post-processing separately | not captured |
| Total server | 35537.383 ms |
| Total client observed | 35721 ms |

## Final Status

**FAIL**.

The payload correction succeeded and the Combined CRM+RAG pipeline executed. Policy selection and evidence retrieval were correct. PASS criteria failed at Completeness (`completenessPass=false` after one regeneration), Groundedness (`0.38 < 0.7888`), Safety (grounding fallback), final citation presence, and final-answer completeness. No further request or production change was made.


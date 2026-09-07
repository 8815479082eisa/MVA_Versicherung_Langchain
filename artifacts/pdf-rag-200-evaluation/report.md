# PDF RAG-only 200-case evaluation

> Automated runtime and deterministic reference-evidence validation. It is not independent human validation.

- Cases: 200
- Operational completion: 73.50%
- Fallback rate: 20.00%
- RAG-only route accuracy: 73.00%
- Expected PDF source hit rate: 26.50%
- Exact reference page hit rate: 16.00%
- Groundedness pass rate (threshold 0.7888): 60.40%
- Mean groundedness score: 0.6402
- Mean reference token recall: 23.73%
- Mean claim-support rate: 41.50%
- Citation presence: 41.50%
- Citation link precision: 100.00%

## Latency (milliseconds)

Mean 33359.6 | Median 36284.2 | P90 63637.4 | P95 70677.7 | P99 76436.4 | Max 81260.2

## Error counts

- `reference_recall_below_0.5`: 176
- `expected_page_missing`: 168
- `expected_source_missing`: 147
- `route_mismatch`: 54
- `operational_failure`: 53
- `fallback`: 40
- `groundedness_failed`: 40

## Per-PDF results

| PDF | Cases | Source hit | Page hit | Groundedness pass | Mean groundedness | Mean reference recall | Mean latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `assistance-brochure` | 10 | 30.00% | 30.00% | 100.00% | 0.8723 | 21.46% | 42330.1 |
| `assistance-sti` | 20 | 30.00% | 30.00% | 75.00% | 0.8015 | 32.63% | 39443.2 |
| `brochure-household-contents-and-private-liability` | 10 | 0.00% | 0.00% | 100.00% | 0.8612 | 17.29% | 32973.6 |
| `brochure-services` | 8 | 12.50% | 12.50% | 100.00% | 0.9620 | 15.20% | 36459.5 |
| `buildings-insurance-sti` | 25 | 4.00% | 0.00% | 5.56% | 0.3192 | 7.85% | 29118.7 |
| `household-contents-private-liability-sti` | 35 | 74.29% | 17.14% | 100.00% | 0.8665 | 12.09% | 18602.0 |
| `legal-protection-sti` | 15 | 6.67% | 6.67% | 10.00% | 0.1660 | 12.75% | 34695.8 |
| `motor-vehicle-insurance-product-sheet` | 10 | 50.00% | 50.00% | 71.43% | 0.6508 | 39.66% | 34746.6 |
| `motor-vehicle-insurance-sti` | 40 | 20.00% | 20.00% | 88.89% | 0.8938 | 19.93% | 43081.9 |
| `mutual-provisions-pkv` | 20 | 0.00% | 0.00% | 0.00% | 0.3800 | 5.77% | 30335.7 |
| `rental-guarantee-insurance-sti` | 7 | 28.57% | 28.57% | 100.00% | 0.9152 | 29.49% | 37345.8 |

## Interpretation boundary

Reference token recall and claim-support are deterministic proxies, not expert insurance review. Groundedness is read from the production evidence guardrail; a failed or fallback answer is a retrieval/runtime risk, not automatically a confirmed hallucination. The saved per-case JSONL contains the full payload and reference evidence needed for manual audit.

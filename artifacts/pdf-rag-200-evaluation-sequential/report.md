# PDF RAG-only 200-case evaluation

> Automated runtime and deterministic reference-evidence validation. It is not independent human validation.

- Cases: 200
- Operational completion: 98.50%
- Fallback rate: 36.00%
- RAG-only route accuracy: 98.50%
- Expected PDF source hit rate: 57.00%
- Exact reference page hit rate: 42.50%
- Groundedness pass rate (threshold 0.7888): 62.76%
- Mean groundedness score: 0.6848
- Mean reference token recall: 42.48%
- Mean claim-support rate: 62.94%
- Citation presence: 62.94%
- Citation link precision: 100.00%

## Latency (milliseconds)

Mean 20633.4 | Median 19502.2 | P90 37927.0 | P95 46580.3 | P99 56772.5 | Max 59896.7

## Error counts

- `reference_recall_below_0.5`: 122
- `expected_page_missing`: 115
- `expected_source_missing`: 86
- `groundedness_failed`: 73
- `fallback`: 72
- `operational_failure`: 3
- `route_mismatch`: 3

## Per-PDF results

| PDF | Cases | Source hit | Page hit | Groundedness pass | Mean groundedness | Mean reference recall | Mean latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `assistance-brochure` | 10 | 90.00% | 90.00% | 100.00% | 0.8609 | 75.33% | 26528.7 |
| `assistance-sti` | 20 | 65.00% | 60.00% | 60.00% | 0.7338 | 59.87% | 31355.6 |
| `brochure-household-contents-and-private-liability` | 10 | 0.00% | 0.00% | 100.00% | 0.8713 | 19.24% | 11716.0 |
| `brochure-services` | 8 | 62.50% | 62.50% | 62.50% | 0.8218 | 60.12% | 27770.7 |
| `buildings-insurance-sti` | 25 | 8.00% | 4.00% | 8.00% | 0.3241 | 12.89% | 15129.6 |
| `household-contents-private-liability-sti` | 35 | 100.00% | 22.86% | 100.00% | 0.8641 | 16.59% | 13010.5 |
| `legal-protection-sti` | 15 | 26.67% | 26.67% | 26.67% | 0.3178 | 27.39% | 19282.0 |
| `motor-vehicle-insurance-product-sheet` | 10 | 90.00% | 90.00% | 90.00% | 0.8186 | 77.23% | 21648.5 |
| `motor-vehicle-insurance-sti` | 40 | 75.00% | 75.00% | 81.08% | 0.8375 | 70.48% | 26827.5 |
| `mutual-provisions-pkv` | 20 | 0.00% | 0.00% | 0.00% | 0.3800 | 8.68% | 15319.6 |
| `rental-guarantee-insurance-sti` | 7 | 100.00% | 100.00% | 100.00% | 0.9446 | 95.01% | 25163.2 |

## Interpretation boundary

Reference token recall and claim-support are deterministic proxies, not expert insurance review. Groundedness is read from the production evidence guardrail; a failed or fallback answer is a retrieval/runtime risk, not automatically a confirmed hallucination. The saved per-case JSONL contains the full payload and reference evidence needed for manual audit.

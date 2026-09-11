# PDF RAG-only 200-case evaluation

> Automated runtime and deterministic reference-evidence validation. It is not independent human validation.

- Cases: 200
- Operational completion: 92.50%
- Fallback rate: 0.00%
- RAG-only route accuracy: 92.50%
- Expected PDF source hit rate: 34.00%
- Exact reference page hit rate: 17.50%
- Groundedness pass rate (threshold 0.7888): 72.92%
- Mean groundedness score: 0.7982
- Mean reference token recall: 18.20%
- Mean claim-support rate: 36.76%
- Citation presence: 16.22%
- Citation link precision: 100.00%

## Latency (milliseconds)

Mean 119439.2 | Median 121108.9 | P90 167212.7 | P95 180700.7 | P99 181631.5 | Max 181718.2

## Error counts

- `reference_recall_below_0.5`: 173
- `expected_page_missing`: 165
- `expected_source_missing`: 132
- `groundedness_failed`: 26
- `operational_failure`: 15
- `route_mismatch`: 15

## Per-PDF results

| PDF | Cases | Source hit | Page hit | Groundedness pass | Mean groundedness | Mean reference recall | Mean latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `assistance-brochure` | 10 | 10.00% | 10.00% | 100.00% | 1.0000 | 7.86% | 158374.5 |
| `assistance-sti` | 20 | 30.00% | 25.00% | 75.00% | 0.8438 | 23.91% | 162400.9 |
| `brochure-household-contents-and-private-liability` | 10 | 0.00% | 0.00% | 0.00% | 0.0000 | 3.46% | 139125.0 |
| `brochure-services` | 8 | 25.00% | 25.00% | 100.00% | 1.0000 | 28.38% | 119864.4 |
| `buildings-insurance-sti` | 25 | 92.00% | 0.00% | 100.00% | 1.0000 | 7.35% | 138722.3 |
| `household-contents-private-liability-sti` | 35 | 8.57% | 2.86% | 37.50% | 0.4167 | 6.94% | 129778.9 |
| `legal-protection-sti` | 15 | 86.67% | 40.00% | 92.86% | 0.9464 | 19.92% | 113590.4 |
| `motor-vehicle-insurance-product-sheet` | 10 | 50.00% | 50.00% | 55.56% | 0.7315 | 30.09% | 76798.2 |
| `motor-vehicle-insurance-sti` | 40 | 37.50% | 37.50% | 80.00% | 0.9052 | 35.26% | 89487.6 |
| `mutual-provisions-pkv` | 20 | 0.00% | 0.00% | 20.00% | 0.2714 | 4.39% | 101059.4 |
| `rental-guarantee-insurance-sti` | 7 | 0.00% | 0.00% | 0.00% | 0.4167 | 3.36% | 89008.5 |

## Interpretation boundary

Reference token recall and claim-support are deterministic proxies, not expert insurance review. Groundedness is read from the production evidence guardrail; a failed or fallback answer is a retrieval/runtime risk, not automatically a confirmed hallucination. The saved per-case JSONL contains the full payload and reference evidence needed for manual audit.

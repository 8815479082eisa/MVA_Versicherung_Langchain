# Final-System Reference-Based Answer Evaluation

> This is an automated, reference-based technical validation. It is not independent human validation.

- Cases: 100
- Evaluable: 100
- Automated overall pass: 89/100 (89.00%)

## Per-route results

| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 25 | 25 | 96.00% | 100.00% | 96.00% | n/a | n/a | n/a | n/a | n/a |
| `retrieval_only` | 25 | 25 | 80.00% | 100.00% | n/a | 88.00% | 84.00% | 100.00% | 85.33% | 71.76% |
| `combined` | 25 | 25 | 84.00% | 96.00% | 96.00% | 86.00% | 96.00% | 100.00% | 85.60% | 63.74% |
| `denied` | 25 | 25 | 96.00% | 96.00% | n/a | n/a | n/a | n/a | n/a | n/a |

## Error catalog

- `requirement_recall_below_threshold`: 7
- `expected_document_citation_missing`: 5
- `expected_document_source_missing`: 4
- `groundedness_failed`: 3
- `crm_fact_incomplete_or_incorrect`: 2
- `route_mismatch`: 2
- `unsafe_or_missing_denial`: 1

## Interpretation boundary

*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.

The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.

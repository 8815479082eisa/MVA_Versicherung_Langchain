# Final-System Reference-Based Answer Evaluation

> This is an automated, reference-based technical validation. It is not independent human validation.

- Cases: 80
- Evaluable: 80
- Automated overall pass: 73/80 (91.25%)

## Per-route results

| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100.00% | 100.00% | 100.00% | n/a | n/a | n/a | n/a | n/a |
| `retrieval_only` | 20 | 20 | 80.00% | 100.00% | n/a | 85.00% | 80.00% | 100.00% | 76.67% | 68.99% |
| `combined` | 20 | 20 | 85.00% | 100.00% | 95.00% | 85.00% | 95.00% | 100.00% | 84.43% | 61.64% |
| `denied` | 20 | 20 | 100.00% | 100.00% | n/a | n/a | n/a | n/a | n/a | n/a |

## Error catalog

- `requirement_recall_below_threshold`: 6
- `expected_document_citation_missing`: 5
- `expected_document_source_missing`: 5
- `groundedness_failed`: 4
- `crm_fact_incomplete_or_incorrect`: 1

## Interpretation boundary

*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.

The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.

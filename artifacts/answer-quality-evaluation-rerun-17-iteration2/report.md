# Final-System Reference-Based Answer Evaluation

> This is an automated, reference-based technical validation. It is not independent human validation.

- Cases: 17
- Evaluable: 17
- Automated overall pass: 3/17 (17.65%)

## Per-route results

| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `retrieval_only` | 8 | 8 | 25.00% | 100.00% | n/a | 31.25% | 37.50% | 100.00% | 37.50% | 32.53% |
| `combined` | 9 | 9 | 11.11% | 100.00% | 33.33% | 11.11% | 33.33% | 100.00% | 33.33% | 23.00% |
| `denied` | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Error catalog

- `requirement_recall_below_threshold`: 14
- `expected_document_citation_missing`: 11
- `expected_document_source_missing`: 11
- `groundedness_failed`: 11
- `crm_fact_incomplete_or_incorrect`: 6

## Interpretation boundary

*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.

The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.

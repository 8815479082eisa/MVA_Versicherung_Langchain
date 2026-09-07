# Final-System Reference-Based Answer Evaluation

> This is an automated, reference-based technical validation. It is not independent human validation.

- Cases: 3
- Evaluable: 3
- Automated overall pass: 1/3 (33.33%)

## Per-route results

| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `retrieval_only` | 2 | 2 | 0.00% | 100.00% | n/a | 0.00% | 0.00% | n/a | 0.00% | 0.00% |
| `combined` | 1 | 1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 69.23% |
| `denied` | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Error catalog

- `expected_document_citation_missing`: 2
- `expected_document_source_missing`: 2
- `groundedness_failed`: 2
- `requirement_recall_below_threshold`: 2

## Interpretation boundary

*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.

The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.

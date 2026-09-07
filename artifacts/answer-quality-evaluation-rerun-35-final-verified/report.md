# Final-System Reference-Based Answer Evaluation

> This is an automated, reference-based technical validation. It is not independent human validation.

- Cases: 35
- Evaluable: 35
- Automated overall pass: 31/35 (88.57%)

## Per-route results

| Expected route | Cases | Evaluable | Overall pass | Route accuracy | CRM fact recall | Requirement recall | Expected-source citation | Citation link precision | Citation coverage* | Claim support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 5 | 5 | 100.00% | 100.00% | 100.00% | n/a | n/a | n/a | n/a | n/a |
| `retrieval_only` | 12 | 12 | 91.67% | 100.00% | n/a | 91.67% | 91.67% | 100.00% | 87.50% | 77.90% |
| `combined` | 17 | 17 | 82.35% | 100.00% | 100.00% | 85.29% | 100.00% | 100.00% | 97.06% | 62.37% |
| `denied` | 1 | 1 | 100.00% | 100.00% | n/a | n/a | n/a | n/a | n/a | n/a |

## Error catalog

- `requirement_recall_below_threshold`: 4
- `expected_document_citation_missing`: 1
- `expected_document_source_missing`: 1
- `groundedness_failed`: 1

## Interpretation boundary

*Citation link precision checks whether every explicit PDF marker maps to returned source metadata. Citation coverage is a deterministic marker-based proxy: a terminal source marker is treated as applying to the whole answer. Neither value is an ALCE/NLI score.

The metrics validate technically specified requirements, synthetic CRM facts, source metadata, lexical claim support, and safety invariants. Extra top-k retrieval results are reported descriptively and are not treated as citation errors. The results do not establish expert-level insurance correctness, user comprehension, user trust, or production readiness.

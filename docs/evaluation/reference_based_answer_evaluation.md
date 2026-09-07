# Reference-based answer evaluation (80 final-system cases)

This workflow evaluates the complete API response for the four final routes. It is an
automated, reference-based technical validation and must not be described as independent
human validation.

## Inputs

- `data/benchmarks/routing/routing_eval_80.jsonl`: 20 CRM-only, 20 retrieval-only,
  20 combined and 20 denied cases.
- `data/benchmarks/answer_quality/reference_spec_v1.json`: versioned answer
  requirements and accepted formulations.
- `data/synthetic/crm/*.csv`: deterministic CRM ground truth.
- `/api/ask`: full answer, route, source and diagnostic payload.

## What is checked

| Route | Automated checks |
| --- | --- |
| CRM-only | exact presence of the requested CSV facts, route, status and safety invariants |
| Retrieval-only | required concepts, expected document/source, explicit citation linkage, marker-based citation coverage, lexical claim support and runtime groundedness |
| Combined | all CRM-only and retrieval-only checks; both evidence channels must succeed |
| Denied | denied route, safe response and absence of CRM/document access |

Returned top-k documents that are not explicitly cited are reported descriptively and are
not classified as citation errors. Citation link precision checks whether explicit PDF
markers map to returned source metadata. Citation coverage is a deterministic proxy: a
terminal source marker is treated as applying to the whole answer. It is not an ALCE/NLI
measurement.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_final_system_answers.py --mode both
```

The collector writes every response atomically and resumes by case ID. To regenerate only
the metrics from saved full responses:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_final_system_answers.py --mode evaluate
```

Use `--overwrite-responses` only when a new model/API run is intended. `--routes`, `--start`
and `--limit` can select a subset.

## Outputs

All outputs are written to `artifacts/answer-quality-evaluation/`:

- `responses_full.jsonl`: auditable full API payloads;
- `per_case_results.jsonl` and `.csv`: case-level scores and failure reasons;
- `summary.json`: aggregate and per-route metrics with Wilson intervals;
- `error_catalog.json`: failed cases and missing requirements/facts;
- `report.md`: compact thesis-oriented summary;
- `collection_metadata.json`: health/configuration snapshot for the collection run.

## Current run

The 2026-08-23 run evaluated all 80 cases. Operational completion was 100%, routing
accuracy was 98.75%, and the strict automated overall pass rate was 45/80 (56.25%; Wilson
95% interval 45.34%–66.59%). Per-route pass rates were 75% for CRM-only, 40% for
retrieval-only, 15% for combined and 95% for denied.

These values show technical behavior against declared references. They do not establish
expert insurance correctness, user understanding, trust, usability or production
readiness.

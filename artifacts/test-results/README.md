# Test Results Overview

This index summarizes all currently present test-result folders under `artifacts/test-results/`.

| Test Result Label | run_id | Associated Pipeline | Associated Test Setup | Processed | Failed | EM | Token F1 | Distinguishing Characteristics |
|---|---|---|---|---:|---:|---:|---:|---|
| `test-result-20260325T002026Z` | `20260325T002226Z` | `pipeline-20260325` | `test-setup-first-20-EM-F1` | 20 | 0 | 0.000000 | 0.343531 | Oldest preliminary 20-sample run; folder label and run_id differ; tagged by run_id form (`test-result-20260325T002226Z`). |
| `test-result-20260326T213943Z` | `20260326T213943Z` | `pipeline-20260326` | `test-setup-first-20-EM-F1` | 20 | 0 | 0.000000 | 0.351242 | Registered post-reranker-fallback result; strong tag linkage (`pipeline-20260326`, `test-result-20260326T213943Z`); canonical line-level evidence documented from GPU run. |
| `test-result-20260326T222652Z` | `20260326T222652Z` | `pipeline-20260326` (inferred) | `test-setup-first-20-EM-F1` (inferred) | 20 | 0 | 0.0 | 0.3699221767271236 | In-process FastAPI TestClient execution evidence; includes `summary.json` and JSONL with `source_ids`; no explicit tag or commit-hash linkage. |
| `test-result-20260326T233038Z` | `20260326T233038Z` | `pipeline-20260326` (inferred) | `test-setup-first-20-EM-F1` (audit-export equivalent) | 20 | 0 | 0.0 | 0.3699221767271236 | Exported from `audit.log` (`2026-03-26T23:*`) using `scripts/evaluation/export_eval_from_audit.py`; includes JSONL + summary + command metadata. |

## Notes on Traceability and Consistency
- All listed results use 20-sample InsuranceQA evaluation with EM/F1 semantics, but their registration maturity differs.
- `test-result-20260325T002026Z` contains a naming inconsistency between folder label and run id.
- `test-result-20260326T222652Z` is currently less formally registered (no commit-hash file / no dedicated tag).
- `test-result-20260326T233038Z` is a post-hoc export artifact (audit-derived), not a direct raw evaluator emission.
- For formal comparability, prefer tagged results with explicit pipeline + setup linkage.

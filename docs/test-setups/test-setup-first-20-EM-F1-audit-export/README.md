# test-setup-first-20-EM-F1-audit-export

## Dataset
InsuranceQA

## Split
test (equivalent first-20 benchmark query set, reconstructed from audit rows)

## Sample Size
20

## Metrics
- Exact Match (EM)
- Token F1

## Execution Mode
Post-hoc export from `data/processed/logs/audit.log` into eval JSONL format.

## Export Script
- `scripts/evaluation/export_eval_from_audit.py`
- wrapper: `scripts/export_eval_from_audit.py`

## Run Parameters (Recorded Example)
- audit-log: `data/processed/logs/audit.log`
- timestamp-prefix: `2026-03-26T23:`
- run-id: `20260326T233038Z`
- out: `artifacts/test-results/test-result-20260326T233038Z/eval_insuranceqa_results.jsonl`
- summary-out: `artifacts/test-results/test-result-20260326T233038Z/summary.json`

## Associated Test Results
- `artifacts/test-results/test-result-20260326T233038Z`

## Notes
- This setup does not replay model inference; it converts already logged answers.
- Useful when the original online runtime is not available but audit evidence exists.

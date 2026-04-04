# Test Result: test-result-20260326T233038Z

## 1. Test Result ID
- Folder label: `test-result-20260326T233038Z`
- Internal run_id: `20260326T233038Z`

## 2. Objective
- Reconstruct a 20-sample InsuranceQA EM/F1 result from `audit.log` using export tooling.

## 3. Pipeline Version Used
- `pipeline-20260326`
- Reference: `pipeline-reference.txt`
- Snapshot: `snapshots/pipeline-readme-snapshot.md`

## 4. Test Setup Used
- `test-setup-first-20-EM-F1-audit-export`
- Reference: `test-setup-reference.txt`
- Snapshot: `snapshots/test-setup-readme-snapshot.md`

## 5. Commit Hash
- `5a14d52ff21a02ba806e7172da393bc717011483` (inferred)
- Source: `commit-hash.txt`

## 6. Run Command
- `python scripts/evaluation/export_eval_from_audit.py --audit-log data/processed/logs/audit.log --dataset-jsonl data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl --timestamp-prefix 2026-03-26T23: --run-id 20260326T233038Z --out artifacts/test-results/test-result-20260326T233038Z/eval_insuranceqa_results.jsonl --summary-out artifacts/test-results/test-result-20260326T233038Z/summary.json`
- Source: `run-command.txt`

## 7. Input / Dataset / Scope
- Dataset: `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl`
- Split: `test` (reconstructed first 20)
- Scope: first 20 samples
- Execution mode: post-hoc export from `data/processed/logs/audit.log`

## 8. Output Files
- `eval_insuranceqa_results.jsonl`
- `summary.json`
- `metadata.json`
- `commit-hash.txt`
- `run-command.txt`
- `pipeline-reference.txt`
- `test-setup-reference.txt`

## 9. Main Results
- Processed: `20`
- Failed: `0`
- Exact Match: `0.0`
- Token F1: `0.3699221767271236`

## 10. Notes / Open Issues / Next Step
- Result is audit-derived and does not replay online inference.
- `commit-hash.txt` was missing in the original artifact and is now inferred from repository timeline.

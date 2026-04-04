# Test Result: test-result-20260325T002026Z

## 1. Test Result ID
- Folder label: `test-result-20260325T002026Z`
- Internal run_id: `20260325T002226Z`

## 2. Objective
- Preliminary baseline 20-sample InsuranceQA EM/F1 run through live API evaluation.

## 3. Pipeline Version Used
- `pipeline-20260325`
- Reference: `pipeline-reference.txt`
- Snapshot: `snapshots/pipeline-readme-snapshot.md`

## 4. Test Setup Used
- `test-setup-first-20-EM-F1`
- Reference: `test-setup-reference.txt`
- Snapshot: `snapshots/test-setup-readme-snapshot.md`

## 5. Commit Hash
- `7d2adf72297540e1a86baf5d893ea82d96cee695`
- Source: `commit-hash.txt`

## 6. Run Command
- `/usr/bin/python3 scripts/eval_insuranceqa.py --api-url http://localhost:8000/api/ask --split test --max-samples 20 --count 20 --timeout 3600`
- Source: `run-command.txt`

## 7. Input / Dataset / Scope
- Dataset: `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl`
- Split: `test`
- Scope: first 20 samples
- Execution mode: external HTTP API (`/api/ask`)

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
- Token F1: `0.343531`

## 10. Notes / Open Issues / Next Step
- Folder timestamp (`002026Z`) differs from internal run_id timestamp (`002226Z`).
- For strict comparability, prefer the run_id value in machine-readable files.

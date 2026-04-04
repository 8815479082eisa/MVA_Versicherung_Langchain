# Test Result: test-result-20260326T213943Z

## 1. Test Result ID
- Folder label: `test-result-20260326T213943Z`
- Internal run_id: `20260326T213943Z`

## 2. Objective
- 20-sample InsuranceQA EM/F1 run after reranker fallback registration.

## 3. Pipeline Version Used
- `pipeline-20260326`
- Reference: `pipeline-reference.txt`
- Snapshot: `snapshots/pipeline-readme-snapshot.md`

## 4. Test Setup Used
- `test-setup-first-20-EM-F1`
- Reference: `test-setup-reference.txt`
- Snapshot: `snapshots/test-setup-readme-snapshot.md`

## 5. Commit Hash
- `5a14d52ff21a02ba806e7172da393bc717011483`
- Source: `commit-hash.txt`

## 6. Run Command
- `/usr/bin/python3 scripts/eval_insuranceqa.py --api-url http://localhost:8000/api/ask --split test --max-samples 20 --count 20 --timeout 3600`
- Source: `run-command.txt`

## 7. Input / Dataset / Scope
- Dataset: InsuranceQA test split
- Scope: first 20 samples
- Execution mode: external HTTP API (`/api/ask`)

## 8. Output Files
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
- Token F1: `0.351242`

## 10. Notes / Open Issues / Next Step
- No local `eval_insuranceqa_results.jsonl` was present in this artifact.
- Metrics are preserved from documented run evidence and summarized in `summary.json`.

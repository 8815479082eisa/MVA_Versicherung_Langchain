# Test Result: test-result-20260326T222652Z

## 1. Test Result ID
- Folder label: `test-result-20260326T222652Z`
- Internal run_id: `20260326T222652Z`

## 2. Objective
- 20-sample InsuranceQA EM/F1 run for local reproducibility using in-process FastAPI `TestClient`.

## 3. Pipeline Version Used
- `pipeline-20260326`
- Reference: `pipeline-reference.txt`
- Snapshot: `snapshots/pipeline-readme-snapshot.md`

## 4. Test Setup Used
- `test-setup-first-20-EM-F1-testclient`
- Reference: `test-setup-reference.txt`
- Snapshot: `snapshots/test-setup-readme-snapshot.md`

## 5. Commit Hash
- `5a14d52ff21a02ba806e7172da393bc717011483` (inferred)
- Source: `commit-hash.txt`

## 6. Run Command
- `python -m fastapi.testclient -> src.main:/api/ask, split=test, max-samples=20, count=20, seed=42`
- Source: `run-command.txt`

## 7. Input / Dataset / Scope
- Dataset: `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl`
- Split: `test`
- Scope: first 20 samples
- Execution mode: in-process FastAPI `TestClient`

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
- `commit-hash.txt` was missing in the original artifact and is now inferred from repository timeline.
- This run includes `source_ids` in the JSONL output schema.

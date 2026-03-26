# test-result-20260326T222652Z

## Overview
Local 20-sample InsuranceQA test artifact containing JSONL output and a machine-readable summary file.  
This result appears to be generated with a FastAPI `TestClient` path (in-process API testing), not clearly with an external HTTP server process.

## Timestamp / Run Identification
- test-result label: `test-result-20260326T222652Z`
- run_id: `20260326T222652Z` (from `summary.json` and JSONL lines)
- execution date/time: 2026-03-26 22:26:52 UTC (inferred from label/run_id format)

## Associated Pipeline
- Pipeline Label: `pipeline-20260326` (functional equivalent inferred from date, output characteristics, and nearby registrations)
- direct explicit pipeline reference file: not present in this folder

## Associated Test Setup
- Test-Setup Label: `test-setup-first-20-EM-F1` (functional equivalent inferred from command and metrics)
- Dataset: `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl` (from `summary.json`)
- Split: `test` (from `run-command.txt`)
- Sample Size: `20` (from `summary.json` and JSONL count)
- Metrics: `Exact Match (EM)`, `Token F1`

## Execution Summary
- Processed: `20`
- Failed: `0`
- Exact Match: `0.0`
- Token F1: `0.3699221767271236`

## Files
- Output JSONL: `eval_insuranceqa_results.jsonl` (includes `source_ids` fields and `run_id=20260326T222652Z`)
- Run command: `run-command.txt`
- Summary: `summary.json`
- Commit hash reference: not present
- Local README: `README.md` (this file)

## Technical Context
- `run-command.txt` indicates an in-process FastAPI TestClient invocation: `python -m fastapi.testclient -> src.main:/api/ask`.
- Output schema includes `source_ids`, which is not present in older registered result JSONL files.
- Reranking mode cannot be proven directly from files in this folder; not clearly identifiable from repository state for this exact run.

## Distinguishing Notes
- Highest Token F1 among current 20-sample artifacts (`0.369922...`).
- Only current result folder with a structured `summary.json`.
- Contains local JSONL output file directly in artifact folder (unlike `test-result-20260326T213943Z`).
- As of current state, Git tag linkage for this label is not present.

## Traceability
- Result-to-pipeline/test-setup mapping is reconstructed by evidence, but not explicitly committed via a dedicated registration commit message/tag.
- No `commit-hash.txt` in this folder; exact source commit is not clearly identifiable from repository state.

# test-result-20260325T002026Z

## Overview
Preliminary InsuranceQA evaluation artifact for a 20-sample API-based RAG run.  
Important: the folder label (`...002026Z`) differs from the internal run id (`...002226Z`).

## Timestamp / Run Identification
- test-result label: `test-result-20260325T002026Z`
- run_id (from JSONL): `20260325T002226Z`
- execution date/time: 2026-03-25 

## Associated Pipeline
- Pipeline Label: `pipeline-20260325`
- Pipeline documentation: `docs/pipelines/pipeline-20260325/README.md`

## Associated Test Setup
- Test-Setup Label: `test-setup-first-20-EM-F1`
- Dataset: InsuranceQA (`data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl`)
- Split: `test`
- Sample Size: `20`
- Metrics: `Exact Match (EM)`, `Token F1`

## Execution Summary
- Processed: `20`
- Failed: `0`
- Exact Match: `0.000000`
- Token F1: `0.343531`

## Files
- Output JSONL: `eval_insuranceqa_results.jsonl` (contains `run_id=20260325T002226Z`, idx 0..19)
- Run command: `run-command.txt`
- Commit hash reference: `commit-hash.txt` (`7d2adf72297540e1a86baf5d893ea82d96cee695`)
- Local README: `README.md`

## Technical Context
- Run executed via `/api/ask` endpoint using the evaluation script command in `run-command.txt`.
- This result is a small preliminary run (20 samples), not a full 1k benchmark.
- Reranker fallback status is not explicitly documented in this artifact and is therefore not clearly identifiable from repository state for this specific run.

## Distinguishing Notes
- Only result folder where directory timestamp and `run_id` are different (`002026Z` vs `002226Z`).
- Linked Git tag exists for `test-result-20260325T002226Z` (run id), not for the folder name.
- Older baseline among currently present artifacts.

## Traceability
- Registered in commit history around: `c07ade8` (registration commit message references `test-result-20260325T002226Z`).
- Associated pipeline tag: `pipeline-20260325`.
- Associated test-result tag: `test-result-20260325T002226Z`.

# test-result-20260326T213943Z

## Overview
20-sample InsuranceQA test result registered after enabling reranking fallback support in the pipeline (`FlagEmbedding` -> `sentence-transformers` CrossEncoder fallback path).

## Timestamp / Run Identification
- test-result label: `test-result-20260326T213943Z`
- run_id: `20260326T213943Z`
- execution date/time: 2026-03-26 21:39:43 UTC (inferred from run_id format)

## Associated Pipeline
- Pipeline Label: `pipeline-20260326`
- Pipeline documentation: `docs/pipelines/pipeline-20260326/README.md`

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
- Token F1: `0.351242`
- Index range: `idx 0..19` (from provided run evidence)

## Files
- Output JSONL: not present in this folder (canonical line-level evidence was provided from GPU runtime and documented in this README)
- Run command: `run-command.txt`
- Commit hash reference: `commit-hash.txt` (`5a14d52ff21a02ba806e7172da393bc717011483`)
- Local README: `README.md`

## Technical Context
- Run executed against `http://localhost:8000/api/ask` in remote `/workspace` runtime.
- Result context explicitly tied to reranker-fallback-capable pipeline generation.
- Earlier instability symptoms (`500` / connection refused) are not observed in this specific run (`Failed=0`).
- EM remains strict and zero; token overlap metric indicates partial semantic alignment.

## Distinguishing Notes
- This is the tagged post-reranker-fallback result (`test-result-20260326T213943Z`).
- Compared with `test-result-20260325T002026Z`, F1 is slightly higher (`0.351242` vs `0.343531`).
- Compared with `test-result-20260326T222652Z`, this artifact has stronger registry linkage (pipeline tag/commit registration), but no local JSONL file checked in.

## Traceability
- Registration commit: `4db5cdf` (plus later README refinement commit `53e8cfc`).
- Associated tags: `pipeline-20260326`, `test-result-20260326T213943Z`.
- Uses the shared setup `docs/test-setups/test-setup-first-20-EM-F1/README.md`.

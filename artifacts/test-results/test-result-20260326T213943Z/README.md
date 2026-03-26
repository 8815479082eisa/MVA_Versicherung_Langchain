# test-result-20260326T213943Z

## Pipeline Label
pipeline-20260326

## Test Setup Label
test-setup-first-20-EM-F1

## Execution Summary
- Processed: 20
- Failed: 0
- Exact Match: 0.000000
- Token F1: 0.351242
- run_id: 20260326T213943Z
- Index range: idx 0..19

## Output File
eval_insuranceqa_results.jsonl

## Notes
This result was generated after enabling the reranker fallback path (CrossEncoder) for cases where `FlagEmbedding` is unavailable.
The run was executed in the remote runtime (`/workspace`) against `http://localhost:8000/api/ask`.
The user provided record-level evidence (`idx` 0..19) from the GPU runtime for this run id.
The local workspace file `data/processed/eval_outputs/insuranceqa/eval_insuranceqa_results.jsonl` currently points to a different run and must not be used as the canonical source for this result.
Canonical source for this registered result is the GPU-side output belonging to `run_id=20260326T213943Z`.

## Record-Level Observations (from provided JSONL lines)
- Language/output issues previously reported (German generic fallback) are not visible in this run; responses are in English.
- No backend transport failures in this run (`Failed: 0`), so earlier `500/connection` instability is not present for this specific execution.
- EM is `0.0` although several answers are semantically close; this is expected for strict exact-match.
- Token F1 `0.351242` is plausible because many answers are verbose, templated, or partially off-context.
- Several responses show instruction/prompt leakage patterns (e.g. "TASK:", "THOUGHT:", generic step lists), which likely depresses F1.

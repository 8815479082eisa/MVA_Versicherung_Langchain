# Full Mode Test Failure Report

- Test folder: `test-result-20260524T211702Z-full-qa200-safety200-qwen35-9b-fast`
- Intended mode: `full`
- Intended scope: `200 QA queries + 200 safety cases`
- Started at: `2026-05-24T21:17:02Z`
- Status: `not completed`

## Intended Runtime Configuration

```env
ANSWER_MODEL=qwen3.5:9b
PREFERRED_ANSWER_MODEL=qwen3.5:9b
OLLAMA_MODEL=qwen3.5:9b

ROUTER_MODEL=functiongemma:270m
SELF_CHECK_MODEL=llama3.2:3b
QUERY_REWRITE_MODEL=llama3.2:3b
COMPRESSOR_MODEL=llama3.2:3b

QUERY_REWRITE_ENABLED=false
ENABLE_CONTEXT_COMPRESSION=false
ANSWER_STYLE=concise
```

The runtime config check resolved the answer model correctly:

```text
resolved_configured_answer_model=qwen3.5:9b
resolved_preferred_answer_model=qwen3.5:9b
answer_model_matches_preference=true
```

## Command

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_thesis.py `
  --mode full `
  --qa-mode live `
  --dataset-jsonl data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl `
  --max-samples 200 `
  --count 200 `
  --audit-log data/processed/logs/audit.log `
  --audit-limit 200 `
  --safety-file data/benchmarks/safety/thesis_safety_mix_200.jsonl `
  --support-threshold 0.2 `
  --out-dir artifacts/test-results/test-result-20260524T211702Z-full-qa200-safety200-qwen35-9b-fast
```

## What Happened

The evaluation process started but did not produce result files within the 6 hour timeout window.

No completed output files were generated:

- no `summary.md`
- no `summary.json`
- no `metadata.json`
- no `qa_items.jsonl`
- no `security_items.jsonl`

The audit log did not show new QA rows for `qwen3.5:9b`. The latest audit rows observed during troubleshooting still belonged to the previous successful `qwen2.5:7b-instruct` run.

## Direct Model Health Check

A direct Ollama request was sent to the same endpoint with a minimal prompt:

```text
model: qwen3.5:9b
prompt: Answer in one short sentence: what is insurance?
timeout: 90 seconds
```

Result:

```text
ERROR: The operation has timed out
```

This indicates that the remote Ollama endpoint did not return even a short response from `qwen3.5:9b` within 90 seconds.

## Diagnosis

The failure was caused by the `qwen3.5:9b` model being non-responsive or too slow on the configured remote Ollama endpoint:

```env
OLLAMA_BASE_URL=http://139.174.72.125:41144
```

The evaluation pipeline itself was not the primary failure point. The previous full mode run with `qwen2.5:7b-instruct` completed successfully under the same evaluation framework.

## Outcome

The `qwen3.5:9b` full mode run was aborted after timeout and the still-running Python evaluation process was stopped manually.

This run should not be used as a completed benchmark result.

## Recommendation

Use the previously successful fast configuration for full mode evaluation:

```env
ANSWER_MODEL=qwen2.5:7b-instruct
PREFERRED_ANSWER_MODEL=qwen2.5:7b-instruct
OLLAMA_MODEL=qwen2.5:7b-instruct
```

Before attempting another full run with `qwen3.5:9b`, first run a small direct model health check and a smoke evaluation, for example:

```powershell
# Direct Ollama model check should return quickly before starting full evaluation.
```

Only start a full `200 QA + 200 safety` run if the model responds reliably to short prompts.

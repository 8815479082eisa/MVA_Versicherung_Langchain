# test-setup-thesis-8-metrics

## Purpose
This setup is the reproducible thesis evaluation path for the unified 8-metric framework:

- Retrieval
- Generated Answer
- Citation
- Security

The full run is intended to be executed manually later in the GPU container. Codex should not run the full evaluation automatically.

## Runtime Configuration Priority
The runtime now resolves configuration in this order:

1. Explicit shell environment variables
2. `.env` values
3. Code defaults

This matters especially for answer-model selection. If `ANSWER_MODEL=qwen3.5:4b` is exported in the shell, `.env` must not overwrite it.

## Correct Backend Entrypoint
Start the backend with:

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Do not use `src.api.main:app` or `src.api.app:app` for the thesis run. The active FastAPI entrypoint is `src.main:app`.

## Recommended Thesis Eval Environment
Set the environment before starting the backend or running the evaluation:

```bash
export SAFETY_BACKEND=nemo
export NEMO_INPUT_ENABLED=true
export NEMO_CONTEXT_ENABLED=true
export NEMO_OUTPUT_ENABLED=true
export NEMO_ENFORCE_INPUT=true
export NEMO_ENFORCE_OUTPUT=false
export QUERY_REWRITE_ENABLED=false
export ANSWER_MODEL=qwen3.5:4b
export PREFERRED_ANSWER_MODEL=qwen3.5:4b
export OLLAMA_MODEL=qwen3.5:4b
export ROUTER_MODEL=functiongemma:270m
export INSURANCEQA_EXACT_MATCH_SHORTCUT=false
```

Alternative if output enforcement should stay enabled:

```bash
export NEMO_ENFORCE_OUTPUT=true
export SAFETY_MIN_GROUNDEDNESS=0.35
```

## Preflight Check
Before the backend start, verify the resolved runtime config without starting the pipeline:

```bash
python scripts/tools/print_runtime_config.py
```

The output includes:

- `ANSWER_MODEL`
- `PREFERRED_ANSWER_MODEL`
- `OLLAMA_MODEL`
- `ROUTER_MODEL`
- `SAFETY_BACKEND`
- `NEMO_ENFORCE_OUTPUT`
- `QUERY_REWRITE_ENABLED`
- `INSURANCEQA_EXACT_MATCH_SHORTCUT`
- `resolved_configured_answer_model`
- `resolved_preferred_answer_model`
- `answer_model_matches_preference`
- the source of the resolved answer model (`shell_env`, `dotenv`, `default`, `fallback`)

If the configured answer model does not match the preferred answer model, the output will also show a warning and any `.env` conflicts that were overridden by shell env.

## Health Check
After backend startup, verify the health endpoint:

```bash
curl http://localhost:8000/health
```

The response now includes:

- `configured_answer_model`
- `preferred_answer_model`
- `answer_model_matches_preference`
- `configured_answer_model_source`
- `query_rewrite_enabled`
- `nemo_enforce_output`
- `safety_backend`

## Recommended Full 200+200 Evaluation Command
Run the full thesis evaluation later in the GPU container:

```bash
python scripts/evaluation/evaluate_thesis.py \
  --mode full \
  --qa-mode live \
  --dataset-jsonl data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl \
  --max-samples 200 \
  --count 200 \
  --audit-log data/processed/logs/audit.log \
  --audit-limit 200 \
  --safety-file data/benchmarks/safety/thesis_safety_mix_200.jsonl \
  --support-threshold 0.2 \
  --out-dir "$RESULT_DIR"
```

## Output Artifacts
The run writes a result folder under `artifacts/test-results/` or the provided `--out-dir`.

Expected files:

- `metadata.json`
- `summary.json`
- `summary.md`
- `qa_items.jsonl`
- `qa_audit_rows.jsonl`
- `security_items.jsonl`
- `commit-hash.txt`
- `branch.txt`
- `run-command.txt`
- `setup/qa_sample.jsonl`
- `setup/security_cases.json`

## Reproducibility Notes
`metadata.json` and `summary.md` now record the runtime configuration needed for the thesis:

- configured and preferred answer model
- whether they match
- source of the configured answer model
- query rewrite enabled/disabled
- Nemo output enforcement enabled/disabled
- minimum groundedness threshold
- expected and processed QA counts
- expected and processed safety counts
- whether the run was complete or incomplete
- QA fallback reason counts from the selected audit rows

If a run processes fewer items than expected, `summary.md` marks it as incomplete instead of looking like a full 200+200 run.

## Harmless GPU Warning
The warning below is treated as non-critical when the backend otherwise starts normally:

```text
GPU device discovery failed: device_discovery.cc:92 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
```

This warning is typically emitted by external device-discovery code and does not come from the thesis evaluation logic itself. Missing `/sys/class/drm/...` paths should not be treated as a hard error for the evaluation run.

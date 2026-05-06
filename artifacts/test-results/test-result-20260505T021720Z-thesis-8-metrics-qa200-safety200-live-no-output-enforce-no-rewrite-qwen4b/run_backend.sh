#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source .venv/bin/activate
source "/workspace/artifacts/test-results/test-result-20260505T021720Z-thesis-8-metrics-qa200-safety200-live-no-output-enforce-no-rewrite-qwen4b/runtime-env.sh"

python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

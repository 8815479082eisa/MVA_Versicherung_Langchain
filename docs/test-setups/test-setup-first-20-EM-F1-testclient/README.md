# test-setup-first-20-EM-F1-testclient

## Dataset
InsuranceQA

## Split
test

## Sample Size
20

## Metrics
- Exact Match (EM)
- Token F1

## Execution Mode
In-process FastAPI `TestClient` invocation (`src.main:/api/ask`) instead of external HTTP service.

## Run Parameters
- split: test
- max-samples: 20
- count: 20
- seed: 42

## Typical Command Signature
`python -m fastapi.testclient -> src.main:/api/ask, split=test, max-samples=20, count=20, seed=42`

## Associated Test Results
- `artifacts/test-results/test-result-20260326T222652Z`

## Notes
- Intended for local reproducibility when server orchestration is not needed.
- Output schema in the recorded run includes `source_ids`.

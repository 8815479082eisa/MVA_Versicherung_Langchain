# Test Setups Overview

This index tracks all registered test setup variants and their linked test-result artifacts.

| Test Setup Label | Dataset | Split | Sample Size | Execution Mode | Associated Test Results |
|---|---|---|---:|---|---|
| `test-setup-first-20-EM-F1` | InsuranceQA | test | 20 | External HTTP API (`/api/ask`) | `test-result-20260325T002026Z`, `test-result-20260326T213943Z` |
| `test-setup-first-20-EM-F1-testclient` | InsuranceQA | test | 20 | In-process FastAPI TestClient | `test-result-20260326T222652Z` |
| `test-setup-first-20-EM-F1-audit-export` | InsuranceQA | test (reconstructed) | 20 | Post-hoc export from `audit.log` | `test-result-20260326T233038Z` |
| `test-setup-thesis-8-metrics` | InsuranceQA + direct-query safety | mixed | 20 QA + 20 safety | Unified thesis evaluator (`live` or `audit` QA + direct safety checker) | pending |

## Notes
- All listed setups use EM/F1 scoring semantics compatible with `scripts/evaluation/eval_insuranceqa.py`.
- Setup variants differ by execution path (live API vs in-process vs audit export), not by benchmark dataset.
- `test-setup-thesis-8-metrics` extends the benchmark scope beyond EM/F1 by combining retrieval, answer, citation, and security metrics in one run artifact.

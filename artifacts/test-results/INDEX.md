# Test Results Index

This index provides a compact, run-centric overview of all test executions under `artifacts/test-results/`.

| Test Result ID | Date (UTC) | Pipeline | Test Setup | Purpose | Available Files | Status |
|---|---|---|---|---|---|---|
| `test-result-20260325T002026Z` | 2026-03-25 | `pipeline-20260325` | `test-setup-first-20-EM-F1` | Baseline preliminary external API run (first 20) | README, metadata, command, commit hash, summary, JSONL, refs, snapshots | completed |
| `test-result-20260326T213943Z` | 2026-03-26 | `pipeline-20260326` | `test-setup-first-20-EM-F1` | Post-reranker-fallback external API run (first 20) | README, metadata, command, commit hash, summary, refs, snapshots | completed_with_documented_metrics |
| `test-result-20260326T222652Z` | 2026-03-26 | `pipeline-20260326` | `test-setup-first-20-EM-F1-testclient` | In-process FastAPI TestClient run (first 20) | README, metadata, command, commit hash, summary, JSONL, refs, snapshots | completed |
| `test-result-20260326T233038Z` | 2026-03-26 | `pipeline-20260326` | `test-setup-first-20-EM-F1-audit-export` | Audit log export to eval artifact (first 20 equivalent) | README, metadata, command, commit hash, summary, JSONL, refs, snapshots | completed_audit_export |
| `test-result-20260403T233950Z-safety-direct-query` | 2026-04-03 | `pipeline-20260327` | `test-setup-safety-direct-query-v1` | Direct-query safety validation (input attacks) | process report, metadata, command, commit hash, summary, responses, setup | completed |
| `test-result-20260404T235516Z-safety-direct-query` | 2026-04-04 | `pipeline-20260327` | `test-setup-safety-direct-query-v1` | Direct-query safety validation rerun | process report, metadata, command, commit hash, summary, responses, setup | completed |
| `test-result-20260405T213536Z-safety-direct-query-20-rerun` | 2026-04-05 | `pipeline-20260327` | `test-setup-safety-direct-query-v1` | Extended direct-query run (20 attacks) | metadata, command, responses, setup | completed |
| `test-result-20260405T224624Z-safety-direct-query-20-after-helper-fix` | 2026-04-05 | `pipeline-20260327` | `test-setup-safety-direct-query-v1` | Direct-query rerun after helper fix | metadata, responses, setup | completed |

## Structure Standard

Each run folder is self-contained and uses this structure:

- `README.md`
- `metadata.json`
- `run-command.txt`
- `commit-hash.txt`
- `summary.json`
- `eval_insuranceqa_results.jsonl` (if available)
- `pipeline-reference.txt`
- `test-setup-reference.txt`
- `snapshots/pipeline-readme-snapshot.md`
- `snapshots/test-setup-readme-snapshot.md`

## Naming Convention

Recommended naming for new runs:
- `test-result-{timestamp}-{short-purpose}`

Examples:
- `test-result-20260326T213943Z-baseline-first-20-em-f1`
- `test-result-20260327T101500Z-safety-layer-direct-query`

Existing folders keep their original names for backward compatibility.

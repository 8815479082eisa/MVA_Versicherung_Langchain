# EspoCRM MCP integration report

Date: 2026-07-26  
Repository: `MVA_Versicherung_Langchain_main`  
EspoCRM: 10.0.3  
Extension: MVA Insurance Data Model 1.0.0

## 1. Outcome

The synthetic insurance CRM integration is operational for read-only REST,
MCP, and CRM-only Agent queries.

- `mva-espocrm` and `mva-espocrm-db` are healthy.
- EspoCRM UI, `app-check`, and `db:check` passed.
- `MvaPolicy` and `MvaClaim` are installed and available.
- The dataset contains exactly 10 Contacts, 15 Policies, and 8 Claims.
- A second seed run created no duplicates.
- The temporary seed user is disabled.
- The final API user has read-only access to Contact, MvaPolicy, and MvaClaim.
- Backend health reported `crmEnabled=true`, `crmReady=true`, and
  `crmError=null`.
- Direct REST, all five CRM MCP tools, CRM-only routing, and forbidden-scope
  routing passed live validation.
- Retrieval-only and combined E2E generation did not pass on this machine.
  They are recorded as failures, not as completed or skipped.

## 2. Starting-state validation

The controlled phase began with:

| Check | Result |
|---|---|
| `mva-espocrm` | healthy |
| `mva-espocrm-db` | healthy |
| UI `http://localhost:8080` | HTTP 200 |
| `bin/command app-check` | all checks OK |
| `bin/command db:check` | OK |
| Currency | EUR only; rate 1 |
| Extension | installed, version 1.0.0 |
| `MvaPolicy`, `MvaClaim` metadata | available |
| Initial CRM records | 0 / 0 / 0 |
| Initial API users | 0 |
| Initial `CRM_ENABLED` | false |

No MariaDB query, volume removal, core EspoCRM modification, extension
reinstallation, document reindex, or write-capable Agent tool was used.

## 3. Seed result

The temporary API user `mva-seed-temp` used role `MVA Seed Temporary`.
For Contact, MvaPolicy, and MvaClaim it had only read and create access.
Edit, delete, stream, export, mass update, assignment, and unrelated entity
permissions were denied.

The seeder used `http://localhost:8080` and synthetic fixtures only.

| Entity | First run created | First run skipped | Second run created | Second run skipped | Final count |
|---|---:|---:|---:|---:|---:|
| Contact | 10 | 0 | 0 | 10 | 10 |
| MvaPolicy | 15 | 0 | 0 | 15 | 15 |
| MvaClaim | 8 | 0 | 0 | 8 | 8 |

Required scenario validation:

- Lara Neumann
- masked application output for `lara.neumann@example.test`
- `TEST-KFZ-2026-1001`
- Motor Insurance / Partial Coverage / Active
- deductible 150 EUR
- `TEST-CLM-2026-2001`
- Glass Damage / Under Review
- Policy-to-customer, claim-to-customer, and claim-to-policy links present

After seeding, the temporary key was removed from each execution environment
and `mva-seed-temp` was disabled. It is not used by the application.

## 4. Final read-only API access

Role: `MVA Agent Read Only`

Installed API username: `mva_rag_read_only`.
This differs only in spelling from the requested `mva-rag-readonly`; the role,
authentication type, and effective permissions are the requested ones.

Effective ACL for Contact, MvaPolicy, and MvaClaim:

| Permission | Value |
|---|---|
| Read | all |
| Create | no |
| Edit | no |
| Delete | no |
| Stream | no |
| Export | no |

Mass update, assignment, user, user calendar, portal, group email, follower,
message, audit, mention, data privacy, and lock permissions are also denied.
The API user is active, uses API-key authentication, and has only the expected
role.

The key is stored only in ignored `.env.crm`; it was never printed, copied to
the report, or committed.

## 5. Runtime and Docker

Local ignored configuration:

```dotenv
CRM_ENABLED=true
ESPOCRM_BASE_URL=http://espocrm
ESPOCRM_PUBLIC_URL=http://localhost:8080
ESPOCRM_API_KEY=<read-only key, never commit>
```

Validated health response:

```json
{
  "status": "ok",
  "pipelineReady": true,
  "pipelineInitializing": false,
  "pipelineInitError": null,
  "crmEnabled": true,
  "crmReady": true,
  "crmError": null
}
```

Docker-specific reproducibility fixes made during live startup:

- CPU-only PyTorch is installed so CUDA packages are not pulled.
- `build-essential` is installed because `annoy` needs a compiler.
- `pypdf` is declared because `PyPDFLoader` requires it.
- `transformers<5` is declared because FlagEmbedding failed with
  `transformers 5.x`.
- the existing host Hugging Face cache is mounted, avoiding a repeated
  4.3-GB `bge-m3` download;
- the healthcheck uses Python standard-library HTTP instead of absent `curl`;
- PDF hash keys use the filename, so Windows and Docker paths do not make
  unchanged PDFs appear stale.

The saved 28 PDF hashes matched all 28 current PDFs. Chroma remained at 8,551
records; no reindex was performed.

## 6. Live REST and security validation

Read operations with the final API key:

| Operation | Result |
|---|---|
| Find Lara Neumann | success |
| Read `TEST-KFZ-2026-1001` | success |
| Read `TEST-CLM-2026-2001` | success |

Safe negative permission checks used valid request bodies and non-existent
record identifiers where applicable:

| Operation | HTTP result |
|---|---:|
| Create Contact | 403 |
| Update MvaPolicy | 403 |
| Change MvaClaim | 403 |
| Delete Contact | 403 |
| Export Contacts | 403 |

Counts after the negative tests remained exactly 10 / 15 / 8.

## 7. Live MCP validation

The MCP server exposed exactly:

1. `find_customer`
2. `get_customer_policies`
3. `get_policy`
4. `get_customer_claims`
5. `get_claim_status`

All five live calls succeeded. Lara was found with masked email output, two
active policies were returned, `TEST-KFZ-2026-1001` contained the 150 EUR
deductible, and `TEST-CLM-2026-2001` returned `Under Review`.

The MCP adapter is read-only. It contains no create, update, delete, export,
bulk enumeration, or arbitrary-record tool.

## 8. End-to-end routing and live queries

Deterministic routing:

| Query | Planned mode |
|---|---|
| General partial-coverage question | retrieval-only |
| Lara active policies | CRM-only |
| Lara glass damage and claim status | combined |
| Export all customers and policies | denied |

Live results:

| Scenario | Result | Observed latency |
|---|---|---:|
| CRM-only | HTTP 200; correct Lara facts and both policies | 293.2 ms wall, 268.9 ms API |
| Forbidden export | HTTP 403; no disclosure | 15.9 ms wall |
| Retrieval-only | failed; full request exceeded 360 s | >360,000 ms |
| Combined | failed in benchmark with HTTP 500 | not measured successfully |

The CRM-only answer included `TEST-KFZ-2026-1001`, Partial Coverage, and the
150 EUR deductible. It also returned EspoCRM source objects.

The failed retrieval investigation found three independent runtime issues:

1. `transformers 5.x` was incompatible with FlagEmbedding. This was fixed by
   declaring `transformers<5`.
2. Docker initially pointed at container loopback for Ollama. The local
   ignored environment was corrected to `host.docker.internal:11434`.
3. The machine had insufficient immediately available RAM for the unchanged
   7B answer model while Docker and the embedding model were resident. After
   reclaiming WSL filesystem cache, a direct Qwen call succeeded in 19.35 s,
   but the full request still exceeded 360 s in a CPU-only Phi guardrail step.
   No LLM output-token limit is configured, so that step can run for minutes.

No Ollama model, retrieval parameter, reranker parameter, guardrail threshold,
PDF, embedding, or Chroma index was changed to force a pass.

## 9. Latency benchmark

Benchmark artifact:
`reports/espocrm_latency_benchmark.json`

Configuration:

- 5 warm-up calls
- 50 measured direct REST calls
- 50 measured persistent MCP calls
- 1 measured E2E call after an initial scenario call
- E2E timeout for this diagnostic run: 30 s

### Direct EspoCRM REST

| Metric | Value |
|---|---:|
| Cold | 71.436 ms |
| Minimum | 27.734 ms |
| p50 | 31.587 ms |
| p95 | 37.806 ms |
| Maximum | 42.573 ms |
| Mean | 32.381 ms |
| Success | 50 / 50 |
| Timeouts | 0 |

Target p95 below 500 ms: passed.

### Persistent MCP

| Metric | Value |
|---|---:|
| Process startup, one time | 4,890.353 ms |
| Cold call after startup | 103.463 ms |
| Minimum | 53.885 ms |
| p50 | 67.748 ms |
| p95 | 102.335 ms |
| Maximum | 138.824 ms |
| Mean | 72.309 ms |
| Success | 50 / 50 |
| Timeouts | 0 |

MCP overhead over direct REST:

| Metric | Value |
|---|---:|
| p50 | 36.161 ms |
| p95 | 64.529 ms |
| Mean | 39.928 ms |

Target additional MCP p95 below 100–200 ms: passed.

### Agent endpoint

- warm CRM-only benchmark call: 85.707 ms, 1 / 1 successful;
- independently observed CRM-only wall latency: 293.2 ms;
- retrieval-only: HTTP 500 in the benchmark after Ollama was stopped
  following the >360 s diagnostic timeout;
- combined: HTTP 500 for the same unavailable local LLM runtime;
- therefore no successful retrieval-only or combined p50/p95 is claimed.

## 10. Docker resource observations

Benchmark snapshot:

| Container | CPU | RAM | PIDs |
|---|---:|---:|---:|
| `mva-backend` | 67.37% | 83.75 MiB | 4 |
| `mva-espocrm` | 0.00% | 125.6 MiB | 11 |
| `mva-espocrm-db` | 0.02% | 159.2 MiB | 17 |

The backend snapshot occurred while failed RAG scenarios and worker reload were
being handled, so it is not a steady-state RAG memory measurement. During a
final fully loaded, healthy snapshot the backend used 1.612 GiB and 6.66% CPU.
EspoCRM used 126 MiB and 0.00% CPU; MariaDB used 159.2 MiB and 0.03% CPU.

The evidence therefore supports retaining EspoCRM: its measured request
latency and resource use are small relative to local embedding and LLM costs.
No Baserow A/B test is justified by these measurements.

## 11. Tests

Final combined command covered CRM fixtures, orchestration, MCP tools, REST
client behavior, routing, API routing, MCP lifecycle/connections, retrieval
hash portability, and the live EspoCRM test.

Result:

```text
40 passed, 0 failed, 0 skipped, 22 warnings in 50.57s
```

The warnings are third-party Pydantic/LangChain/NeMo deprecations. The live
EspoCRM test passed and was not skipped.

Additional live scripts:

- `scripts/seed_espocrm.py`: first and second runs passed;
- `scripts/test_mcp_crm_client.py`: five-tool discovery and all five calls
  passed;
- `scripts/benchmark_espocrm_mcp.py`: completed and wrote the JSON artifact.

## 12. Files changed in this phase

Key phase-specific changes:

- `.env.crm` (ignored): enabled CRM and retained the read-only key;
- `.env` (ignored): Docker-reachable local Ollama URL;
- `docker/docker-compose.yml`: CPU dependency installation, cache mount, and
  portable healthcheck;
- `requirements.txt`: `pypdf` and `transformers<5`;
- `src/api/rag_service.py`: portable PDF hash key;
- `src/main.py`: sanitized exception logging for unexpected API failures;
- `scripts/test_mcp_crm_client.py`: exact five-tool live validation;
- `scripts/benchmark_espocrm_mcp.py`: backend resources, configurable E2E
  timeout, and explicit failed-scenario reporting;
- `tests/unit/test_retrieval_service.py`: Windows/Docker hash regression;
- `reports/espocrm_latency_benchmark.json`: actual benchmark output;
- `reports/espocrm_mcp_integration_report.md`: this report;
- `docs/espocrm_mcp_integration.md`: operating documentation.

Existing unrelated user changes in the dirty worktree were preserved.

## 13. Remaining limitation and next action

No EspoCRM UI step remains. Keep `mva-seed-temp` disabled and keep
`.env.crm` untracked.

The only incomplete live acceptance steps are retrieval-only and combined E2E
generation. They require sufficient RAM/GPU capacity or an approved generation
runtime adjustment. On this machine, first add an explicit, thesis-approved
LLM output limit and a backend request timeout, or run the unchanged models on
a host with more free memory/GPU. Then rerun the two scenarios and the E2E
benchmark. Do not change CRM or switch to Baserow to address this LLM bottleneck.

## 14. LLM runtime stabilization addendum — 2026-07-26

This dated addendum preserves the previous baseline above.

| Validation item | Result |
|---|---|
| CRM-only live E2E | PASSED |
| Five-tool read-only MCP contract | PASSED |
| Five live write/export denials | PASSED |
| Seed user remains disabled | PASSED |
| Focused suite including live EspoCRM | PASSED |
| Retrieval-only complete E2E | FAILED |
| Combined complete E2E | FAILED |
| Combined HTTP 206 partial contract | PASSED |
| Chroma count 8,551 before/after | PASSED |

The final focused command was:

```powershell
docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests
```

Result:

```text
134 passed, 0 failed, 0 skipped, 226 warnings in 131.41s
```

CRM-only measured 167.260 ms in the final benchmark. Retrieval-only returned
HTTP 504 after 47,500.054 ms with `LLM_STAGE_TIMEOUT/self_check`. Combined
returned HTTP 206 partial after 45,551.799 ms; CRM facts were retained and the
unavailable knowledge result was not fabricated. EspoCRM used about
128.2–128.5 MiB RAM, while the backend used 1.93–2.161 GiB.

Full diagnostics, implementation details, commands, and limitations are in:

- `reports/llm_runtime_stabilization_report.md`
- `reports/llm_runtime_benchmark.json`
- `docs/llm_runtime_stabilization.md`

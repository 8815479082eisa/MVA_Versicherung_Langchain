# EspoCRM MCP integration

This document describes the installed read-only EspoCRM integration, its safe
operating procedure, and the validated live state on 2026-07-26.

## Architecture

```text
FastAPI /api/ask
  |
  +-- deterministic insurance query planner
       |
       +-- retrieval-only -> existing document RAG
       +-- CRM-only ------> persistent CRM MCP runtime
       +-- combined ------> CRM MCP + existing document RAG
       +-- denied --------> HTTP 403

CRM MCP runtime
  |
  +-- find_customer
  +-- get_customer_policies
  +-- get_policy
  +-- get_customer_claims
  +-- get_claim_status
       |
       +-- EspoCRM REST API (read-only API user)
```

The Agent has no generic EspoCRM client and no create, update, delete, export,
or bulk-list tool.

## Installed components

- EspoCRM 10.0.3
- MariaDB 11.4
- MVA Insurance Data Model 1.0.0
- Contact
- MvaPolicy
- MvaClaim
- synthetic fixtures under `data/synthetic`
- read-only REST client
- persistent stdio MCP server and client
- deterministic routing and FastAPI integration

Ports:

| Service | Port |
|---|---:|
| Backend | 8000 |
| Frontend | 5173 |
| EspoCRM | 8080 |
| Nginx | 80 / 443 |
| Local Ollama | 11434 |

## Safe configuration

Copy the example if `.env.crm` does not exist:

```powershell
Copy-Item .env.crm.example .env.crm
```

Configure the ignored file without printing the key:

```dotenv
CRM_ENABLED=true
ESPOCRM_BASE_URL=http://espocrm
ESPOCRM_PUBLIC_URL=http://localhost:8080
ESPOCRM_API_KEY=<read-only API key>
```

Rules:

- never commit `.env.crm`;
- never put the EspoCRM administrator password in application configuration;
- never use the disabled seed key for runtime access;
- never paste an API key into a report, test output, or chat;
- do not expose the CRM MCP server directly outside the backend process.

The current local `.env` uses:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Docker and the Windows host can both resolve this name in the validated Docker
Desktop setup. If Ollama runs on another machine, use that approved endpoint
instead.

## Start the stack

From `docker`:

```powershell
docker compose --env-file ..\.env.crm `
  -f docker-compose.yml `
  -f docker-compose.crm.yml `
  up -d
```

To recreate only the backend:

```powershell
docker compose --env-file ..\.env.crm `
  -f docker-compose.yml `
  -f docker-compose.crm.yml `
  up -d --force-recreate --no-deps backend
```

Do not use `docker compose down --volumes`; the CRM data is in named volumes.

The backend image command installs CPU-only PyTorch, `build-essential`, and
the declared requirements. The host Hugging Face cache is mounted at
`/root/.cache/huggingface`, so the existing 4.3-GB `bge-m3` cache is reused.
The first fresh container build is still slower than a prebuilt backend image.

## Verify the runtime

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}"
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:8080
docker exec mva-espocrm bin/command app-check
docker exec mva-espocrm bin/command db:check
```

Required health fields:

```json
{
  "crmEnabled": true,
  "crmReady": true,
  "crmError": null
}
```

`pipelineReady` describes the separate RAG warmup. A CRM-only query can remain
fast even though local LLM generation is slow.

## Roles and users

### Temporary seed role

Role: `MVA Seed Temporary`  
API user: `mva-seed-temp`

Contact, MvaPolicy, and MvaClaim:

- Read: all
- Create: yes
- Edit: no
- Delete: no
- Stream: no
- Export: no

All unrelated entity and special permissions are denied.

This user is currently disabled. Enable it only for a controlled reseed, use a
temporary process environment variable, and disable it immediately afterward.
Never store the temporary key in `.env.crm`.

### Final application role

Role: `MVA Agent Read Only`  
Installed API user: `mva_rag_read_only`

Contact, MvaPolicy, and MvaClaim:

- Read: all
- Create: no
- Edit: no
- Delete: no
- Stream: no
- Export: no

All unrelated entities and special permissions are denied. The installed
username uses underscores; it is functionally the requested
`mva-rag-readonly` account.

## Synthetic data

Current exact counts:

| Entity | Count |
|---|---:|
| Contact | 10 |
| MvaPolicy | 15 |
| MvaClaim | 8 |

The seeder is idempotent. The validated second run created 0 records and
skipped 10 Contacts, 15 Policies, and 8 Claims.

Required scenario:

- Lara Neumann
- `TEST-KFZ-2026-1001`
- Motor Insurance
- Partial Coverage
- Active
- deductible 150 EUR
- `TEST-CLM-2026-2001`
- Glass Damage
- Under Review

Controlled reseed command:

```powershell
$env:ESPOCRM_SEED_BASE_URL = "http://localhost:8080"
$env:ESPOCRM_SEED_API_KEY = "<temporary seed key>"
try {
  .\.venv\Scripts\python.exe scripts\seed_espocrm.py
}
finally {
  Remove-Item Env:ESPOCRM_SEED_API_KEY -ErrorAction SilentlyContinue
}
```

Disable `mva-seed-temp` in EspoCRM immediately afterward.

## Live validation

Run the exact MCP client:

```powershell
.\.venv\Scripts\python.exe scripts\test_mcp_crm_client.py
```

It asserts exact discovery of the five read-only tools and invokes all five
against Lara, her policy, and her claim.

Run the live test with the read-only key present only in the current process:

```powershell
$env:CRM_ENABLED = "true"
$env:ESPOCRM_PUBLIC_URL = "http://localhost:8080"
$env:ESPOCRM_API_KEY = "<read-only key>"
try {
  .\.venv\Scripts\python.exe -m pytest `
    tests\integration\test_espocrm_live.py -q
}
finally {
  Remove-Item Env:ESPOCRM_API_KEY -ErrorAction SilentlyContinue
}
```

Validated security results:

- Lara read: allowed;
- policy read: allowed;
- claim read: allowed;
- Contact create: HTTP 403;
- policy update: HTTP 403;
- claim change: HTTP 403;
- delete: HTTP 403;
- export: HTTP 403.

## End-to-end examples

```powershell
$body = @{
  question = "Which active policies does Lara Neumann have?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/ask `
  -ContentType application/json `
  -Body $body
```

Validated results:

| Scenario | Status |
|---|---|
| CRM-only Lara policies | HTTP 200 |
| Forbidden export | HTTP 403 |
| Retrieval-only | failed: >360 s local LLM timeout |
| Combined | failed: local LLM unavailable/HTTP 500 in benchmark |

The CRM-only request returned the correct two active Lara policies and took
293.2 ms wall time in the independent live check. The later warm benchmark
call took 85.707 ms.

The E2E retrieval failures do not indicate an EspoCRM bottleneck. The local
machine runs Ollama on four CPU cores, has tight available RAM, and has no
configured LLM output-token limit. Do not mark retrieval-only or combined E2E
as passed until they produce successful responses.

## Benchmark

Run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_espocrm_mcp.py
```

Useful controls:

```powershell
$env:CRM_BENCHMARK_CALLS = "50"
$env:CRM_BENCHMARK_WARMUP_CALLS = "5"
$env:CRM_E2E_BENCHMARK_CALLS = "1"
$env:CRM_E2E_BENCHMARK_TIMEOUT_SECONDS = "30"
```

Actual 2026-07-26 results:

| Metric | Result |
|---|---:|
| REST cold | 71.436 ms |
| REST p50 / p95 | 31.587 / 37.806 ms |
| REST success / timeout | 50/50 / 0 |
| MCP process startup | 4,890.353 ms |
| MCP cold call | 103.463 ms |
| MCP p50 / p95 | 67.748 / 102.335 ms |
| MCP success / timeout | 50/50 / 0 |
| MCP overhead p50 / p95 | 36.161 / 64.529 ms |
| Warm CRM-only E2E | 85.707 ms |

The JSON artifact is:
`reports/espocrm_latency_benchmark.json`.

In the final healthy snapshot EspoCRM used 126 MiB RAM and 0.00% CPU, MariaDB
used 159.2 MiB and 0.03% CPU, and the fully loaded RAG backend used 1.612 GiB
and 6.66% CPU. CRM REST and MCP latency therefore meet the engineering targets
and do not justify a switch to Baserow.

## Tests

The final focused live and regression run produced:

```text
40 passed, 0 failed, 0 skipped, 22 warnings in 50.57s
```

The warnings are third-party deprecations. The live EspoCRM test was executed
and passed; it was not skipped.

## Troubleshooting

### `crmReady=false`

1. Confirm `CRM_ENABLED=true`.
2. Confirm the Docker URL is `http://espocrm`.
3. Confirm the read-only API user is active.
4. Confirm only the ignored `.env.crm` holds the key.
5. Check sanitized backend logs; do not print environment values.

### Docker tries to reindex unchanged PDFs

The integration normalizes saved PDF hash keys to filenames, so Windows and
Docker paths compare correctly. Verify the saved and current hash sets before
starting. Do not delete Chroma or force a reindex as a shortcut.

### `XLMRobertaTokenizer ... prepare_for_model`

This indicates incompatible `transformers 5.x`. The requirements declare
`transformers<5`.

### Ollama cannot allocate a CPU buffer

This is a host RAM problem, not an EspoCRM problem. Stop unnecessary workloads
or use a machine with sufficient RAM/GPU. Do not silently change the thesis
models. Any output limit or timeout change must be approved and documented,
then the retrieval-only and combined acceptance tests must be rerun.

### Very slow full RAG response

The measured CRM contribution is below a tenth of a second when warm. In the
historical baseline, a Phi guardrail prompt could generate for minutes because
no stage-specific output limit was set. The stabilized runtime is documented
in `docs/llm_runtime_stabilization.md`; inspect its stage diagnostics before
changing any limit.

## Remaining work

No EspoCRM UI action remains. Keep the seed user disabled.

The local LLM runtime now has stage-specific output limits, timeouts, retry
limits, readiness fields, pure-retrieval diagnostics, and combined partial
semantics. See `docs/llm_runtime_stabilization.md`. This does not change the
EspoCRM model, data, five-tool read-only MCP contract, or CRM-only independence.
Live retrieval-only and combined status is recorded separately in
`reports/llm_runtime_stabilization_report.md`; do not switch CRM products to
solve an Ollama resource problem.

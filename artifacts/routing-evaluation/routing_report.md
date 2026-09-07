# Routing Evaluation Report

Generated: `2026-08-22T08:58:14.883684+00:00`
Dataset: `C:\Users\mirae\MVA_Versicherung_Langchain_main\data\benchmarks\routing\routing_eval_80.jsonl`
Cases: 80 (20 per route)

## Method

- `LOCAL` reproduces deterministic safety prechecks and the lexical route planner without external services.
- `API` calls `/api/ask`; route correctness is measured independently from downstream operational success.
- A denied request is operationally successful with HTTP 200 (safety fallback) or 403 (planner denial).
- Non-denied requests are operationally successful with HTTP 200 or 206.

## LOCAL evaluation

- Cases: 80
- Routing accuracy: 98.75%
- Macro F1: 98.75%
- Operational success rate: 100.00%
- Stage accuracy: 100.00%
- Mean latency: 0.282 ms

### Per-route metrics

| Route | Support | Correct | Precision | Recall | F1 | Operational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `retrieval_only` | 20 | 20 | 95.24% | 100.00% | 97.56% | 100.00% |
| `combined` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `denied` | 20 | 19 | 100.00% | 95.00% | 97.44% | 100.00% |

### Confusion matrix

| Expected / Actual | crm_only | retrieval_only | combined | denied | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 0 | 0 | 0 | 0 |
| `retrieval_only` | 0 | 20 | 0 | 0 | 0 |
| `combined` | 0 | 0 | 20 | 0 | 0 |
| `denied` | 0 | 1 | 0 | 19 | 0 |

Misclassified IDs: route-denied-002

Operational failure IDs: none

### Operational error counts

```json
{}
```

## API evaluation

- Cases: 80
- Routing accuracy: 98.75%
- Macro F1: 98.75%
- Operational success rate: 23.75%
- Stage accuracy: n/a
- Mean latency: 3929.913 ms

### Per-route metrics

| Route | Support | Correct | Precision | Recall | F1 | Operational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100.00% | 100.00% | 100.00% | 0.00% |
| `retrieval_only` | 20 | 20 | 95.24% | 100.00% | 97.56% | 0.00% |
| `combined` | 20 | 20 | 100.00% | 100.00% | 100.00% | 0.00% |
| `denied` | 20 | 19 | 100.00% | 95.00% | 97.44% | 95.00% |

### Confusion matrix

| Expected / Actual | crm_only | retrieval_only | combined | denied | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 0 | 0 | 0 | 0 |
| `retrieval_only` | 0 | 20 | 0 | 0 | 0 |
| `combined` | 0 | 0 | 20 | 0 | 0 |
| `denied` | 0 | 1 | 0 | 19 | 0 |

Misclassified IDs: route-denied-002

Operational failure IDs: route-crm-001, route-crm-002, route-crm-003, route-crm-004, route-crm-005, route-crm-006, route-crm-007, route-crm-008, route-crm-009, route-crm-010, route-crm-011, route-crm-012, route-crm-013, route-crm-014, route-crm-015, route-crm-016, route-crm-017, route-crm-018, route-crm-019, route-crm-020, route-rag-001, route-rag-002, route-rag-003, route-rag-004, route-rag-005, route-rag-006, route-rag-007, route-rag-008, route-rag-009, route-rag-010, route-rag-011, route-rag-012, route-rag-013, route-rag-014, route-rag-015, route-rag-016, route-rag-017, route-rag-018, route-rag-019, route-rag-020, route-combined-001, route-combined-002, route-combined-003, route-combined-004, route-combined-005, route-combined-006, route-combined-007, route-combined-008, route-combined-009, route-combined-010, route-combined-011, route-combined-012, route-combined-013, route-combined-014, route-combined-015, route-combined-016, route-combined-017, route-combined-018, route-combined-019, route-combined-020, route-denied-002

### Operational error counts

```json
{
  "CRM_REQUEST_FAILED": 40,
  "GUARDRAIL_INVALID_OUTPUT": 21
}
```

### API health snapshot

```json
{
  "reachable": true,
  "status_code": 200,
  "payload": {
    "status": "ok",
    "message": "Backend laeuft",
    "pipelineReady": true,
    "pipelineInitializing": false,
    "pipelineInitError": null,
    "crmEnabled": true,
    "crmReady": true,
    "crmError": null,
    "answerProvider": "openai",
    "ollamaReachable": false,
    "llmReady": false,
    "llmError": "Ollama readiness check failed (ConnectTimeout).",
    "answerModelReady": true,
    "guardrailModelReady": false,
    "embeddingReady": true,
    "retrievalReady": true,
    "insuranceqaExactMatchShortcut": false,
    "safetyEnabled": true,
    "safetyMode": "enforce",
    "answer_provider": "openai",
    "openai_api_key_configured": true,
    "configured_answer_model": "gpt-4o-mini",
    "preferred_answer_model": "gpt-4o-mini",
    "answer_model_matches_preference": true,
    "configured_answer_model_source": "dotenv",
    "selfCheckEnabled": false,
    "answerCompletenessEnabled": false,
    "query_rewrite_enabled": false,
    "nemo_enforce_output": true,
    "safety_backend": "nemo"
  }
}
```

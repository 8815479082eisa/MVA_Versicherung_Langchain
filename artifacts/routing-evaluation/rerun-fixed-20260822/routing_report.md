# Routing Evaluation Report

Generated: `2026-08-22T09:57:03.316900+00:00`
Dataset: `data\benchmarks\routing\routing_eval_80.jsonl`
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
- Operational success rate: 98.75%
- Downstream operational completion: 100.00%
- Stage accuracy: 100.00%
- Mean latency: 0.461 ms

### Per-route metrics

| Route | Support | Correct | Precision | Recall | F1 | Operational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `retrieval_only` | 20 | 20 | 95.24% | 100.00% | 97.56% | 100.00% |
| `combined` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `denied` | 20 | 19 | 100.00% | 95.00% | 97.44% | 95.00% |

### Confusion matrix

| Expected / Actual | crm_only | retrieval_only | combined | denied | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 0 | 0 | 0 | 0 |
| `retrieval_only` | 0 | 20 | 0 | 0 | 0 |
| `combined` | 0 | 0 | 20 | 0 | 0 |
| `denied` | 0 | 1 | 0 | 19 | 0 |

Misclassified IDs: route-denied-002

Operational failure IDs: route-denied-002

### Operational error counts

```json
{}
```

## API evaluation

- Cases: 80
- Routing accuracy: 98.75%
- Macro F1: 98.75%
- Operational success rate: 98.75%
- Downstream operational completion: 100.00%
- Stage accuracy: n/a
- Mean latency: 2412.921 ms

### Per-route metrics

| Route | Support | Correct | Precision | Recall | F1 | Operational |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `retrieval_only` | 20 | 20 | 95.24% | 100.00% | 97.56% | 100.00% |
| `combined` | 20 | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| `denied` | 20 | 19 | 100.00% | 95.00% | 97.44% | 95.00% |

### Confusion matrix

| Expected / Actual | crm_only | retrieval_only | combined | denied | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crm_only` | 20 | 0 | 0 | 0 | 0 |
| `retrieval_only` | 0 | 20 | 0 | 0 | 0 |
| `combined` | 0 | 0 | 20 | 0 | 0 |
| `denied` | 0 | 1 | 0 | 19 | 0 |

Misclassified IDs: route-denied-002

Operational failure IDs: route-denied-002

### Operational error counts

```json
{}
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
    "llmReady": true,
    "llmError": null,
    "answerModelReady": true,
    "guardrailModelReady": true,
    "ollamaGuardrailModelReady": false,
    "guardrailEmbeddingModel": "sentence-transformers/all-MiniLM-L6-v2",
    "guardrailEmbeddingDimension": 384,
    "guardrailModelError": null,
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

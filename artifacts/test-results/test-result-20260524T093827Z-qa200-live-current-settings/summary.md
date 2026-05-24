# Thesis Evaluation Summary

- Run ID: `20260524T093918Z`
- Generated at: `2026-05-24T09:39:18.646078+00:00`
- Run status: `complete`
- Answer model: `rnj-1:8b`
- Answer model source: `dotenv`
- Preferred answer model: `rnj-1:8b`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 0.9750 |
| Retrieval | Retrieval Context Precision | 0.6360 |
| Generated Answer | Exact Match | 0.0000 |
| Generated Answer | Token F1 | 0.2312 |
| Citation | Source Presence Rate | 0.8050 |
| Citation | Citation Support Rate | 0.7891 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: True
- Minimum groundedness: 0.3

## Safety Outcome Breakdown

- QA fallback reasons: low_groundedness=37, nemo_post_generation_runtime_error=1, pii_detected_in_answer=30, pii_detected_in_context=35, prompt_injection_signal_in_answer=1

## QA Diagnostics

- Requested QA items: 200
- Expected QA items: 200
- Selected QA items: 200
- Processed QA items: 200
- Failed QA items: 0
- Fallback answers: 39
- Mean inline citation coverage: 0.0660

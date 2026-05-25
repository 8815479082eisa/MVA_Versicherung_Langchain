# Thesis Evaluation Summary

- Run ID: `20260524T164212Z`
- Generated at: `2026-05-24T16:42:12.987391+00:00`
- Run status: `complete`
- Answer model: `qwen2.5:7b-instruct`
- Answer model source: `dotenv`
- Preferred answer model: `qwen2.5:7b-instruct`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 1.0000 |
| Retrieval | Retrieval Context Precision | 0.9180 |
| Generated Answer | Exact Match | 0.0000 |
| Generated Answer | Token F1 | 0.5561 |
| Citation | Source Presence Rate | 1.0000 |
| Citation | Citation Support Rate | 0.6983 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: True
- Minimum groundedness: 0.2

## Safety Outcome Breakdown

- QA fallback reasons: pii_detected_in_answer=1, pii_detected_in_context=9

## QA Diagnostics

- Requested QA items: 200
- Expected QA items: 200
- Selected QA items: 200
- Processed QA items: 200
- Failed QA items: 0
- Fallback answers: 0
- Mean inline citation coverage: 0.3300

## Security Diagnostics

- Expected safety cases: 200
- Processed safety cases: 200
- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

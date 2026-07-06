# Thesis Evaluation Summary

- Run ID: `20260525T235304Z`
- Generated at: `2026-05-25T23:53:04.671447+00:00`
- Run status: `complete`
- Answer model: `qwen2.5:7b-instruct`
- Answer model source: `dotenv`
- Preferred answer model: `qwen2.5:7b-instruct`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 0.9750 |
| Retrieval | Retrieval Context Precision | 0.6350 |
| Generated Answer | Exact Match | 0.0000 |
| Generated Answer | Token F1 | 0.1730 |
| Citation | Source Presence Rate | 0.9100 |
| Citation | Citation Support Rate | 0.5971 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: True
- Minimum groundedness: 0.2

## Safety Outcome Breakdown

- QA fallback reasons: low_groundedness=18, pii_detected_in_answer=15, pii_detected_in_context=43

## QA Diagnostics

- Requested QA items: 200
- Expected QA items: 200
- Selected QA items: 200
- Processed QA items: 200
- Failed QA items: 0
- Fallback answers: 18
- Mean inline citation coverage: 0.7500

## Security Diagnostics

- Expected safety cases: 200
- Processed safety cases: 200
- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

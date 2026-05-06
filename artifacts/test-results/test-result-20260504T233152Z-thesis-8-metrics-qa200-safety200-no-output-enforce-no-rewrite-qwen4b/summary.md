# Thesis Evaluation Summary

- Run ID: `20260504T233307Z`
- Generated at: `2026-05-04T23:33:07.873737+00:00`
- Answer model: `functiongemma:270m`
- Preferred answer model: `qwen3.5:4b`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 1.0000 |
| Retrieval | Retrieval Context Precision | 0.4400 |
| Generated Answer | Exact Match | 0.0500 |
| Generated Answer | Token F1 | 0.2481 |
| Citation | Source Presence Rate | 0.3000 |
| Citation | Citation Support Rate | 0.3000 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: True
- Minimum groundedness: 0.7

## Safety Outcome Breakdown

- QA fallback reasons: low_groundedness=14

## QA Diagnostics

- Processed QA items: 20
- Failed QA items: 0
- Fallback answers: 14
- Mean inline citation coverage: 0.0000

## Security Diagnostics

- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

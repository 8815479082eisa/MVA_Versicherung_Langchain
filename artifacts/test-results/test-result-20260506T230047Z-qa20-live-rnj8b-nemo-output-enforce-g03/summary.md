# Thesis Evaluation Summary

- Run ID: `20260506T230052Z`
- Generated at: `2026-05-06T23:00:52.482586+00:00`
- Run status: `complete`
- Answer model: `rnj-1:8b`
- Answer model source: `dotenv`
- Preferred answer model: `rnj-1:8b`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 1.0000 |
| Retrieval | Retrieval Context Precision | 0.4400 |
| Generated Answer | Exact Match | 0.1500 |
| Generated Answer | Token F1 | 0.6782 |
| Citation | Source Presence Rate | 0.9000 |
| Citation | Citation Support Rate | 0.8274 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: True
- Minimum groundedness: 0.3

## Safety Outcome Breakdown

- QA fallback reasons: low_groundedness=2

## QA Diagnostics

- Requested QA items: 20
- Expected QA items: 20
- Selected QA items: 20
- Processed QA items: 20
- Failed QA items: 0
- Fallback answers: 2
- Mean inline citation coverage: 0.0638

## Security Diagnostics

- Expected safety cases: 200
- Processed safety cases: 200
- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

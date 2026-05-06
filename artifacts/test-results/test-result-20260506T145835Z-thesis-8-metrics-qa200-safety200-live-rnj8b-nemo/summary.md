# Thesis Evaluation Summary

- Run ID: `20260506T145840Z`
- Generated at: `2026-05-06T14:58:40.915107+00:00`
- Run status: `complete`
- Answer model: `rnj-1:8b`
- Answer model source: `shell_env`
- Preferred answer model: `rnj-1:8b`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 0.9900 |
| Retrieval | Retrieval Context Precision | 0.4680 |
| Generated Answer | Exact Match | 0.0550 |
| Generated Answer | Token F1 | 0.6012 |
| Citation | Source Presence Rate | 1.0000 |
| Citation | Citation Support Rate | 0.9050 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: False
- Minimum groundedness: 0.7
- .env conflicts overridden by shell env: ['PDF_DIRECTORY']

## Safety Outcome Breakdown

- QA fallback reasons: pii_detected_in_context=4

## QA Diagnostics

- Requested QA items: 200
- Expected QA items: 200
- Selected QA items: 200
- Processed QA items: 200
- Failed QA items: 0
- Fallback answers: 0
- Mean inline citation coverage: 0.0458

## Security Diagnostics

- Expected safety cases: 200
- Processed safety cases: 200
- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

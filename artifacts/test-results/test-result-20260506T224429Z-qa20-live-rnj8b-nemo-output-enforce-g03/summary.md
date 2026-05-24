# Thesis Evaluation Summary

- Run ID: `20260506T224434Z`
- Generated at: `2026-05-06T22:44:34.703022+00:00`
- Run status: `complete`
- Answer model: `rnj-1:8b`
- Answer model source: `shell_env`
- Preferred answer model: `rnj-1:8b`
- QA mode: `live`

## Metric Summary

| Area | Metric | Value |
|---|---|---:|
| Retrieval | Retrieval Support Hit Rate | 1.0000 |
| Retrieval | Retrieval Context Precision | 0.4400 |
| Generated Answer | Exact Match | 0.1000 |
| Generated Answer | Token F1 | 0.6184 |
| Citation | Source Presence Rate | 1.0000 |
| Citation | Citation Support Rate | 0.9054 |
| Security | Attack Block Rate | 1.0000 |
| Security | Benign Allow Rate | 1.0000 |

## Runtime Config

- Query rewrite enabled: False
- Output enforcement enabled: False
- Minimum groundedness: 0.7
- .env conflicts overridden by shell env: ['NEMO_ENFORCE_OUTPUT', 'PDF_DIRECTORY', 'SAFETY_MIN_GROUNDEDNESS']

## QA Diagnostics

- Requested QA items: 20
- Expected QA items: 20
- Selected QA items: 20
- Processed QA items: 20
- Failed QA items: 0
- Fallback answers: 0
- Mean inline citation coverage: 0.0381

## Security Diagnostics

- Expected safety cases: 200
- Processed safety cases: 200
- Attack cases: 70
- Benign cases: 130
- False negatives: 0
- False positives: 0

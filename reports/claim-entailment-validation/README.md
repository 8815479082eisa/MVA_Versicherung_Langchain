# Claim entailment validation — 2026-09-06

- Related unit suites: 144 passed. After the final quote-validation hardening, the 25 evaluator/adapter tests passed again.
- Live provider regressions: 11/11 expected outcomes (see live-regressions.json).
- Original answer replay: 19 supported atomic claims, score 1.0.
- Full API request: HTTP 200; 19 supported claims, score 1.0; no contradiction cap or output fallback (see api-response.json).
- API groundedness stage: approximately 32.2 seconds; reranking: 20.4 seconds; answer generation: 3.6 seconds.
- Provider boundary is mocked in unit tests. Live regressions and API validation use the configured OpenAI judge.
- These are targeted regressions, not a calibrated estimate of general semantic evaluation accuracy.

# Targeted RAG Fixes

## Causes and Changes

- `src/core/answer_completeness.py`: quoted-clause questions inherited unrelated product requirements from product words in document titles. Do not inject product boilerplate for these requests; retain explicit CRM requirements.
- `src/core/insurance_tool_routing.py`: brochure/STI source names were not distinguished reliably. Resolve the explicit household brochure, services brochure and assistance STI requests to their respective files. Ignore quoted clause text when inferring a source, so "electrical assistance" does not override an explicitly named household document.
- `src/api/rag_service.py`: recovery only recognized the legacy `low_groundedness` reason. Recognize claim-aware reasons; retry technical/uncertain judgments separately from targeted answer repair. Keep unrelated safety blockers. Give the generator concise question-specific instructions about table rows, conditions and citations.
- `src/core/claim_groundedness.py`: select judging evidence from the sources actually cited by the answer; validate source/page bindings. Retry invalid judge quotes once, without relaxing provenance validation. Preserve negation/conditional operators and numeric boundaries when tolerating OCR differences.
- `src/guardrails/integrations/nemo_official.py`: missing NeMo output actions previously caused HTTP 503. Execute the same output safety action directly when the dialog/embedding path does not return an action, and cache this recovery path for that runtime. No unvalidated allow path was added.
- `src/guardrails/integrations/nemo_actions.py`: redaction no longer prevents escalation to fallback when factual support fails.

## Tests

- 105 relevant unit tests passed, plus 18 subtests, on the final code.
- 35 NeMo runtime tests passed in the preceding run; the subsequent change only tightened quote numeric boundaries, covered by the final unit run.
- New regressions cover source selection, quoted-clause boilerplate, retry versus repair, fabricated answer citations, source-scoped judging and exact judge-quote repair.
- Existing quote tests now reject changing `if` to `unless` and numeric prefix matches such as `10` versus `100`.
- One diagnostic test expected the obsolete `low_groundedness` reason. Updated it to `groundedness_sensitive_unsupported`; the unsupported-answer fallback assertion remains unchanged.
- `git diff --check` passed (only Windows line-ending warnings).

## Evaluation Protocol and Limits

The saved 100-question dataset is replayed through the actual `/api/ask` endpoint on port 8001. The first 15 results used six workers, then collection resumed with two workers after request timeouts. After result 17, quoted clause text was excluded from source inference (the source selections for the first 17 questions are unchanged). Completed results, including errors, are preserved; no completed result is silently retried or replaced. The client timeout is 240 seconds and the application's own request timeout is unchanged. Raw responses preserve diagnostics. Earlier interrupted development runs are separate artifacts and are excluded from `responses.jsonl` and its summary.

The historical baseline contains 100 HTTP 503 responses. These are service failures, not 100 confirmed hallucinations. The baseline ran sequentially, so latency is not directly comparable with the concurrent rerun.

`report.md`, `review.jsonl` and `questions-and-answers.csv` contain the completed rerun. `answered` is a runtime category, not a factual correctness label. No hallucination rate is inferred from the production groundedness evaluator alone.

Remaining risks include claim-extraction audit failures, uncertain quote verification, imperfect table interpretation and request timeouts. The missing NeMo ONNX cache artifact is not repaired by these code changes; the recovery path applies the same safety checks without depending on that output-dialog artifact. An independent factual audit is still required before claiming an unchanged or reduced hallucination rate.

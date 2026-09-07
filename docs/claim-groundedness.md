# Claim entailment groundedness

The active output guardrail uses `claim_entailment_v1`. The v4/v5 lexical
scorers remain available for historical experiments but are not a runtime fallback.

## Evaluation

1. Preserve answer sentences and list headings, then extract self-contained atomic
   claims using structured model output. Every answer unit must be accounted for.
2. Check preservation of sensitive values and independently audit completeness of
   extraction **without evidence**. This prevents confusing a false answer with
   an unfaithful extraction of that answer.
3. Rank overlapping evidence windows using subject/location matches, lexical
   relevance and the existing semantic retrieval/reranker order. Ranking values
   are never passed into the final groundedness score.
4. Judge each claim against selected evidence and full context for exceptions.
   Return `supported`, `contradicted`, `insufficient_evidence`, or `unknown`.
   Require verified quotes and explicit checks for all 12 sensitive categories.
5. Validate quote provenance with OCR-normalized local-span alignment, then check
   normalized numeric/currency/date facts, duration
   units, location anchors, coverage taxonomy, structured CRM fields and scoped
   polarity. Missing values withhold support; they do not prove contradiction.
   Conditions, exclusions and product semantics also require explicit semantic
   checks. Keyword equality is not used to establish those relationships.

The score is the fraction of atomic claims supported. A high-confidence
contradiction requires a model verdict with confidence >= 0.85, matched subject
and scope, verified evidence, and an explicit conflicting sensitive check.
Only this case applies the 0.25 cap. Missing location evidence downgrades a
contradiction to insufficient evidence. All atomic claims must be supported for
the active safety gate to allow the answer; the former v5 calibration threshold
does not alone establish acceptance. Unknown and insufficient evidence receive
no contradiction cap, but also receive no support credit.

## Runtime

The judge uses the existing OpenAI Responses adapter and credentials, with strict
JSON schema and `store=false`. `GROUNDEDNESS_MODEL` optionally overrides the
configured answer model (currently `gpt-4o-mini`). Three bounded model calls
perform extraction, extraction audit and entailment. Their shared budget is
at most 80 seconds and is below the configured guardrail timeout (90 seconds).
Minor PDF spacing, hyphenation and narrowly paraphrased connective wording do not
invalidate an otherwise traceable citation. Fabricated sensitive values and unrelated
quotes still fail provenance. Transport failures, malformed output, incomplete extraction or exceeded bounds
produce `unknown`; lexical scoring never substitutes for the failed judge.

Diagnostics include each claim, relation, confidence, verified evidence IDs and
quotes, 12 sensitive checks, deterministic issues, support fraction and caps.
The model confidence is a self-reported gate, not a calibrated probability.

## Validation and limits

Unit tests isolate the provider, verify semantic labels control scoring, preserve
true contradictions, reject fabricated citations, retain list scope, normalize
OCR artifacts, and veto missing sensitive facts. Existing NeMo/PII tests mock
only the provider boundary, retaining the actual evaluator and safety gate.

Live tests cover Kosovo paraphrases and contradiction, repair paraphrases,
amount/currency/duration conflicts, conditional paraphrases, dropped conditions,
and a different geographic scope. A correct answer rejected by the previous
guardrail was also replayed against its eight original retrieved chunks.

This is model-based entailment, not a formally verified NLI system. Additional
provider calls add latency/cost. The new score is not interchangeable with the
historical v5 metric; evaluation datasets and thresholds must be versioned.
Broader multilingual and adversarial calibration remains necessary before
claiming a general accuracy or false-rejection rate.

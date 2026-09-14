# Combined RAG + CRM Fallback — Diagnostic & Improvement Plan

> Working document. Keep it in the repo (`docs/COMBINED_FALLBACK_PLAN.md`) and update it
> at the end of every working session. It is the hand-off state between sessions:
> a new session should be able to read **only this file** and continue.
>
> Code map below was read on branch `Abschluss-Arbeit`. Local working copy is
> `MVA_Versicherung_Langchain_main` — **verify line numbers before trusting them**;
> the file/function names are what matter, not the numbers.

---

## 0. Scope

**Problem statement:** In `combined` mode (CRM + document retrieval together) the pipeline
returns a fallback / partial answer for many queries where a correct answer should be possible.

**Out of scope until the fallback is understood:** refactoring, performance work, prompt
rewriting, model swaps. Do not mix a correctness investigation with an optimization pass —
it destroys the ability to attribute any change to any outcome.

---

## 1. Code map of the combined path

Entry: `src/main.py` — `/answer` handler.

| Step | Location | What happens |
|---|---|---|
| 1. Route | `src/core/insurance_tool_routing.py` → `plan_insurance_query()` | Keyword/regex based. Emits `QueryMode.{RETRIEVAL_ONLY, CRM_ONLY, COMBINED, DENIED}` |
| 2. CRM guard | `main.py` `if query_plan.uses_crm:` | 503 if `_crm_runtime_ready` is false |
| 3. CRM exec | `main.py` `_execute_crm_with_diagnostics(query_plan)` | Early return on `AMBIGUOUS_CUSTOMER` / `CUSTOMER_NOT_FOUND` |
| 4. CRM select | `select_crm_context()` → `_crm_context_documents()` | CRM rows become `Document` objects |
| 5. Query build | `build_knowledge_query()` + `enrich_knowledge_query(product_hints=...)` | Retrieval query is rewritten using CRM hints |
| 6. RAG | `_run_rag_with_total_timeout(..., additional_context_docs=crm_context_documents)` | Hands CRM docs into `src/api/rag_service.py` |
| 7. Context merge | `rag_service.py` `context_docs = reranked_docs + additional_context_docs` | **CRM docs are appended AFTER rerank** |
| 8. Context rail | `safety_checker.check_context_safety(context_docs)` | Sees CRM docs, i.e. sees PII |
| 9. Generation | | |
| 10. Post rail + groundedness | `check_*` → `action in {"block","fallback"}` | `SETTINGS.safety.min_groundedness` |
| 11. Completeness | `main.py` `_answer_completeness_warning()` | Downgrades to `status="partial"`, HTTP 206 |

Fallback texts are category-specific and are the cheapest signal available:
`SETTINGS.safety.{security,grounding,pii,context}_fallback_text` (see `rag_service.py`
`safety_fallback_texts`). **Which text comes back tells you which rail fired.**

---

## 2. Ranked hypotheses

Each is falsifiable from diagnostics alone. Do not fix anything before one is confirmed.

**H1 — Context rail flags CRM PII (highest prior).**
Step 7 appends CRM documents containing names, emails, dates of birth, addresses, policy
numbers into `context_docs`, and step 8 runs the context safety rail over exactly that list.
In `retrieval_only` mode the rail never sees personal data; in `combined` it always does.
*Signature:* fallback text == `context_fallback_text` or `pii_fallback_text`;
`safetyDecision.reasonCode` points at the context/PII stage.

**H2 — Groundedness threshold is invalid for mixed evidence.**
`min_groundedness` and `config/groundedness_calibration.json` were calibrated on document-only
runs (see `scripts/calibrate_*groundedness*`). A combined answer asserts CRM facts that do not
appear in any retrieved PDF chunk, so claim-support scoring drops below threshold and the
answer is replaced by `grounding_fallback_text`.
*Signature:* fallback text == `grounding_fallback_text`; `safetyDecision.groundedness` just
under `threshold`; answer content was actually correct.

**H3 — `fail_closed` converts infrastructure flakiness into fallback.**
Every rail has `except Exception: if fail_closed and is_active: SafetyResult(action="fallback")`.
Combined mode has a larger context and one extra external call, so guardrail LLM timeouts are
more likely. A timeout is then indistinguishable from a real safety decision.
*Signature:* `stage`/`error_type` present in `safetyDecision` details; latency near the
guardrail timeout; non-deterministic — same query sometimes works.

**H4 — CRM docs bypass rerank and get truncated at context assembly.**
They are concatenated after rerank (step 7), so any downstream token/`top_k` budget cut may
drop precisely the CRM evidence the answer needs, leaving the answer ungrounded.
*Signature:* `crmContextSummary` diagnostic shows the facts, final `sourceIds` do not.

**H5 — Router mislabels queries.**
`plan_insurance_query()` is regex/keyword based. A query needing both may land as
`CRM_ONLY` or `RETRIEVAL_ONLY`.
*Signature:* `diagnostics.route` disagrees with the expected route.

**H6 — Completeness heuristic over-triggers.**
`_answer_completeness_warning()` downgrades good answers to `partial`/206.
*Signature:* HTTP 206 with a non-empty, correct `answer`.

---

## 3. Phase 1 — Reproduce and classify (NO code changes)

The repo already has strong instrumentation (`diagnostics.record_evidence`,
`src/core/runtime_diagnostics.py`, `artifacts/test-results/`). **Do not add logging before
reading what already exists.**

1. Build a fixed set of ~10 failing combined queries. Store as
   `tests/fixtures/combined_failing_queries.json` (query, expected route, expected facts).
   This set never changes during the investigation — it is the ruler.
2. Write `scripts/tools/triage_combined.py`: POST each query to `/answer`, and write ONE row
   per query to `artifacts/test-results/combined-triage.csv`:
   `query, http_status, route, reason_code, fallback_category, groundedness, threshold,
   guardrail_stage, error_type, latency_ms, n_sources, crm_docs_in_final_sources (bool)`
3. Run it. **Read only the CSV.** Never paste raw logs or full JSON diagnostics into a session.
4. Classify each row against H1–H6. Write the counts into section 7 below.

**Exit criterion:** every failing query is assigned to exactly one hypothesis, with evidence.
If a hypothesis has zero rows, drop it. Fix in descending order of row count.

---

## 4. Phase 2 — Fix, one hypothesis at a time

For each confirmed hypothesis, in its own branch and its own session:

1. Write a failing unit test first, in `tests/unit/`, with CRM data mocked — no live Ollama,
   no live EspoCRM. Fast tests are what make iteration cheap.
2. Make the minimal change.
3. Re-run `pytest tests/unit -q` and the triage script.
4. Record before/after counts in this document.
5. Commit with the hypothesis id in the message: `H1: scope context rail to non-CRM docs`.

Likely shapes of the fixes (do not implement before confirming):

- **H1:** the context rail should treat CRM documents as an authorized evidence class, not as
  leaked PII — tag CRM `Document.metadata` with a provenance marker and exempt/downgrade it
  in the rail rather than disabling the rail.
- **H2:** groundedness must score CRM claims against CRM evidence and document claims against
  document evidence, then combine — a single threshold across both evidence types cannot be
  correct. Recalibrate with `scripts/calibrate_extended_groundedness_threshold.py` on a
  combined-mode set, and keep the document-only threshold intact.
- **H3:** separate `action="fallback"` from `action="error"`. A rail that crashed or timed out
  must surface as an operational error with its own reason code, never as a safety verdict.
  This is also a thesis-grade argument: fail-closed is correct, but it must be *attributable*.
- **H4:** give CRM docs a reserved budget in context assembly so they cannot be evicted.
- **H5:** expand routing tests in `tests/unit/test_insurance_tool_routing.py` with the real
  failing queries; consider an LLM-assisted router only if the regex approach plateaus.
- **H6:** tighten the completeness heuristic to fire only when required facts are absent.

---

### 4.1 F1 — post stage falls back on a groundedness score of 1.0 (C1, C6)

Diagnosed, not implemented. **F1 is misfiled as H2. It is an H4-family defect:** CRM evidence is
evicted — not at context assembly, but *inside the groundedness evaluator*. The threshold is
never the deciding factor, so recalibrating `min_groundedness` would not move a single case.

#### Mechanism

| # | Location | What happens |
|---|---|---|
| 1 | `src/api/rag_service.py:4916` | `context_docs = reranked_docs + additional_context_docs` — CRM docs **are** present. Correct. |
| 2 | `src/api/rag_service.py:4997` | `check_answer_safety(original_query, context_docs, answer)` — evaluator receives them. Correct. |
| 3 | `src/core/claim_groundedness.py:320` | Citation regex is **PDF-only**: `r"\[([^\[\],]+\.pdf),\s*page\s+(\d+)\]"`. The generator cites CRM facts as `[CRM: TEST-KFZ-2026-1001]`, which this pattern cannot see. |
| 4 | `src/core/claim_groundedness.py:331-348` | `doc_label(doc)` → `(source_filename, page)`. A CRM `Document` has no `.pdf` source and no page, so its label can never be a member of `labels`; line 346 returns `False` for the CRM index. |
| 5 | `src/core/claim_groundedness.py:600-606` | `allowed_docs` is built by calling `_answer_citations_match` once per doc index. The CRM index fails, so **every CRM evidence window is filtered out of `candidates`** before `rank_evidence` (line 607). |
| 6 | `src/core/claim_groundedness.py:965, 974` | Judge never sees the CRM row → CRM claims return `relation="unknown"`, `provenance_valid=False`, and — being amounts/status — `is_sensitive_unsupported=True`. |
| 7 | `src/core/claim_groundedness.py:762-764` | `decided_count` excludes `unknown`. `score = supported / decided`. **The 1.0 is an artifact of the missing claims, not a sign of health.** |
| 8 | `src/guardrails/integrations/nemo_actions.py:1186-1195` | `sensitive_unsupported_count > 0` hard gate → `allow=False`, `action="fallback"`. It is an `elif` chain; the score comparison at `nemo_actions.py:1224-1237` is **never reached**. |
| 9 | `src/api/rag_service.py:5233-5240` | `if not safety_post_result.allow:` → answer replaced by `grounding_fallback_text`, `sources = []` (5236), decision `post_fallback`. |

Raw evidence (`artifacts/test-results/combined-triage/raw/`):

| Case | score | relation counts | sens_unsup | prov_fail | reason code |
|---|---|---|---|---|---|
| C1 | 1.0 | `{supported: 2, unknown: 3}` | 3 | 3 | `groundedness_sensitive_unsupported` |
| C6 | 1.0 | `{supported: 2, unknown: 2}` | 2 | 2 | `groundedness_sensitive_unsupported` |

Every `unknown` row is a CRM fact (`"The deductible recorded for policy TEST-KFZ-2026-1001 is
150 EUR"`, `"Amira Hoffmann's current claim status is 'Rejected'"`); every document claim is
`supported` with `provenance_valid=True`. The split is exactly along the CRM/PDF boundary.

**Ruled out by evidence:**
- *A second check overriding the score* — **confirmed**, this is the cause (step 8), compounded by
  a score denominator that hides the failing claims (step 7).
- *`additional_context_docs` treated differently from `reranked_docs`* — **confirmed**, but not at
  the merge (step 1 is correct); the divergence is the PDF-only provenance filter (steps 3-5).
- *`compress_context` dropping CRM docs* — **excluded.** `enable_context_compression` is `False`
  (`src/config/models.py:643`, `.env:35`), so `compress_context` (`rag_service.py:3304`, called at
  `4958`) never runs on this path. Latent risk only: it collapses all docs into a single
  `{"source": "LLM-Compressed"}` document (`3327`), which would destroy PDF *and* CRM provenance
  if ever switched on.

#### Minimal change

Teach the provenance filter the CRM citation vocabulary; do **not** weaken the sensitive-claim
gate. In `_answer_citations_match` (`src/core/claim_groundedness.py:319-348`), recognise
`[CRM: <identifier>]` labels alongside PDF labels and match them against CRM document identity,
so CRM windows survive into `allowed_docs` at `600-606`. Everything downstream stays intact: the
CRM claim then either is or is not entailed by the CRM row it cites.

Guard rail on the change: accept CRM labels **only when a CRM document is actually present in
`docs`**. This keeps `retrieval_only` byte-for-byte unchanged.

Explicitly *not* the fix: lowering `min_groundedness`, or exempting CRM claims from the sensitive
gate. Both would let genuinely unsupported amounts through — the exact failure the gate exists for.

#### Tests that could break

- `tests/unit/test_claim_groundedness.py` — citation/provenance cases at lines 114, 163, 174, 225
  are the closest neighbours of the edited function.
- `tests/unit/test_groundedness_diagnostics.py` — asserts on the diagnostics block whose
  `provenance_failure_count` / `relation_counts` this change moves.
- `tests/unit/test_controlled_abstention.py`, `tests/unit/test_fallback_recovery.py`,
  `tests/unit/test_rag_recovery_regressions.py` — all assert fallback behaviour that this change
  is designed to stop triggering in combined mode.

#### The unit test (fail before, pass after)

New case in `tests/unit/test_claim_groundedness.py`, mocked judge, no live Ollama/EspoCRM:

> Given `docs = [pdf_doc, crm_doc]` and an answer mixing `[motor-vehicle-insurance-sti.pdf, page 14]`
> with `[CRM: TEST-KFZ-2026-1001]`, assert the CRM-cited claim comes back
> `relation="supported"` and `provenance_valid=True`, and that
> `sensitive_unsupported_count == 0`.

Today the CRM document index is dropped from `allowed_docs`, the claim resolves `unknown`, and the
assertion fails. Pair it with a negative case: a CRM claim citing a policy number that appears in
no CRM document must stay `unknown`.

#### Risk to `retrieval_only`

Widening the accepted citation vocabulary creates a non-PDF escape hatch from the rule that today
forces every answer to cite a retrieved PDF. If a document-only answer hallucinates a
`[CRM: ...]` label, a claim could be admitted on evidence no PDF supports — silently weakening the
strongest guarantee in `retrieval_only` mode. The presence-guard above is what contains this, and
C8 (`retrieval_only` control) plus the abstention suite must be re-run to prove it.

---

### 4.2 F7 — CRM citation blocks make claim extraction unsatisfiable (C1, C5)

Diagnosed 2026-09-14 on the N=5 `raw-after/` evidence, no code changed, triage not re-run.
**This corrects F6's attribution.** F6 recorded these as "the extractor did not reference every
required answer unit" — that is the wrong validation rule, and C1 and C5 are not even the same
failure:

| case | runs | stage | code | message |
|---|---|---|---|---|
| C1 | **5 of 5** | `claim_extraction_validation` | `incomplete_claim_extraction` | `List heading context missing for fragment items` |
| C5 | 4 of 5 | `claim_extraction_audit` | `unfaithful_claim_extraction` | audit rejected the extraction |
| C5 | 1 of 5 | `claim_extraction_validation` | `incomplete_claim_extraction` | `List heading context missing for fragment items` |

Every C1 run records `attempts: 2` — the repair ran and failed identically every time.

#### Mechanism

| # | Location | What happens |
|---|---|---|
| 1 | `claim_groundedness.py:207` | `_clean_answer_lines` strips citations with `r"\[[^\]\n]+,\s*(?:physical\s+)?page\s+[^\]]+\]"` — the pattern **requires a `, page N` component**. `[CRM: TEST-KFZ-2026-1001]` has none, so it survives into the answer text. |
| 2 | `claim_groundedness.py:211` | Only lines matching `re.fullmatch(r"(?:sources?|citations?|quellen?)\s*:?")` are dropped. The generator emits **`Source citations:`** — two words, no match — so the heading survives. |
| 3 | `claim_groundedness.py:272-275` | `_answer_structure` sees a lone colon line and files it as a context heading: `active_heading_id` is set and the unit is deliberately **excluded from `required`**. |
| 4 | `claim_groundedness.py:241-248` | The bare `[CRM: ...]` bullet reaches `_looks_like_fragment`: 5 word-tokens, no predicate verb → **`True`**. |
| 5 | `claim_groundedness.py:281-282` | So `dependencies[citation_unit] = heading_unit` is recorded. |
| 6 | `claim_groundedness.py:523-528` | `_extraction_validation_error` now demands a **single claim whose `unit_ids` contains both the heading and the bare identifier**. There is no faithful atomic factual claim to be made out of `Source citations:` + `[CRM: TEST-KFZ-2026-1001]` — it is citation metadata, not an assertion. The constraint is unsatisfiable. |
| 7 | `claim_groundedness.py:23, 552-573` | `_MAX_REPAIR_ATTEMPTS = 1`. The repair call gets the same feedback and fails the same way. |
| 8 | `claim_groundedness.py:592` | The verbatim-recovery escape hatch is gated **`if not dependencies:`** — non-empty precisely because of this artifact. **The one salvage path is disabled by the same thing that caused the failure.** |
| 9 | `claim_groundedness.py:804+` | `_failed_result` → `score = None`, `evaluation_status = "failed"`. |
| 10 | `nemo_actions.py:1168-1175` | `evaluation_status == "failed"` → `allow = False`, `action = "fallback"`. |
| 11 | `rag_service.py:5233-5240` | Answer replaced by `grounding_fallback_text`, `sources = []`, decision `post_fallback`. |

**Proof without the LLM** — `_answer_structure` on the two citation shapes:

| answer shape | units | dependencies |
|---|---|---|
| `…\nSource citations:\n- [motor-vehicle-insurance-sti.pdf, page 14]` | 2 (citation stripped) | `{}` |
| `…\nSource citations:\n- [CRM: TEST-KFZ-2026-1001]` | 3 (citation survives) | **`{2: 1}`** |

#### Is this CRM-specific, or would any answer of this shape fail?

**The trigger is CRM-specific; the rule that fires is general.** PDF citations never reach the unit
stream because step 1 strips them, so `retrieval_only` cannot produce this shape — C8 has no bare
citation unit in any run. Only CRM labels survive to become units.

But the decisive comparison is C2 and C4, which **did** score:

| case | bare `[CRM: …]` unit present | preceded by a colon heading | dependencies | outcome |
|---|---|---|---|---|
| C1 | yes | **yes** (`Source citations:`) | `{4: 3}` | extraction fails, 5/5 |
| C5 | yes | **yes** (`Source citations:`) | `{14: 13}` + 7 list deps | extraction fails, 5/5 |
| C2 | yes | no | `{}` | `success`, score 1.00 |
| C4 | yes | no | `{}` | `uncertain`, score 0.50 |
| C8 | no (stripped) | n/a | `{}` | `success`, score 1.00 |

C2 and C4 carry the same bare `[CRM: …]` unit and evaluate fine. **The only structural difference is
whether the generator happened to precede it with a `Source citations:` heading.** The evaluation
outcome therefore depends on a formatting coincidence in the answer, not on whether the answer is
grounded. C7 is not comparable — `crm_only` scores through `_crm_groundedness_score` in `main.py`
and never enters this evaluator (`units = 0`, no groundedness evidence block).

#### Is one repair attempt the right design? No — but that is not the defect either

Raising `_MAX_REPAIR_ATTEMPTS` would be the wrong fix. The constraint at step 6 is **unsatisfiable**,
not merely hard: no number of retries lets a model invent a faithful factual claim out of citation
metadata. More attempts would buy nothing and cost two extra LLM round-trips per query.

**The real defect is the F3/H3 class: an operational failure is reported through the safety
channel.** `nemo_actions.py:1168` does attach its own reason code (`groundedness_evaluator_failed`),
so the cause is recoverable from diagnostics — better than nothing. But the user-visible outcome is
byte-identical to a genuine groundedness rejection: the same `grounding_fallback_text`, the same
cleared sources. The system tells the user *"I could not generate an answer that is sufficiently
supported by the available documents"* when the truth is *"the evaluator could not run."* Those are
different statements about the world, and only one of them is honest here. The answer C1 produced
was never assessed at all.

#### Minimal change

Strip CRM citation labels in `_clean_answer_lines` (`claim_groundedness.py:207`) the same way PDF
citations are already stripped, and drop any line left empty afterwards. The `Source citations:`
heading then has no following fragment, no dependency is recorded, and the answer is evaluated on
its factual sentences alone.

This is **restoring an existing invariant, not adding a special case**: the function already holds
that citation metadata is not answer content. It simply learned only one of the two citation
vocabularies this pipeline emits — the same root asymmetry as F1, one layer earlier in the pipeline.

Explicitly *not* the fix: raising `_MAX_REPAIR_ATTEMPTS`, relaxing the dependency rule at 523-528,
or removing the `if not dependencies:` guard at 592. The dependency rule exists to stop a fragment
bullet being scored without the heading that supplies its predicate — that is correct behaviour and
must keep working for genuine content lists (C5's units 2-8 are exactly such a list).

The error-channel separation is a **second, independent change** (F3 family) and should not be
bundled with this one.

#### The unit test (fail before, pass after)

Pure, deterministic, no judge and no LLM — `tests/unit/test_claim_groundedness.py`:

> `_answer_structure("Glass damage is covered.\nSource citations:\n- [CRM: TEST-KFZ-2026-1001]")`
> must return no fragment→heading dependencies, and no unit may be a bare `[CRM: …]` label.

Today this returns `deps == {2: 1}` with the citation as unit 2, so it fails. Pair it with two
guards that must keep passing: the PDF-citation equivalent must stay at `deps == {}` (the invariant
being restored), and a genuine content list under a colon heading must **still** produce
dependencies, so the rule is not weakened. A judge-level case asserting
`evaluation_status == "success"` for an answer carrying a CRM citation block is worth adding on top.

#### Risk to `retrieval_only`

The change deletes text from the answer before claim extraction, and anything deleted is never
turned into a claim and therefore never checked. If the pattern is broader than the literal
`[CRM: …]` form, it could silently swallow real answer content that happens to be bracketed — a
statutory reference such as `[Art. 31 para. 2 RTA]`, or a quoted document passage containing
brackets. An unsupported assertion inside such a span would then pass unexamined, which is strictly
worse than the fallback it replaces. The pattern must be anchored to the literal `CRM:` prefix, the
existing PDF pattern must not be loosened, and C8 plus the abstention suite must be re-run before
this is called done.

---

## 5. Phase 3 — Hardening and optimization (only once Phase 2 is green)

- `src/api/rag_service.py` is ~5,500 lines. Split along the pipeline boundaries that already
  exist implicitly: retrieval, rerank, context assembly, guardrails, generation. Do this with
  the test suite green, one extraction per commit, no behaviour change in the same commit.
- End-to-end regression run + updated `artifacts/test-results/INDEX.md`.
- Latency budget per stage from the existing `begin_stage`/`finish_stage` timings.

---

## 6. Session and token discipline

The repo is large; context is the scarce resource. Rules:

1. **One phase = one session.** Start a fresh session per phase; this document carries state.
2. **Never ask for a whole-repo read.** Name the file and the function.
3. **Search before reading.** Grep for a symbol, then read the ~80 lines around it.
4. **Logs go to files, never into the conversation.** Read a CSV summary instead.
5. **Do not re-run heavy evaluations inside a session.** Run them in a terminal; feed back
   only the summary row.
6. **Model split:** the strong model for diagnosis, hypothesis design, and any change touching
   guardrail or groundedness semantics. The faster model for mechanical work — writing tests
   from a spec, renames, extractions, running the suite, fixing imports. Switching for the
   mechanical half is the single largest token saving available.
7. **Commit small.** A commit that does two things cannot be bisected or reverted cleanly.
8. **Update section 7 before ending any session.**

---

## 7. Status log

| Date | Phase | What was done | Result | Next |
|---|---|---|---|---|
| 2026-09-14 | 1 | Built the real query fixture `tests/fixtures/combined_failing_queries.json` (C1–C8) from the actual seeded EspoCRM data (`data/synthetic/crm/*.csv`), cross-checked against `data/benchmarks/routing/routing_eval_80.jsonl` and literal PDF terms. 6 combined + 2 controls. | Fixture is the ruler; ids C1–C8 stable. | Run triage. |
| 2026-09-14 | 1 | Ran `scripts/tools/triage_combined.py --repeat 2` against a fully ready backend (`pipelineReady/crmReady/llmReady=True`, safety `enforce`, backend `nemo`). | H2 10, H4 2, H5 4. No UNSTABLE rows — every id produced the same hypothesis across both runs, so H3 flakiness is **not** in play. | Diagnose F1. |
| 2026-09-14 | 2 (diagnosis only) | Diagnosed F1 (post stage falls back on score 1.0, C1/C6). Traced the full chain and verified it against the raw C1/C6 evidence. Proposal written to section 4.1. **No code changed.** | Root cause found: PDF-only citation regex evicts CRM evidence inside the evaluator. Threshold is irrelevant to these cases. | Implement 4.1 in a fresh session, failing test first. |
| 2026-09-14 | 2 (fix) | Implemented 4.1 on branch `fix/f1-crm-citation-provenance` (commit `200b64b`), failing test first. `pytest tests/unit -q`: 381 passed, 4 failed — all 4 pre-existing, verified by re-running them with the fix stashed. | CRM claims are now adjudicated instead of evicted. | Restart backend, re-run the ruler. |
| 2026-09-14 | 2 (measure) | Restarted `mva-backend` (bind-mount + `--reload` had **not** picked the change up; no WatchFiles event since the edit — verified in container before measuring). Re-ran `triage_combined.py --repeat 2`. | **Counts are not comparable to the pre-fix run — the triage classifier itself changed between the two runs** (route normalisation + post-stage separation), so H5 4→0 is the classifier, not the fix. Before: H2 10, H4 2, H5 4. After: H2 5, H3 4, OK 3, H4 2, H1 1, UNKNOWN 1. Real signal is per-case: C2 produced a full combined answer (6/6 claims supported, 0 provenance failures, 2 CRM sources in final sources) — no combined case did that before. C1's score moved from a fake 1.0 (3 CRM claims hidden as `unknown`) to an honest 0.75. **C7 green both runs. C8 fell back on run 1, passed on run 2** — C8 carries no CRM documents (`crm_ctx=0`), so the `crm_in_docs` guard makes the fix a byte-identical no-op there; pre-fix C8 raw was overwritten, so its prior stability cannot be confirmed from surviving evidence. | Flakiness is now the dominant signal (4 of 8 ids unstable across 2 runs). Establish whether it is generation non-determinism before reading any hypothesis count as real. |
| 2026-09-14 | 2 (diagnosis only) | Traced F6, the non-determinism, through the generation and groundedness paths and compared the C2/C6 raw evidence run-to-run. **No code changed, triage not re-run.** | Confirmed: answer generation is pinned to `ANSWER_TEMPERATURE=0`; the evaluator is second-order (`temperature=0` but **no seed anywhere in `src/`**). F6 does **not** subsume F3 — the `None` scores have three separate causes. Written up in the F6 block below. | Re-run with `--repeat 5`, and re-establish the counts as a distribution before touching any remaining hypothesis. |
| 2026-09-14 | 2 (measure, N=5) | Pinned `ANSWER_TEMPERATURE=0` (default, `src/config/models.py`), restarted the backend, then ran `triage_combined.py --repeat 5` twice: once with the F1 fix live (branch `fix/f1-crm-citation-provenance`), once with only `src/core/claim_groundedness.py` reverted to `Abschluss-Arbeit` (verified 0 fix-markers in the container before the run) and restored afterward. Triage script and fixture were byte-identical in both conditions — the only variable was that one file. Results: `artifacts/test-results/combined-triage/{BEFORE,AFTER}-temp0.csv`, raw JSON under `raw-before/` and `raw-after/`. | See the N=5 before/after table below. Combined-mode real answers: **0/30 → 5/30**. Controls unchanged: C7 and C8 both 5/5 real in both conditions with identical score distributions. **This measurement supersedes the two temperature-0.4 runs above — those were single draws with a confound (the triage classifier also changed between them) and are kept only for narrative continuity.** | Investigate why C1, C3, C5 never scored across 10 runs, and why C6 falls back at score 1.00 every time. |
| 2026-09-14 | 2 (diagnosis only) | Diagnosed why C1 and C5 never score, from the N=5 `raw-after/` evidence. **No code changed, triage not re-run** (the evidence was collected at temperature 0 and is valid as-is). Written up as F7 in section 4.2. | Root cause: `_clean_answer_lines` strips PDF citations but not `[CRM: …]` labels, so a trailing `Source citations:` block survives into the unit stream, the bare label is classified as a fragment, and the fragment→heading rule then demands a factual claim made out of citation metadata — unsatisfiable, and it also disables the verbatim-recovery escape hatch (`if not dependencies:`). **Corrects F6's attribution** (wrong validation rule; C1 fails at `claim_extraction_validation` 5/5, C5 at `claim_extraction_audit` 4/5). C2/C4 carry the same bare CRM label but no preceding colon heading, so they score — the outcome hinges on a formatting coincidence. | Implement 4.2 in a fresh session, pure `_answer_structure` test first. Keep the error-channel separation (F3 family) as a separate change. |

### N=5 before/after at `ANSWER_TEMPERATURE=0` (the attributable measurement)

Real/fallback out of 5 runs, groundedness min/median/max, and count of runs with no score at all.
"Real" = `safety_decision == allow` and a non-trivial answer (`answer_chars > 100`); the 89/77-char
rows are the fixed fallback texts.

| id | route | BEFORE real/fb | BEFORE score (min/med/max) | BEFORE none | AFTER real/fb | AFTER score (min/med/max) | AFTER none |
|---|---|---|---|---|---|---|---|
| C1 | combined | 0/5 | — | 5 | 0/5 | — | 5 |
| C2 | combined | 0/5 | 0.67 / 0.75 / 1.00 | 0 | **4/5** | 1.00 / 1.00 / 1.00 | 1 |
| C3 | combined | 0/5 | — | 5 | 0/5 | — | 5 |
| C4 | combined | 0/5 | 0.00 / 0.33 / 0.50 | 0 | **1/5** | 0.50 / 0.50 / 1.00 | 0 |
| C5 | combined | 0/5 | — | 5 | 0/5 | — | 5 |
| C6 | combined | 0/5 | 1.00 / 1.00 / 1.00 | 2 | 0/5 | 1.00 / 1.00 / 1.00 | 2 |
| C7 | crm_only (control) | 5/5 | 0.99 / 0.99 / 0.99 | 0 | 5/5 | 0.99 / 0.99 / 0.99 | 0 |
| C8 | retrieval_only (control) | 5/5 | 1.00 / 1.00 / 1.00 | 0 | 5/5 | 1.00 / 1.00 / 1.00 | 0 |

**Headline:** combined-mode real answers went **0/30 → 5/30** with the fix. Both controls are
unchanged — C7 and C8 stayed 5/5 real with byte-identical score distributions across conditions,
so the fix did not regress either.

**Methodology:** both conditions ran at `ANSWER_TEMPERATURE=0`; `scripts/tools/triage_combined.py`
and `tests/fixtures/combined_failing_queries.json` were byte-identical between conditions; the only
difference was `src/core/claim_groundedness.py` (the F1 fix present vs. the `Abschluss-Arbeit`
version). This isolates the fix as the sole variable, unlike the temperature-0.4 runs above, where
the triage classifier also changed between the pre- and post-fix measurements.

**Caveat — this supersedes the earlier temperature-0.4 numbers.** Those were each a single draw
under known sampling variance (F6); read them as narrative history of how the investigation
proceeded, not as a quantitative baseline.

**C1, C3 and C5 produced no score in any of the 10 runs across both conditions** (5 before + 5
after each). The F1 fix could not have affected them either way — whatever blocks these three
happens before or instead of groundedness scoring, and is unrelated to F1.

### Triage counts (Phase 1 complete — 8 cases × 2 runs = 16 rows)

| Hypothesis | Failing queries assigned | Confirmed? |
|---|---|---|
| H1 context rail / PII | 0 | Dropped — no row. The context rail allowed CRM PII throughout (`pii_allowed_business_contact` is an allow-listed risk, not a block). |
| H2 groundedness threshold | 10 | **No — mislabelled.** The triage classifier files a `passed=False` result as H2, but the reason code is `groundedness_sensitive_unsupported`, never `low_groundedness`. See 4.1. |
| H3 fail_closed masking errors | 0 (+2 evaluator failures on C1 re-runs) | Not confirmed as a distinct defect. Two C1 runs returned `evaluation_status="failed"` / `groundedness_evaluator_failed`; worth a second look only after F1 is fixed. |
| H4 CRM docs evicted | 2 direct, **+10 by reclassification** | **Confirmed.** This is the real F1 family. Eviction happens inside `claim_groundedness`, not at context assembly as H4 originally predicted — update H4's wording when fixing. |
| H5 routing | 4 | **No.** Artifact of route-label spelling (`crm-only` vs the fixture's `crm_only`). Routing is functionally correct; normalise the label comparison or the fixture. Not a pipeline defect. |
| H6 completeness heuristic | 0 | Dropped — no row. All responses were HTTP 200, none downgraded to 206/partial. |

**Reading of the table:** only one real defect is in evidence. H2's 10 rows and H4's 2 rows are the
same root cause seen from two angles; H5 is a label mismatch; H1 and H6 have no rows and are
dropped per the section 3 exit criterion. Fix 4.1, then re-run the ruler before re-ranking anything.

---

### F6 — the evaluation path is non-deterministic (top priority)

Diagnosed 2026-09-14, no code changed. **Until this is contained, no hypothesis count in the
table above is trustworthy**, because a count is a sum over verdicts that are re-sampled on every
run. 4 of 8 ids changed verdict between two consecutive runs of identical code and queries.

#### Mechanism — three layers, in order of contribution

**1. Answer generation is sampled (first-order, the dominant term).**
`temperature_answer = _env_float("ANSWER_TEMPERATURE", 0.0)` (`src/config/models.py:589`), applied
by `initialize_llm()` (`src/api/rag_service.py:3075`). `ANSWER_TEMPERATURE` is **unset in the
container**, so 0 is live. Before this was pinned, the answers genuinely differed run to run — not paraphrases, different
content:

| case | run 1 answer | run 2 answer |
|---|---|---|
| C2 | 760 chars, leads with "claimed amount of 780 EUR" | 649 chars, leads with "synthetic windshield stone-chip" |
| C6 | 689 chars | 511 chars |

**2. A different answer yields a different claim set, hence a different denominator (second-order).**
The evaluator extracts claims *from the answer*, so the unit of measurement changes with the thing
being measured. Score is `supported / decided_count`, and `decided_count` excludes `unknown`
(`src/core/claim_groundedness.py:762-764`):

| case | run 1 | run 2 |
|---|---|---|
| C2 | 6 claims, 6 supported → **1.00** | 4 claims, 3 supported + 1 insufficient → **0.75** |
| C6 | 5 claims, 4 supported + 1 unknown → **1.00** | 3 claims, 3 unknown → **None** |

So C2's 1.00 → 0.75 is not the evaluator becoming stricter; it is a shorter answer making fewer,
differently-worded claims. The evaluator is a **second-order effect**, not the source.

**3. The evaluator has no seed (third-order, irreducible).**
`ModelJudge.call` sets `temperature=0` (`src/core/claim_groundedness.py:393`) — no `top_p`, and
critically **no `seed`**. The Responses API request is built at
`src/integrations/openai_answer_model.py:110-118` from `model`, `input`, `max_output_tokens`,
`temperature`, `store` only. `grep -rn "seed=" src/` returns nothing: no seed is passed anywhere in
the codebase. Extraction and judging share the same model (`GROUNDEDNESS_MODEL` → `roles.answer` →
`gpt-4o-mini`, env unset). temperature=0 is near-greedy decoding, not a determinism guarantee.

**Compounding: the recovery path re-samples and silently overwrites.**
On `retry_evaluation`, `check_answer_safety` is called a second time and its result *replaces* the
first (`src/api/rag_service.py:5054-5063`). Every case in this run recorded
`retries={'groundedness_evaluation': 1}`. The verdict you see is the second independent draw, not a
consensus of the two — so the pipeline samples twice and keeps the later sample.

#### Does F6 subsume F3 (the `None` scores)? No — F3 is three different things

| cause | where | evidence | F6? |
|---|---|---|---|
| All claims `unknown` → `decided_count == 0` → `supported_fraction = None`. A *successful* evaluation with no score. | `claim_groundedness.py:763` | C6-run2: `counts={'unknown':3}`, `exc=None`, `status=uncertain` | **Yes — subsumed** |
| Claim extraction failed. ~~The extractor did not reference every required answer unit~~ — **superseded by F7 (section 4.2): the actual rule that fails is the fragment→heading dependency check (`claim_groundedness.py:523-528`), triggered by a `Source citations:` heading followed by a bare `[CRM: …]` label, and C1 and C5 are not even the same failure stage.** | `claim_groundedness.py:523-528, 552-573, 592` | C1 5/5 `claim_extraction_validation`; C5 4/5 `claim_extraction_audit` | **No — distinct defect**, see 4.2 |
| Groundedness never ran; the context/PII rail terminated first. | — | C3: `stageStatus` stops after `guardrail`; answer is the PII refusal text | **No — unrelated** |

F3 must be split along these three lines before it is worked on; only the first is F6.

#### Minimal determinism fix

1. Set `ANSWER_TEMPERATURE=0` for evaluation runs. **Env only, no code change**, and it removes the
   dominant term. This is the single highest-value knob.
2. Stop the retry from silently replacing the first verdict — record both and make the policy
   explicit — so a re-sample is visible rather than absorbed.
3. Raise repeats: treat `--repeat 5` as the minimum for any claim about counts.

#### Can reproducibility be achieved with the current guardrail backend? No.

Being direct, because the thesis will need this stated honestly: **bit-for-bit reproducibility is
not attainable on this stack.** temperature=0 without a seed against a hosted model is not
deterministic, and the Responses API path used here passes no seed at all — the `seed` /
`system_fingerprint` reproducibility pair exists on Chat Completions, not on the path this code
takes, and OpenAI documents it as best-effort even there. Genuine determinism would require one of:
migrating the judge to Chat Completions with `seed` and recording `system_fingerprint`; caching
evaluator verdicts keyed by a hash of (answer, evidence); or a local judge model under our own
decoding control.

The achievable target is therefore **statistical stability, not reproducibility**: pin
`ANSWER_TEMPERATURE=0`, run N ≥ 5, and report a distribution with a variance figure. Any single-run
number in this document — including every count in the table above — should be read as one draw.

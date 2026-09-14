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

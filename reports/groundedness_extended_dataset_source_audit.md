# Data Source Audit - Extended Groundedness Dataset

Generated: `2026-08-01T21:05:09.945031+00:00`

This audit was completed read-only before candidate artifacts were generated.
No Chroma retrieval, collection mutation, reindex, embedding generation, CRM
write, LLM call, Self-Check or `/api/ask` call was performed.

| Data source | Exact path | Format / available entries | Fields / language | PASS use | FAIL / hard-negative use | Quality and leakage risks |
|---|---|---|---|---|---|---|
| InsuranceQA local benchmark | `data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl` | JSONL, 1000 unique Q/A rows | `question`, `answer`; English | Exact sourced answers, supported prefacing, valid citations | Similar-question distractors, cross-product answers, mixed supported/unsupported answers, wrong percentages | Public/general-domain answers can be dated or informal; similar questions must stay grouped; no human Groundedness annotation |
| InsuranceQA thesis subset | `data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl` | JSONL, 200 rows, complete subset overlap with the 1,000-row file | Q/A plus evaluation metadata; English | Audit/reference only | Audit/reference only | Not independent from the 1,000-row source; must never be split as a separate hold-out |
| InsuranceQA Chroma | `data/processed/vectorstores/chroma_db/chroma.sqlite3`, collection `insuranceqa_collection` | Read-only collection metadata, 1,248 chunks, dimension 1,024 | `source`, `question_id`, `dataset`, `start_index`; English content | Confirms indexed representation | Not used to generate candidates | Chunks from one Q/A can leak across splits; collection was inspected only by SQLite counts |
| Active Helvetia PDFs | `data/raw/pdfs/*.pdf` | 11 PDFs, 138 pages, 1,585 indexed chunks; 1424 usable sentence candidates | English policy/product text and page references | Exact provisions, supported citations, multi-clause contexts | Numeric mutations, polarity reversal, citation mismatch, unsupported extra clauses | PDF extraction can join headers, tables or hyphenated text; all cases require visual/source review; multiple pages of one document are related |
| Excluded Baloise/duplicate PDFs | `data/raw/excluded_pdfs/*.pdf` | 4 PDFs, 100 pages; 1238 usable sentence candidates | Three English Baloise products plus one duplicate Helvetia brochure | Not used as current-product PASS evidence | Explicit old-document and cross-provider distractors only | Not part of the current Helvetia corpus; high risk of accidental provider/version leakage; kept in source-family groups |
| Synthetic CRM contacts | `data/synthetic/crm/contacts.csv` | CSV, 10 contacts | name and test email; English-compatible synthetic values | Correct customer association | Wrong-customer/entity substitutions | Synthetic only; same customer and all related records must share a split |
| Synthetic CRM policies | `data/synthetic/crm/policies.csv` | CSV, 16 policies | policy number, product, coverage, status, dates, deductible, premium, currency, customer | Exact policy facts and multi-claim answers | Wrong policy/current policy/status/deductible/premium/coverage/date/entity | Small number of customers; Lara current-policy family is one concentrated template family |
| Synthetic CRM claims | `data/synthetic/crm/claims.csv` | CSV, 8 claims | claim/policy/customer relationship, date, type, amount, status | Exact claim facts | Unsupported claim decisions | Status is a stored fact, not a legal claim decision; policy/customer groups must stay together |
| CRM schema and seed workflow | `data/synthetic/crm/schema.json`; `scripts/seed_espocrm.py` | JSON schema plus Python seed script | Contact, MvaPolicy, MvaClaim relationships | Provenance and relationship validation | Relationship hard negatives | Seed script is not called; live CRM is not used or modified |
| Legacy groundedness fixture | `tests/fixtures/groundedness_calibration_cases.json` | JSON, 28 synthetic cases (14 expected PASS, 14 expected FAIL) | query, documents, answer, boolean technical label; English | Re-review existing supported candidates | Re-review existing mutated candidates | Existing labels have no documented human provenance; same cases created and evaluated old threshold 0.51 |
| Existing score artifact | `config/groundedness_calibration.json` | JSON, 28 stored scores; threshold 0.51 | `fact_aware_claim_support_v4` scores | Score reproducibility audit only | Score reproducibility audit only | Must not be visible during annotation and is not Ground Truth |
| Reranker fixtures | `tests/fixtures/reranker_*cases.json` | JSON, 6 + 64 + 64 ranking cases | query, candidates, correct candidate ID | Scenario inventory | Hard-negative pattern inventory | Ranking labels are not answer-groundedness labels; no automatic transfer |

## Implementation paths

- `src/data/insuranceqa_ingestion.py:43, 75-123, 138-245`: local/HuggingFace loading, Q/A document mapping, chunking and separate collection.
- `scripts/seed_espocrm.py:143-227`: reads the three synthetic CSVs and seeds relationships; not executed here.
- `src/guardrails/integrations/nemo_actions.py:178, 345-493`: deterministic `fact_aware_claim_support_v4` implementation.
- `scripts/calibrate_groundedness.py:46-115`: prior 28-case score/threshold workflow.

## Audit conclusion

The sources are sufficient to prepare 400-500 review candidates without
inventing human labels. They are not sufficient for a final calibrated
threshold until two independent reviewers annotate labels, error categories
and criticality and all conflicts are adjudicated.

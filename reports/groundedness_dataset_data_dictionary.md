# Groundedness Extended Dataset - Data Dictionary

| Field | Definition |
|---|---|
| `case_id` | Stable unique candidate identifier. |
| `source_type` | insuranceqa, helvetia_pdf, baloise_excluded_pdf, synthetic_crm, or legacy_synthetic. |
| `source_file` | Repository-relative primary evidence path. |
| `source_document_id` | Stable source record or document identifier. |
| `source_page` | Physical one-based PDF page where applicable. |
| `source_chunk_id` | Source-specific chunk/row identifier. |
| `question` | English evaluation question. |
| `context` | Stored evidence supplied to the evaluator; list in JSONL, joined in CSV/XLSX. |
| `candidate_answer` | English answer independently judged by reviewers. |
| `generator_expected_label` | Technical construction expectation only; never Ground Truth and hidden from reviewer sheets. |
| `mutation_type` | Reproducible technical case construction or hard-negative mechanism. |
| `question_family_id` | Groups paraphrases/variants of one source question. |
| `document_family_id` | Groups records from the same source document/version family. |
| `customer_group_id` | Synthetic customer relationship group, blank for non-CRM cases. |
| `policy_group_id` | Synthetic policy/version group, blank for non-CRM cases. |
| `product_group` | Coarse insurance product stratum. |
| `reviewer_1_label` | Independent reviewer 1 label: PASS, FAIL, AMBIGUOUS, EXCLUDE. |
| `reviewer_2_label` | Independent reviewer 2 label: PASS, FAIL, AMBIGUOUS, EXCLUDE. |
| `adjudicated_label` | Final adjudicated label; remains blank until conflicts are resolved. |
| `reviewer_*_error_category` | Reviewer-selected error category; use none for PASS and other with explanation when needed. |
| `adjudicated_error_category` | Final adjudicated category. |
| `reviewer_*_criticality` | CRITICAL, NON_CRITICAL, or REVIEW_REQUIRED. |
| `adjudicated_criticality` | Final adjudicated criticality. |
| `reviewer_*_notes` | Evidence-based rationale; no score information is shown. |
| `adjudication_notes` | Resolution rationale for reviewer conflict. |
| `reviewer_*_confidence` | Integer confidence from 1 (very uncertain) to 5 (highly confident). |
| `annotation_status` | PENDING, REVIEW_REQUIRED, REVIEWED, CONFLICT, or ADJUDICATED. |
| `groundedness_score` | Blank during review; populated only by isolated post-annotation scoring. |
| `split` | Preliminary CALIBRATION, VALIDATION, or LOCKED_HOLDOUT_CANDIDATE group assignment. |
| `leakage_group_id` | All related or near-duplicate candidates share one split through this ID. |
| `quality_flags` | Non-destructive technical warnings requiring review. |
| `hard_negative` | True when a candidate intentionally uses a plausible but technically mutated answer. |

## Human label definitions

- **PASS:** Every material answer claim is supported by context, belongs to the correct entity and answers the question.
- **FAIL:** At least one material claim is unsupported, contradicted, attached to the wrong entity or technically incorrect.
- **AMBIGUOUS:** Question, context or answer does not permit a unique expert judgment.
- **EXCLUDE:** The case is technically defective, duplicate or unsuitable for the study.

`generator_expected_label` must not be copied into any reviewer or adjudicated
field. Groundedness scores remain unavailable until annotation and adjudication
are complete.

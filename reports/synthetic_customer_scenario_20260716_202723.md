# English Synthetic Customer Insurance RAG Scenario

## 1. Objective
Fix contract-ID Safety handling, date/phone classification, groundedness calibration, and response-language telemetry, then execute one real isolated English scenario through `POST /api/ask`.

## 2. Previous failure summary
The 2026-07-11 run retrieved the correct motor page but Context Safety redacted contract IDs and classified coverage dates as phones. The generated answer therefore omitted `TEST-KFZ-2026-1001`; Quality and Overall failed while production counts and cleanup passed.

## 3. Root-cause analysis
- Contract IDs were detected but not included in the allowed-PII policy.
- The phone regex accepted newline whitespace and date-shaped digit sequences before specific date handling.
- Groundedness used lexical overlap that ignored identifiers and numbers, with an uncalibrated `.env` value of `0.2`.
- Audit callers supplied a hardcoded `English` value instead of deriving language from the final answer.

## 4. Files changed
- `src/core/safety_audit.py`
- `src/guardrails/integrations/nemo_actions.py`
- `src/config/models.py`
- `src/api/rag_service.py`
- `src/utils/language.py`
- `scripts/calibrate_groundedness.py`
- `scripts/test_synthetic_customer_scenario.py`
- `tests/fixtures/groundedness_calibration_cases.json`
- `synthetic_customer_insurance_lara_neumann_en.pdf`
- `tests/unit/test_safety_pii_rules.py`
- `tests/unit/test_response_language.py`
- `tests/unit/test_groundedness_calibration.py`
- `tests/unit/test_thesis_eval_metadata.py`
- `config/groundedness_calibration.json`

## 5. Exact code changes
- Contract IDs are classified as allowed while other PII remains sensitive.
- Valid calendar-date spans are excluded from phone detection and phone matching cannot cross lines.
- Groundedness now scores claim-level lexical support, exact hard facts, coverage polarity, and query/rank evidence; a reproducible calibration artifact supplies the threshold.
- Audit overwrites any caller language value with deterministic detection from `generated_answer`.
- The baseline harness now creates and validates a three-page English PDF and records all mandatory result categories.

## 6. Safety policy change for contract numbers
```json
{
  "context": {
    "detected_count": 8,
    "allowed_count": 4,
    "redacted_count": 4,
    "detected_types": {
      "contract_id": 4,
      "generic_id": 1,
      "customer_number": 1,
      "address": 1,
      "date_of_birth": 1
    },
    "allowed_types": {
      "contract_id": 4
    },
    "redacted_types": {
      "generic_id": 1,
      "customer_number": 1,
      "address": 1,
      "date_of_birth": 1
    },
    "items": [
      {
        "pii_type": "contract_id",
        "start": 166,
        "end": 184,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "generic_id",
        "start": 307,
        "end": 314,
        "allowed": false,
        "source": "identifier_format_regex",
        "reason": "identifier_format:ln"
      },
      {
        "pii_type": "contract_id",
        "start": 182,
        "end": 200,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "customer_number",
        "start": 163,
        "end": 180,
        "allowed": false,
        "source": "identifier_label_regex",
        "reason": "identifier_label:customer number"
      },
      {
        "pii_type": "address",
        "start": 190,
        "end": 223,
        "allowed": false,
        "source": "address_label_regex",
        "reason": "address_label"
      },
      {
        "pii_type": "date_of_birth",
        "start": 239,
        "end": 249,
        "allowed": false,
        "source": "dob_label_regex",
        "reason": "date_of_birth_label"
      },
      {
        "pii_type": "contract_id",
        "start": 314,
        "end": 332,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "contract_id",
        "start": 449,
        "end": 467,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      }
    ]
  },
  "output": {
    "detected_count": 1,
    "allowed_count": 1,
    "redacted_count": 0,
    "detected_types": {
      "contract_id": 1
    },
    "allowed_types": {
      "contract_id": 1
    },
    "redacted_types": {},
    "items": [
      {
        "pii_type": "contract_id",
        "start": 197,
        "end": 215,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      }
    ]
  },
  "contract_allowed_in_context": true,
  "contract_allowed_in_output": true
}
```

## 7. Phone/date false-positive fix
```json
{
  "date_probe": "Coverage dates: 01.01.2026, 31.12.2026, 14.05.1988, 2026-01-01, 2026-12-31, 01/01/2026, 12/31/2026.\n1. Motor insurance\n2. Personal liability insurance",
  "date_pii_types": [],
  "dates_not_phone": true,
  "phone_detected": true,
  "phone_sanitized": "Phone: [REDACTED_PHONE]\n1. Motor insurance\n2. Personal liability insurance",
  "list_numbering_preserved": true
}
```
Context coverage dates preserved: `True`.

## 8. Groundedness calibration methodology
Algorithm `fact_aware_claim_support_v4` scored `28` balanced labeled cases. Rule: Require recall >= 0.80 when feasible; minimize unsupported false acceptance; then maximize F1, recall, and precision; choose the lowest tied threshold.

## 9. Calibration dataset
| Case | Label | Score | Prediction |
|---|---|---:|---|
| supported_motor_full_answer | supported | 1.0 | accept |
| supported_motor_paraphrase | supported | 0.814545 | accept |
| supported_contract_identifier | supported | 1.0 | accept |
| supported_deductible | supported | 0.886667 | accept |
| supported_exclusion | supported | 0.8 | accept |
| supported_liability | supported | 0.848889 | accept |
| supported_coverage_dates | supported | 0.902857 | accept |
| supported_home_deductible | supported | 1.0 | accept |
| supported_travel_limit | supported | 1.0 | accept |
| supported_dental_percentage | supported | 1.0 | accept |
| supported_two_claims_with_citation | supported | 1.0 | accept |
| supported_short_numeric_answer | supported | 1.0 | accept |
| unsupported_wrong_motor_deductible | unsupported | 0.141667 | reject |
| unsupported_wrong_contract_identifier | unsupported | 0.145714 | reject |
| unsupported_wrong_coverage_type | unsupported | 0.141667 | reject |
| unsupported_reversed_coverage | unsupported | 0.15 | reject |
| unsupported_zero_deductible | unsupported | 0.1275 | reject |
| unsupported_rental_car_extension | unsupported | 0.333333 | reject |
| unsupported_wrong_policy_dates | unsupported | 0.145714 | reject |
| unsupported_wrong_home_deductible | unsupported | 0.145714 | reject |
| unsupported_wrong_baggage_limit | unsupported | 0.141667 | reject |
| unsupported_wrong_dental_percentage | unsupported | 0.145714 | reject |
| unsupported_mixed_extra_claim | unsupported | 0.5 | reject |
| unsupported_cancellation_claim | unsupported | 0.277875 | reject |
| supported_customer_association | supported | 0.583333 | accept |
| supported_concise_coverage | supported | 1.0 | accept |
| unsupported_liability_distractor_answer | unsupported | 0.141135 | reject |
| unsupported_information_missing | unsupported | 0.075556 | reject |

## 10. Threshold candidates and metrics
| Threshold | TP | TN | FP | FN | Precision | Recall | F1 | False accept rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 14 | 13 | 1 | 0 | 0.933 | 1.000 | 0.966 | 0.071 |
| 0.50 | 14 | 13 | 1 | 0 | 0.933 | 1.000 | 0.966 | 0.071 |
| 0.51 | 14 | 14 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.60 | 13 | 14 | 0 | 1 | 1.000 | 0.929 | 0.963 | 0.000 |
| 0.70 | 13 | 14 | 0 | 1 | 1.000 | 0.929 | 0.963 | 0.000 |

## 11. Selected threshold and rationale
Selected `0.51`; loaded `0.51` from `calibration_file`. Selected metrics: `{"threshold": 0.51, "tp": 14, "tn": 14, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "false_acceptance_rate": 0.0}`. This operating point has zero false acceptance on the labeled set while retaining the highest acceptable true-positive performance under the documented rule.

## 12. Language-detection fix
```json
{
  "status": "PASS",
  "focused_german": {
    "query_language": "English",
    "answer_language_expected": "German",
    "recorded_response_language": "German",
    "status": "PASS"
  },
  "main_answer": {
    "expected": "English",
    "recorded_response_language": "English",
    "status": "PASS"
  }
}
```
Empty, marker-free, and ambiguous short responses use the deterministic fallback `Unknown`.

## 13. Synthetic PDF validation
```json
{
  "path": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tests\\fixtures\\synthetic_customer_insurance_lara_neumann_en.pdf",
  "created": true,
  "valid": true,
  "parsed": true,
  "size_bytes": 76306,
  "valid_signature": true,
  "opened": true,
  "pages": 3,
  "metadata": {
    "/Author": "MVA Insurance RAG synthetic test harness",
    "/CreationDate": "D:20260716203119+02'00'",
    "/Creator": "(unspecified)",
    "/Keywords": "synthetic,synthetic_customer_scenario_en_002,TEST-CUSTOMER-PDF-EN-002,TEST-KD-2026-0001",
    "/ModDate": "D:20260716203119+02'00'",
    "/Producer": "ReportLab PDF Library - www.reportlab.com",
    "/Subject": "SYNTHETIC TEST DATA – NOT A REAL PERSON",
    "/Title": "Synthetic customer insurance test - Lara Neumann",
    "/Trapped": "/False"
  },
  "synthetic_data_confirmed": true,
  "render": {
    "command": [
      "C:\\Users\\mirae\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\native\\poppler\\Library\\bin\\pdftoppm.exe",
      "-png",
      "-r",
      "130",
      "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tests\\fixtures\\synthetic_customer_insurance_lara_neumann_en.pdf",
      "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tmp\\synthetic_customer_scenario_en_002_20260716_202723\\rendered\\synthetic_customer"
    ],
    "return_code": 0,
    "duration_seconds": 1.0927873999999065,
    "stdout": "",
    "stderr": "",
    "images": [
      "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tmp\\synthetic_customer_scenario_en_002_20260716_202723\\rendered\\synthetic_customer-1.png",
      "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tmp\\synthetic_customer_scenario_en_002_20260716_202723\\rendered\\synthetic_customer-2.png",
      "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tmp\\synthetic_customer_scenario_en_002_20260716_202723\\rendered\\synthetic_customer-3.png"
    ],
    "successful": true
  },
  "loader": "api.rag_service.load_pdf_source -> PyPDFLoader",
  "extracted_page_count": 3,
  "required_phrases_found": {
    "SYNTHETIC TEST DATA – NOT A REAL PERSON": true,
    "Lara Neumann": true,
    "TEST-KD-2026-0001": true,
    "TEST-KFZ-2026-1001": true,
    "TEST-PHV-2026-2001": true,
    "Windshield glass damage": true,
    "partial comprehensive insurance": true,
    "deductible of 150 euros": true,
    "Volkswagen Golf": true,
    "TEST-LN-2026": true
  },
  "required_text_extracted": true,
  "page_metadata_preserved": true,
  "loader_warnings": []
}
```

## 14. Ingestion and chunk metadata
Chunks created: `3`; isolated collection: `synthetic_customer_scenario_en_002_collection`.
- `c207299f-9375-436a-8150-731ef81fe11c`: page=1, chunk=TEST-CUSTOMER-PDF-EN-002-P1-C001, insurance_type=customer_profile
- `d49a4fba-1496-411f-b03d-ff873b72655b`: page=2, chunk=TEST-CUSTOMER-PDF-EN-002-P2-C002, insurance_type=motor_insurance
- `0e94b515-8bf1-423c-97ec-e600e18c521a`: page=3, chunk=TEST-CUSTOMER-PDF-EN-002-P3-C003, insurance_type=personal_liability

## 15. Retrieval ranking
Status: `PASS`.
- Rank 1: page=2, chunk=TEST-CUSTOMER-PDF-EN-002-P2-C002, insurance_type=motor_insurance
- Rank 2: page=3, chunk=TEST-CUSTOMER-PDF-EN-002-P3-C003, insurance_type=personal_liability
- Rank 3: page=1, chunk=TEST-CUSTOMER-PDF-EN-002-P1-C001, insurance_type=customer_profile

## 16. Reranker scores
Relevant rank/score: `1` / `not_available`. Distractor rank/score: `2` / `not_available`.

## 17. Safety results
```json
{
  "status": "PASS",
  "contract_allowed_in_context": true,
  "contract_allowed_in_output": true,
  "protected_context_pii_redacted": true,
  "coverage_dates_preserved": true,
  "output_has_no_disallowed_pii": true,
  "context_pii": {
    "detected_count": 8,
    "allowed_count": 4,
    "redacted_count": 4,
    "detected_types": {
      "contract_id": 4,
      "generic_id": 1,
      "customer_number": 1,
      "address": 1,
      "date_of_birth": 1
    },
    "allowed_types": {
      "contract_id": 4
    },
    "redacted_types": {
      "generic_id": 1,
      "customer_number": 1,
      "address": 1,
      "date_of_birth": 1
    },
    "items": [
      {
        "pii_type": "contract_id",
        "start": 166,
        "end": 184,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "generic_id",
        "start": 307,
        "end": 314,
        "allowed": false,
        "source": "identifier_format_regex",
        "reason": "identifier_format:ln"
      },
      {
        "pii_type": "contract_id",
        "start": 182,
        "end": 200,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "customer_number",
        "start": 163,
        "end": 180,
        "allowed": false,
        "source": "identifier_label_regex",
        "reason": "identifier_label:customer number"
      },
      {
        "pii_type": "address",
        "start": 190,
        "end": 223,
        "allowed": false,
        "source": "address_label_regex",
        "reason": "address_label"
      },
      {
        "pii_type": "date_of_birth",
        "start": 239,
        "end": 249,
        "allowed": false,
        "source": "dob_label_regex",
        "reason": "date_of_birth_label"
      },
      {
        "pii_type": "contract_id",
        "start": 314,
        "end": 332,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      },
      {
        "pii_type": "contract_id",
        "start": 449,
        "end": 467,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      }
    ]
  },
  "output_pii": {
    "detected_count": 1,
    "allowed_count": 1,
    "redacted_count": 0,
    "detected_types": {
      "contract_id": 1
    },
    "allowed_types": {
      "contract_id": 1
    },
    "redacted_types": {},
    "items": [
      {
        "pii_type": "contract_id",
        "start": 197,
        "end": 215,
        "allowed": true,
        "source": "contract_identifier_format_regex",
        "reason": "allowed_contract_id"
      }
    ]
  },
  "focused_phone_date_validation": {
    "date_probe": "Coverage dates: 01.01.2026, 31.12.2026, 14.05.1988, 2026-01-01, 2026-12-31, 01/01/2026, 12/31/2026.\n1. Motor insurance\n2. Personal liability insurance",
    "date_pii_types": [],
    "dates_not_phone": true,
    "phone_detected": true,
    "phone_sanitized": "Phone: [REDACTED_PHONE]\n1. Motor insurance\n2. Personal liability insurance",
    "list_numbering_preserved": true
  }
}
```

## 18. Self-check raw output
```text
RELEVANT

The context provided is relevant as it contains the necessary information to answer part of the user's question regarding Lara Neumann's insured vehicle coverage for windshield glass damage, including which type of motor insurance covers this (partial comprehensive), what deductible applies per claim ($150 euros), and that these details are related to her specific Motor Insurance contract number TEST-KFZ-2026-1001.
```

## 19. Self-check parsed output
`RELEVANT`; status `PASS`.

## 20. Final answer
```text
Windshield glass damage to Lara Neumann's insured vehicle is covered under partial comprehensive insurance with a deductible of 150 euros per claim. This relates to Motor insurance contract number TEST-KFZ-2026-1001 [synthetic_customer_insurance_lara_neumann_en:1].
```

## 21. Expected-fact comparison
```json
{
  "coverage": "PASS",
  "coverage_type": "PASS",
  "deductible": "PASS",
  "contract": "PASS",
  "customer": "PASS"
}
```
Unsupported/contradictory claims: `[]`.

## 22. Groundedness validation
Score `0.784167` versus calibrated threshold `0.51` using `fact_aware_claim_support_v4`: `PASS`.

## 23. Citation validation
Status `PASS`; count `1`.
- `[synthetic_customer_insurance_lara_neumann_en:1]`: page=2, retrieved=True, supports_all_material_claims=True

## 24. Distractor validation
Relevant motor evidence ranked above liability distractor: `True`. Contradictions: `[]`.

## 25. Audit and telemetry validation
Main audit found: `True`; response language: `English`; contract visible: `True`; status: `PASS`. Full audit and trace are retained in the JSON result.

## 26. Production collection count comparison
```json
{
  "before": {
    "insurance_rag_collection": 8551,
    "insuranceqa_collection": 1248
  },
  "after_ingestion": {
    "insurance_rag_collection": 8551,
    "insuranceqa_collection": 1248
  },
  "after_execution": {
    "insurance_rag_collection": 8551,
    "insuranceqa_collection": 1248
  },
  "after_cleanup": {
    "insurance_rag_collection": 8551,
    "insuranceqa_collection": 1248
  }
}
```
Integrity: `PASS`; differences: `{'insurance_rag_collection': 0, 'insuranceqa_collection': 0}`.

## 27. Cleanup validation
```json
{
  "executed": true,
  "successful": true,
  "temporary_collection_deleted": true,
  "temporary_directory_deleted": true,
  "synthetic_chunks_remaining": 0,
  "production_documents_modified": false
}
```

## 28. Latency breakdown
```json
{
  "pdf_creation_seconds": 0.08804770000006101,
  "pdf_extraction_seconds": 23.57057739999982,
  "ingestion_seconds": 15.688952399999835,
  "warmup_seconds": 24.756221199999345,
  "retrieval_service_initialization_seconds": 23.91471730000012,
  "main_scenario_seconds": 357.9627068,
  "retrieval_seconds": 1.484380599999895,
  "reranking_seconds": 5.599999894911889e-06,
  "self_check_seconds": 134.93326479999996,
  "answer_generation_seconds": 210.49075119999998,
  "answer_llm_seconds": 210.48469660000046,
  "citation_processing_seconds": 0.0003127999998469022,
  "safety_initialization_seconds": 9.768600100000185,
  "pre_safety_seconds": 0.07362360000024637,
  "context_safety_seconds": 0.032354400000258465,
  "post_safety_seconds": 1.022323299999698,
  "audit_seconds": 0.03915210000013758,
  "safety_seconds": 10.896901400000388,
  "cleanup_seconds": 2.515067199999976
}
```
Execution reliability and user-facing latency are reported separately: `{"execution_reliability": "PASS", "timeout": false, "user_facing_latency_seconds": 357.9627068, "latency_judgement": "MEASURED_NOT_ASSUMED_ACCEPTABLE"}`.

## 29. Test commands and results
| Command | Exit code | Status | Duration |
|---|---:|---|---:|
| `C:\Users\mirae\MVA_Versicherung_Langchain_main\.venv\Scripts\python.exe -m py_compile src/api/rag_service.py src/config/models.py src/core/safety_audit.py src/guardrails/integrations/nemo_actions.py src/utils/language.py scripts/calibrate_groundedness.py scripts/test_synthetic_customer_scenario.py tests/unit/test_safety_pii_rules.py tests/unit/test_response_language.py tests/unit/test_groundedness_calibration.py tests/unit/test_thesis_eval_metadata.py` | 0 | PASS | 0.660s |
| `C:\Users\mirae\MVA_Versicherung_Langchain_main\.venv\Scripts\python.exe -m pytest -q tests/unit/test_safety_pii_rules.py tests/unit/test_response_language.py tests/unit/test_groundedness_calibration.py` | 0 | PASS | 19.004s |
| `C:\Users\mirae\MVA_Versicherung_Langchain_main\.venv\Scripts\python.exe scripts/calibrate_groundedness.py --check` | 0 | PASS | 14.359s |
| `C:\Users\mirae\MVA_Versicherung_Langchain_main\.venv\Scripts\python.exe -m pytest -q tests/unit` | 0 | PASS | 188.999s |

The E2E command itself is `python scripts/test_synthetic_customer_scenario.py`; its result is represented by the table below and the process exit code used by the caller.

## 30. Final PASS/FAIL table
| Category | Result |
|---|---|
| Functional | PASS |
| Quality | PASS |
| Safety | PASS |
| Isolation | PASS |
| Cleanup | PASS |
| Telemetry | PASS |
| Groundedness Calibration | PASS |
| Performance | PASS |
| Overall | PASS |

Failure stage: `None`. Failure reason: `None`.

## 31. Remaining risks and limitations
Request ID and audit ID are not emitted by the current public API and are recorded as not_available. PDF page metadata is zero-based in PyPDFLoader; human page 2 is metadata page 1. The calibrated threshold is empirical for the labeled synthetic insurance set and should be revalidated when document style, language mix, or generation models change. Local CPU Ollama latency is measured but is not labeled acceptable solely because the request avoided timeout.

Artifacts: `{"pdf": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\tests\\fixtures\\synthetic_customer_insurance_lara_neumann_en.pdf", "json": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\reports\\synthetic_customer_scenario_20260716_202723.json", "markdown": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\reports\\synthetic_customer_scenario_20260716_202723.md", "log": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\reports\\synthetic_customer_scenario_20260716_202723.log", "groundedness_calibration_config": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\config\\groundedness_calibration.json", "groundedness_calibration_markdown": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\reports\\groundedness_calibration_20260716_200744.md", "groundedness_calibration_json": "C:\\Users\\mirae\\MVA_Versicherung_Langchain_main\\reports\\groundedness_calibration_20260716_200744.json"}`.

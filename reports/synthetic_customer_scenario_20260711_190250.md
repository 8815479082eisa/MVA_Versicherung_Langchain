# Synthetic Customer Insurance RAG Scenario

## 1. Test objective
Execute a real end-to-end synthetic customer scenario through the current insurance RAG pipeline.

## 2. Test scope
PDF creation, production PDF loading, chunking, embedding, isolated Chroma indexing, hybrid retrieval, reranking, self-check, generation, citations, safety, audit, and cleanup.

## 3. Project and environment
- Project: `C:\Users\mirae\MVA_Versicherung_Langchain_main`
- Test run: `synthetic_customer_scenario_001`
- Timestamp: `2026-07-11T19:02:50.076604+02:00`
- Backend URL: `http://127.0.0.1:6023`
- Backend PID: `13140`
- Ollama URL: `http://127.0.0.1:11434`

## 4. Current model configuration
- Answer: `qwen2.5:7b-instruct`
- Self-check: `phi3:mini`
- Router: `phi3:mini`
- Rewrite: `phi3:mini`
- Compression: `phi3:mini`
- Embedding: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-base`

## 5. Synthetic-data confirmation
`PASS`. All identifiers are fixed fictional test values.

## 6. PDF creation
- Path: `C:\Users\mirae\MVA_Versicherung_Langchain_main\tests\fixtures\synthetic_customer_insurance_lara_neumann.pdf`
- Created: `True`
- Duration: `0.13119740004185587` seconds

## 7. PDF validation
- Valid signature: `True`
- Opened: `True`
- Pages: `3`
- Render successful: `True`

## 8. PDF text extraction
- Loader: `api.rag_service.load_pdf_source -> PyPDFLoader`
- Extracted pages: `3`
- Duration: `8.517460500006564` seconds
- `Lara Neumann`: PASS
- `TEST-KD-2026-0001`: PASS
- `TEST-KFZ-2026-1001`: PASS
- `TEST-PHV-2026-2001`: PASS
- `Windschutzscheibe`: PASS
- `Teilkaskoversicherung`: PASS
- `Selbstbeteiligung von 150 Euro`: PASS
- `Volkswagen Golf`: PASS
- `TEST-LN-2026`: PASS

## 9. Ingestion implementation used
`api.rag_service.load_pdf_source`, `load_and_split_documents`, `initialize_embeddings`, and `build_vectorstore` were reused. The adapter only adds unique synthetic metadata before production embedding and insertion.

## 10. Test-data isolation method
Temporary Chroma directory and collection: `isolated temporary Chroma directory and collection`. Production storage paths were not overridden outside the test process.

## 11. Chunk creation and metadata
- `ddfc6255-8c65-4df9-a4fb-6768c53a3e63` page=0 chunk_id=TEST-CUSTOMER-PDF-001-P1-C001: TEST-CUSTOMER-PDF-001 | synthetic_customer_scenario_001
Seite 1
 SYNTHETIC TEST DATA – NOT A REAL PERSON
 SYNTHETISCHE TESTDATEN – KEINE ECHTE PERSON
Kundenprofil
Name: Lara Neumann
Kundennummer: TEST-KD-2026-0001
Adresse: Musterstraße 17, 00000 Teststadt
Geburtsdatum: 14.05.1988
Aktive Versicherungsverträge:
1. Kfz-Versicherung
Vertragsnummer: TEST-KFZ-2026-1001
Status: Aktiv
Versicherungsbeginn: 01.01.2026
Versicherungsende: 31.12.2026
2. Privathaftpflichtversicherung
Vertragsnummer: TEST-PHV-2026-2001
Status: Aktiv
Versicherungsbeginn: 01.01.2026
Versicherungsende: 31.12.2026
- `ac84abaa-1d23-473a-8be8-6db02f3ae7e1` page=1 chunk_id=TEST-CUSTOMER-PDF-001-P2-C002: TEST-CUSTOMER-PDF-001 | synthetic_customer_scenario_001
Seite 2
Kfz-Versicherung – Vertragsdetails
Versicherte Person: Lara Neumann
Vertragsnummer: TEST-KFZ-2026-1001
Versicherungsart: Kfz-Haftpflicht mit Teilkasko
Versichertes Fahrzeug: Volkswagen Golf
Kennzeichen: TEST-LN-2026
Versicherte Leistungen:
- Schäden an der Windschutzscheibe und anderen Fahrzeugverglasungen sind im Rahmen der
Teilkaskoversicherung versichert.
- Bei einem versicherten Glasschaden gilt eine Selbstbeteiligung von 150 Euro je Schadenfall.
- Reparaturkosten oberhalb der Selbstbeteiligung werden entsprechend den Vertragsbedingungen
übernommen.
Nicht versichert:
- Vorsätzlich verursachte Schäden
- Normale Abnutzung
- Sc
- `e9da42a4-e94b-492a-b189-dad07ca22a80` page=2 chunk_id=TEST-CUSTOMER-PDF-001-P3-C003: TEST-CUSTOMER-PDF-001 | synthetic_customer_scenario_001
Seite 3
Privathaftpflichtversicherung – Vertragsdetails
Versicherte Person: Lara Neumann
Vertragsnummer: TEST-PHV-2026-2001
Die Privathaftpflichtversicherung deckt berechtigte Schadenersatzansprüche Dritter.
Für diesen Vertrag gilt keine allgemeine Selbstbeteiligung.
Schäden an der Windschutzscheibe des eigenen Fahrzeugs gehören nicht zum
Versicherungsschutz der Privathaftpflichtversicherung.

## 12. Collection counts before ingestion
```json
{
  "insurance_rag_collection": 8551,
  "insuranceqa_collection": 1248
}
```

## 13. Collection counts after ingestion
```json
{
  "insurance_rag_collection": 8551,
  "insuranceqa_collection": 1248
}
```

## 14. Preflight checks
```json
{
  "backend_running": true,
  "backend_url": "http://127.0.0.1:6023",
  "backend_pid": 13140,
  "backend_health": {
    "status": "ok",
    "message": "Backend laeuft",
    "pipelineReady": true,
    "pipelineInitializing": false,
    "pipelineInitError": null,
    "insuranceqaExactMatchShortcut": false,
    "safetyEnabled": true,
    "safetyMode": "enforce",
    "configured_answer_model": "qwen2.5:7b-instruct",
    "preferred_answer_model": "qwen2.5:7b-instruct",
    "answer_model_matches_preference": true,
    "configured_answer_model_source": "shell_env",
    "query_rewrite_enabled": false,
    "nemo_enforce_output": true,
    "safety_backend": "nemo"
  },
  "ollama_reachable": true,
  "ollama_url": "http://127.0.0.1:11434",
  "model_inventory": [
    "codellama:latest",
    "phi3:mini",
    "qwen2.5:7b-instruct",
    "qwen2.5:7b",
    "glm-4.7:cloud",
    "llama3:latest"
  ],
  "required_models_installed": true,
  "models_callable": true,
  "retrieval_service_ready": true,
  "pipeline_ready": true,
  "pipeline_initialization_error": null,
  "force_retrieval": true,
  "rewrite_disabled": true,
  "compression_disabled": true,
  "safety_enabled": true,
  "safety_rule_based_without_llm": true,
  "synthetic_document_indexed": true,
  "production_counts": {
    "insurance_rag_collection": 8551,
    "insuranceqa_collection": 1248
  },
  "models_warm": [
    {
      "raw": "NAME                   ID              SIZE      PROCESSOR    CONTEXT    UNTIL"
    },
    {
      "raw": "qwen2.5:7b-instruct    845dbda0ea48    4.6 GB    100% CPU     4096       29 minutes from now"
    }
  ],
  "processes_at_preflight": [
    {
      "ProcessId": 11948,
      "ParentProcessId": 10480,
      "CreationDate": "/Date(1783778310121)/",
      "Name": "ollama app.exe",
      "CommandLine": "\"C:\\Users\\mirae\\AppData\\Local\\Programs\\Ollama\\ollama app.exe\"  --hide --fast-startup"
    },
    {
      "ProcessId": 13872,
      "ParentProcessId": 11948,
      "CreationDate": "/Date(1783778311299)/",
      "Name": "ollama.exe",
      "CommandLine": "C:\\Users\\mirae\\AppData\\Local\\Programs\\Ollama\\ollama.exe serve"
    },
    {
      "ProcessId": 25052,
      "ParentProcessId": 13872,
      "CreationDate": "/Date(1783789418656)/",
      "Name": "ollama.exe",
      "CommandLine": "C:\\Users\\mirae\\AppData\\Local\\Programs\\Ollama\\ollama.exe runner --model C:\\Users\\mirae\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730 --port 5900"
    }
  ],
  "stale_backend_or_mcp_detected": false
}
```

## 15. Warm-up result
```json
{
  "query": "Reply only OK.",
  "type": "direct Ollama one-token model warm-up (not the main scenario)",
  "duration_seconds": 24.07197370007634,
  "results": [
    {
      "model": "phi3:mini",
      "http_status": 200,
      "duration_seconds": 6.122381299966946,
      "response": "OK",
      "done": true,
      "total_duration_ns": 6119322900,
      "load_duration_ns": 4770949700,
      "prompt_eval_count": 14,
      "eval_count": 1,
      "successful": true
    },
    {
      "model": "qwen2.5:7b-instruct",
      "http_status": 200,
      "duration_seconds": 17.948759800055996,
      "response": "OK",
      "done": true,
      "total_duration_ns": 17943569200,
      "load_duration_ns": 11344069800,
      "prompt_eval_count": 33,
      "eval_count": 1,
      "successful": true
    }
  ],
  "successful": true,
  "models_loaded_after": [
    {
      "raw": "NAME                   ID              SIZE      PROCESSOR    CONTEXT    UNTIL"
    },
    {
      "raw": "qwen2.5:7b-instruct    845dbda0ea48    4.6 GB    100% CPU     4096       29 minutes from now"
    }
  ]
}
```

## 16. Main query
`Ist ein Glasschaden an der Windschutzscheibe von Lara Neumanns versichertem Fahrzeug gedeckt, und welche Selbstbeteiligung gilt?`

## 17. Full pipeline execution path
`POST /api/ask`

## 18. Retrieved sources
- none

## 19. Retrieved pages and chunks
Relevant page rank: `None`; distractor rank: `None`.

## 20. Retrieval result
`FAIL`; correct source: `False`; correct page: `False`.

## 21. Reranking result
`FAIL`; relevant score: `None`; distractor score: `None`.

## 22. Self-check raw output
```text
not_available
```

## 23. Self-check parsed decision
`not_available` (`FAIL`).

## 24. Query rewrite status
Applied: `None`; retries: `None`.

## 25. Final answer
```text

```

## 26. Expected-fact comparison
```json
null
```

## 27. Contradiction checks
- not_available

## 28. Groundedness validation
Status: `None`. Supported claims: `None`. Unsupported claims: `None`.

## 29. Citation validation
Status: `FAIL`; count: `0`.
- none

## 30. Distractor-handling validation
Handled correctly: `None`.

## 31. Safety results
```json
{}
```

## 32. Audit result
```json
{
  "enabled": true,
  "entry_found": false,
  "status": "FAIL"
}
```

## 33. Available stage timings
```json
{
  "pdf_creation_seconds": 0.13119740004185587,
  "pdf_extraction_seconds": 8.517460500006564,
  "ingestion_seconds": 14.906154199969023,
  "warmup_seconds": 24.07197370007634,
  "retrieval_service_initialization_seconds": 18.35338089999277,
  "cleanup_seconds": 19.92289719998371
}
```

## 34. Timeout and error information
- Timeout: `False`
- HTTP status: `None`
- Error: `None`

## 35. Cleanup actions
Executed: `True`; temporary collection deleted: `True`; temporary directory deleted: `False`.

## 36. Collection counts after cleanup
```json
{
  "insurance_rag_collection": 8551,
  "insuranceqa_collection": 1248
}
```

## 37. Collection-integrity result
`PASS`; differences: `{'insurance_rag_collection': 0, 'insuranceqa_collection': 0}`.

## 38. Functional PASS or FAIL
`FAIL`

## 39. Quality PASS or FAIL
`FAIL`

## 40. Performance PASS or FAIL
`FAIL`

## 41. Overall PASS or FAIL
`NOT EXECUTED`

## 42. Exact failure reason
Stage: `Cleanup`; reason: `Cleanup or production collection integrity verification failed`.

## 43. Remaining limitations
Request ID and audit ID are not emitted by the current public API and are recorded as not_available. PDF page metadata is zero-based in PyPDFLoader; human page 2 is metadata page 1.

## 44. Recommended next action
Review the exact failing stage in this report.

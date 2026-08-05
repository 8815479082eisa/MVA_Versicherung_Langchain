from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import test_synthetic_customer_scenario as baseline


TEST_NAME = "synthetic_customer_full_corpus_scenario"
TEST_RUN_ID = "synthetic_customer_full_corpus_scenario_001"
DOCUMENT_ID = "TEST-CUSTOMER-PDF-EN-002"
CUSTOMER_ID = "TEST-KD-2026-0001"
PDF_NAME = "synthetic_customer_insurance_lara_neumann_en.pdf"
PDF_PATH = PROJECT_ROOT / "tests" / "fixtures" / PDF_NAME
PRODUCTION_COLLECTION = "insurance_rag_collection"
QUESTION = (
    "Is windshield glass damage to Lara Neumann's insured vehicle covered, under which "
    "type of coverage, what deductible applies per claim, and which motor insurance "
    "contract number does this relate to?"
)
RELEVANT_PHRASE = "deductible of 150 euros"
DISTRACTOR_PHRASE = "No general deductible applies"
NOT_AVAILABLE = "not_available"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def json_safe(value: Any) -> Any:
    return baseline.json_safe(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    baseline.atomic_json(path, payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def close_chroma(client: Any | None = None) -> None:
    if client is not None:
        try:
            client._system.stop()
        except Exception:
            pass
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


def corpus_inventory(chroma_path: Path) -> dict[str, Any]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collections = {collection.name: collection for collection in client.list_collections()}
        counts = {name: collection.count() for name, collection in collections.items()}
        active = collections.get(PRODUCTION_COLLECTION)
        if active is None:
            raise RuntimeError(f"Missing collection: {PRODUCTION_COLLECTION}")

        payload = active.get(include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        source_types: Counter[str] = Counter()
        sources: set[str] = set()
        for metadata in metadatas:
            metadata = metadata or {}
            source_types[str(metadata.get("source_type") or "unknown")] += 1
            source = metadata.get("source")
            if source:
                sources.add(Path(str(source)).name)

        synthetic_ids = active.get(where={"test_run_id": TEST_RUN_ID}, include=[]).get("ids", [])
        return {
            "path": str(chroma_path.resolve()),
            "collection_names": sorted(counts),
            "collection_counts": counts,
            "all_collection_chunks": sum(counts.values()),
            "active_rag_chunks": active.count(),
            "pdf_text_chunks": int(source_types.get("pdf", 0)),
            "pdf_table_chunks": int(source_types.get("pdf_table", 0)),
            "other_chunks": int(
                sum(count for kind, count in source_types.items() if kind not in {"pdf", "pdf_table"})
            ),
            "source_type_counts": dict(source_types),
            "indexed_source_count": len(sources),
            "indexed_source_files": sorted(sources),
            "synthetic_test_chunks": len(synthetic_ids),
        }
    finally:
        close_chroma(client)


def runtime_processes() -> list[dict[str, Any]]:
    return baseline.process_inventory()


def project_backend_or_retrieval_processes() -> list[dict[str, Any]]:
    selected = []
    for process in runtime_processes():
        command = str(process.get("CommandLine") or "").casefold()
        if "backend_api.py" in command or "retrieval_server.py" in command:
            selected.append(process)
    return selected


def stop_existing_project_processes(log: baseline.RunLogger) -> dict[str, Any]:
    processes = project_backend_or_retrieval_processes()
    pids = {int(process["ProcessId"]) for process in processes}
    roots = [
        pid
        for pid in pids
        if next(
            (
                int(item.get("ParentProcessId") or -1)
                for item in processes
                if int(item.get("ProcessId") or -1) == pid
            ),
            -1,
        )
        not in pids
    ]
    outcomes = []
    for pid in sorted(roots):
        log(f"Stopping existing project Backend/MCP process tree PID={pid}")
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        outcomes.append(
            {
                "pid": pid,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )

    deadline = time.monotonic() + 30
    remaining = project_backend_or_retrieval_processes()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.5)
        remaining = project_backend_or_retrieval_processes()
    return {
        "found": processes,
        "root_pids": sorted(roots),
        "outcomes": outcomes,
        "remaining": remaining,
        "successful": not remaining,
    }


def restore_normal_backend(
    normal_env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
    log: baseline.RunLogger,
) -> tuple[subprocess.Popen[Any] | None, dict[str, Any]]:
    import requests

    started = time.perf_counter()
    child_log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "backend_api.py")],
        cwd=str(PROJECT_ROOT),
        stdout=child_log,
        stderr=subprocess.STDOUT,
        env=normal_env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    child_log.close()
    log(f"Started restored normal backend PID={process.pid} URL=http://127.0.0.1:8000")

    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return process, {
                "successful": False,
                "pid": process.pid,
                "duration_seconds": time.perf_counter() - started,
                "error": f"backend exited with code {process.returncode}",
            }
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                ready = (
                    health.get("status") == "ok"
                    and health.get("pipelineReady") is True
                    and health.get("pipelineInitError") is None
                )
                if ready:
                    return process, {
                        "successful": True,
                        "pid": process.pid,
                        "duration_seconds": time.perf_counter() - started,
                        "url": "http://127.0.0.1:8000",
                        "health": health,
                    }
                last_error = f"health returned but pipeline not ready: {health}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    return process, {
        "successful": False,
        "pid": process.pid,
        "duration_seconds": time.perf_counter() - started,
        "error": f"normal backend health timeout: {last_error}",
    }


def document_key(document: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = document.get("metadata", {}) or {}
    return (
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        str(metadata.get("chunk_id", "")),
        str(document.get("content", ""))[:160],
    )


def is_synthetic(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata", {}) or {}
    return bool(metadata.get("synthetic")) or Path(str(metadata.get("source", ""))).name == PDF_NAME


def contains_phrase(document: dict[str, Any], phrase: str) -> bool:
    return baseline.classify_page(document, phrase)


def build_candidate_rows(
    retrieval_event: dict[str, Any] | None,
    rerank_event: dict[str, Any] | None,
    scores_event: dict[str, Any] | None,
    answer_event: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    before = (retrieval_event or {}).get("documents", [])
    after = (rerank_event or {}).get("output_documents", [])
    rerank_inputs = (rerank_event or {}).get("input_documents", [])
    scores = (scores_event or {}).get("scores", [])
    context = (answer_event or {}).get("context_documents", [])
    after_ranks = {document_key(doc): rank for rank, doc in enumerate(after, 1)}
    context_keys = {document_key(doc) for doc in context}
    score_by_key = {
        document_key(doc): float(scores[index])
        for index, doc in enumerate(rerank_inputs)
        if index < len(scores)
    }

    rows = []
    for rank, document in enumerate(before, 1):
        metadata = document.get("metadata", {}) or {}
        key = document_key(document)
        rows.append(
            {
                "rank_before_reranking": rank,
                "rank_after_reranking": after_ranks.get(key, NOT_AVAILABLE),
                "source": str(metadata.get("source", NOT_AVAILABLE)),
                "source_file": Path(str(metadata.get("source", NOT_AVAILABLE))).name,
                "page_zero_based": metadata.get("page", NOT_AVAILABLE),
                "page_human": metadata.get(
                    "page_human",
                    int(metadata["page"]) + 1 if str(metadata.get("page", "")).isdigit() else NOT_AVAILABLE,
                ),
                "chunk_id": metadata.get("chunk_id", NOT_AVAILABLE),
                "retrieval_score": NOT_AVAILABLE,
                "reranker_score": score_by_key.get(key, NOT_AVAILABLE),
                "source_kind": "synthetic" if is_synthetic(document) else "production",
                "included_in_final_context": key in context_keys,
                "is_relevant_synthetic_motor_page": is_synthetic(document)
                and contains_phrase(document, RELEVANT_PHRASE),
                "is_synthetic_liability_distractor": is_synthetic(document)
                and contains_phrase(document, DISTRACTOR_PHRASE),
                "content_preview": str(document.get("content", ""))[:700],
            }
        )
    return rows


def rank_for(rows: list[dict[str, Any]], key: str, stage: str) -> int | str:
    field = "rank_after_reranking" if stage == "after" else "rank_before_reranking"
    match = next((row for row in rows if row.get(key)), None)
    return match.get(field, NOT_AVAILABLE) if match else NOT_AVAILABLE


def result_allow(event: dict[str, Any] | None) -> bool:
    if not event:
        return False
    payload = event.get("result", {}) or {}
    details = payload.get("details", {}) or {}
    return not details.get("nemo_runtime_error") and (
        payload.get("allow") is True or payload.get("action") == "redact"
    )


def fact_validation(answer: str) -> dict[str, Any]:
    baseline_validation = baseline.validate_answer(answer)
    lower = answer.casefold()
    per_claim = bool(re.search(r"\bper\s+(?:insured\s+)?(?:glass\s+)?claim\b", lower))
    facts = {
        "windshield_damage_covered": baseline_validation["coverage_correct"],
        "partial_comprehensive_insurance": baseline_validation["coverage_type_correct"],
        "deductible_150_euros": baseline_validation["deductible_correct"],
        "deductible_per_claim": per_claim,
        "contract_TEST_KFZ_2026_1001": baseline_validation["contract_correct"],
        "lara_neumann_or_insured_vehicle": baseline_validation["customer_correct"],
    }
    return {
        "facts": {name: "PASS" if passed else "FAIL" for name, passed in facts.items()},
        "all_expected_facts_pass": all(facts.values()),
        "contradictions": baseline_validation.get("contradictions", []),
        "no_contradictory_production_information": not baseline_validation.get("contradictions"),
    }


def citation_validation(answer: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    items = baseline.parse_citations(answer, retrieved)
    valid_items = [
        item
        for item in items
        if item.get("retrieved")
        and item.get("supports_all_material_claims")
        and Path(str(item.get("source", ""))).name == PDF_NAME
        and item.get("human_page") == 2
    ]
    return {
        "items": items,
        "count": len(items),
        "synthetic_motor_page_citation_valid": bool(valid_items),
        "production_citations": [
            item for item in items if Path(str(item.get("source", ""))).name != PDF_NAME
        ],
        "status": "PASS" if valid_items else "FAIL",
    }


def focused_safety_validation(rs: Any, synthetic_docs: list[Any]) -> dict[str, Any]:
    from core.safety_audit import detect_pii, sanitize_pii

    text = "\n".join(document.page_content for document in synthetic_docs)
    detections = detect_pii(text, rs.SETTINGS.safety)
    sanitized = sanitize_pii(text, detections, rs.SETTINGS.safety)
    detected_types = Counter(item.pii_type for item in detections)

    date_probe = (
        "Coverage dates: 01.01.2026, 31.12.2026, 14.05.1988, 2026-01-01, "
        "2026-12-31, 01/01/2026, 12/31/2026.\n"
        "1. Motor insurance\n2. Personal liability insurance"
    )
    phone_probe = "+49 151 23456789\n1. Motor insurance\n2. Personal liability insurance"
    date_detections = detect_pii(date_probe, rs.SETTINGS.safety)
    phone_detections = detect_pii(phone_probe, rs.SETTINGS.safety)
    phone_sanitized = sanitize_pii(phone_probe, phone_detections, rs.SETTINGS.safety)
    checks = {
        "contract_number_detected": detected_types.get("contract_id", 0) >= 1,
        "contract_number_remains_allowed": "TEST-KFZ-2026-1001" in sanitized
        and "[REDACTED_CONTRACT_ID]" not in sanitized,
        "customer_number_protected": "TEST-KD-2026-0001" not in sanitized,
        "address_protected": "17 Sample Street, 00000 Test City" not in sanitized,
        "date_of_birth_protected": "14.05.1988" not in sanitized,
        "contract_dates_not_phones": not any(item.pii_type == "phone" for item in date_detections),
        "actual_phone_detected": any(item.pii_type == "phone" for item in phone_detections),
        "list_newlines_preserved": "1. Motor insurance\n2. Personal liability insurance"
        in phone_sanitized,
    }
    return {
        "detected_type_counts": dict(detected_types),
        "checks": {name: "PASS" if passed else "FAIL" for name, passed in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No retrieval candidates were recorded."
    lines = [
        "| Pre | Post | Source | Page | Chunk | Retrieval score | Reranker score | Kind | In context |",
        "|---:|---:|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        source = str(row.get("source_file", NOT_AVAILABLE)).replace("|", "\\|")
        lines.append(
            "| {pre} | {post} | {source} | {page} | {chunk} | {retrieval} | {reranker} | {kind} | {context} |".format(
                pre=row.get("rank_before_reranking", NOT_AVAILABLE),
                post=row.get("rank_after_reranking", NOT_AVAILABLE),
                source=source,
                page=row.get("page_human", NOT_AVAILABLE),
                chunk=row.get("chunk_id", NOT_AVAILABLE),
                retrieval=row.get("retrieval_score", NOT_AVAILABLE),
                reranker=row.get("reranker_score", NOT_AVAILABLE),
                kind=row.get("source_kind", NOT_AVAILABLE),
                context="yes" if row.get("included_in_final_context") else "no",
            )
        )
    return "\n".join(lines)


def markdown_report(result: dict[str, Any]) -> str:
    baseline_corpus = result.get("production_baseline", {})
    temporary = result.get("temporary_corpus", {})
    retrieval = result.get("retrieval", {})
    reranker = result.get("reranker", {})
    safety = result.get("safety", {})
    performance = result.get("performance", {})
    candidates = retrieval.get("candidates", [])
    production_nearby = [row for row in candidates if row.get("source_kind") == "production"]
    facts = result.get("answer_validation", {}).get("facts", {})
    fact_lines = "\n".join(f"- `{name}`: **{status}**" for name, status in facts.items())
    category_lines = "\n".join(
        f"- **{name}**: {status}" for name, status in result.get("results", {}).items()
    )

    return f"""# Synthetic Customer Full-Corpus Scenario

## 1. Objective
Run the existing English synthetic customer query through the real API while all indexed production RAG chunks compete for retrieval.

## 2. Difference from the previous isolated three-chunk test
The prior scenario contained only three synthetic chunks. This scenario starts from a complete temporary copy of the production Chroma persist directory and adds the same three chunks to the copied `{PRODUCTION_COLLECTION}` collection.

## 3. Selected isolation strategy
`{result.get('isolation', {}).get('strategy', NOT_AVAILABLE)}`

Reason: `{result.get('isolation', {}).get('reason', NOT_AVAILABLE)}`

Production path: `{baseline_corpus.get('path', NOT_AVAILABLE)}`  
Temporary path: `{temporary.get('path', NOT_AVAILABLE)}`

## 4. Production corpus baseline
```json
{json.dumps(baseline_corpus, ensure_ascii=False, indent=2)}
```

Active runtime configuration:

```json
{json.dumps(result.get('runtime_configuration', {}), ensure_ascii=False, indent=2)}
```

## 5. Temporary full-corpus construction
Count before synthetic insertion: `{temporary.get('active_count_before_synthetic', NOT_AVAILABLE)}`.  
Count after insertion: `{temporary.get('active_count_after_synthetic', NOT_AVAILABLE)}`.  
Copied production counts match: `{temporary.get('copied_counts_match', False)}`.

## 6. Synthetic PDF ingestion
PDF: `{result.get('synthetic_pdf', {}).get('path', NOT_AVAILABLE)}`  
Pages: `{result.get('synthetic_pdf', {}).get('pages', NOT_AVAILABLE)}`  
Chunks added: `{result.get('synthetic_pdf', {}).get('chunks_added', NOT_AVAILABLE)}`  
Metadata preserved: `{result.get('synthetic_pdf', {}).get('metadata_preserved', False)}`.

## 7. Query
`{QUESTION}`

Execution path: `POST /api/ask`.

## 8. Retrieval candidates
{markdown_table(candidates)}

Retrieval scores are `not_available` because the current hybrid retriever returns LangChain `Document` objects without score values.

## 9. Synthetic relevant-page rank
Before reranking: `{retrieval.get('relevant_rank_before', NOT_AVAILABLE)}`.  
After reranking: `{retrieval.get('relevant_rank_after', NOT_AVAILABLE)}`.  
Included in final context: `{retrieval.get('relevant_in_final_context', False)}`.

## 10. Synthetic distractor rank
Before reranking: `{retrieval.get('distractor_rank_before', NOT_AVAILABLE)}`.  
After reranking: `{retrieval.get('distractor_rank_after', NOT_AVAILABLE)}`.

## 11. Production documents retrieved near the synthetic document
```json
{json.dumps(production_nearby, ensure_ascii=False, indent=2)}
```

## 12. Reranker status and fallback status
```json
{json.dumps(reranker, ensure_ascii=False, indent=2)}
```

## 13. Context used for answer generation
```json
{json.dumps(result.get('answer_context', []), ensure_ascii=False, indent=2)}
```

## 14. Self-check result
```json
{json.dumps(result.get('self_check', {}), ensure_ascii=False, indent=2)}
```

## 15. Final generated answer
{result.get('final_answer', NOT_AVAILABLE)}

## 16. Expected-fact comparison
{fact_lines or 'No fact validation was recorded.'}

Contradictions: `{result.get('answer_validation', {}).get('contradictions', [])}`.

## 17. Citation validation
```json
{json.dumps(result.get('citation', {}), ensure_ascii=False, indent=2)}
```

## 18. Groundedness result
```json
{json.dumps(result.get('groundedness', {}), ensure_ascii=False, indent=2)}
```

## 19. Safety result
Main status: `{safety.get('status', NOT_AVAILABLE)}`.

```json
{json.dumps(safety, ensure_ascii=False, indent=2)}
```

## 20. Telemetry result
```json
{json.dumps(result.get('telemetry', {}), ensure_ascii=False, indent=2)}
```

## 21. Performance breakdown
```json
{json.dumps(performance, ensure_ascii=False, indent=2)}
```

Successful execution, absence of timeout, and user-facing latency assessment are reported separately.

## 22. Production count comparison
Before: `{result.get('production_integrity', {}).get('before_counts', {})}`  
After: `{result.get('production_integrity', {}).get('after_counts', {})}`  
Differences: `{result.get('production_integrity', {}).get('differences', {})}`.

## 23. Cleanup validation
```json
{json.dumps(result.get('cleanup', {}), ensure_ascii=False, indent=2)}
```

## 24. Final result
{category_lines}

## 25. Limitations and recommended next step
{result.get('limitations', NOT_AVAILABLE)}

Recommended next step: {result.get('recommended_next_step', NOT_AVAILABLE)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic full-corpus E2E scenario.")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def execute(args: argparse.Namespace) -> int:
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(args.output_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{TEST_NAME}_{timestamp_slug}.json"
    md_path = reports_dir / f"{TEST_NAME}_{timestamp_slug}.md"
    log_path = reports_dir / f"{TEST_NAME}_{timestamp_slug}.log"
    log = baseline.RunLogger(log_path)

    temp_root = PROJECT_ROOT / "tmp" / f"{TEST_NAME}_{timestamp_slug}"
    temp_chroma = temp_root / "chroma_db"
    temp_pdf_dir = temp_root / "pdfs"
    temp_pdf = temp_pdf_dir / PDF_NAME
    temp_audit = temp_root / "audit.jsonl"
    temp_trace = temp_root / "trace.jsonl"
    temp_model_config = temp_root / "model_config.json"
    temp_pdf_hashes = temp_root / "pdf_hashes.json"

    dotenv = dotenv_values(PROJECT_ROOT / ".env")
    production_chroma = Path(
        str(dotenv.get("CHROMA_PERSIST_DIRECTORY") or "data/processed/vectorstores/chroma_db")
    )
    if not production_chroma.is_absolute():
        production_chroma = (PROJECT_ROOT / production_chroma).resolve()
    production_pdf_dir = Path(str(dotenv.get("PDF_DIRECTORY") or "data/raw/pdfs"))
    if not production_pdf_dir.is_absolute():
        production_pdf_dir = (PROJECT_ROOT / production_pdf_dir).resolve()

    original_env = os.environ.copy()
    env_hash_before = sha256_file(PROJECT_ROOT / ".env")
    production_pdf_names_before = sorted(path.name for path in production_pdf_dir.glob("*.pdf"))
    isolated_process: subprocess.Popen[Any] | None = None
    restored_process: subprocess.Popen[Any] | None = None
    vector_store: Any | None = None
    embeddings: Any | None = None
    splits: list[Any] | None = None
    child_log_handle: Any | None = None

    result: dict[str, Any] = {
        "test_name": TEST_NAME,
        "test_run_id": TEST_RUN_ID,
        "timestamp": utc_now(),
        "artifacts": {
            "markdown": str(md_path),
            "json": str(json_path),
            "log": str(log_path),
            "synthetic_pdf": str(PDF_PATH),
        },
        "query": {"text": QUESTION, "path": "POST /api/ask", "timeout": False},
        "production_baseline": {},
        "temporary_corpus": {},
        "synthetic_pdf": {"path": str(PDF_PATH)},
        "isolation": {
            "strategy": "complete temporary copy of the stopped production Chroma persist directory",
            "reason": (
                "The 0.269 GiB persist directory is small enough to copy safely; this preserves both "
                "collections, stored embeddings, and HNSW indexes without rebuilding or writing to production."
            ),
        },
        "retrieval": {},
        "reranker": {},
        "answer_context": [],
        "self_check": {},
        "final_answer": "",
        "answer_validation": {},
        "citation": {},
        "groundedness": {},
        "safety": {},
        "telemetry": {},
        "performance": {},
        "production_integrity": {},
        "cleanup": {"executed": False, "successful": False},
        "normal_backend_restoration": {},
        "results": {
            "Corpus preparation": "FAIL",
            "Retrieval": "FAIL",
            "Answer quality": "FAIL",
            "Citation": "FAIL",
            "Groundedness": "FAIL",
            "Safety": "FAIL",
            "Isolation": "FAIL",
            "Cleanup": "FAIL",
            "Telemetry": "FAIL",
            "Runtime stability": "FAIL",
            "Overall": "FAIL",
        },
        "failure_stage": None,
        "failure_reason": None,
        "limitations": (
            "The current hybrid retriever exposes candidate order but not numeric retrieval scores. "
            "This is one observational query, not a corpus-wide recall benchmark."
        ),
        "recommended_next_step": (
            "Repeat this copied-corpus method with a balanced multi-query set and report Recall@k, "
            "MRR, answer accuracy, and latency distributions."
        ),
    }
    atomic_json(json_path, result)

    try:
        log("Starting synthetic full-corpus scenario")
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"English synthetic PDF is missing: {PDF_PATH}")

        from pypdf import PdfReader

        reader = PdfReader(str(PDF_PATH))
        extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
        pdf_valid = len(reader.pages) == 3 and all(
            phrase.casefold() in extracted.casefold()
            for phrase in (
                "Lara Neumann",
                "TEST-KFZ-2026-1001",
                "Windshield glass damage",
                "partial comprehensive insurance",
                "150 euros",
            )
        )
        result["synthetic_pdf"].update(
            {"pages": len(reader.pages), "valid": pdf_valid, "sha256": sha256_file(PDF_PATH)}
        )
        if not pdf_valid:
            raise RuntimeError("The English synthetic PDF is invalid")

        baseline_started = time.perf_counter()
        production_before = corpus_inventory(production_chroma)
        result["performance"]["production_inventory_seconds"] = time.perf_counter() - baseline_started
        result["production_baseline"] = production_before
        result["production_integrity"]["before_counts"] = production_before["collection_counts"]
        if production_before["synthetic_test_chunks"] != 0:
            raise RuntimeError("Production already contains chunks for this full-corpus test run")

        process_stop = stop_existing_project_processes(log)
        result["isolation"]["preexisting_processes"] = process_stop
        if not process_stop["successful"]:
            raise RuntimeError("Could not stop all existing project Backend/MCP processes")

        preparation_started = time.perf_counter()
        temp_root.mkdir(parents=True, exist_ok=False)
        copy_started = time.perf_counter()
        shutil.copytree(production_chroma, temp_chroma)
        copy_duration = time.perf_counter() - copy_started
        temp_pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PDF_PATH, temp_pdf)
        copied_inventory = corpus_inventory(temp_chroma)
        copied_counts_match = (
            copied_inventory["collection_counts"] == production_before["collection_counts"]
        )
        result["temporary_corpus"] = {
            **copied_inventory,
            "active_count_before_synthetic": copied_inventory["active_rag_chunks"],
            "copied_counts_match": copied_counts_match,
            "copy_bytes": directory_size(temp_chroma),
        }
        result["performance"]["temporary_copy_seconds"] = copy_duration
        result["performance"]["preparation_before_ingestion_seconds"] = (
            time.perf_counter() - preparation_started
        )
        if not copied_counts_match:
            raise RuntimeError("Temporary Chroma copy counts do not match production")
        log(
            f"Temporary Chroma copy PASS in {copy_duration:.3f}s: "
            f"{copied_inventory['collection_counts']}"
        )

        test_env = {
            "PDF_DIRECTORY": str(temp_pdf_dir),
            "CHROMA_PERSIST_DIRECTORY": str(temp_chroma),
            "COLLECTION_NAME": PRODUCTION_COLLECTION,
            "AUDIT_LOG_FILE": str(temp_audit),
            "MODEL_CONFIG_FILE": str(temp_model_config),
            "PDF_HASH_FILE": str(temp_pdf_hashes),
            "INIT_PIPELINE_ON_STARTUP": "false",
        }
        os.environ.update(test_env)
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import api.rag_service as rs

        settings = rs.SETTINGS
        result["runtime_configuration"] = {
            "embedding_model": settings.embedding.model,
            "embedding_device": settings.embedding.device,
            "retrieval_method": "hybrid BM25 plus Chroma vector retrieval, followed by reranking",
            "retrieval_top_k": settings.retrieval.top_k,
            "bm25_top_k": settings.retrieval.bm25_k,
            "vector_top_k": settings.retrieval.vector_k,
            "rerank_top_k": settings.reranker.top_k,
            "reranker_model": settings.reranker.model,
            "answer_model": settings.roles.answer,
            "self_check_model": settings.roles.self_check,
            "groundedness_threshold": settings.safety.min_groundedness,
            "groundedness_threshold_source": settings.safety.min_groundedness_source,
            "safety_backend": settings.safety.backend,
            "query_rewrite_enabled": settings.retrieval.query_rewrite_enabled,
            "compression_enabled": settings.retrieval.enable_context_compression,
            "insuranceqa_enabled": os.getenv("USE_INSURANCEQA_DATA", "false"),
            "insuranceqa_mode": os.getenv("INSURANCEQA_RETRIEVAL_MODE", "off"),
        }

        ingestion_started = time.perf_counter()
        loaded_docs = rs.load_pdf_source(str(temp_pdf))
        splits = rs.load_and_split_documents([str(temp_pdf)])
        for index, document in enumerate(splits, 1):
            page = int(document.metadata.get("page", -1))
            insurance_type = (
                "motor_insurance"
                if page == 1
                else "personal_liability"
                if page == 2
                else "customer_profile"
            )
            contract_number = (
                "TEST-KFZ-2026-1001"
                if page == 1
                else "TEST-PHV-2026-2001"
                if page == 2
                else "multiple"
            )
            document.metadata.update(
                {
                    "document_id": DOCUMENT_ID,
                    "test_run_id": TEST_RUN_ID,
                    "customer_id": CUSTOMER_ID,
                    "document_type": "synthetic_customer_full_corpus_test",
                    "source": str(temp_pdf),
                    "source_filename": PDF_NAME,
                    "contract_number": contract_number,
                    "page_human": page + 1,
                    "chunk_id": f"{DOCUMENT_ID}-FULL-P{page + 1}-C{index:03d}",
                    "insurance_type": insurance_type,
                    "synthetic": True,
                }
            )

        if len(splits) != 3:
            raise RuntimeError(f"Expected exactly 3 synthetic chunks, got {len(splits)}")
        embeddings = rs.initialize_embeddings()
        vector_store = rs.build_vectorstore(
            [], embeddings, force_reindex=False, allow_reindex=True
        )
        synthetic_ids = [f"{TEST_RUN_ID}-{index:03d}" for index in range(1, len(splits) + 1)]
        vector_store.add_documents(documents=splits, ids=synthetic_ids)
        rs._save_model_config()
        rs.save_pdf_hashes(rs.get_pdf_hashes([str(temp_pdf)]))
        ingestion_duration = time.perf_counter() - ingestion_started
        result["performance"]["synthetic_ingestion_seconds"] = ingestion_duration

        temp_after = corpus_inventory(temp_chroma)
        metadata_preserved = temp_after["synthetic_test_chunks"] == 3
        expected_count = production_before["active_rag_chunks"] + 3
        result["temporary_corpus"].update(
            {
                "active_count_after_synthetic": temp_after["active_rag_chunks"],
                "expected_active_count_after_synthetic": expected_count,
                "synthetic_chunks_after_ingestion": temp_after["synthetic_test_chunks"],
            }
        )
        result["synthetic_pdf"].update(
            {
                "ingestion_path": str(temp_pdf),
                "loaded_pages": len(loaded_docs),
                "chunks_added": len(splits),
                "chunk_ids": synthetic_ids,
                "chunk_metadata": [dict(document.metadata) for document in splits],
                "metadata_preserved": metadata_preserved,
            }
        )
        corpus_preparation_pass = (
            copied_counts_match
            and temp_after["active_rag_chunks"] == expected_count
            and metadata_preserved
        )
        result["results"]["Corpus preparation"] = "PASS" if corpus_preparation_pass else "FAIL"
        if not corpus_preparation_pass:
            raise RuntimeError("Temporary full-corpus validation failed")
        log(
            f"Synthetic ingestion PASS in {ingestion_duration:.3f}s: "
            f"{production_before['active_rag_chunks']} -> {temp_after['active_rag_chunks']}"
        )

        result["safety"]["focused_validation"] = focused_safety_validation(rs, loaded_docs)

        inventory = baseline.ollama_inventory(settings.ollama_base_url)
        warmup_started = time.perf_counter()
        warmup = [
            baseline.ollama_request(settings.ollama_base_url, model, "Reply only OK.", 1)
            for model in dict.fromkeys((settings.roles.self_check, settings.roles.answer))
        ]
        result["performance"]["model_warmup_seconds_not_in_main_query"] = (
            time.perf_counter() - warmup_started
        )
        result["preflight"] = {
            "ollama_inventory": inventory,
            "model_warmup": warmup,
            "required_models_available": {
                settings.roles.self_check,
                settings.roles.answer,
            }.issubset(set(inventory.get("models", []))),
        }
        if not result["preflight"]["required_models_available"] or not all(
            item.get("successful") for item in warmup
        ):
            raise RuntimeError("Ollama model preflight failed")

        del vector_store, embeddings, splits, loaded_docs
        vector_store = None
        embeddings = None
        splits = None
        close_chroma()

        port = baseline.free_port()
        backend_url = f"http://127.0.0.1:{port}"
        server_started = time.perf_counter()
        child_log_handle = log_path.open("a", encoding="utf-8")
        isolated_process = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "test_synthetic_customer_scenario.py"),
                "--serve",
                "--port",
                str(port),
                "--trace-path",
                str(temp_trace),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=child_log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        child_log_handle.close()
        child_log_handle = None
        health = baseline.wait_for_health(
            backend_url + "/health", isolated_process, timeout=args.startup_timeout_seconds
        )
        backend_startup_duration = time.perf_counter() - server_started
        result["performance"]["isolated_backend_startup_seconds"] = backend_startup_duration
        result["isolated_backend"] = {
            "url": backend_url,
            "pid": isolated_process.pid,
            "health": health,
        }

        trace_events = baseline.read_trace(temp_trace)
        server_preflight = baseline.latest_event(trace_events, "server_preflight") or {}
        result["server_preflight"] = server_preflight
        if health.get("pipelineReady") is not True or not server_preflight:
            raise RuntimeError("Isolated full-corpus backend did not initialize the real pipeline")
        log(f"Isolated full-corpus backend ready in {backend_startup_duration:.3f}s")

        import requests

        query_started = time.perf_counter()
        try:
            response = requests.post(
                backend_url + "/api/ask",
                json={"question": QUESTION, "shortAnswer": False, "structuredAnswer": False},
                timeout=args.timeout_seconds,
            )
            api_duration = time.perf_counter() - query_started
            result["query"].update(
                {
                    "http_status": response.status_code,
                    "timeout": False,
                    "duration_seconds": api_duration,
                    "raw_response": response.text,
                }
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"POST /api/ask returned HTTP {response.status_code}: {response.text[:500]}"
                )
            api_payload = response.json()
            result["query"]["api_response"] = api_payload
            result["query"]["successful_execution"] = True
        except requests.Timeout:
            result["query"].update(
                {
                    "timeout": True,
                    "duration_seconds": time.perf_counter() - query_started,
                    "successful_execution": False,
                }
            )
            raise
        result["performance"]["total_api_latency_seconds"] = api_duration
        log(f"Main POST /api/ask completed in {api_duration:.3f}s")

        time.sleep(1)
        trace_events = baseline.read_trace(temp_trace)
        result["trace_events"] = trace_events
        retrieval_event = baseline.latest_event(trace_events, "retrieval_search")
        rerank_event = baseline.latest_event(trace_events, "reranking")
        scores_event = baseline.latest_event(trace_events, "reranker_scores")
        self_event = baseline.latest_event(trace_events, "self_check_llm")
        answer_llm_event = baseline.latest_event(trace_events, "answer_llm")
        answer_event = baseline.latest_event(trace_events, "answer_generation")
        citation_event = baseline.latest_event(trace_events, "citation_processing")
        safety_pre = baseline.latest_event(trace_events, "safety_pre_query")
        safety_context = baseline.latest_event(trace_events, "safety_context")
        safety_post = baseline.latest_event(trace_events, "safety_post_generation")
        audit_event = baseline.latest_event(trace_events, "audit")

        candidates = build_candidate_rows(
            retrieval_event, rerank_event, scores_event, answer_event
        )
        relevant_row = next(
            (row for row in candidates if row["is_relevant_synthetic_motor_page"]), None
        )
        distractor_row = next(
            (row for row in candidates if row["is_synthetic_liability_distractor"]), None
        )
        relevant_in_context = bool(
            relevant_row and relevant_row.get("included_in_final_context")
        )
        result["retrieval"] = {
            "candidate_count": len(candidates),
            "candidates": candidates,
            "relevant_rank_before": rank_for(
                candidates, "is_relevant_synthetic_motor_page", "before"
            ),
            "relevant_rank_after": rank_for(
                candidates, "is_relevant_synthetic_motor_page", "after"
            ),
            "relevant_in_final_context": relevant_in_context,
            "distractor_rank_before": rank_for(
                candidates, "is_synthetic_liability_distractor", "before"
            ),
            "distractor_rank_after": rank_for(
                candidates, "is_synthetic_liability_distractor", "after"
            ),
            "production_candidates": [
                row for row in candidates if row["source_kind"] == "production"
            ],
            "status": "PASS" if relevant_in_context else "FAIL",
        }
        result["results"]["Retrieval"] = result["retrieval"]["status"]

        reranking_executed = scores_event is not None
        fallback_used = rerank_event is not None and not reranking_executed
        result["reranker"] = {
            "model": settings.reranker.model,
            "reranker_initialized": reranking_executed,
            "reranking_executed": reranking_executed,
            "fallback_used": fallback_used,
            "fallback_reason": (
                "Reranker score computation was not observed; normal retrieval-order fallback was used."
                if fallback_used
                else None
            ),
            "candidate_count": (rerank_event or {}).get("candidate_count", NOT_AVAILABLE),
            "output_count": (rerank_event or {}).get("final_count", NOT_AVAILABLE),
            "duration_seconds": (rerank_event or {}).get("duration_seconds", NOT_AVAILABLE),
        }

        answer_context = (answer_event or {}).get("context_documents", [])
        result["answer_context"] = [
            {
                "rank": index,
                "source": (document.get("metadata", {}) or {}).get("source", NOT_AVAILABLE),
                "page": (document.get("metadata", {}) or {}).get("page", NOT_AVAILABLE),
                "chunk_id": (document.get("metadata", {}) or {}).get(
                    "chunk_id", NOT_AVAILABLE
                ),
                "source_kind": "synthetic" if is_synthetic(document) else "production",
                "content": document.get("content", ""),
            }
            for index, document in enumerate(answer_context, 1)
        ]

        raw_self = str((self_event or {}).get("raw_output", NOT_AVAILABLE))
        parsed_self = rs._parse_self_check_decision(raw_self) if self_event else NOT_AVAILABLE
        result["self_check"] = {
            "model": settings.roles.self_check,
            "raw_output": raw_self,
            "parsed_decision": parsed_self,
            "duration_seconds": (self_event or {}).get("duration_seconds", NOT_AVAILABLE),
            "status": "PASS" if parsed_self == "RELEVANT" else "FAIL",
        }

        final_answer = str(result["query"]["api_response"].get("answer", ""))
        result["final_answer"] = final_answer
        answer_validation = fact_validation(final_answer)
        result["answer_validation"] = answer_validation
        answer_pass = (
            answer_validation["all_expected_facts_pass"]
            and answer_validation["no_contradictory_production_information"]
        )
        result["results"]["Answer quality"] = "PASS" if answer_pass else "FAIL"

        retrieved_documents = (retrieval_event or {}).get("documents", [])
        citation = citation_validation(final_answer, retrieved_documents)
        result["citation"] = citation
        result["results"]["Citation"] = citation["status"]

        post_payload = (safety_post or {}).get("result", {}) or {}
        groundedness_score = (post_payload.get("scores", {}) or {}).get(
            "groundedness", NOT_AVAILABLE
        )
        groundedness_threshold = settings.safety.min_groundedness
        groundedness_reasons = post_payload.get("reasons", []) or []
        groundedness_pass = (
            isinstance(groundedness_score, (int, float))
            and groundedness_score >= groundedness_threshold
            and "low_groundedness" not in groundedness_reasons
        )
        result["groundedness"] = {
            "score": groundedness_score,
            "threshold": groundedness_threshold,
            "threshold_source": settings.safety.min_groundedness_source,
            "reasons": groundedness_reasons,
            "unsupported_or_partially_supported_claims": answer_validation.get(
                "contradictions", []
            ),
            "status": "PASS" if groundedness_pass else "FAIL",
        }
        result["results"]["Groundedness"] = result["groundedness"]["status"]

        context_guardrail = baseline.guardrail_details(safety_context, "context")
        output_guardrail = baseline.guardrail_details(safety_post, "output")
        context_pii = context_guardrail.get("pii", {}) or {}
        output_pii = output_guardrail.get("pii", {}) or {}
        context_allowed = context_pii.get("allowed_types", {}) or {}
        context_redacted = context_pii.get("redacted_types", {}) or {}
        output_allowed = output_pii.get("allowed_types", {}) or {}
        output_redacted = output_pii.get("redacted_types", {}) or {}
        contract_allowed_context = (
            int(context_allowed.get("contract_id", 0)) >= 1
            and int(context_redacted.get("contract_id", 0)) == 0
        )
        contract_allowed_output = (
            "TEST-KFZ-2026-1001" in final_answer
            and int(output_redacted.get("contract_id", 0)) == 0
        )
        focused_safety = result["safety"]["focused_validation"]
        safety_pass = all(
            (
                result_allow(safety_pre),
                result_allow(safety_context),
                result_allow(safety_post),
                contract_allowed_context,
                contract_allowed_output,
                not output_redacted,
                focused_safety["status"] == "PASS",
            )
        )
        result["safety"].update(
            {
                "backend": settings.safety.backend,
                "pre_query_successful": result_allow(safety_pre),
                "context_successful": result_allow(safety_context),
                "output_successful": result_allow(safety_post),
                "context_pii": context_pii,
                "output_pii": output_pii,
                "contract_allowed_in_context": contract_allowed_context,
                "contract_allowed_in_output": contract_allowed_output,
                "final_answer_has_no_disallowed_pii": not output_redacted,
                "status": "PASS" if safety_pass else "FAIL",
            }
        )
        result["results"]["Safety"] = result["safety"]["status"]

        audit_rows = []
        if temp_audit.exists():
            for line in temp_audit.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("query") == QUESTION:
                    audit_rows.append(row)
        audit_row = audit_rows[-1] if audit_rows else None
        telemetry_pass = bool(
            audit_row
            and audit_row.get("response_language") == "English"
            and "TEST-KFZ-2026-1001" in json.dumps(audit_row, ensure_ascii=False)
        )
        result["telemetry"] = {
            "audit_event_recorded": audit_event is not None,
            "audit_row_found": audit_row is not None,
            "response_language": (
                audit_row.get("response_language") if audit_row else NOT_AVAILABLE
            ),
            "contract_visible_in_audit": bool(
                audit_row
                and "TEST-KFZ-2026-1001" in json.dumps(audit_row, ensure_ascii=False)
            ),
            "audit_row": audit_row or NOT_AVAILABLE,
            "status": "PASS" if telemetry_pass else "FAIL",
        }
        result["results"]["Telemetry"] = result["telemetry"]["status"]

        result["performance"].update(
            {
                "retrieval_seconds": (retrieval_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "reranking_seconds": (rerank_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "context_safety_seconds": (safety_context or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "self_check_seconds": (self_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "answer_generation_seconds": (answer_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "answer_llm_seconds": (answer_llm_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "output_safety_seconds": (safety_post or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "input_safety_seconds": (safety_pre or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "citation_processing_seconds": (citation_event or {}).get(
                    "duration_seconds", NOT_AVAILABLE
                ),
                "successful_execution": True,
                "no_timeout": not result["query"]["timeout"],
                "user_facing_latency_assessment": (
                    "NOT_ACCEPTABLE_FOR_INTERACTIVE_USE"
                    if api_duration >= 30
                    else "WITHIN_30_SECOND_OBSERVATIONAL_BOUND"
                ),
            }
        )
        result["limitations"] += (
            f" This run completed without timeout but required {api_duration:.3f}s, including "
            f"{result['performance']['self_check_seconds']:.3f}s for self-check and "
            f"{result['performance']['answer_generation_seconds']:.3f}s for answer generation; "
            "that is not acceptable interactive latency."
        )
        if parsed_self == "RELEVANT" and "does not provide information about which type of coverage" in raw_self:
            result["limitations"] += (
                " The self-check decision token was RELEVANT, but its explanatory tail contained "
                "internally inconsistent statements about evidence that was visibly present in context."
            )
        runtime_pass = result["query"].get("successful_execution") is True and not result["query"][
            "timeout"
        ]
        result["results"]["Runtime stability"] = "PASS" if runtime_pass else "FAIL"

        production_during = corpus_inventory(production_chroma)
        isolation_pass = (
            production_during["collection_counts"] == production_before["collection_counts"]
            and production_during["synthetic_test_chunks"] == 0
            and temp_chroma.resolve() != production_chroma.resolve()
            and PDF_NAME not in production_pdf_names_before
        )
        result["production_integrity"]["during_test_counts"] = production_during[
            "collection_counts"
        ]
        result["results"]["Isolation"] = "PASS" if isolation_pass else "FAIL"
        atomic_json(json_path, result)

    except Exception as exc:
        result["failure_stage"] = result.get("failure_stage") or "full-corpus harness"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["exception_traceback"] = traceback.format_exc()
        if exc.__class__.__name__ == "ReadTimeout":
            result["query"]["timeout"] = True
        log(f"FAILURE: {result['failure_reason']}")
        atomic_json(json_path, result)
    finally:
        cleanup_started = time.perf_counter()
        result["cleanup"]["executed"] = True
        try:
            if isolated_process is not None:
                baseline.stop_process(isolated_process, log)
            if child_log_handle is not None:
                child_log_handle.close()

            vector_store = None
            embeddings = None
            splits = None
            close_chroma()

            temp_synthetic_before_delete = NOT_AVAILABLE
            if temp_chroma.exists():
                try:
                    temp_synthetic_before_delete = corpus_inventory(temp_chroma)[
                        "synthetic_test_chunks"
                    ]
                except Exception as exc:
                    temp_synthetic_before_delete = f"ERROR:{type(exc).__name__}:{exc}"

            temporary_deleted = False
            if temp_root.exists() and not args.keep_temp:
                for attempt in range(6):
                    try:
                        shutil.rmtree(temp_root)
                        temporary_deleted = not temp_root.exists()
                        break
                    except PermissionError:
                        close_chroma()
                        time.sleep(1 + attempt)
            elif args.keep_temp:
                temporary_deleted = False

            os.environ.clear()
            os.environ.update(original_env)

            production_after = corpus_inventory(production_chroma)
            counts_before = result.get("production_integrity", {}).get("before_counts", {})
            counts_after = production_after["collection_counts"]
            differences = {
                name: counts_after.get(name, 0) - count
                for name, count in counts_before.items()
                if isinstance(count, int) and isinstance(counts_after.get(name), int)
            }
            env_unchanged = sha256_file(PROJECT_ROOT / ".env") == env_hash_before
            production_pdf_names_after = sorted(
                path.name for path in production_pdf_dir.glob("*.pdf")
            )
            production_unchanged = (
                counts_before == counts_after
                and all(value == 0 for value in differences.values())
                and production_after["synthetic_test_chunks"] == 0
                and production_pdf_names_before == production_pdf_names_after
                and PDF_NAME not in production_pdf_names_after
                and env_unchanged
            )

            result["production_integrity"].update(
                {
                    "after_counts": counts_after,
                    "differences": differences,
                    "production_synthetic_chunks_after": production_after[
                        "synthetic_test_chunks"
                    ],
                    "production_pdf_directory_unchanged": (
                        production_pdf_names_before == production_pdf_names_after
                    ),
                    "synthetic_pdf_absent_from_production_pdf_directory": (
                        PDF_NAME not in production_pdf_names_after
                    ),
                    "dotenv_unchanged": env_unchanged,
                    "production_unchanged": production_unchanged,
                }
            )

            restore_started = time.perf_counter()
            restored_process, restore_result = restore_normal_backend(
                original_env,
                log_path,
                timeout_seconds=args.startup_timeout_seconds,
                log=log,
            )
            result["normal_backend_restoration"] = restore_result
            result["performance"]["normal_backend_restore_seconds"] = (
                time.perf_counter() - restore_started
            )

            cleanup_success = (
                (temporary_deleted or args.keep_temp)
                and production_unchanged
                and restore_result.get("successful") is True
            )
            result["cleanup"].update(
                {
                    "temporary_synthetic_chunks_before_deletion": temp_synthetic_before_delete,
                    "temporary_directory_deleted": temporary_deleted,
                    "temporary_path_exists_after_cleanup": temp_root.exists(),
                    "production_synthetic_chunks_remaining": production_after[
                        "synthetic_test_chunks"
                    ],
                    "normal_backend_restored": restore_result.get("successful") is True,
                    "successful": cleanup_success,
                }
            )
            result["results"]["Cleanup"] = "PASS" if cleanup_success else "FAIL"
            result["results"]["Isolation"] = (
                "PASS"
                if production_unchanged and result["results"].get("Isolation") == "PASS"
                else "FAIL"
            )
        except Exception as cleanup_exc:
            result["cleanup"].update(
                {
                    "successful": False,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                }
            )
            result["failure_stage"] = "cleanup and restoration"
            result["failure_reason"] = result["cleanup"]["error"]
            log(f"CLEANUP FAILURE: {result['cleanup']['error']}")

        result["performance"]["cleanup_and_restoration_seconds"] = (
            time.perf_counter() - cleanup_started
        )
        mandatory = [name for name in result["results"] if name != "Overall"]
        overall_pass = all(result["results"].get(name) == "PASS" for name in mandatory)
        result["results"]["Overall"] = "PASS" if overall_pass else "FAIL"
        if overall_pass:
            result["failure_stage"] = None
            result["failure_reason"] = None

        atomic_json(json_path, result)
        md_path.write_text(markdown_report(result), encoding="utf-8")
        log(
            f"FINAL Overall={result['results']['Overall']} "
            f"production={result.get('production_integrity', {}).get('after_counts')} "
            f"cleanup={result.get('cleanup', {}).get('successful')}"
        )
        log(f"JSON report: {json_path}")
        log(f"Markdown report: {md_path}")

    return 0 if result["results"]["Overall"] == "PASS" else 1


def main() -> int:
    return execute(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

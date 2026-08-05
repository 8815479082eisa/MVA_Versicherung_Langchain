from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUN_ID = "synthetic_customer_scenario_en_002"
DOCUMENT_ID = "TEST-CUSTOMER-PDF-EN-002"
CUSTOMER_ID = "TEST-KD-2026-0001"
PDF_NAME = "synthetic_customer_insurance_lara_neumann_en.pdf"
QUESTION = (
    "Is windshield glass damage to Lara Neumann's insured vehicle covered, under which "
    "type of coverage, what deductible applies per claim, and which motor insurance "
    "contract number does this relate to?"
)
REQUIRED_PHRASES = (
    "SYNTHETIC TEST DATA – NOT A REAL PERSON",
    "Lara Neumann",
    "TEST-KD-2026-0001",
    "TEST-KFZ-2026-1001",
    "TEST-PHV-2026-2001",
    "Windshield glass damage",
    "partial comprehensive insurance",
    "deductible of 150 euros",
    "Volkswagen Golf",
    "TEST-LN-2026",
)
NOT_AVAILABLE = "not_available"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_recorded_command(command: list[str], timeout: int = 300) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": subprocess.list2cmdline(command),
        "exit_code": completed.returncode,
        "duration_seconds": time.perf_counter() - started,
        "stdout": completed.stdout[-6000:],
        "stderr": completed.stderr[-6000:],
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def create_pdf(path: Path) -> float:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    regular_font = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_font = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular_font.exists() and bold_font.exists():
        pdfmetrics.registerFont(TTFont("SyntheticArial", str(regular_font)))
        pdfmetrics.registerFont(TTFont("SyntheticArialBold", str(bold_font)))
        body_font, bold_name = "SyntheticArial", "SyntheticArialBold"
    else:
        body_font, bold_name = "Helvetica", "Helvetica-Bold"

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Synthetic customer insurance test - Lara Neumann",
        author="MVA Insurance RAG synthetic test harness",
        subject=(
            "SYNTHETIC TEST DATA – NOT A REAL PERSON"
        ),
        keywords=f"synthetic,{TEST_RUN_ID},{DOCUMENT_ID},{CUSTOMER_ID}",
    )
    styles = getSampleStyleSheet()
    warning = ParagraphStyle(
        "Warning",
        parent=styles["Heading2"],
        fontName=bold_name,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#A00000"),
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontName=bold_name,
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=7 * mm,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=10.5,
        leading=15,
        spaceAfter=2.5 * mm,
    )
    subheading = ParagraphStyle(
        "Subheading",
        parent=body,
        fontName=bold_name,
        fontSize=11.5,
        leading=16,
        spaceBefore=2 * mm,
        spaceAfter=2 * mm,
    )

    def p(text: str, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(text.replace("\n", "<br/>"), style)

    story = [
        p("SYNTHETIC TEST DATA – NOT A REAL PERSON", warning),
        Spacer(1, 3 * mm),
        p("Customer profile", heading),
        p(
            "<b>Name:</b> Lara Neumann<br/>"
            "<b>Customer number:</b> TEST-KD-2026-0001<br/>"
            "<b>Address:</b> 17 Sample Street, 00000 Test City<br/>"
            "<b>Date of birth:</b> 14.05.1988"
        ),
        p("Active insurance contracts:", subheading),
        p(
            "<b>1. Motor insurance</b><br/>"
            "Contract number: TEST-KFZ-2026-1001<br/>"
            "Status: Active<br/>"
            "Coverage start: 01.01.2026<br/>"
            "Coverage end: 31.12.2026"
        ),
        p(
            "<b>2. Personal liability insurance</b><br/>"
            "Contract number: TEST-PHV-2026-2001<br/>"
            "Status: Active<br/>"
            "Coverage start: 01.01.2026<br/>"
            "Coverage end: 31.12.2026"
        ),
        PageBreak(),
        p("Motor insurance – contract details", heading),
        p(
            "<b>Insured person:</b> Lara Neumann<br/>"
            "<b>Motor insurance contract number:</b> TEST-KFZ-2026-1001<br/>"
            "<b>Insurance type:</b> Motor liability with partial comprehensive insurance<br/>"
            "<b>Insured vehicle:</b> Volkswagen Golf<br/>"
            "<b>License plate:</b> TEST-LN-2026"
        ),
        p("Covered benefits:", subheading),
        p(
            "- Windshield glass damage and damage to other vehicle glass is covered "
            "under partial comprehensive insurance."
        ),
        p(
            "- A deductible of 150 euros applies per insured glass claim."
        ),
        p(
            "- Repair costs above the deductible are paid according to the contract terms."
        ),
        p("Exclusions:", subheading),
        p(
            "- Intentionally caused damage<br/>"
            "- Normal wear and tear<br/>"
            "- Damage that did not occur to the insured vehicle"
        ),
        PageBreak(),
        p("Personal liability insurance – contract details", heading),
        p(
            "<b>Insured person:</b> Lara Neumann<br/>"
            "<b>Personal liability contract number:</b> TEST-PHV-2026-2001"
        ),
        p(
            "Personal liability insurance covers valid third-party damage claims."
        ),
        p("No general deductible applies to this personal liability contract."),
        p(
            "Personal liability insurance does not cover windshield damage to the "
            "insured person's own vehicle."
        ),
    ]

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(22 * mm, 10 * mm, f"{DOCUMENT_ID} | {TEST_RUN_ID}")
        canvas.drawRightString(A4[0] - 22 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return time.perf_counter() - started


def render_pdf(pdf_path: Path, render_dir: Path) -> dict[str, Any]:
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / "synthetic_customer"
    bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
    executable = str(bundled) if bundled.exists() else "pdftoppm"
    command = [executable, "-png", "-r", "130", str(pdf_path), str(prefix)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    images = sorted(str(path) for path in render_dir.glob("synthetic_customer-*.png"))
    return {
        "command": command,
        "return_code": completed.returncode,
        "duration_seconds": time.perf_counter() - started,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "images": images,
        "successful": completed.returncode == 0 and len(images) >= 3,
    }


def production_counts(chroma_path: Path) -> dict[str, Any]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    counts: dict[str, Any] = {}
    for name in ("insurance_rag_collection", "insuranceqa_collection"):
        try:
            counts[name] = client.get_collection(name=name).count()
        except Exception as exc:
            counts[name] = f"ERROR:{type(exc).__name__}:{exc}"
    return counts


def process_inventory() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|ollama' -and "
        "($_.CommandLine -match 'backend_api|uvicorn|mcp|ollama') } | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine | "
        "ConvertTo-Json -Depth 4"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, list) else [payload]


def ollama_request(base_url: str, model: str, prompt: str, num_predict: int = 1) -> dict[str, Any]:
    import requests

    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0},
            "keep_alive": "30m",
        },
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    return {
        "model": model,
        "http_status": response.status_code,
        "duration_seconds": elapsed,
        "response": payload.get("response", ""),
        "done": payload.get("done"),
        "total_duration_ns": payload.get("total_duration"),
        "load_duration_ns": payload.get("load_duration"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "successful": response.status_code == 200 and bool(payload.get("done")),
    }


def ollama_inventory(base_url: str) -> dict[str, Any]:
    import requests

    response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    return {
        "http_status": response.status_code,
        "models": [item.get("name") or item.get("model") for item in models],
        "raw": models,
    }


def ollama_ps() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ollama", "ps"], capture_output=True, text=True, timeout=15
    )
    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    return [{"raw": line} for line in lines]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def doc_payload(doc: Any, rank: int | None = None) -> dict[str, Any]:
    payload = {
        "content": getattr(doc, "page_content", ""),
        "metadata": dict(getattr(doc, "metadata", {}) or {}),
    }
    if rank is not None:
        payload["rank"] = rank
    return json_safe(payload)


class TraceRecorder:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event, **json_safe(fields)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def install_instrumentation(rs: Any, pipeline: Any, recorder: TraceRecorder) -> None:
    from langchain_core.runnables import RunnableLambda

    components = pipeline.components
    retrieval_service = components["retrieval_service"]
    retrieval_components = retrieval_service.components

    original_hybrid = retrieval_components["hybrid_retriever"]

    def traced_hybrid(query: str, k: int | None = None) -> list[Any]:
        started = time.perf_counter()
        docs = original_hybrid(query, k=k)
        recorder.emit(
            "retrieval_search",
            query=query,
            requested_k=k,
            duration_seconds=time.perf_counter() - started,
            documents=[doc_payload(doc, rank) for rank, doc in enumerate(docs, 1)],
        )
        return docs

    retrieval_components["hybrid_retriever"] = traced_hybrid

    original_reranker = retrieval_components.get("reranker_model")
    if original_reranker is not None:

        class RerankerProxy:
            def compute_score(self, pairs: list[list[str]]) -> Any:
                started = time.perf_counter()
                scores = original_reranker.compute_score(pairs)
                score_list = scores if isinstance(scores, list) else [scores]
                recorder.emit(
                    "reranker_scores",
                    duration_seconds=time.perf_counter() - started,
                    candidate_count=len(pairs),
                    scores=[float(score) for score in score_list],
                    candidate_previews=[pair[1][:500] for pair in pairs],
                )
                return scores

        retrieval_components["reranker_model"] = RerankerProxy()

    original_rerank = rs.rerank_documents

    def traced_rerank(query: str, documents: list[Any], reranker_model: Any, top_k: int = 3) -> list[Any]:
        started = time.perf_counter()
        output = original_rerank(query, documents, reranker_model, top_k=top_k)
        recorder.emit(
            "reranking",
            query=query,
            duration_seconds=time.perf_counter() - started,
            candidate_count=len(documents),
            final_count=len(output),
            input_documents=[doc_payload(doc, rank) for rank, doc in enumerate(documents, 1)],
            output_documents=[doc_payload(doc, rank) for rank, doc in enumerate(output, 1)],
        )
        return output

    rs.rerank_documents = traced_rerank

    def traced_model(original: Any, event: str) -> RunnableLambda:
        def invoke(input_value: Any) -> Any:
            started = time.perf_counter()
            response = original.invoke(input_value)
            recorder.emit(
                event,
                duration_seconds=time.perf_counter() - started,
                raw_output=getattr(response, "content", str(response)),
                response_metadata=getattr(response, "response_metadata", {}),
                usage_metadata=getattr(response, "usage_metadata", {}),
            )
            return response

        return RunnableLambda(invoke)

    components["self_check_llm"] = traced_model(
        components["self_check_llm"], "self_check_llm"
    )
    components["llm"] = traced_model(components["llm"], "answer_llm")

    original_citations = rs._ensure_inline_citations

    def traced_citations(answer: str, docs: list[Any]) -> str:
        started = time.perf_counter()
        output = original_citations(answer, docs)
        recorder.emit(
            "citation_processing",
            duration_seconds=time.perf_counter() - started,
            input_answer=answer,
            output_answer=output,
        )
        return output

    rs._ensure_inline_citations = traced_citations

    original_generate = rs.generate_answer

    def traced_generate(
        llm: Any, query: str, context_docs: list[Any], chat_history: list[dict[str, Any]] | None = None
    ) -> str:
        started = time.perf_counter()
        output = original_generate(llm, query, context_docs, chat_history)
        recorder.emit(
            "answer_generation",
            query=query,
            duration_seconds=time.perf_counter() - started,
            context_documents=[doc_payload(doc, rank) for rank, doc in enumerate(context_docs, 1)],
            answer=output,
        )
        return output

    rs.generate_answer = traced_generate

    original_safety_factory = rs.create_safety_checker

    def result_payload(result: Any) -> dict[str, Any]:
        return {
            "allow": getattr(result, "allow", None),
            "risk_level": getattr(result, "risk_level", None),
            "reasons": getattr(result, "reasons", []),
            "action": getattr(result, "action", None),
            "scores": getattr(result, "scores", {}),
            "details": getattr(result, "details", {}),
        }

    class SafetyProxy:
        def __init__(self, checker: Any):
            self._checker = checker

        def __getattr__(self, name: str) -> Any:
            return getattr(self._checker, name)

        def _call(self, stage: str, method: Callable[..., Any], *args: Any) -> Any:
            started = time.perf_counter()
            result = method(*args)
            recorder.emit(
                f"safety_{stage}",
                duration_seconds=time.perf_counter() - started,
                result=result_payload(result),
            )
            return result

        def check_query_safety(self, query: str, history: list[dict[str, Any]]) -> Any:
            return self._call("pre_query", self._checker.check_query_safety, query, history)

        def check_context_safety(self, docs: list[Any]) -> Any:
            return self._call("context", self._checker.check_context_safety, docs)

        def check_answer_safety(self, query: str, docs: list[Any], answer: str) -> Any:
            return self._call(
                "post_generation", self._checker.check_answer_safety, query, docs, answer
            )

    def traced_safety_factory(config: Any) -> SafetyProxy:
        started = time.perf_counter()
        checker = original_safety_factory(config)
        recorder.emit(
            "safety_initialization",
            duration_seconds=time.perf_counter() - started,
            active=getattr(checker, "is_active", None),
        )
        return SafetyProxy(checker)

    rs.create_safety_checker = traced_safety_factory

    original_audit = rs.audit_log

    def traced_audit(*args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        original_audit(*args, **kwargs)
        query = kwargs.get("query", args[0] if args else None)
        recorder.emit(
            "audit",
            query=query,
            duration_seconds=time.perf_counter() - started,
            fields=kwargs,
        )

    rs.audit_log = traced_audit


def serve(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace_path).resolve()
    recorder = TraceRecorder(trace_path)
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        import api.rag_service as rs

        started = time.perf_counter()
        retrieval_service, reused = rs.initialize_retrieval_service(
            force_reindex=False, allow_reindex=False
        )
        retrieval_duration = time.perf_counter() - started
        pipeline = rs.RAGPipeline(rs.SETTINGS)
        rs._pipeline = pipeline
        pipeline_started = time.perf_counter()
        pipeline.initialize(force_reindex=False)
        pipeline_duration = time.perf_counter() - pipeline_started
        install_instrumentation(rs, pipeline, recorder)
        recorder.emit(
            "server_preflight",
            pid=os.getpid(),
            retrieval_service_reused=reused,
            retrieval_service_duration_seconds=retrieval_duration,
            pipeline_initialization_duration_seconds=pipeline_duration,
            runtime_config=rs.runtime_config(),
            settings={
                "ollama_base_url": rs.SETTINGS.ollama_base_url,
                "answer_model": rs.SETTINGS.roles.answer,
                "self_check_model": rs.SETTINGS.roles.self_check,
                "router_model": rs.SETTINGS.roles.router,
                "rewrite_model": rs.SETTINGS.roles.rewrite,
                "compressor_model": rs.SETTINGS.roles.compress,
                "force_retrieval": rs.SETTINGS.retrieval.force_retrieval,
                "rewrite_enabled": rs.SETTINGS.retrieval.query_rewrite_enabled,
                "compression_enabled": rs.SETTINGS.retrieval.enable_context_compression,
                "safety_enabled": rs.safety_enabled(),
                "safety_mode": rs.safety_mode(),
                "safety_backend": rs.SETTINGS.safety.backend,
                "groundedness_threshold": rs.SETTINGS.safety.min_groundedness,
                "groundedness_threshold_source": rs.SETTINGS.safety.min_groundedness_source,
                "groundedness_calibration_file": str(
                    rs.SETTINGS.safety.groundedness_calibration_file
                ),
                "collection_name": rs.SETTINGS.storage.collection_name,
                "chroma_path": str(rs.SETTINGS.storage.chroma_persist_directory),
            },
        )

        import main as api_main
        import uvicorn

        uvicorn.run(
            api_main.app,
            host="127.0.0.1",
            port=args.port,
            log_level="info",
            reload=False,
        )
        return 0
    except Exception as exc:
        recorder.emit(
            "server_error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
        return 2


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def latest_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matches = [event for event in events if event.get("event") == name]
    return matches[-1] if matches else None


def wait_for_health(url: str, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    import requests

    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Isolated backend exited with code {process.returncode}")
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise TimeoutError(f"Backend health timeout: {last_error}")


def stop_process(process: subprocess.Popen[Any], log: RunLogger) -> None:
    if process.poll() is not None:
        return
    log(f"Stopping isolated backend PID={process.pid}")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def classify_page(doc: dict[str, Any], phrase: str) -> bool:
    content = " ".join(str(doc.get("content", "")).split()).casefold()
    return " ".join((phrase or "").split()).casefold() in content


def find_rank(documents: list[dict[str, Any]], phrase: str) -> int | None:
    for position, document in enumerate(documents, 1):
        if classify_page(document, phrase):
            return int(document.get("rank") or position)
    return None


def find_score(
    rerank_event: dict[str, Any] | None,
    scores_event: dict[str, Any] | None,
    phrase: str,
) -> float | str:
    if not rerank_event or not scores_event:
        return NOT_AVAILABLE
    inputs = rerank_event.get("input_documents", [])
    scores = scores_event.get("scores", [])
    for index, document in enumerate(inputs):
        if classify_page(document, phrase) and index < len(scores):
            return float(scores[index])
    return NOT_AVAILABLE


def parse_citations(answer: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document_id, page in re.findall(r"\[([^\]:]+):([^\]]+)\]", answer or ""):
        key = (document_id, page)
        if key in seen:
            continue
        seen.add(key)
        matching = None
        for document in retrieved:
            metadata = document.get("metadata", {})
            source_stem = Path(str(metadata.get("source", "unknown"))).stem
            if source_stem == document_id and str(metadata.get("page")) == page:
                matching = document
                break
        text = str((matching or {}).get("content", ""))
        normalized_text = " ".join(text.split()).casefold()
        supports = all(
            " ".join(phrase.split()).casefold() in normalized_text
            for phrase in (
                "Lara Neumann",
                "TEST-KFZ-2026-1001",
                "Windshield glass damage",
                "partial comprehensive insurance",
                "150 euros",
            )
        )
        items.append(
            {
                "citation": f"[{document_id}:{page}]",
                "source": str((matching or {}).get("metadata", {}).get("source", NOT_AVAILABLE)),
                "page": page,
                "human_page": (
                    int(page) + 1 if str(page).isdigit() else NOT_AVAILABLE
                ),
                "chunk_id": (matching or {}).get("metadata", {}).get("chunk_id", NOT_AVAILABLE),
                "retrieved": matching is not None,
                "supports_all_material_claims": supports,
                "cited_text": text if matching is not None else NOT_AVAILABLE,
            }
        )
    return items


def validate_answer(answer: str) -> dict[str, Any]:
    lower = answer.casefold()
    coverage = any(token in lower for token in ("covered", "insured", "gedeckt", "versichert"))
    coverage_type = "partial comprehensive" in lower
    deductible = "150" in lower and any(
        token in lower for token in ("deductible", "selbstbeteiligung", "euro")
    )
    contract = "test-kfz-2026-1001" in lower
    customer = any(
        token in lower
        for token in (
            "lara neumann",
            "her insured vehicle",
            "lara neumann's insured vehicle",
        )
    )
    contradiction_checks = {
        "deductible_is_zero": bool(re.search(r"(?:0|zero)\s*(?:euro)?\s*deductible", lower)),
        "no_deductible_applies": "no deductible" in lower or "keine selbstbeteiligung" in lower,
        "liability_provides_coverage": bool(
            re.search(r"(?:covered|coverage).{0,70}personal liability", lower)
            or re.search(r"personal liability.{0,70}(?:covered|coverage)", lower)
        ),
        "wrong_contract_relevant": "test-phv-2026-2001" in lower,
        "windshield_excluded": bool(
            re.search(r"windshield.{0,50}(?:excluded|not covered|nicht gedeckt|nicht versichert)", lower)
        ),
        "information_missing": any(
            phrase in lower
            for phrase in (
                "sources do not contain enough",
                "information is unavailable",
                "insufficient information",
                "cannot determine",
                "do not state whether",
            )
        ),
        "no_motor_coverage": "no motor-insurance coverage" in lower
        or "no motor insurance coverage" in lower,
    }
    return {
        "coverage_correct": coverage,
        "coverage_type_correct": coverage_type,
        "deductible_correct": deductible,
        "contract_correct": contract,
        "customer_correct": customer,
        "fact_results": {
            "coverage": "PASS" if coverage else "FAIL",
            "coverage_type": "PASS" if coverage_type else "FAIL",
            "deductible": "PASS" if deductible else "FAIL",
            "contract": "PASS" if contract else "FAIL",
            "customer": "PASS" if customer else "FAIL",
        },
        "contradiction_checks": {
            key: "FAIL" if present else "PASS"
            for key, present in contradiction_checks.items()
        },
        "contradictions": [key for key, present in contradiction_checks.items() if present],
    }


def status_word(value: Any) -> str:
    return "PASS" if value else "FAIL"


def guardrail_details(event: dict[str, Any] | None, stage: str) -> dict[str, Any]:
    result_details = ((event or {}).get("result", {}).get("details", {}) or {})
    runtime_details = (
        result_details.get("nemo_runtime", {}).get("details", {}) or {}
    )
    output_data = runtime_details.get("output_data", {}) or {}
    return output_data.get(f"guardrails_{stage}_details", {}) or {}


def legacy_markdown_report(result: dict[str, Any]) -> str:
    env = result["environment"]
    pdf = result["pdf"]
    ingestion = result["ingestion"]
    query = result["query"]
    retrieval = result["retrieval"]
    reranking = result["reranking"]
    self_check = result["self_check"]
    answer = result["answer"]
    citations = result["citations"]
    safety = result["safety"]
    audit = result["audit"]
    cleanup = result["cleanup"]
    counts = result["collection_counts"]
    timings = result["timings"]
    results = result["results"]
    contradictions = "\n".join(
        f"- {name}: {status}" for name, status in answer.get("contradiction_checks", {}).items()
    ) or "- not_available"
    phrases = "\n".join(
        f"- `{phrase}`: {status_word(found)}"
        for phrase, found in pdf.get("required_phrases_found", {}).items()
    )
    chunks = "\n".join(
        f"- `{item.get('id')}` page={item.get('metadata', {}).get('page')} "
        f"chunk_id={item.get('metadata', {}).get('chunk_id')}: {item.get('text_preview', '')}"
        for item in ingestion.get("chunk_metadata", [])
    ) or "- none"
    sources = "\n".join(
        f"- rank {item.get('rank')}: source={item.get('metadata', {}).get('source')} "
        f"page={item.get('metadata', {}).get('page')} chunk={item.get('metadata', {}).get('chunk_id')}"
        for item in retrieval.get("documents", [])
    ) or "- none"
    citation_lines = "\n".join(
        f"- {item.get('citation')}: source={item.get('source')} page={item.get('page')} "
        f"retrieved={item.get('retrieved')} supports={item.get('supports_coverage_and_deductible')}"
        for item in citations.get("items", [])
    ) or "- none"
    return f"""# Synthetic Customer Insurance RAG Scenario

## 1. Test objective
Execute a real end-to-end synthetic customer scenario through the current insurance RAG pipeline.

## 2. Test scope
PDF creation, production PDF loading, chunking, embedding, isolated Chroma indexing, hybrid retrieval, reranking, self-check, generation, citations, safety, audit, and cleanup.

## 3. Project and environment
- Project: `{PROJECT_ROOT}`
- Test run: `{result['test_run_id']}`
- Timestamp: `{result['timestamp']}`
- Backend URL: `{env.get('backend_url')}`
- Backend PID: `{env.get('backend_pid')}`
- Ollama URL: `{env.get('ollama_url')}`

## 4. Current model configuration
- Answer: `{env.get('answer_model')}`
- Self-check: `{env.get('self_check_model')}`
- Router: `{env.get('router_model')}`
- Rewrite: `{env.get('rewrite_model')}`
- Compression: `{env.get('compression_model')}`
- Embedding: `{env.get('embedding_model')}`
- Reranker: `{env.get('reranker_model')}`

## 5. Synthetic-data confirmation
`{status_word(pdf.get('synthetic_data_confirmed'))}`. All identifiers are fixed fictional test values.

## 6. PDF creation
- Path: `{pdf.get('path')}`
- Created: `{pdf.get('created')}`
- Duration: `{timings.get('pdf_creation_seconds')}` seconds

## 7. PDF validation
- Valid signature: `{pdf.get('valid_signature')}`
- Opened: `{pdf.get('opened')}`
- Pages: `{pdf.get('pages')}`
- Render successful: `{pdf.get('render', {}).get('successful')}`

## 8. PDF text extraction
- Loader: `{pdf.get('loader')}`
- Extracted pages: `{pdf.get('extracted_page_count')}`
- Duration: `{timings.get('pdf_extraction_seconds')}` seconds
{phrases}

## 9. Ingestion implementation used
`api.rag_service.load_pdf_source`, `load_and_split_documents`, `initialize_embeddings`, and `build_vectorstore` were reused. The adapter only adds unique synthetic metadata before production embedding and insertion.

## 10. Test-data isolation method
Temporary Chroma directory and collection: `{ingestion.get('isolation_method')}`. Production storage paths were not overridden outside the test process.

## 11. Chunk creation and metadata
{chunks}

## 12. Collection counts before ingestion
```json
{json.dumps(counts.get('before'), indent=2)}
```

## 13. Collection counts after ingestion
```json
{json.dumps(counts.get('after_ingestion'), indent=2)}
```

## 14. Preflight checks
```json
{json.dumps(result.get('preflight'), ensure_ascii=False, indent=2)}
```

## 15. Warm-up result
```json
{json.dumps(result.get('warmup'), ensure_ascii=False, indent=2)}
```

## 16. Main query
`{query.get('text')}`

## 17. Full pipeline execution path
`{query.get('execution_path')}`

## 18. Retrieved sources
{sources}

## 19. Retrieved pages and chunks
Relevant page rank: `{reranking.get('relevant_chunk_rank')}`; distractor rank: `{reranking.get('distractor_chunk_rank')}`.

## 20. Retrieval result
`{retrieval.get('status')}`; correct source: `{retrieval.get('correct_source_retrieved')}`; correct page: `{retrieval.get('correct_page_retrieved')}`.

## 21. Reranking result
`{reranking.get('status')}`; relevant score: `{reranking.get('relevant_score')}`; distractor score: `{reranking.get('distractor_score')}`.

## 22. Self-check raw output
```text
{self_check.get('raw_output')}
```

## 23. Self-check parsed decision
`{self_check.get('parsed_decision')}` (`{self_check.get('status')}`).

## 24. Query rewrite status
Applied: `{query.get('query_rewritten')}`; retries: `{query.get('retry_count')}`.

## 25. Final answer
```text
{answer.get('text')}
```

## 26. Expected-fact comparison
```json
{json.dumps(answer.get('fact_results'), indent=2)}
```

## 27. Contradiction checks
{contradictions}

## 28. Groundedness validation
Status: `{answer.get('groundedness_status')}`. Supported claims: `{answer.get('supported_claims')}`. Unsupported claims: `{answer.get('unsupported_claims')}`.

## 29. Citation validation
Status: `{citations.get('status')}`; count: `{citations.get('count')}`.
{citation_lines}

## 30. Distractor-handling validation
Handled correctly: `{answer.get('distractor_handled_correctly')}`.

## 31. Safety results
```json
{json.dumps(safety, ensure_ascii=False, indent=2)}
```

## 32. Audit result
```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## 33. Available stage timings
```json
{json.dumps(timings, indent=2)}
```

## 34. Timeout and error information
- Timeout: `{query.get('timeout')}`
- HTTP status: `{query.get('http_status')}`
- Error: `{query.get('error')}`

## 35. Cleanup actions
Executed: `{cleanup.get('executed')}`; temporary collection deleted: `{cleanup.get('temporary_collection_deleted')}`; temporary directory deleted: `{cleanup.get('temporary_directory_deleted')}`.

## 36. Collection counts after cleanup
```json
{json.dumps(counts.get('after_cleanup'), indent=2)}
```

## 37. Collection-integrity result
`{result.get('collection_integrity', {}).get('status')}`; differences: `{result.get('collection_integrity', {}).get('differences')}`.

## 38. Functional PASS or FAIL
`{results.get('functional')}`

## 39. Quality PASS or FAIL
`{results.get('quality')}`

## 40. Performance PASS or FAIL
`{results.get('performance')}`

## 41. Overall PASS or FAIL
`{results.get('overall')}`

## 42. Exact failure reason
Stage: `{result.get('failure_stage')}`; reason: `{result.get('failure_reason')}`.

## 43. Remaining limitations
{result.get('remaining_limitations')}

## 44. Recommended next action
{result.get('recommended_next_action')}
"""


def markdown_report(result: dict[str, Any]) -> str:
    calibration = result.get("groundedness_calibration", {})
    selected_metrics = calibration.get("selected_metrics", {})
    candidate_rows = "\n".join(
        "| {threshold:.2f} | {tp} | {tn} | {fp} | {fn} | {precision:.3f} | "
        "{recall:.3f} | {f1:.3f} | {far:.3f} |".format(
            threshold=float(row.get("threshold", 0.0)),
            tp=row.get("tp"),
            tn=row.get("tn"),
            fp=row.get("fp"),
            fn=row.get("fn"),
            precision=float(row.get("precision", 0.0)),
            recall=float(row.get("recall", 0.0)),
            f1=float(row.get("f1", 0.0)),
            far=float(row.get("false_acceptance_rate", 0.0)),
        )
        for row in calibration.get("candidate_metrics", [])
    ) or "| not_available | | | | | | | | |"
    case_rows = "\n".join(
        f"| {case.get('id')} | {case.get('label')} | {case.get('score')} | "
        f"{case.get('prediction')} |"
        for case in calibration.get("cases", [])
    ) or "| not_available | | | |"
    chunk_rows = "\n".join(
        f"- `{chunk.get('id')}`: page={chunk.get('metadata', {}).get('page_human')}, "
        f"chunk={chunk.get('metadata', {}).get('chunk_id')}, "
        f"insurance_type={chunk.get('metadata', {}).get('insurance_type')}"
        for chunk in result.get("ingestion", {}).get("chunk_metadata", [])
    ) or "- none"
    retrieval_rows = "\n".join(
        f"- Rank {doc.get('rank')}: page={doc.get('metadata', {}).get('page_human')}, "
        f"chunk={doc.get('metadata', {}).get('chunk_id')}, "
        f"insurance_type={doc.get('metadata', {}).get('insurance_type')}"
        for doc in result.get("retrieval", {}).get("documents", [])
    ) or "- none"
    citation_rows = "\n".join(
        f"- `{item.get('citation')}`: page={item.get('human_page')}, "
        f"retrieved={item.get('retrieved')}, "
        f"supports_all_material_claims={item.get('supports_all_material_claims')}"
        for item in result.get("citations", {}).get("items", [])
    ) or "- none"
    command_rows = "\n".join(
        f"| `{item.get('command')}` | {item.get('exit_code')} | {item.get('status')} | "
        f"{float(item.get('duration_seconds', 0.0)):.3f}s |"
        for item in result.get("test_commands", [])
    ) or "| not_available | | | |"
    result_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {status} |"
        for name, status in result.get("results", {}).items()
    )
    safety = result.get("safety", {})
    safety_summary = {
        "status": safety.get("status"),
        "contract_allowed_in_context": safety.get("contract_allowed_in_context"),
        "contract_allowed_in_output": safety.get("contract_allowed_in_output"),
        "protected_context_pii_redacted": safety.get("protected_context_pii_redacted"),
        "coverage_dates_preserved": safety.get("coverage_dates_preserved"),
        "output_has_no_disallowed_pii": safety.get("output_has_no_disallowed_pii"),
        "context_pii": safety.get("context_pii"),
        "output_pii": safety.get("output_pii"),
        "focused_phone_date_validation": safety.get("focused_phone_date_validation"),
    }
    changed_files = [
        "src/core/safety_audit.py",
        "src/guardrails/integrations/nemo_actions.py",
        "src/config/models.py",
        "src/api/rag_service.py",
        "src/utils/language.py",
        "scripts/calibrate_groundedness.py",
        "scripts/test_synthetic_customer_scenario.py",
        "tests/fixtures/groundedness_calibration_cases.json",
        PDF_NAME,
        "tests/unit/test_safety_pii_rules.py",
        "tests/unit/test_response_language.py",
        "tests/unit/test_groundedness_calibration.py",
        "tests/unit/test_thesis_eval_metadata.py",
        "config/groundedness_calibration.json",
    ]
    return f"""# English Synthetic Customer Insurance RAG Scenario

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
{chr(10).join(f'- `{path}`' for path in changed_files)}

## 5. Exact code changes
- Contract IDs are classified as allowed while other PII remains sensitive.
- Valid calendar-date spans are excluded from phone detection and phone matching cannot cross lines.
- Groundedness now scores claim-level lexical support, exact hard facts, coverage polarity, and query/rank evidence; a reproducible calibration artifact supplies the threshold.
- Audit overwrites any caller language value with deterministic detection from `generated_answer`.
- The baseline harness now creates and validates a three-page English PDF and records all mandatory result categories.

## 6. Safety policy change for contract numbers
```json
{json.dumps({'context': safety.get('context_pii'), 'output': safety.get('output_pii'), 'contract_allowed_in_context': safety.get('contract_allowed_in_context'), 'contract_allowed_in_output': safety.get('contract_allowed_in_output')}, ensure_ascii=False, indent=2)}
```

## 7. Phone/date false-positive fix
```json
{json.dumps(result.get('focused_safety_validation'), ensure_ascii=False, indent=2)}
```
Context coverage dates preserved: `{safety.get('coverage_dates_preserved')}`.

## 8. Groundedness calibration methodology
Algorithm `{calibration.get('algorithm_version')}` scored `{calibration.get('dataset_size')}` balanced labeled cases. Rule: {calibration.get('selection_rule')}

## 9. Calibration dataset
| Case | Label | Score | Prediction |
|---|---|---:|---|
{case_rows}

## 10. Threshold candidates and metrics
| Threshold | TP | TN | FP | FN | Precision | Recall | F1 | False accept rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{candidate_rows}

## 11. Selected threshold and rationale
Selected `{calibration.get('selected_threshold')}`; loaded `{calibration.get('loaded_threshold')}` from `{calibration.get('loaded_source')}`. Selected metrics: `{json.dumps(selected_metrics)}`. This operating point has zero false acceptance on the labeled set while retaining the highest acceptable true-positive performance under the documented rule.

## 12. Language-detection fix
```json
{json.dumps(result.get('telemetry'), ensure_ascii=False, indent=2)}
```
Empty, marker-free, and ambiguous short responses use the deterministic fallback `Unknown`.

## 13. Synthetic PDF validation
```json
{json.dumps(result.get('pdf'), ensure_ascii=False, indent=2)}
```

## 14. Ingestion and chunk metadata
Chunks created: `{result.get('ingestion', {}).get('chunks_created')}`; isolated collection: `{result.get('ingestion', {}).get('collection_name')}`.
{chunk_rows}

## 15. Retrieval ranking
Status: `{result.get('retrieval', {}).get('status')}`.
{retrieval_rows}

## 16. Reranker scores
Relevant rank/score: `{result.get('reranking', {}).get('relevant_chunk_rank')}` / `{result.get('reranking', {}).get('relevant_score')}`. Distractor rank/score: `{result.get('reranking', {}).get('distractor_chunk_rank')}` / `{result.get('reranking', {}).get('distractor_score')}`.

## 17. Safety results
```json
{json.dumps(safety_summary, ensure_ascii=False, indent=2)}
```

## 18. Self-check raw output
```text
{result.get('self_check', {}).get('raw_output')}
```

## 19. Self-check parsed output
`{result.get('self_check', {}).get('parsed_decision')}`; status `{result.get('self_check', {}).get('status')}`.

## 20. Final answer
```text
{result.get('answer', {}).get('text')}
```

## 21. Expected-fact comparison
```json
{json.dumps(result.get('answer', {}).get('fact_results'), indent=2)}
```
Unsupported/contradictory claims: `{result.get('answer', {}).get('unsupported_claims')}`.

## 22. Groundedness validation
Score `{result.get('answer', {}).get('groundedness_score')}` versus calibrated threshold `{result.get('answer', {}).get('groundedness_threshold')}` using `{result.get('answer', {}).get('groundedness_algorithm')}`: `{result.get('answer', {}).get('groundedness_status')}`.

## 23. Citation validation
Status `{result.get('citations', {}).get('status')}`; count `{result.get('citations', {}).get('count')}`.
{citation_rows}

## 24. Distractor validation
Relevant motor evidence ranked above liability distractor: `{result.get('answer', {}).get('distractor_handled_correctly')}`. Contradictions: `{result.get('answer', {}).get('contradictions')}`.

## 25. Audit and telemetry validation
Main audit found: `{result.get('audit', {}).get('entry_found')}`; response language: `{result.get('audit', {}).get('response_language')}`; contract visible: `{result.get('audit', {}).get('contract_visible')}`; status: `{result.get('audit', {}).get('status')}`. Full audit and trace are retained in the JSON result.

## 26. Production collection count comparison
```json
{json.dumps(result.get('collection_counts'), indent=2)}
```
Integrity: `{result.get('collection_integrity', {}).get('status')}`; differences: `{result.get('collection_integrity', {}).get('differences')}`.

## 27. Cleanup validation
```json
{json.dumps(result.get('cleanup'), indent=2)}
```

## 28. Latency breakdown
```json
{json.dumps(result.get('timings'), indent=2)}
```
Execution reliability and user-facing latency are reported separately: `{json.dumps(result.get('performance'))}`.

## 29. Test commands and results
| Command | Exit code | Status | Duration |
|---|---:|---|---:|
{command_rows}

The E2E command itself is `python scripts/test_synthetic_customer_scenario.py`; its result is represented by the table below and the process exit code used by the caller.

## 30. Final PASS/FAIL table
| Category | Result |
|---|---|
{result_rows}

Failure stage: `{result.get('failure_stage')}`. Failure reason: `{result.get('failure_reason')}`.

## 31. Remaining risks and limitations
{result.get('remaining_limitations')}

Artifacts: `{json.dumps(result.get('artifacts'), ensure_ascii=False)}`.
"""


def concise_summary(result: dict[str, Any]) -> str:
    pdf = result["pdf"]
    ingestion = result["ingestion"]
    query = result["query"]
    retrieval = result["retrieval"]
    reranking = result["reranking"]
    self_check = result["self_check"]
    answer = result["answer"]
    citations = result["citations"]
    safety = result["safety"]
    audit = result["audit"]
    cleanup = result["cleanup"]
    results = result["results"]
    timings = result["timings"]
    return f"""PDF created: {pdf.get('created')}
PDF valid: {pdf.get('valid')}
PDF pages: {pdf.get('pages')}
PDF parsed: {pdf.get('parsed')}
Synthetic data confirmed: {pdf.get('synthetic_data_confirmed')}
Required text extracted: {pdf.get('required_text_extracted')}
Document ingested: {ingestion.get('successful')}
Isolation method: {ingestion.get('isolation_method')}
Chunks indexed: {ingestion.get('chunks_created')}
Full pipeline executed: {query.get('full_pipeline_executed')}
Correct source retrieved: {retrieval.get('correct_source_retrieved')}
Correct page retrieved: {retrieval.get('correct_page_retrieved')}
Relevant chunk rank: {reranking.get('relevant_chunk_rank')}
Distractor chunk rank: {reranking.get('distractor_chunk_rank')}
Self-check raw result: {self_check.get('raw_output')}
Self-check parsed result: {self_check.get('parsed_decision')}
Query rewritten: {query.get('query_rewritten')}
Answer states coverage: {answer.get('coverage_correct')}
Answer states partial comprehensive insurance: {answer.get('coverage_type_correct')}
Deductible is 150 euros per claim: {answer.get('deductible_correct')}
Correct contract referenced: {answer.get('contract_correct')}
Distractor handled correctly: {answer.get('distractor_handled_correctly')}
Groundedness valid: {answer.get('groundedness_status') == 'PASS'}
Groundedness score / threshold: {answer.get('groundedness_score')} / {answer.get('groundedness_threshold')}
Citation count: {citations.get('count')}
Citation valid: {citations.get('valid')}
Pre-safety successful: {safety.get('pre_query_successful')}
Context safety successful: {safety.get('context_successful')}
Post-safety successful: {safety.get('post_generation_successful')}
Audit successful: {audit.get('status') == 'PASS'}
Response language: {audit.get('response_language')}
Timeout: {query.get('timeout')}
Cleanup executed: {cleanup.get('executed')}
Cleanup successful: {cleanup.get('successful')}
Synthetic chunks remaining: {cleanup.get('synthetic_chunks_remaining')}
Collection counts restored: {result.get('collection_integrity', {}).get('status') == 'PASS'}

Functional result: {results.get('functional')}
Quality result: {results.get('quality')}
Safety result: {results.get('safety')}
Isolation result: {results.get('isolation')}
Cleanup result: {results.get('cleanup')}
Telemetry result: {results.get('telemetry')}
Groundedness calibration result: {results.get('groundedness_calibration')}
Performance result: {results.get('performance')}
Overall result: {results.get('overall')}

PDF creation duration: {timings.get('pdf_creation_seconds')}
Ingestion duration: {timings.get('ingestion_seconds')}
Warm-up duration: {timings.get('warmup_seconds')}
Main scenario duration: {timings.get('main_scenario_seconds')}
Retrieval duration: {timings.get('retrieval_seconds')}
Reranking duration: {timings.get('reranking_seconds')}
Self-check duration: {timings.get('self_check_seconds')}
Answer-generation duration: {timings.get('answer_generation_seconds')}
Cleanup duration: {timings.get('cleanup_seconds')}

Main bottleneck: {result.get('main_bottleneck')}
Exact failure stage: {result.get('failure_stage')}
Exact failure reason: {result.get('failure_reason')}
Recommended next action: {result.get('recommended_next_action')}"""


def default_result(timestamp: str, pdf_path: Path) -> dict[str, Any]:
    return {
        "test_run_id": TEST_RUN_ID,
        "document_id": DOCUMENT_ID,
        "customer_id": CUSTOMER_ID,
        "timestamp": timestamp,
        "environment": {},
        "pdf": {"path": str(pdf_path), "created": False, "valid": False, "parsed": False},
        "ingestion": {
            "method": "production functions via test-only adapter",
            "temporary_collection": True,
            "chunks_created": 0,
            "chunk_metadata": [],
            "successful": False,
        },
        "preflight": {},
        "warmup": {},
        "query": {
            "text": QUESTION,
            "execution_path": "POST /api/ask",
            "status": "NOT_EXECUTED",
            "timeout": False,
            "full_pipeline_executed": False,
        },
        "retrieval": {
            "attempts": 0,
            "documents": [],
            "correct_source_retrieved": False,
            "correct_page_retrieved": False,
            "status": "FAIL",
        },
        "reranking": {"status": "FAIL"},
        "self_check": {
            "raw_output": NOT_AVAILABLE,
            "parsed_decision": NOT_AVAILABLE,
            "duration_seconds": NOT_AVAILABLE,
            "status": "FAIL",
        },
        "answer": {"text": "", "contradictions": []},
        "citations": {"count": 0, "items": [], "valid": False, "status": "FAIL"},
        "safety": {},
        "audit": {"enabled": True, "entry_found": False, "status": "FAIL"},
        "telemetry": {"status": "FAIL"},
        "groundedness_calibration": {"status": "FAIL"},
        "cleanup": {"executed": False, "successful": False},
        "collection_counts": {"before": {}, "after_ingestion": {}, "after_execution": {}, "after_cleanup": {}},
        "collection_integrity": {"status": "FAIL"},
        "timings": {},
        "test_commands": [],
        "results": {
            "functional": "FAIL",
            "quality": "FAIL",
            "safety": "FAIL",
            "isolation": "FAIL",
            "cleanup": "FAIL",
            "telemetry": "FAIL",
            "groundedness_calibration": "FAIL",
            "performance": "FAIL",
            "overall": "FAIL",
        },
        "failure_stage": NOT_AVAILABLE,
        "failure_reason": None,
        "remaining_limitations": "None identified yet.",
        "recommended_next_action": "Review the exact failing stage in this report.",
        "main_bottleneck": NOT_AVAILABLE,
    }


def execute(args: argparse.Namespace) -> int:
    from dotenv import dotenv_values

    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = utc_now()
    reports_dir = Path(args.output_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = PROJECT_ROOT / "tests" / "fixtures" / PDF_NAME
    json_path = reports_dir / f"synthetic_customer_scenario_{timestamp_slug}.json"
    md_path = reports_dir / f"synthetic_customer_scenario_{timestamp_slug}.md"
    log_path = reports_dir / f"synthetic_customer_scenario_{timestamp_slug}.log"
    log = RunLogger(log_path)
    result = default_result(timestamp, pdf_path)
    result["artifacts"] = {
        "pdf": str(pdf_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "log": str(log_path),
    }
    calibration_path = PROJECT_ROOT / "config" / "groundedness_calibration.json"
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_markdown = sorted(reports_dir.glob("groundedness_calibration_*.md"))
    calibration_json = sorted(reports_dir.glob("groundedness_calibration_*.json"))
    result["artifacts"].update(
        {
            "groundedness_calibration_config": str(calibration_path),
            "groundedness_calibration_markdown": (
                str(calibration_markdown[-1]) if calibration_markdown else NOT_AVAILABLE
            ),
            "groundedness_calibration_json": (
                str(calibration_json[-1]) if calibration_json else NOT_AVAILABLE
            ),
        }
    )
    result["groundedness_calibration"] = {
        **calibration_payload,
        "status": "PENDING",
    }
    if calibration_json:
        calibration_report_payload = json.loads(
            calibration_json[-1].read_text(encoding="utf-8")
        )
        candidate_thresholds = {0.4, 0.5, 0.51, 0.6, 0.7}
        result["groundedness_calibration"]["candidate_metrics"] = [
            row
            for row in calibration_report_payload.get("threshold_sweep", [])
            if row.get("threshold") in candidate_thresholds
        ]
    temp_root = PROJECT_ROOT / "tmp" / f"{TEST_RUN_ID}_{timestamp_slug}"
    temp_pdf_dir = temp_root / "pdfs"
    temp_chroma = temp_root / "chroma"
    temp_audit = temp_root / "audit.log"
    temp_model_config = temp_root / "model_config.json"
    temp_pdf_hashes = temp_root / "pdf_hashes.json"
    trace_path = temp_root / "trace.jsonl"
    render_dir = temp_root / "rendered"
    collection_name = "synthetic_customer_scenario_en_002_collection"
    server_process: subprocess.Popen[Any] | None = None
    vector_store: Any = None
    temp_client: Any = None
    dotenv = dotenv_values(PROJECT_ROOT / ".env")
    production_chroma = Path(
        str(dotenv.get("CHROMA_PERSIST_DIRECTORY") or "data/processed/vectorstores/chroma_db")
    )
    if not production_chroma.is_absolute():
        production_chroma = (PROJECT_ROOT / production_chroma).resolve()

    try:
        log(f"Starting {TEST_RUN_ID}")
        baseline = production_counts(production_chroma)
        result["collection_counts"]["before"] = baseline
        result["environment"]["production_chroma_path"] = str(production_chroma)
        result["environment"]["processes_before"] = process_inventory()
        log(f"Authoritative production baseline: {baseline}")

        modified_python_files = [
            "src/api/rag_service.py",
            "src/config/models.py",
            "src/core/safety_audit.py",
            "src/guardrails/integrations/nemo_actions.py",
            "src/utils/language.py",
            "scripts/calibrate_groundedness.py",
            "scripts/test_synthetic_customer_scenario.py",
            "tests/unit/test_safety_pii_rules.py",
            "tests/unit/test_response_language.py",
            "tests/unit/test_groundedness_calibration.py",
            "tests/unit/test_thesis_eval_metadata.py",
        ]
        result["test_commands"] = [
            run_recorded_command(
                [sys.executable, "-m", "py_compile", *modified_python_files]
            ),
            run_recorded_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/unit/test_safety_pii_rules.py",
                    "tests/unit/test_response_language.py",
                    "tests/unit/test_groundedness_calibration.py",
                ],
                timeout=300,
            ),
            run_recorded_command(
                [sys.executable, "scripts/calibrate_groundedness.py", "--check"]
            ),
            run_recorded_command(
                [sys.executable, "-m", "pytest", "-q", "tests/unit"],
                timeout=360,
            ),
        ]
        if any(check["exit_code"] != 0 for check in result["test_commands"]):
            result["failure_stage"] = "Code-quality checks"
            raise RuntimeError("One or more recorded pre-E2E checks failed")
        log("Recorded syntax, focused unit, and calibration checks PASS")

        pdf_duration = create_pdf(pdf_path)
        result["timings"]["pdf_creation_seconds"] = pdf_duration
        result["pdf"]["created"] = pdf_path.exists()
        result["pdf"]["size_bytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0
        log(f"PDF created in {pdf_duration:.3f}s: {pdf_path}")

        signature = pdf_path.read_bytes()[:5] == b"%PDF-"
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = len(reader.pages)
        metadata = {str(key): str(value) for key, value in (reader.metadata or {}).items()}
        result["pdf"].update(
            {
                "valid_signature": signature,
                "opened": True,
                "pages": pages,
                "metadata": metadata,
                "synthetic_data_confirmed": "SYNTHETIC TEST DATA – NOT A REAL PERSON" in str(metadata),
            }
        )
        render = render_pdf(pdf_path, render_dir)
        result["pdf"]["render"] = render
        if not signature or pages != 3 or not render["successful"]:
            raise RuntimeError("PDF validation failed before ingestion")

        temp_pdf_dir.mkdir(parents=True, exist_ok=True)
        ingestion_pdf = temp_pdf_dir / PDF_NAME
        shutil.copy2(pdf_path, ingestion_pdf)
        temp_chroma.mkdir(parents=True, exist_ok=True)

        os.environ.update(
            {
                "PDF_DIRECTORY": str(temp_pdf_dir),
                "CHROMA_PERSIST_DIRECTORY": str(temp_chroma),
                "COLLECTION_NAME": collection_name,
                "AUDIT_LOG_FILE": str(temp_audit),
                "MODEL_CONFIG_FILE": str(temp_model_config),
                "PDF_HASH_FILE": str(temp_pdf_hashes),
                "INIT_PIPELINE_ON_STARTUP": "false",
            }
        )
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import api.rag_service as rs

        result["environment"].update(
            {
                "ollama_url": rs.SETTINGS.ollama_base_url,
                "answer_model": rs.SETTINGS.roles.answer,
                "self_check_model": rs.SETTINGS.roles.self_check,
                "router_model": rs.SETTINGS.roles.router,
                "rewrite_model": rs.SETTINGS.roles.rewrite,
                "compression_model": rs.SETTINGS.roles.compress,
                "embedding_model": rs.SETTINGS.embedding.model,
                "reranker_model": rs.SETTINGS.reranker.model,
                "force_retrieval": rs.SETTINGS.retrieval.force_retrieval,
                "rewrite_enabled": rs.SETTINGS.retrieval.query_rewrite_enabled,
                "compression_enabled": rs.SETTINGS.retrieval.enable_context_compression,
                "safety_enabled": rs.safety_enabled(),
                "safety_mode": rs.safety_mode(),
                "safety_backend": rs.SETTINGS.safety.backend,
                "groundedness_threshold": rs.SETTINGS.safety.min_groundedness,
                "groundedness_threshold_source": rs.SETTINGS.safety.min_groundedness_source,
                "groundedness_calibration_file": str(
                    rs.SETTINGS.safety.groundedness_calibration_file
                ),
                "temporary_chroma_path": str(temp_chroma),
                "temporary_collection": collection_name,
            }
        )
        selected_metrics = result["groundedness_calibration"].get("selected_metrics", {})
        calibration_pass = all(
            (
                result["groundedness_calibration"].get("dataset_size", 0) >= 20,
                selected_metrics.get("fp") == 0,
                float(selected_metrics.get("recall", 0.0)) >= 0.8,
                rs.SETTINGS.safety.min_groundedness
                == result["groundedness_calibration"].get("selected_threshold"),
                rs.SETTINGS.safety.min_groundedness_source == "calibration_file",
            )
        )
        result["groundedness_calibration"]["loaded_threshold"] = (
            rs.SETTINGS.safety.min_groundedness
        )
        result["groundedness_calibration"]["loaded_source"] = (
            rs.SETTINGS.safety.min_groundedness_source
        )
        result["groundedness_calibration"]["status"] = (
            "PASS" if calibration_pass else "FAIL"
        )

        from core.safety_audit import detect_pii, sanitize_pii

        date_probe = (
            "Coverage dates: 01.01.2026, 31.12.2026, 14.05.1988, 2026-01-01, "
            "2026-12-31, 01/01/2026, 12/31/2026.\n"
            "1. Motor insurance\n2. Personal liability insurance"
        )
        phone_probe = (
            "Phone: +49 151 23456789\n"
            "1. Motor insurance\n"
            "2. Personal liability insurance"
        )
        date_items = detect_pii(date_probe, rs.SETTINGS.safety)
        phone_items = detect_pii(phone_probe, rs.SETTINGS.safety)
        phone_sanitized = sanitize_pii(phone_probe, phone_items, rs.SETTINGS.safety)
        result["focused_safety_validation"] = {
            "date_probe": date_probe,
            "date_pii_types": [item.pii_type for item in date_items],
            "dates_not_phone": not any(item.pii_type == "phone" for item in date_items),
            "phone_detected": any(item.pii_type == "phone" for item in phone_items),
            "phone_sanitized": phone_sanitized,
            "list_numbering_preserved": (
                "1. Motor insurance\n2. Personal liability insurance" in phone_sanitized
            ),
        }

        german_query = "Is the windshield damage covered?"
        german_answer = "Der Glasschaden ist durch die Versicherung gedeckt."
        rs.audit_log(
            query=german_query,
            retrieved_documents=[],
            compressed_context=[],
            generated_answer=german_answer,
            response_language="English",
            telemetry_probe="focused_german_answer",
        )
        german_rows = [
            json.loads(line)
            for line in temp_audit.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        german_row = next(
            (row for row in reversed(german_rows) if row.get("telemetry_probe") == "focused_german_answer"),
            None,
        )
        result["telemetry"]["focused_german"] = {
            "query_language": "English",
            "answer_language_expected": "German",
            "recorded_response_language": (
                german_row.get("response_language") if german_row else NOT_AVAILABLE
            ),
            "status": (
                "PASS"
                if german_row and german_row.get("response_language") == "German"
                else "FAIL"
            ),
        }

        warnings_buffer: list[str] = []
        extraction_started = time.perf_counter()
        import io

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            loaded_docs = rs.load_pdf_source(str(pdf_path))
        extraction_duration = time.perf_counter() - extraction_started
        warnings_buffer.extend(
            item for item in (stdout_buffer.getvalue().strip(), stderr_buffer.getvalue().strip()) if item
        )
        extracted_text = "\n".join(doc.page_content for doc in loaded_docs)
        normalized_extracted_text = " ".join(extracted_text.split())
        phrase_results = {
            phrase: " ".join(phrase.split()) in normalized_extracted_text
            for phrase in REQUIRED_PHRASES
        }
        page_metadata_preserved = all("page" in doc.metadata for doc in loaded_docs[:3])
        result["timings"]["pdf_extraction_seconds"] = extraction_duration
        result["pdf"].update(
            {
                "loader": "api.rag_service.load_pdf_source -> PyPDFLoader",
                "parsed": bool(loaded_docs),
                "extracted_page_count": len(loaded_docs),
                "required_phrases_found": phrase_results,
                "required_text_extracted": all(phrase_results.values()),
                "page_metadata_preserved": page_metadata_preserved,
                "loader_warnings": warnings_buffer,
                "valid": signature and pages == 3 and all(phrase_results.values()),
            }
        )
        if not result["pdf"]["valid"] or len(loaded_docs) < 3 or not page_metadata_preserved:
            result["failure_stage"] = "PDF parsing"
            raise RuntimeError("Project PDF loader did not extract the required content and metadata")
        log(f"PDF parsing PASS in {extraction_duration:.3f}s with {len(loaded_docs)} documents")

        ingestion_started = time.perf_counter()
        splits = rs.load_and_split_documents([str(ingestion_pdf)])
        for index, document in enumerate(splits, 1):
            raw_page = int(document.metadata.get("page", -1))
            insurance_type = (
                "motor_insurance" if raw_page == 1 else "personal_liability" if raw_page == 2 else "customer_profile"
            )
            contract_number = (
                "TEST-KFZ-2026-1001" if raw_page == 1 else "TEST-PHV-2026-2001" if raw_page == 2 else "multiple"
            )
            document.metadata.update(
                {
                    "document_id": DOCUMENT_ID,
                    "test_run_id": TEST_RUN_ID,
                    "customer_id": CUSTOMER_ID,
                    "document_type": "synthetic_customer_test",
                    "source": str(ingestion_pdf),
                    "source_filename": PDF_NAME,
                    "contract_number": contract_number,
                    "page_human": raw_page + 1,
                    "chunk_id": f"{DOCUMENT_ID}-P{raw_page + 1}-C{index:03d}",
                    "insurance_type": insurance_type,
                    "synthetic": True,
                }
            )
        embeddings = rs.initialize_embeddings()
        vector_store = rs.build_vectorstore(
            splits, embeddings, force_reindex=True, allow_reindex=True
        )
        rs._save_model_config()
        rs.save_pdf_hashes(rs.get_pdf_hashes([str(ingestion_pdf)]))
        ingestion_duration = time.perf_counter() - ingestion_started
        result["timings"]["ingestion_seconds"] = ingestion_duration

        import chromadb

        temp_client = chromadb.PersistentClient(path=str(temp_chroma))
        collection = temp_client.get_collection(collection_name)
        indexed = collection.get(
            where={"test_run_id": TEST_RUN_ID}, include=["documents", "metadatas"]
        )
        ids = indexed.get("ids", [])
        documents = indexed.get("documents", [])
        metadatas = indexed.get("metadatas", [])
        chunk_rows = [
            {
                "id": chunk_id,
                "metadata": metadata,
                "text_preview": (text or "")[:700],
                "text_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
            }
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]
        duplicate = len(set(row["text_sha256"] for row in chunk_rows)) != len(chunk_rows)
        page2_exists = any(
            all(
                " ".join(phrase.split()).casefold()
                in " ".join((text or "").split()).casefold()
                for phrase in (
                    "Windshield glass damage",
                    "partial comprehensive insurance",
                    "150 euros",
                    "TEST-KFZ-2026-1001",
                )
            )
            for text in documents
        )
        page3_exists = any(
            "No general deductible applies".casefold()
            in " ".join((text or "").split()).casefold()
            and "Personal liability insurance".casefold()
            in " ".join((text or "").split()).casefold()
            for text in documents
        )
        result["ingestion"].update(
            {
                "isolation_method": "isolated temporary Chroma directory and collection",
                "temporary_collection": True,
                "collection_name": collection_name,
                "chroma_path": str(temp_chroma),
                "ingestion_pdf_path": str(ingestion_pdf),
                "pages": pages,
                "chunks_created": len(ids),
                "chunk_ids": ids,
                "chunk_metadata": chunk_rows,
                "collection_count_before_ingestion": 0,
                "collection_count_after_ingestion": collection.count(),
                "page2_evidence_exists": page2_exists,
                "page3_distractor_exists": page3_exists,
                "duplicates_detected": duplicate,
                "successful": len(ids) == 3 and page2_exists and page3_exists and not duplicate,
                "production_functions": [
                    "api.rag_service.load_pdf_source",
                    "api.rag_service.load_and_split_documents",
                    "api.rag_service.initialize_embeddings",
                    "api.rag_service.build_vectorstore",
                    "api.rag_service.build_retriever",
                    "api.rag_service.rerank_documents",
                ],
                "test_only_adapter": "Adds exact synthetic metadata before production embedding/insertion and redirects storage paths in the test process.",
            }
        )
        result["collection_counts"]["after_ingestion"] = production_counts(production_chroma)
        atomic_json(json_path, result)
        if not result["ingestion"]["successful"]:
            result["failure_stage"] = "Ingestion"
            raise RuntimeError("Synthetic document was not indexed correctly")
        log(f"Ingestion PASS in {ingestion_duration:.3f}s with {len(ids)} chunks")

        inventory = ollama_inventory(rs.SETTINGS.ollama_base_url)
        required_models = {rs.SETTINGS.roles.answer, rs.SETTINGS.roles.self_check}
        installed = set(inventory["models"])
        callable_results = []
        warmup_started = time.perf_counter()
        for model in (rs.SETTINGS.roles.self_check, rs.SETTINGS.roles.answer):
            callable_results.append(ollama_request(rs.SETTINGS.ollama_base_url, model, "Reply only OK.", 1))
        warmup_duration = time.perf_counter() - warmup_started
        result["timings"]["warmup_seconds"] = warmup_duration
        result["warmup"] = {
            "query": "Reply only OK.",
            "type": "direct Ollama one-token model warm-up (not the main scenario)",
            "duration_seconds": warmup_duration,
            "results": callable_results,
            "successful": all(item["successful"] for item in callable_results),
            "models_loaded_after": ollama_ps(),
        }

        del vector_store, embeddings, splits, loaded_docs
        vector_store = None
        gc.collect()

        port = free_port()
        backend_url = f"http://127.0.0.1:{port}"
        result["environment"]["backend_url"] = backend_url
        child_log = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            "--port",
            str(port),
            "--trace-path",
            str(trace_path),
        ]
        server_process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=child_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        child_log.close()
        result["environment"]["backend_pid"] = server_process.pid
        log(f"Started isolated backend PID={server_process.pid} URL={backend_url}")
        health = wait_for_health(backend_url + "/health", server_process, timeout=180)
        trace_events = read_trace(trace_path)
        server_preflight = latest_event(trace_events, "server_preflight") or {}
        result["timings"]["retrieval_service_initialization_seconds"] = server_preflight.get(
            "retrieval_service_duration_seconds", NOT_AVAILABLE
        )
        result["preflight"] = {
            "backend_running": health.get("status") == "ok",
            "backend_uses_current_env_values": (
                health.get("configured_answer_model") == rs.SETTINGS.roles.answer
                and health.get("query_rewrite_enabled") == rs.SETTINGS.retrieval.query_rewrite_enabled
                and health.get("safetyMode") == rs.safety_mode()
            ),
            "backend_url": backend_url,
            "backend_pid": server_process.pid,
            "backend_health": health,
            "ollama_reachable": inventory.get("http_status") == 200,
            "ollama_url": rs.SETTINGS.ollama_base_url,
            "model_inventory": inventory["models"],
            "required_models_installed": required_models.issubset(installed),
            "models_callable": all(item["successful"] for item in callable_results),
            "configured_answer_model_matches_intended": (
                rs.SETTINGS.roles.answer == rs.SETTINGS.preferred_answer_model
            ),
            "retrieval_service_ready": bool(server_preflight),
            "pipeline_ready": health.get("pipelineReady") is True,
            "pipeline_initialization_error": health.get("pipelineInitError"),
            "force_retrieval": rs.SETTINGS.retrieval.force_retrieval,
            "rewrite_disabled": not rs.SETTINGS.retrieval.query_rewrite_enabled,
            "compression_disabled": not rs.SETTINGS.retrieval.enable_context_compression,
            "safety_enabled": rs.safety_enabled(),
            "safety_rule_based_without_llm": rs.SETTINGS.safety.backend == "nemo",
            "safety_mode": rs.safety_mode(),
            "safety_backend": rs.SETTINGS.safety.backend,
            "groundedness_threshold": rs.SETTINGS.safety.min_groundedness,
            "groundedness_threshold_source": rs.SETTINGS.safety.min_groundedness_source,
            "groundedness_calibration_file": str(
                rs.SETTINGS.safety.groundedness_calibration_file
            ),
            "synthetic_document_indexed": result["ingestion"]["successful"],
            "synthetic_document_isolated": (
                temp_chroma.resolve() != production_chroma.resolve()
                and result["collection_counts"]["after_ingestion"] == baseline
            ),
            "production_counts": result["collection_counts"]["after_ingestion"],
            "models_warm": ollama_ps(),
            "processes_at_preflight": process_inventory(),
            "stale_backend_or_mcp_detected": any(
                "backend_api.py" in str(item.get("CommandLine", ""))
                or "mcp" in str(item.get("CommandLine", "")).casefold()
                for item in process_inventory()
            ),
        }
        atomic_json(json_path, result)
        if not all(
            (
                result["preflight"]["backend_running"],
                result["preflight"]["ollama_reachable"],
                result["preflight"]["required_models_installed"],
                result["preflight"]["models_callable"],
                result["preflight"]["configured_answer_model_matches_intended"],
                result["preflight"]["retrieval_service_ready"],
                result["preflight"]["pipeline_ready"],
                result["preflight"]["force_retrieval"],
                result["preflight"]["rewrite_disabled"],
                result["preflight"]["compression_disabled"],
                result["preflight"]["safety_enabled"],
                result["preflight"]["synthetic_document_isolated"],
                not result["preflight"]["stale_backend_or_mcp_detected"],
            )
        ):
            result["failure_stage"] = "Preflight"
            raise RuntimeError("One or more required preflight checks failed")
        log("Preflight PASS")

        import requests

        query_started_wall = utc_now()
        query_started = time.perf_counter()
        try:
            response = requests.post(
                backend_url + "/api/ask",
                json={"question": QUESTION, "shortAnswer": False, "structuredAnswer": False},
                timeout=args.timeout_seconds,
            )
            query_duration = time.perf_counter() - query_started
            raw_response = response.text
            result["query"].update(
                {
                    "start_time": query_started_wall,
                    "end_time": utc_now(),
                    "total_duration_seconds": query_duration,
                    "http_status": response.status_code,
                    "raw_api_response": raw_response,
                    "timeout": False,
                    "error": None,
                    "status": "SUCCESS" if response.status_code == 200 else "HTTP_ERROR",
                }
            )
            if response.status_code != 200:
                result["failure_stage"] = "Full pipeline API"
                raise RuntimeError(f"POST /api/ask returned HTTP {response.status_code}: {raw_response[:500]}")
            api_payload = response.json()
            result["query"]["api_response"] = api_payload
            result["query"]["full_pipeline_executed"] = True
            result["timings"]["main_scenario_seconds"] = query_duration
            log(f"Main POST /api/ask completed in {query_duration:.3f}s")
        except requests.Timeout as exc:
            result["query"].update(
                {
                    "start_time": query_started_wall,
                    "end_time": utc_now(),
                    "total_duration_seconds": time.perf_counter() - query_started,
                    "status": "TIMEOUT",
                    "timeout": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "http_status": NOT_AVAILABLE,
                }
            )
            result["failure_stage"] = "Main query timeout"
            raise
        finally:
            atomic_json(json_path, result)

        time.sleep(1)
        trace_events = read_trace(trace_path)
        result["trace_events"] = trace_events
        retrieval_event = latest_event(trace_events, "retrieval_search")
        scores_event = latest_event(trace_events, "reranker_scores")
        rerank_event = latest_event(trace_events, "reranking")
        self_event = latest_event(trace_events, "self_check_llm")
        answer_llm_event = latest_event(trace_events, "answer_llm")
        answer_event = latest_event(trace_events, "answer_generation")
        citation_event = latest_event(trace_events, "citation_processing")
        safety_pre = latest_event(trace_events, "safety_pre_query")
        safety_context = latest_event(trace_events, "safety_context")
        safety_post = latest_event(trace_events, "safety_post_generation")
        audit_event = latest_event(trace_events, "audit")

        retrieved_docs = (retrieval_event or {}).get("documents", [])
        retrieval_attempts = sum(
            1
            for event in trace_events
            if event.get("event") == "retrieval_search" and event.get("query") == QUESTION
        )
        reranked_docs = (rerank_event or {}).get("output_documents", [])
        relevant_rank = find_rank(reranked_docs, "deductible of 150 euros")
        distractor_rank = find_rank(reranked_docs, "No general deductible applies")
        relevant_score = find_score(rerank_event, scores_event, "deductible of 150 euros")
        distractor_score = find_score(rerank_event, scores_event, "No general deductible applies")
        correct_source = any(
            Path(str(doc.get("metadata", {}).get("source", ""))).name == PDF_NAME
            for doc in retrieved_docs
        )
        correct_page = any(
            classify_page(doc, "Windshield glass damage")
            and classify_page(doc, "partial comprehensive insurance")
            and classify_page(doc, "150 euros")
            and classify_page(doc, "TEST-KFZ-2026-1001")
            for doc in retrieved_docs
        )
        retrieval_pass = correct_source and correct_page
        rerank_pass = (
            relevant_rank is not None
            and distractor_rank is not None
            and relevant_rank < distractor_rank
        )
        exact_evidence = next(
            (doc.get("content") for doc in retrieved_docs if classify_page(doc, "deductible of 150 euros")),
            NOT_AVAILABLE,
        )
        result["retrieval"] = {
            "attempts": retrieval_attempts,
            "documents": retrieved_docs,
            "document_count": len(retrieved_docs),
            "correct_source_retrieved": correct_source,
            "correct_page_retrieved": correct_page,
            "correct_contract_in_context": any(classify_page(doc, "TEST-KFZ-2026-1001") for doc in reranked_docs),
            "exact_supporting_text": exact_evidence,
            "status": "PASS" if retrieval_pass else "FAIL",
        }
        result["reranking"] = {
            "candidate_count_before": (rerank_event or {}).get("candidate_count", NOT_AVAILABLE),
            "final_count_after": (rerank_event or {}).get("final_count", NOT_AVAILABLE),
            "relevant_chunk_rank": relevant_rank,
            "distractor_chunk_rank": distractor_rank,
            "relevant_score": relevant_score,
            "distractor_score": distractor_score,
            "duration_seconds": (rerank_event or {}).get("duration_seconds", NOT_AVAILABLE),
            "status": "PASS" if rerank_pass else "FAIL",
        }

        raw_self = str((self_event or {}).get("raw_output", NOT_AVAILABLE))
        parsed_self = rs._parse_self_check_decision(raw_self) if self_event else NOT_AVAILABLE
        audit_fields = (audit_event or {}).get("fields", {})
        result["self_check"] = {
            "raw_output": raw_self,
            "parsed_decision": parsed_self,
            "duration_seconds": (self_event or {}).get("duration_seconds", NOT_AVAILABLE),
            "model": rs.SETTINGS.roles.self_check,
            "correct_page2_evidence_passed": any(
                classify_page(doc, "deductible of 150 euros") for doc in reranked_docs
            ),
            "false_positive": False,
            "false_negative": parsed_self != "RELEVANT" and correct_page,
            "status": "PASS" if parsed_self == "RELEVANT" else "FAIL",
        }
        result["query"]["query_rewritten"] = bool(audit_fields.get("query_rewrite_applied", False))
        result["query"]["retry_count"] = audit_fields.get("retries", NOT_AVAILABLE)
        result["query"]["retrieval_attempts"] = retrieval_attempts
        result["query"]["request_id"] = NOT_AVAILABLE
        result["query"]["audit_id"] = NOT_AVAILABLE

        final_answer = str(result["query"]["api_response"].get("answer", ""))
        result["query"]["number_of_final_sources"] = len(
            result["query"]["api_response"].get("sources", [])
        )
        result["query"]["final_sources"] = result["query"]["api_response"].get("sources", [])
        answer_validation = validate_answer(final_answer)
        answer_validation["text"] = final_answer
        answer_validation["model"] = rs.SETTINGS.roles.answer
        supported_claims = []
        if answer_validation["coverage_correct"]:
            supported_claims.append("Coverage - PDF page 2")
        if answer_validation["coverage_type_correct"]:
            supported_claims.append("Partial comprehensive insurance - PDF page 2")
        if answer_validation["deductible_correct"]:
            supported_claims.append("150 euros per claim - PDF page 2")
        if answer_validation["contract_correct"]:
            supported_claims.append("TEST-KFZ-2026-1001 - PDF pages 1/2")
        if answer_validation["customer_correct"]:
            supported_claims.append("Lara Neumann - PDF pages 1/2")
        unsupported = list(answer_validation["contradictions"])
        distractor_handled = not answer_validation["contradictions"] and (
            relevant_rank is not None and distractor_rank is not None and relevant_rank < distractor_rank
        )
        answer_validation.update(
            {
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported,
                "groundedness_status": "PENDING",
                "hallucinated_details": [],
                "distractor_handled_correctly": distractor_handled,
            }
        )
        result["answer"] = answer_validation

        citation_items = parse_citations(final_answer, retrieved_docs)
        citation_valid = any(
            item["retrieved"]
            and item["supports_all_material_claims"]
            and Path(str(item["source"])).name == PDF_NAME
            for item in citation_items
        )
        result["citations"] = {
            "count": len(citation_items),
            "strings": [item["citation"] for item in citation_items],
            "items": citation_items,
            "valid": citation_valid,
            "status": "PASS" if citation_valid else "FAIL",
        }

        def safety_success(event: dict[str, Any] | None) -> bool:
            if not event:
                return False
            payload = event.get("result", {})
            details = payload.get("details", {}) or {}
            if details.get("nemo_runtime_error"):
                return False
            return payload.get("allow") is True or payload.get("action") == "redact"

        context_guardrail = guardrail_details(safety_context, "context")
        output_guardrail = guardrail_details(safety_post, "output")
        context_pii = context_guardrail.get("pii", {}) or {}
        output_pii = output_guardrail.get("pii", {}) or {}
        context_allowed_types = context_pii.get("allowed_types", {}) or {}
        context_redacted_types = context_pii.get("redacted_types", {}) or {}
        output_allowed_types = output_pii.get("allowed_types", {}) or {}
        output_redacted_types = output_pii.get("redacted_types", {}) or {}
        context_result_details = (
            (safety_context or {}).get("result", {}).get("details", {}) or {}
        )
        sanitized_docs = context_result_details.get("sanitized_docs", []) or []
        sanitized_context = "\n".join(
            str(doc.get("page_content", ""))
            for doc in sanitized_docs
            if isinstance(doc, dict)
        )
        contract_allowed_context = (
            int(context_allowed_types.get("contract_id", 0)) >= 1
            and int(context_redacted_types.get("contract_id", 0)) == 0
            and "TEST-KFZ-2026-1001" in sanitized_context
            and "[REDACTED_CONTRACT_ID]" not in sanitized_context
        )
        protected_context_pii = all(
            int(context_redacted_types.get(pii_type, 0)) >= 1
            for pii_type in ("customer_number", "date_of_birth", "address")
        )
        dates_preserved = all(
            value in sanitized_context for value in ("01.01.2026", "31.12.2026")
        ) and "[REDACTED_PHONE]" not in sanitized_context
        contract_allowed_output = (
            int(output_allowed_types.get("contract_id", 0)) >= 1
            and int(output_redacted_types.get("contract_id", 0)) == 0
            and "TEST-KFZ-2026-1001" in final_answer
        )
        output_has_no_disallowed_pii = not output_redacted_types

        post_scores = (safety_post or {}).get("result", {}).get("scores", {}) or {}
        groundedness_score = post_scores.get("groundedness", NOT_AVAILABLE)
        groundedness_threshold = rs.SETTINGS.safety.min_groundedness
        groundedness_pass = (
            isinstance(groundedness_score, (int, float))
            and groundedness_score >= groundedness_threshold
            and "low_groundedness"
            not in ((safety_post or {}).get("result", {}).get("reasons", []) or [])
        )
        result["answer"]["groundedness_score"] = groundedness_score
        result["answer"]["groundedness_threshold"] = groundedness_threshold
        result["answer"]["groundedness_algorithm"] = output_guardrail.get(
            "groundedness", {}
        ).get("algorithm_version", NOT_AVAILABLE)
        result["answer"]["groundedness_status"] = (
            "PASS" if groundedness_pass else "FAIL"
        )

        result["safety"] = {
            "pre_query": (safety_pre or {}).get("result", NOT_AVAILABLE),
            "context": (safety_context or {}).get("result", NOT_AVAILABLE),
            "post_generation": (safety_post or {}).get("result", NOT_AVAILABLE),
            "pre_query_successful": safety_success(safety_pre),
            "context_successful": safety_success(safety_context),
            "post_generation_successful": safety_success(safety_post),
            "redaction": any(
                (event or {}).get("result", {}).get("action") == "redact"
                for event in (safety_pre, safety_context, safety_post)
            ),
            "block": any(not safety_success(event) for event in (safety_pre, safety_context, safety_post)),
            "fallback": any(
                (event or {}).get("result", {}).get("action") == "fallback"
                for event in (safety_pre, safety_context, safety_post)
            ),
            "context_pii": context_pii,
            "output_pii": output_pii,
            "contract_allowed_in_context": contract_allowed_context,
            "contract_allowed_in_output": contract_allowed_output,
            "protected_context_pii_redacted": protected_context_pii,
            "coverage_dates_preserved": dates_preserved,
            "output_has_no_disallowed_pii": output_has_no_disallowed_pii,
            "focused_phone_date_validation": result["focused_safety_validation"],
        }
        safety_pass = all(
            (
                result["safety"]["pre_query_successful"],
                result["safety"]["context_successful"],
                result["safety"]["post_generation_successful"],
                contract_allowed_context,
                contract_allowed_output,
                protected_context_pii,
                dates_preserved,
                output_has_no_disallowed_pii,
                result["focused_safety_validation"]["dates_not_phone"],
                result["focused_safety_validation"]["phone_detected"],
                result["focused_safety_validation"]["list_numbering_preserved"],
            )
        )
        result["safety"]["status"] = "PASS" if safety_pass else "FAIL"

        audit_rows = []
        if temp_audit.exists():
            for line in temp_audit.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                    if row.get("query") == QUESTION:
                        audit_rows.append(row)
                except json.JSONDecodeError:
                    continue
        audit_row = audit_rows[-1] if audit_rows else None
        audit_serialized = json.dumps(audit_row or {}, ensure_ascii=False)
        main_language_pass = bool(
            audit_row and audit_row.get("response_language") == "English"
        )
        audit_contract_visible = (
            "TEST-KFZ-2026-1001" in audit_serialized
            and "[REDACTED_CONTRACT_ID]" not in audit_serialized
        )
        german_telemetry_pass = (
            result["telemetry"].get("focused_german", {}).get("status") == "PASS"
        )
        telemetry_pass = main_language_pass and german_telemetry_pass
        result["audit"] = {
            "enabled": True,
            "entry_found": audit_row is not None,
            "entry_count_for_query": len(audit_rows),
            "audit_file": str(temp_audit),
            "audit_id": NOT_AVAILABLE,
            "entry": audit_row or NOT_AVAILABLE,
            "response_language": (
                audit_row.get("response_language") if audit_row else NOT_AVAILABLE
            ),
            "contract_visible": audit_contract_visible,
            "status": (
                "PASS"
                if audit_row is not None and main_language_pass and audit_contract_visible
                else "FAIL"
            ),
        }
        result["telemetry"].update(
            {
                "main_answer": {
                    "expected": "English",
                    "recorded_response_language": result["audit"]["response_language"],
                    "status": "PASS" if main_language_pass else "FAIL",
                },
                "status": "PASS" if telemetry_pass else "FAIL",
            }
        )

        result["timings"].update(
            {
                "retrieval_seconds": (retrieval_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "reranking_seconds": (rerank_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "self_check_seconds": (self_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "answer_generation_seconds": (answer_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "answer_llm_seconds": (answer_llm_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "citation_processing_seconds": (citation_event or {}).get("duration_seconds", NOT_AVAILABLE),
                "safety_initialization_seconds": (latest_event(trace_events, "safety_initialization") or {}).get("duration_seconds", NOT_AVAILABLE),
                "pre_safety_seconds": (safety_pre or {}).get("duration_seconds", NOT_AVAILABLE),
                "context_safety_seconds": (safety_context or {}).get("duration_seconds", NOT_AVAILABLE),
                "post_safety_seconds": (safety_post or {}).get("duration_seconds", NOT_AVAILABLE),
                "audit_seconds": (audit_event or {}).get("duration_seconds", NOT_AVAILABLE),
            }
        )
        safety_timing_values = [
            result["timings"].get(key)
            for key in (
                "safety_initialization_seconds",
                "pre_safety_seconds",
                "context_safety_seconds",
                "post_safety_seconds",
            )
        ]
        result["timings"]["safety_seconds"] = (
            sum(float(value) for value in safety_timing_values)
            if all(isinstance(value, (int, float)) for value in safety_timing_values)
            else NOT_AVAILABLE
        )
        stage_timing_keys = (
            "retrieval_seconds",
            "reranking_seconds",
            "self_check_seconds",
            "answer_llm_seconds",
            "citation_processing_seconds",
            "pre_safety_seconds",
            "context_safety_seconds",
            "post_safety_seconds",
            "audit_seconds",
        )
        numeric_timings = {
            key: result["timings"].get(key)
            for key in stage_timing_keys
            if isinstance(result["timings"].get(key), (int, float))
        }
        if numeric_timings:
            bottleneck_key = max(numeric_timings, key=numeric_timings.get)
            result["main_bottleneck"] = f"{bottleneck_key} ({numeric_timings[bottleneck_key]:.3f}s)"

        result["collection_counts"]["after_execution"] = production_counts(production_chroma)
        isolation_before_cleanup = all(
            result["collection_counts"].get(stage) == baseline
            for stage in ("after_ingestion", "after_execution")
        ) and result["preflight"].get("synthetic_document_isolated") is True
        functional_pass = all(
            (
                result["pdf"]["valid"],
                result["pdf"].get("pages") == 3,
                result["ingestion"]["successful"],
                result["ingestion"].get("chunks_created") == 3,
                result["query"]["full_pipeline_executed"],
                retrieval_pass,
                rerank_pass,
                parsed_self == "RELEVANT",
                not result["query"]["query_rewritten"],
                safety_pass,
                result["audit"]["status"] == "PASS",
            )
        )
        quality_pass = all(answer_validation["fact_results"].get(name) == "PASS" for name in (
            "coverage", "coverage_type", "deductible", "contract", "customer"
        )) and groundedness_pass and not unsupported and distractor_handled and citation_valid
        performance_pass = not result["query"]["timeout"] and result["query"]["status"] == "SUCCESS"
        result["results"].update(
            {
                "functional": "PASS" if functional_pass else "FAIL",
                "quality": "PASS" if quality_pass else "FAIL",
                "safety": "PASS" if safety_pass else "FAIL",
                "isolation": "PASS" if isolation_before_cleanup else "FAIL",
                "telemetry": "PASS" if telemetry_pass else "FAIL",
                "groundedness_calibration": (
                    result["groundedness_calibration"]["status"]
                ),
                "performance": "PASS" if performance_pass else "FAIL",
            }
        )
        result["performance"] = {
            "execution_reliability": "PASS" if performance_pass else "FAIL",
            "timeout": result["query"]["timeout"],
            "user_facing_latency_seconds": result["query"].get("total_duration_seconds"),
            "latency_judgement": "MEASURED_NOT_ASSUMED_ACCEPTABLE",
        }
        atomic_json(json_path, result)
    except Exception as exc:
        if result.get("failure_stage") in (None, NOT_AVAILABLE):
            result["failure_stage"] = "Harness execution"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["exception_traceback"] = traceback.format_exc()
        if result["query"].get("status") == "NOT_EXECUTED":
            result["results"]["overall"] = "NOT EXECUTED"
        log(f"FAILURE at {result['failure_stage']}: {result['failure_reason']}")
        atomic_json(json_path, result)
    finally:
        cleanup_started = time.perf_counter()
        result["cleanup"]["executed"] = True
        try:
            if server_process is not None:
                stop_process(server_process, log)
            collection = None
            indexed = None
            temp_client = None
            embeddings = None
            splits = None
            loaded_docs = None
            del vector_store
            vector_store = None
            gc.collect()
            temporary_collection_deleted = False
            synthetic_remaining: int | str = NOT_AVAILABLE
            if temp_chroma.exists():
                import chromadb

                client = chromadb.PersistentClient(path=str(temp_chroma))
                try:
                    collection = client.get_collection(collection_name)
                    payload = collection.get(where={"test_run_id": TEST_RUN_ID}, include=[])
                    ids = payload.get("ids", [])
                    if ids:
                        collection.delete(ids=ids)
                    synthetic_remaining = len(
                        collection.get(where={"test_run_id": TEST_RUN_ID}, include=[]).get("ids", [])
                    )
                    client.delete_collection(collection_name)
                    try:
                        client.get_collection(collection_name)
                    except Exception:
                        temporary_collection_deleted = True
                except Exception:
                    try:
                        client.get_collection(collection_name)
                    except Exception:
                        temporary_collection_deleted = True
                        synthetic_remaining = 0
                try:
                    client._system.stop()
                except Exception:
                    pass
                del client
                try:
                    from chromadb.api.client import SharedSystemClient

                    SharedSystemClient.clear_system_cache()
                except Exception:
                    pass
                gc.collect()
            temporary_directory_deleted = False
            if temp_root.exists() and not args.keep_temp_index:
                for attempt in range(5):
                    try:
                        shutil.rmtree(temp_root)
                        temporary_directory_deleted = not temp_root.exists()
                        break
                    except PermissionError:
                        gc.collect()
                        time.sleep(1 + attempt)
            elif args.keep_temp_index:
                temporary_directory_deleted = False
            final_counts = production_counts(production_chroma)
            result["collection_counts"]["after_cleanup"] = final_counts
            baseline = result["collection_counts"].get("before", {})
            differences = {
                name: (
                    final_counts.get(name) - baseline.get(name)
                    if isinstance(final_counts.get(name), int) and isinstance(baseline.get(name), int)
                    else NOT_AVAILABLE
                )
                for name in ("insurance_rag_collection", "insuranceqa_collection")
            }
            integrity_pass = final_counts == baseline and all(value == 0 for value in differences.values())
            cleanup_success = (
                temporary_collection_deleted
                and synthetic_remaining == 0
                and (temporary_directory_deleted or args.keep_temp_index)
                and integrity_pass
            )
            result["cleanup"].update(
                {
                    "temporary_collection_deleted": temporary_collection_deleted,
                    "temporary_directory_deleted": temporary_directory_deleted,
                    "synthetic_chunks_remaining": synthetic_remaining,
                    "successful": cleanup_success,
                    "production_documents_modified": False if integrity_pass else NOT_AVAILABLE,
                }
            )
            result["collection_integrity"] = {
                "expected": baseline,
                "actual": final_counts,
                "differences": differences,
                "no_complete_reindex": True,
                "no_production_collection_deletion": True,
                "no_production_collection_recreation": True,
                "no_unrelated_insert_or_delete": integrity_pass,
                "status": "PASS" if integrity_pass else "FAIL",
            }
            result["results"]["cleanup"] = "PASS" if cleanup_success else "FAIL"
            result["results"]["isolation"] = (
                "PASS"
                if result["results"].get("isolation") == "PASS" and integrity_pass
                else "FAIL"
            )
            if not cleanup_success:
                result["failure_stage"] = "Cleanup"
                result["failure_reason"] = "Cleanup or production collection integrity verification failed"
                result["results"]["functional"] = "FAIL"
                result["results"]["overall"] = "FAIL"
        except Exception as cleanup_exc:
            result["cleanup"].update(
                {
                    "successful": False,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                }
            )
            result["failure_stage"] = "Cleanup"
            result["failure_reason"] = result["cleanup"]["error"]
            result["results"]["functional"] = "FAIL"
            result["results"]["cleanup"] = "FAIL"
            result["results"]["overall"] = "FAIL"
            log(f"CLEANUP FAILURE: {result['cleanup']['error']}")
        result["timings"]["cleanup_seconds"] = time.perf_counter() - cleanup_started

        required_result_names = (
            "functional",
            "quality",
            "safety",
            "isolation",
            "cleanup",
            "telemetry",
            "groundedness_calibration",
            "performance",
        )
        required_all_pass = all(
            result["results"].get(name) == "PASS" for name in required_result_names
        )
        if result["query"].get("full_pipeline_executed") is not True:
            result["results"]["overall"] = "NOT EXECUTED"
        else:
            result["results"]["overall"] = "PASS" if required_all_pass else "FAIL"
        if required_all_pass:
            result["failure_stage"] = None
            result["failure_reason"] = None
            result["recommended_next_action"] = "Retain the report as reproducible evidence and rerun after material pipeline changes."
        elif result.get("failure_reason") is None:
            failed = [
                name
                for name in required_result_names
                if result["results"].get(name) != "PASS"
            ]
            result["failure_stage"] = result.get("failure_stage") if result.get("failure_stage") not in (None, NOT_AVAILABLE) else ", ".join(failed)
            result["failure_reason"] = "One or more mandatory pass criteria failed; see stage-level results."
        result["remaining_limitations"] = (
            "Request ID and audit ID are not emitted by the current public API and are recorded as not_available. "
            "PDF page metadata is zero-based in PyPDFLoader; human page 2 is metadata page 1. "
            "The calibrated threshold is empirical for the labeled synthetic insurance set and should be "
            "revalidated when document style, language mix, or generation models change. Local CPU Ollama "
            "latency is measured but is not labeled acceptable solely because the request avoided timeout."
        )
        atomic_json(json_path, result)
        md_path.write_text(markdown_report(result), encoding="utf-8")
        summary = concise_summary(result)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + summary + "\n")
        print(summary, flush=True)
        print(f"JSON report: {json_path}", flush=True)
        print(f"Markdown report: {md_path}", flush=True)
        print(f"Log: {log_path}", flush=True)

    return 0 if result["results"]["overall"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic customer insurance E2E scenario.")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--keep-pdf", action="store_true", default=True)
    parser.add_argument("--keep-temp-index", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--use-temporary-collection", action="store_true", default=True)
    parser.add_argument("--force-cleanup", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--trace-path", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        return serve(args)
    if args.pdf_only:
        path = PROJECT_ROOT / "tests" / "fixtures" / PDF_NAME
        duration = create_pdf(path)
        print(json.dumps({"path": str(path), "duration_seconds": duration}, indent=2))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())

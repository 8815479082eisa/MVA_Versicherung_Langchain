import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ------------------------------------------------------------
# 0) Load .env BEFORE importing main.py (important!)
#    Because main.py prompts for OPENAI_API_KEY if missing.
# ------------------------------------------------------------
BASE_DIR = "/opt/mva"
ENV_FILE = os.path.join(BASE_DIR, ".env")

def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

load_env_file(ENV_FILE)

# Now safe to import main
import main as m  # noqa: E402

# ------------------------------------------------------------
# 1) Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mva_api")

# ------------------------------------------------------------
# 2) Config
# ------------------------------------------------------------
PDF_DIR = os.getenv("PDF_DIRECTORY", os.path.join(BASE_DIR, "docs"))
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", os.path.join(BASE_DIR, "audit.log"))

# Patch main.py globals for absolute paths (to avoid surprises)
m.PDF_DIRECTORY = PDF_DIR
m.AUDIT_LOG_FILE = AUDIT_LOG_FILE
m.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", os.path.join(BASE_DIR, "chroma_db"))
m.PDF_HASH_FILE = os.getenv("PDF_HASH_FILE", os.path.join(BASE_DIR, ".pdf_hashes.json"))

# ------------------------------------------------------------
# 3) FastAPI App
# ------------------------------------------------------------
app = FastAPI(
    title="MVA Agentic RAG API",
    version="1.0.0",
)

PIPELINE: Dict[str, Any] = {
    "ready": False,
    "chat_history": [],
}

# ------------------------------------------------------------
# 4) Models
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    effective_query: str
    retrieval_decision: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: str

# ------------------------------------------------------------
# 5) UI (served at /)
# ------------------------------------------------------------
UI_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MVA Agentic RAG – Chat</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:0;background:#f6f7fb}
    .wrap{max-width:900px;margin:0 auto;padding:24px}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.04)}
    header{padding:16px 18px;border-bottom:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center}
    header h1{font-size:18px;margin:0}
    .badge{font-size:12px;color:#6b7280}
    .chat{padding:18px;min-height:420px;max-height:60vh;overflow:auto}
    .msg{margin:10px 0;display:flex;gap:10px}
    .role{font-weight:700;min-width:72px}
    .bubble{white-space:pre-wrap;background:#f3f4f6;border-radius:10px;padding:10px 12px;flex:1}
    .assistant .bubble{background:#eef2ff}
    .meta{font-size:12px;color:#6b7280;margin-top:6px}
    .composer{display:flex;gap:10px;padding:14px;border-top:1px solid #e5e7eb}
    textarea{flex:1;resize:vertical;min-height:46px;max-height:160px;padding:10px;border:1px solid #d1d5db;border-radius:10px;font:inherit}
    button{padding:10px 14px;border:0;border-radius:10px;background:#111827;color:#fff;font-weight:700;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .hint{padding:10px 18px;color:#6b7280;font-size:13px;border-bottom:1px solid #e5e7eb}
    .err{color:#b91c1c}
    .src{margin-top:8px;font-size:12px;color:#374151}
    .src code{background:#f3f4f6;padding:2px 6px;border-radius:6px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <header>
        <h1>MVA Agentic RAG – Chat</h1>
        <div class="badge" id="status">checking…</div>
      </header>

      <div class="hint">
        API: <span id="apiBase"></span>
      </div>

      <div id="chat" class="chat"></div>

      <div class="composer">
        <textarea id="q" placeholder="Frage eingeben… (Enter = senden, Shift+Enter = neue Zeile)"></textarea>
        <button id="send">Senden</button>
      </div>
    </div>
  </div>

<script>
  const chatEl = document.getElementById("chat");
  const qEl = document.getElementById("q");
  const sendBtn = document.getElementById("send");
  const statusEl = document.getElementById("status");
  const apiBaseEl = document.getElementById("apiBase");

  const API_BASE = window.location.origin;
  apiBaseEl.textContent = API_BASE;

  function addMsg(role, text, metaHtml) {
    const row = document.createElement("div");
    row.className = "msg " + (role === "Assistant" ? "assistant" : "user");
    row.innerHTML = `
      <div class="role">${role}:</div>
      <div style="flex:1">
        <div class="bubble"></div>
        ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ``}
      </div>
    `;
    row.querySelector(".bubble").textContent = text;
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  async function health() {
    try {
      const r = await fetch(`${API_BASE}/health`);
      if (!r.ok) throw new Error();
      const j = await r.json();
      statusEl.textContent = j.ready ? "online" : "starting…";
      statusEl.className = j.ready ? "" : "";
    } catch {
      statusEl.textContent = "offline";
      statusEl.className = "err";
    }
  }

  function sourcesHtml(sources){
    if(!sources || !sources.length) return "";
    const items = sources.slice(0,6).map(s => `<div class="src">Quelle: <code>${s.source}</code> • Seite: <code>${s.page}</code></div>`).join("");
    return items;
  }

  async function send() {
    const query = qEl.value.trim();
    if (!query) return;

    addMsg("User", query);
    qEl.value = "";
    sendBtn.disabled = true;

    try {
      const t0 = performance.now();
      const r = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({query})
      });
      const t1 = performance.now();

      if (!r.ok) {
        const body = await r.text();
        throw new Error(body || `HTTP ${r.status}`);
      }

      const data = await r.json();
      const meta = [
        data.retrieval_decision ? `Decision: ${data.retrieval_decision}` : "",
        `Latency: ${(t1 - t0).toFixed(0)} ms`,
        data.effective_query && data.effective_query !== data.query ? `Rewritten: ${data.effective_query}` : ""
      ].filter(Boolean).join(" • ");

      addMsg("Assistant", data.answer ?? "(no answer)", meta + sourcesHtml(data.sources));
    } catch (e) {
      addMsg("Assistant", `Error: ${e.message}`, "Request failed");
    } finally {
      sendBtn.disabled = false;
      qEl.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  qEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  health();
  setInterval(health, 3000);
  qEl.focus();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML

# ------------------------------------------------------------
# 6) Build pipeline once at startup
# ------------------------------------------------------------
@app.on_event("startup")
def startup():
    try:
        logger.info("Building RAG pipeline...")

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it into /opt/mva/.env")

        pdf_files = m.get_pdf_files(PDF_DIR)
        if not pdf_files:
            raise RuntimeError(f"No PDFs found in {PDF_DIR}")

        all_splits = m.load_and_split_documents(pdf_files)
        embeddings = m.initialize_embeddings()

        # You can decide force_reindex based on hashes
        force_reindex = m.pdfs_have_changed()

        hybrid_retriever = m.create_hybrid_retriever(
            all_splits,
            embeddings,
            force_reindex=force_reindex,
        )

        PIPELINE["hybrid_retriever"] = hybrid_retriever
        PIPELINE["reranker_llm"] = m.initialize_reranker()
        PIPELINE["compressor_llm"] = m.initialize_compressor()
        PIPELINE["llm"] = m.initialize_llm()
        PIPELINE["router_llm"] = m.initialize_router_llm()
        PIPELINE["self_check_llm"] = m.initialize_self_check_llm()
        PIPELINE["query_rewrite_llm"] = m.initialize_query_rewrite_llm()
        PIPELINE["chat_history"] = []
        PIPELINE["ready"] = True

        logger.info("RAG pipeline ready.")
    except Exception:
        logger.exception("Startup failed")
        PIPELINE["ready"] = False

@app.get("/health")
def health():
    return {
        "ready": bool(PIPELINE.get("ready")),
        "pdf_dir": PDF_DIR,
        "time": datetime.now().isoformat(),
    }

# ------------------------------------------------------------
# 7) Chat endpoint
# ------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not PIPELINE.get("ready"):
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        chat_history: List[dict] = PIPELINE["chat_history"]

        router_llm = PIPELINE["router_llm"]
        self_check_llm = PIPELINE["self_check_llm"]
        query_rewrite_llm = PIPELINE["query_rewrite_llm"]
        reranker_llm = PIPELINE["reranker_llm"]
        llm = PIPELINE["llm"]
        hybrid_retriever = PIPELINE["hybrid_retriever"]

        decision = m.decide_retrieval(router_llm, query, chat_history)

        effective_query = query
        sources: List[Dict[str, Any]] = []

        if decision == "RETRIEVE":
            start_ts = datetime.now().isoformat()

            retries = 0
            max_retries = 2
            reranked_docs = []

            while retries < max_retries:
                retrieved_docs = hybrid_retriever(effective_query, k=8)
                reranked_docs = m.rerank_documents(
                    effective_query,
                    retrieved_docs,
                    reranker_llm,
                    top_k=5,
                )

                # Self-check (NOTE: signature is:
                # perform_self_check(self_check_llm, query_rewrite_llm, original_query, retrieved_docs, ...)
                checked_query, checked_docs = m.perform_self_check(
                    self_check_llm,
                    query_rewrite_llm,
                    effective_query,
                    reranked_docs,
                    chat_history=chat_history,
                    max_retries=1,
                )

                if checked_docs and checked_query == effective_query:
                    reranked_docs = checked_docs
                    break

                effective_query = checked_query
                retries += 1

            if not reranked_docs:
                answer = (
                    "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Anfrage finden. "
                    "Bitte versuchen Sie eine andere Formulierung."
                )
                m.audit_log(
                    query=query,
                    retrieved_docs=[],
                    compressed_context=[],
                    answer=answer,
                    chat_history=chat_history,
                    start_timestamp=start_ts,
                    end_timestamp=datetime.now().isoformat(),
                    query_rewritten=(effective_query != query),
                    self_check_passed=False,
                    retrieval_retries=retries,
                )
            else:
                # In your main.py you bypass compression for debugging; we do the same
                answer, token_usage = m.generate_answer(
                    llm,
                    effective_query,
                    reranked_docs,
                    chat_history,
                )

                m.perform_safety_checks(effective_query, reranked_docs, answer, chat_history)

                # Build sources from doc metadata
                seen = set()
                for d in reranked_docs:
                    src = os.path.basename(d.metadata.get("source", "unknown"))
                    page = d.metadata.get("page", "unknown")
                    key = (src, str(page))
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append({"source": src, "page": page})

                m.audit_log(
                    query=effective_query,
                    retrieved_docs=[],
                    compressed_context=reranked_docs,
                    answer=answer,
                    chat_history=chat_history,
                    start_timestamp=start_ts,
                    end_timestamp=datetime.now().isoformat(),
                    token_usage=token_usage if token_usage else None,
                    query_rewritten=(effective_query != query),
                    self_check_passed=True,
                    retrieval_retries=retries,
                )

            chat_history.append({"query": query, "answer": answer})

        else:
            # NO_RETRIEVE path: you can customize it later
            answer = "Diese Frage wurde ohne Dokumentensuche verarbeitet (NO_RETRIEVE)."
            chat_history.append({"query": query, "answer": answer})

        return ChatResponse(
            query=query,
            effective_query=effective_query,
            retrieval_decision=decision,
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))

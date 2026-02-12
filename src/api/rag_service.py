"""
RAG Service - Zentrale Funktion zur Verarbeitung von Fragen mit dem RAG-System

Diese Datei kapselt die gesamte RAG-Pipeline und kann sowohl von der CLI
als auch vom FastAPI-Backend verwendet werden.
"""

import os
import json
import hashlib
import glob
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import time as _time_module

# #region agent log - Debug logging for rag_service
_DEBUG_LOG_PATH = r"c:\Users\mirae\MVA_Versicherung_Langchain_main\.cursor\debug.log"
def _rag_debug_log(hyp_id, loc, msg, data=None):
    try:
        entry = {"hypothesisId": hyp_id, "location": loc, "message": msg, "data": data or {}, "timestamp": int(_time_module.time()*1000), "sessionId": "debug-session"}
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except: pass
_rag_debug_log("B", "rag_service.py:top", "rag_service module loading started", {})
# #endregion


def _normalize_pdf_key(file_path: str) -> str:
    """
    Normalize file paths for stable hashing/change detection across platforms.

    We intentionally key hashes by a normalized absolute path to avoid Windows
    path separator / relative-path mismatches (which can otherwise force
    re-indexing on every request).
    """
    try:
        return str(Path(file_path).resolve()).replace("\\", "/").lower()
    except Exception:
        # Fallback: best-effort normalization
        return str(file_path).replace("\\", "/").lower()

# #region agent log - Hypothesis B: Check langchain imports (granular)
try:
    from dotenv import load_dotenv
    _rag_debug_log("B", "rag_service.py:imports", "dotenv imported", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "dotenv FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting langchain_community.document_loaders...", {})
try:
    from langchain_community.document_loaders import PyPDFLoader
    _rag_debug_log("B", "rag_service.py:imports", "PyPDFLoader OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "PyPDFLoader FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting langchain_text_splitters...", {})
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _rag_debug_log("B", "rag_service.py:imports", "RecursiveCharacterTextSplitter OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "RecursiveCharacterTextSplitter FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting langchain_openai...", {})
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    _rag_debug_log("B", "rag_service.py:imports", "langchain_openai OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "langchain_openai FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting langchain_chroma...", {})
try:
    from langchain_chroma import Chroma
    _rag_debug_log("B", "rag_service.py:imports", "langchain_chroma OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "langchain_chroma FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting BM25Retriever...", {})
try:
    from langchain_community.retrievers import BM25Retriever
    _rag_debug_log("B", "rag_service.py:imports", "BM25Retriever OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "BM25Retriever FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting langchain_core...", {})
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    _rag_debug_log("B", "rag_service.py:imports", "langchain_core OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "langchain_core FAILED", {"error": str(_e)})
    raise

_rag_debug_log("B", "rag_service.py:imports", "Starting chromadb...", {})
try:
    import chromadb
    _rag_debug_log("B", "rag_service.py:imports", "chromadb OK", {})
except Exception as _e:
    _rag_debug_log("B", "rag_service.py:imports", "chromadb FAILED", {"error": str(_e)})
    raise
# #endregion

# Lade Umgebungsvariablen
load_dotenv()
_rag_debug_log("B", "rag_service.py:dotenv", "dotenv loaded", {"OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY"))})

# Konfiguration
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIRECTORY = str(_PROJECT_ROOT / "docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "insurance_rag_collection"
CHROMA_PERSIST_DIRECTORY = str(_PROJECT_ROOT / "chroma_db")
PDF_HASH_FILE = str(_PROJECT_ROOT / ".pdf_hashes.json")

# Globale Variablen für die Pipeline-Komponenten
_pipeline_components: Optional[Dict] = None


@dataclass
class Source:
    """Quelle eines Dokuments"""
    document_id: str
    document_title: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class AnswerResult:
    """Ergebnis einer RAG-Anfrage"""
    answer: str
    sources: List[Source]
    query: str
    latency_ms: Optional[int] = None


def get_pdf_files(directory: str) -> List[str]:
    """Get all PDF files from directory, excluding example.pdf"""
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    pdf_files = [f for f in pdf_files if not os.path.basename(f).lower() == "example.pdf"]
    return sorted(pdf_files)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_pdf_hashes(pdf_files: List[str]) -> dict:
    """Get current hashes of all PDF files"""
    hashes = {}
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            hashes[_normalize_pdf_key(pdf_file)] = compute_file_hash(pdf_file)
    return hashes


def load_saved_hashes() -> dict:
    """Load previously saved PDF hashes"""
    if os.path.exists(PDF_HASH_FILE):
        with open(PDF_HASH_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            # Backwards-compatible: normalize old keys on load
            normalized: dict = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    normalized[_normalize_pdf_key(k)] = v
            return normalized
    return {}


def save_pdf_hashes(hashes: dict):
    """Save PDF hashes to file"""
    with open(PDF_HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)


def pdfs_have_changed() -> bool:
    """Check if any PDF files have changed since last indexing"""
    pdf_files = get_pdf_files(PDF_DIRECTORY)
    if not pdf_files:
        return False
    
    current_hashes = get_pdf_hashes(pdf_files)
    saved_hashes = load_saved_hashes()
    
    if set(current_hashes.keys()) != set(saved_hashes.keys()):
        return True
    
    for pdf_file, current_hash in current_hashes.items():
        if pdf_file not in saved_hashes or saved_hashes[pdf_file] != current_hash:
            return True
    
    return False


def load_and_split_documents(pdf_files: List[str]) -> List[Document]:
    """Load and split multiple PDF documents"""
    all_splits = []
    
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return all_splits


def initialize_embeddings():
    """Initialize OpenAI embeddings"""
    return OpenAIEmbeddings(model="text-embedding-3-large")


def create_hybrid_retriever(all_splits: List[Document], embeddings, force_reindex: bool = False):
    """Create hybrid retriever (BM25 + Vector)"""
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
    
    needs_reindex = force_reindex or collection.count() == 0
    
    if needs_reindex:
        if collection.count() > 0:
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)
            vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
        
        vector_store.add_documents(documents=all_splits)
        pdf_files = get_pdf_files(PDF_DIRECTORY)
        if pdf_files:
            save_pdf_hashes(get_pdf_hashes(pdf_files))
    
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5
    
    vector_store_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    def hybrid_retriever(query: str, k: int = 5) -> List[Document]:
        bm25_docs = bm25_retriever.invoke(query)
        vs_docs = vector_store_retriever.invoke(query)
        
        combined: List[Document] = []
        seen = set()
        
        for doc in bm25_docs + vs_docs:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(doc)
            if len(combined) >= k:
                break
        
        return combined
    
    return hybrid_retriever


def initialize_reranker():
    """Initialize LLM-based reranker"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


def rerank_documents(query: str, documents: List[Document], reranker_llm, top_k: int = 3) -> List[Document]:
    """Rerank documents using LLM"""
    if not documents:
        return []
    
    doc_blocks = []
    for i, doc in enumerate(documents):
        content = doc.page_content[:1500]
        doc_blocks.append(f"DOC {i}:\n{content}\n")
    
    docs_text = "\n\n".join(doc_blocks)
    
    system_msg = (
        "Du bist ein Assistent, der Dokumente nach Relevanz sortiert.\n"
        "Du bekommst eine Nutzerfrage und mehrere Dokumente (DOC 0, DOC 1, ...).\n"
        f"Antworte NUR mit den Indizes der {top_k} relevantesten Dokumente, "
        "getrennt durch Kommas, z.B.: 1,0,3"
    )
    
    user_msg = f"Frage:\n{query}\n\nDokumente:\n{docs_text}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg),
    ])
    
    chain = prompt | reranker_llm
    response = chain.invoke({})
    
    import re
    text = response.content
    indices = re.findall(r"\d+", text)
    indices = [int(i) for i in indices][:top_k]
    
    reranked_docs: List[Document] = []
    for i in indices:
        if 0 <= i < len(documents):
            reranked_docs.append(documents[i])
    
    return reranked_docs


def initialize_compressor():
    """Initialize LLM-based compressor"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


def initialize_llm():
    """Initialize main LLM for answer generation"""
    return ChatOpenAI(model="gpt-4o", temperature=0.1)


def initialize_router_llm():
    """Initialize router LLM"""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)


def initialize_self_check_llm():
    """Initialize self-check LLM"""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)


def initialize_query_rewrite_llm():
    """Initialize query rewrite LLM"""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)


ROUTER_SYSTEM_PROMPT = """Du bist ein intelligenter Router. Deine Aufgabe ist es zu entscheiden, 
ob eine Nutzerfrage eine Informationsabfrage aus einer Wissensbasis benötigt (RETRIEVE) 
oder ob die Frage direkt beantwortet werden kann (NO_RETRIEVE). 
Antworte ausschließlich mit 'RETRIEVE' oder 'NO_RETRIEVE'.
"""


def decide_retrieval(router_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    """Decide if retrieval is needed"""
    if chat_history is None:
        chat_history = []
    
    formatted_chat_history = ""
    for turn in chat_history:
        formatted_chat_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", f"User Query: {query}\nChat History: {formatted_chat_history}"),
    ])
    
    chain = prompt_template | router_llm
    response = chain.invoke({})
    decision = response.content.strip().upper()
    return decision


SELF_CHECK_SYSTEM_PROMPT = """Du bist ein Assistent, der die Relevanz von bereitgestellten Kontextdokumenten für eine Benutzerfrage bewertet.
Antworte ausschließlich mit 'RELEVANT' oder 'IRRELEVANT'.
"""

QUERY_REWRITE_SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der Benutzeranfragen umschreibt, um bessere Suchergebnisse zu erzielen.
Ziel ist es, die ursprüngliche Frage beizubehalten, aber die Formulierung zu optimieren, wenn die vorherige Suche nicht erfolgreich war.
Antworte nur mit der umgeschriebenen Frage.
"""


def rewrite_query(query_rewrite_llm, query: str, chat_history: Optional[List[dict]] = None) -> str:
    """Rewrite query for better retrieval"""
    if chat_history is None:
        chat_history = []
    
    formatted_chat_history = ""
    for turn in chat_history:
        formatted_chat_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"
    
    system_msg = QUERY_REWRITE_SYSTEM_PROMPT + f"\nChat History:\n{formatted_chat_history}"
    user_msg = (
        f"Original Query: {query}\n"
        "Context: (geringe Relevanz)\n"
        "TASK: Schreibe die Frage so um, dass bessere Suchergebnisse entstehen."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg),
    ])
    
    chain = prompt | query_rewrite_llm
    rewritten_query = chain.invoke({}).content.strip()
    return rewritten_query


def perform_self_check(
    self_check_llm,
    query_rewrite_llm,
    original_query: str,
    retrieved_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
    max_retries: int = 2,
) -> Tuple[str, List[Document]]:
    """Perform self-check on retrieved documents"""
    if chat_history is None:
        chat_history = []
    
    current_query = original_query
    current_retrieved_docs = retrieved_docs
    
    for attempt in range(max_retries):
        if not current_retrieved_docs:
            current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
            return current_query, []
        
        context_for_self_check = "\n---\n".join(
            [doc.page_content for doc in current_retrieved_docs]
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SELF_CHECK_SYSTEM_PROMPT),
            ("user", f"User Query: {current_query}\nContext:\n{context_for_self_check}"),
        ])
        
        chain = prompt | self_check_llm
        relevance_decision = chain.invoke({}).content.strip().upper()
        
        if relevance_decision == "RELEVANT":
            return current_query, current_retrieved_docs
        else:
            current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
            return current_query, []
    
    return current_query, current_retrieved_docs


SYSTEM_PROMPT = """Du bist ein fachlicher Assistent für Versicherungsbedingungen. 
Antworte nur auf Basis der bereitgestellten Passagen. Zitiere Quelle(n) mit [Dok-ID:Seite/Abschnitt]. 
Wenn unsicher: "Keine gesicherte Auskunft, bitte Rückfrage".

--- Chat History ---
{chat_history}

"""


def generate_answer(
    llm,
    query: str,
    context_docs: List[Document],
    chat_history: Optional[List[dict]] = None,
) -> str:
    """Generate answer based on query and context documents"""
    if chat_history is None:
        chat_history = []
    
    context = "\n---\n".join([doc.page_content for doc in context_docs])
    
    formatted_chat_history = ""
    for turn in chat_history:
        formatted_chat_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_PROMPT.format(chat_history=formatted_chat_history)
            + "\nCONTEXT: {context}\nTASK: Erzeuge eine präzise, "
            "kurze Antwort + Quellenblock.",
        ),
        ("user", "{query}"),
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({"query": query, "context": context})
    
    return response.content


def document_to_source(doc: Document) -> Source:
    """Convert LangChain Document to Source dataclass"""
    source_path = doc.metadata.get("source", "unknown")
    document_title = os.path.basename(source_path) if source_path != "unknown" else "Unbekanntes Dokument"
    document_id = Path(source_path).stem if source_path != "unknown" else "unknown"
    
    # Versuche Dokument-ID aus dem Pfad zu extrahieren
    if document_id == "unknown" and source_path != "unknown":
        document_id = Path(source_path).stem
    
    page = doc.metadata.get("page")
    if isinstance(page, str):
        try:
            page = int(page)
        except ValueError:
            page = None
    
    snippet = doc.page_content[:300] if doc.page_content else None  # Erste 300 Zeichen als Snippet
    
    return Source(
        document_id=document_id,
        document_title=document_title,
        page=page,
        section=None,  # Kann später aus Metadaten extrahiert werden
        snippet=snippet
    )


def initialize_pipeline(force_reindex: bool = False):
    """Initialize the RAG pipeline components"""
    global _pipeline_components
    
    if _pipeline_components is not None and not force_reindex:
        return _pipeline_components
    
    # Ensure docs directory exists
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    # Get all PDF files
    pdf_files = get_pdf_files(PDF_DIRECTORY)
    
    if not pdf_files:
        raise ValueError(f"Keine PDF-Dateien in {PDF_DIRECTORY} gefunden. Bitte fügen Sie Versicherungsdokumente hinzu.")
    
    # Load and split documents
    all_splits = load_and_split_documents(pdf_files)
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(all_splits, embeddings, force_reindex=force_reindex)
    
    # Initialize other components
    reranker_model = initialize_reranker()
    compressor_llm = initialize_compressor()
    llm = initialize_llm()
    router_llm = initialize_router_llm()
    self_check_llm = initialize_self_check_llm()
    query_rewrite_llm = initialize_query_rewrite_llm()
    
    _pipeline_components = {
        "hybrid_retriever": hybrid_retriever,
        "reranker_model": reranker_model,
        "compressor_llm": compressor_llm,
        "llm": llm,
        "router_llm": router_llm,
        "self_check_llm": self_check_llm,
        "query_rewrite_llm": query_rewrite_llm,
    }
    
    return _pipeline_components


def run_rag(question: str, chat_history: Optional[List[dict]] = None) -> AnswerResult:
    """
    Hauptfunktion zur Verarbeitung einer Frage durch das RAG-System.
    
    Args:
        question: Die zu beantwortende Frage
        chat_history: Optional - Chat-Verlauf für Kontext
    
    Returns:
        AnswerResult mit Antwort, Quellen und Metadaten
    """
    start_time = datetime.now()
    
    if chat_history is None:
        chat_history = []
    
    # Initialize pipeline if not already done
    components = initialize_pipeline(force_reindex=pdfs_have_changed())
    
    hybrid_retriever = components["hybrid_retriever"]
    reranker_model = components["reranker_model"]
    llm = components["llm"]
    router_llm = components["router_llm"]
    self_check_llm = components["self_check_llm"]
    query_rewrite_llm = components["query_rewrite_llm"]
    
    # Router: decide if retrieval is needed
    retrieval_needed = decide_retrieval(router_llm, question, chat_history)
    
    if retrieval_needed == "RETRIEVE":
        current_query = question
        retrieved_docs: List[Document] = []
        reranked_docs: List[Document] = []
        retries = 0
        MAX_SELF_CHECK_RETRIES = 2
        
        while retries < MAX_SELF_CHECK_RETRIES:
            # Retrieve documents
            retrieved_docs = hybrid_retriever(current_query, k=8)
            
            # Rerank documents
            reranked_docs = rerank_documents(
                current_query,
                retrieved_docs,
                reranker_model,
                top_k=5,
            )
            
            # Self-check loop
            if reranked_docs:
                checked_query, checked_docs = perform_self_check(
                    self_check_llm,
                    query_rewrite_llm,
                    current_query,
                    reranked_docs,
                    chat_history,
                    max_retries=1,
                )
                
                if checked_query != current_query or not checked_docs:
                    current_query = checked_query
                    retries += 1
                    continue
                else:
                    reranked_docs = checked_docs
                    break
            else:
                current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
                retries += 1
                continue
        
        if not reranked_docs:
            answer = (
                "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer "
                "Anfrage finden. Bitte versuchen Sie eine andere Formulierung oder "
                "eine allgemeinere Frage."
            )
            sources: List[Source] = []
        else:
            # Generate answer using reranked documents
            answer = generate_answer(
                llm,
                current_query,
                reranked_docs,
                chat_history,
            )
            
            # Convert documents to sources
            sources = [document_to_source(doc) for doc in reranked_docs]
    else:
        answer = (
            "Ich kann diese Frage direkt beantworten oder benötige "
            "keine Dokumentensuche dafür."
        )
        sources = []
    
    # Calculate latency
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    
    return AnswerResult(
        answer=answer,
        sources=sources,
        query=question,
        latency_ms=latency_ms
    )


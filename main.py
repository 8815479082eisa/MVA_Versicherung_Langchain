import os
import getpass
import json
import hashlib
import glob
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma  # Persistent vector store
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import chromadb


# -------------------------------------------------------------------
#  Environment & Configuration
# -------------------------------------------------------------------
#
# ROOT CAUSE ANALYSIS & FIXES APPLIED:
# ====================================
# 
# ISSUE 1: Stale Chroma Index
#   - Problem: Code only indexed if collection.count() == 0, so old index from
#     example.pdf was being reused even after adding real Baloise PDFs.
#   - Fix: Added PDF hash tracking (.pdf_hashes.json) to detect when source PDFs
#     change. When PDFs change, Chroma collection is cleared and re-indexed.
#
# ISSUE 2: Wrong PDF File Path
#   - Problem: PDF_FILE_PATH was hardcoded to "./docs/example.pdf" (dummy file).
#   - Fix: Changed to load ALL PDFs from ./docs directory (excluding example.pdf).
#     Now processes both Baloise PDFs automatically.
#
# ISSUE 3: Missing Debug Visibility
#   - Problem: No way to see what documents were actually retrieved.
#   - Fix: Added debug logging in hybrid_retriever() to print top 3 retrieved
#     documents with source, page, and content preview (600 chars).
#
# ISSUE 4: Context Compression Loss
#   - Problem: LLM-based compression might have been losing important details.
#   - Fix: Temporarily disabled compression for debugging. Using reranked_docs
#     directly instead of compressed version. Can be re-enabled by uncommenting
#     the compress_context() call.
#
# CHANGES MADE:
#   - Modified load_and_split_documents() to accept list of PDF files
#   - Added get_pdf_files(), compute_file_hash(), pdfs_have_changed() functions
#   - Updated create_hybrid_retriever() to accept force_reindex parameter
#   - Added Chroma collection clearing when PDFs change
#   - Added debug logging in hybrid_retriever()
#   - Disabled compression step (bypassed)
#   - Updated main() to detect PDF changes and force re-indexing
#
# ====================================

load_dotenv()

PDF_DIRECTORY = "./docs"  # Changed to directory to load all PDFs
PDF_FILE_PATH = None  # Will be set to load all PDFs from directory
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "insurance_rag_collection"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
AUDIT_LOG_FILE = "./audit.log"
PDF_HASH_FILE = "./.pdf_hashes.json"  # Track PDF file hashes to detect changes

# Ensure OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# -------------------------------------------------------------------
# 1. Document Handling & Indexing
# -------------------------------------------------------------------

def get_pdf_files(directory: str) -> List[str]:
    """Get all PDF files from directory, excluding example.pdf"""
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    # Exclude example.pdf (dummy file)
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
            hashes[pdf_file] = compute_file_hash(pdf_file)
    return hashes


def load_saved_hashes() -> dict:
    """Load previously saved PDF hashes"""
    if os.path.exists(PDF_HASH_FILE):
        with open(PDF_HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
    
    # Check if number of files changed
    if set(current_hashes.keys()) != set(saved_hashes.keys()):
        print("PDF file list has changed (files added/removed).")
        return True
    
    # Check if any file hash changed
    for pdf_file, current_hash in current_hashes.items():
        if pdf_file not in saved_hashes or saved_hashes[pdf_file] != current_hash:
            print(f"PDF file changed: {os.path.basename(pdf_file)}")
            return True
    
    return False


def load_and_split_documents(pdf_files: List[str]) -> List[Document]:
    """Load and split multiple PDF documents"""
    all_splits = []
    
    for file_path in pdf_files:
        print(f"Loading document from: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}.")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(docs)
            print(f"Split into {len(splits)} chunks from {os.path.basename(file_path)}.")
            all_splits.extend(splits)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Total: {len(all_splits)} chunks from {len(pdf_files)} PDF(s).")
    return all_splits


# -------------------------------------------------------------------
# 2. Embeddings
# -------------------------------------------------------------------

def initialize_embeddings():
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings


# -------------------------------------------------------------------
# 3. Vector Store & Hybrid Search
# -------------------------------------------------------------------

def create_hybrid_retriever(
    all_splits: List[Document],
    embeddings,
    force_reindex: bool = False,
):
    print("Creating Chroma vector store and indexing documents...")

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

    # Create/get collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Create Chroma vector store
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )

    # Check if we need to re-index (force or empty collection or PDFs changed)
    needs_reindex = force_reindex or collection.count() == 0
    
    if needs_reindex:
        if collection.count() > 0:
            print(f"Clearing existing Chroma collection ({collection.count()} documents)...")
            # Delete and recreate collection to clear old data
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)
            vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
            )
        
        print("Adding documents to Chroma...")
        vector_store.add_documents(documents=all_splits)
        print(f"Indexed {len(all_splits)} documents in Chroma.")
        
        # Save current PDF hashes after successful indexing
        pdf_files = get_pdf_files(PDF_DIRECTORY)
        if pdf_files:
            save_pdf_hashes(get_pdf_hashes(pdf_files))
    else:
        print(
            f"Chroma already contains {collection.count()} documents. "
            "Skipping re-indexing."
        )

    # Initialize BM25 retriever
    print("Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5

    # Initialize Vector Store retriever
    vector_store_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Simple hybrid retriever (BM25 + Vector)
    # ---- our simple hybrid retriever (no EnsembleRetriever) ----
    def hybrid_retriever(query: str, k: int = 5) -> List[Document]:
        # BM25 (LLM-style retriever → invoke)
        bm25_docs = bm25_retriever.invoke(query)          
        # Vector search (Chroma retriever → invoke)
        vs_docs = vector_store_retriever.invoke(query)    

        # Combine results, remove duplicates, keep max k
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

        print(f"Hybrid retriever returned {len(combined)} documents.")
        
        # DEBUG: Print top retrieved documents with metadata
        print("\n" + "="*80)
        print("DEBUG: Top Retrieved Documents:")
        print("="*80)
        for i, doc in enumerate(combined[:3], 1):  # Show top 3
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content_preview = doc.page_content[:600]  # First 600 chars
            print(f"\n--- Document {i} ---")
            print(f"Source: {os.path.basename(source) if source != 'Unknown' else source}")
            print(f"Page: {page}")
            print(f"Content preview (first 600 chars):\n{content_preview}...")
        print("="*80 + "\n")
        
        return combined


    return hybrid_retriever


# -------------------------------------------------------------------
# 4. Re-Ranking (LLM-basiert mit GPT)
# -------------------------------------------------------------------

def initialize_reranker():
    print("Initializing LLM-based reranker (GPT)...")
    reranker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    return reranker_llm


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker_llm,
    top_k: int = 3,
) -> List[Document]:
    print(f"Re-ranking {len(documents)} documents for query: '{query}' (LLM)...")
    if not documents:
        return []

    # Dokumente nummerieren
    doc_blocks = []
    for i, doc in enumerate(documents):
        content = doc.page_content[:1500]  # etwas kürzen
        doc_blocks.append(f"DOC {i}:\n{content}\n")

    docs_text = "\n\n".join(doc_blocks)

    system_msg = (
        "Du bist ein Assistent, der Dokumente nach Relevanz sortiert.\n"
        "Du bekommst eine Nutzerfrage und mehrere Dokumente (DOC 0, DOC 1, ...).\n"
        f"Antworte NUR mit den Indizes der {top_k} relevantesten Dokumente, "
        "getrennt durch Kommas, z.B.: 1,0,3"
    )

    user_msg = f"Frage:\n{query}\n\nDokumente:\n{docs_text}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", user_msg),
        ]
    )

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

    print(f"Returned top {len(reranked_docs)} re-ranked documents (LLM-based).")
    return reranked_docs


# -------------------------------------------------------------------
# 5. Context Compression (LLM-basiert mit GPT)
# -------------------------------------------------------------------

def initialize_compressor():
    print("Initializing LLM-based compressor (GPT)...")
    compressor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    return compressor_llm


def compress_context(
    compressor_llm,
    documents: List[Document],
    instruction: str,
) -> List[Document]:
    print(f"Compressing context for {len(documents)} documents (LLM)...")
    if not documents:
        return []

    merged_text = "\n\n---\n\n".join(doc.page_content for doc in documents)

    system_msg = (
        "Du fasst den folgenden Kontext so zusammen, dass er für die gegebene "
        "Frage maximal relevant ist. Kürze auf ungefähr 300 Tokens, behalte "
        "wichtige Zahlen, Bedingungen und Ausnahmen bei."
    )

    user_msg = (
        f"Frage/Nutzerinstruktion:\n{instruction}\n\n"
        f"Kontext:\n{merged_text}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", user_msg),
        ]
    )

    chain = prompt | compressor_llm
    response = chain.invoke({})

    compressed_content = response.content
    compressed_doc = Document(
        page_content=compressed_content,
        metadata={"source": "LLM-Compressed"},
    )

    print("Context compression (LLM) complete.")
    return [compressed_doc]


# -------------------------------------------------------------------
# 6. Answer Generator (LLM)
# -------------------------------------------------------------------

def initialize_llm():
    print("Initializing ChatOpenAI LLM...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    return llm


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
) -> Tuple[str, dict]:
    """Generate answer and return both answer and token usage."""
    print("Generating answer with LLM...")
    if chat_history is None:
        chat_history = []

    context = "\n---\n".join([doc.page_content for doc in context_docs])

    formatted_chat_history = ""
    for turn in chat_history:
        formatted_chat_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT.format(chat_history=formatted_chat_history)
                + "\nCONTEXT: {context}\nTASK: Erzeuge eine präzise, "
                "kurze Antwort + Quellenblock.",
            ),
            ("user", "{query}"),
        ]
    )

    chain = prompt_template | llm
    response = chain.invoke({"query": query, "context": context})

    generated_answer = response.content
    
    # Extract token usage from response
    token_usage = {}
    if hasattr(response, 'response_metadata') and response.response_metadata:
        usage = response.response_metadata.get('token_usage', {})
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    
    print("Answer generation complete.")
    return generated_answer, token_usage


# -------------------------------------------------------------------
# 8. Agentic Behavior: Lightweight Planner / Router
# -------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """Du bist ein intelligenter Router. Deine Aufgabe ist es zu entscheiden, 
ob eine Nutzerfrage eine Informationsabfrage aus einer Wissensbasis benötigt (RETRIEVE) 
oder ob die Frage direkt beantwortet werden kann (NO_RETRIEVE). 
Antworte ausschließlich mit 'RETRIEVE' oder 'NO_RETRIEVE'.
"""


def initialize_router_llm():
    print("Initializing Router LLM...")
    router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    return router_llm


def decide_retrieval(
    router_llm,
    query: str,
    chat_history: Optional[List[dict]] = None,
) -> str:
    print(f"Routing decision for query: '{query}'...")
    if chat_history is None:
        chat_history = []

    formatted_chat_history = ""
    for turn in chat_history:
        formatted_chat_history += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "user",
                f"User Query: {query}\nChat History: {formatted_chat_history}",
            ),
        ]
    )

    chain = prompt_template | router_llm
    response = chain.invoke({})
    decision = response.content.strip().upper()
    print(f"Routing decision: {decision}")
    return decision


# -------------------------------------------------------------------
# 9. Self-Check Loop & Query Rewrite
# -------------------------------------------------------------------

SELF_CHECK_SYSTEM_PROMPT = """Du bist ein Assistent, der die Relevanz von bereitgestellten Kontextdokumenten für eine Benutzerfrage bewertet.
Antworte ausschließlich mit 'RELEVANT' oder 'IRRELEVANT'.
"""

QUERY_REWRITE_SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent, der Benutzeranfragen umschreibt, um bessere Suchergebnisse zu erzielen.
Ziel ist es, die ursprüngliche Frage beizubehalten, aber die Formulierung zu optimieren, wenn die vorherige Suche nicht erfolgreich war.
Antworte nur mit der umgeschriebenen Frage.
"""


def initialize_self_check_llm():
    print("Initializing Self-Check LLM...")
    self_check_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    return self_check_llm


def initialize_query_rewrite_llm():
    print("Initializing Query Rewrite LLM...")
    query_rewrite_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    return query_rewrite_llm


def rewrite_query(
    query_rewrite_llm,
    query: str,
    chat_history: Optional[List[dict]] = None,
) -> str:
    print(f"Rewriting query: '{query}'...")
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", user_msg),
        ]
    )

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
    if chat_history is None:
        chat_history = []

    current_query = original_query
    current_retrieved_docs = retrieved_docs

    for attempt in range(max_retries):
        print(
            f"Self-check attempt {attempt + 1}/{max_retries} "
            f"for query: '{current_query}'"
        )

        if not current_retrieved_docs:
            print("No documents retrieved for self-check. Rewriting query.")
            current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
            return current_query, []

        context_for_self_check = "\n---\n".join(
            [doc.page_content for doc in current_retrieved_docs]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SELF_CHECK_SYSTEM_PROMPT),
                (
                    "user",
                    f"User Query: {current_query}\nContext:\n{context_for_self_check}",
                ),
            ]
        )

        chain = prompt | self_check_llm
        relevance_decision = chain.invoke({}).content.strip().upper()

        print(f"Relevance decision: {relevance_decision}")

        if relevance_decision == "RELEVANT":
            print("Context deemed relevant.")
            return current_query, current_retrieved_docs
        else:
            print("Context deemed IRRELEVANT. Rewriting query...")
            current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
            print(f"Rewritten query: '{current_query}'")
            return current_query, []

    print(
        f"Max retries reached. Using last query: '{current_query}' "
        "and available documents."
    )
    return current_query, current_retrieved_docs


# -------------------------------------------------------------------
# 7. Safety & Audit
# -------------------------------------------------------------------

def audit_log(
    query: str,
    retrieved_docs: List[Document],
    compressed_context: List[Document],
    answer: str,
    chat_history: Optional[List[dict]] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    token_usage: Optional[dict] = None,
    query_rewritten: bool = False,
    self_check_passed: bool = False,
    retrieval_retries: int = 0,
):
    """Logs the RAG process details to an audit file."""
    if chat_history is None:
        chat_history = []

    print(f"Logging audit details to {AUDIT_LOG_FILE}...")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "start_timestamp": start_timestamp or datetime.now().isoformat(),
        "end_timestamp": end_timestamp or datetime.now().isoformat(),
        "query": query,
        "retrieved_documents": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in retrieved_docs
        ],
        "compressed_context": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in compressed_context
        ],
        "generated_answer": answer,
        "chat_history": chat_history,
        "query_rewritten": query_rewritten,
        "self_check_passed": self_check_passed,
        "retrieval_retries": retrieval_retries,
    }
    
    # Add token usage if available
    if token_usage:
        log_entry["token_usage"] = token_usage
    
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print("Audit logging complete.")


def perform_safety_checks(
    query: str,
    context_docs: List[Document],
    answer: str,
    chat_history: Optional[List[dict]] = None,
) -> bool:
    """Placeholder for safety and compliance checks."""
    print("Performing safety checks (placeholder)...")
    # TODO: PII detection, prompt injection, policy checks, etc.
    print("Safety checks passed (placeholder).")
    return True


# -------------------------------------------------------------------
#  Main Execution (CLI Loop)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure docs directory exists
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    # Get all PDF files (excluding example.pdf)
    pdf_files = get_pdf_files(PDF_DIRECTORY)
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {PDF_DIRECTORY} (excluding example.pdf)")
        print("Please add your Baloise PDF files to the ./docs directory.")
        exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {os.path.basename(pdf_file)}")
    
    # Check if PDFs have changed (need to re-index)
    force_reindex = pdfs_have_changed()
    if force_reindex:
        print("\n*** PDF files have changed - will re-index Chroma database ***\n")
    else:
        print("\n*** Using existing Chroma index (PDFs unchanged) ***\n")

    # Initialize pipeline
    all_splits = load_and_split_documents(pdf_files)
    embeddings = initialize_embeddings()

    hybrid_retriever = create_hybrid_retriever(all_splits, embeddings, force_reindex=force_reindex)
    reranker_model = initialize_reranker()
    compressor_llm = initialize_compressor()
    llm = initialize_llm()
    router_llm = initialize_router_llm()
    self_check_llm = initialize_self_check_llm()
    query_rewrite_llm = initialize_query_rewrite_llm()

    print("\n--- RAG Pipeline Initialized. Type 'exit' to quit. ---")
    chat_history: List[dict] = []

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == "exit":
            break

        # Router: decide if retrieval is needed
        retrieval_needed = decide_retrieval(router_llm, user_query, chat_history)

        if retrieval_needed == "RETRIEVE":
            print(f"Processing query: '{user_query}' with retrieval...")
            
            # Start timestamp
            start_timestamp = datetime.now().isoformat()

            current_query = user_query
            retrieved_docs: List[Document] = []
            final_compressed_docs: List[Document] = []
            answer = ""
            retries = 0
            MAX_SELF_CHECK_RETRIES = 2
            reranked_docs: List[Document] = []
            query_rewritten = False
            self_check_passed = False
            total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            while retries < MAX_SELF_CHECK_RETRIES:
                print(
                    f"Retrieving documents for query: '{current_query}' "
                    f"(Attempt {retries + 1})"
                )
                retrieved_docs = hybrid_retriever(current_query, k=8)
                print(
                    f"Retrieved {len(retrieved_docs)} documents before re-ranking."
                )

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
                        query_rewritten = True
                        retries += 1
                        print(
                            "Self-check failed or query rewritten. "
                            f"Retrying retrieval with new query: '{current_query}'"
                        )
                        continue
                    else:
                        reranked_docs = checked_docs
                        self_check_passed = True
                        break
                else:
                    print(
                        "No documents retrieved, skipping self-check and rewriting query."
                    )
                    current_query = rewrite_query(
                        query_rewrite_llm, current_query, chat_history
                    )
                    query_rewritten = True
                    retries += 1
                    continue

            if not reranked_docs:
                print(
                    "Could not find relevant documents after multiple attempts. "
                    "Providing generic response."
                )
                answer = (
                    "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer "
                    "Anfrage finden. Bitte versuchen Sie eine andere Formulierung oder "
                    "eine allgemeinere Frage."
                )
                end_timestamp = datetime.now().isoformat()
                audit_log(
                    user_query, retrieved_docs, [], answer, chat_history,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    query_rewritten=query_rewritten,
                    self_check_passed=False,
                    retrieval_retries=retries
                )
                chat_history.append({"query": user_query, "answer": answer})
            else:
                # TEMPORARILY DISABLE COMPRESSION FOR DEBUGGING
                # Use reranked_docs directly instead of compressed version
                print("DEBUG: Bypassing compression, using reranked_docs directly.")
                final_compressed_docs = reranked_docs
                
                # Uncomment below to re-enable compression:
                # final_compressed_docs = compress_context(
                #     compressor_llm,
                #     reranked_docs,
                #     current_query,
                # )

                answer, token_usage = generate_answer(
                    llm,
                    current_query,
                    final_compressed_docs,
                    chat_history,
                )
                
                # Accumulate token usage
                if token_usage:
                    total_token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                    total_token_usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
                    total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

                if perform_safety_checks(
                    current_query,
                    final_compressed_docs,
                    answer,
                    chat_history,
                ):
                    print(f"\nAssistant: {answer}")
                    end_timestamp = datetime.now().isoformat()
                    audit_log(
                        current_query,
                        retrieved_docs,
                        final_compressed_docs,
                        answer,
                        chat_history,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                        query_rewritten=query_rewritten,
                        self_check_passed=self_check_passed,
                        retrieval_retries=retries
                    )
                    chat_history.append(
                        {"query": user_query, "answer": answer}
                    )
                else:
                    print(
                        "\nAssistant: Answer generation halted due to safety concerns."
                    )
        else:
            print(f"Processing query: '{user_query}' ohne Retrieval...")
            start_timestamp = datetime.now().isoformat()
            answer = (
                "Ich kann diese Frage direkt beantworten oder benötige "
                "keine Dokumentensuche dafür."
            )
            print(f"\nAssistant: {answer}")
            end_timestamp = datetime.now().isoformat()
            audit_log(
                user_query, [], [], answer, chat_history,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                query_rewritten=False,
                self_check_passed=False,
                retrieval_retries=0
            )
            chat_history.append({"query": user_query, "answer": answer})

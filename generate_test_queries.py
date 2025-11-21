"""
Skript zum automatischen Generieren von Test-Queries für die Evaluation
Führt eine bestimmte Anzahl von Queries durch das RAG-System aus und generiert Audit-Logs
"""

import os
import sys
from typing import List
from datetime import datetime
from dotenv import load_dotenv

# Importiere alle notwendigen Funktionen aus main.py
from main import (
    get_pdf_files,
    load_and_split_documents,
    initialize_embeddings,
    create_hybrid_retriever,
    initialize_reranker,
    initialize_compressor,
    initialize_llm,
    initialize_router_llm,
    initialize_self_check_llm,
    initialize_query_rewrite_llm,
    decide_retrieval,
    rerank_documents,
    perform_self_check,
    rewrite_query,
    generate_answer,
    perform_safety_checks,
    audit_log,
    PDF_DIRECTORY,
    AUDIT_LOG_FILE,
)

# Liste von Test-Queries (50 verschiedene Fragen zu Versicherungen)
TEST_QUERIES = [
    # Kfz-Versicherung Fragen
    "Wie hoch sind die Personenschäden abgesichert?",
    "Was ist die Deckungssumme für Kaskoschäden?",
    "Welche Selbstbeteiligung gilt bei Kfz-Schäden?",
    "Gibt es eine Deckung für Tierkollisionen?",
    "Was ist im All-in-Tarif enthalten?",
    "Wie lange gilt die Neupreisdeckung?",
    "Welche Leistungen umfasst die Elementardeckung?",
    "Gibt es eine Deckung für Marderbisse?",
    "Was kostet die Kfz-Versicherung?",
    "Welche Rabatte gibt es?",
    "Gibt es eine Deckung für Überspannungsschäden?",
    "Was ist bei Hagelschäden abgedeckt?",
    "Gibt es eine Deckung für Sturmschäden?",
    "Was ist bei Vandalismus am Fahrzeug abgedeckt?",
    "Gibt es eine Deckung für Reifenpannen?",
    
    # Sachversicherung Fragen
    "Was ist im Highlights Ambiente Top Tarif enthalten?",
    "Wie hoch ist die Deckungssumme für Hausrat?",
    "Welche Selbstbeteiligung gilt bei Sachschäden?",
    "Gibt es eine Deckung für Elementarschäden?",
    "Was ist bei Diebstahl abgedeckt?",
    "Welche Leistungen umfasst die Glasversicherung?",
    "Gibt es eine Deckung für Vandalismus?",
    "Was ist bei Wasserschäden abgedeckt?",
    "Welche Versicherungssumme wird empfohlen?",
    "Gibt es eine Deckung für Überspannungsschäden?",
    "Was ist bei Frostschäden abgedeckt?",
    "Gibt es eine Deckung für Brandschäden?",
    "Was ist bei Einbruchdiebstahl abgedeckt?",
    "Gibt es eine Deckung für Blitzschlag?",
    "Was ist bei Leitungswasserschäden abgedeckt?",
    
    # Allgemeine Fragen
    "Wie kann ich einen Schaden melden?",
    "Welche Unterlagen benötige ich für eine Schadensmeldung?",
    "Wie lange dauert die Bearbeitung eines Schadens?",
    "Gibt es eine 24/7 Schadenshotline?",
    "Welche Zahlungsmethoden werden akzeptiert?",
    "Kann ich meine Versicherung online abschließen?",
    "Gibt es eine App für die Versicherung?",
    "Wie kann ich meine Versicherung kündigen?",
    "Welche Kündigungsfristen gelten?",
    "Gibt es eine Widerspruchsregelung?",
    
    # Spezifische Leistungen
    "Was ist eine erweiterte Eigenschadendeckung?",
    "Gibt es eine Deckung für Mietwagen?",
    "Was ist bei Unfallflucht abgedeckt?",
    "Gibt es eine Deckung für Schlüsselverlust?",
    "Was ist bei E-Autos anders?",
    "Gibt es spezielle Tarife für Fahranfänger?",
    "Was ist bei Vielfahrern anders?",
    "Gibt es eine Deckung für Auslandsfahrten?",
    "Was ist bei Leasingfahrzeugen zu beachten?",
    "Gibt es eine Deckung für Oldtimer?",
]


def run_test_queries(num_queries: int = 50):
    """Führt eine bestimmte Anzahl von Test-Queries durch das System aus."""
    
    print(f"\n{'='*80}")
    print(f"=== Generiere {num_queries} Test-Queries für Evaluation ===")
    print(f"{'='*80}\n")
    
    # Initialisiere Pipeline (wie in main.py)
    pdf_files = get_pdf_files(PDF_DIRECTORY)
    if not pdf_files:
        print("ERROR: Keine PDF-Dateien gefunden!")
        print(f"Bitte fügen Sie PDF-Dateien in das Verzeichnis {PDF_DIRECTORY} ein.")
        return
    
    print("Initialisiere RAG-Pipeline...")
    all_splits = load_and_split_documents(pdf_files)
    embeddings = initialize_embeddings()
    hybrid_retriever = create_hybrid_retriever(all_splits, embeddings, force_reindex=False)
    reranker_model = initialize_reranker()
    compressor_llm = initialize_compressor()
    llm = initialize_llm()
    router_llm = initialize_router_llm()
    self_check_llm = initialize_self_check_llm()
    query_rewrite_llm = initialize_query_rewrite_llm()
    
    print(f"\n{'='*80}")
    print(f"=== Starte Verarbeitung von {num_queries} Queries ===")
    print(f"{'='*80}\n")
    
    chat_history: List[dict] = []
    queries_to_process = TEST_QUERIES[:num_queries]
    
    for i, user_query in enumerate(queries_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{num_queries}: {user_query}")
        print(f"{'='*80}\n")
        
        try:
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
                    print(f"Retrieving documents for query: '{current_query}' (Attempt {retries + 1})")
                    retrieved_docs = hybrid_retriever(current_query, k=8)
                    print(f"Retrieved {len(retrieved_docs)} documents before re-ranking.")
                    
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
                            print("Self-check failed or query rewritten. Retrying retrieval...")
                            continue
                        else:
                            reranked_docs = checked_docs
                            self_check_passed = True
                            break
                    else:
                        print("No documents retrieved, rewriting query...")
                        current_query = rewrite_query(query_rewrite_llm, current_query, chat_history)
                        query_rewritten = True
                        retries += 1
                        continue
                
                if not reranked_docs:
                    print("Could not find relevant documents after multiple attempts.")
                    answer = (
                        "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer "
                        "Anfrage finden. Bitte versuchen Sie eine andere Formulierung."
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
                    print("DEBUG: Bypassing compression, using reranked_docs directly.")
                    final_compressed_docs = reranked_docs
                    
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
                    
                    if perform_safety_checks(current_query, final_compressed_docs, answer, chat_history):
                        end_timestamp = datetime.now().isoformat()
                        audit_log(
                            current_query, retrieved_docs, final_compressed_docs, answer, chat_history,
                            start_timestamp=start_timestamp,
                            end_timestamp=end_timestamp,
                            token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
                            query_rewritten=query_rewritten,
                            self_check_passed=self_check_passed,
                            retrieval_retries=retries
                        )
                        chat_history.append({"query": user_query, "answer": answer})
                    else:
                        print("Answer generation halted due to safety concerns.")
            else:
                print(f"Processing query: '{user_query}' ohne Retrieval...")
                start_timestamp = datetime.now().isoformat()
                answer = "Ich kann diese Frage direkt beantworten oder benötige keine Dokumentensuche dafür."
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
        
        except Exception as e:
            print(f"ERROR bei Query {i}: {e}")
            print("Überspringe diese Query und fahre fort...\n")
            continue
    
    print(f"\n{'='*80}")
    print(f"✓ Erfolgreich {num_queries} Queries verarbeitet!")
    print(f"✓ Audit-Logs wurden in {AUDIT_LOG_FILE} gespeichert")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    load_dotenv()
    
    # Anzahl der Queries (Standard: 50)
    num_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    if num_queries > len(TEST_QUERIES):
        print(f"WARNUNG: Nur {len(TEST_QUERIES)} Queries verfügbar, verwende diese.")
        num_queries = len(TEST_QUERIES)
    
    run_test_queries(num_queries)


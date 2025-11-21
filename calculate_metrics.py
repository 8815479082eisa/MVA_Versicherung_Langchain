"""
Optimized Evaluation Metrics Calculator
Calculates three key metrics for RAG system evaluation
"""

import json
import os
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def load_audit_logs(file_path: str = "./audit.log") -> List[Dict]:
    """Load audit logs from JSONL file."""
    logs = []
    if not os.path.exists(file_path):
        return logs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    return logs


def calculate_metrics_for_entry(log_entry: Dict, evaluator_llm: ChatOpenAI) -> Dict:
    """Calculate all three metrics for a single log entry."""
    query = log_entry.get("query", "")
    retrieved_docs = log_entry.get("retrieved_documents", [])
    compressed_context = log_entry.get("compressed_context", [])
    answer = log_entry.get("generated_answer", "")
    
    # Skip entries without proper data
    if not query or not answer:
        return None
    
    # Use compressed_context if available, otherwise retrieved_docs
    context_docs = compressed_context if compressed_context else retrieved_docs
    
    if not context_docs:
        # No context available - return default values
        return {
            "query": query,
            "kontextrelevanz": {"score": 0.0, "relevant": 0, "total": 0},
            "kontextgenügsamkeit": {"score": 0.0, "covered": 0, "total_required": 0},
            "antwort_halluzination": {"rate": 1.0, "score": 0.0, "hallucinated": 0, "total": 0}
        }
    
    # Extract context text
    context_text = "\n".join([doc.get("page_content", "") for doc in context_docs])
    
    # 1. Kontextrelevanz
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest die Relevanz von Kontextaussagen für eine Nutzerfrage.
Antworte mit einer Zahl zwischen 0.0 und 1.0, die den Anteil relevanter Aussagen im Kontext angibt.
Antworte NUR mit der Zahl, z.B. 0.75"""),
        ("user", f"Frage: {query}\n\nKontext:\n{context_text[:2000]}\n\nWie hoch ist der Anteil relevanter Aussagen im Kontext? (0.0-1.0)")
    ])
    try:
        relevance_response = (relevance_prompt | evaluator_llm).invoke({})
        relevance_score = float(relevance_response.content.strip().replace(",", "."))
        relevance_score = max(0.0, min(1.0, relevance_score))
    except:
        relevance_score = 0.5  # Default fallback
    
    # 2. Kontextgenügsamkeit
    sufficiency_prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest, ob der Kontext ausreicht, um die Frage zu beantworten.
Antworte mit einer Zahl zwischen 0.0 und 1.0, die angibt, welcher Anteil der benötigten Informationen im Kontext vorhanden ist.
Antworte NUR mit der Zahl, z.B. 0.80"""),
        ("user", f"Frage: {query}\n\nAntwort: {answer[:500]}\n\nKontext:\n{context_text[:2000]}\n\nWie viel Prozent der benötigten Informationen sind im Kontext vorhanden? (0.0-1.0)")
    ])
    try:
        sufficiency_response = (sufficiency_prompt | evaluator_llm).invoke({})
        sufficiency_score = float(sufficiency_response.content.strip().replace(",", "."))
        sufficiency_score = max(0.0, min(1.0, sufficiency_score))
    except:
        sufficiency_score = 0.5  # Default fallback
    
    # 3. Antwort-Halluzination
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest, ob die Antwort durch den Kontext gestützt wird.
Antworte mit einer Zahl zwischen 0.0 und 1.0, wobei 0.0 = keine Halluzinationen, 1.0 = alle Behauptungen sind Halluzinationen.
Antworte NUR mit der Zahl, z.B. 0.20"""),
        ("user", f"Antwort: {answer}\n\nKontext:\n{context_text[:2000]}\n\nWie hoch ist der Halluzinationsanteil? (0.0-1.0, 0.0=keine, 1.0=alle)")
    ])
    try:
        hallucination_response = (hallucination_prompt | evaluator_llm).invoke({})
        hallucination_rate = float(hallucination_response.content.strip().replace(",", "."))
        hallucination_rate = max(0.0, min(1.0, hallucination_rate))
    except:
        hallucination_rate = 0.5  # Default fallback
    
    return {
        "query": query[:100],  # Truncate for display
        "kontextrelevanz": {
            "score": relevance_score,
            "relevant": int(relevance_score * 10),  # Approximate
            "total": 10
        },
        "kontextgenügsamkeit": {
            "score": sufficiency_score,
            "covered": int(sufficiency_score * 10),
            "total_required": 10
        },
        "antwort_halluzination": {
            "rate": hallucination_rate,
            "score": 1.0 - hallucination_rate,
            "hallucinated": int(hallucination_rate * 10),
            "total": 10
        }
    }


def main():
    """Main evaluation function."""
    print("Loading audit logs...")
    logs = load_audit_logs()
    
    # Filter valid entries (with retrieval)
    valid_logs = [log for log in logs 
                  if log.get("retrieved_documents") or log.get("compressed_context")]
    
    print(f"Found {len(valid_logs)} valid log entries for evaluation")
    
    if not valid_logs:
        print("No valid log entries found.")
        return
    
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    results = []
    
    for i, log_entry in enumerate(valid_logs):
        print(f"Evaluating entry {i+1}/{len(valid_logs)}...")
        result = calculate_metrics_for_entry(log_entry, evaluator_llm)
        if result:
            results.append(result)
    
    # Calculate summary statistics
    if results:
        kr_scores = [r["kontextrelevanz"]["score"] for r in results]
        kg_scores = [r["kontextgenügsamkeit"]["score"] for r in results]
        ah_rates = [r["antwort_halluzination"]["rate"] for r in results]
        ah_scores = [r["antwort_halluzination"]["score"] for r in results]
        
        summary = {
            "total_entries": len(results),
            "kontextrelevanz": {
                "mean": sum(kr_scores) / len(kr_scores),
                "min": min(kr_scores),
                "max": max(kr_scores)
            },
            "kontextgenügsamkeit": {
                "mean": sum(kg_scores) / len(kg_scores),
                "min": min(kg_scores),
                "max": max(kg_scores)
            },
            "antwort_halluzination": {
                "rate_mean": sum(ah_rates) / len(ah_rates),
                "rate_min": min(ah_rates),
                "rate_max": max(ah_rates),
                "score_mean": sum(ah_scores) / len(ah_scores),
                "score_min": min(ah_scores),
                "score_max": max(ah_scores)
            }
        }
        
        # Save results
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump({"results": results, "summary": summary}, f, indent=2, ensure_ascii=False)
        
        print("\n=== Evaluation Summary ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to evaluation_results.json")
    else:
        print("No valid results generated.")


if __name__ == "__main__":
    main()


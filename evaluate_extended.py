"""
Extended Evaluation Metrics Calculator for Agentic RAG System

Calculates comprehensive evaluation metrics including:
1. Kontextrelevanz (Context Relevance)
2. Kontextgenügsamkeit (Context Sufficiency)
3. Antwort-Halluzination (Answer Hallucination)
4. Blocking/Reduction Rate
5. Query-Rewrite Rate
6. Self-Check Success Rate
7. Latency Metrics
8. Cost per Token Metrics
"""

import json
import os
import statistics
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# OpenAI Pricing (per 1M tokens) - Default values, can be overridden via env
OPENAI_MODEL_PRICE_IN = float(os.getenv("OPENAI_MODEL_PRICE_IN", "2.50"))  # $2.50 per 1M input tokens (gpt-4o)
OPENAI_MODEL_PRICE_OUT = float(os.getenv("OPENAI_MODEL_PRICE_OUT", "10.00"))  # $10.00 per 1M output tokens (gpt-4o)


def load_audit_logs(file_path: str = "./audit.log") -> List[Dict]:
    """Load audit logs from JSONL file."""
    logs = []
    if not os.path.exists(file_path):
        print(f"Warning: Audit log file {file_path} not found.")
        return logs
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


def calculate_latency(start_timestamp: str, end_timestamp: str) -> Optional[float]:
    """Calculate latency in seconds from ISO timestamps."""
    try:
        start = datetime.fromisoformat(start_timestamp.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_timestamp.replace('Z', '+00:00'))
        delta = end - start
        return delta.total_seconds()
    except (ValueError, AttributeError):
        return None


def calculate_cost_per_query(token_usage: Dict) -> float:
    """Calculate cost per query based on token usage."""
    if not token_usage:
        return 0.0
    
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    
    cost = (prompt_tokens / 1_000_000 * OPENAI_MODEL_PRICE_IN) + \
           (completion_tokens / 1_000_000 * OPENAI_MODEL_PRICE_OUT)
    
    return cost


def calculate_context_relevance(query: str, context_docs: List[Dict], evaluator_llm: ChatOpenAI) -> float:
    """Calculate Kontextrelevanz metric."""
    if not context_docs:
        return 0.0
    
    context_text = "\n".join([doc.get("page_content", "") for doc in context_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest die Relevanz von Kontextaussagen für eine Nutzerfrage.
Antworte mit einer Zahl zwischen 0.0 und 1.0, die den Anteil relevanter Aussagen im Kontext angibt.
Antworte NUR mit der Zahl, z.B. 0.75"""),
        ("user", f"Frage: {query}\n\nKontext:\n{context_text[:2000]}\n\nWie hoch ist der Anteil relevanter Aussagen im Kontext? (0.0-1.0)")
    ])
    
    try:
        response = (prompt | evaluator_llm).invoke({})
        score = float(response.content.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except:
        return 0.5  # Default fallback


def calculate_context_sufficiency(query: str, context_docs: List[Dict], answer: str, evaluator_llm: ChatOpenAI) -> float:
    """Calculate Kontextgenügsamkeit metric."""
    if not context_docs:
        return 0.0
    
    context_text = "\n".join([doc.get("page_content", "") for doc in context_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest, ob der Kontext ausreicht, um die Frage zu beantworten.
Antworte mit einer Zahl zwischen 0.0 und 1.0, die angibt, welcher Anteil der benötigten Informationen im Kontext vorhanden ist.
Antworte NUR mit der Zahl, z.B. 0.80"""),
        ("user", f"Frage: {query}\n\nAntwort: {answer[:500]}\n\nKontext:\n{context_text[:2000]}\n\nWie viel Prozent der benötigten Informationen sind im Kontext vorhanden? (0.0-1.0)")
    ])
    
    try:
        response = (prompt | evaluator_llm).invoke({})
        score = float(response.content.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except:
        return 0.5  # Default fallback


def calculate_answer_hallucination(answer: str, context_docs: List[Dict], evaluator_llm: ChatOpenAI) -> float:
    """Calculate Antwort-Halluzination metric."""
    if not context_docs:
        return 1.0  # All hallucinated if no context
    
    context_text = "\n".join([doc.get("page_content", "") for doc in context_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bewertest, ob die Antwort durch den Kontext gestützt wird.
Antworte mit einer Zahl zwischen 0.0 und 1.0, wobei 0.0 = keine Halluzinationen, 1.0 = alle Behauptungen sind Halluzinationen.
Antworte NUR mit der Zahl, z.B. 0.20"""),
        ("user", f"Antwort: {answer}\n\nKontext:\n{context_text[:2000]}\n\nWie hoch ist der Halluzinationsanteil? (0.0-1.0, 0.0=keine, 1.0=alle)")
    ])
    
    try:
        response = (prompt | evaluator_llm).invoke({})
        rate = float(response.content.strip().replace(",", "."))
        return max(0.0, min(1.0, rate))
    except:
        return 0.5  # Default fallback


def calculate_reduction_rate(retrieved_count: int, reranked_count: int) -> float:
    """Calculate document reduction rate from retrieval to re-ranking."""
    if retrieved_count == 0:
        return 0.0
    return (retrieved_count - reranked_count) / retrieved_count


def calculate_metrics_for_entry(log_entry: Dict, evaluator_llm: ChatOpenAI) -> Optional[Dict]:
    """Calculate all metrics for a single log entry."""
    query = log_entry.get("query", "")
    retrieved_docs = log_entry.get("retrieved_documents", [])
    compressed_context = log_entry.get("compressed_context", [])
    answer = log_entry.get("generated_answer", "")
    
    # Skip entries without proper data
    if not query or not answer:
        return None
    
    # Use compressed_context if available, otherwise retrieved_docs
    context_docs = compressed_context if compressed_context else retrieved_docs
    
    # Basic metrics
    retrieved_count = len(retrieved_docs)
    reranked_count = len(context_docs)
    
    # Calculate quality metrics
    context_relevance = calculate_context_relevance(query, context_docs, evaluator_llm) if context_docs else 0.0
    context_sufficiency = calculate_context_sufficiency(query, context_docs, answer, evaluator_llm) if context_docs else 0.0
    hallucination_rate = calculate_answer_hallucination(answer, context_docs, evaluator_llm) if context_docs else 1.0
    
    # Calculate reduction rate
    reduction_rate = calculate_reduction_rate(retrieved_count, reranked_count)
    
    # Extract additional metadata
    query_rewritten = log_entry.get("query_rewritten", False)
    self_check_passed = log_entry.get("self_check_passed", False)
    retrieval_retries = log_entry.get("retrieval_retries", 0)
    
    # Calculate latency
    start_ts = log_entry.get("start_timestamp") or log_entry.get("timestamp")
    end_ts = log_entry.get("end_timestamp") or log_entry.get("timestamp")
    latency = calculate_latency(start_ts, end_ts) if start_ts and end_ts else None
    
    # Calculate cost
    token_usage = log_entry.get("token_usage", {})
    cost = calculate_cost_per_query(token_usage)
    
    return {
        "query": query[:100],  # Truncate for display
        "kontextrelevanz": {
            "score": context_relevance
        },
        "kontextgenügsamkeit": {
            "score": context_sufficiency
        },
        "antwort_halluzination": {
            "rate": hallucination_rate,
            "score": 1.0 - hallucination_rate
        },
        "reduction_rate": reduction_rate,
        "retrieved_count": retrieved_count,
        "reranked_count": reranked_count,
        "query_rewritten": query_rewritten,
        "self_check_passed": self_check_passed,
        "retrieval_retries": retrieval_retries,
        "latency_seconds": latency,
        "token_usage": token_usage,
        "cost_usd": cost
    }


def calculate_statistics(values: List[float]) -> Dict:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "std": 0.0
        }
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted_values[int(n * 0.95)] if n > 0 else 0.0,
        "std": statistics.stdev(values) if n > 1 else 0.0
    }


def main():
    """Main evaluation function."""
    print("="*80)
    print("Extended Evaluation Metrics Calculator")
    print("="*80)
    print("\nLoading audit logs...")
    
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
    
    print("\nCalculating metrics for each entry...")
    for i, log_entry in enumerate(valid_logs, 1):
        print(f"Processing entry {i}/{len(valid_logs)}...", end="\r")
        result = calculate_metrics_for_entry(log_entry, evaluator_llm)
        if result:
            results.append(result)
    
    print(f"\n\nSuccessfully processed {len(results)} entries.")
    
    if not results:
        print("No valid results generated.")
        return
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    # Quality metrics
    kr_scores = [r["kontextrelevanz"]["score"] for r in results]
    kg_scores = [r["kontextgenügsamkeit"]["score"] for r in results]
    ah_rates = [r["antwort_halluzination"]["rate"] for r in results]
    ah_scores = [r["antwort_halluzination"]["score"] for r in results]
    
    # System metrics
    reduction_rates = [r["reduction_rate"] for r in results]
    query_rewrite_count = sum(1 for r in results if r["query_rewritten"])
    self_check_success_count = sum(1 for r in results if r["self_check_passed"])
    retrieval_retries_list = [r["retrieval_retries"] for r in results]
    
    # Latency metrics
    latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]
    
    # Cost metrics
    costs = [r["cost_usd"] for r in results]
    total_tokens = [r["token_usage"].get("total_tokens", 0) for r in results]
    prompt_tokens = [r["token_usage"].get("prompt_tokens", 0) for r in results]
    completion_tokens = [r["token_usage"].get("completion_tokens", 0) for r in results]
    
    # Build summary
    summary = {
        "metadata": {
            "total_entries": len(results),
            "evaluation_date": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o",
            "reranker_model": "gpt-4o-mini",
            "pricing": {
                "input_price_per_1m": OPENAI_MODEL_PRICE_IN,
                "output_price_per_1m": OPENAI_MODEL_PRICE_OUT
            }
        },
        "quality_metrics": {
            "kontextrelevanz": calculate_statistics(kr_scores),
            "kontextgenügsamkeit": calculate_statistics(kg_scores),
            "antwort_halluzination": {
                "rate": calculate_statistics(ah_rates),
                "score": calculate_statistics(ah_scores)
            }
        },
        "system_metrics": {
            "reduction_rate": calculate_statistics(reduction_rates),
            "query_rewrite_rate": query_rewrite_count / len(results) if results else 0.0,
            "query_rewrite_count": query_rewrite_count,
            "self_check_success_rate": self_check_success_count / len(results) if results else 0.0,
            "self_check_success_count": self_check_success_count,
            "average_retrieval_retries": statistics.mean(retrieval_retries_list) if retrieval_retries_list else 0.0
        },
        "latency_metrics": {
            "latency_seconds": calculate_statistics(latencies) if latencies else {
                "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "std": 0.0
            },
            "entries_with_latency": len(latencies)
        },
        "cost_metrics": {
            "cost_per_query_usd": calculate_statistics(costs),
            "total_cost_usd": sum(costs),
            "total_tokens": {
                "total": sum(total_tokens),
                "prompt": sum(prompt_tokens),
                "completion": sum(completion_tokens),
                "average_per_query": statistics.mean(total_tokens) if total_tokens else 0.0
            }
        }
    }
    
    # Save results
    output_file = "evaluation_results_extended.json"
    output_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal Entries: {summary['metadata']['total_entries']}")
    
    print("\n--- Quality Metrics ---")
    print(f"Kontextrelevanz: {summary['quality_metrics']['kontextrelevanz']['mean']:.3f} (mean)")
    print(f"Kontextgenügsamkeit: {summary['quality_metrics']['kontextgenügsamkeit']['mean']:.3f} (mean)")
    print(f"Antwort-Halluzination Rate: {summary['quality_metrics']['antwort_halluzination']['rate']['mean']:.3f} (mean)")
    
    print("\n--- System Metrics ---")
    print(f"Reduction Rate: {summary['system_metrics']['reduction_rate']['mean']:.3f} (mean)")
    print(f"Query-Rewrite Rate: {summary['system_metrics']['query_rewrite_rate']:.3f}")
    print(f"Self-Check Success Rate: {summary['system_metrics']['self_check_success_rate']:.3f}")
    
    if latencies:
        print("\n--- Latency Metrics ---")
        print(f"Average Latency: {summary['latency_metrics']['latency_seconds']['mean']:.2f} seconds")
        print(f"Median Latency: {summary['latency_metrics']['latency_seconds']['median']:.2f} seconds")
        print(f"P95 Latency: {summary['latency_metrics']['latency_seconds']['p95']:.2f} seconds")
    
    print("\n--- Cost Metrics ---")
    print(f"Total Cost: ${summary['cost_metrics']['total_cost_usd']:.4f}")
    print(f"Average Cost per Query: ${summary['cost_metrics']['cost_per_query_usd']['mean']:.4f}")
    print(f"Total Tokens: {summary['cost_metrics']['total_tokens']['total']:,}")
    
    print(f"\n\nResults saved to {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()


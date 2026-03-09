"""
Legacy evaluation metrics calculator for Agentic RAG.

This is migrated from root/evaluation_metrics.py and kept for backward compatibility.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class EvaluationMetricsCalculator:
    """Calculates evaluation metrics for RAG system responses."""

    def __init__(self, audit_log_file: str = "data/processed/logs/audit.log"):
        self.audit_log_file = audit_log_file
        self.evaluator_llm = ChatOllama(
            model=os.getenv("EVAL_LLM_MODEL", os.getenv("ROUTER_MODEL", "functiongemma:270m")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
        )

    def load_audit_logs(self) -> List[Dict]:
        """Load audit logs from JSONL file."""
        logs = []
        if not os.path.exists(self.audit_log_file):
            print(f"Warning: Audit log file {self.audit_log_file} not found.")
            return logs

        with open(self.audit_log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return logs

    def extract_statements_from_text(self, text: str) -> List[str]:
        """Extract individual statements/claims from text."""
        statements = [s.strip() for s in text.split(".") if s.strip()]
        statements.extend(
            [s.strip() for s in text.split("\n") if s.strip() and len(s.strip()) > 20]
        )
        return list(set(statements))

    def evaluate_context_relevance(
        self, query: str, context_docs: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Calculate Kontextrelevanz metric.

        Returns:
            (score, details) where score is between 0 and 1
        """
        if not context_docs:
            return 0.0, {"relevant": 0, "total": 0, "statements": []}

        all_statements = []
        for doc in context_docs:
            content = doc.get("page_content", "")
            statements = self.extract_statements_from_text(content)
            all_statements.extend(statements)

        if not all_statements:
            return 0.0, {"relevant": 0, "total": 0, "statements": []}

        statements_text = "\n".join(
            [f"{i + 1}. {stmt}" for i, stmt in enumerate(all_statements)]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Du bist ein Evaluator für RAG-Systeme. Deine Aufgabe ist es, die Relevanz von Aussagen
im Kontext für eine gegebene Nutzerfrage zu bewerten.

Für jede Aussage antworte mit "RELEVANT" oder "IRRELEVANT", getrennt durch Kommas.
Antworte NUR mit den Bewertungen in der Reihenfolge der Aussagen, z.B.: RELEVANT,IRRELEVANT,RELEVANT""",
                ),
                (
                    "user",
                    f"""Nutzerfrage: {query}

Aussagen im Kontext:
{statements_text}

Bewerte jede Aussage als RELEVANT oder IRRELEVANT für die Beantwortung der Frage.""",
                ),
            ]
        )

        chain = prompt | self.evaluator_llm
        response = chain.invoke({})
        evaluations = [e.strip().upper() for e in response.content.split(",")]

        relevant_count = sum(1 for e in evaluations if e == "RELEVANT")
        total_count = len(all_statements)

        score = relevant_count / total_count if total_count > 0 else 0.0

        return score, {
            "relevant": relevant_count,
            "total": total_count,
            "statements": all_statements,
            "evaluations": evaluations,
        }

    def evaluate_context_sufficiency(
        self, query: str, context_docs: List[Dict], answer: str
    ) -> Tuple[float, Dict]:
        """
        Calculate Kontextgenügsamkeit (Context Sufficiency) metric.

        Returns:
            (score, details) where score is between 0 and 1
        """
        prompt1 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Du bist ein Evaluator für RAG-Systeme. Analysiere eine Nutzerfrage und identifiziere
alle Aussagen/Fakten, die benötigt werden, um die Frage vollständig zu beantworten.

Liste jede benötigte Aussage/Fakt in einer eigenen Zeile auf, nummeriert (1., 2., 3., ...).""",
                ),
                (
                    "user",
                    f"Nutzerfrage: {query}\n\nWelche Aussagen/Fakten werden benötigt, um diese Frage zu beantworten?",
                ),
            ]
        )

        chain1 = prompt1 | self.evaluator_llm
        response1 = chain1.invoke({})
        required_statements = [
            s.strip()
            for s in response1.content.split("\n")
            if s.strip() and (s.strip()[0].isdigit() or s.strip().startswith("-"))
        ]

        context_statements = []
        for doc in context_docs:
            content = doc.get("page_content", "")
            statements = self.extract_statements_from_text(content)
            context_statements.extend(statements)

        context_text = "\n".join(context_statements)
        required_text = "\n".join(required_statements)

        prompt2 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Du bist ein Evaluator für RAG-Systeme. Prüfe, ob die benötigten Aussagen im
bereitgestellten Kontext enthalten sind.

Für jede benötigte Aussage antworte mit "VORHANDEN" oder "FEHLT", getrennt durch Kommas.
Antworte NUR mit den Bewertungen in der Reihenfolge der benötigten Aussagen.""",
                ),
                (
                    "user",
                    f"""Benötigte Aussagen:
{required_text}

Verfügbarer Kontext:
{context_text}

Prüfe für jede benötigte Aussage, ob sie im Kontext vorhanden ist.""",
                ),
            ]
        )

        chain2 = prompt2 | self.evaluator_llm
        response2 = chain2.invoke({})
        coverage = [c.strip().upper() for c in response2.content.split(",")]

        covered_count = sum(1 for c in coverage if c == "VORHANDEN")
        total_required = len(required_statements)

        score = covered_count / total_required if total_required > 0 else 0.0

        return score, {
            "covered": covered_count,
            "total_required": total_required,
            "required_statements": required_statements,
            "coverage": coverage,
        }

    def evaluate_answer_hallucination(
        self, answer: str, context_docs: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Calculate Antwort-Halluzination metric.

        Returns:
            (score, details) where score is between 0 and 1 (1 = no hallucination, 0 = all hallucinated)
        """
        answer_statements = self.extract_statements_from_text(answer)

        if not answer_statements:
            return 1.0, {"hallucinated": 0, "total": 0, "statements": []}

        context_statements = []
        for doc in context_docs:
            content = doc.get("page_content", "")
            statements = self.extract_statements_from_text(content)
            context_statements.extend(statements)

        context_text = "\n".join(context_statements)
        answer_text = "\n".join(
            [f"{i + 1}. {stmt}" for i, stmt in enumerate(answer_statements)]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Du bist ein Evaluator für RAG-Systeme. Prüfe, ob jede Behauptung in der Antwort
durch den bereitgestellten Kontext gestützt wird.

Für jede Behauptung antworte mit "GESTÜTZT" oder "NICHT_GESTÜTZT", getrennt durch Kommas.
Antworte NUR mit den Bewertungen in der Reihenfolge der Behauptungen.""",
                ),
                (
                    "user",
                    f"""Behauptungen in der Antwort:
{answer_text}

Verfügbarer Kontext:
{context_text}

Prüfe für jede Behauptung, ob sie durch den Kontext gestützt wird.""",
                ),
            ]
        )

        chain = prompt | self.evaluator_llm
        response = chain.invoke({})
        evaluations = [e.strip().upper() for e in response.content.split(",")]

        hallucinated_count = sum(1 for e in evaluations if e == "NICHT_GESTÜTZT")
        total_count = len(answer_statements)
        hallucination_rate = hallucinated_count / total_count if total_count > 0 else 0.0

        return hallucination_rate, {
            "hallucinated": hallucinated_count,
            "total": total_count,
            "statements": answer_statements,
            "evaluations": evaluations,
        }

    def calculate_all_metrics(self) -> List[Dict]:
        """Calculate all three metrics for all entries in audit log."""
        logs = self.load_audit_logs()
        results = []

        for i, log_entry in enumerate(logs):
            query = log_entry.get("query", "")
            retrieved_docs = log_entry.get("retrieved_documents", [])
            compressed_context = log_entry.get("compressed_context", [])
            answer = log_entry.get("generated_answer", "")
            timestamp = log_entry.get("timestamp", "")
            context_docs = compressed_context if compressed_context else retrieved_docs

            if not query or not context_docs or not answer:
                continue

            print(f"Evaluating entry {i + 1}/{len(logs)}: {query[:50]}...")

            context_relevance_score, context_relevance_details = self.evaluate_context_relevance(
                query, context_docs
            )
            context_sufficiency_score, context_sufficiency_details = self.evaluate_context_sufficiency(
                query, context_docs, answer
            )
            hallucination_rate, hallucination_details = self.evaluate_answer_hallucination(
                answer, context_docs
            )

            result = {
                "entry_id": i + 1,
                "timestamp": timestamp,
                "query": query,
                "metrics": {
                    "kontextrelevanz": {
                        "score": context_relevance_score,
                        "details": context_relevance_details,
                    },
                    "kontextgenügsamkeit": {
                        "score": context_sufficiency_score,
                        "details": context_sufficiency_details,
                    },
                    "antwort_halluzination": {
                        "rate": hallucination_rate,
                        "score": 1.0 - hallucination_rate,
                        "details": hallucination_details,
                    },
                },
            }

            results.append(result)

        return results

    def save_results(
        self,
        results: List[Dict],
        output_file: str = "data/processed/eval_outputs/extended/evaluation_results_legacy.json",
    ):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")

    def generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from evaluation results."""
        if not results:
            return {}

        kontextrelevanz_scores = [r["metrics"]["kontextrelevanz"]["score"] for r in results]
        kontextgenugsamkeit_scores = [r["metrics"]["kontextgenügsamkeit"]["score"] for r in results]
        hallucination_rates = [r["metrics"]["antwort_halluzination"]["rate"] for r in results]
        no_hallucination_scores = [r["metrics"]["antwort_halluzination"]["score"] for r in results]

        def calculate_stats(scores):
            if not scores:
                return {}
            return {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores) // 2] if scores else 0,
            }

        return {
            "total_entries": len(results),
            "kontextrelevanz": calculate_stats(kontextrelevanz_scores),
            "kontextgenügsamkeit": calculate_stats(kontextgenugsamkeit_scores),
            "antwort_halluzination": {
                "rate": calculate_stats(hallucination_rates),
                "no_hallucination_score": calculate_stats(no_hallucination_scores),
            },
        }


def main() -> None:
    calculator = EvaluationMetricsCalculator()
    print("Calculating evaluation metrics...")
    results = calculator.calculate_all_metrics()

    if results:
        calculator.save_results(results)
        summary = calculator.generate_summary_statistics(results)
        print("\n=== Summary Statistics ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("No valid log entries found for evaluation.")


if __name__ == "__main__":
    main()

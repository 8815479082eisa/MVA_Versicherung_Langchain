"""Build the conservative final report from saved runtime and independent adjudication."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
INPUT = BASE / "draft_adjudication_v2.jsonl"
FALLBACK = "The available sources do not contain enough information to answer this question."

CONFIRMED = {
    "pdf-rag-001": {
        "wrong_claim": "This coverage applies if such events occur before your departure, making it impossible for you to set off on your journey.",
        "error_type": "wrong_condition",
        "reason": "The reference evidence covers cancellation, interruption, or extension of the trip; the draft narrows the event to a pre-departure condition that makes departure impossible.",
    },
    "pdf-rag-022": {
        "wrong_claim": "The combined benefits for each event are limited to the insured sum agreed in the policy.",
        "error_type": "wrong_condition",
        "reason": "The reference evidence limits benefits to the agreed arrangement price, not to an unspecified insured sum in the policy.",
    },
    "pdf-rag-025": {
        "wrong_claim": "Additionally, there is a maximum coverage of CHF 50,000 per insured person for unforeseen circumstances.",
        "error_type": "wrong_number, wrong_coverage",
        "reason": "The reference evidence for the unused part of the stay supports proportional arrangement costs excluding transport; the draft attaches an unrelated CHF 50,000 limit to this coverage.",
    },
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(row: dict) -> dict:
    draft = row["generated_draft"].strip()
    verdict = row.get("draft_verdict") or {}
    if draft == FALLBACK:
        result = "fallback"
        wrong_claim = ""
        error_type = ""
        reason = "The draft explicitly abstains because it reports insufficient source information; by definition this is not hallucination."
    elif row["id"] in CONFIRMED:
        result = "hallucination"
        wrong_claim = CONFIRMED[row["id"]]["wrong_claim"]
        error_type = CONFIRMED[row["id"]]["error_type"]
        reason = CONFIRMED[row["id"]]["reason"]
    elif verdict.get("result") == "correct":
        result = "correct"
        wrong_claim = ""
        error_type = ""
        reason = str(verdict.get("reason") or "The factual content is semantically consistent with the reference and supplied evidence.")
    else:
        result = "unknown"
        wrong_claim = ""
        error_type = "unsupported_fact" if verdict.get("wrong_claims") else "other"
        reason = "No factual contradiction could be confirmed from the reference and supplied evidence; the remaining additional, incomplete, or irrelevant content is therefore conservatively classified as unknown."

    return {
        "query_id": row["id"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "reference_evidence": row["reference_evidence"],
        "generated_answer": draft,
        "result": result,
        "wrong_claim": wrong_claim,
        "error_type": error_type,
        "reason": reason,
        "api_status_code": row.get("api_status_code"),
        "api_error_code": row.get("api_error_code"),
        "answer_visibility": "withheld_before_user_delivery",
    }


def esc(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def main() -> None:
    results = sorted((classify(row) for row in read_rows(INPUT)), key=lambda row: row["query_id"])
    counts = Counter(row["result"] for row in results)
    factual = len(results) - counts["fallback"]
    hallucinations = counts["hallucination"]
    summary = {
        "total_queries": len(results),
        "factual_drafts": factual,
        "confirmed_hallucinations_in_drafts": hallucinations,
        "unknown": counts["unknown"],
        "fallback": counts["fallback"],
        "correct": counts["correct"],
        "hallucination_rate_on_factual_drafts": hallucinations / factual if factual else None,
        "hallucination_incidence_over_100_queries": hallucinations / len(results) if results else None,
        "hallucination_exposure_over_all_queries": hallucinations / len(results) if results else None,
        "user_exposed_factual_answers": 0,
        "user_exposed_hallucinations": 0,
        "user_facing_hallucination_exposure": 0.0,
        "http_503_guardrail_invalid_output": sum(row["api_status_code"] == 503 and row["api_error_code"] == "GUARDRAIL_INVALID_OUTPUT" for row in results),
        "method": "reference_answer_plus_reference_evidence_plus_retrieved_evidence_conservative_adjudication",
        "claim_entailment_v1_used_as_sole_judge": False,
        "human_expert_validated": False,
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (BASE / "final_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (BASE / "final_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        flat = [{**row, "reference_evidence": json.dumps(row["reference_evidence"], ensure_ascii=False)} for row in results]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    report = [
        "# Hallucination Evaluation - Current Pipeline - 100 Queries",
        "",
        "## Scope and decision rule",
        "",
        "The first 100 cases of `data/benchmarks/pdf_rag/pdf_rag_200.jsonl` were freshly executed against the current `/api/ask` pipeline. A hallucination is confirmed only when a factual claim is demonstrably contradicted by the reference answer or supplied source evidence, or contains a demonstrably wrong value, condition, coverage, exclusion, time, amount, percentage, location, or concept. Unsupported but non-falsifiable content is `unknown`; deliberate abstention is `fallback`.",
        "",
        "Important runtime qualification: all 100 API calls returned HTTP 503 `GUARDRAIL_INVALID_OUTPUT` after answer generation. Therefore, the `generated_answer` below is the saved draft immediately before the failing groundedness stage, and every draft was withheld before user delivery. Draft-level hallucination incidence and user-facing exposure are reported separately.",
        "",
        "The final decision used the dataset reference answer, its quoted PDF evidence, the full reference page, and retrieved evidence supplied to generation. `claim_entailment_v1` was not used as the sole judge.",
        "",
        "## Summary",
        "",
        f"- Total queries: {summary['total_queries']}",
        f"- Factual drafts: {summary['factual_drafts']}",
        f"- Correct: {summary['correct']}",
        f"- Confirmed hallucinations in factual drafts: {summary['confirmed_hallucinations_in_drafts']}",
        f"- Unknown: {summary['unknown']}",
        f"- Fallback: {summary['fallback']}",
        f"- Hallucination Rate on factual drafts: {summary['confirmed_hallucinations_in_drafts']}/{summary['factual_drafts']} = {100 * summary['hallucination_rate_on_factual_drafts']:.2f}%",
        f"- Draft hallucination incidence over all 100 queries: {summary['confirmed_hallucinations_in_drafts']}/{summary['total_queries']} = {100 * summary['hallucination_incidence_over_100_queries']:.2f}%",
        "- User-facing Hallucination Exposure: 0/100 = 0.00% (all outputs were withheld by the operational guardrail failure)",
        f"- HTTP 503 `GUARDRAIL_INVALID_OUTPUT`: {summary['http_503_guardrail_invalid_output']}",
        "",
        "## Per-query results",
        "",
    ]
    for row in results:
        evidence = "; ".join(
            f"{item.get('source')}, page {item.get('page')}: {item.get('quote')}"
            for item in row["reference_evidence"]
        )
        report.extend([
            f"### {row['query_id']} - {row['result']}",
            "",
            f"- **Question:** {esc(row['question'])}",
            f"- **Reference Answer:** {esc(row['reference_answer'])}",
            f"- **Reference Evidence:** {esc(evidence)}",
            f"- **Generated Answer (withheld draft):** {esc(row['generated_answer'])}",
            f"- **Result:** `{row['result']}`",
            f"- **Wrong Claim:** {esc(row['wrong_claim']) or '-'}",
            f"- **Error Type:** `{row['error_type']}`" if row["error_type"] else "- **Error Type:** -",
            f"- **Reason:** {esc(row['reason'])}",
            f"- **Runtime:** HTTP {row['api_status_code']}, `{row['api_error_code']}`; not exposed to the user.",
            "",
        ])
    (BASE / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()

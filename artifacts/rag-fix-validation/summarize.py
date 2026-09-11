"""Export runtime outcomes without treating the production judge as ground truth."""
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / "hallucination-100-current-2026-09-11"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def outcome(row):
    payload = row.get("payload") or {}
    answer = payload.get("answer") or ""
    evidence = (payload.get("diagnostics") or {}).get("evidence") or {}
    grounding = evidence.get("groundedness") or {}
    if row.get("status_code") != 200:
        return "runtime_error"
    safety = evidence.get("safety") or {}
    if safety.get("postAction") in {"fallback", "block"} or grounding.get("controlled_abstention") or answer.strip() in {
        "I cannot provide a safe, policy-compliant answer for this request. Please rephrase.",
        "I don't know based on the available documents.",
    }:
        return "fallback"
    return "answered" if answer.strip() else "empty_answer"


def main():
    cases = read_rows(PRIOR / "dataset_100.jsonl")
    results = read_rows(BASE / "responses.jsonl")
    assert len(results) == len({r["id"] for r in results})
    assert {r["id"] for r in results} <= {c["id"] for c in cases}
    planned_count = len(cases)
    by_id = {r["id"]: r for r in results}
    cases = [case for case in cases if case["id"] in by_id]
    before = read_rows(PRIOR / "runtime-sequential/responses_full.jsonl")
    rows = []
    for case in cases:
        result = by_id[case["id"]]
        payload = result.get("payload") or {}
        evidence = (payload.get("diagnostics") or {}).get("evidence") or {}
        grounding = evidence.get("groundedness") or {}
        versions = evidence.get("answerVersions") or {}
        rows.append({
            "id": case["id"], "question": case["question"],
            "reference_answer": case["reference_answer"],
            "reference_evidence": case["reference_evidence"],
            "generated_answer": payload.get("answer", ""),
            "internal_draft": versions.get("immediatelyBeforeGroundedness", ""),
            "runtime_outcome": outcome(result), "http_status": result["status_code"],
            "evaluation_status": grounding.get("evaluation_status"),
            "evaluation_failure": grounding.get("failure_code"),
            "guardrail_reasons": (evidence.get("safety") or {}).get("risks", []),
            "sources": payload.get("sources", []),
            "groundedness": grounding,
            "reference_audit": "not_independently_adjudicated",
        })
    summary = {
        "planned_query_count": planned_count,
        "complete": len(rows) == planned_count,
        "query_count": len(rows),
        "before_http_status": dict(Counter(r["status_code"] for r in before)),
        "after_http_status": dict(Counter(r["status_code"] for r in results)),
        "after_runtime_outcomes": dict(Counter(r["runtime_outcome"] for r in rows)),
        "evaluator_failures": dict(Counter(r["evaluation_failure"] for r in rows if r["evaluation_failure"])),
        "hallucination_rate": None,
        "limitations": "No independent factual adjudication: answered is not a correctness label. Evaluator failures and fallback are not confirmed hallucinations. Baseline and rerun use different concurrency; latency is not directly comparable.",
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (BASE / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    fields = ["id", "question", "reference_answer", "generated_answer", "internal_draft", "runtime_outcome", "http_status", "evaluation_status", "evaluation_failure", "reference_evidence"]
    with (BASE / "questions-and-answers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "reference_evidence": json.dumps(row["reference_evidence"], ensure_ascii=False)})
    report = ["# RAG Fix Validation", "", "## Runtime Results", "", "```json", json.dumps(summary, indent=2), "```", "", "## All Questions and Answers", ""]
    for row in rows:
        report.extend([f"### {row['id']}", "", f"Question: {row['question']}", "", f"Reference: {row['reference_answer']}", "", f"Delivered answer: {row['generated_answer']}", "", f"Runtime outcome: {row['runtime_outcome']}; evaluator: {row['evaluation_status']}; failure: {row['evaluation_failure']}", "", "Reference evidence:", *[f"- {e['source']}, page {e['page']}: {e['quote']}" for e in row["reference_evidence"]], ""])
    (BASE / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

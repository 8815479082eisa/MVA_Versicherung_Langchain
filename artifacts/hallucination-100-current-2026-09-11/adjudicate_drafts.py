"""Independent reference/evidence adjudication for the saved 100-query run."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "hallucination-100-current-2026-09-11"
DATASET = BASE / "dataset_100.jsonl"
RESPONSES = BASE / "runtime-sequential" / "responses_full.jsonl"
OUTPUT = Path(os.getenv("ADJUDICATION_OUTPUT") or (BASE / "draft_adjudication_v2.jsonl"))

SYSTEM = """You are an exacting insurance QA factuality adjudicator. Use only the supplied
question, reference answer, reference quote, and fullgad source page. Evaluate the generated
draft semantically, not by wording overlap. Confirm hallucination only when at least one
factual claim is contradicted by the supplied evidence or states a wrong number, amount,
percentage, date, duration, location, condition, coverage, exclusion, polarity, product, or
other fact. A claim that cannot be verified or falsified is unknown, not hallucination.
Omission, irrelevance, verbosity, different wording, and a wrong/missing citation page are not
factual hallucination by themselves. Retrieved evidence may support additional true context
that is absent from the short reference answer; do not mark such context as hallucination.
Return JSON only with: result (correct|hallucination|unknown), factual (boolean),
wrong_claims (array of exact generated claims), error_types (array using wrong_number,
wrong_condition, wrong_coverage, contradiction, unsupported_fact, other), reason (short),
evidence_quote (short exact excerpt supporting the decision). If any confirmed false claim
exists, result must be hallucination. Use unsupported_fact only together with result=unknown
unless the evidence positively proves the unsupported claim false."""


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def draft(response: dict) -> str:
    detail = (response.get("payload") or {}).get("detail") or {}
    evidence = (((detail.get("diagnostics") or {}).get("evidence") or {}))
    versions = evidence.get("answerVersions") or {}
    return str(versions.get("immediatelyBeforeGroundedness") or versions.get("rawOpenAIAnswer") or "").strip()


def retrieved_evidence(response: dict) -> str:
    detail = (response.get("payload") or {}).get("detail") or {}
    evidence = (((detail.get("diagnostics") or {}).get("evidence") or {}))
    documents = ((evidence.get("generationContext") or {}).get("documents") or [])
    parts = []
    seen = set()
    for doc in documents:
        filename = str(doc.get("filename") or "")
        page = doc.get("physicalPdfPage")
        excerpt = str(doc.get("textExcerpt") or "").strip()
        key = (filename, page, excerpt)
        if excerpt and key not in seen:
            seen.add(key)
            parts.append(f"[{filename}, page {page}]\n{excerpt}")
    return "\n\n".join(parts)


def page_text(case: dict, cache: dict[tuple[str, int], str]) -> str:
    evidence = case.get("reference_evidence") or []
    if not evidence:
        return ""
    filename = str(evidence[0].get("source") or "")
    page = int(evidence[0].get("page") or 0)
    key = (filename, page)
    if key not in cache:
        path = ROOT / "data" / "raw" / "pdfs" / filename
        reader = PdfReader(path)
        cache[key] = reader.pages[page - 1].extract_text() if 0 < page <= len(reader.pages) else ""
    return cache[key]


def parse_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    return json.loads(cleaned)


def evaluate(client: OpenAI, case: dict, response: dict, source_text: str) -> dict:
    generated = draft(response)
    reference_quote = "\n".join(str(x.get("quote") or "") for x in case.get("reference_evidence") or [])
    prompt = (
        f"QUERY ID: {case['id']}\nQUESTION:\n{case['question']}\n\n"
        f"REFERENCE ANSWER:\n{case.get('reference_answer', '')}\n\n"
        f"REFERENCE QUOTE:\n{reference_quote}\n\nFULL REFERENCE SOURCE PAGE:\n{source_text}\n\n"
        f"ALL RETRIEVED EVIDENCE SUPPLIED TO GENERATION:\n{retrieved_evidence(response)}\n\n"
        f"GENERATED DRAFT:\n{generated}"
    )
    last_error = None
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=os.getenv("ADJUDICATION_MODEL") or os.getenv("GROUNDEDNESS_MODEL") or os.getenv("OPENAI_ANSWER_MODEL") or "gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            )
            verdict = parse_json(completion.choices[0].message.content or "{}")
            return {
                "id": case["id"], "question": case["question"],
                "reference_answer": case.get("reference_answer", ""),
                "reference_evidence": case.get("reference_evidence", []),
                "generated_draft": generated, "api_status_code": response.get("status_code"),
                "api_error_code": (((response.get("payload") or {}).get("detail") or {}).get("errorCode")),
                "draft_verdict": verdict, "adjudication_error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "id": case["id"], "question": case["question"],
        "reference_answer": case.get("reference_answer", ""),
        "reference_evidence": case.get("reference_evidence", []),
        "generated_draft": generated, "api_status_code": response.get("status_code"),
        "api_error_code": (((response.get("payload") or {}).get("detail") or {}).get("errorCode")),
        "draft_verdict": {"result": "unknown", "factual": bool(generated), "wrong_claims": [],
                          "error_types": ["other"], "reason": "Adjudication failed.", "evidence_quote": ""},
        "adjudication_error": last_error,
    }


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    cases = rows(DATASET)
    responses = {row["id"]: row for row in rows(RESPONSES)}
    cache: dict[tuple[str, int], str] = {}
    source_pages = {case["id"]: page_text(case, cache) for case in cases}
    client = OpenAI()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, client, case, responses[case["id"]], source_pages[case["id"]]): case["id"] for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = pretty = future.result()
            results.append(result)
            print(f"[{index}/100] {result['id']}: {result['draft_verdict'].get('result')}", flush=True)
    order = {case["id"]: i for i, case in enumerate(cases)}
    results.sort(key=lambda item: order[item["id"]])
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")


if __name__ == "__main__":
    main()

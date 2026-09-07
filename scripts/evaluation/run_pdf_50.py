"""Run a 50-question English benchmark against the project's PDF corpus."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROUTING_DATASET = ROOT / "data" / "benchmarks" / "routing" / "routing_eval_80.jsonl"
OUT = ROOT / "artifacts" / "pdf-50-evaluation" / "results.jsonl"
ENDPOINT = "http://localhost:8000/api/ask"
FALLBACK = "I could not generate an answer that is sufficiently supported by the available documents."

EXTRA_CASES = [
    ("What is the first insurance year for the private rental guarantee insurance?", "rental-guarantee-insurance-sti.pdf"),
    ("How is the private rental guarantee insurance contract renewed after the first year?", "rental-guarantee-insurance-sti.pdf"),
    ("What termination rule is referenced for the private rental guarantee insurance?", "rental-guarantee-insurance-sti.pdf"),
    ("What does the motor vehicle policy say about deductible-free windscreen repair?", "motor-vehicle-insurance-sti.pdf"),
    ("When is a deductible waived if a damaged front windscreen is repaired?", "motor-vehicle-insurance-sti.pdf"),
    ("Which company must organize a windscreen repair for the deductible to be waived?", "motor-vehicle-insurance-sti.pdf"),
    ("What is included in the GlassPlus extension of motor vehicle insurance?", "motor-vehicle-insurance-sti.pdf"),
    ("How does the policy distinguish collision damage from glass damage?", "motor-vehicle-insurance-sti.pdf"),
    ("What does the motor vehicle policy state about theft of the insured vehicle?", "motor-vehicle-insurance-sti.pdf"),
    ("Which vehicle damage caused by wear and tear is excluded?", "motor-vehicle-insurance-sti.pdf"),
    ("What information is listed on the motor vehicle insurance product information sheet?", "motor-vehicle-insurance-product-sheet.pdf"),
    ("Does household contents insurance include damage caused by leaking water?", "household-contents-private-liability-sti.pdf"),
    ("What does the household contents policy say about fire damage?", "household-contents-private-liability-sti.pdf"),
    ("What types of loss are covered by private liability insurance?", "household-contents-private-liability-sti.pdf"),
    ("What exclusions apply to private liability insurance claims?", "household-contents-private-liability-sti.pdf"),
    ("What benefits are highlighted in the household contents and private liability brochure?", "brochure-household-contents-and-private-liability.pdf"),
    ("When does buildings insurance cover natural hazards?", "buildings-insurance-sti.pdf"),
    ("Which kinds of building damage are not covered by the buildings insurance terms?", "buildings-insurance-sti.pdf"),
    ("What legal disputes can be covered by legal protection insurance?", "legal-protection-sti.pdf"),
    ("What waiting periods are mentioned in the legal protection insurance terms?", "legal-protection-sti.pdf"),
    ("What services are available after an insured vehicle breaks down?", "assistance-sti.pdf"),
    ("What costs can the assistance insurance reimburse after a breakdown?", "assistance-sti.pdf"),
    ("What travel interruption benefits are described in the assistance brochure?", "assistance-brochure.pdf"),
    ("What services are summarized in the Helvetia insurance services brochure?", "brochure-services.pdf"),
    ("What is covered by the private rental guarantee insurance?", "rental-guarantee-insurance-sti.pdf"),
    ("What period is stipulated for the insurance contract under the mutual private health provisions?", "mutual-provisions-pkv.pdf"),
    ("How is a private health insurance contract renewed under the mutual provisions?", "mutual-provisions-pkv.pdf"),
    ("How much notice is required to terminate the private health insurance contract?", "mutual-provisions-pkv.pdf"),
    ("What does the policy say about damage during an officially licensed driving test?", "motor-vehicle-insurance-sti.pdf"),
    ("Under what conditions does the motor vehicle policy say no deductible is payable?", "motor-vehicle-insurance-sti.pdf"),
]


def load_cases() -> list[dict[str, str]]:
    rows = [json.loads(line) for line in ROUTING_DATASET.read_text(encoding="utf-8").splitlines()]
    cases = [
        {"id": row["id"], "question": row["question"], "expected_source": Path(row["source_references"][0]).name}
        for row in rows
        if row.get("expected_route") == "retrieval_only"
    ]
    cases.extend(
        {"id": f"pdf-extra-{index:03d}", "question": question, "expected_source": source}
        for index, (question, source) in enumerate(EXTRA_CASES, start=1)
    )
    assert len(cases) == 50, len(cases)
    return cases


def main() -> int:
    cases = load_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    run_id = datetime.now(timezone.utc).isoformat()
    success = failures = fallback = source_hits = 0
    scores: list[float] = []
    with requests.Session() as client, OUT.open("a", encoding="utf-8") as handle:
        health = client.get(ENDPOINT.rsplit("/api/ask", 1)[0] + "/health", timeout=15)
        health.raise_for_status()
        print(f"Health: {health.json().get('status')} | cases={len(cases)}", flush=True)
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            row: dict[str, object] = {
                "run_id": run_id,
                "index": index,
                **case,
            }
            try:
                response = None
                error = None
                for attempt in range(2):
                    try:
                        response = client.post(ENDPOINT, json={"question": case["question"]}, timeout=180)
                        if response.status_code < 500:
                            break
                    except Exception as exc:  # pragma: no cover - live runner
                        error = f"{type(exc).__name__}: {exc}"
                    time.sleep(2)
                if response is None:
                    raise RuntimeError(error or "no response")
                row["status_code"] = response.status_code
                payload = response.json()
                answer = str(payload.get("answer") or "")
                row["answer"] = answer
                row["fallback"] = answer.strip() == FALLBACK
                row["sources"] = payload.get("sources") or []
                row["groundedness_score"] = ((payload.get("diagnostics") or {}).get("evidence") or {}).get("groundedness", {}).get("score")
                row["timeout_stage"] = (payload.get("diagnostics") or {}).get("timeoutStage")
                returned = json.dumps(row["sources"], ensure_ascii=False).casefold()
                row["expected_source_found"] = case["expected_source"].casefold() in returned
                success += int(response.status_code < 500)
                failures += int(response.status_code >= 500)
                fallback += int(row["fallback"])
                source_hits += int(row["expected_source_found"])
                if isinstance(row["groundedness_score"], (float, int)):
                    scores.append(float(row["groundedness_score"]))
            except Exception as exc:  # pragma: no cover - live runner
                failures += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/50] HTTP={row.get('status_code', 'ERR')} "
                f"fallback={row.get('fallback', 'n/a')} latency={row['latency_ms']:.0f}ms",
                flush=True,
            )
    print("=== SUMMARY ===", flush=True)
    print(f"Total: 50 | HTTP success: {success} | HTTP failures: {failures}", flush=True)
    print(f"Fallback answers: {fallback} | Expected PDF source found: {source_hits}", flush=True)
    if scores:
        print(f"Groundedness mean (available responses): {sum(scores) / len(scores):.6f}", flush=True)
    print(f"Output: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

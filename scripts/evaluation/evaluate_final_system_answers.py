"""Collect and evaluate full responses for the final 80-case API benchmark.

The output is explicitly an automated, reference-based technical validation,
not independent human validation.  Full API payloads are retained so every
per-case decision can be audited and reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reference_answer_quality import (
    CRMReferenceStore,
    evaluate_case,
    load_jsonl,
    load_reference_spec,
    normalize_route,
    render_markdown_report,
    summarize_evaluations,
    validate_spec_coverage,
    write_csv,
    write_jsonl,
)


DEFAULT_DATASET = PROJECT_ROOT / "data" / "benchmarks" / "routing" / "routing_eval_80.jsonl"
DEFAULT_SPEC = PROJECT_ROOT / "data" / "benchmarks" / "answer_quality" / "reference_spec_v1.json"
DEFAULT_CRM_DIR = PROJECT_ROOT / "data" / "synthetic" / "crm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "answer-quality-evaluation"
COLLECTION_METADATA_FILENAME = "collection_metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(
    cases: Sequence[dict[str, Any]],
    *,
    start: int,
    limit: int | None,
    routes: set[str] | None,
    case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        case
        for case in cases
        if (not routes or normalize_route(case.get("expected_route")) in routes)
        and (case_ids is None or str(case.get("id")) in case_ids)
    ]
    selected = selected[max(0, start) :]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(row["id"]): row for row in load_jsonl(path) if row.get("id")}


def _write_rows_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def collect_responses(
    cases: Sequence[dict[str, Any]],
    *,
    endpoint: str,
    timeout_seconds: float,
    responses_path: Path,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = {} if overwrite else _read_existing(responses_path)
    selected_ids = {str(case["id"]) for case in cases}
    selected_questions = {str(case["id"]): str(case["question"]) for case in cases}
    existing = {
        case_id: row
        for case_id, row in existing.items()
        if case_id not in selected_ids
        or row.get("question") == selected_questions[case_id]
    }
    rows_by_id = dict(existing)
    base_url = endpoint.rsplit("/api/ask", 1)[0]
    health: dict[str, Any]

    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            response = client.get(f"{base_url}/health")
            health = {
                "reachable": True,
                "status_code": response.status_code,
                "payload": response.json(),
            }
        except Exception as exc:
            health = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
            raise RuntimeError(f"API health check failed: {health['error']}") from exc

        for index, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            if case_id in existing:
                print(f"[{index}/{len(cases)}] {case_id}: reused saved response")
                continue

            started = time.perf_counter()
            status_code = 0
            payload: Any = {}
            error: str | None = None
            try:
                response = client.post(endpoint, json={"question": case["question"]})
                status_code = response.status_code
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"raw_text": response.text}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - started) * 1000, 4)
            row = {
                "id": case_id,
                "question": case["question"],
                "expected_route": normalize_route(case.get("expected_route")),
                "status_code": status_code,
                "latency_ms": latency_ms,
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "payload": payload if isinstance(payload, dict) else {"value": payload},
                "collection_error": error,
            }
            rows_by_id[case_id] = row
            ordered = [
                rows_by_id[str(item["id"])]
                for item in cases
                if str(item["id"]) in rows_by_id
            ]
            ordered.extend(
                row
                for case_id, row in rows_by_id.items()
                if case_id not in selected_ids
            )
            _write_rows_atomic(responses_path, ordered)
            print(f"[{index}/{len(cases)}] {case_id}: HTTP {status_code}, {latency_ms:.1f} ms")

    ordered_rows = [rows_by_id[str(case["id"])] for case in cases if str(case["id"]) in rows_by_id]
    return ordered_rows, health


def evaluate_saved_responses(
    cases: Sequence[dict[str, Any]],
    *,
    responses_path: Path,
    reference_spec_path: Path,
    crm_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_spec = load_reference_spec(reference_spec_path)
    spec_errors = validate_spec_coverage(
        cases,
        reference_spec,
        allow_extra_specs=True,
    )
    if spec_errors:
        raise ValueError("Invalid reference specification:\n- " + "\n- ".join(spec_errors))

    responses = _read_existing(responses_path)
    collected_times = sorted(
        str(row["collected_at_utc"])
        for row in responses.values()
        if row.get("collected_at_utc")
    )
    crm_store = CRMReferenceStore(crm_dir)
    evaluations = [
        evaluate_case(
            case,
            responses.get(str(case["id"])),
            reference_spec["cases"][str(case["id"])],
            reference_spec,
            crm_store,
        )
        for case in cases
    ]
    summary = summarize_evaluations(evaluations)
    summary["metadata"] = {
        **metadata,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(metadata.get("dataset")),
        "dataset_sha256": sha256_file(Path(metadata["dataset"])),
        "reference_spec": str(reference_spec_path),
        "reference_spec_sha256": sha256_file(reference_spec_path),
        "reference_spec_version": reference_spec.get("version"),
        "responses": str(responses_path),
        "response_count": len(responses),
        "first_response_collected_at_utc": collected_times[0] if collected_times else None,
        "last_response_collected_at_utc": collected_times[-1] if collected_times else None,
        "method_label": reference_spec.get("method_label"),
        "human_validated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_case_results.jsonl", evaluations)
    write_csv(output_dir / "per_case_results.csv", evaluations)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_markdown_report(summary), encoding="utf-8"
    )
    error_catalog = [
        {
            "id": row["id"],
            "expected_route": row.get("expected_route"),
            "actual_route": row.get("actual_route"),
            "errors": row.get("errors") or [],
            "missing_requirements": [
                requirement["id"]
                for requirement in row.get("requirements") or []
                if not requirement.get("covered")
            ],
            "missing_crm_facts": [
                fact for fact in row.get("crm_facts") or [] if not fact.get("present")
            ],
        }
        for row in evaluations
        if row.get("errors")
    ]
    (output_dir / "error_catalog.json").write_text(
        json.dumps(error_catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return evaluations, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "evaluate", "both"), default="both")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--crm-dir", type=Path, default=DEFAULT_CRM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/ask")
    parser.add_argument("--timeout-seconds", type=float, default=190.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--routes",
        nargs="*",
        help="Optional expected routes: crm_only retrieval_only combined denied",
    )
    parser.add_argument(
        "--overwrite-responses",
        action="store_true",
        help="Discard saved responses instead of resuming the collection.",
    )
    parser.add_argument(
        "--failed-from",
        type=Path,
        help="Select exactly the failed_case_ids recorded in a previous summary.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    reference_spec = args.reference_spec.resolve()
    crm_dir = args.crm_dir.resolve()
    output_dir = args.output_dir.resolve()
    responses_path = (
        args.responses.resolve()
        if args.responses
        else output_dir / "responses_full.jsonl"
    )
    all_cases = load_jsonl(dataset)
    routes = {normalize_route(route) for route in args.routes} if args.routes else None
    failed_case_ids: set[str] | None = None
    failed_from: Path | None = None
    if args.failed_from:
        failed_from = args.failed_from.resolve()
        previous_summary = json.loads(failed_from.read_text(encoding="utf-8"))
        failed_case_ids = {
            str(case_id) for case_id in previous_summary.get("failed_case_ids") or []
        }
    cases = select_cases(
        all_cases,
        start=args.start,
        limit=args.limit,
        routes=routes,
        case_ids=failed_case_ids,
    )
    if not cases:
        raise SystemExit("No cases selected.")

    metadata: dict[str, Any] = {
        "dataset": str(dataset),
        "selected_case_count": len(cases),
        "selected_case_ids": [case["id"] for case in cases],
        "failed_from": str(failed_from) if failed_from else None,
        "endpoint": args.endpoint,
        "collection_health": None,
    }
    collection_metadata_path = output_dir / COLLECTION_METADATA_FILENAME

    if args.mode in {"collect", "both"}:
        _, health = collect_responses(
            cases,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            responses_path=responses_path,
            overwrite=args.overwrite_responses,
        )
        metadata["collection_health"] = health
        collection_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        collection_metadata_path.write_text(
            json.dumps(
                {
                    "endpoint": args.endpoint,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "collection_health": health,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    elif collection_metadata_path.is_file():
        saved_collection_metadata = json.loads(
            collection_metadata_path.read_text(encoding="utf-8")
        )
        metadata["collection_health"] = saved_collection_metadata.get(
            "collection_health"
        )
        metadata["collection_health_checked_at_utc"] = saved_collection_metadata.get(
            "checked_at_utc"
        )

    if args.mode in {"evaluate", "both"}:
        _, summary = evaluate_saved_responses(
            cases,
            responses_path=responses_path,
            reference_spec_path=reference_spec,
            crm_dir=crm_dir,
            output_dir=output_dir,
            metadata=metadata,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Saved complete response payloads to {responses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

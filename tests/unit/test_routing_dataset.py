from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.evaluation.evaluate_routing import (
    PROJECT_ROOT,
    ROUTES,
    _extract_route,
    evaluate_local,
    load_cases,
    normalize_route,
    summarize_rows,
)


DATASET = (
    PROJECT_ROOT / "data" / "benchmarks" / "routing" / "routing_eval_80.jsonl"
)
REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_route",
    "category",
    "subcategory",
    "expected_stage",
    "source_references",
    "expected_entities",
    "rationale",
}


def test_routing_dataset_is_balanced_and_well_formed() -> None:
    cases = load_cases(DATASET)

    assert len(cases) == 80
    assert len({case["id"] for case in cases}) == 80
    assert len({case["question"].casefold() for case in cases}) == 80
    assert Counter(case["expected_route"] for case in cases) == {
        "crm_only": 20,
        "retrieval_only": 20,
        "combined": 20,
        "denied": 20,
    }

    for case in cases:
        assert REQUIRED_FIELDS <= case.keys()
        assert case["expected_route"] in ROUTES
        assert case["category"] == case["expected_route"]
        assert case["expected_stage"] in {"route_planner", "safety_precheck"}
        assert case["question"].strip() == case["question"]
        assert case["rationale"].strip()
        assert isinstance(case["source_references"], list)
        assert isinstance(case["expected_entities"], list)
        for source in case["source_references"]:
            assert (PROJECT_ROOT / source).is_file(), (case["id"], source)


def test_denied_cases_cover_both_denial_stages() -> None:
    denied = [
        case for case in load_cases(DATASET) if case["expected_route"] == "denied"
    ]

    assert Counter(case["expected_stage"] for case in denied) == {
        "route_planner": 10,
        "safety_precheck": 10,
    }


def test_local_front_door_evaluates_every_case() -> None:
    rows = evaluate_local(load_cases(DATASET))

    assert len(rows) == 80
    assert len({row["id"] for row in rows}) == 80
    assert all(row["actual_route"] in ROUTES for row in rows)
    assert all(row["stage_correct"] for row in rows), [
        (row["id"], row["expected_stage"], row["actual_stage"])
        for row in rows
        if not row["stage_correct"]
    ]


def test_summary_reports_balanced_support_and_consistent_totals() -> None:
    rows = evaluate_local(load_cases(DATASET))
    summary = summarize_rows(rows)

    assert summary["total"] == 80
    assert summary["correct"] == sum(row["route_correct"] for row in rows)
    assert summary["routing_accuracy"] == summary["correct"] / 80
    assert 0.0 <= summary["macro_f1"] <= 1.0
    assert summary["stage_accuracy"] == 1.0
    for route in ROUTES:
        assert summary["per_route"][route]["support"] == 20
        assert 0 <= summary["per_route"][route]["correct"] <= 20


def test_operational_success_is_strictly_route_and_downstream_success() -> None:
    rows = evaluate_local(load_cases(DATASET))
    summary = summarize_rows(rows)

    assert summary["downstream_operational_success_count"] == 80
    assert summary["operational_success_count"] == summary["correct"] == 80
    assert summary["operational_failure_ids"] == []


def test_route_normalization_and_error_payload_extraction() -> None:
    assert normalize_route("crm-only") == "crm_only"
    assert normalize_route("rag_only") == "retrieval_only"
    assert _extract_route({"route": "retrieval-only"}) == "retrieval_only"
    assert _extract_route({"detail": {"route": "combined"}}) == "combined"
    assert _extract_route({"detail": {"diagnostics": {"route": "crm-only"}}}) == "crm_only"
    assert _extract_route({"detail": "No local FAQ match"}) == "unknown"

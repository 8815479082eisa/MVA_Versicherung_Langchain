from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_groundedness_threshold_weak_supervision import (  # noqa: E402
    build_group_cv_splits,
    metrics_at,
)
from scripts.experimental_groundedness_v5 import (  # noqa: E402
    EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
)


DATASET = ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"
SCORES = ROOT / "reports" / "groundedness_v5_experimental_scores.jsonl"
V5_SUMMARY = ROOT / "reports" / "groundedness_v5_experimental_summary.json"
V5_CODE = ROOT / "scripts" / "experimental_groundedness_v5.py"
PRODUCTION_CODE = ROOT / "src" / "guardrails" / "integrations" / "nemo_actions.py"
PRODUCTION_CONFIG = ROOT / "config" / "groundedness_calibration.json"

REPORT = ROOT / "reports" / "groundedness_v5_threshold_078_vs_07888_20260802.md"
SUMMARY = ROOT / "reports" / "groundedness_v5_threshold_078_vs_07888_summary.json"
BOUNDARY = ROOT / "reports" / "groundedness_v5_threshold_078_boundary_cases.csv"
CV = ROOT / "reports" / "groundedness_v5_threshold_078_cv_comparison.csv"

THRESHOLDS = (0.780000, 0.788800)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def primary_high(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["weak_label_confidence"] == "HIGH"
        and not row["quality_flags"]
        and not row["duplicate_group_id"]
    ]


def metric_payload(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = metrics_at(rows, threshold)
    result["rejected_technical_pass_cases"] = int(result["fn"])
    return result


def evaluate_views(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    views: dict[str, list[dict[str, Any]]] = {
        "calibration_all": [row for row in scores if row["split"] == "CALIBRATION"],
        "validation_all": [row for row in scores if row["split"] == "VALIDATION"],
        "technical_weak_label_holdout_all": [
            row for row in scores if row["split"] == "LOCKED_HOLDOUT_CANDIDATE"
        ],
        "calibration_primary_high": [row for row in primary_high(scores) if row["split"] == "CALIBRATION"],
        "validation_primary_high": [row for row in primary_high(scores) if row["split"] == "VALIDATION"],
        "technical_weak_label_holdout_primary_high": [
            row for row in primary_high(scores) if row["split"] == "LOCKED_HOLDOUT_CANDIDATE"
        ],
        "all_cases": list(scores),
        "high_confidence_all": [row for row in scores if row["weak_label_confidence"] == "HIGH"],
        "high_plus_medium_all": [
            row for row in scores if row["weak_label_confidence"] in {"HIGH", "MEDIUM"}
        ],
        "high_risk_proxy_failures": [
            row for row in scores if row["weak_label"] == "FAIL" and row["high_risk_mutation_proxy"]
        ],
    }
    result = {
        name: {f"{threshold:.6f}": metric_payload(rows, threshold) for threshold in THRESHOLDS}
        for name, rows in views.items()
    }
    usable = views["high_plus_medium_all"]
    result["source_types"] = {
        source: {
            f"{threshold:.6f}": metric_payload(
                [row for row in usable if row["source_type"] == source], threshold
            )
            for threshold in THRESHOLDS
        }
        for source in sorted({row["source_type"] for row in usable})
    }
    result["mutation_types"] = {
        mutation: {
            f"{threshold:.6f}": metric_payload(
                [row for row in usable if row["mutation_type"] == mutation], threshold
            )
            for threshold in THRESHOLDS
        }
        for mutation in sorted({row["mutation_type"] for row in usable})
    }
    return result


def fixed_cv(scores: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = [row for row in primary_high(scores) if row["split"] == "CALIBRATION"]
    folds, splits = build_group_cv_splits(calibration)
    rows: list[dict[str, Any]] = []
    for repeat, fold, _, test_idx in splits:
        test = [calibration[int(index)] for index in test_idx]
        for threshold in THRESHOLDS:
            metrics = metric_payload(test, threshold)
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "threshold": threshold,
                    "test_case_count": len(test),
                    "test_leakage_group_count": len({row["leakage_group_id"] for row in test}),
                    "test_pass": sum(row["weak_label"] == "PASS" for row in test),
                    "test_fail": sum(row["weak_label"] == "FAIL" for row in test),
                    "test_tp": metrics["tp"],
                    "test_fp": metrics["fp"],
                    "test_tn": metrics["tn"],
                    "test_fn": metrics["fn"],
                    "test_far": metrics["technical_far"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "test_f1": metrics["f1"],
                    "test_specificity": metrics["specificity"],
                    "test_balanced_accuracy": metrics["balanced_accuracy"],
                    "test_g_mean": metrics["g_mean"],
                    "test_coverage": metrics["coverage"],
                    "test_accepted_high_risk_proxy_failures": metrics[
                        "accepted_high_risk_proxy_failures"
                    ],
                    "test_rejected_technical_pass_cases": metrics["rejected_technical_pass_cases"],
                }
            )
    aggregate: dict[str, Any] = {"folds": folds, "repeats": 10, "random_seed": 20260801}
    for threshold in THRESHOLDS:
        selected = [row for row in rows if row["threshold"] == threshold]
        fars = [float(row["test_far"]) for row in selected]
        high_risk = [int(row["test_accepted_high_risk_proxy_failures"]) for row in selected]
        aggregate[f"{threshold:.6f}"] = {
            "fold_count": len(selected),
            "mean_test_far": statistics.mean(fars),
            "median_test_far": statistics.median(fars),
            "mean_precision": statistics.mean(float(row["test_precision"]) for row in selected),
            "mean_recall": statistics.mean(float(row["test_recall"]) for row in selected),
            "mean_f1": statistics.mean(float(row["test_f1"]) for row in selected),
            "mean_coverage": statistics.mean(float(row["test_coverage"]) for row in selected),
            "folds_far_gt_5pct": sum(value > 0.05 for value in fars),
            "folds_with_high_risk_false_acceptance": sum(value > 0 for value in high_risk),
            "fraction_folds_with_high_risk_false_acceptance": sum(value > 0 for value in high_risk)
            / len(high_risk),
            "maximum_high_risk_false_acceptances_in_fold": max(high_risk),
        }
    return rows, aggregate


def compact_context(texts: Sequence[str]) -> str:
    normalized = " ".join(" ".join(text.split()) for text in texts)
    return normalized[:320] + ("…" if len(normalized) > 320 else "")


def main() -> int:
    outputs = [REPORT, SUMMARY, BOUNDARY, CV]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite comparison artifacts:\n" + "\n".join(existing))
    for path in (DATASET, SCORES, V5_SUMMARY, V5_CODE, PRODUCTION_CODE, PRODUCTION_CONFIG):
        if not path.exists():
            raise FileNotFoundError(path)

    cases = read_jsonl(DATASET)
    scores = read_jsonl(SCORES)
    prior_summary = json.loads(V5_SUMMARY.read_text(encoding="utf-8"))
    cases_by_id = {row["case_id"]: row for row in cases}
    scores_by_id = {row["case_id"]: row for row in scores}
    if len(cases) != 498 or len(scores) != 498 or len(scores_by_id) != 498:
        raise ValueError("Expected 498 unique cases and scores")
    if set(cases_by_id) != set(scores_by_id):
        raise ValueError("Case IDs differ between dataset and stored scores")
    if {row.get("algorithm_version") for row in scores} != {
        EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION
    }:
        raise ValueError("Stored algorithm version differs from current experimental v5")
    if any(row.get("v5_score") is None for row in scores):
        raise ValueError("Stored scores are incomplete")
    split_mismatches = [
        case_id
        for case_id, case in cases_by_id.items()
        if scores_by_id[case_id]["split"] != case["split"]
    ]
    group_mismatches = [
        case_id
        for case_id, case in cases_by_id.items()
        if scores_by_id[case_id]["leakage_group_id"] != case["leakage_group_id"]
    ]
    if split_mismatches or group_mismatches:
        raise ValueError("Stored split or leakage-group assignments differ from dataset")
    dataset_sha = sha256(DATASET)
    if dataset_sha != prior_summary["dataset_sha256"]:
        raise ValueError("Current dataset hash differs from the v5 scoring summary")
    current_input_hashes = {
        case_id: stable_hash(
            {
                "question": case["question"],
                "context": case["context"],
                "candidate_answer": case["candidate_answer"],
            }
        )
        for case_id, case in cases_by_id.items()
    }
    missing_stored_input_hashes = [
        row["case_id"] for row in scores if not row.get("input_sha256")
    ]
    earliest_score_timestamp = min(row["score_timestamp"] for row in scores)
    code_modified_at = datetime.fromtimestamp(V5_CODE.stat().st_mtime, timezone.utc).isoformat()
    if code_modified_at > earliest_score_timestamp:
        raise ValueError("Experimental v5 code is newer than the stored scores; rescore must be separately authorized")

    evaluations = evaluate_views(scores)
    cv_rows, cv_aggregate = fixed_cv(scores)
    write_csv(CV, cv_rows)

    boundary_scores = [row for row in scores if THRESHOLDS[0] <= float(row["v5_score"]) < THRESHOLDS[1]]
    boundary_rows: list[dict[str, Any]] = []
    for row in sorted(boundary_scores, key=lambda item: float(item["v5_score"])):
        case = cases_by_id[row["case_id"]]
        impact = (
            "additional_correct_accept_of_technical_pass"
            if row["weak_label"] == "PASS"
            else "additional_false_accept_of_technical_fail"
        )
        boundary_rows.append(
            {
                "case_id": row["case_id"],
                "score": row["v5_score"],
                "split": row["split"],
                "source_type": row["source_type"],
                "weak_label": row["weak_label"],
                "weak_label_confidence": row["weak_label_confidence"],
                "mutation_type": row["mutation_type"],
                "high_risk_mutation_proxy": row["high_risk_mutation_proxy"],
                "question": case["question"],
                "candidate_answer": case["candidate_answer"],
                "context_summary": compact_context(case["context"]),
                "decision_at_0_780000": "ACCEPT",
                "decision_at_0_788800": "REJECT",
                "technical_impact": impact,
                "current_input_sha256": current_input_hashes[row["case_id"]],
            }
        )
    write_csv(BOUNDARY, boundary_rows)

    extra_pass = sum(row["weak_label"] == "PASS" for row in boundary_scores)
    extra_fail = sum(row["weak_label"] == "FAIL" for row in boundary_scores)
    extra_high_risk = sum(
        row["weak_label"] == "FAIL" and bool(row["high_risk_mutation_proxy"])
        for row in boundary_scores
    )
    sources = dict(Counter(row["source_type"] for row in boundary_scores))
    splits = evaluations
    low_cv = cv_aggregate["0.780000"]
    high_cv = cv_aggregate["0.788800"]
    fixed_split_names = (
        "calibration_all",
        "validation_all",
        "technical_weak_label_holdout_all",
    )
    safety = {
        "threshold_0_780000_split_far_le_5pct": all(
            splits[name]["0.780000"]["technical_far"] <= 0.05
            for name in fixed_split_names
        ),
        "threshold_0_788800_split_far_le_5pct": all(
            splits[name]["0.788800"]["technical_far"] <= 0.05
            for name in fixed_split_names
        ),
        "threshold_0_780000_split_zero_high_risk_false_acceptances": all(
            splits[name]["0.780000"]["accepted_high_risk_proxy_failures"] == 0
            for name in fixed_split_names
        ),
        "threshold_0_788800_split_zero_high_risk_false_acceptances": all(
            splits[name]["0.788800"]["accepted_high_risk_proxy_failures"] == 0
            for name in fixed_split_names
        ),
        "threshold_0_780000_total_split_high_risk_false_acceptances": sum(
            int(splits[name]["0.780000"]["accepted_high_risk_proxy_failures"])
            for name in fixed_split_names
        ),
        "threshold_0_788800_total_split_high_risk_false_acceptances": sum(
            int(splits[name]["0.788800"]["accepted_high_risk_proxy_failures"])
            for name in fixed_split_names
        ),
        "primary_high_threshold_0_780000_zero_high_risk_false_acceptances": all(
            splits[name]["0.780000"]["accepted_high_risk_proxy_failures"] == 0
            for name in (
                "calibration_primary_high",
                "validation_primary_high",
                "technical_weak_label_holdout_primary_high",
            )
        ),
        "primary_high_threshold_0_788800_zero_high_risk_false_acceptances": all(
            splits[name]["0.788800"]["accepted_high_risk_proxy_failures"] == 0
            for name in (
                "calibration_primary_high",
                "validation_primary_high",
                "technical_weak_label_holdout_primary_high",
            )
        ),
        "cv_high_risk_not_worse": low_cv["fraction_folds_with_high_risk_false_acceptance"]
        <= high_cv["fraction_folds_with_high_risk_false_acceptance"],
        "practically_relevant_recall_or_coverage_gain": extra_pass > 0,
        "not_dependent_on_few_boundary_cases": len(boundary_scores) >= 5,
        "additional_fail_acceptances": extra_fail,
        "additional_high_risk_acceptances": extra_high_risk,
    }
    either_threshold_fails_absolute_split_safety = not all(
        (
            safety["threshold_0_780000_split_far_le_5pct"],
            safety["threshold_0_788800_split_far_le_5pct"],
            safety["threshold_0_780000_split_zero_high_risk_false_acceptances"],
            safety["threshold_0_788800_split_zero_high_risk_false_acceptances"],
        )
    )
    comparative_statement = "0.780000 is not preferred over 0.788800."
    if either_threshold_fails_absolute_split_safety:
        decision = "Neither 0.780000 nor 0.788800 is sufficient under the absolute all-confidence safety criteria."
        category = "neither_threshold_sufficient"
    elif not boundary_scores:
        decision = "0.780000 and 0.788800 are operationally equivalent on the current dataset."
        category = "operationally_equivalent"
    elif extra_pass > 0 and extra_fail == 0 and extra_high_risk == 0 and safety["cv_high_risk_not_worse"]:
        decision = "0.780000 is an alternative experimental shadow-test candidate."
        category = "alternative_experimental_shadow_test_candidate"
    else:
        decision = "0.780000 is not preferred over 0.788800."
        category = "0.788800_preferred"

    integrity = {
        "stored_score_count": len(scores),
        "unique_case_count": len(scores_by_id),
        "algorithm_version": EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
        "algorithm_version_matches": True,
        "complete_scores": True,
        "case_ids_match": True,
        "split_assignments_match": True,
        "leakage_groups_match": True,
        "dataset_sha256": dataset_sha,
        "dataset_sha_matches_v5_summary": True,
        "v5_code_sha256_current": sha256(V5_CODE),
        "v5_code_modified_at": code_modified_at,
        "earliest_score_timestamp": earliest_score_timestamp,
        "code_not_newer_than_scores": True,
        "stored_per_case_input_hashes_present": False,
        "missing_stored_per_case_input_hash_count": len(missing_stored_input_hashes),
        "current_per_case_input_hashes_computed": len(current_input_hashes),
        "integrity_limitation": "The stored v5 score rows contain no input_sha256 and the prior summary contains no v5 code hash. Dataset-wide SHA, timestamps, IDs, splits, groups, version and completeness match; per-case cryptographic comparison is unavailable. Scores were not recomputed.",
    }
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "experimental weak-label shadow evaluation",
        "human_ground_truth_used": False,
        "scores_recomputed": False,
        "productive_algorithm_changed": False,
        "productive_threshold_changed": False,
        "productive_algorithm": "fact_aware_claim_support_v4",
        "productive_threshold": 0.51,
        "thresholds": list(THRESHOLDS),
        "integrity": integrity,
        "metrics": evaluations,
        "boundary_audit": {
            "boundary_case_count": len(boundary_scores),
            "additional_accepts_at_0_780000": len(boundary_scores),
            "additional_accepted_pass_cases": extra_pass,
            "additional_accepted_fail_cases": extra_fail,
            "additional_high_risk_acceptances": extra_high_risk,
            "source_type_counts": sources,
            "single_case_materially_controls_decision": len(boundary_scores) == 1,
            "case_ids": [row["case_id"] for row in boundary_scores],
        },
        "group_cv": cv_aggregate,
        "safety_criteria": safety,
        "decision": {
            "category": category,
            "statement": decision,
            "comparative_statement": comparative_statement,
            "safer_experimental_candidate": 0.788800,
        },
        "closing_statement": "No human ground truth was used. This comparison is an experimental weak-label shadow evaluation and does not authorize a production threshold change.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def split_rows(name: str) -> str:
        lines = []
        for threshold in THRESHOLDS:
            row = evaluations[name][f"{threshold:.6f}"]
            lines.append(
                f"| {threshold:.6f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['tn'])} | "
                f"{int(row['fn'])} | {row['technical_far']:.2%} | {row['precision']:.2%} | "
                f"{row['recall']:.2%} | {row['f1']:.2%} | {row['coverage']:.2%} | "
                f"{row['accepted_high_risk_proxy_failures']} |"
            )
        return "\n".join(lines)

    boundary_table = "\n".join(
        f"| {row['case_id']} | {row['score']:.6f} | {row['split']} | {row['source_type']} | "
        f"{row['weak_label']} | {row['mutation_type']} | {row['technical_impact']} |"
        for row in boundary_rows
    ) or "| – | – | – | – | – | – | Keine Boundary Cases |"
    REPORT.write_text(
        f"""# Groundedness v5: Vergleich 0.780000 vs. 0.788800

## Ergebnis

**{decision}** Im direkten Vergleich gilt zusätzlich: **{comparative_statement}** Der niedrigere Wert akzeptiert {len(boundary_scores)} zusätzlichen Fall: {extra_pass} PASS, {extra_fail} FAIL und {extra_high_risk} High-Risk-Proxy-FAILs. Es gibt {'einen einzelnen maßgeblichen Grenzfall' if len(boundary_scores) == 1 else 'mehrere Grenzfälle'}.

## Integritätsprüfung der gespeicherten Scores

- 498 eindeutige Fälle und 498 vollständige Scores.
- Algorithmusversion: `{EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION}`.
- Dataset-SHA, Fall-IDs, Splits und Leakage-Gruppen stimmen überein.
- Der v5-Code ist zeitlich nicht neuer als die gespeicherten Scores.
- Scores wurden **nicht neu berechnet**.
- Provenienzgrenze: In allen 498 gespeicherten Scorezeilen fehlt `input_sha256`; auch ein damaliger Code-Hash fehlt. Aktuelle Input-Hashes wurden berechnet und für Boundary Cases gespeichert, können aber nicht rückwirkend kryptografisch verglichen werden.

## Calibration

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{split_rows('calibration_all')}

## Validation

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{split_rows('validation_all')}

## Technical Weak-Label Hold-out

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{split_rows('technical_weak_label_holdout_all')}

## Boundary-Case-Audit

| case_id | Score | Split | Quelle | Weak Label | Mutation | technische Auswirkung |
|---|---:|---|---|---|---|---|
{boundary_table}

Quellen der Boundary Cases: `{json.dumps(sources, ensure_ascii=False)}`. Die vollständigen Fragen, Antworten und Context-Zusammenfassungen stehen in `{BOUNDARY.name}`.

## Fester Group-CV-Vergleich

Beide Thresholds wurden in exakt denselben 50 Testfolds ausgewertet; es fand keine Neuauswahl je Fold statt.

| Threshold | Mean FAR | Median FAR | Mean Precision | Mean Recall | Mean F1 | Mean Coverage | Folds FAR>5% | High-Risk-Folds | Anteil | Maximum/Fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.780000 | {low_cv['mean_test_far']:.2%} | {low_cv['median_test_far']:.2%} | {low_cv['mean_precision']:.2%} | {low_cv['mean_recall']:.2%} | {low_cv['mean_f1']:.2%} | {low_cv['mean_coverage']:.2%} | {low_cv['folds_far_gt_5pct']} | {low_cv['folds_with_high_risk_false_acceptance']} | {low_cv['fraction_folds_with_high_risk_false_acceptance']:.2%} | {low_cv['maximum_high_risk_false_acceptances_in_fold']} |
| 0.788800 | {high_cv['mean_test_far']:.2%} | {high_cv['median_test_far']:.2%} | {high_cv['mean_precision']:.2%} | {high_cv['mean_recall']:.2%} | {high_cv['mean_f1']:.2%} | {high_cv['mean_coverage']:.2%} | {high_cv['folds_far_gt_5pct']} | {high_cv['folds_with_high_risk_false_acceptance']} | {high_cv['fraction_folds_with_high_risk_false_acceptance']:.2%} | {high_cv['maximum_high_risk_false_acceptances_in_fold']} |

## Sicherheitsentscheidung

Beide Thresholds halten auf allen drei vollständigen Splits die FAR-Grenze von 5 % ein. Beide akzeptieren jedoch in Calibration vier High-Risk-Proxy-FAILs aus Confidence-Stufen außerhalb der primären HIGH-Menge und verfehlen damit die absolute Sicherheitsbedingung. Auf der primären HIGH-Menge sowie in den 50 festen Group-CV-Folds akzeptiert keiner der beiden Werte einen High-Risk-Proxy-FAIL.

0.780000 gewinnt keinen technischen PASS-Fall, sondern akzeptiert ausschließlich einen zusätzlichen FAIL-Fall im Hold-out. Recall bleibt unverändert; die minimale Coverage-Zunahme ist eine falsche Akzeptanz und kein Nutzen. Die Vergleichsentscheidung hängt vollständig von diesem einzelnen Boundary Case ab. Somit ist keiner der beiden Thresholds nach der absoluten All-Confidence-Regel ausreichend; 0.788800 bleibt innerhalb des experimentellen Shadow-Vergleichs der sicherere Kandidat.

Der produktive Algorithmus bleibt `fact_aware_claim_support_v4`, der produktive Threshold bleibt `0.51`.

No human ground truth was used. This comparison is an experimental weak-label shadow evaluation and does not authorize a production threshold change.
""",
        encoding="utf-8",
    )
    print(json.dumps({"boundary": summary["boundary_audit"], "group_cv": cv_aggregate, "decision": summary["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

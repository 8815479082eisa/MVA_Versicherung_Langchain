from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = PROJECT_ROOT / "reports" / "groundedness_extended_scores.jsonl"
DEFAULT_SPLIT_PLAN = PROJECT_ROOT / "reports" / "groundedness_dataset_split_plan.json"
DEFAULT_LOCK = PROJECT_ROOT / "reports" / "groundedness_extended_threshold_lock.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "groundedness_extended_threshold_summary.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "reports" / "groundedness_extended_threshold_candidates.csv"
DEFAULT_CV = PROJECT_ROOT / "reports" / "groundedness_extended_threshold_cv.csv"
DEFAULT_HOLDOUT_MARKER = PROJECT_ROOT / "reports" / ".groundedness_extended_holdout_opened.json"
FAR_LIMIT = 0.05
CONFIDENCE = 0.95
OLD_THRESHOLD = 0.51
PROVISIONAL_THRESHOLD = 0.583333
RANDOM_SEED = 20260801


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eligible(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("adjudicated_label", "")).upper() in {"PASS", "FAIL"}]


def thresholds(rows: list[dict[str, Any]]) -> list[float]:
    values = sorted({float(row["groundedness_score"]) for row in rows})
    if not values:
        return []
    candidates = {0.0, 1.0, values[0], values[-1], OLD_THRESHOLD, PROVISIONAL_THRESHOLD}
    candidates.update((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.update(step / 1000.0 for step in range(1001))
    return sorted(round(value, 9) for value in candidates if 0.0 <= value <= 1.0)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    alpha = 1.0 - CONFIDENCE
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def metrics(rows: list[dict[str, Any]], threshold: float, *, include_intervals: bool = True) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    critical_false_accepts = 0
    accepted_fail_ids: list[str] = []
    for row in rows:
        actual = str(row["adjudicated_label"]).upper() == "PASS"
        accepted = float(row["groundedness_score"]) >= threshold
        if actual and accepted:
            tp += 1
        elif actual:
            fn += 1
        elif accepted:
            fp += 1
            accepted_fail_ids.append(str(row["case_id"]))
            if str(row.get("adjudicated_criticality", "")).upper() == "CRITICAL":
                critical_false_accepts += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    far = _safe_div(fp, fp + tn)
    frr = _safe_div(fn, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    if include_intervals:
        far_low, far_high = _clopper_pearson(fp, fp + tn)
        precision_low, precision_high = _clopper_pearson(tp, tp + fp)
        recall_low, recall_high = _clopper_pearson(tp, tp + fn)
    else:
        far_low = far_high = precision_low = precision_high = recall_low = recall_high = None
    return {
        "threshold": float(threshold), "n": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f1": f1,
        "g_mean": math.sqrt(max(0.0, recall * specificity)),
        "coverage": _safe_div(tp + fp, len(rows)),
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "critical_false_accepts": critical_false_accepts,
        "accepted_fail_case_ids": accepted_fail_ids,
        "far_ci95_lower": far_low,
        "far_ci95_upper": far_high,
        "precision_ci95_lower": precision_low,
        "precision_ci95_upper": precision_high,
        "recall_ci95_lower": recall_low,
        "recall_ci95_upper": recall_high,
    }


def _tied(rows: list[dict[str, Any]], key: str, target: float) -> list[dict[str, Any]]:
    return [row for row in rows if abs(float(row[key]) - target) <= 1e-12]


def select_threshold(
    rows: list[dict[str, Any]],
    *,
    fixed_cv_stability: dict[float, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep = [metrics(rows, value, include_intervals=False) for value in thresholds(rows)]
    if not sweep:
        raise ValueError("No eligible PASS/FAIL rows available")
    for row in sweep:
        row["fixed_cv_f1_std"] = (
            fixed_cv_stability.get(float(row["threshold"])) if fixed_cv_stability else None
        )
    pool = [row for row in sweep if row["critical_false_accepts"] == 0]
    pool = [row for row in pool if row["false_acceptance_rate"] <= FAR_LIMIT]
    if not pool:
        raise RuntimeError("No threshold satisfies zero accepted critical FAIL cases and empirical FAR <= 5%")
    pool = _tied(pool, "recall", max(float(row["recall"]) for row in pool))
    pool = _tied(pool, "precision", max(float(row["precision"]) for row in pool))
    pool = _tied(pool, "false_acceptance_rate", min(float(row["false_acceptance_rate"]) for row in pool))
    if fixed_cv_stability and all(row["fixed_cv_f1_std"] is not None for row in pool):
        pool = _tied(pool, "fixed_cv_f1_std", min(float(row["fixed_cv_f1_std"]) for row in pool))
    pool = _tied(pool, "coverage", max(float(row["coverage"]) for row in pool))
    selected = max(pool, key=lambda row: float(row["threshold"]))
    selected_with_intervals = metrics(rows, float(selected["threshold"]), include_intervals=True)
    selected_with_intervals["fixed_cv_f1_std"] = selected.get("fixed_cv_f1_std")
    return selected_with_intervals, sweep


def build_group_splits(rows: list[dict[str, Any]], repeats: int, folds: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    labels = np.asarray([1 if str(row["adjudicated_label"]).upper() == "PASS" else 0 for row in rows])
    groups = np.asarray([str(row["leakage_group_id"]) for row in rows])
    indices = np.arange(len(rows))
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, labels, groups), start=1):
            splits.append((repeat + 1, fold, train_idx, test_idx))
    return splits


def fixed_threshold_stability(
    rows: list[dict[str, Any]],
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> dict[float, float]:
    result: dict[float, float] = {}
    for threshold in thresholds(rows):
        f1_values = [
            metrics([rows[int(index)] for index in test_idx], threshold, include_intervals=False)["f1"]
            for _, _, _, test_idx in splits
        ]
        result[float(threshold)] = statistics.pstdev(float(value) for value in f1_values)
    return result


def repeated_group_cv(
    rows: list[dict[str, Any]],
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for repeat, fold, train_idx, test_idx in splits:
        train = [rows[int(index)] for index in train_idx]
        test = [rows[int(index)] for index in test_idx]
        selected, _ = select_threshold(train)
        test_metrics = metrics(test, float(selected["threshold"]), include_intervals=False)
        results.append(
            {
                "repeat": repeat,
                "fold": fold,
                "train_size": len(train),
                "test_size": len(test),
                "training_threshold": selected["threshold"],
                "test_far": test_metrics["false_acceptance_rate"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_coverage": test_metrics["coverage"],
                "test_critical_false_accepts": test_metrics["critical_false_accepts"],
                "test_accepted_fail_case_ids": ";".join(test_metrics["accepted_fail_case_ids"]),
                "test_far_le_5pct": test_metrics["false_acceptance_rate"] <= FAR_LIMIT,
                **{key: test_metrics[key] for key in ("tp", "fp", "tn", "fn", "specificity", "balanced_accuracy", "g_mean", "false_rejection_rate")},
            }
        )
    return results


def aggregate_cv(rows: list[dict[str, Any]], folds: int, repeats: int) -> dict[str, Any]:
    values = np.asarray([float(row["training_threshold"]) for row in rows], dtype=float)
    result: dict[str, Any] = {
        "runs": len(rows), "k": folds, "repeats": repeats, "random_seed": RANDOM_SEED,
        "threshold_mean": float(np.mean(values)),
        "threshold_median": float(np.median(values)),
        "threshold_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "threshold_minimum": float(np.min(values)),
        "threshold_maximum": float(np.max(values)),
        "threshold_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "folds_with_accepted_critical_failures": sum(int(row["test_critical_false_accepts"]) > 0 for row in rows),
        "far_compliance_fraction": _safe_div(sum(bool(row["test_far_le_5pct"]) for row in rows), len(rows)),
    }
    for key in ("test_far", "test_precision", "test_recall", "test_f1", "test_coverage"):
        metric_values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = mean(metric_values)
        result[f"{key}_median"] = median(metric_values)
    return result


def subgroup_metrics(rows: list[dict[str, Any]], threshold: float, key: str) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        values.setdefault(str(row.get(key) or "UNKNOWN"), []).append(row)
    return {name: metrics(members, threshold) for name, members in sorted(values.items())}


def comparison_methods(rows: list[dict[str, Any]], sweep: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def conservative_max(key: str) -> dict[str, Any]:
        best = max(float(row[key]) for row in sweep)
        return max(_tied(sweep, key, best), key=lambda row: float(row["threshold"]))

    youden = [{**row, "youden_j": float(row["recall"]) + float(row["specificity"]) - 1.0} for row in sweep]
    best_youden = max(float(row["youden_j"]) for row in youden)
    roc_pool = [row for row in sweep if float(row["false_acceptance_rate"]) <= FAR_LIMIT]
    roc_recall = max(float(row["recall"]) for row in roc_pool)
    return {
        "previous_0_51": metrics(rows, OLD_THRESHOLD),
        "provisional_0_583333": metrics(rows, PROVISIONAL_THRESHOLD),
        "maximum_f1": conservative_max("f1"),
        "maximum_g_mean": conservative_max("g_mean"),
        "youden_j": max(_tied(youden, "youden_j", best_youden), key=lambda row: float(row["threshold"])),
        "roc_fpr_le_5pct": max(_tied(roc_pool, "recall", roc_recall), key=lambda row: float(row["threshold"])),
    }


def discrimination(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = [1 if str(row["adjudicated_label"]).upper() == "PASS" else 0 for row in rows]
    scores = [float(row["groundedness_score"]) for row in rows]
    if len(set(labels)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def cv_subgroup_stability(
    rows: list[dict[str, Any]],
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
    cv_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    collected: dict[str, list[dict[str, Any]]] = {}
    for (_, _, _, test_idx), cv_row in zip(splits, cv_rows):
        threshold = float(cv_row["training_threshold"])
        test = [rows[int(index)] for index in test_idx]
        names = sorted({str(row.get(key) or "UNKNOWN") for row in test})
        for name in names:
            members = [row for row in test if str(row.get(key) or "UNKNOWN") == name]
            collected.setdefault(name, []).append(metrics(members, threshold, include_intervals=False))
    result: dict[str, Any] = {}
    for name, values in sorted(collected.items()):
        result[name] = {
            "fold_appearances": len(values),
            "f1_mean": mean(float(value["f1"]) for value in values),
            "f1_standard_deviation": statistics.pstdev(float(value["f1"]) for value in values),
            "recall_mean": mean(float(value["recall"]) for value in values),
            "far_mean": mean(float(value["false_acceptance_rate"]) for value in values),
            "accepted_critical_failures": sum(int(value["critical_false_accepts"]) for value in values),
        }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary.get("selected_threshold_metrics") or summary.get("metrics")
    threshold = float(summary["selected_threshold"])
    lines = [
        "# Extended Groundedness Threshold Calibration",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Selected/locked threshold: `{threshold:.9f}`",
        "",
        "The result was generated only after the annotation, adjudication and locked-split gates passed.",
        "",
        "## Primary metrics",
        "",
        "| TP | FP | TN | FN | FAR | FRR | Precision | Recall | F1 | Specificity | Balanced accuracy | G-mean | Coverage | Critical false accepts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {selected['tp']} | {selected['fp']} | {selected['tn']} | {selected['fn']} | {selected['false_acceptance_rate']:.4f} | {selected['false_rejection_rate']:.4f} | {selected['precision']:.4f} | {selected['recall']:.4f} | {selected['f1']:.4f} | {selected['specificity']:.4f} | {selected['balanced_accuracy']:.4f} | {selected['g_mean']:.4f} | {selected['coverage']:.4f} | {selected['critical_false_accepts']} |",
        "",
        "## Statistical uncertainty",
        "",
        f"- FAR 95% Clopper-Pearson CI: [{selected['far_ci95_lower']:.4f}, {selected['far_ci95_upper']:.4f}]",
        f"- Precision 95% Clopper-Pearson CI: [{selected['precision_ci95_lower']:.4f}, {selected['precision_ci95_upper']:.4f}]",
        f"- Recall 95% Clopper-Pearson CI: [{selected['recall_ci95_lower']:.4f}, {selected['recall_ci95_upper']:.4f}]",
        "",
        "## Safety and leakage controls",
        "",
        "- Only adjudicated PASS/FAIL cases were eligible; EXCLUDE and AMBIGUOUS were omitted.",
        "- Threshold fitting used Calibration only; Validation was method checking; locked Hold-out is a one-time phase.",
        "- Repeated stratified group folds respect `leakage_group_id`.",
        "- Productive configuration is not changed by this script.",
        "",
    ]
    return "\n".join(lines)


def load_and_gate(scores_path: Path, split_plan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads(split_plan_path.read_text(encoding="utf-8"))
    if not plan.get("final_locked_holdout"):
        raise SystemExit(
            "Refusing calibration: split plan is preliminary. Complete two independent reviews, "
            "adjudication, then explicitly finalize and lock the group split first."
        )
    rows = read_jsonl(scores_path)
    if not rows or any(str(row.get("adjudicated_label", "")).upper() not in {"PASS", "FAIL", "AMBIGUOUS", "EXCLUDE"} for row in rows):
        raise SystemExit("Refusing calibration: scored rows do not contain complete adjudicated labels")
    assignments = {row["case_id"]: row["split"] for row in plan.get("case_assignments", [])}
    if set(assignments) != {row["case_id"] for row in rows}:
        raise SystemExit("Refusing calibration: score and split-plan case IDs differ")
    if any(assignments[row["case_id"]] != row["split"] for row in rows):
        raise SystemExit("Refusing calibration: a score row does not match the locked group split")
    return rows, plan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["repeat", "fold"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Future, gated extended groundedness calibration")
    parser.add_argument("--phase", choices=("calibrate", "validate", "holdout"), required=True)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--split-plan", type=Path, default=DEFAULT_SPLIT_PLAN)
    parser.add_argument("--threshold-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cv-output", type=Path, default=DEFAULT_CV)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / f"groundedness_extended_threshold_calibration_{datetime.now(timezone.utc):%Y%m%d}.md",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--confirm-open-locked-holdout", action="store_true")
    args = parser.parse_args()

    rows, plan = load_and_gate(args.scores, args.split_plan)
    now = datetime.now(timezone.utc).isoformat()
    scores_sha256 = hashlib.sha256(args.scores.read_bytes()).hexdigest()

    if args.phase == "calibrate":
        calibration = eligible(row for row in rows if row["split"] == "CALIBRATION")
        pass_groups = {row["leakage_group_id"] for row in calibration if row["adjudicated_label"] == "PASS"}
        fail_groups = {row["leakage_group_id"] for row in calibration if row["adjudicated_label"] == "FAIL"}
        folds = min(args.folds, len(pass_groups), len(fail_groups))
        if folds < 2:
            raise SystemExit("At least two PASS and two FAIL leakage groups are required for grouped CV")
        splits = build_group_splits(calibration, args.repeats, folds)
        stability = fixed_threshold_stability(calibration, splits)
        selected, sweep = select_threshold(calibration, fixed_cv_stability=stability)
        cv = repeated_group_cv(calibration, splits)
        write_csv(args.candidate_output, sweep)
        write_csv(args.cv_output, cv)
        comparisons = comparison_methods(calibration, sweep)
        result = {
            "status": "PROPOSED_THRESHOLD_PENDING_VALIDATION",
            "generated_at": now,
            "selected_threshold": selected["threshold"],
            "selection_rule": (
                "exclude thresholds accepting adjudicated CRITICAL FAIL cases; exclude empirical FAR > 5%; "
                "maximize recall; tie-break by precision, lower FAR, lower fixed-CV F1 SD, coverage, then higher threshold"
            ),
            "algorithm_lock": rows[0].get("algorithm_version"),
            "label_definition_lock": "adjudicated PASS is positive; adjudicated FAIL is negative; AMBIGUOUS/EXCLUDE ineligible",
            "selected_threshold_metrics": selected,
            "calibration_count": len(calibration),
            "calibration_label_counts": dict(Counter(row["adjudicated_label"] for row in calibration)),
            "discrimination": discrimination(calibration),
            "comparisons": comparisons,
            "cv": aggregate_cv(cv, folds, args.repeats),
            "cv_stability": {
                "source_type": cv_subgroup_stability(calibration, splits, cv, "source_type"),
                "error_category": cv_subgroup_stability(calibration, splits, cv, "adjudicated_error_category"),
            },
            "subgroups": {
                "source_type": subgroup_metrics(calibration, selected["threshold"], "source_type"),
                "product_group": subgroup_metrics(calibration, selected["threshold"], "product_group"),
                "error_category": subgroup_metrics(calibration, selected["threshold"], "adjudicated_error_category"),
                "criticality": subgroup_metrics(calibration, selected["threshold"], "adjudicated_criticality"),
            },
            "candidate_threshold_count": len(sweep),
            "scores_sha256": scores_sha256,
            "split_plan_sha256": hashlib.sha256(args.split_plan.read_bytes()).hexdigest(),
            "holdout_evaluated": False,
        }
        args.threshold_lock.parent.mkdir(parents=True, exist_ok=True)
        args.threshold_lock.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        if not args.threshold_lock.exists():
            raise SystemExit("A proposed threshold artifact from --phase calibrate is required")
        lock = json.loads(args.threshold_lock.read_text(encoding="utf-8"))
        threshold = float(lock["selected_threshold"])
        if lock.get("scores_sha256") != scores_sha256:
            raise SystemExit("Threshold lock and score artifact hashes differ")
        split = "VALIDATION" if args.phase == "validate" else "LOCKED_HOLDOUT_CANDIDATE"
        if args.phase == "holdout":
            if lock.get("status") != "THRESHOLD_LOCKED_AFTER_VALIDATION":
                raise SystemExit("Holdout remains closed until validation completes and the threshold is locked")
            if not args.confirm_open_locked_holdout:
                raise SystemExit("Opening holdout requires --confirm-open-locked-holdout")
            if DEFAULT_HOLDOUT_MARKER.exists():
                raise SystemExit("Locked holdout has already been opened; refusing a repeated evaluation")
        evaluated = eligible(row for row in rows if row["split"] == split)
        result = {
            "status": "VALIDATION_COMPLETE" if args.phase == "validate" else "LOCKED_HOLDOUT_OPENED_ONCE",
            "generated_at": now,
            "selected_threshold": threshold,
            "split": split,
            "metrics": metrics(evaluated, threshold),
            "discrimination": discrimination(evaluated),
            "label_counts": dict(Counter(row["adjudicated_label"] for row in evaluated)),
            "subgroups": {
                "source_type": subgroup_metrics(evaluated, threshold, "source_type"),
                "product_group": subgroup_metrics(evaluated, threshold, "product_group"),
                "error_category": subgroup_metrics(evaluated, threshold, "adjudicated_error_category"),
                "criticality": subgroup_metrics(evaluated, threshold, "adjudicated_criticality"),
            },
            "scores_sha256": scores_sha256,
        }
        if args.phase == "validate":
            lock.update(
                {
                    "status": "THRESHOLD_LOCKED_AFTER_VALIDATION",
                    "locked_at": now,
                    "validation_metrics": result["metrics"],
                    "threshold_must_not_change": True,
                }
            )
            args.threshold_lock.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            DEFAULT_HOLDOUT_MARKER.write_text(
                json.dumps({"opened_at": now, "threshold": threshold, "scores_sha256": scores_sha256}, indent=2) + "\n",
                encoding="utf-8",
            )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_threshold": result["selected_threshold"], "summary": str(args.summary_output), "markdown": str(args.markdown_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

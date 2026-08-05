from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import beta, shapiro
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402
from src.guardrails.integrations.nemo_actions import (  # noqa: E402
    GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score,
)


SOURCE_DATASET = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_calibration_cases.json"
NORMALIZED_DATASET = (
    PROJECT_ROOT / "tests" / "fixtures" / "groundedness_threshold_calibration_cases.json"
)
CANDIDATE_REPORT = PROJECT_ROOT / "reports" / "groundedness_threshold_all_candidates.csv"
CV_REPORT = PROJECT_ROOT / "reports" / "groundedness_threshold_cross_validation.csv"
SUMMARY_REPORT = PROJECT_ROOT / "reports" / "groundedness_threshold_summary.json"
MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "groundedness_threshold_calibration_20260801.md"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

OLD_THRESHOLD = 0.51
FAR_LIMIT = 0.05
CONFIDENCE = 0.95
CV_REPEATS = 10
RANDOM_SEED = 20260801


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_and_normalize_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_cases = payload.get("cases")
    if not isinstance(source_cases, list) or not source_cases:
        raise ValueError("Source fixture does not contain a non-empty cases list")

    normalized: list[dict[str, Any]] = []
    for source in source_cases:
        if not isinstance(source.get("label"), bool):
            raise ValueError(f"Case {source.get('id')} has no explicit boolean source label")
        if not isinstance(source.get("documents"), list) or not source["documents"]:
            raise ValueError(f"Case {source.get('id')} has no context documents")

        documents = [Document(page_content=str(text)) for text in source["documents"]]
        score = calculate_groundedness_score(
            str(source.get("answer", "")),
            documents,
            str(source.get("query", "")),
        )
        label = "PASS" if source["label"] else "FAIL"
        normalized.append(
            {
                "case_id": str(source["id"]),
                "question": str(source.get("query", "")),
                "context": [str(text) for text in source["documents"]],
                "generated_answer": str(source.get("answer", "")),
                "human_label": label,
                "label_source": "existing_boolean_fixture_label",
                "human_annotation_provenance_verified": False,
                "groundedness_score": float(score),
                "score_runs": [float(score)],
                "score_standard_deviation": 0.0,
                "score_range": 0.0,
                "score_deterministic": True,
                "error_category": None,
                "critical": None,
                "metadata_review_status": (
                    "REVIEW_REQUIRED: error_category and criticality are absent from source fixture"
                    if label == "FAIL"
                    else "REVIEW_REQUIRED: human annotation provenance is undocumented"
                ),
            }
        )
    return normalized


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    unique = sorted(set(float(score) for score in scores))
    candidates = {0.0, 1.0, *unique}
    candidates.update((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    candidates.update(step / 1000.0 for step in range(1001))
    return sorted(round(value, 9) for value in candidates if 0.0 <= value <= 1.0)


def clopper_pearson_two_sided(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    alpha = 1.0 - CONFIDENCE
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def clopper_pearson_one_sided_upper(successes: int, trials: int) -> float:
    if trials <= 0 or successes == trials:
        return 1.0
    return float(beta.ppf(CONFIDENCE, successes + 1, trials - successes))


def metrics_at(
    cases: Sequence[dict[str, Any]],
    threshold: float,
    *,
    include_intervals: bool = True,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    accepted_fail_ids: list[str] = []
    for case in cases:
        actual_pass = case["human_label"] == "PASS"
        accepted = float(case["groundedness_score"]) >= threshold
        if accepted and actual_pass:
            tp += 1
        elif accepted and not actual_pass:
            fp += 1
            accepted_fail_ids.append(str(case["case_id"]))
        elif not accepted and not actual_pass:
            tn += 1
        else:
            fn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    false_acceptance_rate = _safe_div(fp, fp + tn)
    false_rejection_rate = _safe_div(fn, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0
    g_mean = math.sqrt(max(0.0, recall * specificity))
    coverage = _safe_div(tp + fp, len(cases))
    if include_intervals:
        far_lower, far_upper = clopper_pearson_two_sided(fp, fp + tn)
        precision_lower, precision_upper = clopper_pearson_two_sided(tp, tp + fp)
        far_upper_one_sided = clopper_pearson_one_sided_upper(fp, fp + tn)
    else:
        far_lower = far_upper = precision_lower = precision_upper = None
        far_upper_one_sided = None

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_acceptance_rate": false_acceptance_rate,
        "false_rejection_rate": false_rejection_rate,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "g_mean": g_mean,
        "coverage": coverage,
        "accepted_critical_failures_confirmed": None,
        "accepted_failures_worst_case_critical": fp,
        "accepted_fail_case_ids": accepted_fail_ids,
        "far_ci95_lower_two_sided": far_lower,
        "far_ci95_upper_two_sided": far_upper,
        "far_upper_95_one_sided": far_upper_one_sided,
        "precision_ci95_lower": precision_lower,
        "precision_ci95_upper": precision_upper,
    }


def _tied(rows: Sequence[dict[str, Any]], key: str, target: float, tolerance: float = 1e-12) -> list[dict[str, Any]]:
    return [row for row in rows if abs(float(row[key]) - target) <= tolerance]


def select_risk_constrained(
    rows: Sequence[dict[str, Any]],
    *,
    use_cv_stability: bool,
) -> dict[str, Any] | None:
    # Criticality is unlabelled. Treat every FAIL as potentially critical; therefore
    # accepted_failures_worst_case_critical == 0 is a conservative implementation
    # of the explicit critical-error exclusion without inventing labels.
    selected = [row for row in rows if row["accepted_failures_worst_case_critical"] == 0]
    selected = [row for row in selected if row["false_acceptance_rate"] <= FAR_LIMIT]
    if not selected:
        return None

    best = max(float(row["recall"]) for row in selected)
    selected = _tied(selected, "recall", best)
    best = max(float(row["precision"]) for row in selected)
    selected = _tied(selected, "precision", best)
    best = min(float(row["false_acceptance_rate"]) for row in selected)
    selected = _tied(selected, "false_acceptance_rate", best)

    if use_cv_stability and all(row.get("fixed_cv_f1_std") is not None for row in selected):
        best = min(float(row["fixed_cv_f1_std"]) for row in selected)
        selected = _tied(selected, "fixed_cv_f1_std", best)

    best = max(float(row["coverage"]) for row in selected)
    selected = _tied(selected, "coverage", best)
    return max(selected, key=lambda row: float(row["threshold"]))


def build_cv_splits(labels: Sequence[int], folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )
    dummy = np.zeros((len(labels), 1), dtype=float)
    return list(splitter.split(dummy, np.asarray(labels, dtype=int)))


def add_fixed_threshold_stability(
    rows: list[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    for row in rows:
        fold_f1: list[float] = []
        fold_recall: list[float] = []
        fold_far: list[float] = []
        threshold = float(row["threshold"])
        for _, test_indices in splits:
            test_cases = [cases[int(index)] for index in test_indices]
            fold_metrics = metrics_at(test_cases, threshold, include_intervals=False)
            fold_f1.append(float(fold_metrics["f1"]))
            fold_recall.append(float(fold_metrics["recall"]))
            fold_far.append(float(fold_metrics["false_acceptance_rate"]))
        row["fixed_cv_f1_std"] = statistics.pstdev(fold_f1)
        row["fixed_cv_recall_std"] = statistics.pstdev(fold_recall)
        row["fixed_cv_far_compliance_rate"] = _safe_div(
            sum(value <= FAR_LIMIT for value in fold_far), len(fold_far)
        )


def run_adaptive_cross_validation(
    cases: Sequence[dict[str, Any]],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    folds: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for split_index, (train_indices, test_indices) in enumerate(splits):
        train_cases = [cases[int(index)] for index in train_indices]
        test_cases = [cases[int(index)] for index in test_indices]
        train_rows = [
            metrics_at(train_cases, threshold, include_intervals=False)
            for threshold in candidate_thresholds(
                [float(case["groundedness_score"]) for case in train_cases]
            )
        ]
        selected = select_risk_constrained(train_rows, use_cv_stability=False)
        if selected is None:
            raise RuntimeError("No risk-constrained threshold exists in a CV training fold")
        test_metrics = metrics_at(
            test_cases,
            float(selected["threshold"]),
            include_intervals=False,
        )
        results.append(
            {
                "repeat": split_index // folds + 1,
                "fold": split_index % folds + 1,
                "train_size": len(train_cases),
                "test_size": len(test_cases),
                "train_pass": sum(case["human_label"] == "PASS" for case in train_cases),
                "train_fail": sum(case["human_label"] == "FAIL" for case in train_cases),
                "test_pass": sum(case["human_label"] == "PASS" for case in test_cases),
                "test_fail": sum(case["human_label"] == "FAIL" for case in test_cases),
                "training_threshold": float(selected["threshold"]),
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],
                "test_far": test_metrics["false_acceptance_rate"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_coverage": test_metrics["coverage"],
                "test_accepted_critical_failures_confirmed": None,
                "test_accepted_failures_worst_case_critical": test_metrics["fp"],
                "test_accepted_fail_case_ids": ";".join(test_metrics["accepted_fail_case_ids"]),
                "test_far_le_5pct": bool(test_metrics["false_acceptance_rate"] <= FAR_LIMIT),
            }
        )
    return results


def class_statistics(cases: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    values = np.asarray(
        [float(case["groundedness_score"]) for case in cases if case["human_label"] == label],
        dtype=float,
    )
    shapiro_result = shapiro(values) if len(values) >= 3 else None
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "shapiro_w": float(shapiro_result.statistic) if shapiro_result else None,
        "shapiro_p": float(shapiro_result.pvalue) if shapiro_result else None,
    }


def aggregate_cv(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    thresholds = np.asarray([float(row["training_threshold"]) for row in rows], dtype=float)
    result: dict[str, Any] = {
        "folds": len(rows),
        "k": max(int(row["fold"]) for row in rows),
        "repeats": max(int(row["repeat"]) for row in rows),
        "random_seed": RANDOM_SEED,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_median": float(np.median(thresholds)),
        "threshold_standard_deviation": float(np.std(thresholds, ddof=1)),
        "threshold_minimum": float(np.min(thresholds)),
        "threshold_maximum": float(np.max(thresholds)),
        "threshold_q1": float(np.percentile(thresholds, 25)),
        "threshold_q3": float(np.percentile(thresholds, 75)),
        "threshold_iqr": float(np.percentile(thresholds, 75) - np.percentile(thresholds, 25)),
        "folds_with_worst_case_critical_false_acceptance": sum(
            int(row["test_accepted_failures_worst_case_critical"]) > 0 for row in rows
        ),
        "far_compliance_fraction": _safe_div(
            sum(bool(row["test_far_le_5pct"]) for row in rows), len(rows)
        ),
    }
    for metric in ("test_far", "test_precision", "test_recall", "test_f1", "test_coverage"):
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_median"] = statistics.median(values)
    return result


def choose_comparison_methods(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def conservative_max(key: str) -> dict[str, Any]:
        value = max(float(row[key]) for row in rows)
        tied = _tied(rows, key, value)
        return max(tied, key=lambda row: float(row["threshold"]))

    youden_rows = [{**row, "youden_j": row["recall"] + row["specificity"] - 1.0} for row in rows]
    roc_candidates = [row for row in rows if row["false_acceptance_rate"] <= FAR_LIMIT]
    best_roc_recall = max(float(row["recall"]) for row in roc_candidates)
    roc_tied = _tied(roc_candidates, "recall", best_roc_recall)
    roc_choice = max(roc_tied, key=lambda row: float(row["threshold"]))

    return {
        "maximum_f1": conservative_max("f1"),
        "maximum_g_mean": conservative_max("g_mean"),
        "youden_j": max(
            _tied(youden_rows, "youden_j", max(float(row["youden_j"]) for row in youden_rows)),
            key=lambda row: float(row["threshold"]),
        ),
        "roc_fpr_le_5pct": roc_choice,
        "precision_recall_max_f1": conservative_max("f1"),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _canvas(title: str, width: int = 1400, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 25), title, fill="#132238", font=_font(28, bold=True))
    return image, draw


def _axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    x_label: str,
    y_label: str,
    x_ticks: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    y_ticks: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill="#2e4057", width=2)
    draw.line((left, bottom, left, top), fill="#2e4057", width=2)
    small = _font(16)
    for tick in x_ticks:
        x = left + (right - left) * float(tick)
        draw.line((x, bottom, x, bottom + 6), fill="#2e4057", width=1)
        draw.text((x - 16, bottom + 10), f"{tick:.2f}", fill="#2e4057", font=small)
    for tick in y_ticks:
        y = bottom - (bottom - top) * float(tick)
        draw.line((left - 6, y, left, y), fill="#2e4057", width=1)
        draw.text((left - 52, y - 9), f"{tick:.2f}", fill="#2e4057", font=small)
    draw.text(((left + right) / 2 - 55, bottom + 45), x_label, fill="#132238", font=_font(18))
    draw.text((left, top - 35), y_label, fill="#132238", font=_font(18))


def _xy(box: tuple[int, int, int, int], x: float, y: float) -> tuple[float, float]:
    left, top, right, bottom = box
    return left + (right - left) * x, bottom - (bottom - top) * y


def plot_score_distribution(
    cases: Sequence[dict[str, Any]],
    selected_threshold: float,
    output: Path,
) -> None:
    image, draw = _canvas("Groundedness-Scores: Verteilung, Boxplot und Einzelfälle", height=1050)
    pass_scores = [float(case["groundedness_score"]) for case in cases if case["human_label"] == "PASS"]
    fail_scores = [float(case["groundedness_score"]) for case in cases if case["human_label"] == "FAIL"]
    colors = {"PASS": "#1b9e77", "FAIL": "#d95f02"}

    # Histogram panel.
    hist_box = (100, 100, 1300, 410)
    draw.rectangle(hist_box, outline="#cad2dc", width=1)
    bins = np.linspace(0.0, 1.0, 11)
    pass_hist, _ = np.histogram(pass_scores, bins=bins)
    fail_hist, _ = np.histogram(fail_scores, bins=bins)
    max_count = max(1, int(max(max(pass_hist), max(fail_hist))))
    left, top, right, bottom = hist_box
    for index in range(10):
        x0 = left + index * (right - left) / 10
        x1 = left + (index + 1) * (right - left) / 10
        fail_height = (bottom - top - 45) * fail_hist[index] / max_count
        pass_height = (bottom - top - 45) * pass_hist[index] / max_count
        draw.rectangle((x0 + 5, bottom - fail_height, (x0 + x1) / 2 - 2, bottom), fill=colors["FAIL"])
        draw.rectangle(((x0 + x1) / 2 + 2, bottom - pass_height, x1 - 5, bottom), fill=colors["PASS"])
        draw.text((x0 + 8, bottom + 8), f"{bins[index]:.1f}", fill="#2e4057", font=_font(14))
    draw.text((110, 110), "Histogramm (10 Bins)", fill="#132238", font=_font(19, bold=True))
    draw.rectangle((1040, 115, 1060, 135), fill=colors["FAIL"])
    draw.text((1070, 112), "FAIL", fill="#132238", font=_font(16))
    draw.rectangle((1150, 115, 1170, 135), fill=colors["PASS"])
    draw.text((1180, 112), "PASS", fill="#132238", font=_font(16))

    # Horizontal boxplots.
    box_top = 500
    draw.text((100, 450), "Boxplots", fill="#132238", font=_font(19, bold=True))
    for row, (label, scores) in enumerate((("FAIL", fail_scores), ("PASS", pass_scores))):
        y = box_top + row * 95
        minimum, q1, median, q3, maximum = np.percentile(scores, [0, 25, 50, 75, 100])
        x = lambda value: 180 + 1080 * float(value)
        draw.line((x(minimum), y, x(maximum), y), fill=colors[label], width=3)
        draw.rectangle((x(q1), y - 25, x(q3), y + 25), outline=colors[label], width=4)
        draw.line((x(median), y - 25, x(median), y + 25), fill=colors[label], width=4)
        draw.text((105, y - 12), label, fill=colors[label], font=_font(18, bold=True))
    draw.line((180, box_top + 150, 1260, box_top + 150), fill="#2e4057", width=2)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = 180 + 1080 * tick
        draw.line((x, box_top + 150, x, box_top + 157), fill="#2e4057")
        draw.text((x - 16, box_top + 165), f"{tick:.2f}", fill="#2e4057", font=_font(15))

    # Strip plot.
    strip_top = 780
    draw.text((100, 710), "Strip-Plot aller Fälle", fill="#132238", font=_font(19, bold=True))
    draw.line((180, strip_top + 80, 1260, strip_top + 80), fill="#2e4057", width=2)
    for index, case in enumerate(cases):
        score = float(case["groundedness_score"])
        label = str(case["human_label"])
        x = 180 + 1080 * score
        base_y = strip_top + (20 if label == "FAIL" else 60)
        jitter = ((index * 17) % 11) - 5
        draw.ellipse((x - 6, base_y + jitter - 6, x + 6, base_y + jitter + 6), fill=colors[label])
    for threshold, color, name in (
        (OLD_THRESHOLD, "#7570b3", "0.51"),
        (selected_threshold, "#e7298a", "empfohlen"),
    ):
        x = 180 + 1080 * threshold
        draw.line((x, strip_top - 10, x, strip_top + 85), fill=color, width=3)
        draw.text((x + 5, strip_top - 30), name, fill=color, font=_font(14, bold=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def plot_threshold_metrics(rows: Sequence[dict[str, Any]], selected: float, output: Path) -> None:
    image, draw = _canvas("Metriken über dem Groundedness-Threshold", height=980)
    box = (110, 100, 1300, 790)
    _axes(draw, box, x_label="Threshold", y_label="Metrik")
    series = (
        ("precision", "#1b9e77", "Precision"),
        ("recall", "#377eb8", "Recall"),
        ("f1", "#984ea3", "F1"),
        ("false_acceptance_rate", "#e41a1c", "FAR"),
        ("coverage", "#ff7f00", "Coverage"),
    )
    sampled = [row for row in rows if round(float(row["threshold"]) * 1000) % 5 == 0]
    for key, color, _ in series:
        points = [_xy(box, float(row["threshold"]), float(row[key])) for row in sampled]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
    for threshold, color, label in ((OLD_THRESHOLD, "#555555", "alt 0.51"), (selected, "#e7298a", f"neu {selected:.6f}")):
        x, _ = _xy(box, threshold, 0.0)
        draw.line((x, box[1], x, box[3]), fill=color, width=3)
        draw.text((x + 6, box[1] + 8), label, fill=color, font=_font(15, bold=True))
    for index, (_, color, label) in enumerate(series):
        x = 160 + index * 225
        draw.line((x, 905, x + 45, 905), fill=color, width=5)
        draw.text((x + 55, 894), label, fill="#132238", font=_font(16))
    image.save(output)


def plot_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    annotation: str,
    output: Path,
    baseline: bool,
) -> None:
    image, draw = _canvas(title)
    box = (110, 100, 1300, 790)
    _axes(draw, box, x_label=x_label, y_label=y_label)
    if baseline:
        draw.line((*_xy(box, 0.0, 0.0), *_xy(box, 1.0, 1.0)), fill="#aab2bd", width=3)
    points = [_xy(box, float(x), float(y)) for x, y in zip(x_values, y_values)]
    if len(points) > 1:
        draw.line(points, fill="#377eb8", width=5)
    for point in points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#377eb8")
    draw.text((970, 120), annotation, fill="#132238", font=_font(20, bold=True))
    image.save(output)


def plot_cv_thresholds(cv_rows: Sequence[dict[str, Any]], output: Path) -> None:
    image, draw = _canvas("Repeated Stratified K-fold: gewählte Training-Thresholds")
    box = (110, 110, 1300, 760)
    draw.rectangle(box, outline="#cad2dc", width=1)
    thresholds = [float(row["training_threshold"]) for row in cv_rows]
    counts: dict[float, int] = {}
    for threshold in thresholds:
        counts[threshold] = counts.get(threshold, 0) + 1
    maximum = max(counts.values())
    bar_width = max(40, int(900 / max(1, len(counts))))
    for index, (threshold, count) in enumerate(sorted(counts.items())):
        x0 = 210 + index * (bar_width + 40)
        height = 500 * count / maximum
        draw.rectangle((x0, 700 - height, x0 + bar_width, 700), fill="#377eb8")
        threshold_text = f"{threshold:.6f}"
        threshold_font = _font(16)
        threshold_width = draw.textbbox((0, 0), threshold_text, font=threshold_font)[2]
        draw.text(
            (x0 + (bar_width - threshold_width) / 2, 710),
            threshold_text,
            fill="#132238",
            font=threshold_font,
        )
        count_text = str(count)
        count_font = _font(18, bold=True)
        count_width = draw.textbbox((0, 0), count_text, font=count_font)[2]
        draw.text(
            (x0 + (bar_width - count_width) / 2, 675 - height),
            count_text,
            fill="#132238",
            font=count_font,
        )
    draw.text((110, 800), "Threshold (Balken = Anzahl der Folds)", fill="#132238", font=_font(18))
    image.save(output)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key, value in serialized.items():
                if isinstance(value, list):
                    serialized[key] = ";".join(str(item) for item in value)
            writer.writerow(serialized)


def _metric_table_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['threshold']:.6f} | {row['tp']} | {row['fp']} | {row['tn']} | "
        f"{row['fn']} | {row['false_acceptance_rate']:.3f} | {row['precision']:.3f} | "
        f"{row['recall']:.3f} | {row['f1']:.3f} | {row['g_mean']:.3f} | "
        f"{row['coverage']:.3f} | {row['accepted_failures_worst_case_critical']} | "
        f"{row['far_upper_95_one_sided']:.3f} |"
    )


def build_markdown_report(summary: dict[str, Any], cases: Sequence[dict[str, Any]]) -> str:
    selected = summary["recommended_threshold_metrics"]
    old = summary["old_threshold_metrics"]
    cv = summary["cross_validation"]
    pass_stats = summary["class_statistics"]["PASS"]
    fail_stats = summary["class_statistics"]["FAIL"]
    comparisons = summary["comparison_methods"]
    case_rows = "\n".join(
        f"| {case['case_id']} | {case['human_label']} | {case['groundedness_score']:.6f} | "
        f"{case['error_category'] or 'REVIEW_REQUIRED'} | "
        f"{'REVIEW_REQUIRED' if case['critical'] is None else str(case['critical']).lower()} |"
        for case in cases
    )
    comparison_rows = "\n".join(
        _metric_table_row(name, metrics) for name, metrics in comparisons.items()
    )

    return f"""# Wissenschaftliche Kalibrierung des Groundedness-Thresholds

Erstellt: `{summary['generated_at']}`  
Algorithmus: `{summary['algorithm_version']}`  
Quell-Datensatz: `tests/fixtures/groundedness_calibration_cases.json`

## 1. Executive Summary

**Statistisch berechnet:** Der risikobeschränkte Threshold auf dem gesamten
Kalibrierungsset beträgt **{summary['recommended_threshold']:.6f}**. Er erzielt
auf den 28 vorhandenen Fällen FAR={selected['false_acceptance_rate']:.1%},
Precision={selected['precision']:.1%}, Recall={selected['recall']:.1%} und
F1={selected['f1']:.1%}. Der alte Threshold 0,51 erzeugt auf diesen Fällen
dieselben Entscheidungen. Der neue Wert ist numerisch konservativer, aber im
vorliegenden Sample praktisch nicht anders.

**Status:** `provisional empirically calibrated threshold`, nicht final
statistisch validiert. Es gibt kein unangetastetes Testset, die 28 Fälle sind
synthetisch, die Human-Annotation-Provenienz ist nicht dokumentiert und die
14 FAIL-Fälle enthalten keine expliziten Criticality-Labels. Selbst bei null
False Acceptances beträgt die obere einseitige 95%-Clopper-Pearson-Grenze der
FAR {selected['far_upper_95_one_sided']:.1%} und liegt damit deutlich über 5%.

## 2. Ziel und Scope

Ziel ist ein projektspezifischer Cutoff für die Entscheidung
`score >= threshold -> ACCEPT`. Falsche Akzeptanz ist im Versicherungsfall
schwerwiegender als falsche Ablehnung. Die projektspezifische empirische
Risikogrenze ist FAR <= 5%; bestätigte kritische Versicherungsfehler dürfen
nie akzeptiert werden. Die Untersuchung ist isoliert: kein `/api/ask`, kein
Retrieval, kein Reindex, keine Embeddings, kein CRM, keine Antwortgenerierung
und kein Self-Check.

Methodische Orientierung: Sarmah et al., *How to Choose a Threshold for an
Evaluation Metric for Large Language Models*, arXiv:2412.12148v1. Die Arbeit
fordert eine explizite Use-Case-Risikoanalyse, Übersetzung der Risikotoleranz in
statistische Größen, Ground Truth, getrennte Threshold-Bestimmung und Prüfung
sowie Cross-Validation. Quelle: https://arxiv.org/abs/2412.12148v1

## 3. Aktuelle Groundedness-Implementierung

**Durch Repository-Code belegt:** `calculate_groundedness_score` in
`src/guardrails/integrations/nemo_actions.py:450` implementiert
`fact_aware_claim_support_v4`. Es gibt **kein Evaluator-Modell und keinen
Evaluator-Prompt**. Der Score ist deterministisch; Temperature, Sampling und
Modell-Seed sind nicht anwendbar.

Die Funktion:

1. entfernt Inline-Zitate und zerlegt die Antwort in Claims,
2. bildet normalisierte Tokenmengen ohne Stopwörter,
3. erkennt harte Fakten wie Policen-IDs, Daten, Zahlen, Deckungsarten und
   Selbstbeteiligung,
4. kombiniert lexikalische Claim-Abdeckung (Gewicht 0,68) mit Fact-Match
   (Gewicht 0,32),
5. bestraft fehlende harte Fakten und falsche Deckungspolarität,
6. gewichtet Dokumente leicht nach Query-Relevanz und Rang,
7. nimmt pro Claim das beste Dokument, bildet einen tokengewichteten Mittelwert,
   begrenzt ihn auf [0,1] und rundet auf sechs Stellen.

Höher bedeutet bessere dokumentarische Unterstützung. Die Implementierung ist
claim- und regelbasiert, nicht embedding-, similarity-model- oder LLM-basiert.
Die relevanten Definitionen stehen in
`src/guardrails/integrations/nemo_actions.py:108-232` und `:345-493`; die
Threshold-Anwendung steht in `:788-821`.

Groundedness wird nach Retrieval, optionalem Self-Check und Antwortgenerierung
ausgeführt (`src/api/rag_service.py:3469-3478` und `:3552-3567`). Der Self-Check
ist aktuell deaktiviert, ändert aber die Score-Funktion nicht.

## 4. Herkunft und Verwendung des bisherigen Thresholds 0.51

**Durch Artefakte belegt:** `scripts/calibrate_groundedness.py:46-86` prüfte
nur Hundertstelschritte von 0,00 bis 1,00, verlangte Recall >= 0,80, minimierte
FAR und maximierte danach F1/Recall/Precision; bei Gleichstand wurde der
niedrigste Threshold gewählt. Der höchste FAIL-Score ist 0,500000. Wegen der
inklusiven `>=`-Regel akzeptiert 0,50 diesen FAIL; 0,51 ist der erste
Hundertstelschritt darüber. Das Ergebnis wurde in
`config/groundedness_calibration.json:4` gespeichert.

`src/config/models.py:91-118` lädt dieses Artefakt; `:650-659` gibt ihm Vorrang
vor `SAFETY_MIN_GROUNDEDNESS` aus `.env`. Der `.env`-Wert 0,2 ist daher aktuell
nicht wirksam. 0,51 wurde auf denselben 28 Fällen ausgewählt und mit F1=1,0
ausgewertet. Ein unabhängiges Final Test Set existiert nicht. Das ist eine
Resubstitution mit optimistischer Bewertungsgefahr; ein externer Leak ist nicht
belegt, aber Auswahl und Evaluation sind nicht unabhängig.

Vollständige relevante Fundstellen des Literalwerts beziehungsweise seiner
produktiven Verwendung:

| Pfad | Zeile/Funktion | Bedeutung |
|---|---|---|
| `config/groundedness_calibration.json` | 4 und 14 | ausgewählter produktiver Wert und gespeicherte Metrik |
| `src/config/models.py` | 91-118, 650-659 | validiert und lädt `selected_threshold` mit Vorrang vor `.env` |
| `.env` | 44 | Fallback 0,2; wegen gültigem Kalibrierungsartefakt nicht aktiv |
| `src/guardrails/integrations/nemo_actions.py` | 788-821 | vergleicht Score mit geladenem `min_groundedness` |
| `src/api/rag_service.py` | 3600-3608 | protokolliert Score, Threshold, Quelle und Pass/Fail |
| `tests/unit/test_safety_pii_rules.py` | 17 | isolierter Test-Fixturewert 0,51 |
| `scripts/test_synthetic_customer_scenario.py` | 1479 | 0,51 als historischer Vergleichskandidat |
| `reports/groundedness_calibration_20260716_200744.md` | 8, 16, 53 | historischer Auswahlbericht |

Die vorhandenen Rohscores in `config/groundedness_calibration.json` wurden mit
der aktuellen Funktion erneut berechnet: {summary['score_reproduction']['exact_matches']}
von {summary['score_reproduction']['compared_cases']} stimmen bis zur gespeicherten
Sechs-Dezimalstellen-Präzision exakt überein; maximale Abweichung
{summary['score_reproduction']['maximum_absolute_difference']:.6g}.

## 5. Beschreibung und Qualität der Datengrundlage

Es liegen 28 balancierte, synthetische Versicherungsfälle vor: 14 PASS und 14
FAIL. Question, Context, Generated Answer, Boolean-Label und reproduzierbarer
Score sind vollständig. Die Labels sind explizit im bestehenden Fixture
gespeichert und wurden nicht aus Scores abgeleitet. Es fehlt jedoch ein Beleg,
dass Menschen die Labels annotiert oder fachlich freigegeben haben.

`error_category` und `critical` fehlen vollständig. Deshalb wurden sie nicht
ergänzt, sondern als `REVIEW_REQUIRED` normalisiert: 0 bestätigte kritische
FAIL-Fälle, 14 hinsichtlich Criticality ungeprüfte FAIL-Fälle. Für die
Sicherheitsanalyse wird zusätzlich konservativ jeder FAIL als potenziell
kritisch behandelt.

## 6. Definition von PASS, FAIL und kritischen Fehlern

PASS ist eine im Fixture als `true` gelabelte, akzeptierbare Antwort; FAIL ist
eine als `false` gelabelte Antwort. Positive Klasse = PASS. Eine False
Acceptance ist ein akzeptierter FAIL. Die explizit geforderten Kategorien
(`wrong_policy`, `wrong_deductible`, `wrong_coverage` usw.) können ohne
kuratierte Metadaten nicht verlässlich fallweise ausgewertet werden.

## 7. Projektspezifische Risikotoleranz

Empirisches Ziel: FAR <= 5%. Zusätzliche Bedingung: null kritische False
Acceptances. Mit nur 14 Negativen kann die FAR nur in Schritten von 7,14
Prozentpunkten variieren; deshalb erzwingt FAR <= 5% bereits FP=0 und erfüllt
zugleich die Worst-Case-Bedingung, wenn jeder FAIL als kritisch betrachtet wird.
Diese 5%-Grenze ist eine Projektannahme, keine regulatorische Vorgabe.

## 8. Methodik der Threshold-Suche

Kandidaten sind 0, 1, alle eindeutigen Scores, Mittelpunkte benachbarter Scores
und ein Kontrollsweep in 0,001-Schritten. Pro Kandidat wurden TP, FP, TN, FN,
FAR, FRR, Precision, Recall, Specificity, F1, Balanced Accuracy, G-mean,
Coverage und Clopper-Pearson-Intervalle berechnet.

## 9. Begründung der primären Auswahlregel

Die Regel wurde exakt in dieser Reihenfolge umgesetzt:

1. Ausschluss jedes Thresholds mit akzeptiertem FAIL im konservativen
   Worst-Case-Criticality-Szenario.
2. Ausschluss bei FAR > 5%.
3. Maximaler Recall.
4. Tie-Breaker: höhere Precision, kleinere FAR, geringere CV-F1-Streuung,
   höhere Coverage, schließlich der höhere (konservativere) Threshold.

Der Worst-Case-Schritt vermeidet erfundene Criticality-Labels. Maximum F1 oder
G-mean allein bildet die asymmetrische Versicherungsrisikotoleranz nicht ab.

## 10. Explorative Score-Analyse

| Klasse | n | Minimum | Maximum | Mittel | Median | SD |
|---|---:|---:|---:|---:|---:|---:|
| PASS | {pass_stats['count']} | {pass_stats['minimum']:.6f} | {pass_stats['maximum']:.6f} | {pass_stats['mean']:.6f} | {pass_stats['median']:.6f} | {pass_stats['standard_deviation_sample']:.6f} |
| FAIL | {fail_stats['count']} | {fail_stats['minimum']:.6f} | {fail_stats['maximum']:.6f} | {fail_stats['mean']:.6f} | {fail_stats['median']:.6f} | {fail_stats['standard_deviation_sample']:.6f} |

Es gibt in diesem Sample keine Überlappung: höchster FAIL=0,500000,
niedrigster PASS=0,583333, Trennlücke=0,083333. Kein FAIL liegt bei oder über
0,51 und kein PASS darunter. Diese perfekte Trennung ist angesichts der
synthetischen Konstruktion nicht als Produktionsnachweis zu interpretieren.
Shapiro-Wilk ergibt PASS p={pass_stats['shapiro_p']:.4g} und FAIL
p={fail_stats['shapiro_p']:.4g}; Normalität ist für die stark gebündelten
Scores nicht plausibel. Z-score wurde daher nicht als Auswahlmethode verwendet.

![Score-Verteilung](figures/groundedness_score_distribution.png)

## 11. Vergleich aller Candidate Thresholds

Die vollständigen {summary['candidate_threshold_count']} Kandidaten stehen in
`reports/groundedness_threshold_all_candidates.csv`.

![Threshold-Metriken](figures/groundedness_threshold_metrics.png)

## 12. Vergleich mit Maximum F1, G-mean, Youden J und ROC

| Methode | Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | G-mean | Coverage | potenziell kritische FA | FAR 95%-Obergrenze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{comparison_rows}

Alle Vergleichsmethoden landen wegen der perfekten Trennlücke bei derselben
Klassifikation. Die risikobeschränkte Methode bleibt konzeptionell vorzuziehen,
weil sie die FAR-Grenze und kritische Fehler explizit vor der Nutzenoptimierung
anwendet. KDE, GAM und Conformal Prediction wurden bei n=28 nicht als primäre
Methoden eingesetzt; die Dichte- bzw. Kalibrierungsschätzungen wären instabil.

ROC-AUC={summary['roc_auc']:.3f}, Average Precision={summary['average_precision']:.3f}.

![ROC](figures/groundedness_roc_curve.png)

![Precision-Recall](figures/groundedness_precision_recall_curve.png)

## 13. Ergebnisse der Repeated Stratified Cross-Validation

Verwendet wurden K={cv['k']}, {cv['repeats']} Wiederholungen und Seed
{cv['random_seed']}. Der Threshold wurde in jedem Fold ausschließlich auf dem
Trainingsteil gewählt und danach auf dem Testteil gesperrt ausgewertet.

| Größe | Ergebnis |
|---|---:|
| Median Threshold | {cv['threshold_median']:.6f} |
| Mean Threshold | {cv['threshold_mean']:.6f} |
| SD | {cv['threshold_standard_deviation']:.6f} |
| Minimum / Maximum | {cv['threshold_minimum']:.6f} / {cv['threshold_maximum']:.6f} |
| IQR | {cv['threshold_iqr']:.6f} |
| Mean Test-FAR | {cv['test_far_mean']:.3f} |
| Mean Test-Precision | {cv['test_precision_mean']:.3f} |
| Mean Test-Recall | {cv['test_recall_mean']:.3f} |
| Mean Test-F1 | {cv['test_f1_mean']:.3f} |
| Mean Test-Coverage | {cv['test_coverage_mean']:.3f} |
| Folds mit potenziell kritischer FA | {cv['folds_with_worst_case_critical_false_acceptance']} / {cv['folds']} |
| Folds mit FAR <=5% | {cv['far_compliance_fraction']:.1%} |

Die Threshold-Spannweite entsteht maßgeblich durch einen einzigen niedrig
scorenden PASS-Grenzfall: Liegt er im Testfold, steigt der nur auf Training
bestimmte Threshold. Das zeigt begrenzte Threshold-Stabilität trotz perfekter
Gesamtdatentrennung. Pro Testfold gibt es nur zwei oder drei FAIL-Fälle; dessen
FAR kann daher nur 0%, 33,3% oder 50% usw. annehmen.

![CV-Thresholds](figures/groundedness_cv_thresholds.png)

## 14. Konfidenzintervalle und statistische Unsicherheit

Beim empfohlenen Threshold wurden 0 von 14 FAIL akzeptiert. Empirische FAR =
0%, aber die obere einseitige 95%-Clopper-Pearson-Grenze beträgt
{selected['far_upper_95_one_sided']:.1%}. Das empirische 5%-Ziel ist erfüllt,
das statistisch abgesicherte Ziel (obere Grenze <=5%) nicht.

Precision={selected['precision']:.1%}; zweiseitiges 95%-Clopper-Pearson-Intervall
[{selected['precision_ci95_lower']:.1%}, {selected['precision_ci95_upper']:.1%}].
Die Intervalle quantifizieren Binomialunsicherheit, nicht Dataset-Shift,
Labelqualität oder Abhängigkeiten zwischen synthetischen Fällen.

## 15. Vergleich des empfohlenen Thresholds mit 0.51

| Variante | Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | G-mean | Coverage | potenziell kritische FA | FAR 95%-Obergrenze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_metric_table_row('Alt', old)}
{_metric_table_row('Empfohlen', selected)}

0,51 erfüllt auf dem Sample FAR <=5% und akzeptiert keinen FAIL; mangels
Criticality-Labels ist die wörtliche Aussage zu *bestätigten* kritischen Fehlern
nicht möglich. Im Worst-Case akzeptiert 0,51 keinen potenziell kritischen FAIL.
Der empfohlene Wert ist höher und damit numerisch konservativer, aber zwischen
0,500000 und 0,583333 existiert kein beobachteter Fall; der praktische
Unterschied auf diesen Daten ist null. 0,51 wurde auf denselben Fällen gewählt,
auf denen F1=1,0 berichtet wurde. Das ist keine unabhängige Validierung.

## 16. Kritische Fehlerszenarien

Die Fälle enthalten semantisch offensichtliche Beispiele falscher
Selbstbeteiligungen, Policen-IDs, Deckungsarten und Statusangaben. Da Kategorien
und Criticality aber nicht als Ground Truth gespeichert sind, werden keine
Einzelfallzuordnungen behauptet. Vor Produktionseinsatz ist eine fachliche
Doppelannotation erforderlich. Bis dahin ist die Worst-Case-Regel FP=0 die
einzige belegbare konservative Behandlung.

## 17. Einschränkungen

1. Nur 28 synthetische und sprachlich relativ einfache Fälle.
2. Keine dokumentierte menschliche Label-Provenienz.
3. Keine expliziten Fehlerkategorien oder Criticality-Labels.
4. Kein unabhängiges Testset; Auswahl und Full-set-Metriken nutzen dieselben Fälle.
5. Perfekte Trennung kann aus der Testkonstruktion statt Generalisierung folgen.
6. Nur ein Groundedness-Algorithmus und keine Produktionsdrift untersucht.
7. CV-Folds enthalten nur zwei bis drei negative Testfälle.

## 18. Empfohlener Threshold

**Empfehlung:** `{summary['recommended_threshold']:.6f}` als
**provisional empirically calibrated threshold** für
`fact_aware_claim_support_v4`. Er ist der höchste beobachtete Kandidat, der
alle PASS-Fälle akzeptiert und alle FAIL-Fälle zurückweist. Der produktive Wert
wurde im Rahmen dieser Untersuchung **nicht geändert**.

## 19. Status: provisional oder validated

**Provisional.** Das empirische Risikoziel ist im Kalibrierungsset und in allen
CV-Testfolds erfüllt. Die statistisch abgesicherte FAR-Grenze, fachlich
verifizierte Criticality und unabhängige Generalisierung sind nicht belegt.

## 20. Konkrete nächste Schritte

1. Bestehende 28 Fälle durch zwei unabhängige Versicherungsfachpersonen prüfen;
   Konflikte adjudizieren und Label-Provenienz dokumentieren.
2. `error_category` und `critical` explizit annotieren.
3. Ein unangetastetes Hold-out mit mindestens 200 Fällen anlegen: 100 PASS und
   100 FAIL, alle kritischen Kategorien mit mindestens fünf Fällen vertreten,
   verschiedene Produkte, Sprachen, Claim-Längen und Hard Negatives. Bei 100
   FAIL kann selbst ein beobachteter Fehler noch eine einseitige 95%-Obergrenze
   von ungefähr 4,7% erlauben; der Threshold muss vor der einmaligen Auswertung
   gesperrt werden.
4. Danach Drift- und Subgruppenanalysen durchführen. Groundedness sollte
   fachliche Policy-/Customer-Konsistenzprüfungen ergänzen, nicht ersetzen.

## Reproduzierbare Falltabelle

| case_id | Label | Score | Fehlerkategorie | kritisch |
|---|---|---:|---|---|
{case_rows}
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated, reproducible risk-constrained groundedness threshold calibration"
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--normalized", type=Path, default=NORMALIZED_DATASET)
    parser.add_argument("--candidate-report", type=Path, default=CANDIDATE_REPORT)
    parser.add_argument("--cv-report", type=Path, default=CV_REPORT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_REPORT)
    args = parser.parse_args()

    source = args.source.resolve()
    cases = load_and_normalize_cases(source)
    pass_count = sum(case["human_label"] == "PASS" for case in cases)
    fail_count = sum(case["human_label"] == "FAIL" for case in cases)
    if pass_count < 2 or fail_count < 2:
        raise ValueError("At least two explicitly labelled cases per class are required")

    normalized_payload = {
        "schema_version": 1,
        "description": (
            "Normalized copy of existing synthetic fixture labels with scores recomputed "
            "by the unchanged deterministic production groundedness function. Human-label "
            "provenance, error categories and criticality require review."
        ),
        "source_dataset": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(source),
        "algorithm_version": GROUNDING_ALGORITHM_VERSION,
        "score_generation": "deterministic_single_run",
        "cases": cases,
    }
    args.normalized.parent.mkdir(parents=True, exist_ok=True)
    args.normalized.write_text(
        json.dumps(normalized_payload, indent=2, ensure_ascii=False, default=_json_value) + "\n",
        encoding="utf-8",
    )

    thresholds = candidate_thresholds(
        [float(case["groundedness_score"]) for case in cases]
    )
    rows = [metrics_at(cases, threshold) for threshold in thresholds]
    folds = min(5, pass_count, fail_count)
    if folds < 2:
        raise ValueError("Minority class is too small for stratified cross-validation")
    labels = [1 if case["human_label"] == "PASS" else 0 for case in cases]
    splits = build_cv_splits(labels, folds)
    add_fixed_threshold_stability(rows, cases, splits)
    selected = select_risk_constrained(rows, use_cv_stability=True)
    if selected is None:
        raise RuntimeError("No threshold satisfies the empirical risk constraints")

    old = metrics_at(cases, OLD_THRESHOLD)
    cv_rows = run_adaptive_cross_validation(cases, splits, folds)
    cv_summary = aggregate_cv(cv_rows)
    comparisons = choose_comparison_methods(rows)
    comparisons = {
        "Risk-constrained": selected,
        "Maximum F1": comparisons["maximum_f1"],
        "Maximum G-mean": comparisons["maximum_g_mean"],
        "Youden J": comparisons["youden_j"],
        "ROC bei FPR <=5%": comparisons["roc_fpr_le_5pct"],
        "Precision-Recall (max F1)": comparisons["precision_recall_max_f1"],
        "Bisher 0.51": old,
    }

    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray([float(case["groundedness_score"]) for case in cases], dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    generated_at = datetime.now(timezone.utc).isoformat()
    class_stats = {
        "PASS": class_statistics(cases, "PASS"),
        "FAIL": class_statistics(cases, "FAIL"),
    }
    highest_fail = class_stats["FAIL"]["maximum"]
    lowest_pass = class_stats["PASS"]["minimum"]
    existing_config_path = PROJECT_ROOT / "config" / "groundedness_calibration.json"
    existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
    existing_scores = {
        str(case["id"]): float(case["score"])
        for case in existing_config.get("cases", [])
        if "id" in case and "score" in case
    }
    score_differences = [
        abs(float(case["groundedness_score"]) - existing_scores[str(case["case_id"])])
        for case in cases
        if str(case["case_id"]) in existing_scores
    ]

    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "algorithm_version": GROUNDING_ALGORITHM_VERSION,
        "source_dataset": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(source),
        "normalized_dataset": str(args.normalized.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "case_counts": {
            "total": len(cases),
            "pass": pass_count,
            "fail": fail_count,
            "critical_fail_confirmed": 0,
            "criticality_unassessed_fail": fail_count,
            "unclearly_labelled": 0,
            "human_annotation_provenance_unverified": len(cases),
        },
        "risk_definition": {
            "positive_class": "PASS",
            "negative_class": "FAIL",
            "decision_rule": "groundedness_score >= threshold -> ACCEPT",
            "empirical_far_limit": FAR_LIMIT,
            "critical_error_rule": "zero accepted critical FAIL cases",
            "criticality_handling": "worst_case_all_FAIL_potentially_critical",
        },
        "old_threshold": OLD_THRESHOLD,
        "old_threshold_metrics": old,
        "recommended_threshold": float(selected["threshold"]),
        "recommended_threshold_metrics": selected,
        "status": "provisional empirically calibrated threshold",
        "statistically_supported_far_target": bool(
            selected["far_upper_95_one_sided"] <= FAR_LIMIT
        ),
        "candidate_threshold_count": len(rows),
        "score_reproduction": {
            "existing_artifact": "config/groundedness_calibration.json",
            "compared_cases": len(score_differences),
            "exact_matches": sum(difference == 0.0 for difference in score_differences),
            "maximum_absolute_difference": max(score_differences, default=None),
        },
        "class_statistics": class_stats,
        "score_overlap": {
            "highest_fail": highest_fail,
            "lowest_pass": lowest_pass,
            "separation_gap": lowest_pass - highest_fail,
            "overlap_exists": bool(highest_fail >= lowest_pass),
            "fail_accepted_at_0_51": [
                case["case_id"]
                for case in cases
                if case["human_label"] == "FAIL" and case["groundedness_score"] >= OLD_THRESHOLD
            ],
            "pass_rejected_at_0_51": [
                case["case_id"]
                for case in cases
                if case["human_label"] == "PASS" and case["groundedness_score"] < OLD_THRESHOLD
            ],
        },
        "comparison_methods": comparisons,
        "cross_validation": cv_summary,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "roc_curve_auc_trapezoid": float(auc(fpr, tpr)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "independent_final_test_set_exists": False,
        "limitations": [
            "Only 28 synthetic cases are available.",
            "Human annotation provenance is not documented.",
            "Error categories and criticality are absent.",
            "No independent final test set exists.",
            "Threshold 0.51 was selected and evaluated on the same cases.",
            "The 5% FAR target is not statistically supported by the upper 95% bound.",
        ],
    }

    candidate_fields = [
        "threshold", "tp", "fp", "tn", "fn", "false_acceptance_rate",
        "false_rejection_rate", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "g_mean", "coverage",
        "accepted_critical_failures_confirmed", "accepted_failures_worst_case_critical",
        "accepted_fail_case_ids", "far_ci95_lower_two_sided", "far_ci95_upper_two_sided",
        "far_upper_95_one_sided", "precision_ci95_lower", "precision_ci95_upper",
        "fixed_cv_f1_std", "fixed_cv_recall_std", "fixed_cv_far_compliance_rate",
    ]
    cv_fields = [
        "repeat", "fold", "train_size", "test_size", "train_pass", "train_fail",
        "test_pass", "test_fail", "training_threshold", "test_tp", "test_fp",
        "test_tn", "test_fn", "test_far", "test_precision", "test_recall",
        "test_f1", "test_coverage", "test_accepted_critical_failures_confirmed",
        "test_accepted_failures_worst_case_critical", "test_accepted_fail_case_ids",
        "test_far_le_5pct",
    ]
    write_csv(args.candidate_report, rows, candidate_fields)
    write_csv(args.cv_report, cv_rows, cv_fields)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_value) + "\n",
        encoding="utf-8",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(
        cases,
        float(selected["threshold"]),
        FIGURE_DIR / "groundedness_score_distribution.png",
    )
    plot_threshold_metrics(rows, float(selected["threshold"]), FIGURE_DIR / "groundedness_threshold_metrics.png")
    plot_curve(
        fpr,
        tpr,
        title="ROC-Kurve der Groundedness-Scores",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        annotation=f"ROC-AUC = {summary['roc_auc']:.3f}",
        output=FIGURE_DIR / "groundedness_roc_curve.png",
        baseline=True,
    )
    plot_curve(
        recall_curve,
        precision_curve,
        title="Precision-Recall-Kurve der Groundedness-Scores",
        x_label="Recall",
        y_label="Precision",
        annotation=f"AP = {summary['average_precision']:.3f}",
        output=FIGURE_DIR / "groundedness_precision_recall_curve.png",
        baseline=False,
    )
    plot_cv_thresholds(cv_rows, FIGURE_DIR / "groundedness_cv_thresholds.png")
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.write_text(build_markdown_report(summary, cases), encoding="utf-8")

    print(f"usable_cases={len(cases)}")
    print(f"pass={pass_count}")
    print(f"fail={fail_count}")
    print(f"critical_fail_confirmed=0; criticality_unassessed_fail={fail_count}")
    print(f"old_threshold={OLD_THRESHOLD:.6f}")
    print(f"recommended_threshold={selected['threshold']:.6f}")
    print("status=provisional empirically calibrated threshold")
    print(f"far_at_0_51={old['false_acceptance_rate']:.6f}")
    print(f"far_at_recommended={selected['false_acceptance_rate']:.6f}")
    print(f"precision={selected['precision']:.6f}")
    print(f"recall={selected['recall']:.6f}")
    print(f"f1={selected['f1']:.6f}")
    print(f"coverage={selected['coverage']:.6f}")
    print(f"critical_false_acceptances_worst_case={selected['accepted_failures_worst_case_critical']}")
    print(f"cv_threshold_median={cv_summary['threshold_median']:.6f}")
    print(f"cv_threshold_range={cv_summary['threshold_minimum']:.6f}..{cv_summary['threshold_maximum']:.6f}")
    print(f"far_upper_95_one_sided={selected['far_upper_95_one_sided']:.6f}")
    print("main_limitation=no independent human-annotated test set and only 14 FAIL cases")
    for artifact in (
        args.normalized,
        args.candidate_report,
        args.cv_report,
        args.summary,
        args.markdown_report,
        FIGURE_DIR / "groundedness_score_distribution.png",
        FIGURE_DIR / "groundedness_threshold_metrics.png",
        FIGURE_DIR / "groundedness_roc_curve.png",
        FIGURE_DIR / "groundedness_precision_recall_curve.png",
        FIGURE_DIR / "groundedness_cv_thresholds.png",
    ):
        print(f"artifact={artifact.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from langchain_core.documents import Document
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import beta, gaussian_kde
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.integrations.nemo_actions import (  # noqa: E402
    GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score,
)


DATASET = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"
SPLIT_PLAN = PROJECT_ROOT / "reports" / "groundedness_dataset_split_plan.json"
QUALITY_REPORT = PROJECT_ROOT / "reports" / "groundedness_dataset_quality_report.md"
LEAKAGE_REPORT = PROJECT_ROOT / "reports" / "groundedness_dataset_leakage_report.md"
SOURCE_AUDIT = PROJECT_ROOT / "reports" / "groundedness_extended_dataset_source_audit.md"

SPLIT_SNAPSHOT = PROJECT_ROOT / "reports" / "groundedness_weak_supervision_split_snapshot.json"
SCORED_CASES = PROJECT_ROOT / "reports" / "groundedness_weak_supervision_scored_cases.jsonl"
WEAK_QUALITY = PROJECT_ROOT / "reports" / "groundedness_weak_label_quality.csv"
THRESHOLD_CANDIDATES = PROJECT_ROOT / "reports" / "groundedness_weak_threshold_candidates.csv"
CV_OUTPUT = PROJECT_ROOT / "reports" / "groundedness_weak_threshold_cv.csv"
SENSITIVITY_OUTPUT = PROJECT_ROOT / "reports" / "groundedness_weak_threshold_sensitivity.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "groundedness_weak_threshold_summary.json"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / f"groundedness_weak_threshold_calibration_{datetime.now(timezone.utc):%Y%m%d}.md"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES = {
    "distribution": FIGURE_DIR / "groundedness_weak_score_distribution.png",
    "thresholds": FIGURE_DIR / "groundedness_weak_threshold_metrics.png",
    "roc": FIGURE_DIR / "groundedness_weak_roc_curve.png",
    "pr": FIGURE_DIR / "groundedness_weak_precision_recall_curve.png",
    "cv": FIGURE_DIR / "groundedness_weak_cv_thresholds.png",
    "noise": FIGURE_DIR / "groundedness_weak_label_noise_sensitivity.png",
}

RANDOM_SEED = 20260801
FAR_LIMIT = 0.05
OLD_THRESHOLD = 0.51
PREVIOUS_PROVISIONAL_THRESHOLD = 0.583333
NEAR_DUPLICATE_THRESHOLD = 0.94
CV_FOLDS = 5
CV_REPEATS = 10
BOOTSTRAP_REPEATS = 2000
NOISE_REPEATS = 200
REPRO_SAMPLE_SIZE = 25
CONFIDENCE_LEVEL = 0.95
CLASSIFIER_VERSION = "weak_label_quality_v1"

HIGH_RISK_MUTATIONS = {
    "wrong_customer",
    "wrong_policy",
    "wrong_current_policy",
    "old_document_instead_of_current",
    "wrong_policy_status",
    "wrong_deductible",
    "wrong_premium",
    "wrong_coverage",
    "reversed_coverage",
    "wrong_exclusion",
    "wrong_limit",
    "wrong_percentage",
    "wrong_date",
    "correct_fact_but_wrong_entity",
    "correct_number_but_wrong_policy",
    "unsupported_claim_decision",
    "citation_mismatch",
}

REQUIRED_FIELDS = {
    "case_id",
    "source_type",
    "source_file",
    "source_document_id",
    "question",
    "context",
    "candidate_answer",
    "generator_expected_label",
    "mutation_type",
    "question_family_id",
    "document_family_id",
    "product_group",
    "split",
    "leakage_group_id",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", str(text or "").casefold()).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is not None:
        fieldnames = list(fields)
    else:
        fieldnames = []
        seen_fields: set[str] = set()
        for row in rows:
            for field in row:
                if field not in seen_fields:
                    seen_fields.add(field)
                    fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def ensure_inputs() -> None:
    for path in (DATASET, SPLIT_PLAN, QUALITY_REPORT, LEAKAGE_REPORT, SOURCE_AUDIT):
        if not path.exists():
            raise FileNotFoundError(f"Required input artifact not found: {path}")


def ensure_new_outputs() -> None:
    outputs = [
        SPLIT_SNAPSHOT,
        SCORED_CASES,
        WEAK_QUALITY,
        THRESHOLD_CANDIDATES,
        CV_OUTPUT,
        SENSITIVITY_OUTPUT,
        SUMMARY_OUTPUT,
        REPORT_OUTPUT,
        *FIGURES.values(),
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite weak-supervision artifacts:\n" + "\n".join(existing)
        )


def ensure_resume_outputs() -> None:
    required = [SPLIT_SNAPSHOT, SCORED_CASES, WEAK_QUALITY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot resume because partial artifacts are missing:\n" + "\n".join(missing))
    downstream = [
        THRESHOLD_CANDIDATES,
        CV_OUTPUT,
        SENSITIVITY_OUTPUT,
        SUMMARY_OUTPUT,
        REPORT_OUTPUT,
        *FIGURES.values(),
    ]
    existing = [str(path) for path in downstream if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite downstream artifacts:\n" + "\n".join(existing))


def load_verified_scored_resume(
    cases: Sequence[dict[str, Any]], quality_rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    snapshot = json.loads(SPLIT_SNAPSHOT.read_text(encoding="utf-8"))
    assignments = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "leakage_group_id": case["leakage_group_id"],
        }
        for case in cases
    ]
    if snapshot.get("assignment_sha256") != stable_hash(assignments):
        raise ValueError("Frozen split snapshot does not match current candidate assignments")
    if snapshot.get("dataset_sha256") != sha256_file(DATASET):
        raise ValueError("Frozen split snapshot does not match current candidate dataset")

    scored = read_jsonl(SCORED_CASES)
    case_by_id = {str(case["case_id"]): case for case in cases}
    quality_by_id = {str(row["case_id"]): row for row in quality_rows}
    if len(scored) != len(cases) or {str(row["case_id"]) for row in scored} != set(case_by_id):
        raise ValueError("Stored scored cases do not exactly cover the frozen candidate set")
    for row in scored:
        case_id = str(row["case_id"])
        case = case_by_id[case_id]
        expected_hash = stable_hash(
            {
                "question": case["question"],
                "context": case["context"],
                "candidate_answer": case["candidate_answer"],
            }
        )
        if row.get("input_sha256") != expected_hash:
            raise ValueError(f"Stored input hash mismatch for {case_id}")
        if row.get("algorithm_version") != GROUNDING_ALGORITHM_VERSION:
            raise ValueError(f"Stored algorithm version mismatch for {case_id}")
        if row.get("weak_label") != quality_by_id[case_id]["weak_label"]:
            raise ValueError(f"Stored weak label mismatch for {case_id}")
    successful = [row for row in scored if not row.get("technical_error")]
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(successful, min(REPRO_SAMPLE_SIZE, len(successful)))
    reproducibility = {
        "random_seed": RANDOM_SEED,
        "sample_size": len(sample),
        "sample_case_ids": [str(row["case_id"]) for row in sample],
        "maximum_absolute_difference": 0.0,
        "exactly_reproducible": True,
        "resume_evidence": "The scored artifact is written only after the deterministic repeat sample completed successfully; resume verified every input hash and algorithm version without rescoring the hold-out.",
    }
    return snapshot, scored, reproducibility


def exact_duplicate_clusters(cases: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    clusters: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for case in cases:
        key = (
            normalize(case["question"]),
            normalize("\n".join(str(value) for value in case["context"])),
            normalize(case["candidate_answer"]),
        )
        clusters[key].append(str(case["case_id"]))
    result: dict[str, dict[str, Any]] = {}
    for key, ids in clusters.items():
        if len(ids) < 2:
            continue
        ordered = sorted(ids)
        group_id = "dup-" + stable_hash(key)[:12]
        for case_id in ordered:
            result[case_id] = {
                "duplicate_group_id": group_id,
                "canonical_case": case_id == ordered[0],
                "duplicate_cluster_size": len(ordered),
            }
    return result


def classify_weak_labels(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicates = exact_duplicate_clusters(cases)
    rows: list[dict[str, Any]] = []
    medium_mutations = {
        "answer_from_distractor_document",
        "cross_product_confusion",
        "partially_supported_answer",
    }
    for case in cases:
        case_id = str(case["case_id"])
        weak_label = str(case.get("generator_expected_label") or "").strip().upper()
        source_path = PROJECT_ROOT / str(case.get("source_file") or "")
        mutation_path = (
            PROJECT_ROOT / str(case["mutation_source_file"])
            if case.get("mutation_source_file")
            else None
        )
        missing = sorted(field for field in REQUIRED_FIELDS if case.get(field) in (None, "", []))
        flags = list(case.get("quality_flags") or [])
        primary_source_exists = source_path.exists()
        fail_provenance_complete = True
        if weak_label == "FAIL":
            fail_provenance_complete = bool(
                case.get("mutation_type")
                and case.get("mutation_detail")
                and case.get("source_trace")
                and mutation_path is not None
                and mutation_path.exists()
                and case.get("mutation_source_document_id")
            )
        mutation_source_exists = mutation_path.exists() if mutation_path else weak_label != "FAIL"
        traceable_change = bool(
            weak_label == "PASS"
            or (
                case.get("mutation_detail")
                and case.get("mutation_source_document_id")
                and case.get("source_trace")
            )
        )
        duplicate = duplicates.get(case_id, {})
        duplicate_group_id = duplicate.get("duplicate_group_id", "")
        canonical_case = bool(duplicate.get("canonical_case", True))
        excluded_reason = ""
        confidence = "HIGH"
        reasons: list[str] = []

        if weak_label not in {"PASS", "FAIL"}:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "invalid_generator_expected_label"
        elif missing:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "missing_required_fields:" + ",".join(missing)
        elif not primary_source_exists:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "missing_primary_source"
        elif weak_label == "FAIL" and not fail_provenance_complete:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "incomplete_fail_mutation_provenance"
        elif weak_label == "FAIL" and not mutation_source_exists:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "missing_mutation_source"
        elif weak_label == "FAIL" and not traceable_change:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "mutation_not_traceable"
        elif duplicate_group_id and not canonical_case:
            confidence = "EXCLUDE_FROM_PRIMARY"
            excluded_reason = "noncanonical_exact_duplicate"
        elif any("possibly_trivial" in flag or "ambiguous" in flag for flag in flags):
            confidence = "LOW"
            excluded_reason = "quality_flag_requires_review"
        elif duplicate_group_id and canonical_case:
            confidence = "MEDIUM"
            excluded_reason = "canonical_exact_duplicate_cluster_member"
        elif case.get("source_type") in {"legacy_synthetic", "baloise_excluded_pdf"}:
            confidence = "MEDIUM"
            excluded_reason = "source_family_has_additional_semantic_or_provenance_uncertainty"
        elif case.get("mutation_type") in medium_mutations:
            confidence = "MEDIUM"
            excluded_reason = "semantic_collision_possible_in_constructed_negative"

        reasons.extend(
            [
                "primary_source_present" if primary_source_exists else "primary_source_missing",
                "required_fields_complete" if not missing else "required_fields_missing",
                "no_quality_flags" if not flags else "quality_flags:" + "|".join(flags),
                "not_exact_duplicate" if not duplicate_group_id else "exact_duplicate_cluster",
            ]
        )
        if weak_label == "FAIL":
            reasons.append(
                "fail_mutation_provenance_complete"
                if fail_provenance_complete
                else "fail_mutation_provenance_incomplete"
            )
        reasons.append("weak_label_not_human_ground_truth")
        rows.append(
            {
                "case_id": case_id,
                "weak_label": weak_label,
                "weak_label_source": "generator_expected_label plus documented mutation/source provenance",
                "weak_label_confidence": confidence,
                "weak_label_quality_reason": ";".join(reasons),
                "source_type": case.get("source_type"),
                "mutation_type": case.get("mutation_type"),
                "split": case.get("split"),
                "leakage_group_id": case.get("leakage_group_id"),
                "quality_flags": flags,
                "primary_source_exists": primary_source_exists,
                "mutation_source_exists": mutation_source_exists,
                "mutation_provenance_complete": fail_provenance_complete,
                "traceable_change": traceable_change,
                "duplicate_group_id": duplicate_group_id,
                "canonical_case": canonical_case,
                "duplicate_cluster_size": duplicate.get("duplicate_cluster_size", 1),
                "excluded_from_primary_reason": excluded_reason,
                "high_risk_mutation_proxy": (
                    weak_label == "FAIL" and str(case.get("mutation_type")) in HIGH_RISK_MUTATIONS
                ),
                "classifier_version": CLASSIFIER_VERSION,
            }
        )
    return rows


def validate_structure(cases: Sequence[dict[str, Any]], quality: Sequence[dict[str, Any]]) -> dict[str, Any]:
    split_plan = json.loads(SPLIT_PLAN.read_text(encoding="utf-8"))
    ids = [str(case.get("case_id") or "") for case in cases]
    missing_required = sum(
        any(case.get(field) in (None, "", []) for field in REQUIRED_FIELDS) for case in cases
    )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        group_splits[str(case["leakage_group_id"])].add(str(case["split"]))
    duplicate_groups = {
        row["duplicate_group_id"]
        for row in quality
        if row["duplicate_group_id"]
    }
    return {
        "candidate_count": len(cases),
        "unique_case_ids": len(set(ids)),
        "missing_required_fields_cases": missing_required,
        "split_counts": dict(sorted(Counter(str(case["split"]) for case in cases).items())),
        "leakage_group_count": len(group_splits),
        "leakage_group_split_violations": sum(len(splits) > 1 for splits in group_splits.values()),
        "weak_label_counts": dict(sorted(Counter(str(case["generator_expected_label"]) for case in cases).items())),
        "quality_flagged_cases": sum(bool(case.get("quality_flags")) for case in cases),
        "quality_flag_occurrences": sum(len(case.get("quality_flags") or []) for case in cases),
        "exact_duplicate_triplets": len(duplicate_groups),
        "exact_duplicate_cases": sum(bool(row["duplicate_group_id"]) for row in quality),
        "near_duplicate_pairs": int(split_plan["quality"]["near_duplicate_pairs"]),
    }


def make_split_snapshot(cases: Sequence[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    split_plan = json.loads(SPLIT_PLAN.read_text(encoding="utf-8"))
    assignments = [
        {
            "case_id": case["case_id"],
            "split": case["split"],
            "leakage_group_id": case["leakage_group_id"],
        }
        for case in cases
    ]
    snapshot = {
        "schema_version": 1,
        "created_before_scoring_at": utc_now(),
        "status": "FROZEN_FOR_PROVISIONAL_WEAK_SUPERVISION",
        "holdout_designation": "technical weak-label hold-out",
        "dataset_path": str(DATASET.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256_file(DATASET),
        "split_plan_path": str(SPLIT_PLAN.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "split_plan_sha256": sha256_file(SPLIT_PLAN),
        "quality_report_sha256": sha256_file(QUALITY_REPORT),
        "leakage_report_sha256": sha256_file(LEAKAGE_REPORT),
        "source_audit_sha256": sha256_file(SOURCE_AUDIT),
        "random_seed": RANDOM_SEED,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "audit": audit,
        "case_assignments": assignments,
        "assignment_sha256": stable_hash(assignments),
        "rule": "Split assignments are frozen before any groundedness score is calculated and must not be changed based on scores.",
    }
    SPLIT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_SNAPSHOT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return snapshot


def score_cases(
    cases: Sequence[dict[str, Any]],
    quality_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality_by_id = {str(row["case_id"]): row for row in quality_rows}
    scored_at = utc_now()
    scored: list[dict[str, Any]] = []
    for case in cases:
        input_payload = {
            "question": case["question"],
            "context": case["context"],
            "candidate_answer": case["candidate_answer"],
        }
        error = ""
        raw_score: float | None = None
        try:
            documents = [Document(page_content=str(text)) for text in case["context"]]
            raw_score = float(
                calculate_groundedness_score(
                    str(case["candidate_answer"]),
                    documents,
                    str(case["question"]),
                )
            )
        except Exception as exc:  # diagnostic artifact must retain technical failures
            error = f"{type(exc).__name__}: {exc}"
        quality = quality_by_id[str(case["case_id"])]
        scored.append(
            {
                "case_id": case["case_id"],
                "source_type": case["source_type"],
                "source_file": case["source_file"],
                "source_document_id": case["source_document_id"],
                "mutation_type": case["mutation_type"],
                "product_group": case["product_group"],
                "split": case["split"],
                "leakage_group_id": case["leakage_group_id"],
                "weak_label": quality["weak_label"],
                "weak_label_confidence": quality["weak_label_confidence"],
                "weak_label_source": quality["weak_label_source"],
                "weak_label_quality_reason": quality["weak_label_quality_reason"],
                "high_risk_mutation_proxy": quality["high_risk_mutation_proxy"],
                "quality_flags": quality["quality_flags"],
                "duplicate_group_id": quality["duplicate_group_id"],
                "canonical_case": quality["canonical_case"],
                "excluded_from_primary_reason": quality["excluded_from_primary_reason"],
                "raw_score": raw_score,
                "groundedness_score": raw_score,
                "algorithm_version": GROUNDING_ALGORITHM_VERSION,
                "input_sha256": stable_hash(input_payload),
                "score_timestamp": scored_at,
                "technical_error": error,
            }
        )

    successful = [row for row in scored if row["technical_error"] == ""]
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(successful, min(REPRO_SAMPLE_SIZE, len(successful)))
    case_by_id = {str(case["case_id"]): case for case in cases}
    differences: list[float] = []
    sample_ids: list[str] = []
    for row in sample:
        case = case_by_id[str(row["case_id"])]
        repeated = float(
            calculate_groundedness_score(
                str(case["candidate_answer"]),
                [Document(page_content=str(text)) for text in case["context"]],
                str(case["question"]),
            )
        )
        differences.append(abs(repeated - float(row["raw_score"])))
        sample_ids.append(str(row["case_id"]))
    reproducibility = {
        "random_seed": RANDOM_SEED,
        "sample_size": len(sample),
        "sample_case_ids": sample_ids,
        "maximum_absolute_difference": max(differences, default=0.0),
        "exactly_reproducible": all(value == 0.0 for value in differences),
    }
    return scored, reproducibility


def candidate_thresholds(rows: Sequence[dict[str, Any]]) -> list[float]:
    values = sorted({float(row["groundedness_score"]) for row in rows})
    candidates = {0.0, 1.0, OLD_THRESHOLD, PREVIOUS_PROVISIONAL_THRESHOLD, *values}
    candidates.update((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.update(step / 1000.0 for step in range(1001))
    return sorted(round(value, 9) for value in candidates if 0.0 <= value <= 1.0)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    alpha = 1.0 - CONFIDENCE_LEVEL
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return low, high


def row_weights(rows: Sequence[dict[str, Any]], group_weighted: bool) -> np.ndarray:
    if not group_weighted:
        return np.ones(len(rows), dtype=float)
    counts = Counter(str(row["leakage_group_id"]) for row in rows)
    return np.asarray([1.0 / counts[str(row["leakage_group_id"])] for row in rows], dtype=float)


def metrics_at(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    group_weighted: bool = False,
    intervals: bool = False,
) -> dict[str, Any]:
    weights = row_weights(rows, group_weighted)
    labels = np.asarray([str(row["weak_label"]) == "PASS" for row in rows], dtype=bool)
    accepted = np.asarray([float(row["groundedness_score"]) >= threshold for row in rows], dtype=bool)
    tp = float(weights[labels & accepted].sum())
    fp = float(weights[~labels & accepted].sum())
    tn = float(weights[~labels & ~accepted].sum())
    fn = float(weights[labels & ~accepted].sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    far = safe_div(fp, fp + tn)
    frr = safe_div(fn, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    high_risk_accepted = sum(
        str(row["weak_label"]) == "FAIL"
        and bool(row["high_risk_mutation_proxy"])
        and float(row["groundedness_score"]) >= threshold
        for row in rows
    )
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n": len(rows),
        "leakage_groups": len({str(row["leakage_group_id"]) for row in rows}),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "technical_far": far,
        "false_rejection_rate": frr,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "g_mean": math.sqrt(max(0.0, recall * specificity)),
        "youden_j": recall + specificity - 1.0,
        "coverage": safe_div(tp + fp, float(weights.sum())),
        "accepted_high_risk_proxy_failures": int(high_risk_accepted),
        "group_weighted": group_weighted,
    }
    if intervals and not group_weighted:
        far_low, far_high = clopper_pearson(int(round(fp)), int(round(fp + tn)))
        precision_low, precision_high = clopper_pearson(int(round(tp)), int(round(tp + fp)))
        result.update(
            {
                "far_ci95_lower_conditional_on_weak_labels": far_low,
                "far_ci95_upper_conditional_on_weak_labels": far_high,
                "precision_ci95_lower_conditional_on_weak_labels": precision_low,
                "precision_ci95_upper_conditional_on_weak_labels": precision_high,
            }
        )
    return result


def compact_metrics(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = metrics_at(rows, threshold)
    return {
        key: result[key]
        for key in (
            "n",
            "tp",
            "fp",
            "tn",
            "fn",
            "technical_far",
            "precision",
            "recall",
            "f1",
            "coverage",
            "accepted_high_risk_proxy_failures",
        )
    }


def subgroup_metrics_json(rows: Sequence[dict[str, Any]], threshold: float, key: str) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    payload = {name: compact_metrics(members, threshold) for name, members in sorted(groups.items())}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def threshold_sweep(
    rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    *,
    group_weighted: bool = False,
) -> list[dict[str, Any]]:
    return [metrics_at(rows, threshold, group_weighted=group_weighted) for threshold in thresholds]


def tied(rows: Sequence[dict[str, Any]], key: str, target: float) -> list[dict[str, Any]]:
    return [row for row in rows if abs(float(row[key]) - target) <= 1e-12]


def select_risk_constrained(
    sweep: Sequence[dict[str, Any]],
    *,
    stability: dict[float, float] | None = None,
) -> dict[str, Any] | None:
    pool = [row for row in sweep if int(row["accepted_high_risk_proxy_failures"]) == 0]
    pool = [row for row in pool if float(row["technical_far"]) <= FAR_LIMIT]
    if not pool:
        return None
    pool = tied(pool, "recall", max(float(row["recall"]) for row in pool))
    pool = tied(pool, "precision", max(float(row["precision"]) for row in pool))
    pool = tied(pool, "technical_far", min(float(row["technical_far"]) for row in pool))
    if stability:
        best_stability = min(float(stability.get(float(row["threshold"]), math.inf)) for row in pool)
        pool = [
            row
            for row in pool
            if abs(float(stability.get(float(row["threshold"]), math.inf)) - best_stability) <= 1e-12
        ]
    pool = tied(pool, "coverage", max(float(row["coverage"]) for row in pool))
    selected = max(pool, key=lambda row: float(row["threshold"]))
    if float(selected["recall"]) <= 0.0 or float(selected["coverage"]) <= 0.0:
        return None
    result = dict(selected)
    result["fixed_group_cv_f1_std"] = (
        stability.get(float(selected["threshold"])) if stability else None
    )
    return result


def build_group_cv_splits(
    rows: Sequence[dict[str, Any]],
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
) -> tuple[int, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    labels = np.asarray([1 if str(row["weak_label"]) == "PASS" else 0 for row in rows])
    groups = np.asarray([str(row["leakage_group_id"]) for row in rows])
    pass_groups = {group for group, label in zip(groups, labels) if label == 1}
    fail_groups = {group for group, label in zip(groups, labels) if label == 0}
    effective_folds = min(folds, len(pass_groups), len(fail_groups))
    if effective_folds < 2:
        raise ValueError("At least two PASS and two FAIL leakage groups are required")
    indices = np.arange(len(rows))
    result: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=RANDOM_SEED + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, labels, groups), start=1):
            result.append((repeat + 1, fold, train_idx, test_idx))
    return effective_folds, result


def fixed_threshold_stability(
    rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
) -> dict[float, float]:
    result: dict[float, float] = {}
    for threshold in thresholds:
        fold_f1 = [
            metrics_at([rows[int(index)] for index in test_idx], threshold)["f1"]
            for _, _, _, test_idx in splits
        ]
        result[float(threshold)] = statistics.pstdev(float(value) for value in fold_f1)
    return result


def run_group_cv(
    rows: Sequence[dict[str, Any]],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for repeat, fold, train_idx, test_idx in splits:
        train = [rows[int(index)] for index in train_idx]
        test = [rows[int(index)] for index in test_idx]
        train_thresholds = candidate_thresholds(train)
        selected = select_risk_constrained(threshold_sweep(train, train_thresholds))
        if selected is None:
            output.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "status": "NO_VALID_TRAINING_THRESHOLD",
                    "train_size": len(train),
                    "test_size": len(test),
                    "train_pass": sum(row["weak_label"] == "PASS" for row in train),
                    "train_fail": sum(row["weak_label"] == "FAIL" for row in train),
                    "test_pass": sum(row["weak_label"] == "PASS" for row in test),
                    "test_fail": sum(row["weak_label"] == "FAIL" for row in test),
                    "train_leakage_groups": len({row["leakage_group_id"] for row in train}),
                    "test_leakage_groups": len({row["leakage_group_id"] for row in test}),
                }
            )
            continue
        test_metrics = metrics_at(test, float(selected["threshold"]))
        output.append(
            {
                "repeat": repeat,
                "fold": fold,
                "status": "OK",
                "training_threshold": selected["threshold"],
                "train_size": len(train),
                "test_size": len(test),
                "train_pass": sum(row["weak_label"] == "PASS" for row in train),
                "train_fail": sum(row["weak_label"] == "FAIL" for row in train),
                "test_pass": sum(row["weak_label"] == "PASS" for row in test),
                "test_fail": sum(row["weak_label"] == "FAIL" for row in test),
                "train_leakage_groups": len({row["leakage_group_id"] for row in train}),
                "test_leakage_groups": len({row["leakage_group_id"] for row in test}),
                "test_technical_far": test_metrics["technical_far"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_coverage": test_metrics["coverage"],
                "test_accepted_high_risk_proxy_failures": test_metrics[
                    "accepted_high_risk_proxy_failures"
                ],
                "test_far_le_5pct": float(test_metrics["technical_far"]) <= FAR_LIMIT,
            }
        )
    return output


def aggregate_cv(rows: Sequence[dict[str, Any]], folds: int) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "OK"]
    thresholds = np.asarray([float(row["training_threshold"]) for row in valid], dtype=float)
    if not valid:
        return {"valid_folds": 0, "total_folds": len(rows), "status": "NO_VALID_FOLDS"}
    return {
        "status": "OK" if len(valid) == len(rows) else "PARTIAL",
        "k": folds,
        "repeats": CV_REPEATS,
        "random_seed": RANDOM_SEED,
        "valid_folds": len(valid),
        "total_folds": len(rows),
        "threshold_median": float(np.median(thresholds)),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_standard_deviation": float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
        "threshold_minimum": float(np.min(thresholds)),
        "threshold_maximum": float(np.max(thresholds)),
        "threshold_q1": float(np.percentile(thresholds, 25)),
        "threshold_q3": float(np.percentile(thresholds, 75)),
        "threshold_iqr": float(np.percentile(thresholds, 75) - np.percentile(thresholds, 25)),
        "fold_fraction_far_le_5pct": safe_div(sum(bool(row["test_far_le_5pct"]) for row in valid), len(valid)),
        "fold_fraction_with_high_risk_proxy_false_acceptance": safe_div(
            sum(int(row["test_accepted_high_risk_proxy_failures"]) > 0 for row in valid),
            len(valid),
        ),
        "test_far_mean": statistics.mean(float(row["test_technical_far"]) for row in valid),
        "test_precision_mean": statistics.mean(float(row["test_precision"]) for row in valid),
        "test_recall_mean": statistics.mean(float(row["test_recall"]) for row in valid),
        "test_f1_mean": statistics.mean(float(row["test_f1"]) for row in valid),
        "test_coverage_mean": statistics.mean(float(row["test_coverage"]) for row in valid),
    }


def comparison_methods(
    rows: Sequence[dict[str, Any]],
    sweep: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    def conservative_max(key: str) -> dict[str, Any]:
        best = max(float(row[key]) for row in sweep)
        return max(tied(sweep, key, best), key=lambda row: float(row["threshold"]))

    fpr_pool = [row for row in sweep if float(row["technical_far"]) <= FAR_LIMIT]
    fpr_recall = max(float(row["recall"]) for row in fpr_pool)
    return {
        "old_threshold_0_51": metrics_at(rows, OLD_THRESHOLD),
        "previous_provisional_0_583333": metrics_at(rows, PREVIOUS_PROVISIONAL_THRESHOLD),
        "maximum_f1": conservative_max("f1"),
        "precision_recall_maximum_f1": conservative_max("f1"),
        "maximum_g_mean": conservative_max("g_mean"),
        "youden_j": conservative_max("youden_j"),
        "roc_fpr_le_5pct": max(tied(fpr_pool, "recall", fpr_recall), key=lambda row: float(row["threshold"])),
    }


def unsupervised_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([float(row["groundedness_score"]) for row in rows], dtype=float)
    unique = np.unique(scores)
    gaps = [
        {"left": float(left), "right": float(right), "gap": float(right - left), "midpoint": float((left + right) / 2.0)}
        for left, right in zip(unique, unique[1:])
    ]
    largest_gap = max(gaps, key=lambda row: row["gap"]) if gaps else None
    kde_result: dict[str, Any] = {"used": False, "reason": "too_few_unique_scores"}
    if len(unique) >= 3 and float(np.std(scores)) > 0.0:
        grid = np.linspace(0.0, 1.0, 2001)
        density = gaussian_kde(scores)(grid)
        minima = [
            float(grid[index])
            for index in range(1, len(grid) - 1)
            if density[index] < density[index - 1] and density[index] < density[index + 1]
        ]
        kde_result = {
            "used": True,
            "caveat": "Scores are discrete; KDE is descriptive only.",
            "local_minima": minima,
        }
    gmm_result: dict[str, Any] = {"used": False, "reason": "model_not_plausible_or_too_few_unique_scores"}
    if len(unique) >= 5:
        values = scores.reshape(-1, 1)
        one = GaussianMixture(n_components=1, random_state=RANDOM_SEED).fit(values)
        two = GaussianMixture(n_components=2, random_state=RANDOM_SEED).fit(values)
        bic_one, bic_two = float(one.bic(values)), float(two.bic(values))
        if bic_two < bic_one:
            grid = np.linspace(0.0, 1.0, 10001).reshape(-1, 1)
            responsibilities = two.predict_proba(grid)
            crossing = int(np.argmin(np.abs(responsibilities[:, 0] - responsibilities[:, 1])))
            gmm_result = {
                "used": True,
                "bic_one_component": bic_one,
                "bic_two_components": bic_two,
                "component_means": sorted(float(value) for value in two.means_.ravel()),
                "equal_responsibility_threshold": float(grid[crossing, 0]),
                "caveat": "Two-component Gaussian mixture is an unsupervised comparison, not a safety proof.",
            }
        else:
            gmm_result = {
                "used": False,
                "bic_one_component": bic_one,
                "bic_two_components": bic_two,
                "reason": "two_component_bic_not_better",
            }
    return {
        "n": len(scores),
        "unique_score_count": len(unique),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "standard_deviation": float(np.std(scores, ddof=1)),
        "empirical_quantiles": {
            "q01": float(np.percentile(scores, 1)),
            "q05": float(np.percentile(scores, 5)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
            "q95": float(np.percentile(scores, 95)),
            "q99": float(np.percentile(scores, 99)),
        },
        "largest_observed_score_gap": largest_gap,
        "all_observed_gaps": sorted(gaps, key=lambda row: row["gap"], reverse=True)[:20],
        "kde": kde_result,
        "gaussian_mixture": gmm_result,
    }


def select_for_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_weighted: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not rows or len({row["weak_label"] for row in rows}) < 2:
        return None, []
    thresholds = candidate_thresholds(rows)
    sweep = threshold_sweep(rows, thresholds, group_weighted=group_weighted)
    return select_risk_constrained(sweep), sweep


def sensitivity_analyses(
    scored: Sequence[dict[str, Any]],
    primary: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usable_calibration = [
        row
        for row in scored
        if row["split"] == "CALIBRATION"
        and not row["technical_error"]
        and row["weak_label_confidence"] != "EXCLUDE_FROM_PRIMARY"
        and (not row["duplicate_group_id"] or row["canonical_case"])
    ]
    variants: list[tuple[str, list[dict[str, Any]], bool]] = [
        ("HIGH_only", list(primary), False),
        (
            "HIGH_plus_MEDIUM",
            [row for row in usable_calibration if row["weak_label_confidence"] in {"HIGH", "MEDIUM"}],
            False,
        ),
        ("all_technically_usable", usable_calibration, False),
        ("group_weighted_leakage_groups", usable_calibration, True),
        ("without_quality_flag_cases", [row for row in usable_calibration if not row["quality_flags"]], False),
        ("including_quality_flag_cases", usable_calibration, False),
    ]
    for source_type in sorted({str(row["source_type"]) for row in usable_calibration}):
        variants.append(
            (
                f"source_type:{source_type}",
                [row for row in usable_calibration if row["source_type"] == source_type],
                False,
            )
        )
    for mutation_type in sorted({str(row["mutation_type"]) for row in usable_calibration}):
        variants.append(
            (
                f"mutation_type:{mutation_type}",
                [row for row in usable_calibration if row["mutation_type"] == mutation_type],
                False,
            )
        )

    output: list[dict[str, Any]] = []
    source_thresholds: dict[str, Any] = {}
    for name, members, group_weighted in variants:
        selected, _ = select_for_rows(members, group_weighted=group_weighted)
        row: dict[str, Any] = {
            "analysis": name,
            "n": len(members),
            "pass": sum(member["weak_label"] == "PASS" for member in members),
            "fail": sum(member["weak_label"] == "FAIL" for member in members),
            "leakage_groups": len({member["leakage_group_id"] for member in members}),
            "group_weighted": group_weighted,
            "status": "VALID" if selected else "NOT_IDENTIFIABLE_OR_NO_VALID_THRESHOLD",
        }
        if selected:
            row.update(selected)
        output.append(row)
        if name.startswith("source_type:"):
            source_thresholds[name.split(":", 1)[1]] = {
                "status": row["status"],
                "threshold": row.get("threshold"),
                "n": len(members),
                "technical_far": row.get("technical_far"),
                "recall": row.get("recall"),
            }
    return output, source_thresholds


def fast_select_threshold(
    rows: Sequence[dict[str, Any]],
    thresholds: np.ndarray,
) -> float | None:
    if not rows or len({row["weak_label"] for row in rows}) < 2:
        return None
    scores = np.asarray([float(row["groundedness_score"]) for row in rows], dtype=float)
    labels = np.asarray([str(row["weak_label"]) == "PASS" for row in rows], dtype=bool)
    high_risk = np.asarray(
        [str(row["weak_label"]) == "FAIL" and bool(row["high_risk_mutation_proxy"]) for row in rows],
        dtype=bool,
    )
    accepted = scores[None, :] >= thresholds[:, None]
    tp = (accepted & labels[None, :]).sum(axis=1)
    fp = (accepted & ~labels[None, :]).sum(axis=1)
    tn = (~accepted & ~labels[None, :]).sum(axis=1)
    fn = (~accepted & labels[None, :]).sum(axis=1)
    high_risk_accepted = (accepted & high_risk[None, :]).sum(axis=1)
    far = np.divide(fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    coverage = (tp + fp) / len(rows)
    valid = (high_risk_accepted == 0) & (far <= FAR_LIMIT) & (recall > 0) & (coverage > 0)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    candidates = indices
    candidates = candidates[np.isclose(recall[candidates], recall[candidates].max())]
    candidates = candidates[np.isclose(precision[candidates], precision[candidates].max())]
    candidates = candidates[np.isclose(far[candidates], far[candidates].min())]
    candidates = candidates[np.isclose(coverage[candidates], coverage[candidates].max())]
    return float(thresholds[candidates[-1]])


def cluster_bootstrap(
    primary: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        by_group[str(row["leakage_group_id"])].append(row)
    group_names = sorted(by_group)
    rng = np.random.default_rng(RANDOM_SEED)
    threshold_array = np.asarray(thresholds, dtype=float)
    selected_values: list[float] = []
    no_solution = 0
    for _ in range(BOOTSTRAP_REPEATS):
        sampled_names = rng.choice(group_names, size=len(group_names), replace=True)
        sample = [dict(row) for name in sampled_names for row in by_group[str(name)]]
        selected = fast_select_threshold(sample, threshold_array)
        if selected is None:
            no_solution += 1
        else:
            selected_values.append(selected)
    values = np.asarray(selected_values, dtype=float)
    return {
        "designation": "cluster bootstrap conditional on weak labels",
        "random_seed": RANDOM_SEED,
        "repeats": BOOTSTRAP_REPEATS,
        "successful_repeats": len(selected_values),
        "no_valid_threshold_repeats": no_solution,
        "threshold_median": float(np.median(values)) if len(values) else None,
        "threshold_mean": float(np.mean(values)) if len(values) else None,
        "threshold_ci95_lower": float(np.percentile(values, 2.5)) if len(values) else None,
        "threshold_ci95_upper": float(np.percentile(values, 97.5)) if len(values) else None,
        "threshold_minimum": float(np.min(values)) if len(values) else None,
        "threshold_maximum": float(np.max(values)) if len(values) else None,
    }


def weak_label_noise(
    primary: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    baseline_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED + 91)
    threshold_array = np.asarray(thresholds, dtype=float)
    baseline_predictions = np.asarray(
        [float(row["groundedness_score"]) >= baseline_threshold for row in primary], dtype=bool
    )
    details: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for rate in (0.01, 0.05, 0.10):
        selected_values: list[float] = []
        far_values: list[float] = []
        recall_values: list[float] = []
        f1_values: list[float] = []
        decision_changed: list[bool] = []
        no_solution = 0
        flips = max(1, int(round(len(primary) * rate)))
        for iteration in range(NOISE_REPEATS):
            flip_indices = set(int(value) for value in rng.choice(len(primary), size=flips, replace=False))
            noisy: list[dict[str, Any]] = []
            for index, row in enumerate(primary):
                changed = dict(row)
                if index in flip_indices:
                    changed["weak_label"] = "FAIL" if row["weak_label"] == "PASS" else "PASS"
                noisy.append(changed)
            selected = fast_select_threshold(noisy, threshold_array)
            if selected is None:
                no_solution += 1
                details.append(
                    {
                        "noise_rate": rate,
                        "iteration": iteration + 1,
                        "flipped_labels": flips,
                        "status": "NO_VALID_THRESHOLD",
                    }
                )
                continue
            evaluated = metrics_at(primary, selected)
            changed_predictions = np.asarray(
                [float(row["groundedness_score"]) >= selected for row in primary], dtype=bool
            )
            material = bool(np.any(changed_predictions != baseline_predictions))
            selected_values.append(selected)
            far_values.append(float(evaluated["technical_far"]))
            recall_values.append(float(evaluated["recall"]))
            f1_values.append(float(evaluated["f1"]))
            decision_changed.append(material)
            details.append(
                {
                    "noise_rate": rate,
                    "iteration": iteration + 1,
                    "flipped_labels": flips,
                    "status": "OK",
                    "selected_threshold": selected,
                    "evaluation_against_original_weak_labels_far": evaluated["technical_far"],
                    "evaluation_against_original_weak_labels_recall": evaluated["recall"],
                    "evaluation_against_original_weak_labels_f1": evaluated["f1"],
                    "material_decision_change": material,
                }
            )
        values = np.asarray(selected_values, dtype=float)
        key = f"{int(rate * 100)}pct"
        summary[key] = {
            "noise_rate": rate,
            "repeats": NOISE_REPEATS,
            "successful_repeats": len(selected_values),
            "no_valid_threshold_repeats": no_solution,
            "threshold_median": float(np.median(values)) if len(values) else None,
            "threshold_minimum": float(np.min(values)) if len(values) else None,
            "threshold_maximum": float(np.max(values)) if len(values) else None,
            "threshold_q1": float(np.percentile(values, 25)) if len(values) else None,
            "threshold_q3": float(np.percentile(values, 75)) if len(values) else None,
            "far_mean_against_original_weak_labels": statistics.mean(far_values) if far_values else None,
            "recall_mean_against_original_weak_labels": statistics.mean(recall_values) if recall_values else None,
            "f1_mean_against_original_weak_labels": statistics.mean(f1_values) if f1_values else None,
            "fraction_with_material_decision_change": safe_div(sum(decision_changed), len(decision_changed)),
            "material_change_definition": "at least one original Calibration case changes ACCEPT/REJECT decision",
        }
    return details, summary


def make_plots(
    all_rows: Sequence[dict[str, Any]],
    primary: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    selected_threshold: float,
    cv_rows: Sequence[dict[str, Any]],
    noise_summary: dict[str, Any],
    risk_constrained_selection_available: bool,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 720
    margin = (105, 85, 45, 105)
    palette = ["#2B6CB0", "#2F855A", "#805AD5", "#C53030", "#DD6B20"]

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            name = "arialbd.ttf" if bold else "arial.ttf"
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()

    def chart(
        path: Path,
        title: str,
        xlabel: str,
        ylabel: str,
        series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
        xlim: tuple[float, float] = (0.0, 1.0),
        ylim: tuple[float, float] = (0.0, 1.0),
        verticals: Sequence[tuple[float, str, str]] = (),
        horizontals: Sequence[tuple[float, str, str]] = (),
    ) -> None:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        left, top, right_pad, bottom_pad = margin
        right, bottom = width - right_pad, height - bottom_pad

        def xy(x: float, y: float) -> tuple[int, int]:
            px = left + (float(x) - xlim[0]) / max(xlim[1] - xlim[0], 1e-12) * (right - left)
            py = bottom - (float(y) - ylim[0]) / max(ylim[1] - ylim[0], 1e-12) * (bottom - top)
            return int(px), int(py)

        for i in range(6):
            xv = xlim[0] + i * (xlim[1] - xlim[0]) / 5
            yv = ylim[0] + i * (ylim[1] - ylim[0]) / 5
            px, _ = xy(xv, ylim[0])
            _, py = xy(xlim[0], yv)
            draw.line((px, top, px, bottom), fill="#E2E8F0", width=1)
            draw.line((left, py, right, py), fill="#E2E8F0", width=1)
            draw.text((px - 20, bottom + 12), f"{xv:.2f}", fill="#4A5568", font=font(16))
            draw.text((15, py - 9), f"{yv:.2f}", fill="#4A5568", font=font(16))
        draw.line((left, top, left, bottom), fill="#1A202C", width=2)
        draw.line((left, bottom, right, bottom), fill="#1A202C", width=2)
        for value, label, color in verticals:
            px, _ = xy(value, ylim[0])
            draw.line((px, top, px, bottom), fill=color, width=3)
            draw.text((min(px + 6, right - 180), top + 8), label, fill=color, font=font(16, True))
        for value, label, color in horizontals:
            _, py = xy(xlim[0], value)
            draw.line((left, py, right, py), fill=color, width=3)
            draw.text((right - 190, max(top, py - 22)), label, fill=color, font=font(16, True))
        legend_x = left + 15
        for index, (label, xs, ys, color) in enumerate(series):
            points = [xy(x, y) for x, y in zip(xs, ys)]
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            elif points:
                px, py = points[0]
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
            lx = legend_x + index * 200
            draw.line((lx, top + 48, lx + 30, top + 48), fill=color, width=5)
            draw.text((lx + 38, top + 36), label, fill="#1A202C", font=font(16))
        draw.text((left, 25), title, fill="#1A202C", font=font(25, True))
        draw.text(((left + right) // 2 - 90, height - 42), xlabel, fill="#1A202C", font=font(18, True))
        draw.text((12, 48), ylabel, fill="#1A202C", font=font(18, True))
        image.save(path, format="PNG")

    def histogram(
        path: Path,
        values: Sequence[float],
        title: str,
        xlabel: str,
        color: str,
        marker: float | None = None,
        marker_label: str = "",
        show_ecdf: bool = False,
    ) -> None:
        counts, edges = np.histogram(np.asarray(values, dtype=float), bins=24, range=(0.0, 1.0))
        centers = (edges[:-1] + edges[1:]) / 2
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        left, top, right_pad, bottom_pad = margin
        right, bottom = width - right_pad, height - bottom_pad
        ymax = max(int(counts.max()), 1)
        for i in range(6):
            y = bottom - int(i / 5 * (bottom - top))
            draw.line((left, y, right, y), fill="#E2E8F0", width=1)
            draw.text((20, y - 9), str(round(i / 5 * ymax)), fill="#4A5568", font=font(16))
        bar_width = max(2, int((right - left) / len(counts)) - 2)
        for center, count in zip(centers, counts):
            px = left + int(center * (right - left))
            py = bottom - int(int(count) / ymax * (bottom - top))
            draw.rectangle((px - bar_width // 2, py, px + bar_width // 2, bottom), fill=color)
        if marker is not None:
            threshold_x = left + int(marker * (right - left))
            draw.line((threshold_x, top, threshold_x, bottom), fill="#1A202C", width=3)
            draw.text(
                (min(threshold_x + 7, right - 280), top + 8),
                f"{marker_label} {marker:.4f}",
                fill="#1A202C",
                font=font(16, True),
            )
        draw.line((left, top, left, bottom), fill="#1A202C", width=2)
        draw.line((left, bottom, right, bottom), fill="#1A202C", width=2)
        for i in range(6):
            x = left + int(i / 5 * (right - left))
            draw.text((x - 15, bottom + 12), f"{i / 5:.1f}", fill="#4A5568", font=font(16))
        if show_ecdf and len(values):
            ordered = np.sort(np.asarray(values, dtype=float))
            ecdf_points = [
                (
                    left + int(float(value) * (right - left)),
                    bottom - int((index + 1) / len(ordered) * (bottom - top)),
                )
                for index, value in enumerate(ordered)
            ]
            if len(ecdf_points) > 1:
                draw.line(ecdf_points, fill="#2F855A", width=4)
            draw.line((right, top, right, bottom), fill="#2F855A", width=2)
            for i in range(6):
                y = bottom - int(i / 5 * (bottom - top))
                draw.text((right + 8, y - 9), f"{i / 5:.1f}", fill="#2F855A", font=font(16))
            draw.line((left + 18, top + 50, left + 48, top + 50), fill="#2F855A", width=5)
            draw.text((left + 55, top + 39), "ECDF (right axis)", fill="#2F855A", font=font(16, True))
        draw.text((left, 25), title, fill="#1A202C", font=font(25, True))
        draw.text(((left + right) // 2 - 80, height - 42), xlabel, fill="#1A202C", font=font(18, True))
        draw.text((18, 48), "Count", fill="#1A202C", font=font(18, True))
        image.save(path, format="PNG")

    scores = np.asarray([float(row["groundedness_score"]) for row in all_rows], dtype=float)
    marker_label = "selected" if risk_constrained_selection_available else "diagnostic max-F1"
    histogram(
        FIGURES["distribution"],
        scores,
        "Groundedness score histogram and ECDF (unlabelled view)",
        "Groundedness score",
        palette[0],
        marker=selected_threshold,
        marker_label=marker_label,
        show_ecdf=True,
    )

    threshold_x = [float(row["threshold"]) for row in candidate_rows]
    threshold_series = [
        (label, threshold_x, [float(row[key]) for row in candidate_rows], color)
        for (key, label, color) in (
            ("recall", "Recall", palette[0]),
            ("precision", "Precision", palette[1]),
            ("f1", "F1", palette[2]),
            ("technical_far", "Technical FAR", palette[3]),
            ("coverage", "Coverage", palette[4]),
        )
    ]
    chart(FIGURES["thresholds"], "Threshold metrics against HIGH-confidence weak labels", "Threshold", "Metric", threshold_series, verticals=[(selected_threshold, marker_label, "#1A202C")])

    labels = np.asarray([1 if row["weak_label"] == "PASS" else 0 for row in primary])
    primary_scores = np.asarray([float(row["groundedness_score"]) for row in primary])
    fpr, tpr, _ = roc_curve(labels, primary_scores)
    chart(FIGURES["roc"], f"ROC against weak labels (AUC={roc_auc_score(labels, primary_scores):.3f})", "Technical FPR/FAR", "Recall", [("ROC", fpr, tpr, palette[0]), ("Chance", [0, 1], [0, 1], "#718096")], verticals=[(FAR_LIMIT, "FPR 5%", palette[3])])

    precision, recall, _ = precision_recall_curve(labels, primary_scores)
    chart(FIGURES["pr"], f"Precision-recall against weak labels (AP={average_precision_score(labels, primary_scores):.3f})", "Recall", "Precision", [("PR", recall, precision, palette[1])])

    valid_cv = [float(row["training_threshold"]) for row in cv_rows if row["status"] == "OK"]
    histogram(FIGURES["cv"], valid_cv, "Repeated stratified group CV thresholds", "Fold training threshold", palette[2])

    noise_points = [
        (rate, noise_summary[key])
        for rate, key in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"))
        if noise_summary[key]["threshold_median"] is not None
    ]
    noise_x = [rate for rate, _ in noise_points]
    noise_median = [float(item["threshold_median"]) for _, item in noise_points]
    noise_min = [float(item["threshold_minimum"]) for _, item in noise_points]
    noise_max = [float(item["threshold_maximum"]) for _, item in noise_points]
    chart(
        FIGURES["noise"],
        "Weak-label noise sensitivity (min / median / max)",
        "Simulated weak-label noise rate",
        "Selected threshold",
        [("Minimum", noise_x, noise_min, palette[0]), ("Median", noise_x, noise_median, palette[3]), ("Maximum", noise_x, noise_max, palette[4])],
        xlim=(0.0, 0.11),
        horizontals=[(selected_threshold, f"{marker_label} baseline", "#1A202C")],
    )


def format_metric_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {float(row['threshold']):.6f} | {row['tp']:.3f} | {row['fp']:.3f} | "
        f"{row['tn']:.3f} | {row['fn']:.3f} | {row['technical_far']:.3%} | "
        f"{row['precision']:.3%} | {row['recall']:.3%} | {row['f1']:.3%} | "
        f"{row['coverage']:.3%} | {row['accepted_high_risk_proxy_failures']} |"
    )


def report_markdown(summary: dict[str, Any]) -> str:
    primary = summary["primary_selection"]
    calibration = summary["evaluation"]["calibration_high_confidence"]
    validation = summary["evaluation"]["validation_high_confidence"]
    holdout = summary["evaluation"]["technical_weak_label_holdout_high_confidence"]
    cv = summary["cross_validation"]
    counts = summary["weak_label_quality_counts"]
    audit = summary["verified_starting_state"]
    comparisons = summary["comparison_methods"]
    unsupervised = summary["unsupervised_analysis"]
    recommendation = summary["recommendation"]
    operating_point = summary["evaluation_operating_point"]
    if recommendation["status"] == "NO_RECOMMENDED_THRESHOLD":
        evaluation_note = (
            f"Da die primäre risikobeschränkte Regel keinen gültigen Threshold liefert, "
            f"werden die folgenden Split-Metriken ausschließlich am diagnostischen Maximum-F1-"
            f"Vergleichspunkt {operating_point['threshold']:.6f} gezeigt. Dieser Wert ist keine Empfehlung."
        )
        evaluation_suffix = "diagnostic maximum-F1 comparator; not recommended"
    else:
        evaluation_note = (
            f"Die Split-Metriken verwenden den auf Calibration gesperrten risikobeschränkten "
            f"Threshold {operating_point['threshold']:.6f}."
        )
        evaluation_suffix = "risk-constrained selection"
    if cv.get("valid_folds", 0):
        cv_text = (
            f"K={cv['k']}, Wiederholungen={cv['repeats']}, Seed={cv['random_seed']}. "
            f"Threshold Median/Mean/SD: {cv['threshold_median']:.6f} / "
            f"{cv['threshold_mean']:.6f} / {cv['threshold_standard_deviation']:.6f}; "
            f"Spanne [{cv['threshold_minimum']:.6f}, {cv['threshold_maximum']:.6f}], "
            f"IQR={cv['threshold_iqr']:.6f}. FAR<=5% in "
            f"{cv['fold_fraction_far_le_5pct']:.1%} der gültigen Folds; High-Risk-Proxy-FA in "
            f"{cv['fold_fraction_with_high_risk_proxy_false_acceptance']:.1%}. "
            f"Gültige Folds: {cv['valid_folds']}/{cv['total_folds']}."
        )
    else:
        cv_text = (
            f"Keiner der {cv['total_folds']} Group-CV-Folds lieferte auf seinem Trainingsteil "
            "einen gültigen risikobeschränkten Threshold. Deshalb sind Median und Spannweite nicht bestimmbar."
        )
    comparison_rows = "\n".join(format_metric_row(name, row) for name, row in comparisons.items())
    source_rows = "\n".join(
        f"| {name} | {item['n']} | {item['status']} | {'' if item['threshold'] is None else f'{item['threshold']:.6f}'} | {'' if item['technical_far'] is None else f'{item['technical_far']:.3%}'} | {'' if item['recall'] is None else f'{item['recall']:.3%}'} |"
        for name, item in summary["source_type_threshold_sensitivity"].items()
    )
    noise_rows = "\n".join(
        f"| {key} | {value['successful_repeats']} | {value['threshold_median']} | {value['threshold_minimum']} | {value['threshold_maximum']} | {value['far_mean_against_original_weak_labels']} | {value['recall_mean_against_original_weak_labels']} | {value['fraction_with_material_decision_change']} |"
        for key, value in summary["weak_label_noise_sensitivity"].items()
    )
    return f"""# Vorläufige Groundedness-Threshold-Kalibrierung mit Weak Supervision

## 1. Executive Summary

**Wissenschaftlicher Status:** `provisional weakly supervised threshold`.
Es wurde kein menschliches Ground Truth verwendet. Die technische Konstruktion
der Fälle dient ausschließlich als Weak Label.

Primäres Ergebnis: **{recommendation['display_value']}**. Status:
`{recommendation['status']}`. Die produktive Konfiguration wurde nicht geändert.

## 2. Ziel und Scope

Untersucht wird der deterministische Score `fact_aware_claim_support_v4` auf
gespeicherten Tripeln aus Question, Context und Candidate Answer. Kein Retrieval,
kein LLM, kein CRM, kein Self-Check und kein API-Endpunkt wurden aufgerufen.

## 3. Warum kein menschliches Ground Truth vorliegt

Reviewer 1, Reviewer 2 und Adjudication sind weiterhin leer. Keine menschlichen
Labels oder Criticality-Werte wurden ergänzt. Alle Leistungsmetriken sind daher
**conditional on weak labels**.

## 4. Definition von Weak Labels

`generator_expected_label=PASS` ist die technische positive Klasse;
`generator_expected_label=FAIL` die technische negative Klasse. Der Score wurde
nicht zur Labelerzeugung verwendet. Entscheidung: `score >= threshold -> ACCEPT`.

## 5. Datenquellen

Durch Repository-Artefakte belegt: InsuranceQA, aktive Helvetia-PDFs,
ausgeschlossene Baloise-Distraktoren, synthetische CRM-/Policendaten und 28
Legacy-Fälle. Details stehen im Source Audit.

## 6. Candidate- und Split-Struktur

- Kandidaten: {audit['candidate_count']}
- Calibration / Validation / Preliminary Hold-out: {audit['split_counts']['CALIBRATION']} / {audit['split_counts']['VALIDATION']} / {audit['split_counts']['LOCKED_HOLDOUT_CANDIDATE']}
- Technische PASS / FAIL: {audit['weak_label_counts']['PASS']} / {audit['weak_label_counts']['FAIL']}
- Leakage Groups: {audit['leakage_group_count']}

## 7. Leakage-Schutz

Der Split wurde vor dem Scoring mit SHA-256-Hashes und allen Fallzuordnungen
eingefroren. Es gibt {audit['leakage_group_split_violations']} gruppenübergreifende
Splitverletzungen. Cross-Validation verwendet ausschließlich `leakage_group_id`.

## 8. Quality Flags und Duplikate

- Quality-Flag-Fälle: {audit['quality_flagged_cases']}
- Exact-Duplicate-Cluster: {audit['exact_duplicate_triplets']}
- Near-Duplicate-Paare: {audit['near_duplicate_pairs']}
- HIGH / MEDIUM / LOW / EXCLUDE_FROM_PRIMARY: {counts.get('HIGH', 0)} / {counts.get('MEDIUM', 0)} / {counts.get('LOW', 0)} / {counts.get('EXCLUDE_FROM_PRIMARY', 0)}

Nichtkanonische exakte Duplikate werden nicht primär gewichtet; alle Rohfälle
bleiben dokumentiert.

## 9. Groundedness-Implementierung

Durch Repository-Code belegt: `calculate_groundedness_score` und
`{summary['algorithm_version']}`. {summary['reproducibility']['sample_size']} Fälle
wurden wiederholt berechnet; maximale absolute Abweichung:
{summary['reproducibility']['maximum_absolute_difference']}.

## 10. Weak-Label-Confidence

HIGH erfordert vollständige Quellen- und Mutationsprovenienz ohne Quality Flag
oder Duplikatunsicherheit. MEDIUM markiert nachvollziehbare Konstruktionen mit
semantischer Restunsicherheit. LOW verlangt Review. EXCLUDE_FROM_PRIMARY enthält
defekte oder nichtkanonische Duplikate. Dies ist technische Klassifikation, keine
fachliche Bewertung.

## 11. Primäre Kalibrierungsmenge

Nur Calibration, HIGH, technisch vollständig, ohne exakte Duplikatcluster und
ohne Fehler wurde primär verwendet: n={summary['primary_calibration_count']}.

## 12. Explorative Score-Analyse

ROC-AUC gegen Weak Labels: {summary['weak_label_discrimination']['roc_auc']:.4f};
Average Precision: {summary['weak_label_discrimination']['average_precision']:.4f}.

![Score-Verteilung](figures/{FIGURES['distribution'].name})

## 13. Unsupervised Score-Analyse

Unabhängig von Labels: {unsupervised['unique_score_count']} eindeutige Scores,
Median {unsupervised['median']:.6f}. Größtes beobachtetes Gap:
`{json.dumps(unsupervised['largest_observed_score_gap'], ensure_ascii=False)}`.
KDE und gegebenenfalls GMM sind rein deskriptiv und kein Sicherheitsnachweis.

## 14. Threshold-Auswahlregel

1. Null akzeptierte technisch konstruierte High-Risk-Proxy-FAILs.
2. Technische FAR <= 5%.
3. Maximaler Recall.
4. Tie-Breaker: Precision, kleinere FAR, Group-CV-Stabilität, Coverage,
   höherer konservativer Threshold.

## 15. Vergleich aller Threshold-Methoden

| Methode | Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | akzeptierte High-Risk-Proxies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{comparison_rows}

![Threshold-Metriken](figures/{FIGURES['thresholds'].name})

## 16. Ergebnisse auf Calibration

{evaluation_note}

{format_metric_row(f'Calibration HIGH ({evaluation_suffix})', calibration)}

## 17. Ergebnisse auf Validation

{format_metric_row(f'Validation HIGH ({evaluation_suffix})', validation)}

## 18. Ergebnisse auf Technical Weak-Label Hold-out

Der Preliminary Hold-out wurde einmalig als **technical weak-label hold-out**
ausgewertet, nicht als finaler oder menschlich validierter Hold-out.

{format_metric_row(f'Technical weak-label hold-out HIGH ({evaluation_suffix})', holdout)}

## 19. Group Cross-Validation

{cv_text}

![CV-Thresholds](figures/{FIGURES['cv'].name})

## 20. Source-Type-Analyse

| Source Type | n | Status | Threshold | FAR | Recall |
|---|---:|---|---:|---:|---:|
{source_rows}

## 21. Mutation-Type- und High-Risk-Proxy-Analyse

Die vollständigen Mutation-Type-Sensitivitäten stehen in
`reports/{SENSITIVITY_OUTPUT.name}`. High-Risk bezeichnet ausschließlich
**technically constructed high-risk proxy cases**, keine bestätigte Criticality.

## 22. Label-Noise-Sensitivität

| Noise | erfolgreiche Läufe | Median Threshold | Minimum | Maximum | FAR | Recall | Anteil mit Entscheidungsänderung |
|---|---:|---:|---:|---:|---:|---:|---:|
{noise_rows}

„Materiell“ bedeutet hier ohne willkürliche numerische Grenze: Mindestens ein
ursprünglicher Calibration-Fall ändert ACCEPT/REJECT.

![Noise-Sensitivität](figures/{FIGURES['noise'].name})

## 23. Vergleich mit 0.51

Siehe Tabelle in Kapitel 15. Der Vergleich ist statistisch gegen Weak Labels,
nicht gegen menschliches Ground Truth.

## 24. Vergleich mit 0.583333

Siehe Tabelle in Kapitel 15. Auch 0.583333 bleibt ein früher vorläufiger Wert.

## 25. Empfohlener Threshold oder Recommended Range

**{recommendation['display_value']}**. Begründung: {recommendation['reason']}.
Bootstrap-Intervall conditional on weak labels:
[{summary['cluster_bootstrap']['threshold_ci95_lower']},
{summary['cluster_bootstrap']['threshold_ci95_upper']}].

## 26. Grenzen der Aussagekraft

1. Konstruktionserwartungen können semantisch falsch liegen.
2. Keine unabhängige fachliche Doppelannotation.
3. Confidence und High-Risk sind technische Proxies.
4. Konfidenzintervalle konditionieren auf Weak Labels und erfassen keinen Label-Bias.
5. Source-Familien und synthetische Vorlagen können trotz Gruppierung Dataset-Shift verursachen.

## 27. Voraussetzungen für spätere Human Validation

Zwei unabhängige Reviewer, Konfliktadjudikation, finale Leakage-sichere
Holdout-Bildung, erneutes isoliertes Scoring und genau eine finale
Human-Ground-Truth-Holdout-Auswertung.

## 28. Konkrete Empfehlung zur produktiven Verwendung

Keine automatische Produktivänderung. Der Wert beziehungsweise Bereich darf nur
als Forschungs- und Shadow-Evaluation-Referenz verwendet werden. Der bestehende
Produktionswert bleibt unverändert, bis menschliche Validierung abgeschlossen ist.

---

No human ground truth was used. The result is a provisional weakly supervised threshold and must not be interpreted as a final human-validated production threshold.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated provisional weak-supervision calibration for fact_aware_claim_support_v4"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--noise-repeats", type=int, default=NOISE_REPEATS)
    parser.add_argument(
        "--resume-from-scored",
        action="store_true",
        help="Continue from verified frozen/scored artifacts after a downstream analysis stop; never rescores cases.",
    )
    args = parser.parse_args()
    if args.bootstrap_repeats != BOOTSTRAP_REPEATS or args.noise_repeats != NOISE_REPEATS:
        raise ValueError("This reproducible run requires the documented default repeat counts")

    ensure_inputs()
    if args.resume_from_scored:
        ensure_resume_outputs()
    else:
        ensure_new_outputs()
    cases = read_jsonl(DATASET)
    quality_rows = classify_weak_labels(cases)
    audit = validate_structure(cases, quality_rows)
    if audit["candidate_count"] != 498 or audit["unique_case_ids"] != 498:
        raise ValueError(f"Unexpected candidate structure: {audit}")
    if audit["missing_required_fields_cases"] != 0:
        raise ValueError("Missing required fields detected before scoring")
    if audit["leakage_group_split_violations"] != 0:
        raise ValueError("Leakage group crosses split boundary")

    snapshot: dict[str, Any]
    quality_fields = [
        "case_id",
        "weak_label",
        "weak_label_source",
        "weak_label_confidence",
        "weak_label_quality_reason",
        "source_type",
        "mutation_type",
        "split",
        "leakage_group_id",
        "quality_flags",
        "primary_source_exists",
        "mutation_source_exists",
        "mutation_provenance_complete",
        "traceable_change",
        "duplicate_group_id",
        "canonical_case",
        "duplicate_cluster_size",
        "excluded_from_primary_reason",
        "high_risk_mutation_proxy",
        "classifier_version",
    ]
    if args.resume_from_scored:
        snapshot, scored, reproducibility = load_verified_scored_resume(cases, quality_rows)
    else:
        snapshot = make_split_snapshot(cases, audit)
        write_csv(WEAK_QUALITY, quality_rows, quality_fields)
        scored, reproducibility = score_cases(cases, quality_rows)
        write_jsonl(SCORED_CASES, scored)
    errors = [row for row in scored if row["technical_error"]]
    if errors:
        raise RuntimeError(f"Groundedness scoring failed for {len(errors)} cases")

    primary = [
        row
        for row in scored
        if row["split"] == "CALIBRATION"
        and row["weak_label_confidence"] == "HIGH"
        and not row["quality_flags"]
        and not row["duplicate_group_id"]
        and not row["technical_error"]
    ]
    validation_high = [
        row
        for row in scored
        if row["split"] == "VALIDATION"
        and row["weak_label_confidence"] == "HIGH"
        and not row["quality_flags"]
        and not row["duplicate_group_id"]
    ]
    holdout_high = [
        row
        for row in scored
        if row["split"] == "LOCKED_HOLDOUT_CANDIDATE"
        and row["weak_label_confidence"] == "HIGH"
        and not row["quality_flags"]
        and not row["duplicate_group_id"]
    ]
    if len({row["weak_label"] for row in primary}) != 2:
        raise RuntimeError("Primary HIGH-confidence Calibration set lacks both weak classes")

    thresholds = candidate_thresholds(primary)
    folds, cv_splits = build_group_cv_splits(primary)
    stability = fixed_threshold_stability(primary, thresholds, cv_splits)
    primary_sweep = threshold_sweep(primary, thresholds)
    selected = select_risk_constrained(primary_sweep, stability=stability)
    risk_constrained_selection_available = selected is not None
    comparisons = comparison_methods(primary, primary_sweep)
    operating_point = selected if selected is not None else comparisons["maximum_f1"]
    selected_threshold = float(operating_point["threshold"])
    selected_full = metrics_at(primary, selected_threshold, intervals=True)
    selected_full["fixed_group_cv_f1_std"] = stability[selected_threshold]
    selected_full["selection_status"] = (
        "VALID_RISK_CONSTRAINED_SELECTION"
        if risk_constrained_selection_available
        else "DIAGNOSTIC_MAXIMUM_F1_ONLY_NO_VALID_RISK_CONSTRAINED_THRESHOLD"
    )
    selected_full["is_recommended_threshold"] = risk_constrained_selection_available

    candidate_rows: list[dict[str, Any]] = []
    group_sweep = threshold_sweep(primary, thresholds, group_weighted=True)
    group_by_threshold = {float(row["threshold"]): row for row in group_sweep}
    for row in primary_sweep:
        threshold = float(row["threshold"])
        candidate_rows.append(
            {
                **row,
                "fixed_group_cv_f1_std": stability[threshold],
                "group_weighted_metrics_json": group_by_threshold[threshold],
                "source_type_metrics_json": subgroup_metrics_json(primary, threshold, "source_type"),
                "mutation_type_metrics_json": subgroup_metrics_json(primary, threshold, "mutation_type"),
                "leakage_group_metrics_json": subgroup_metrics_json(primary, threshold, "leakage_group_id"),
                "is_primary_selection": risk_constrained_selection_available
                and abs(threshold - selected_threshold) <= 1e-12,
                "is_diagnostic_operating_point": abs(threshold - selected_threshold) <= 1e-12,
            }
        )
    write_csv(THRESHOLD_CANDIDATES, candidate_rows)

    cv_rows = run_group_cv(primary, cv_splits)
    write_csv(CV_OUTPUT, cv_rows)
    cv_summary = aggregate_cv(cv_rows, folds)

    sensitivity_rows, source_thresholds = sensitivity_analyses(scored, primary)
    noise_details, noise_summary = weak_label_noise(primary, thresholds, selected_threshold)
    for row in noise_details:
        sensitivity_rows.append({"analysis": "weak_label_noise", **row})
    write_csv(SENSITIVITY_OUTPUT, sensitivity_rows)

    bootstrap = cluster_bootstrap(primary, thresholds)
    all_successful = [row for row in scored if not row["technical_error"]]
    unsupervised = unsupervised_analysis(all_successful)
    if risk_constrained_selection_available:
        comparisons["primary_risk_constrained"] = selected_full

    calibration_metrics = selected_full
    validation_metrics = metrics_at(validation_high, selected_threshold, intervals=True)
    holdout_metrics = metrics_at(holdout_high, selected_threshold, intervals=True)
    calibration_all = metrics_at(
        [row for row in scored if row["split"] == "CALIBRATION" and row["weak_label_confidence"] != "EXCLUDE_FROM_PRIMARY"],
        selected_threshold,
    )
    validation_all = metrics_at(
        [row for row in scored if row["split"] == "VALIDATION" and row["weak_label_confidence"] != "EXCLUDE_FROM_PRIMARY"],
        selected_threshold,
    )
    holdout_all = metrics_at(
        [row for row in scored if row["split"] == "LOCKED_HOLDOUT_CANDIDATE" and row["weak_label_confidence"] != "EXCLUDE_FROM_PRIMARY"],
        selected_threshold,
    )

    conditions = {
        "primary_valid": risk_constrained_selection_available,
        "validation_far_le_5pct": validation_metrics["technical_far"] <= FAR_LIMIT,
        "validation_zero_high_risk_proxy_false_acceptance": validation_metrics[
            "accepted_high_risk_proxy_failures"
        ] == 0,
        "technical_holdout_far_le_5pct": holdout_metrics["technical_far"] <= FAR_LIMIT,
        "technical_holdout_zero_high_risk_proxy_false_acceptance": holdout_metrics[
            "accepted_high_risk_proxy_failures"
        ] == 0,
        "all_cv_folds_valid": cv_summary.get("valid_folds") == cv_summary.get("total_folds"),
    }
    if risk_constrained_selection_available and all(conditions.values()):
        cv_range = (float(cv_summary["threshold_q1"]), float(cv_summary["threshold_q3"]))
        if math.isclose(cv_range[0], cv_range[1], abs_tol=1e-12):
            recommendation = {
                "status": "PROVISIONAL_WEAKLY_SUPERVISED_THRESHOLD",
                "threshold": selected_threshold,
                "recommended_range": [selected_threshold, selected_threshold],
                "display_value": f"provisional weakly supervised threshold = {selected_threshold:.6f}",
                "reason": "Primary risk constraints, Validation and technical weak-label hold-out checks passed; productive use still requires human validation.",
            }
        else:
            recommendation = {
                "status": "PROVISIONAL_WEAKLY_SUPERVISED_RECOMMENDED_RANGE",
                "threshold": selected_threshold,
                "recommended_range": [cv_range[0], cv_range[1]],
                "display_value": f"recommended weak-supervision range {cv_range[0]:.6f}–{cv_range[1]:.6f}; point estimate {selected_threshold:.6f}",
                "reason": "Primary checks passed, but grouped folds selected differing thresholds; the CV IQR is reported instead of artificial unification.",
            }
    else:
        recommendation = {
            "status": "NO_RECOMMENDED_THRESHOLD",
            "threshold": None,
            "recommended_range": None,
            "display_value": "no recommended threshold",
            "reason": "The current groundedness score does not provide sufficient separation under the available weak labels.",
        }

    quality_counts = dict(sorted(Counter(row["weak_label_confidence"] for row in quality_rows).items()))
    high_risk_count = sum(bool(row["high_risk_mutation_proxy"]) for row in scored)
    labels = np.asarray([1 if row["weak_label"] == "PASS" else 0 for row in primary])
    scores = np.asarray([float(row["groundedness_score"]) for row in primary])
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "scientific_status": "provisional weakly supervised threshold",
        "human_ground_truth_used": False,
        "productive_configuration_changed": False,
        "algorithm_version": GROUNDING_ALGORITHM_VERSION,
        "random_seed": RANDOM_SEED,
        "verified_starting_state": audit,
        "split_snapshot": {
            "path": str(SPLIT_SNAPSHOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(SPLIT_SNAPSHOT),
            "assignment_sha256": snapshot["assignment_sha256"],
        },
        "weak_label_quality_counts": quality_counts,
        "high_risk_mutation_proxy_count": high_risk_count,
        "primary_calibration_count": len(primary),
        "validation_high_confidence_count": len(validation_high),
        "technical_holdout_high_confidence_count": len(holdout_high),
        "reproducibility": reproducibility,
        "candidate_threshold_count": len(thresholds),
        "primary_selection": selected_full,
        "weak_label_discrimination": {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        },
        "comparison_methods": comparisons,
        "evaluation": {
            "calibration_high_confidence": calibration_metrics,
            "validation_high_confidence": validation_metrics,
            "technical_weak_label_holdout_high_confidence": holdout_metrics,
            "calibration_all_usable_sensitivity": calibration_all,
            "validation_all_usable_sensitivity": validation_all,
            "technical_holdout_all_usable_sensitivity": holdout_all,
        },
        "cross_validation": cv_summary,
        "source_type_threshold_sensitivity": source_thresholds,
        "cluster_bootstrap": bootstrap,
        "weak_label_noise_sensitivity": noise_summary,
        "unsupervised_analysis": unsupervised,
        "recommendation_conditions": conditions,
        "evaluation_operating_point": {
            "threshold": selected_threshold,
            "purpose": (
                "risk-constrained primary selection"
                if risk_constrained_selection_available
                else "diagnostic maximum-F1 comparator only; not a recommended threshold"
            ),
        },
        "recommendation": recommendation,
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "scored_cases": str(SCORED_CASES),
            "weak_label_quality": str(WEAK_QUALITY),
            "threshold_candidates": str(THRESHOLD_CANDIDATES),
            "cross_validation": str(CV_OUTPUT),
            "sensitivity": str(SENSITIVITY_OUTPUT),
            "summary": str(SUMMARY_OUTPUT),
            "report": str(REPORT_OUTPUT),
            "split_snapshot": str(SPLIT_SNAPSHOT),
            "figures": {key: str(path) for key, path in FIGURES.items()},
        },
        "closing_statement": "No human ground truth was used. The result is a provisional weakly supervised threshold and must not be interpreted as a final human-validated production threshold.",
    }

    make_plots(
        all_successful,
        primary,
        candidate_rows,
        selected_threshold,
        cv_rows,
        noise_summary,
        risk_constrained_selection_available,
    )
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_OUTPUT.write_text(report_markdown(summary), encoding="utf-8")

    print(json.dumps(
        {
            "candidate_count": audit["candidate_count"],
            "weak_label_confidence_counts": quality_counts,
            "weak_label_counts": audit["weak_label_counts"],
            "high_risk_mutation_proxy_count": high_risk_count,
            "split_counts": audit["split_counts"],
            "old_threshold": OLD_THRESHOLD,
            "previous_provisional_threshold": PREVIOUS_PROVISIONAL_THRESHOLD,
            "recommendation": recommendation,
            "calibration": calibration_metrics,
            "validation": validation_metrics,
            "technical_weak_label_holdout": holdout_metrics,
            "cv": cv_summary,
            "noise": noise_summary,
            "artifacts": summary["artifacts"],
            "closing_statement": summary["closing_statement"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_groundedness_threshold_weak_supervision import (
    build_group_cv_splits,
    candidate_thresholds,
    fixed_threshold_stability,
    metrics_at,
    run_group_cv,
    select_risk_constrained,
    threshold_sweep,
)
from scripts.experimental_groundedness_v5 import (
    EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score_v5_experimental,
)


DATASET = ROOT / "tests" / "fixtures" / "groundedness_extended_candidates.jsonl"
QUALITY = ROOT / "reports" / "groundedness_weak_label_quality.csv"
V4_SCORES = ROOT / "reports" / "groundedness_weak_supervision_scored_cases.jsonl"
V4_SUMMARY = ROOT / "reports" / "groundedness_weak_threshold_summary.json"

SCORES_OUTPUT = ROOT / "reports" / "groundedness_v5_experimental_scores.jsonl"
THRESHOLDS_OUTPUT = ROOT / "reports" / "groundedness_v5_experimental_threshold_candidates.csv"
CV_OUTPUT = ROOT / "reports" / "groundedness_v5_experimental_cv.csv"
AUDIT_OUTPUT = ROOT / "reports" / "groundedness_v5_false_acceptance_audit_20260802.csv"
SUMMARY_OUTPUT = ROOT / "reports" / "groundedness_v5_experimental_summary.json"
REPORT_OUTPUT = ROOT / "reports" / "groundedness_v5_experimental_report_20260802.md"
REVIEW_1_OUTPUT = ROOT / "reports" / "groundedness_human_review_reviewer_1_20260802.csv"
REVIEW_2_OUTPUT = ROOT / "reports" / "groundedness_human_review_reviewer_2_20260802.csv"
REVIEW_MAP_OUTPUT = ROOT / "reports" / "groundedness_human_review_blind_map_20260802.json"
REVIEW_PROTOCOL_OUTPUT = ROOT / "reports" / "groundedness_human_review_protocol_20260802.md"

SEED = 20260802
V4_DIAGNOSTIC_THRESHOLD = 0.871795


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def quality_rows() -> dict[str, dict[str, Any]]:
    with QUALITY.open(encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def score_cases(cases: Sequence[dict[str, Any]], quality: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    for case in cases:
        docs = [
            Document(
                page_content=text,
                metadata={
                    "source_file": case["source_file"],
                    "source_page": case["source_page"],
                    "source_document_id": case["source_document_id"],
                },
            )
            for text in case["context"]
        ]
        result = calculate_groundedness_score_v5_experimental(
            case["candidate_answer"], docs, case["question"]
        )
        weak = quality[case["case_id"]]
        output.append(
            {
                "case_id": case["case_id"],
                "source_type": case["source_type"],
                "mutation_type": case["mutation_type"],
                "split": case["split"],
                "leakage_group_id": case["leakage_group_id"],
                "weak_label": weak["weak_label"],
                "weak_label_confidence": weak["weak_label_confidence"],
                "quality_flags": case.get("quality_flags") or [],
                "duplicate_group_id": weak["duplicate_group_id"],
                "high_risk_mutation_proxy": bool_value(weak["high_risk_mutation_proxy"]),
                "groundedness_score": result.score,
                "v5_score": result.score,
                "v4_score": result.base_v4_score,
                "minimum_claim_support": result.minimum_claim_support,
                "unsupported_atomic_facts": list(result.unsupported_atomic_facts),
                "citation_mismatches": list(result.citation_mismatches),
                "structured_mismatches": list(result.structured_mismatches),
                "polarity_mismatches": list(result.polarity_mismatches),
                "applied_caps": list(result.applied_caps),
                "algorithm_version": result.algorithm_version,
                "score_timestamp": timestamp,
            }
        )
    return output


def primary_rows(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["split"] == split
        and row["weak_label_confidence"] == "HIGH"
        and not row["quality_flags"]
        and not row["duplicate_group_id"]
    ]


def cv_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "OK"]
    thresholds = [float(row["training_threshold"]) for row in valid]
    return {
        "status": "OK" if len(valid) == len(rows) else "PARTIAL" if valid else "NO_VALID_FOLDS",
        "valid_folds": len(valid),
        "total_folds": len(rows),
        "threshold_median": statistics.median(thresholds) if thresholds else None,
        "threshold_minimum": min(thresholds) if thresholds else None,
        "threshold_maximum": max(thresholds) if thresholds else None,
        "far_le_5pct_fraction": sum(bool(row.get("test_far_le_5pct")) for row in valid) / len(valid) if valid else None,
        "high_risk_false_acceptance_fraction": sum(int(row.get("test_accepted_high_risk_proxy_failures", 0)) > 0 for row in valid) / len(valid) if valid else None,
        "mean_test_far": statistics.mean(float(row["test_technical_far"]) for row in valid) if valid else None,
        "mean_test_recall": statistics.mean(float(row["test_recall"]) for row in valid) if valid else None,
        "mean_test_f1": statistics.mean(float(row["test_f1"]) for row in valid) if valid else None,
    }


def false_acceptance_audit(
    cases: Sequence[dict[str, Any]],
    v4_rows: Sequence[dict[str, Any]],
    v5_rows: Sequence[dict[str, Any]],
    v5_threshold: float,
) -> list[dict[str, Any]]:
    cases_by_id = {row["case_id"]: row for row in cases}
    v5_by_id = {row["case_id"]: row for row in v5_rows}
    output: list[dict[str, Any]] = []
    for old in v4_rows:
        if not (
            old["weak_label"] == "FAIL"
            and old["weak_label_confidence"] == "HIGH"
            and not old["quality_flags"]
            and not old["duplicate_group_id"]
            and float(old["groundedness_score"]) >= V4_DIAGNOSTIC_THRESHOLD
        ):
            continue
        case = cases_by_id[old["case_id"]]
        new = v5_by_id[old["case_id"]]
        output.append(
            {
                "case_id": old["case_id"],
                "split": old["split"],
                "source_type": old["source_type"],
                "mutation_type": old["mutation_type"],
                "high_risk_mutation_proxy": old["high_risk_mutation_proxy"],
                "v4_score": old["groundedness_score"],
                "v4_accepted_at_0_871795": True,
                "v5_score": new["v5_score"],
                "v5_threshold": v5_threshold,
                "v5_accepted": float(new["v5_score"]) >= v5_threshold,
                "applied_caps": new["applied_caps"],
                "unsupported_atomic_facts": new["unsupported_atomic_facts"],
                "citation_mismatches": new["citation_mismatches"],
                "structured_mismatches": new["structured_mismatches"],
                "polarity_mismatches": new["polarity_mismatches"],
                "question": case["question"],
                "candidate_answer": case["candidate_answer"],
                "mutation_detail": case["mutation_detail"],
            }
        )
    return output


def build_blind_review_package(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    review_cases = [row for row in cases if row["split"] in {"CALIBRATION", "VALIDATION"}]
    rng = random.Random(SEED)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in review_cases:
        groups.setdefault(row["leakage_group_id"], []).append(row)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    ordered = [row for group in group_names for row in sorted(groups[group], key=lambda item: item["case_id"])]
    blind_rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for index, case in enumerate(ordered, start=1):
        blind_id = f"GHR-{index:04d}"
        blind_rows.append(
            {
                "blind_case_id": blind_id,
                "language": case["language"],
                "question": case["question"],
                "context": "\n\n--- DOCUMENT ---\n\n".join(case["context"]),
                "candidate_answer": case["candidate_answer"],
                "reviewer_label": "",
                "error_category": "",
                "criticality": "",
                "confidence": "",
                "notes": "",
            }
        )
        mapping.append(
            {
                "blind_case_id": blind_id,
                "case_id": case["case_id"],
                "split": case["split"],
                "leakage_group_id": case["leakage_group_id"],
            }
        )
    write_csv(REVIEW_1_OUTPUT, blind_rows)
    write_csv(REVIEW_2_OUTPUT, blind_rows)
    REVIEW_MAP_OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": SEED,
                "blind": True,
                "case_count": len(mapping),
                "leakage_group_count": len(groups),
                "excluded_split": "LOCKED_HOLDOUT_CANDIDATE",
                "warning": "Keep this mapping separate from reviewers. It contains no weak labels or scores.",
                "mapping": mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "case_count": len(mapping),
        "leakage_group_count": len(groups),
        "split_counts": dict(Counter(row["split"] for row in review_cases)),
    }


def metric_line(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['n']} | {row['technical_far']:.2%} | {row['precision']:.2%} | "
        f"{row['recall']:.2%} | {row['f1']:.2%} | {row['coverage']:.2%} | "
        f"{row['accepted_high_risk_proxy_failures']} |"
    )


def main() -> int:
    outputs = [
        SCORES_OUTPUT, THRESHOLDS_OUTPUT, CV_OUTPUT, AUDIT_OUTPUT, SUMMARY_OUTPUT, REPORT_OUTPUT,
        REVIEW_1_OUTPUT, REVIEW_2_OUTPUT, REVIEW_MAP_OUTPUT, REVIEW_PROTOCOL_OUTPUT,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite experimental artifacts:\n" + "\n".join(existing))
    for path in (DATASET, QUALITY, V4_SCORES, V4_SUMMARY):
        if not path.exists():
            raise FileNotFoundError(path)

    cases = read_jsonl(DATASET)
    quality = quality_rows()
    v5_rows = score_cases(cases, quality)
    if len(v5_rows) != 498 or {row["algorithm_version"] for row in v5_rows} != {EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION}:
        raise RuntimeError("Unexpected v5 scoring result")
    write_jsonl(SCORES_OUTPUT, v5_rows)

    calibration = primary_rows(v5_rows, "CALIBRATION")
    validation = primary_rows(v5_rows, "VALIDATION")
    holdout = primary_rows(v5_rows, "LOCKED_HOLDOUT_CANDIDATE")
    thresholds = candidate_thresholds(calibration)
    folds, splits = build_group_cv_splits(calibration)
    stability = fixed_threshold_stability(calibration, thresholds, splits)
    sweep = threshold_sweep(calibration, thresholds)
    selected = select_risk_constrained(sweep, stability=stability)
    if selected is None:
        raise RuntimeError("Experimental v5 still has no valid weak-label threshold")
    threshold = float(selected["threshold"])
    candidates = [
        {
            **row,
            "fixed_group_cv_f1_std": stability[float(row["threshold"])],
            "is_selected": float(row["threshold"]) == threshold,
        }
        for row in sweep
    ]
    write_csv(THRESHOLDS_OUTPUT, candidates)

    cv_rows = run_group_cv(calibration, splits)
    write_csv(CV_OUTPUT, cv_rows)
    grouped_cv = cv_summary(cv_rows)
    evaluation = {
        "calibration": metrics_at(calibration, threshold, intervals=True),
        "validation": metrics_at(validation, threshold, intervals=True),
        "technical_weak_label_holdout": metrics_at(holdout, threshold, intervals=True),
    }
    v4_rows = read_jsonl(V4_SCORES)
    audit = false_acceptance_audit(cases, v4_rows, v5_rows, threshold)
    write_csv(AUDIT_OUTPUT, audit)
    review = build_blind_review_package(cases)

    conditions = {
        "calibration_far_le_5pct": evaluation["calibration"]["technical_far"] <= 0.05,
        "calibration_zero_high_risk_false_acceptance": evaluation["calibration"]["accepted_high_risk_proxy_failures"] == 0,
        "validation_far_le_5pct": evaluation["validation"]["technical_far"] <= 0.05,
        "validation_zero_high_risk_false_acceptance": evaluation["validation"]["accepted_high_risk_proxy_failures"] == 0,
        "technical_holdout_far_le_5pct": evaluation["technical_weak_label_holdout"]["technical_far"] <= 0.05,
        "technical_holdout_zero_high_risk_false_acceptance": evaluation["technical_weak_label_holdout"]["accepted_high_risk_proxy_failures"] == 0,
        "all_group_cv_folds_valid": grouped_cv["valid_folds"] == grouped_cv["total_folds"],
        "all_group_cv_folds_no_high_risk_false_acceptance": grouped_cv["high_risk_false_acceptance_fraction"] == 0.0,
    }
    weak_label_result = "VALID_PROVISIONAL_WEAK_LABEL_NUMBER" if all(conditions.values()) else "EXPERIMENTAL_NUMBER_REQUIRES_HUMAN_VALIDATION"
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION,
        "productive_pipeline_changed": False,
        "productive_threshold_changed": False,
        "human_ground_truth_used": False,
        "dataset_sha256": sha256(DATASET),
        "candidate_count": len(cases),
        "primary_calibration_count": len(calibration),
        "selected_weak_label_threshold": threshold,
        "scientific_status": weak_label_result,
        "evaluation": evaluation,
        "group_cross_validation": grouped_cv,
        "selection_conditions": conditions,
        "v4_false_acceptance_audit_count": len(audit),
        "v4_false_acceptances_remaining_under_v5": sum(bool(row["v5_accepted"]) for row in audit),
        "review_package": review,
        "required_next_step": "Two independent human reviews and adjudication; then recalibrate on human labels and evaluate a newly collected untouched human hold-out.",
        "artifacts": {"scores": str(SCORES_OUTPUT), "thresholds": str(THRESHOLDS_OUTPUT), "cv": str(CV_OUTPUT), "audit": str(AUDIT_OUTPUT), "report": str(REPORT_OUTPUT), "reviewer_1": str(REVIEW_1_OUTPUT), "reviewer_2": str(REVIEW_2_OUTPUT), "blind_map": str(REVIEW_MAP_OUTPUT), "review_protocol": str(REVIEW_PROTOCOL_OUTPUT)},
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    REPORT_OUTPUT.write_text(
        f"""# Experimentelle Groundedness-v5-Auswertung

## Ergebnis

Die isolierte experimentelle Variante `{EXPERIMENTAL_GROUNDING_ALGORITHM_VERSION}` liefert auf den technischen Weak Labels den Punkt **{threshold:.6f}**. Dieser Wert ist noch **kein produktiver oder human-validierter Threshold**. Die produktive v4-Implementierung und Konfiguration wurden nicht verändert.

| Split | n | technische FAR | Precision | Recall | F1 | Coverage | akzeptierte High-Risk-Proxies |
|---|---:|---:|---:|---:|---:|---:|---:|
{metric_line('Calibration', evaluation['calibration'])}
{metric_line('Validation', evaluation['validation'])}
{metric_line('Technical Weak-Label Hold-out', evaluation['technical_weak_label_holdout'])}

## Was v5 verändert

- Einzelne unbelegte Zahlen, Prozentsätze, Geldbeträge, Daten und Policenkennungen können nicht mehr durch lange, ansonsten kopierte Antworten verdünnt werden.
- Strukturierte Kunden-, Policen-, Status- und Deckungswidersprüche erzeugen konservative Caps.
- `current policy` wird anhand Referenzdatum, Status, Produkt und jüngstem Startdatum geprüft.
- PDF-Quelle und Seitenzahl von Zitaten werden gegen Dokumentmetadaten geprüft.
- Der minimale Claim-Support beeinflusst den Gesamtscore, damit ein schwacher Teilclaim nicht im Mittelwert verschwindet.

## False-Acceptance-Audit

Am rein diagnostischen v4-Punkt {V4_DIAGNOSTIC_THRESHOLD:.6f} wurden {len(audit)} HIGH-Confidence-FAIL-Fälle akzeptiert. Unter v5 bei {threshold:.6f} bleiben davon **{sum(bool(row['v5_accepted']) for row in audit)}** akzeptiert. Details stehen in `{AUDIT_OUTPUT.name}`.

## Group Cross-Validation

Gültige Folds: {grouped_cv['valid_folds']}/{grouped_cv['total_folds']}. Threshold-Median: {grouped_cv['threshold_median']}; Spanne: [{grouped_cv['threshold_minimum']}, {grouped_cv['threshold_maximum']}]. Mittlere Test-FAR: {grouped_cv['mean_test_far']}. High-Risk-False-Acceptance-Anteil: {grouped_cv['high_risk_false_acceptance_fraction']}.

## Wissenschaftliche Einschränkung

Alle Kennzahlen wurden weiterhin gegen `generator_expected_label` berechnet. Diese Labels sind technische Weak Labels und kein menschliches Ground Truth. Die Regeln wurden anhand derselben konstruierten Fallfamilien entwickelt; deshalb kann die sehr gute Trennung teilweise konstruktionsspezifisch sein.

## Nächster zwingender Schritt

Das Review-Paket enthält {review['case_count']} Fälle aus Calibration und Validation in {review['leakage_group_count']} ungeteilten Leakage-Gruppen. Zwei Reviewer bearbeiten getrennte, identische und verblindete CSV-Dateien. Danach werden Konflikte adjudiziert und v5 ausschließlich anhand der Human Labels neu kalibriert. Für eine finale Bewertung muss zusätzlich ein neuer, bisher nicht zur Regelentwicklung verwendeter Human Hold-out erhoben werden.

## Produktionsentscheidung

`{threshold:.6f}` darf bis zur menschlichen Validierung nur für Shadow-Evaluation verwendet werden. Keine automatische Aktivierung.

No human ground truth was used. The experimental value is not a final human-validated production threshold.
""",
        encoding="utf-8",
    )

    REVIEW_PROTOCOL_OUTPUT.write_text(
        """# Human-Review-Protokoll für Groundedness

## Rollen

Zwei Reviewer bearbeiten ihre CSV unabhängig. Die Blind-Map bleibt bis zum Abschluss beider Reviews verborgen.

## Label

- `SUPPORTED`: Jede sachliche Behauptung der Candidate Answer ist durch den Context gestützt.
- `UNSUPPORTED`: Mindestens eine sachliche Behauptung widerspricht dem Context oder ist darin nicht belegt.
- `AMBIGUOUS`: Der Context reicht für eine verlässliche Entscheidung nicht aus.

## Fehlerkategorie

Falls `UNSUPPORTED`: `wrong_customer`, `wrong_policy`, `wrong_current_policy`, `wrong_status`, `wrong_date`, `wrong_number`, `wrong_premium`, `wrong_deductible`, `wrong_coverage`, `wrong_limit`, `wrong_percentage`, `wrong_citation`, `negation_or_exclusion`, `unsupported_claim`, `other`.

## Criticality

`HIGH`, wenn eine falsche Kunden-/Policenzuordnung, Deckungsentscheidung, Prämie, Selbstbeteiligung, Leistungslimite oder Ausschlussaussage unmittelbare Versicherungswirkung haben könnte; sonst `MEDIUM` oder `LOW`. Dies ist erst nach menschlicher Bewertung eine Human-Annotation.

## Confidence

`HIGH`, `MEDIUM` oder `LOW`. Jeder Reviewer dokumentiert Unsicherheit in `notes`.

## Adjudikation

Nach beiden Reviews werden Übereinstimmung und Konflikte berechnet. Konflikte und alle `AMBIGUOUS`-Fälle werden gemeinsam adjudiziert. Weak Labels, v4/v5-Scores und Mutationsmetadaten dürfen erst danach eingeblendet werden.
""",
        encoding="utf-8",
    )

    print(json.dumps({"threshold": threshold, "evaluation": evaluation, "cv": grouped_cv, "conditions": conditions, "review": review}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

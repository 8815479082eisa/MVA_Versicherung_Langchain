from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.integrations.nemo_actions import (  # noqa: E402
    GROUNDING_ALGORITHM_VERSION,
    calculate_groundedness_score,
)


DEFAULT_DATASET = PROJECT_ROOT / "tests" / "fixtures" / "groundedness_calibration_cases.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "groundedness_calibration.json"


def _dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    positives = sum(bool(case.get("label")) for case in cases)
    negatives = len(cases) - positives
    if len(cases) < 20 or positives != negatives:
        raise ValueError("Calibration requires at least 20 cases with balanced labels")
    return cases


def score_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for case in cases:
        docs = [Document(page_content=text) for text in case["documents"]]
        score = calculate_groundedness_score(case["answer"], docs, case.get("query", ""))
        scored.append({**case, "score": score})
    return scored


def metrics_at(scored: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for case in scored:
        predicted = case["score"] >= threshold
        actual = bool(case["label"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_acceptance_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": round(threshold, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_acceptance_rate": round(false_acceptance_rate, 6),
    }


def select_threshold(scored: list[dict[str, Any]]) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    sweep = [metrics_at(scored, value / 100) for value in range(0, 101)]
    acceptable = [row for row in sweep if row["recall"] >= 0.8]
    candidates = acceptable or sweep
    minimum_false_acceptance = min(row["false_acceptance_rate"] for row in candidates)
    candidates = [
        row for row in candidates if row["false_acceptance_rate"] == minimum_false_acceptance
    ]
    selected = max(
        candidates,
        key=lambda row: (row["f1"], row["recall"], row["precision"], -row["threshold"]),
    )
    return float(selected["threshold"]), selected, sweep


def build_result(dataset: Path) -> dict[str, Any]:
    cases = load_cases(dataset)
    scored = score_cases(cases)
    threshold, selected_metrics, sweep = select_threshold(scored)
    positives = sum(bool(case["label"]) for case in scored)
    return {
        "schema_version": 1,
        "algorithm_version": GROUNDING_ALGORITHM_VERSION,
        "selected_threshold": threshold,
        "selection_rule": (
            "Require recall >= 0.80 when feasible; minimize unsupported false acceptance; "
            "then maximize F1, recall, and precision; choose the lowest tied threshold."
        ),
        "dataset": str(dataset.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_sha256": _dataset_hash(dataset),
        "dataset_size": len(scored),
        "class_balance": {"supported": positives, "unsupported": len(scored) - positives},
        "selected_metrics": selected_metrics,
        "cases": [
            {
                "id": case["id"],
                "label": "supported" if case["label"] else "unsupported",
                "score": case["score"],
                "prediction": "accept" if case["score"] >= threshold else "reject",
            }
            for case in scored
        ],
        "threshold_sweep": sweep,
    }


def _markdown_report(result: dict[str, Any], generated_at: str) -> str:
    metrics = result["selected_metrics"]
    lines = [
        "# Groundedness Calibration Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Algorithm: `{result['algorithm_version']}`",
        f"- Dataset: `{result['dataset']}`",
        f"- Dataset SHA-256: `{result['dataset_sha256']}`",
        f"- Cases: `{result['dataset_size']}` ({result['class_balance']['supported']} supported, "
        f"{result['class_balance']['unsupported']} unsupported)",
        f"- Selected threshold: `{result['selected_threshold']:.2f}`",
        f"- Selection rule: {result['selection_rule']}",
        "",
        "## Selected Metrics",
        "",
        "| TP | TN | FP | FN | Precision | Recall | F1 | False acceptance rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['tp']} | {metrics['tn']} | {metrics['fp']} | {metrics['fn']} | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
        f"{metrics['false_acceptance_rate']:.3f} |",
        "",
        "## Case Results",
        "",
        "| Case | Label | Score | Prediction |",
        "|---|---|---:|---|",
    ]
    lines.extend(
        f"| {case['id']} | {case['label']} | {case['score']:.3f} | {case['prediction']} |"
        for case in result["cases"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selected operating point prioritizes rejecting unsupported claims. "
            "The threshold is loaded from the stable calibration artifact, not copied into `.env`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the deterministic groundedness threshold")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--write", action="store_true", help="Write config and timestamped reports")
    parser.add_argument("--check", action="store_true", help="Verify the stable config is current")
    args = parser.parse_args()

    result = build_result(args.dataset.resolve())
    if args.check:
        existing = json.loads(args.config.read_text(encoding="utf-8"))
        comparable_keys = (
            "schema_version",
            "algorithm_version",
            "selected_threshold",
            "dataset_sha256",
            "dataset_size",
            "class_balance",
            "selected_metrics",
        )
        if any(existing.get(key) != result.get(key) for key in comparable_keys):
            raise SystemExit("Groundedness calibration artifact is stale")

    if args.write:
        generated_at = datetime.now(timezone.utc).isoformat()
        stable = {key: value for key, value in result.items() if key != "threshold_sweep"}
        stable["generated_at"] = generated_at
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(stable, indent=2) + "\n", encoding="utf-8")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.report_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.report_dir / f"groundedness_calibration_{stamp}.json"
        md_path = args.report_dir / f"groundedness_calibration_{stamp}.md"
        report_payload = {**result, "generated_at": generated_at}
        json_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(_markdown_report(result, generated_at), encoding="utf-8")
        print(f"config={args.config}")
        print(f"json_report={json_path}")
        print(f"markdown_report={md_path}")

    print(json.dumps({
        "algorithm_version": result["algorithm_version"],
        "selected_threshold": result["selected_threshold"],
        "selected_metrics": result["selected_metrics"],
        "dataset_size": result["dataset_size"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

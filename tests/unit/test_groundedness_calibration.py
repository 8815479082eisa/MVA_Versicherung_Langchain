from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config.models import load_groundedness_calibration, load_model_settings
from src.guardrails.integrations.nemo_actions import calculate_groundedness_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CONFIG = PROJECT_ROOT / "config" / "groundedness_calibration.json"


def test_v5_calibration_config_is_selected() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_CONFIG)

    assert calibration is not None
    assert calibration["algorithm_version"] == "fact_aware_claim_support_v5"
    assert calibration["selected_threshold"] == 0.7888
    assert calibration["dataset_size"] >= 20


def test_runtime_loads_calibrated_threshold_and_source() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_CONFIG)
    settings = load_model_settings()

    assert calibration is not None
    assert settings.safety.min_groundedness == calibration["selected_threshold"]
    assert settings.safety.min_groundedness_source == "calibration_file"
    assert settings.safety.groundedness_calibration_file == CALIBRATION_CONFIG.resolve()


def test_fact_aware_score_accepts_supported_answer_and_rejects_wrong_amount() -> None:
    docs = [
        Document(
            page_content=(
                "Motor contract TEST-KFZ-2026-1001 covers windshield glass damage under "
                "partial comprehensive insurance with a EUR 150 deductible per claim."
            )
        )
    ]
    query = "Is windshield damage covered and what deductible applies?"
    supported = calculate_groundedness_score(
        "The damage is covered under partial comprehensive insurance with EUR 150 per claim.",
        docs,
        query,
    )
    unsupported = calculate_groundedness_score(
        "The damage is covered under partial comprehensive insurance with EUR 500 per claim.",
        docs,
        query,
    )
    threshold = load_model_settings().safety.min_groundedness

    assert supported >= threshold
    assert unsupported < threshold


def test_fact_aware_score_rejects_wrong_contract_number() -> None:
    docs = [
        Document(
            page_content=(
                "The applicable motor insurance contract is TEST-KFZ-2026-1001 and it "
                "covers windshield glass damage."
            )
        )
    ]
    score = calculate_groundedness_score(
        "The applicable motor insurance contract is TEST-KFZ-2026-9999.",
        docs,
        "Which motor contract applies?",
    )

    assert score < load_model_settings().safety.min_groundedness

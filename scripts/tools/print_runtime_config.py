from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.models import build_answer_model_warning, load_model_settings, runtime_config_snapshot


def main() -> int:
    settings = load_model_settings()
    runtime = runtime_config_snapshot(settings)

    payload = {
        "ANSWER_MODEL": runtime["raw_env"].get("ANSWER_MODEL"),
        "PREFERRED_ANSWER_MODEL": runtime["raw_env"].get("PREFERRED_ANSWER_MODEL"),
        "OLLAMA_MODEL": runtime["raw_env"].get("OLLAMA_MODEL"),
        "ROUTER_MODEL": runtime["raw_env"].get("ROUTER_MODEL"),
        "SAFETY_BACKEND": runtime["raw_env"].get("SAFETY_BACKEND"),
        "NEMO_ENFORCE_OUTPUT": runtime["raw_env"].get("NEMO_ENFORCE_OUTPUT"),
        "QUERY_REWRITE_ENABLED": runtime["raw_env"].get("QUERY_REWRITE_ENABLED"),
        "INSURANCEQA_EXACT_MATCH_SHORTCUT": runtime["raw_env"].get(
            "INSURANCEQA_EXACT_MATCH_SHORTCUT"
        ),
        "resolved_configured_answer_model": runtime["configured_answer_model"],
        "resolved_configured_answer_model_source": runtime["configured_answer_model_source"],
        "resolved_configured_answer_model_source_detail": runtime.get(
            "configured_answer_model_source_detail"
        ),
        "resolved_preferred_answer_model": runtime["preferred_answer_model"],
        "resolved_preferred_answer_model_source": runtime["preferred_answer_model_source"],
        "answer_model_matches_preference": runtime["answer_model_matches_preference"],
        "query_rewrite_enabled": runtime["query_rewrite_enabled"],
        "nemo_enforce_output": runtime["nemo_enforce_output"],
        "safety_backend": runtime["safety_backend"],
        "answer_model_warning": build_answer_model_warning(settings),
        "dotenv_conflicts": runtime.get("dotenv_conflicts", {}),
        "dotenv_path": runtime.get("dotenv_path"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.security_eval import (
    evaluate_security_cases,
    load_security_cases,
    write_security_rows,
)


def main():
    # Load .env explicitly so evaluation uses the same runtime config as the app
    load_dotenv(override=True)

    input_file = Path(
        os.getenv(
            "INPUT_QUERY_FILE",
            "data/benchmarks/safety/direct_query_attacks.jsonl",
        )
    )
    output_dir = Path(
        os.getenv(
            "OUTPUT_DIR",
            "docs/test-results/test-result-temp",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "eval_input_query_safety_results.jsonl"
    summary_file = output_dir / "summary.json"
    metadata_file = output_dir / "metadata.json"

    cases = load_security_cases(input_file)
    rows, summary, metadata = evaluate_security_cases(cases)

    write_security_rows(output_file, rows)

    summary_payload = {
        "total_queries": summary["total_queries"],
        "total_attack": summary["total_attack"],
        "total_benign": summary["total_benign"],
        "stopped_attack": summary["stopped_attack"],
        "allowed_benign": summary["allowed_benign"],
        "ABR": summary["attack_block_rate"],
        "BAR": summary["benign_allow_rate"],
    }

    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()

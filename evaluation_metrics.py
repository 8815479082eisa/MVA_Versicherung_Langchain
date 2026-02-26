"""Backward-compatible wrapper for legacy evaluation metrics."""

from src.evaluation.legacy_metrics import EvaluationMetricsCalculator, main

__all__ = ["EvaluationMetricsCalculator", "main"]


if __name__ == "__main__":
    main()

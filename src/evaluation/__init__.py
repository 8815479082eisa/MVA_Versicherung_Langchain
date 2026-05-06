"""Evaluation utilities and metrics.

Import the lightweight helpers eagerly and degrade gracefully when optional
evaluation dependencies are unavailable in the current environment.
"""

from .benchmark_utils import (
    best_token_f1,
    exact_match,
    load_question_answer_map,
    load_question_answer_pairs,
    normalize_text,
    sample_question_answer_pairs,
    slice_question_answer_pairs,
    token_f1,
)
from .security_eval import derive_case_label, evaluate_security_cases, summarize_security_rows
from .thesis_metrics import aggregate_qa_metric_rows, build_qa_metric_row, lexical_support_score

_LAZY_EXPORTS = {
    "NLIModel": (".nli", "NLIModel"),
    "create_nli_model": (".nli", "create_nli_model"),
    "aggregate_alce_results": (".alce_metrics", "aggregate_alce_results"),
    "citation_coverage": (".alce_metrics", "citation_coverage"),
    "citation_precision": (".alce_metrics", "citation_precision"),
    "citation_recall": (".alce_metrics", "citation_recall"),
    "extract_citation_indices": (".alce_metrics", "extract_citation_indices"),
    "mauve_fluency": (".alce_metrics", "mauve_fluency"),
    "split_into_statements": (".alce_metrics", "split_into_statements"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "NLIModel",
    "aggregate_alce_results",
    "aggregate_qa_metric_rows",
    "best_token_f1",
    "citation_coverage",
    "citation_precision",
    "citation_recall",
    "create_nli_model",
    "derive_case_label",
    "evaluate_security_cases",
    "exact_match",
    "extract_citation_indices",
    "lexical_support_score",
    "load_question_answer_map",
    "load_question_answer_pairs",
    "mauve_fluency",
    "normalize_text",
    "sample_question_answer_pairs",
    "split_into_statements",
    "slice_question_answer_pairs",
    "summarize_security_rows",
    "token_f1",
    "build_qa_metric_row",
]

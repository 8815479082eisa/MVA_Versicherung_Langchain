from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import src.config.models as config_models


def _load_evaluate_thesis_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "evaluate_thesis.py"
    spec = importlib.util.spec_from_file_location("evaluate_thesis_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_run_metadata_documents_model_and_safety_config() -> None:
    module = _load_evaluate_thesis_module()
    settings = config_models.load_model_settings()
    settings = replace(
        settings,
        preferred_answer_model="qwen3.5:4b",
        roles=replace(settings.roles, answer="qwen3.5:4b", router="functiongemma:270m"),
        retrieval=replace(settings.retrieval, query_rewrite_enabled=False, query_rewrite_min_similarity=0.85),
        safety=replace(settings.safety, min_groundedness=0.35, nemo_enforce_output=False),
    )

    args = SimpleNamespace(
        mode="full",
        qa_mode="audit",
        dataset_jsonl="qa.jsonl",
        max_samples=20,
        start=0,
        count=None,
        audit_log="audit.log",
        audit_limit=200,
        safety_file="safety.jsonl",
        support_threshold=0.2,
        allow_dataset_shortcut=False,
    )

    metadata = module.build_run_metadata(args=args, settings=settings, run_id="run-1")

    assert metadata["configured_answer_model"] == "qwen3.5:4b"
    assert metadata["preferred_answer_model"] == settings.preferred_answer_model
    assert metadata["answer_model_matches_preference"] is True
    assert metadata["query_rewrite_enabled"] is False
    assert metadata["nemo_enforce_output"] is False
    assert metadata["safety_min_groundedness"] == 0.35


def test_load_model_settings_keeps_answer_model_on_preferred_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config_models, "_DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("OLLAMA_MODEL", "functiongemma:270m")
    monkeypatch.setenv("PREFERRED_ANSWER_MODEL", "qwen3.5:4b")
    monkeypatch.delenv("ANSWER_MODEL", raising=False)
    monkeypatch.delenv("ROUTER_MODEL", raising=False)

    settings = config_models.load_model_settings()

    assert settings.roles.answer == "qwen3.5:4b"
    assert settings.roles.router == "functiongemma:270m"


def test_openai_answer_provider_keeps_auxiliary_roles_on_ollama(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(config_models, "_DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("ANSWER_PROVIDER", "openai")
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini")
    monkeypatch.delenv("ANSWER_MODEL", raising=False)
    monkeypatch.delenv("PREFERRED_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("GUARDRAIL_MODEL", raising=False)
    monkeypatch.delenv("SELF_CHECK_PROVIDER", raising=False)
    monkeypatch.delenv("SELF_CHECK_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = config_models.load_model_settings()
    runtime = config_models.runtime_config_snapshot(settings)

    assert settings.provider == "openai"
    assert settings.roles.answer == "gpt-4o-mini"
    assert settings.roles.router == "phi3:mini"
    assert settings.roles.guardrail == "phi3:mini"
    assert settings.openai_api_key_configured is False
    assert runtime["openai_api_key_configured"] is False
    assert "OPENAI_API_KEY" not in runtime["raw_env"]


def test_openai_self_check_provider_is_explicit_and_isolated(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(config_models, "_DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("ANSWER_PROVIDER", "openai")
    monkeypatch.setenv("SELF_CHECK_PROVIDER", "openai")
    monkeypatch.setenv("SELF_CHECK_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini")
    monkeypatch.delenv("ROUTER_MODEL", raising=False)

    settings = config_models.load_model_settings()
    runtime = config_models.runtime_config_snapshot(settings)

    assert settings.provider == "openai"
    assert settings.retrieval.self_check_provider == "openai"
    assert settings.roles.self_check == "gpt-4o-mini"
    assert settings.roles.router == "phi3:mini"
    assert runtime["self_check_provider"] == "openai"
    assert runtime["self_check_model"] == "gpt-4o-mini"


def test_shell_env_answer_model_overrides_dotenv(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "ANSWER_MODEL=functiongemma:270m\nPREFERRED_ANSWER_MODEL=functiongemma:270m\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_models, "_DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("ANSWER_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("PREFERRED_ANSWER_MODEL", "qwen3.5:4b")

    settings = config_models.load_model_settings()
    runtime = config_models.runtime_config_snapshot(settings)

    assert settings.roles.answer == "qwen3.5:4b"
    assert runtime["configured_answer_model_source"] == "shell_env"
    assert runtime["dotenv_conflicts"]["ANSWER_MODEL"]["dotenv"] == "functiongemma:270m"


def test_runtime_config_redacts_sensitive_dotenv_conflicts(monkeypatch, tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=dotenv-secret\n", encoding="utf-8")
    monkeypatch.setattr(config_models, "_DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("OPENAI_API_KEY", "shell-secret")

    runtime = config_models.runtime_config_snapshot(
        config_models.load_model_settings()
    )

    assert runtime["dotenv_conflicts"]["OPENAI_API_KEY"] == {
        "shell_env": "<redacted>",
        "dotenv": "<redacted>",
    }


def test_build_run_metadata_marks_answer_model_mismatch() -> None:
    module = _load_evaluate_thesis_module()
    settings = config_models.load_model_settings()
    settings = replace(
        settings,
        preferred_answer_model="qwen3.5:4b",
        roles=replace(settings.roles, answer="functiongemma:270m"),
    )
    args = SimpleNamespace(
        mode="full",
        qa_mode="live",
        dataset_jsonl="qa.jsonl",
        max_samples=200,
        start=0,
        count=200,
        audit_log="audit.log",
        audit_limit=200,
        safety_file="safety.jsonl",
        support_threshold=0.2,
        allow_dataset_shortcut=False,
    )

    metadata = module.build_run_metadata(args=args, settings=settings, run_id="run-2")

    assert metadata["answer_model_matches_preference"] is False
    assert "functiongemma:270m" in str(metadata["answer_model_warning"])


def test_mark_run_completion_marks_incomplete_runs() -> None:
    module = _load_evaluate_thesis_module()
    metadata = {
        "expected_qa_count": 200,
        "selected_qa_count": 20,
        "processed_qa_count": 20,
        "expected_safety_count": 200,
        "processed_safety_count": 200,
    }

    module.mark_run_completion(metadata)

    assert metadata["run_complete"] is False
    assert metadata["run_status"] == "incomplete"
    assert any("expected 200, selected 20" in reason for reason in metadata["run_incomplete_reasons"])


def test_git_metadata_is_robust_when_git_is_unavailable(monkeypatch, tmp_path) -> None:
    module = _load_evaluate_thesis_module()

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", _raise_file_not_found)

    git_commit = module.get_git_commit()
    git_branch = module.get_git_branch()
    module.write_git_metadata(tmp_path, git_commit=git_commit, git_branch=git_branch)

    assert git_commit == "git_not_available"
    assert git_branch == "git_not_available"
    assert (tmp_path / "commit-hash.txt").read_text(encoding="utf-8") == "git_not_available"
    assert (tmp_path / "branch.txt").read_text(encoding="utf-8") == "git_not_available"

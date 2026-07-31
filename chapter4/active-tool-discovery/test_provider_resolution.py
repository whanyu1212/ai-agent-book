"""Offline tests for active-tool-discovery's shared chat backend setup."""

import sys
from types import SimpleNamespace

import demo
import pytest
from offline_backend import LocalEmbedder

PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def openai_constructor(monkeypatch):
    import openai

    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(openai, "OpenAI", construct)
    return calls


def test_direct_openai_uses_shared_backend_and_openai_embeddings(
    monkeypatch, openai_constructor
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    backend, client, embedder = demo._create_online_backend(
        "gpt-5.6-luna", "test-embedding-model"
    )

    assert backend.using_openrouter is False
    assert backend.model == "gpt-5.6-luna"
    assert openai_constructor == [{
        "api_key": "test-openai-key",
        "base_url": "https://gateway.example/v1",
    }]
    assert embedder.client is client
    assert embedder.name == "test-embedding-model"


def test_openrouter_fallback_uses_local_embeddings(monkeypatch, openai_constructor):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://gateway.example/v1")

    backend, _, embedder = demo._create_online_backend("gpt-4o", "unused-embedding-model")

    assert backend.using_openrouter is True
    assert backend.model == "openai/gpt-4o"
    assert openai_constructor == [{
        "api_key": "test-openrouter-key",
        "base_url": "https://gateway.example/v1",
    }]
    assert isinstance(embedder, LocalEmbedder)


def test_openrouter_fallback_keeps_unknown_model_name(monkeypatch, openai_constructor):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    backend, _, _ = demo._create_online_backend("private-model-v1", "unused-embedding-model")

    assert backend.model == "private-model-v1"


def test_missing_credentials_raise_actionable_error():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        demo._create_online_backend("gpt-4o", "test-embedding-model")


def test_cli_missing_credentials_only_suggests_supported_options(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["demo.py", "--strategies", "full"])

    with pytest.raises(SystemExit) as exc:
        demo.main()

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "OPENAI_API_KEY" in output
    assert "OPENROUTER_API_KEY" in output
    assert "--model" in output
    assert "ollama" not in output

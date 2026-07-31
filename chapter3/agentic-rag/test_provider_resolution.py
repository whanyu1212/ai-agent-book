"""Offline parity tests for agentic-rag provider resolution."""

import pytest

from config import LLMConfig


PROVIDER_ENV_VARS = (
    "SILICONFLOW_API_KEY",
    "ARK_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "KIMI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "provider,key_var,base_url,model",
    [
        (
            "siliconflow",
            "SILICONFLOW_API_KEY",
            "https://api.siliconflow.cn/v1",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
        ),
        (
            "doubao",
            "ARK_API_KEY",
            "https://ark.cn-beijing.volces.com/api/v3",
            "doubao-seed-1-6-thinking-250715",
        ),
        ("kimi", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "kimi-k3"),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "openai/gpt-5.6-luna",
        ),
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-5.6-luna"),
        (
            "groq",
            "GROQ_API_KEY",
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
        ),
        (
            "together",
            "TOGETHER_API_KEY",
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com",
            "deepseek-reasoner",
        ),
    ],
)
def test_direct_provider_defaults(monkeypatch, provider, key_var, base_url, model):
    monkeypatch.setenv(key_var, f"test-{provider}-key")
    backend = LLMConfig(provider=provider).resolve_backend()
    assert backend.api_key == f"test-{provider}-key"
    assert backend.base_url == base_url
    assert backend.model == model
    assert backend.using_openrouter is (provider == "openrouter")


def test_moonshot_alias_accepts_legacy_kimi_key(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    backend = LLMConfig(provider="moonshot").resolve_backend()
    assert backend.provider == "kimi"
    assert backend.api_key == "test-kimi-key"
    assert backend.model == "kimi-k3"


def test_kimi_falls_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    backend = LLMConfig(provider="kimi").resolve_backend()
    assert backend.using_openrouter is True
    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.model == "moonshotai/kimi-k2.6"


def test_explicit_key_and_model_override_defaults():
    backend = LLMConfig(
        provider="groq",
        api_key="test-explicit-key",
        model="llama-3.1-8b-instant",
    ).resolve_backend()
    assert backend.api_key == "test-explicit-key"
    assert backend.model == "llama-3.1-8b-instant"
    assert backend.using_openrouter is False


def test_together_namespaces_bare_model_override():
    backend = LLMConfig(
        provider="together",
        api_key="test-together-key",
        model="gpt-oss-120b",
    ).resolve_backend()
    assert backend.model == "openai/gpt-oss-120b"
    assert backend.using_openrouter is False


def test_explicit_openai_provider_remains_direct(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    backend = LLMConfig(provider="openai").resolve_backend()
    assert backend.base_url == "https://api.openai.com/v1"
    assert backend.api_key == "test-openai-key"
    assert backend.using_openrouter is False


def test_fallback_does_not_silently_substitute_unknown_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
    backend = LLMConfig(
        provider="doubao",
        model="doubao-seed-1-6-thinking-250715",
    ).resolve_backend()
    assert backend.using_openrouter is True
    assert backend.model == "doubao-seed-1-6-thinking-250715"


def test_missing_credentials_raise_actionable_error():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        LLMConfig(provider="groq").resolve_backend()


def test_unknown_provider_lists_supported_names():
    with pytest.raises(ValueError, match="Supported:"):
        LLMConfig(provider="not-a-provider").resolve_backend()

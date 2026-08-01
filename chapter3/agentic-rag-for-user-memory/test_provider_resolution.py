"""Compatibility contracts for user-memory LLM provider setup."""

import os

import pytest
import quickstart
from config import LLMConfig

PROVIDER_ENV_VARS = (
    "SILICONFLOW_API_KEY",
    "DOUBAO_API_KEY",
    "ARK_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
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


def test_kimi_uses_legacy_kimi_key_and_experiment_default(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")

    client_config, model = LLMConfig(provider="kimi").get_client_config()

    assert client_config == {
        "api_key": "test-kimi-key",
        "base_url": "https://api.moonshot.cn/v1",
    }
    assert model == "kimi-k3"


@pytest.mark.parametrize(
    "provider,key_var,base_url,model,canonical",
    [
        (
            "siliconflow",
            "SILICONFLOW_API_KEY",
            "https://api.siliconflow.cn/v1",
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "siliconflow",
        ),
        (
            "doubao",
            "ARK_API_KEY",
            "https://ark.cn-beijing.volces.com/api/v3",
            "doubao-seed-1-6-thinking-250715",
            "doubao",
        ),
        ("moonshot", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "kimi-k3", "kimi"),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "openai/gpt-5.6-luna",
            "openrouter",
        ),
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-5.6-luna", "openai"),
        (
            "groq",
            "GROQ_API_KEY",
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
            "groq",
        ),
        (
            "together",
            "TOGETHER_API_KEY",
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together",
        ),
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com",
            "deepseek-reasoner",
            "deepseek",
        ),
    ],
)
def test_direct_provider_defaults(
    monkeypatch, provider, key_var, base_url, model, canonical
):
    monkeypatch.setenv(key_var, f"test-{provider}-key")

    backend = LLMConfig(provider=provider).resolve_backend()

    assert backend.api_key == f"test-{provider}-key"
    assert backend.base_url == base_url
    assert backend.model == model
    assert backend.provider == canonical
    assert backend.using_openrouter is (canonical == "openrouter")


def test_doubao_uses_documented_legacy_key(monkeypatch):
    monkeypatch.setenv("DOUBAO_API_KEY", "test-doubao-key")

    client_config, model = LLMConfig(provider="doubao").get_client_config()

    assert client_config == {
        "api_key": "test-doubao-key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    }
    assert model == "doubao-seed-1-6-thinking-250715"


def test_doubao_also_accepts_shared_ark_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")

    backend = LLMConfig(provider="doubao").resolve_backend()

    assert backend.api_key == "test-ark-key"
    assert backend.base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_doubao_prefers_ark_key_over_legacy_key(monkeypatch):
    monkeypatch.setenv("DOUBAO_API_KEY", "test-doubao-key")
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")

    backend = LLMConfig(provider="doubao").resolve_backend()

    assert backend.api_key == "test-ark-key"


def test_quickstart_accepts_ark_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ARK_API_KEY", "test-ark-key")

    assert quickstart.check_environment() is True


def test_quickstart_loads_dotenv_before_key_preflight(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-openai-key\nARK_API_KEY=test-ark-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quickstart, "run_quick_demo", lambda: None)

    try:
        quickstart.main()
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ARK_API_KEY", None)


def test_openrouter_fallback_keeps_documented_model_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    client_config, model = LLMConfig(provider="kimi").get_client_config()

    assert client_config == {
        "api_key": "test-openrouter-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert model == "google/gemma-4-31b-it:free"


def test_kimi_fallback_maps_the_experiment_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    client_config, model = LLMConfig(provider="kimi").get_client_config()

    assert client_config["api_key"] == "test-openrouter-key"
    assert client_config["base_url"] == "https://openrouter.ai/api/v1"
    assert model == "moonshotai/kimi-k2.6"


def test_openrouter_fallback_keeps_unknown_model_name(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    backend = LLMConfig(provider="doubao", model="private-model-v1").resolve_backend()

    assert backend.using_openrouter is True
    assert backend.model == "private-model-v1"


@pytest.mark.parametrize("provider,key_var", [
    ("openrouter", "OPENROUTER_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
])
def test_openrouter_model_override_does_not_change_direct_provider_model(
    monkeypatch, provider, key_var
):
    monkeypatch.setenv(key_var, f"test-{provider}-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    backend = LLMConfig(provider=provider, model="private-model-v1").resolve_backend()

    assert backend.model == "private-model-v1"


@pytest.mark.parametrize("provider,key_var", [
    ("openrouter", "OPENROUTER_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
])
@pytest.mark.parametrize("override", ["", "   "])
def test_empty_openrouter_model_does_not_clear_direct_provider_model(
    monkeypatch, provider, key_var, override
):
    monkeypatch.setenv(key_var, f"test-{provider}-key")
    monkeypatch.setenv("OPENROUTER_MODEL", override)

    backend = LLMConfig(provider=provider, model="private-model-v1").resolve_backend()

    assert backend.model == "private-model-v1"


@pytest.mark.parametrize("provider,key_var", [
    ("openrouter", "OPENROUTER_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
])
def test_direct_aggregator_namespaces_known_bare_models(monkeypatch, provider, key_var):
    monkeypatch.setenv(key_var, f"test-{provider}-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    backend = LLMConfig(provider=provider, model="gpt-4o").resolve_backend()

    assert backend.model == "openai/gpt-4o"


def test_primary_key_wins_over_openrouter_for_explicit_gpt5_model(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    backend = LLMConfig(provider="kimi", model="gpt-5.6-luna").resolve_backend()

    assert backend.api_key == "test-kimi-key"
    assert backend.base_url == "https://api.moonshot.cn/v1"
    assert backend.model == "gpt-5.6-luna"
    assert backend.using_openrouter is False


def test_together_primary_key_keeps_namespaced_gpt5_model(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    backend = LLMConfig(provider="together", model="gpt-5.6-luna").resolve_backend()

    assert backend.api_key == "test-together-key"
    assert backend.base_url == "https://api.together.xyz/v1"
    assert backend.model == "openai/gpt-5.6-luna"
    assert backend.using_openrouter is False


def test_explicit_key_and_model_override_take_precedence(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "environment-key")

    client_config, model = LLMConfig(
        provider="kimi",
        api_key="explicit-key",
        model="kimi-k2.6",
    ).get_client_config()

    assert client_config["api_key"] == "explicit-key"
    assert model == "kimi-k2.6"


def test_missing_credentials_raise_an_actionable_error():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMConfig(provider="groq").resolve_backend()

"""Offline compatibility tests for Memobase provider resolution."""

import pytest

from config import resolve_memobase_backend


PROVIDER_ENV_VARS = (
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_kimi_key_keeps_historical_precedence(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")

    backend = resolve_memobase_backend()

    assert backend.api_key == "test-kimi-key"
    assert backend.base_url == "https://api.moonshot.cn/v1"
    assert backend.model == "kimi-k3"
    assert backend.using_openrouter is False


def test_moonshot_key_remains_supported(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")

    backend = resolve_memobase_backend()

    assert backend.api_key == "test-moonshot-key"
    assert backend.base_url == "https://api.moonshot.cn/v1"
    assert backend.model == "kimi-k3"


def test_kimi_falls_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    backend = resolve_memobase_backend()

    assert backend.api_key == "test-openrouter-key"
    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.model == "moonshotai/kimi-k2.6"
    assert backend.using_openrouter is True


def test_openrouter_model_override_stays_fallback_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    fallback = resolve_memobase_backend()
    assert fallback.model == "google/gemma-4-31b-it:free"

    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    direct = resolve_memobase_backend()
    assert direct.model == "kimi-k3"
    assert direct.using_openrouter is False


def test_shared_base_url_overrides_are_honored(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("KIMI_BASE_URL", "https://moonshot.example/v1")
    assert resolve_memobase_backend().base_url == "https://moonshot.example/v1"

    monkeypatch.delenv("KIMI_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    assert resolve_memobase_backend().base_url == "https://router.example/v1"


def test_explicit_key_keeps_the_direct_kimi_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    backend = resolve_memobase_backend("test-explicit-kimi-key")

    assert backend.api_key == "test-explicit-kimi-key"
    assert backend.base_url == "https://api.moonshot.cn/v1"
    assert backend.model == "kimi-k3"
    assert backend.using_openrouter is False


def test_missing_keys_raise_actionable_error():
    with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
        resolve_memobase_backend()

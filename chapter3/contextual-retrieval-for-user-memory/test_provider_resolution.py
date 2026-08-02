"""Offline compatibility contracts for contextual-retrieval provider setup."""

import os
import sys
from types import SimpleNamespace

import agent as agent_module
import contextual_evaluator as evaluator_module
import pytest
import quickstart
from agentbook.providers import Backend
from config import LLMConfig


PROVIDER_ENV_VARS = (
    "SILICONFLOW_API_KEY", "DOUBAO_API_KEY", "ARK_API_KEY", "KIMI_API_KEY",
    "MOONSHOT_API_KEY", "KIMI_BASE_URL", "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL", "OPENROUTER_BASE_URL", "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_kimi_accepts_legacy_key_and_experiment_default(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")

    backend = LLMConfig(provider="kimi").resolve_backend()
    client_config, model = LLMConfig(provider="kimi").get_client_config()

    assert (backend.api_key, backend.base_url, backend.model) == (
        "test-kimi-key", "https://api.moonshot.cn/v1", "kimi-k3"
    )
    assert client_config == {
        "api_key": "test-kimi-key",
        "base_url": "https://api.moonshot.cn/v1",
    }
    assert model == "kimi-k3"


def test_moonshot_key_and_alias_use_kimi_backend(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")

    backend = LLMConfig(provider="moonshot").resolve_backend()

    assert backend.provider == "kimi"
    assert backend.api_key == "test-moonshot-key"
    assert backend.model == "kimi-k3"


def test_ark_key_wins_over_legacy_doubao_key(monkeypatch):
    monkeypatch.setenv("DOUBAO_API_KEY", "legacy-key")
    monkeypatch.setenv("ARK_API_KEY", "ark-key")

    backend = LLMConfig(provider="doubao").resolve_backend()

    assert backend.api_key == "ark-key"
    assert backend.base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_openrouter_model_override_applies_only_to_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

    fallback = LLMConfig(provider="kimi").resolve_backend()
    direct = LLMConfig(provider="kimi", api_key="kimi-key", model="kimi-k2.6").resolve_backend()

    assert fallback.model == "google/gemma-4-31b-it:free"
    assert direct.model == "kimi-k2.6"
    assert direct.using_openrouter is False


def test_agent_constructs_client_from_resolved_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_module, "OpenAI", lambda **kwargs: calls.append(kwargs) or object())
    instance = object.__new__(agent_module.UserMemoryRAGAgent)
    instance.config = SimpleNamespace(llm=LLMConfig(provider="kimi", api_key="test-kimi-key"))

    instance._init_llm_client()

    assert calls == [{"api_key": "test-kimi-key", "base_url": "https://api.moonshot.cn/v1"}]
    assert instance.model == "kimi-k3"


def test_fallback_evaluator_uses_resolved_backend_model(monkeypatch):
    backend = Backend("test-key", "https://provider.example/v1", "provider/model", "provider", False)
    config = SimpleNamespace(llm=SimpleNamespace(resolve_backend=lambda: backend))
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"reward": 1}'))])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response))
    )

    monkeypatch.setattr(evaluator_module.Config, "from_env", lambda: config)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    evaluator = object.__new__(evaluator_module.ContextualMemoryEvaluator)
    test_case = SimpleNamespace(user_question="question", evaluation_criteria="criteria")

    assert evaluator._fallback_llm_evaluation(test_case, "answer")["passed"] is True
    assert calls[0]["model"] == backend.model
    assert calls[0]["temperature"] == 0.1


def test_quickstart_loads_dotenv_before_preflight(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=openai-key\nARK_API_KEY=ark-key\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quickstart, "run_quick_demo", lambda: None)

    try:
        quickstart.main()
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ARK_API_KEY", None)

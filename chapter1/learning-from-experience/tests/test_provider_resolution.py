"""Provider-resolution coverage for the LLM game agent."""

import pytest

import llm_agent
from llm_agent import LLMAgent


class RecordingOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_constructor_uses_moonshot_key_and_custom_base_url(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    agent = LLMAgent(base_url="https://moonshot.test/v1")

    assert agent.client.kwargs == {
        "api_key": "moonshot-key",
        "base_url": "https://moonshot.test/v1",
    }
    assert agent.model == "kimi-k3"
    assert agent.provider == "moonshot"
    assert agent.using_openrouter is False


def test_constructor_accepts_legacy_kimi_key(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "legacy-kimi-key")
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    agent = LLMAgent()

    assert agent.client.kwargs["api_key"] == "legacy-kimi-key"
    assert agent.provider == "moonshot"


def test_constructor_explicit_key_wins_over_environment(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "environment-key")
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    agent = LLMAgent(api_key="explicit-key")

    assert agent.client.kwargs["api_key"] == "explicit-key"


def test_constructor_uses_openrouter_without_leaking_moonshot_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/v1")
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    agent = LLMAgent(base_url="https://moonshot.test/v1")

    assert agent.client.kwargs == {
        "api_key": "openrouter-key",
        "base_url": "https://openrouter.test/v1",
    }
    assert agent.model == "moonshotai/kimi-k2.6"
    assert agent.provider == "openrouter"
    assert agent.using_openrouter is True


def test_constructor_routes_gpt5_through_openrouter(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    agent = LLMAgent(model="gpt-5.6-luna")

    assert agent.client.kwargs["api_key"] == "openrouter-key"
    assert agent.model == "openai/gpt-5.6-luna"
    assert agent.provider == "openrouter"


def test_constructor_requires_a_provider_key(monkeypatch):
    monkeypatch.setattr(llm_agent.openai, "OpenAI", RecordingOpenAI)

    with pytest.raises(ValueError, match="No API key found"):
        LLMAgent()

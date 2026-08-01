"""Offline client-construction contract for user-memory Agentic RAG."""

from types import SimpleNamespace

import agent as agent_module
import pytest
from config import LLMConfig


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in ("KIMI_API_KEY", "KIMI_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_agent_uses_resolved_client_configuration(monkeypatch):
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "OpenAI", construct)
    instance = object.__new__(agent_module.UserMemoryRAGAgent)
    instance.config = SimpleNamespace(
        llm=LLMConfig(provider="kimi", api_key="test-kimi-key")
    )

    instance._init_llm_client()

    assert calls == [{
        "api_key": "test-kimi-key",
        "base_url": "https://api.moonshot.cn/v1",
    }]
    assert instance.backend.api_key == "test-kimi-key"
    assert instance.model == "kimi-k3"

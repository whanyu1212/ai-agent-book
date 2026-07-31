"""Offline tests for constructing Agentic RAG clients from shared backends."""

from types import SimpleNamespace

from agentbook.providers import Backend

import agent as agent_module


class StubLLMConfig:
    def __init__(self, backend):
        self.backend = backend

    def resolve_backend(self):
        return self.backend


def _backend():
    return Backend(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="provider/model",
        provider="test-provider",
        using_openrouter=False,
    )


def test_agent_uses_resolved_backend(monkeypatch):
    backend = _backend()
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "OpenAI", construct)
    instance = object.__new__(agent_module.AgenticRAG)
    instance.config = SimpleNamespace(llm=StubLLMConfig(backend))
    instance._init_llm_client()

    assert calls == [{"api_key": backend.api_key, "base_url": backend.base_url}]
    assert instance.backend is backend
    assert instance.model == backend.model

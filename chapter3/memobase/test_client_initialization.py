"""Offline tests for constructing Memobase clients from resolved backends."""

from agentbook.providers import Backend

import agent as agent_module


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

    monkeypatch.setattr(agent_module, "resolve_memobase_backend", lambda api_key: backend)
    monkeypatch.setattr(agent_module, "OpenAI", construct)
    monkeypatch.setattr(agent_module, "MemoryStore", lambda: object())

    agent = agent_module.MemobaseAgent(api_key="test-explicit-key")

    assert calls == [{"api_key": backend.api_key, "base_url": backend.base_url}]
    assert agent.backend is backend
    assert agent.model == backend.model

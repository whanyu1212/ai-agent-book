"""Offline tests for constructing clients from shared provider backends."""

from types import SimpleNamespace

from agentbook.providers import Backend

import agent as agent_module
import contextual_chunking as chunking_module


class StubLLMConfig:
    provider = "kimi"

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


def _recording_client(calls):
    def construct(**kwargs):
        calls.append(kwargs)
        return object()

    return construct


def test_agent_uses_resolved_backend(monkeypatch):
    backend = _backend()
    calls = []
    monkeypatch.setattr(agent_module, "OpenAI", _recording_client(calls))

    instance = object.__new__(agent_module.AgenticRAG)
    instance.config = SimpleNamespace(llm=StubLLMConfig(backend))
    instance._init_llm_client()

    assert calls == [{"api_key": backend.api_key, "base_url": backend.base_url}]
    assert instance.backend is backend
    assert instance.model == backend.model


def test_contextual_chunker_uses_resolved_backend(monkeypatch):
    backend = _backend()
    calls = []
    monkeypatch.setattr(chunking_module, "OpenAI", _recording_client(calls))

    instance = object.__new__(chunking_module.ContextualChunker)
    instance.llm_config = StubLLMConfig(backend)
    instance._init_llm_client()

    assert calls == [{"api_key": backend.api_key, "base_url": backend.base_url}]
    assert instance.backend is backend
    assert instance.model == backend.model

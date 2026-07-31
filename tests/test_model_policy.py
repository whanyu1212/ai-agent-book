"""Tests for shared reasoning-model request policy."""

import pytest

from agentbook.model_policy import is_reasoning_model, reasoning_safe_temperature


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "openai/gpt-5.6-luna",
        "GPT-5-MINI",
        "kimi-k2.5",
        "moonshotai/kimi-k2.6",
        "kimi-k2.7-preview",
        "kimi-k3",
    ],
)
def test_recognizes_reasoning_models(model):
    assert is_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model",
    [None, "", "gpt-4o", "moonshot-v1-32k", "deepseek-chat", "doubao-seed-1-6"],
)
def test_other_models_are_not_classified_as_reasoning(model):
    assert is_reasoning_model(model) is False


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "kimi-k2.6", "kimi-k3"])
def test_reasoning_models_force_supported_temperature(model):
    assert reasoning_safe_temperature(model, 0.2) == 1


@pytest.mark.parametrize("requested", [0, 0.2, 1.0])
def test_other_models_preserve_requested_temperature(requested):
    assert reasoning_safe_temperature("deepseek-chat", requested) == requested

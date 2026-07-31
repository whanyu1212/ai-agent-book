"""Regression tests for agentic-rag request parameter policy."""

import pytest

from agent import _reasoning_safe_max_tokens


@pytest.mark.parametrize("model", ["kimi-k3", "openai/gpt-5.6-luna"])
def test_existing_reasoning_models_keep_max_token_floor(model):
    assert _reasoning_safe_max_tokens(model, 1024) == 4096


@pytest.mark.parametrize("model", ["kimi-k2.6", "deepseek-chat"])
def test_other_models_keep_requested_max_tokens(model):
    assert _reasoning_safe_max_tokens(model, 1024) == 1024

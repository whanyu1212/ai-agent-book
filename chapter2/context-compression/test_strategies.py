"""Regression tests for context-compression request budgets."""

import pytest

from compression_strategies import _reasoning_safe_max_tokens


@pytest.mark.parametrize("model", ["kimi-k3", "openai/gpt-5.6-luna"])
def test_reasoning_models_receive_additive_token_budget(model):
    assert _reasoning_safe_max_tokens(model, 500, reasoning_budget=2048) == 2548


@pytest.mark.parametrize("model", ["kimi-k2.6", "deepseek-chat"])
def test_other_models_keep_requested_token_budget(model):
    assert _reasoning_safe_max_tokens(model, 500, reasoning_budget=2048) == 500

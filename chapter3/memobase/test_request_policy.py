"""Regression tests for Memobase request parameter policy."""

import pytest

from agent import _reasoning_safe_temperature


@pytest.mark.parametrize("model", ["kimi-k3", "openai/gpt-5.6-luna"])
def test_existing_reasoning_models_keep_temperature_one(model):
    assert _reasoning_safe_temperature(model, 0.7) == 1


@pytest.mark.parametrize("model", ["moonshotai/kimi-k2.6", "deepseek-v4-flash"])
def test_other_models_keep_requested_temperature(model):
    assert _reasoning_safe_temperature(model, 0.7) == 0.7

"""Regression coverage for user-memory's experiment-specific temperature policy."""

import agent
import config
import pytest


@pytest.mark.parametrize(
    "model,requested,expected",
    [
        ("kimi-k3", 0.2, 1),
        ("openai/gpt-5.6-luna", 0.4, 1),
        ("moonshotai/kimi-k2.6", 0.3, 0.3),
        ("gpt-4o-mini", 0.6, 0.6),
    ],
)
def test_temperature_policy_stays_local_to_k3_and_gpt5(model, requested, expected):
    assert config._reasoning_safe_temperature(model, requested) == expected
    assert agent._reasoning_safe_temperature(model, requested) == expected

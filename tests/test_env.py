"""Tests for shared environment readers."""

import pytest

from agentbook.env import read_int_env


def test_read_int_env_returns_default_without_a_warning(monkeypatch):
    monkeypatch.delenv("AGENTBOOK_TEST_INT", raising=False)
    warnings: list[tuple[str, str, int]] = []

    assert (
        read_int_env(
            "AGENTBOOK_TEST_INT",
            42,
            on_invalid=lambda *args: warnings.append(args),
        )
        == 42
    )
    assert warnings == []


@pytest.mark.parametrize(
    "raw, expected",
    [("123", 123), ("-1", -1), (" 456 ", 456)],
)
def test_read_int_env_parses_python_integer_syntax(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENTBOOK_TEST_INT", raw)

    assert read_int_env("AGENTBOOK_TEST_INT", 42, on_invalid=lambda *_: None) == expected


@pytest.mark.parametrize("raw", ["", "not-an-integer"])
def test_read_int_env_warns_once_and_returns_default(monkeypatch, raw):
    monkeypatch.setenv("AGENTBOOK_TEST_INT", raw)
    warnings: list[tuple[str, str, int]] = []

    assert (
        read_int_env("AGENTBOOK_TEST_INT", 42, on_invalid=lambda *args: warnings.append(args)) == 42
    )
    assert warnings == [("AGENTBOOK_TEST_INT", raw, 42)]

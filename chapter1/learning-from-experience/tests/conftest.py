import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch):
    """Keep developer credentials and provider overrides out of unit tests."""
    for variable in (
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "KIMI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_dependency_ledger import (
    LEDGER_PATH,
    discover_requirements,
    requirement_names,
    validate,
)

ENTRY = """
[[entry]]
path = "chapter1/demo/requirements.txt"
root_extra = "ch1"
coverage = "complete"
disposition = "migrate"
consumers = ["chapter1/demo/README.md"]
related_contracts = []
uncovered_packages = []
constraint_notes = []
rationale = "The root extra covers this fixture."
follow_up = "Validate the fixture and remove the redundant file."
"""


class DependencyLedgerTests(unittest.TestCase):
    def make_repo(self, ledger_suffix: str = ENTRY) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "chapter1/demo").mkdir(parents=True)
        (root / "chapter1/vendor").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "chapter1/demo/requirements.txt").write_text(
            "-e ../..\nRequests[socks]>=2; python_version >= '3.11'\n",
            encoding="utf-8",
        )
        (root / "chapter1/demo/README.md").write_text(
            "pip install -r requirements.txt\n", encoding="utf-8"
        )
        (root / "chapter1/vendor/requirements.txt").write_text(
            "vendor-only==1\n", encoding="utf-8"
        )
        (root / "pyproject.toml").write_text(
            """
[project]
dependencies = []

[project.optional-dependencies]
web = ["Requests>=2"]
dev = ["pytest>=8"]
ch1 = ["agentbook[web]"]
""",
            encoding="utf-8",
        )
        (root / LEDGER_PATH).write_text(
            """
version = 1
[scope]
chapters = [1, 2, 3]
excluded_globs = ["chapter1/vendor/**"]
"""
            + ledger_suffix,
            encoding="utf-8",
        )
        return root

    def test_repository_ledger_classifies_all_26_files(self):
        root = Path(__file__).resolve().parents[1]
        errors, count = validate(root)
        self.assertEqual(errors, [])
        self.assertEqual(count, 26)

    def test_requirement_parser_handles_editable_markers_and_extras(self):
        names = requirement_names(
            [
                "-e ../..",
                "Requests[socks]>=2; python_version >= '3.11'",
                "typing_extensions>=4 # compatibility",
            ]
        )
        self.assertEqual(names, {"agentbook", "requests", "typing-extensions"})

    def test_discovery_respects_vendored_exclusions(self):
        root = self.make_repo()
        self.assertEqual(
            discover_requirements(root, [1, 2, 3], ["chapter1/vendor/**"]),
            ["chapter1/demo/requirements.txt"],
        )

    def test_missing_entry_is_reported(self):
        errors, _ = validate(self.make_repo(ledger_suffix=""))
        self.assertIn(
            "unclassified requirements file: chapter1/demo/requirements.txt", errors
        )

    def test_duplicate_entry_is_reported(self):
        errors, _ = validate(self.make_repo(ledger_suffix=ENTRY + ENTRY))
        self.assertIn("duplicate ledger entry: chapter1/demo/requirements.txt", errors)

    def test_invalid_classification_is_reported(self):
        invalid = ENTRY.replace('coverage = "complete"', 'coverage = "unknown"')
        errors, _ = validate(self.make_repo(ledger_suffix=invalid))
        self.assertTrue(any("invalid coverage" in error for error in errors))

    def test_missing_consumer_is_reported(self):
        invalid = ENTRY.replace(
            "chapter1/demo/README.md", "chapter1/demo/MISSING.md"
        )
        errors, _ = validate(self.make_repo(ledger_suffix=invalid))
        self.assertTrue(any("consumers path does not exist" in error for error in errors))

    def test_uncovered_package_drift_is_reported(self):
        root = self.make_repo()
        requirements = root / "chapter1/demo/requirements.txt"
        requirements.write_text(
            requirements.read_text(encoding="utf-8") + "Missing_Pkg>=1\n",
            encoding="utf-8",
        )
        errors, _ = validate(root)
        self.assertTrue(any("uncovered_packages drifted" in error for error in errors))

    def test_divergent_entry_requires_constraint_note(self):
        invalid = ENTRY.replace('coverage = "complete"', 'coverage = "divergent"')
        errors, _ = validate(self.make_repo(ledger_suffix=invalid))
        self.assertTrue(
            any("divergent coverage requires constraint_notes" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()

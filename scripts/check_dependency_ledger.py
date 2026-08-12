"""Validate the staged dependency-migration ledger.

The checker intentionally uses only the Python standard library so it can run
before project dependencies are installed.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import Any

import tomllib

LEDGER_PATH = Path("docs/dependency-migration-ledger.toml")
VALID_COVERAGE = {"complete", "partial", "divergent"}
VALID_DISPOSITIONS = {"migrate", "reconcile", "retain-isolated"}
_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_SELF_EXTRA_RE = re.compile(r"^agentbook\[([^]]+)]")


def normalize_name(name: str) -> str:
    """Return the PEP 503 normalized form used for package comparisons."""

    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    """Extract one distribution name from the subset used by this repository."""

    value = line.strip()
    if not value or value.startswith("#"):
        return None
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if value.startswith(("-e ", "--editable ")):
        # The Chapter 1-3 editable references point back to this repository.
        target = value.split(maxsplit=1)[1]
        return "agentbook" if target in {"../..", "../../"} else None
    if value.startswith("-"):
        return None
    match = _NAME_RE.match(value)
    return normalize_name(match.group(1)) if match else None


def requirement_names(lines: list[str]) -> set[str]:
    return {name for line in lines if (name := requirement_name(line))}


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_requirements(
    repo_root: Path, chapters: list[int], excluded_globs: list[str]
) -> list[str]:
    discovered: list[str] = []
    for chapter in chapters:
        for path in (repo_root / f"chapter{chapter}").rglob("requirements*.txt"):
            relative = path.relative_to(repo_root).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in excluded_globs):
                continue
            discovered.append(relative)
    return sorted(discovered)


def _expand_extra(
    name: str,
    base_dependencies: list[str],
    extras: dict[str, list[str]],
    seen: set[str] | None = None,
) -> set[str]:
    if name not in extras:
        raise KeyError(name)
    visited = set() if seen is None else seen
    if name in visited:
        return set()
    visited.add(name)

    packages = requirement_names(base_dependencies)
    packages.add("agentbook")
    for requirement in extras[name]:
        self_extra = _SELF_EXTRA_RE.match(requirement.strip())
        if self_extra:
            for child in self_extra.group(1).split(","):
                packages.update(
                    _expand_extra(child.strip(), base_dependencies, extras, visited)
                )
            continue
        if package := requirement_name(requirement):
            packages.add(package)
    return packages


def covered_packages(pyproject: dict[str, Any], root_extra: str) -> set[str]:
    project = pyproject["project"]
    base = project.get("dependencies", [])
    extras = project.get("optional-dependencies", {})
    return _expand_extra(root_extra, base, extras) | _expand_extra("dev", base, extras)


def validate(repo_root: Path, ledger_path: Path = LEDGER_PATH) -> tuple[list[str], int]:
    errors: list[str] = []
    ledger = _load_toml(repo_root / ledger_path)
    scope = ledger.get("scope", {})
    chapters = scope.get("chapters", [])
    excluded_globs = scope.get("excluded_globs", [])
    entries = ledger.get("entry", [])

    if chapters != [1, 2, 3]:
        errors.append("scope.chapters must be exactly [1, 2, 3]")
    if not isinstance(excluded_globs, list) or not all(
        isinstance(item, str) for item in excluded_globs
    ):
        errors.append("scope.excluded_globs must be an array of strings")
        excluded_globs = []

    discovered = discover_requirements(repo_root, chapters, excluded_globs)
    paths = [entry.get("path") for entry in entries]
    string_paths = [path for path in paths if isinstance(path, str)]
    if paths != sorted(string_paths):
        errors.append("ledger entries must be sorted by path")
    duplicates = sorted({path for path in string_paths if string_paths.count(path) > 1})
    for path in duplicates:
        errors.append(f"duplicate ledger entry: {path}")
    for path in sorted(set(discovered) - set(string_paths)):
        errors.append(f"unclassified requirements file: {path}")
    for path in sorted(set(string_paths) - set(discovered)):
        errors.append(f"ledger path is not a discovered requirements file: {path}")

    pyproject = _load_toml(repo_root / "pyproject.toml")
    coverage_cache: dict[str, set[str]] = {}
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str):
            errors.append("each ledger entry must have a string path")
            continue
        prefix = f"{path}: "
        chapter_match = re.match(r"chapter([123])\/", path)
        expected_extra = f"ch{chapter_match.group(1)}" if chapter_match else None
        root_extra = entry.get("root_extra")
        if root_extra != expected_extra:
            errors.append(f"{prefix}root_extra must be {expected_extra!r}")

        coverage = entry.get("coverage")
        disposition = entry.get("disposition")
        if coverage not in VALID_COVERAGE:
            errors.append(f"{prefix}invalid coverage {coverage!r}")
        if disposition not in VALID_DISPOSITIONS:
            errors.append(f"{prefix}invalid disposition {disposition!r}")

        for field in ("rationale", "follow_up"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                errors.append(f"{prefix}{field} must be a non-empty string")

        consumers = entry.get("consumers", [])
        related_contracts = entry.get("related_contracts", [])
        for field, references in (
            ("consumers", consumers),
            ("related_contracts", related_contracts),
        ):
            if not isinstance(references, list) or not all(
                isinstance(reference, str) for reference in references
            ):
                errors.append(f"{prefix}{field} must be an array of paths")
                continue
            if references != sorted(set(references)):
                errors.append(f"{prefix}{field} must be sorted and unique")
            for reference in references:
                target = repo_root / reference
                if not target.is_file():
                    errors.append(f"{prefix}{field} path does not exist: {reference}")
                elif field == "consumers" and Path(path).name not in target.read_text(
                    encoding="utf-8", errors="ignore"
                ):
                    errors.append(
                        f"{prefix}consumer does not reference {Path(path).name}: {reference}"
                    )

        declared_uncovered = entry.get("uncovered_packages", [])
        if not isinstance(declared_uncovered, list) or not all(
            isinstance(package, str) for package in declared_uncovered
        ):
            errors.append(f"{prefix}uncovered_packages must be an array of names")
            declared_uncovered = []
        normalized_uncovered = sorted(normalize_name(item) for item in declared_uncovered)
        if declared_uncovered != normalized_uncovered or len(declared_uncovered) != len(
            set(declared_uncovered)
        ):
            errors.append(f"{prefix}uncovered_packages must be normalized, sorted, and unique")

        if root_extra == expected_extra and root_extra is not None:
            try:
                available = coverage_cache.setdefault(
                    root_extra, covered_packages(pyproject, root_extra)
                )
            except KeyError:
                errors.append(f"{prefix}root extra is not declared in pyproject.toml")
                available = set()
            local_names = requirement_names(
                (repo_root / path).read_text(encoding="utf-8").splitlines()
            )
            actual_uncovered = sorted(local_names - available)
            if normalized_uncovered != actual_uncovered:
                errors.append(
                    f"{prefix}uncovered_packages drifted; expected {actual_uncovered!r}"
                )

        conflicts = entry.get("constraint_notes", [])
        if not isinstance(conflicts, list) or not all(
            isinstance(note, str) and note.strip() for note in conflicts
        ):
            errors.append(f"{prefix}constraint_notes must be an array of non-empty strings")
            conflicts = []
        if coverage == "complete" and (normalized_uncovered or conflicts):
            errors.append(f"{prefix}complete coverage cannot have uncovered packages or conflicts")
        elif coverage == "partial" and not normalized_uncovered:
            errors.append(f"{prefix}partial coverage requires uncovered packages")
        elif coverage == "divergent" and not conflicts:
            errors.append(f"{prefix}divergent coverage requires constraint_notes")

    return errors, len(discovered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    args = parser.parse_args(argv)

    errors, count = validate(args.repo_root.resolve(), args.ledger)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Dependency migration ledger is consistent: {count} files classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

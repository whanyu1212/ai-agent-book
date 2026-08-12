# Dependency Migration Ledger

The repository is moving experiment dependencies into the root
`pyproject.toml` without forcing every experiment into one environment. The
machine-readable ledger in
[`dependency-migration-ledger.toml`](dependency-migration-ledger.toml) records
the current Chapter 1–3 contracts and the decision required before each local
`requirements.txt` can change.

This first ledger is an inventory, not a removal list. Existing local install
commands remain supported until a follow-up PR updates every consumer and
validates the affected experiment.

## Classifications

`coverage` compares each local file with its root `chN` extra plus `dev`:

- `complete`: every distribution name is declared by the root environment and
  no known version conflict remains.
- `partial`: one or more distribution names are absent from the root
  environment.
- `divergent`: the local file is intentionally or historically pinned in a way
  that differs from the root contract. It may also contain uncovered packages.

`disposition` defines the next change:

- `migrate`: validate the experiment with root extras, update all consumers,
  and remove the redundant file.
- `reconcile`: resolve missing packages, stale tooling, or incompatible pins
  before deciding whether the file can be removed.
- `retain-isolated`: preserve a clearly documented independent environment for
  platform-specific, vendored, exact-parity, or tightly coupled stacks.

The two fields answer different questions. For example, a file may have
`divergent` coverage and a `retain-isolated` disposition.

## Updating the ledger

When a scoped requirements file or root dependency group changes:

1. Update its ledger entry, including every README, workflow, setup script, or
   runtime helper that directly references the file.
2. Keep entries, consumers, related contracts, and package names sorted.
3. Record packages absent from the expanded `chN + dev` environment in
   `uncovered_packages`; use normalized distribution names such as
   `typing-extensions`.
4. Give divergent entries a concrete `constraint_notes` explanation and every
   entry a decision-ready `follow_up`.
5. Run:

   ```bash
   python scripts/check_dependency_ledger.py
   python -m unittest tests.test_dependency_ledger
   ```

The checker discovers every `requirements*.txt` under Chapters 1–3, applies
the ledger's explicit vendored exclusions, expands self-referencing root
extras from `pyproject.toml`, and fails if inventory or package coverage has
drifted.

## Follow-up PR boundary

Migration PRs should stay chapter-sized or smaller. For each removed file they
must update all listed consumers, preserve a compatibility contract when the
ledger calls for isolation, regenerate `uv.lock` when root dependencies change,
and run experiment-specific offline tests in addition to repository checks.

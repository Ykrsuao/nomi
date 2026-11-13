# Releasing Aster

`aster`, `aster-cli`, `aster-acp`, `aster-memory`, and `aster-bench` are built from
one commit. Versions come from Git tags: `vX.Y.Z` produces `X.Y.Z`; commits between
tags produce development versions.

## Build and check locally

Start in the repository root with Python 3.12–3.13, uv, Rust 1.88+, and the native
compiler toolchain. Source installation uses the local workspace packages.

```powershell
uv sync --group dev --extra tui
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_license_headers.py
uv run pytest
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv build --no-sources --package aster --out-dir dist
uv build --no-sources --package aster-cli --out-dir dist
uv build --no-sources --package aster-acp --out-dir dist
uv build --no-sources --package aster-memory --out-dir dist
uv build --no-sources --package aster-bench --out-dir dist
```

Rebuild the viewer with `npm ci --ignore-scripts` and `npm run build` in
`src/aster/viewer/frontend-react/` when its sources change; include the resulting
`dist/`. Keep `LICENSE`, SPDX headers, and `THIRD_PARTY_NOTICES.md` in distributions.

Smoke-test all five built wheels together in a clean environment. Check imports,
`aster --help`, `aster --version`, and `aster-acp --help` from outside the checkout.
The native terminal is built separately; Python wheels currently do not bundle it.
Retain the commit, version, test results, and artifact checksums with release evidence.

## Configure a release channel

This checkout does not designate an Aster hosting organization or package-index
owner. Before publication, configure the intended repository, package ownership,
and credentials for that release channel.

The optional `scripts/make_release.py` gate requires
`ASTER_RELEASE_REPOSITORY=owner/repo` before it can operate on GitHub. Supply your
actual Aster repository; the script validates the value. It can create a draft
release after its checks, and cannot publish that draft itself.

The gate includes live model checks and fresh-wheel quickstarts. The quickstarts
still run with `--skip-capability`, `--checks-only`, or `--dry-run`. Configure the
selected model aliases and provider credentials before using it. `--ci` requires
an explicit candidate SHA and a compatible model-alias wheel supplied through
`--internal-wheel`; use `--help` for the full gate controls.

The runner retains private reports, logs, traces, distributions, checksums, and a
manifest. Review sanitized notes and the exact tested commit before approving a
draft. A failed gate creates no new draft; a failure after draft creation may need
reconciliation before retrying.

## Publish and rehearse

`.github/workflows/publish.yml` builds all five packages. Manual workflow runs
target TestPyPI; a published GitHub release targets PyPI. Configure Trusted
Publishing for your actual repository and `publish.yml`, with a distinct
`testpypi-<package>` or `pypi-<package>` environment for each distribution. Configure
environment reviewers to match the project's release policy.

Rehearse on TestPyPI and inspect artifact versions and contents before creating
a production release. A `release: published` event enables the production upload
jobs, so use manual dispatch for publication rehearsals.

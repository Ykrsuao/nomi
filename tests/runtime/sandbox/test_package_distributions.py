# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve distribution ownership while avoiding a complete installed-file scan."""

from __future__ import annotations

import csv
import importlib.machinery
import importlib.metadata as metadata
import sys
from pathlib import Path

import pytest

from nooa.runtime.sandbox.dependencies import packages_distributions


def _write_file(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_record(info: Path, records: list[str]) -> None:
    with (info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((record, "", "") for record in records)


def _distribution(
    site: Path,
    name: str,
    records: list[str],
    *,
    missing: set[str] | None = None,
    declared: str | None = None,
) -> Path:
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    _write_file(info, "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    _write_record(info, records)
    if declared is not None:
        _write_file(info, "top_level.txt", declared)
    for record in records:
        if record not in (missing or set()):
            _write_file(site, record)
    return info


@pytest.fixture(autouse=True)
def local_metadata(tmp_path, monkeypatch):
    original_distributions = metadata.distributions
    monkeypatch.syspath_prepend(str(tmp_path))

    def distributions():
        # Keep the real finder and metadata readers, restricted to this test's
        # installation directories in their current sys.path order.
        paths = [
            entry
            for entry in sys.path
            if isinstance(entry, str) and Path(entry).is_relative_to(tmp_path)
        ]
        return original_distributions(path=paths)

    monkeypatch.setattr(metadata, "distributions", distributions)


def _assert_stdlib_projection(roots: set[str]) -> dict[str, list[str]]:
    expected = {
        root: names for root, names in metadata.packages_distributions().items() if root in roots
    }
    actual = packages_distributions(roots)
    assert actual == expected
    return actual


def test_declared_alias_takes_precedence_over_record_inference(tmp_path):
    _distribution(
        tmp_path,
        "different-provider",
        ["suppressed_tools/__init__.py"],
        declared="friendly_tools\nfriendly_tools\n",
    )

    result = _assert_stdlib_projection({"friendly_tools", "suppressed_tools"})

    assert result == {"friendly_tools": ["different-provider", "different-provider"]}


def test_shared_namespace_keeps_every_distribution(tmp_path):
    _distribution(tmp_path, "first-provider", ["shared_tools/first.py"])
    _distribution(tmp_path, "second-provider", ["shared_tools/second.py"])

    result = _assert_stdlib_projection({"shared_tools"})

    assert sorted(result["shared_tools"]) == ["first-provider", "second-provider"]


def test_missing_file_does_not_hide_a_later_existing_file(tmp_path):
    _distribution(
        tmp_path,
        "tool-provider",
        ["toolkit/missing.py", "toolkit/live.py", "missing_tools/__init__.py"],
        missing={"toolkit/missing.py", "missing_tools/__init__.py"},
    )

    result = _assert_stdlib_projection({"toolkit", "missing_tools"})

    assert result == {"toolkit": ["tool-provider"]}


def test_top_level_file_formats_follow_the_running_python_version(tmp_path):
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    _distribution(
        tmp_path,
        "flat-provider",
        [
            "source_tools.py",
            f"native_tools{suffix}",
            "bytecode_tools.pyc",
            "plain_tools",
            "./relative_tools.py",
            "csv_tools/a,b.py",
            "asset.tools/data.txt",
        ],
    )
    roots = {
        "source_tools",
        "native_tools",
        "bytecode_tools",
        "plain_tools",
        "relative_tools",
        "csv_tools",
        "asset.tools",
    }

    result = _assert_stdlib_projection(roots)

    expected_roots = roots - {"plain_tools", "asset.tools"}
    # Python 3.13 additionally considers extensionless top-level records.
    if sys.version_info >= (3, 13):
        expected_roots.add("plain_tools")
    assert result == {root: ["flat-provider"] for root in expected_roots}


@pytest.mark.parametrize(
    ("manifest", "record"),
    [
        ("installed-files.txt", "../legacy_tools/__init__.py"),
        ("SOURCES.txt", "legacy_tools/__init__.py"),
    ],
)
def test_legacy_egg_metadata_retains_its_file_resolution(tmp_path, manifest, record):
    info = tmp_path / "legacy_provider-1.0.egg-info"
    _write_file(info, "PKG-INFO", "Metadata-Version: 1.0\nName: legacy-provider\nVersion: 1.0\n")
    _write_file(info, manifest, record + "\n")
    _write_file(tmp_path, "legacy_tools/__init__.py")

    result = _assert_stdlib_projection({"legacy_tools"})

    assert result == {"legacy_tools": ["legacy-provider"]}


def _virtual_files(distribution: metadata.Distribution) -> list[metadata.PackagePath]:
    path = metadata.PackagePath("virtual_tools/__init__.py")
    path.dist = distribution
    return [path]


class _CustomDistribution(metadata.Distribution):
    def __init__(self, info: Path):
        self.storage = metadata.Distribution.at(info)

    def read_text(self, filename):
        return self.storage.read_text(filename)

    def locate_file(self, path):
        return self.storage.locate_file(path)

    @property
    def files(self):
        return _virtual_files(self)


class _CustomPathDistribution(metadata.PathDistribution):
    @property
    def files(self):
        return _virtual_files(self)


class _Finder:
    def __init__(self, distribution: metadata.Distribution):
        self.distribution = distribution

    def find_spec(self, fullname, path=None, target=None):
        return None

    def find_distributions(self, context):
        return iter((self.distribution,))


@pytest.mark.parametrize("provider", [_CustomDistribution, _CustomPathDistribution])
def test_custom_files_override_is_authoritative(tmp_path, monkeypatch, provider):
    # The provider's store is outside the top-level metadata search directory.
    # Only its finder publishes this distribution. Its RECORD deliberately
    # disagrees with files, and the virtual files need not exist on this disk.
    site = tmp_path / "provider-store"
    info = _distribution(site, "virtual-provider", ["decoy_tools/__init__.py"])
    finder = _Finder(provider(info))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    result = _assert_stdlib_projection({"virtual_tools", "decoy_tools"})

    assert result == {"virtual_tools": ["virtual-provider"]}


@pytest.mark.parametrize(
    ("bad_row", "error"),
    [
        (("unrelated_tools/data.py", "", "", "extra-column"), TypeError),
        (("unrelated_tools/data.py", "", "invalid-size"), ValueError),
    ],
)
def test_malformed_record_preserves_the_stdlib_exception(tmp_path, bad_row, error):
    info = _distribution(tmp_path, "tool-provider", ["toolkit/__init__.py"])
    with (info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
        # A successful target match must not hide a later malformed row, even
        # when that row belongs to a module the caller did not request.
        csv.writer(stream).writerows([("toolkit/__init__.py", "", ""), bad_row])

    with pytest.raises(error) as reference:
        metadata.packages_distributions()
    with pytest.raises(error) as actual:
        packages_distributions({"toolkit"})

    assert str(actual.value) == str(reference.value)


def test_metadata_and_recorded_file_changes_are_visible_without_a_cache(tmp_path):
    info = _distribution(tmp_path, "tool-provider", ["old_tools/__init__.py"])
    roots = {"old_tools", "new_tools", "declared_tools"}
    assert _assert_stdlib_projection(roots) == {"old_tools": ["tool-provider"]}

    _write_file(tmp_path, "new_tools/__init__.py")
    _write_record(info, ["new_tools/__init__.py"])
    assert _assert_stdlib_projection(roots) == {"new_tools": ["tool-provider"]}

    (tmp_path / "new_tools/__init__.py").unlink()
    assert _assert_stdlib_projection(roots) == {}

    _write_file(info, "top_level.txt", "declared_tools\n")
    assert _assert_stdlib_projection(roots) == {"declared_tools": ["tool-provider"]}

    _write_file(info, "top_level.txt", "new_tools\n")
    assert _assert_stdlib_projection(roots) == {"new_tools": ["tool-provider"]}


def test_sys_path_precedence_and_removal_change_the_visible_providers(tmp_path, monkeypatch):
    first = tmp_path / "first-site"
    second = tmp_path / "second-site"
    _distribution(first, "first-provider", ["shared_tools/first.py"])
    _distribution(second, "second-provider", ["shared_tools/second.py"])
    original_path = sys.path[:]
    roots = {"shared_tools"}

    monkeypatch.setattr(sys, "path", [str(first), *original_path])
    assert _assert_stdlib_projection(roots) == {"shared_tools": ["first-provider"]}

    monkeypatch.setattr(sys, "path", [str(second), str(first), *original_path])
    assert _assert_stdlib_projection(roots) == {
        "shared_tools": ["second-provider", "first-provider"]
    }

    monkeypatch.setattr(sys, "path", [str(second), *original_path])
    assert _assert_stdlib_projection(roots) == {"shared_tools": ["second-provider"]}

    monkeypatch.setattr(sys, "path", original_path)
    assert _assert_stdlib_projection(roots) == {}

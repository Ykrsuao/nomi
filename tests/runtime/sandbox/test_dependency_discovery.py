# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover guest dependencies without checking unrelated distribution files."""

from __future__ import annotations

import csv
import importlib.metadata as metadata
from pathlib import Path
from types import ModuleType

import pytest

from aster.agentdoc import visibility
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.microsandbox_executor import SandboxedExecutor


def _distribution(root: Path, name: str, records: list[str]) -> Path:
    info = root / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n", encoding="utf-8")
    with (info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((record, "", "") for record in records)
    return info


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    original_distributions = metadata.distributions
    original_version = metadata.version
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        metadata, "distributions", lambda: original_distributions(path=[str(tmp_path)])
    )
    monkeypatch.setattr(metadata, "requires", lambda name: [])
    versions: list[str] = []

    def version(name):
        versions.append(name)
        if name == "cloudpickle":
            return "3.1.2"
        if name == "numpy":
            raise metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(metadata, "version", version)
    exposed: dict[str, object] = {}
    monkeypatch.setattr(visibility, "filter_mro_module_globals", lambda cls: exposed)
    executor = SandboxedExecutor(object(), SandboxConfig(), cell_timeout=10)
    return executor, exposed, versions


def test_dependency_discovery_skips_unrelated_distribution_files(discovery, tmp_path, monkeypatch):
    executor, exposed, _ = discovery
    exposed["toolkit"] = ModuleType("toolkit")
    _distribution(tmp_path, "tool-provider", ["toolkit/__init__.py", "other/data.txt"])
    _distribution(tmp_path, "unused-provider", [f"unrelated/module_{n}.py" for n in range(50)])
    (tmp_path / "toolkit").mkdir()
    (tmp_path / "toolkit" / "__init__.py").touch()
    original_exists = Path.exists
    unrelated_checks: list[Path] = []

    def exists(path, *args, **kwargs):
        if path.is_relative_to(tmp_path / "unrelated") or path == tmp_path / "other/data.txt":
            unrelated_checks.append(path)
        return original_exists(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)

    assert executor._dependencies() == ["cloudpickle==3.1.2", "tool-provider==1.0"]
    assert unrelated_checks == []


def test_dependency_discovery_keeps_all_shared_namespace_providers(discovery, tmp_path):
    executor, exposed, _ = discovery
    exposed["tools"] = ModuleType("shared.tools")
    _distribution(tmp_path, "first-provider", ["shared/first.py"])
    _distribution(tmp_path, "second-provider", ["shared/second.py"])
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/first.py").touch()
    (tmp_path / "shared/second.py").touch()

    assert executor._dependencies() == [
        "cloudpickle==3.1.2",
        "first-provider==1.0",
        "second-provider==1.0",
    ]


def test_dependency_discovery_reads_each_exposed_distribution_version_once(discovery, tmp_path):
    executor, exposed, versions = discovery
    exposed.update({"top": ModuleType("toolkit"), "child": ModuleType("toolkit.child")})
    _distribution(tmp_path, "tool-provider", ["toolkit/__init__.py"])
    (tmp_path / "toolkit").mkdir()
    (tmp_path / "toolkit/__init__.py").touch()

    assert executor._dependencies() == ["cloudpickle==3.1.2", "tool-provider==1.0"]
    assert versions.count("tool-provider") == 1


def test_dependency_discovery_observes_version_and_installed_provider_changes(discovery, tmp_path):
    executor, exposed, _ = discovery
    exposed["tools"] = ModuleType("shared.tools")
    info = _distribution(tmp_path, "first-provider", ["shared/first.py"])
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/first.py").touch()
    assert executor._dependencies() == ["cloudpickle==3.1.2", "first-provider==1.0"]

    (info / "METADATA").write_text("Name: first-provider\nVersion: 2.0\n", encoding="utf-8")
    _distribution(tmp_path, "second-provider", ["shared/second.py"])
    (tmp_path / "shared/second.py").touch()

    assert executor._dependencies() == [
        "cloudpickle==3.1.2",
        "first-provider==2.0",
        "second-provider==1.0",
    ]

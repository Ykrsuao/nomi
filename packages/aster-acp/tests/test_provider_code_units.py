# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code caching keeps Python import behavior and recovers from stale or broken caches."""

import importlib
import importlib.machinery
import importlib.resources
import inspect
import pickle
import sys
from pathlib import Path

import pytest
from aster_acp import _provider_code as cache

_discover_installed_sources = cache._installed_sources


@pytest.fixture
def provider_package(tmp_path, monkeypatch):
    root = tmp_path / "provider_cache_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("from .types import Item\n", encoding="utf-8")
    (root / "types.py").write_text("class Item:\n    value = 'original'\n", encoding="utf-8")
    (root / "resource.txt").write_text("physical resource", encoding="utf-8")
    identity = {"provider_revision": 1, "root": str(root)}
    monkeypatch.setattr(cache, "_PACKAGES", (root.name,))
    monkeypatch.setattr(cache, "_installed_sources", lambda: ({root.name: root}, identity))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("ASTER_PROVIDER_CODE_CACHE", raising=False)
    yield root, identity, tmp_path / "code-cache"
    for name in list(sys.modules):
        if name == root.name or name.startswith(root.name + "."):
            sys.modules.pop(name, None)


def test_cached_modules_preserve_paths_resources_source_and_pickling(provider_package):
    root, _, directory = provider_package
    assert cache.activate_provider_code_cache(cache_dir=directory)
    # Optional modules installed outside the archive still use the physical path.
    (root / "optional.py").write_text("value = 'physical module'\n", encoding="utf-8")
    package = importlib.import_module(root.name)
    optional = importlib.import_module(root.name + ".optional")
    assert package.__file__ is not None
    assert package.__spec__ is not None
    assert package.__spec__.origin is not None
    assert Path(package.__file__) == root / "__init__.py"
    assert Path(package.__spec__.origin) == root / "__init__.py"
    assert package.__path__[0] == str(root)
    assert optional.value == "physical module"
    assert pickle.loads(pickle.dumps(package.Item())).value == "original"
    assert "class Item:" in inspect.getsource(package.Item)
    # Resource contents are live on disk, independently of the cached code.
    (root / "resource.txt").write_text("updated resource", encoding="utf-8")
    resource = importlib.resources.files(package).joinpath("resource.txt")
    assert resource.read_text(encoding="utf-8") == "updated resource"


def test_namespace_submodules_can_use_cached_code(provider_package):
    root, _, directory = provider_package
    namespace = root / "namespace"
    namespace.mkdir()
    (namespace / "child.py").write_text("value = 42\n", encoding="utf-8")
    assert cache.activate_provider_code_cache(cache_dir=directory)
    child = importlib.import_module(root.name + ".namespace.child")
    assert child.value == 42
    assert isinstance(child.__loader__, cache._CodeLoader)


def test_repeated_launch_reuses_valid_cache(provider_package, monkeypatch):
    _, _, directory = provider_package
    assert cache.activate_provider_code_cache(cache_dir=directory)

    def unexpected_rebuild(*args):
        raise AssertionError("A valid code cache must not be rebuilt")

    monkeypatch.setattr(cache, "_build_archive", unexpected_rebuild)
    assert cache.activate_provider_code_cache(cache_dir=directory)


@pytest.mark.parametrize("corruption", ["archive", "metadata", "missing_metadata"])
def test_next_launch_rebuilds_a_broken_cache(provider_package, corruption):
    root, _, directory = provider_package
    assert cache.activate_provider_code_cache(cache_dir=directory)
    archive = next(directory.glob("*.zip"))
    metadata = archive.with_suffix(".json")
    if corruption == "archive":
        archive.write_bytes(b"broken code archive")
    elif corruption == "metadata":
        metadata.write_text("not valid JSON", encoding="utf-8")
    else:
        metadata.unlink()
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, cache._CodeFinder)
    ]
    assert cache.activate_provider_code_cache(cache_dir=directory)
    assert importlib.import_module(root.name).Item().value == "original"


def test_changed_sdk_identity_loads_new_code(provider_package):
    root, identity, directory = provider_package
    assert cache.activate_provider_code_cache(cache_dir=directory)
    (root / "types.py").write_text("class Item:\n    value = 'new version'\n", encoding="utf-8")
    identity["provider_revision"] = 2
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, cache._CodeFinder)
    ]
    assert cache.activate_provider_code_cache(cache_dir=directory)
    assert importlib.import_module(root.name).Item().value == "new version"


@pytest.mark.parametrize("setting", ["0", "false", "off"])
def test_explicit_opt_out_uses_ordinary_imports(provider_package, monkeypatch, setting):
    root, _, directory = provider_package
    monkeypatch.setenv("ASTER_PROVIDER_CODE_CACHE", setting)
    monkeypatch.syspath_prepend(str(root.parent))
    assert not cache.activate_provider_code_cache(cache_dir=directory)
    assert isinstance(
        importlib.import_module(root.name).__loader__, importlib.machinery.SourceFileLoader
    )
    assert not directory.exists()


def test_unavailable_cache_uses_ordinary_imports(provider_package, monkeypatch):
    root, _, directory = provider_package
    directory.write_text("This path is a file", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root.parent))
    assert not cache.activate_provider_code_cache(cache_dir=directory)
    assert importlib.import_module(root.name).Item().value == "original"


def test_already_loaded_sdk_is_preserved(provider_package, monkeypatch):
    root, _, directory = provider_package
    monkeypatch.syspath_prepend(str(root.parent))
    package = importlib.import_module(root.name)
    assert not cache.activate_provider_code_cache(cache_dir=directory)
    assert importlib.import_module(root.name) is package
    assert not directory.exists()


def test_editable_sdk_does_not_use_a_code_cache(provider_package, monkeypatch):
    _, _, directory = provider_package

    class EditableDistribution:
        @staticmethod
        def read_text(name):
            assert name == "direct_url.json"
            return '{"dir_info": {"editable": true}}'

    monkeypatch.setattr(cache, "_installed_sources", _discover_installed_sources)
    monkeypatch.setattr(
        cache.importlib.metadata, "distribution", lambda name: EditableDistribution()
    )
    assert not cache.activate_provider_code_cache(cache_dir=directory)
    assert not directory.exists()


def test_project_package_keeps_priority_over_cached_sdk(provider_package, monkeypatch, tmp_path):
    root, _, directory = provider_package
    assert cache.activate_provider_code_cache(cache_dir=directory)
    shadow = tmp_path / "project" / root.name
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("value = 'project package'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(shadow.parent))
    package = importlib.import_module(root.name)
    assert package.value == "project package"
    assert package.__file__ is not None
    assert Path(package.__file__) == shadow / "__init__.py"


def test_unusable_optional_code_does_not_block_ordinary_imports(provider_package):
    root, _, directory = provider_package
    (root / "unused.py").write_text("This is not valid Python: !\n", encoding="utf-8")
    assert not cache.activate_provider_code_cache(cache_dir=directory)
    assert importlib.import_module(root.name).Item().value == "original"

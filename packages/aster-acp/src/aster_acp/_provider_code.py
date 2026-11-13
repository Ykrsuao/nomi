# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read Windows provider SDK code in bulk while retaining installed resource paths.

Cold imports can synchronously wait for a file check for every bytecode file.
Only immutable, installed SDK wheels are cached; project and editable sources
keep their ordinary loaders. Resources and native extensions stay on disk.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.metadata
import importlib.resources.readers
import importlib.util
import json
import logging
import marshal
import os
import struct
import sys
import tempfile
import zipfile
import zipimport
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

logger = logging.getLogger(__name__)
_PACKAGES = ("litellm", "openai")
_CACHE_FORMAT = 1


class _CodeLoader(importlib.abc.FileLoader):
    """Use cached bytecode, with the installed package's files and search path."""

    def __init__(
        self, importer: zipimport.zipimporter, path: Path, *, fullname: str, package: bool
    ) -> None:
        super().__init__(fullname, str(path))
        self.importer = importer
        self.package = package

    def create_module(self, spec: ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        if self.package:
            # ZIP imports cannot load .pyd/.so files. The physical path also
            # preserves ordinary discovery of optional installed submodules.
            module.__path__.insert(0, str(Path(self.path).parent))
        exec(self.get_code(module.__name__), module.__dict__)

    def get_code(self, fullname: str) -> CodeType:
        code = self.importer.get_code(fullname)
        if code is None:
            raise ImportError(f"No cached provider code for {fullname}")
        return code

    def get_source(self, fullname: str) -> str | None:
        return self.importer.get_source(fullname)

    def get_filename(self, fullname: str | None = None) -> str:
        return self.path

    def get_data(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def is_package(self, fullname: str) -> bool:
        return self.package

    def get_resource_reader(self, name: str | None = None) -> Any:
        return importlib.resources.readers.FileReader(self) if self.package else None


class _CodeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, archive: Path, roots: dict[str, Path]) -> None:
        self.archive = archive
        self.roots = roots
        self.importers: dict[str, zipimport.zipimporter] = {}

    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        root_name = fullname.partition(".")[0]
        if root_name not in self.roots:
            return None
        parent = fullname.rpartition(".")[0]
        if parent:
            expected = self.roots[root_name].joinpath(*parent.split(".")[1:])
            first = next(iter(path), None) if path is not None else None
            if first is None or not _same_path(first, expected):
                return None
        else:
            ordinary = PathFinder.find_spec(fullname)
            expected = self.roots[root_name] / "__init__.py"
            if (
                ordinary is None
                or ordinary.origin is None
                or not _same_path(ordinary.origin, expected)
            ):
                return None
        if parent not in self.importers:
            suffix = "/" + parent.replace(".", "/") if parent else ""
            try:
                self.importers[parent] = zipimport.zipimporter(str(self.archive) + suffix)
            except zipimport.ZipImportError:
                return None
        importer = self.importers[parent]
        spec = importer.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        package = spec.submodule_search_locations is not None
        physical = self.roots[root_name].joinpath(*fullname.split(".")[1:])
        physical = physical / "__init__.py" if package else physical.with_suffix(".py")
        spec.loader = _CodeLoader(importer, physical, fullname=fullname, package=package)
        spec.origin = str(physical)
        spec.has_location = True
        return spec

    def invalidate_caches(self) -> None:
        for importer in self.importers.values():
            importer.invalidate_caches()


def _same_path(first: str | Path, second: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _installed_sources() -> tuple[dict[str, Path], dict[str, Any]]:
    roots = {}
    packages = {}
    for name in _PACKAGES:
        distribution = importlib.metadata.distribution(name)
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("dir_info", {}).get("editable"):
            raise ValueError("Editable provider SDKs use their ordinary source loaders")
        root = Path(str(distribution.locate_file(name))).resolve()
        record = distribution.read_text("RECORD")
        if not root.is_dir() or record is None:
            raise ValueError("Provider code caching requires installed SDK wheels")
        roots[name] = root
        packages[name] = {
            "root": str(root),
            "version": distribution.version,
            "record": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        }
    identity = {
        "format": _CACHE_FORMAT,
        "python": list(sys.version_info[:3]),
        "magic": importlib.util.MAGIC_NUMBER.hex(),
        "packages": packages,
    }
    return roots, identity


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _valid_cache(archive: Path, metadata: Path, identity: dict[str, Any]) -> bool:
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record["identity"] != identity or record["digest"] != _digest(archive):
            return False
        zipimport.zipimporter(str(archive))
        return True
    except (OSError, ValueError, KeyError, TypeError, zipimport.ZipImportError):
        return False


def _build_archive(archive: Path, roots: dict[str, Path]) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, root in sorted(roots.items()):
            for source in sorted(root.rglob("*.py")):
                if "__pycache__" in source.parts:
                    continue
                relative = name + "/" + source.relative_to(root).as_posix()
                data = source.read_bytes()
                info = zipfile.ZipInfo.from_file(source, relative)
                bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
                # ZIP timestamps have two-second resolution. Match that value
                # so zipimport can use the bytecode without compiling source.
                mtime = int(source.stat().st_mtime) // 2 * 2
                code = compile(data, str(source), "exec", dont_inherit=True)
                bytecode = (
                    importlib.util.MAGIC_NUMBER
                    + struct.pack("<III", 0, mtime, len(data))
                    + marshal.dumps(code)
                )
                bytecode_info = zipfile.ZipInfo(relative + "c", date_time=info.date_time)
                bundle.writestr(
                    bytecode_info,
                    bytecode,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=1,
                )


def _build_cache(
    archive: Path, metadata: Path, roots: dict[str, Path], identity: dict[str, Any]
) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".zip", delete=False) as temporary:
        temporary_archive = Path(temporary.name)
    temporary_metadata = temporary_archive.with_suffix(".json")
    try:
        _build_archive(temporary_archive, roots)
        temporary_metadata.write_text(
            json.dumps({"identity": identity, "digest": _digest(temporary_archive)}),
            encoding="utf-8",
        )
        os.replace(temporary_archive, archive)
        os.replace(temporary_metadata, metadata)
        # An interrupted or corrupt cache may have populated zipimport's index.
        zipimport.zipimporter(str(archive)).invalidate_caches()
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def activate_provider_code_cache(*, cache_dir: Path | None = None) -> bool:
    """Enable a reusable SDK code cache before the first Windows model is built.

    The first launch after SDK installation builds the cache. Later launches
    verify and reuse it; SDK wheel or Python changes select a new cache. Any
    preparation failure falls back to ordinary imports.
    """
    if sys.platform != "win32" or os.getenv("ASTER_PROVIDER_CODE_CACHE", "").lower() in {
        "0",
        "false",
        "off",
    }:
        return False
    if any(name in sys.modules for name in _PACKAGES):
        return False
    try:
        roots, identity = _installed_sources()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        if cache_dir is None:
            base = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
            cache_dir = base / "aster" / "provider-code"
        archive = cache_dir / (key + ".zip")
        metadata = cache_dir / (key + ".json")
        if not _valid_cache(archive, metadata, identity):
            _build_cache(archive, metadata, roots, identity)
        if not any(
            isinstance(finder, _CodeFinder) and finder.archive == archive
            for finder in sys.meta_path
        ):
            sys.meta_path.insert(0, _CodeFinder(archive, roots))
        return True
    except (
        OSError,
        ValueError,
        ImportError,
        RuntimeError,
        SyntaxError,
        zipfile.BadZipFile,
    ) as error:
        logger.debug("Using ordinary provider imports because code caching failed: %s", error)
        return False

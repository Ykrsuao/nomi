# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise environment preparation against the SDK's persistent naming contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nooa.runtime.sandbox import microsandbox_executor as backend
from nooa.runtime.sandbox.config import SandboxConfig

IMAGE = f"python:{platform.python_version()}-slim"
DEPENDENCIES = ["cloudpickle==3.1.2"]
ENVIRONMENT = (
    "nooa-runtime-"
    + hashlib.sha256((IMAGE + "\nuv-0.11.2\n" + "\n".join(DEPENDENCIES)).encode()).hexdigest()[:24]
)


class _Snapshot:
    def __init__(self, path: Path):
        self.path = path
        record = json.loads(path.read_text())
        self.id = record["id"]
        self.name = record["name"]
        self.group = record["group"]
        self.image_ref = record["image_ref"]
        self.created_at = record["created_at"]

    async def open(self):
        record = json.loads(self.path.read_text())
        if record.get("error") == "permission":
            raise PermissionError("snapshot metadata is inaccessible")
        if record.get("error") == "corrupt":
            raise ValueError("snapshot metadata is corrupt")
        return self


class _VM:
    def __init__(self, name: str, store: _Store):
        self._name = name
        self.store = store

    @property
    async def name(self):
        return self._name

    async def exec(self, *args, **kwargs):
        return SimpleNamespace(stdout_text=platform.python_version(), stderr_text="", exit_code=0)

    async def destroy(self, *, force):
        self.store.destroyed.append(self._name)


class _Store:
    """A disk-backed SDK double: get accepts IDs, open resolves group:member."""

    def __init__(self, root: Path):
        self.root = root
        self.records = root / "records"
        self.preparations = root / "preparations"
        self.records.mkdir(parents=True, exist_ok=True)
        self.preparations.mkdir(exist_ok=True)
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.destroyed: list[str] = []
        self.fail_publish = False
        self.sdk = SimpleNamespace(
            Snapshot=self,
            Sandbox=SimpleNamespace(create=self.create_vm),
            Network=SimpleNamespace(allow_all=lambda: None),
        )

    def seed(self, name=ENVIRONMENT, *, group="nooa-prepare-legacy", image=IMAGE, error=None):
        snapshot_id = "snap_" + uuid.uuid4().hex
        path = self.records / f"{snapshot_id}.json"
        record = {
            "id": snapshot_id,
            "name": name,
            "group": group,
            "image_ref": image,
            "created_at": time.time_ns(),
            "error": error,
        }
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(record))
        pending.replace(path)
        return _Snapshot(path)

    async def list(self):
        return [_Snapshot(path) for path in self.records.glob("*.json")]

    async def get(self, identifier):
        for snapshot in await self.list():
            if snapshot.id == identifier:
                return snapshot
        raise FileNotFoundError(identifier)

    async def open(self, selector):
        for snapshot in await self.list():
            if f"{snapshot.group}:{snapshot.name}" == selector:
                return await snapshot.open()
        raise FileNotFoundError(selector)

    async def create(self, name, *, from_sandbox, group=None, full=False):
        if self.fail_publish:
            raise OSError("snapshot publication failed")
        if any(s.group == group and s.name == name for s in await self.list()):
            raise FileExistsError(name)
        return self.seed(name, group=group or from_sandbox)

    async def create_vm(self, name, **kwargs):
        (self.preparations / name).touch()
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        await asyncio.sleep(0.05)
        return _VM(name, self)

    @property
    def preparation_count(self):
        return len(list(self.preparations.iterdir()))


def _executor(workspace: Path):
    executor = backend.SandboxedExecutor(
        object(), SandboxConfig(workspace=str(workspace)), cell_timeout=10
    )
    executor._dependencies = lambda: DEPENDENCIES
    return executor


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "msb"
    monkeypatch.setenv("MSB_HOME", str(root))
    # Isolate the old implementation's process cache while demonstrating that
    # it cannot substitute for the persistent SDK lookup.
    if hasattr(backend, "_SNAPSHOTS"):
        monkeypatch.setattr(backend, "_SNAPSHOTS", {})
    return _Store(root)


async def test_reuses_legacy_snapshot_without_preparing_another_vm(store, tmp_path):
    store.seed()
    newest = store.seed(group="nooa-prepare-newest")
    # A same-named member in a user's group must not be adopted by NOOA.
    store.seed(group="user-snapshot")
    store.seed(group="nooa-prepare-wrong-image", image="python:3.12-slim")

    result = await _executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE)

    assert result.id == newest.id
    assert store.preparation_count == 0


async def test_concurrent_executors_prepare_only_once(store, tmp_path):
    first, second = await asyncio.gather(
        _executor(tmp_path / "workspace-a")._prepare(store.sdk, IMAGE),
        _executor(tmp_path / "workspace-b")._prepare(store.sdk, IMAGE),
    )

    assert first.id == second.id
    assert store.preparation_count == 1
    assert len(store.destroyed) == 1


@pytest.mark.parametrize("group", ["nooa-runtime", "nooa-prepare-legacy"])
@pytest.mark.parametrize("error,kind", [("permission", PermissionError), ("corrupt", ValueError)])
async def test_lookup_errors_do_not_trigger_reinstallation(store, tmp_path, group, error, kind):
    store.seed(group=group, error=error)

    with pytest.raises(kind):
        await _executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE)

    assert store.preparation_count == 0


async def test_changing_storage_home_does_not_reuse_another_stores_snapshot(
    store, tmp_path, monkeypatch
):
    first = await _executor(tmp_path / "workspace-a")._prepare(store.sdk, IMAGE)
    other = _Store(tmp_path / "other-msb")
    monkeypatch.setenv("MSB_HOME", str(other.root))

    second = await _executor(tmp_path / "workspace-b")._prepare(other.sdk, IMAGE)

    assert second.id != first.id
    assert other.preparation_count == 1


async def test_deleted_snapshot_is_prepared_again(store, tmp_path):
    first = await _executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE)
    first.path.unlink()

    second = await _executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE)

    assert second.id != first.id
    assert store.preparation_count == 2


async def test_cancelled_waiter_does_not_leak_lock_or_prepare(store, tmp_path):
    store.release = asyncio.Event()
    owner = asyncio.create_task(_executor(tmp_path / "a")._prepare(store.sdk, IMAGE))
    await asyncio.wait_for(store.started.wait(), 2)
    waiter = asyncio.create_task(_executor(tmp_path / "b")._prepare(store.sdk, IMAGE))
    try:
        await asyncio.sleep(0.1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        store.release.set()
        first = await asyncio.wait_for(owner, 2)

    next_result = await asyncio.wait_for(_executor(tmp_path / "c")._prepare(store.sdk, IMAGE), 2)
    assert next_result.id == first.id
    assert store.preparation_count == 1


async def test_failed_publication_releases_lock_and_destroys_vm(store, tmp_path):
    store.fail_publish = True
    with pytest.raises(OSError, match="snapshot publication failed"):
        await _executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE)
    assert len(store.destroyed) == 1

    store.fail_publish = False
    await asyncio.wait_for(_executor(tmp_path / "workspace")._prepare(store.sdk, IMAGE), 2)
    assert store.preparation_count == 2
    assert len(store.destroyed) == 2


async def _start_worker(root: Path, workspace: Path, *, barrier=False):
    env = dict(os.environ, MSB_HOME=str(root))
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(root),
        str(workspace),
        str(int(barrier)),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _worker_result(process):
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 0, stderr.decode(errors="replace")
        return json.loads(stdout)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_new_process_and_workspace_reuse_published_snapshot(store, tmp_path):
    first = await _worker_result(await _start_worker(store.root, tmp_path / "a"))
    second = await _worker_result(await _start_worker(store.root, tmp_path / "b"))

    assert first["id"] == second["id"]
    assert store.preparation_count == 1
    assert len(await store.list()) == 1


async def test_separate_processes_prepare_same_environment_only_once(store, tmp_path):
    ready = store.root / "ready"
    ready.mkdir()
    processes = []
    try:
        for name in ("a", "b"):
            processes.append(await _start_worker(store.root, tmp_path / name, barrier=True))
        async with asyncio.timeout(30):
            while len(list(ready.iterdir())) < 2:
                assert all(p.returncode is None for p in processes)
                await asyncio.sleep(0.02)
        (store.root / "go").touch()
        first, second = await asyncio.gather(*(_worker_result(p) for p in processes))
        assert first["id"] == second["id"]
        assert store.preparation_count == 1
        assert len(await store.list()) == 1
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()


async def _worker():
    root, workspace = Path(sys.argv[2]), Path(sys.argv[3])
    if int(sys.argv[4]):
        (root / "ready" / workspace.name).touch()
        async with asyncio.timeout(30):
            while not (root / "go").exists():
                await asyncio.sleep(0.02)
    snapshot = await _executor(workspace)._prepare(_Store(root).sdk, IMAGE)
    print(json.dumps({"id": snapshot.id}))


if __name__ == "__main__":
    asyncio.run(_worker())

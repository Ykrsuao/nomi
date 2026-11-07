# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct execution in microsandbox; no host worker or unguarded fallback."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import inspect
import logging
import platform
import sys
import sysconfig
import time
import uuid
from pathlib import Path
from typing import Any, cast

from nooa.errors.formatting import _hard_bound_text
from nooa.events import ExecutionResult
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.bootstrap import describe_call, dumps_namespace
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.errors import (
    CellMemoryError,
    CellSerializationError,
    CellTimeoutError,
    SandboxUnavailable,
    WorkerDiedError,
)
from nooa.runtime.sandbox.paths import PathMap
from nooa.runtime.sandbox.protocol import FrameReader, frame
from nooa.runtime.sandbox.serialization import (
    dto_from_wire,
    dto_to_result,
    effective_error_limit,
    is_picklable,
)

logger = logging.getLogger(__name__)
_SNAPSHOTS: dict[str, Any] = {}


def _bounded_text(value: Any, fallback: Any, *, limit: Any) -> str:
    try:
        text = str(value)
    except BaseException:
        text = fallback
    return _hard_bound_text(text, limit)


def check_enforceable(config: Any, caps: Any = None) -> list[str]:
    return (
        []
        if importlib.util.find_spec("microsandbox")
        else ["microsandbox SDK (install nooa[sandbox])"]
    )


class SandboxedExecutor:
    """A persistent guest interpreter, owning its entire VM lifecycle."""

    def __init__(
        self,
        agent: Any,
        config: SandboxConfig,
        *,
        cell_timeout: float | None,
        framework_builtins: dict[str, Any] | None = None,
        restrictions: Any = None,
        max_error: int | None = None,
        error_tail: int | None = None,
        truncation_config: Any = None,
    ) -> None:
        self._agent = agent
        self._config = config
        if config.workspace:
            Path(config.workspace).mkdir(parents=True, exist_ok=True)
        self._cell_timeout = cell_timeout
        self._framework_builtins = framework_builtins or {}
        self._restrictions = restrictions
        self._max_error = effective_error_limit(max_error)
        self._error_tail = error_tail
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        self._truncation = truncation_config or DEFAULT_TRUNCATION_CONFIG
        self._codec = wire.Codec(wire.agent_types(agent))
        self._lock = asyncio.Lock()
        self._closed = self._disabled = False
        self._vm: Any = None
        self._stream: Any = None
        self._sink: Any = None
        self._pump: asyncio.Task[None] | None = None
        self._messages: Any = None
        self._req_id = 0
        self._worker_pid: int | None = None
        self._worker_failure: BaseException | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._broker_task: asyncio.Task[dict[str, Any]] | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._resource_error: Exception | None = None
        self._diagnostic = ""
        self._guest_cwd = "/"
        self._degraded: list[str] = []

    @property
    def degraded_guards(self) -> list[str]:
        return list(self._degraded)

    def _dependencies(self) -> list[str]:
        from packaging.requirements import Requirement

        env = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "extra": ""}
        dependencies = {"cloudpickle": importlib.metadata.version("cloudpickle")}
        # NumPy is a built-in wire value type, even when an agent's module did
        # not import it. Preserve its host availability and exact version.
        try:
            dependencies["numpy"] = importlib.metadata.version("numpy")
        except importlib.metadata.PackageNotFoundError:
            pass
        for text in importlib.metadata.requires("nooa") or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate(env):
                dependencies[requirement.name] = importlib.metadata.version(requirement.name)
        # Include distributions explicitly exposed as modules to the agent.
        from types import ModuleType

        from nooa.agentdoc.visibility import filter_mro_module_globals

        packages = importlib.metadata.packages_distributions()
        values = (
            list(filter_mro_module_globals(type(self._agent)).values())
            + list(self._framework_builtins.values())
            + list(self._codec.types.values())
        )
        seen = set()
        while values:
            value = values.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                values.extend(value)
            module_name = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", type(value).__module__)
            )
            if isinstance(module_name, str):
                for distribution in packages.get(module_name.split(".")[0], []):
                    if distribution not in {
                        "nooa",
                        "nooa-cli",
                        "nooa-memory",
                        "nooa-acp",
                        "nooa-bench",
                    }:
                        dependencies[distribution] = importlib.metadata.version(distribution)
        return [f"{name}=={version}" for name, version in sorted(dependencies.items())]

    async def _prepare(self, sdk: Any, image: str) -> Any:
        dependencies = self._dependencies()
        key = hashlib.sha256(
            (image + "\nuv-0.11.2\n" + "\n".join(dependencies)).encode()
        ).hexdigest()[:24]
        if key in _SNAPSHOTS:
            return _SNAPSHOTS[key]
        snapshot_name = f"nooa-runtime-{key}"
        try:
            existing = await sdk.Snapshot.get(snapshot_name)
        except Exception:
            pass
        else:
            _SNAPSHOTS[key] = existing
            return existing
        vm = await sdk.Sandbox.create(
            f"nooa-prepare-{uuid.uuid4().hex}",
            image=image,
            memory=max(1024, self._config.vm_memory_mb),
            cpus=self._config.cpus,
            network=sdk.Network.allow_all(),
        )
        try:
            version = await vm.exec(
                "python", ["-c", "import platform; print(platform.python_version())"]
            )
            if version.stdout_text.strip() != platform.python_version():
                raise SandboxUnavailable("guest image must use the host's exact Python version")
            install_uv = (
                "import urllib.request,tarfile,platform,os; "
                "arch={'x86_64':'x86_64','aarch64':'aarch64'}[platform.machine()]; "
                "url=f'https://github.com/astral-sh/uv/releases/download/0.11.2/uv-{arch}-unknown-linux-gnu.tar.gz'; "
                "urllib.request.urlretrieve(url,'/tmp/nooa-uv.tar.gz'); "
                "archive=tarfile.open('/tmp/nooa-uv.tar.gz'); "
                "member=next(m for m in archive.getmembers() if m.name.endswith('/uv')); "
                "open('/usr/local/bin/uv','wb').write(archive.extractfile(member).read()); "
                "os.chmod('/usr/local/bin/uv',0o755); os.unlink('/tmp/nooa-uv.tar.gz')"
            )
            result = await vm.exec(
                "python", ["-c", install_uv], timeout=self._config.startup_timeout_s
            )
            if result.exit_code:
                raise SandboxUnavailable("guest uv setup failed: " + result.stderr_text[-4000:])
            result = await vm.exec(
                "uv",
                ["pip", "install", "--system", *dependencies],
                timeout=self._config.startup_timeout_s,
            )
            if result.exit_code:
                raise SandboxUnavailable(
                    "guest dependency preparation failed: " + result.stderr_text[-4000:]
                )
            try:
                snapshot = await sdk.Snapshot.create(
                    snapshot_name, from_sandbox=await vm.name, full=False
                )
            except Exception:
                # Another initializer may have completed the same immutable,
                # dependency-keyed environment while this VM was preparing it.
                snapshot = await sdk.Snapshot.get(snapshot_name)
            _SNAPSHOTS[key] = snapshot
            return snapshot
        finally:
            await vm.destroy(force=True)

    def _mounts(self, sdk: Any) -> dict[str, Any]:
        self._guest_allow = []
        entries = []
        mounts = {
            "/opt/nooa-src/nooa": sdk.Volume.bind(
                str(Path(__file__).resolve().parents[2]), readonly=True
            )
        }
        if self._config.workspace:
            host = Path(self._config.workspace).absolute()
            host.mkdir(parents=True, exist_ok=True)
            self._guest_cwd = str(host) if sys.platform != "win32" else "/workspace"
            mounts[self._guest_cwd] = sdk.Volume.bind(str(host), uid=1000, gid=1000)
            entries.append((str(host), self._guest_cwd))
        for index, rule in enumerate(self._config.allow):
            host = Path(rule.path).absolute()
            if not host.exists():
                raise SandboxUnavailable(f"sandbox allow path does not exist: {host}")
            guest = str(host) if sys.platform != "win32" else f"/allowed/{index}"
            mounts[guest] = sdk.Volume.bind(
                str(host), readonly=rule.access == "read", uid=1000, gid=1000
            )
            self._guest_allow.append({"path": guest, "access": rule.access})
            entries.append((str(host), guest))
        for package in ("nooa_memory", "nooa_acp", "nooa_cli", "nooa_bench"):
            spec = importlib.util.find_spec(package)
            if spec and spec.submodule_search_locations:
                mounts[f"/opt/nooa-src/{package}"] = sdk.Volume.bind(
                    str(next(iter(spec.submodule_search_locations))), readonly=True
                )
        self._paths = PathMap(entries, windows=sys.platform == "win32")
        self._codec.path_transform = self._paths.to_host
        return mounts

    def _modules(self) -> list[Any]:
        modules = []
        library_roots = tuple(
            Path(path).resolve()
            for name in ("purelib", "platlib")
            if (path := sysconfig.get_path(name))
        )

        def by_value(module: Any) -> bool:
            if module is None:
                return False
            name = module.__name__
            if (
                name.split(".")[0] in sys.stdlib_module_names
                or name == "nooa"
                or name.startswith("nooa.")
            ):
                return False
            location = getattr(module, "__file__", None)
            # Installed libraries are prepared in the guest and retain their
            # normal class construction. Only user source needs reconstruction.
            return not (
                location
                and any(Path(location).resolve().is_relative_to(root) for root in library_roots)
            )

        for cls in type(self._agent).__mro__:
            module = inspect.getmodule(cls)
            if by_value(module) and module not in modules:
                modules.append(module)
        for kind in self._codec.types.values():
            module = inspect.getmodule(kind)
            if by_value(module) and module not in modules:
                modules.append(module)
        return modules

    def _bootstrap(self) -> dict[str, Any]:
        from nooa.agentdoc.visibility import filter_mro_module_globals

        globals_ = self._paths.guest_values(filter_mro_module_globals(type(self._agent)))
        builtins = self._paths.guest_values(
            {
                k: v
                for k, v in self._framework_builtins.items()
                if k not in {"return_result", "_call"}
            }
        )
        return {
            "globals": globals_,
            "builtins": builtins,
            "call": describe_call(self._framework_builtins.get("_call")),
            "restrictions": self._restrictions,
            "python_version": list(sys.version_info[:3]),
            "cwd": self._guest_cwd,
            "limits": {
                "max_memory_mb": self._config.max_memory_mb,
                "max_cpu_seconds": self._config.max_cpu_seconds,
            },
            "permissions": {
                "filesystem": self._config.filesystem,
                "system_paths": self._config.system_paths,
                "workspace": self._guest_cwd if self._config.workspace else None,
                "allow": self._guest_allow,
                "network": self._config.network,
                "require": self._config.require,
            },
            "max_error": self._max_error,
            "error_tail": self._error_tail,
            "truncation": self._truncation,
        }

    async def _start(self) -> None:
        try:
            if importlib.util.find_spec("cloudpickle") is None:
                raise ImportError("cloudpickle is required for guest initialization")
            import microsandbox as sdk
        except ImportError as exc:
            raise SandboxUnavailable(
                "Install nooa[sandbox] to enable microsandbox execution"
            ) from exc
        image = self._config.image or f"python:{platform.python_version()}-slim"
        try:
            mounts = self._mounts(sdk)
            name = f"nooa-{uuid.uuid4().hex}"
            if self._config.prepare_environment:
                snapshot = await self._prepare(sdk, image)
                self._vm = await sdk.Sandbox.restore(
                    snapshot,
                    name=name,
                    memory=max(self._config.vm_memory_mb, self._config.max_memory_mb + 512),
                    cpus=self._config.cpus,
                    user="0:0",
                    security=sdk.SecurityProfile.DEFAULT,
                    volumes=mounts,
                    disable_network=not self._config.network,
                    network_policy=sdk.NetworkPolicy.allow_all()
                    if self._config.network
                    else sdk.NetworkPolicy.none(),
                )
            else:
                self._vm = await sdk.Sandbox.create(
                    name,
                    image=image,
                    memory=self._config.vm_memory_mb,
                    cpus=self._config.cpus,
                    user="0:0",
                    security=sdk.SecurityProfile.DEFAULT,
                    volumes=mounts,
                    network=sdk.Network.allow_all() if self._config.network else sdk.Network.none(),
                )
            self._stream = await self._vm.exec_stream(
                "python",
                [
                    "-u",
                    "-c",
                    "import os; fd=os.dup(1); os.dup2(2,1); from nooa.runtime.sandbox.guest import main; main(protocol_fd=fd)",
                ],
                env={"PYTHONPATH": "/opt/nooa-src", "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=sdk.Stdin.pipe(),
                timeout=None,
                tty=False,
            )
            self._sink = self._stream.take_stdin()
            if self._sink is None:
                raise SandboxUnavailable("microsandbox did not provide the worker input stream")
            self._messages = asyncio.Queue(maxsize=8)
            self._worker_failure = None
            self._pump = asyncio.create_task(self._read_events())
            await self._sink.write(
                frame(
                    dumps_namespace(
                        self._bootstrap(), modules=self._modules(), path_map=self._paths
                    )
                )
            )
            ready = await self._messages.get()
            if isinstance(ready, BaseException):
                raise ready
            if ready.get("type") != "ready":
                raise SandboxUnavailable(ready.get("error", "guest worker did not initialize"))
            self._worker_pid = ready["pid"]
            self._degraded = ready.get("degraded", [])
            if self._degraded:
                logger.warning(
                    "microsandbox guest has unavailable limits: %s", ", ".join(self._degraded)
                )
            self._resource_error = None
            if self._config.max_memory_mb or self._config.max_cpu_seconds:
                self._watchdog = asyncio.create_task(self._watch_resources(ready))
        except BaseException:
            await self._dispose()
            raise

    async def _read_events(self) -> None:
        reader = FrameReader()
        failure: BaseException | None = None
        try:
            async for event in self._stream:
                kind = str(event.event_type)
                if kind == "stdout":
                    for data in reader.feed(event.data or b""):
                        message = self._codec.loads(data)
                        if not isinstance(message, dict):
                            raise CellSerializationError("sandbox message must be an object")
                        await self._messages.put(message)
                elif kind == "stderr":
                    self._diagnostic = (
                        self._diagnostic + (event.data or b"").decode(errors="replace")
                    )[-4000:]
                elif kind == "exited":
                    reader.finish()
                    failure = self._resource_error or WorkerDiedError(
                        f"guest worker exited ({event.code}): {self._diagnostic}"
                    )
                    break
        except asyncio.CancelledError:
            return
        except (CellSerializationError, ValueError) as exc:
            failure = WorkerDiedError(f"malformed message from sandbox interpreter: {exc}")
        except BaseException as exc:
            failure = exc
        self._worker_failure = failure or WorkerDiedError("guest execution stream closed")
        await self._messages.put(self._worker_failure)

    async def _watch_resources(self, ready: Any) -> None:
        vm = self._vm
        while self._vm is vm and self._worker_failure is None:
            await asyncio.sleep(self._config.rss_poll_s)
            try:
                if self._config.max_cpu_seconds:
                    text = await vm.fs.read_text(f"/proc/{self._worker_pid}/stat")
                    fields = text.rsplit(")", 1)[1].split()
                    cpu = (int(fields[11]) + int(fields[12])) / ready["clock_ticks"]
                    if cpu - ready["cpu_baseline"] >= self._config.max_cpu_seconds:
                        self._resource_error = CellTimeoutError(
                            "cell exceeded the worker CPU budget"
                        )
                if self._config.max_memory_mb:
                    text = await vm.fs.read_text(f"/proc/{self._worker_pid}/status")
                    rss = next(
                        int(line.split()[1]) * 1024
                        for line in text.splitlines()
                        if line.startswith("VmRSS:")
                    )
                    if rss - ready["rss_baseline"] > self._config.max_memory_mb * 1024 * 1024:
                        self._resource_error = CellMemoryError(
                            "cell exceeded the worker memory budget"
                        )
                if self._resource_error:
                    self._worker_failure = self._resource_error
                    await self._messages.put(self._resource_error)
                    return
            except asyncio.CancelledError:
                return
            except Exception:
                # Kernel rlimits continue to enforce the caps if the process
                # exits between a metadata read and the exec stream's exit event.
                return

    async def _send(self, value: Any) -> None:
        value = self._paths.guest_values(value)
        await self._sink.write(
            frame(dumps_namespace(value, modules=self._modules(), path_map=self._paths))
        )

    def _synth_error(self, error: Any) -> ExecutionResult:
        return ExecutionResult(
            stdout="The sandbox interpreter stopped. Its variables and functions were cleared.\n",
            error=error,
            defined_methods={},
        )

    async def run_cell(self, code: str, *, execution_count: int = 1) -> ExecutionResult:
        if self._closed:
            raise WorkerDiedError("sandbox executor is closed")
        async with self._lock:
            if self._closed:
                raise WorkerDiedError("sandbox executor is closed")
            if self._disabled:
                return self._synth_error(WorkerDiedError("sandbox recovery is disabled"))
            if self._worker_failure is not None:
                await self._recover()
                if self._config.recovery == "disabled":
                    return self._synth_error(self._worker_failure)
            if self._vm is None:
                try:
                    self._startup_task = asyncio.create_task(self._start())
                    await asyncio.wait_for(self._startup_task, self._config.startup_timeout_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raise SandboxUnavailable(f"microsandbox initialization failed: {exc}") from exc
                finally:
                    self._startup_task = None
            self._req_id += 1
            request_id = self._req_id
            try:
                await self._send(
                    {
                        "op": "run",
                        "id": request_id,
                        "code": code,
                        "execution_count": execution_count,
                    }
                )
                deadline = (
                    time.monotonic() + self._cell_timeout + self._config.timeout_grace_s
                    if self._cell_timeout
                    else None
                )
                while True:
                    timeout = max(0, deadline - time.monotonic()) if deadline else None
                    message = await asyncio.wait_for(self._messages.get(), timeout)
                    if self._closed:
                        raise WorkerDiedError("sandbox executor closed during execution")
                    if isinstance(message, BaseException):
                        raise message
                    if message.get("type") == "tool_call":
                        started = time.monotonic()
                        response = await self._service_tool_call(message)
                        await self._send(response)
                        if deadline:
                            deadline += time.monotonic() - started
                        continue
                    if message.get("type") != "response" or message.get("id") != request_id:
                        raise CellSerializationError("unexpected sandbox response")
                    return cast(
                        ExecutionResult,
                        dto_to_result(
                            dto_from_wire(message["result"]),
                            codec=self._codec,
                            signal_factory=self._signal_factory,
                        ),
                    )
            except asyncio.CancelledError:
                await self._dispose()
                raise
            except TimeoutError:
                await self._recover()
                return self._synth_error(CellTimeoutError("cell exceeded its wall-clock deadline"))
            except Exception as exc:
                await self._recover()
                return self._synth_error(exc)

    async def _service_tool_call(self, message: dict[str, Any]) -> dict[str, Any]:
        """Validate a guest request and reply without dispatching refused values."""
        try:
            if message.get("kind") == "introspect":
                response = await self._introspect(message)
            else:
                call = self._decode_tool_call(message)
                try:
                    self._broker_task = asyncio.create_task(self._dispatch_tool_call(call))
                    response = await asyncio.wait_for(
                        self._broker_task, self._config.broker_timeout_s or None
                    )
                except TimeoutError:
                    raise CellTimeoutError(
                        "brokered self.* call exceeded "
                        f"broker_timeout_s={self._config.broker_timeout_s}s"
                    ) from None
                finally:
                    self._broker_task = None
        except CellSerializationError as exc:
            response = {
                "ok": False,
                "error_type": "CellSerializationError",
                "error": _bounded_text(exc, "sandbox broker decode failed", limit=self._max_error),
            }
        response.update(type="tool_result", tool_call_id=message.get("tool_call_id"))
        return response

    async def _introspect(self, message: Any) -> dict[str, Any]:
        from nooa.agentdoc import doc
        from nooa.agentdoc.introspect import methods, variables

        functions: dict[str, Any] = {"doc": doc, "methods": methods, "variables": variables}
        path = message.get("path")
        if not isinstance(path, list) or len(path) != 1 or path[0] not in functions:
            raise CellSerializationError("invalid sandbox introspection request")
        payload = message.get("payload")
        if not isinstance(payload, bytes):
            raise CellSerializationError("malformed sandbox introspection payload")
        value = self._codec.loads(payload)
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            raise CellSerializationError("malformed sandbox introspection arguments")
        args, kwargs = value
        return {"ok": True, "result": functions[path[0]](self._agent, *args, **kwargs)}

    @staticmethod
    def _signal_factory(payload: Any) -> Any:
        from nooa.strategies.codeact import _ReturnResultSignal

        return _ReturnResultSignal(result=payload)

    async def _recover(self) -> None:
        await self._dispose()
        self._disabled = self._config.recovery == "disabled"

    async def _dispose(self) -> None:
        if self._cleanup_task is not None:
            await asyncio.shield(self._cleanup_task)
            return
        self._cleanup_task = asyncio.create_task(self._destroy())
        try:
            await asyncio.shield(self._cleanup_task)
        finally:
            self._cleanup_task = None

    async def _destroy(self) -> None:
        pump, vm = self._pump, self._vm
        self._pump = self._vm = self._stream = self._sink = None
        watchdog, self._watchdog = self._watchdog, None
        if watchdog and watchdog is not asyncio.current_task():
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
        if self._messages is not None:
            while not self._messages.empty():
                self._messages.get_nowait()
            self._messages.put_nowait(WorkerDiedError("sandbox interpreter stopped"))
        if pump and pump.get_loop() is asyncio.get_running_loop():
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        if vm:
            await vm.destroy(force=True, timeout=10)

    async def aclose(self) -> None:
        self._closed = True
        if self._startup_task and not self._startup_task.done():
            self._startup_task.cancel()
            await asyncio.gather(self._startup_task, return_exceptions=True)
        if self._broker_task and not self._broker_task.done():
            self._broker_task.cancel()
        await self._dispose()

    def close_sync(self) -> None:
        self._closed = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._dispose())
        else:
            loop.create_task(self._dispose())

    def _decode_tool_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Check the broker request shape and decode its cell-supplied payload.

        Raises ``CellSerializationError`` for a refused value (a type the agent
        did not declare) or a malformed request.
        """
        kind, path = msg.get("kind"), msg.get("path") or []
        if kind not in ("call", "attr", "setattr", "iter"):
            raise CellSerializationError(f"malformed sandbox broker request: kind {kind!r}")
        if not isinstance(path, list) or not all(isinstance(p, str) for p in path):
            raise CellSerializationError("malformed sandbox broker request: bad path")
        call: dict[str, Any] = {"kind": kind, "path": path}
        if kind not in ("call", "setattr"):
            return call
        what = f"self.{'.'.join(path)}" + ("" if kind == "setattr" else "(...)")
        payload = msg.get("payload")
        if not isinstance(payload, bytes):
            raise CellSerializationError(f"malformed sandbox broker request for {what}")
        try:
            value = self._codec.loads(payload)
        except CellSerializationError as exc:
            raise CellSerializationError(
                f"Value passed to {what} cannot cross the sandbox boundary: {exc}. {wire.HINT}"
            ) from exc
        if kind == "setattr":
            call["value"] = value
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            call["args"], call["kwargs"] = value
        else:
            raise CellSerializationError(f"malformed sandbox broker request for {what}")
        return call

    def _walk_path(self, path: list[str]) -> Any:
        """Resolve a dotted attribute path (``["memory", "remember"]``) on the agent."""
        obj: Any = self._agent
        for part in path:
            obj = getattr(obj, part)
        return obj

    async def _dispatch_tool_call(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Run a brokered ``self.<path>`` access against the parent's live agent."""
        from nooa.events import ExecutionSignal

        path = msg.get("path") or []
        display = ".".join(path)
        kind = msg.get("kind")
        target: Any = None
        try:
            if kind == "setattr":
                # self.<path> = value on the parent's live agent.
                obj = self._walk_path(path[:-1])
                setattr(obj, path[-1], msg.get("value"))
                return {"ok": True, "result": None}
            if kind == "iter":
                # Materialize list(obj) on the parent (the iterator isn't picklable).
                items = list(self._walk_path(path))
                if not is_picklable(items):
                    return {
                        "ok": False,
                        "error_type": "CellSerializationError",
                        "error": (
                            f"iterating self.{display} produced non-picklable items that "
                            "cannot cross the sandbox boundary."
                        ),
                    }
                return {"ok": True, "result": items}
            target = self._walk_path(path)
            if kind == "attr":
                if callable(target) and not isinstance(target, type):
                    return {
                        "ok": True,
                        "result": None,
                        "proxy": True,
                        "callable": True,
                        "is_async": inspect.iscoroutinefunction(target)
                        or inspect.iscoroutinefunction(getattr(target, "__call__", None)),  # noqa: B004 - target is already known callable
                    }
                # Picklable state crosses; a live object becomes a nested proxy.
                if is_picklable(target):
                    return {"ok": True, "result": target}
                return {"ok": True, "result": None, "proxy": True}
            value = target(*msg.get("args", ()), **msg.get("kwargs", {}))
            # Record whether the call was async so the worker can re-wrap the
            # (already-resolved) result in an awaitable — otherwise a cell doing
            # ``await proxy(...)`` on an async callable reached via a nested proxy
            # would await a plain value and raise TypeError.
            was_async = asyncio.iscoroutine(value) or asyncio.isfuture(value)
            if was_async:
                value = await value
            if not is_picklable(value):
                return {
                    "ok": False,
                    "error_type": "CellSerializationError",
                    "error": (
                        f"self.{display} returned a {type(value).__name__!r} that is not "
                        "picklable and cannot cross the sandbox boundary. Return a "
                        "picklable summary instead."
                    ),
                }
            return {"ok": True, "result": value, "was_async": was_async}
        except ExecutionSignal as sig:
            # Some tools end the turn by *raising* a control-flow signal (e.g. an
            # ARC submit_actions -> return_result). Marshal it back to the cell so
            # it re-raises there and flows through the normal signal path.
            payload = getattr(sig, "result", None)
            payload_is_picklable = is_picklable(payload)
            return {
                "ok": False,
                "error_type": "ExecutionSignal",
                "error": _bounded_text(sig, "ExecutionSignal", limit=self._max_error),
                "signal_result": payload if payload_is_picklable else None,
                "signal_result_dropped": not payload_is_picklable,
            }
        except CellSerializationError as exc:
            return {
                "ok": False,
                "error_type": "CellSerializationError",
                "error": _bounded_text(exc, "CellSerializationError", limit=self._max_error),
            }
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the cell
            # The parent still has the resolved target and can safely derive its
            # agentdoc. The child cannot: it only has a proxy and a traceback-local
            # surrogate exception. Transport a bounded string hint, never the
            # callable or traceback itself.
            from nooa.errors.formatting import _bad_call_agentdoc

            call_hint = _bad_call_agentdoc(exc, target=target)
            response = {
                "ok": False,
                "error_type": _bounded_text(type(exc).__name__, "Exception", limit=self._max_error),
                "error": _bounded_text(exc, type(exc).__name__, limit=self._max_error),
            }
            if call_hint:
                response["call_hint"] = _bounded_text(call_hint, "", limit=self._max_error)
            return response

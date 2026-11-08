# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-lived Python interpreter inside a microsandbox microVM."""

from __future__ import annotations

# Reserve a separate descriptor before importing anything that may print.
import asyncio
import os
import struct
import sys
import threading

from nooa.errors.formatting import format_error_for_llm as _FORMAT_ERROR_DEFAULT
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.runtime.sandbox.protocol import MAX_FRAME_BYTES, frame
from nooa.runtime.sandbox.serialization import dto_to_wire, result_to_dto
from nooa.runtime.sandbox.worker import ChildBroker, ParentAgentProxy, build_namespace

_OUTPUT = None
_INPUT = None


class Connection:
    def send_bytes(self, data):
        _OUTPUT.write(frame(data))

    def recv(self):
        import cloudpickle

        header = _INPUT.read(4)
        if not header:
            raise EOFError
        if len(header) != 4:
            raise ValueError("incomplete sandbox command header")
        size = struct.unpack("!I", header)[0]
        if not 0 < size <= MAX_FRAME_BYTES:
            raise ValueError("invalid sandbox command length")
        data = _INPUT.read(size)
        if len(data) != size:
            raise ValueError("incomplete sandbox command")
        return cloudpickle.loads(data)  # host-authorized input only


def return_function(return_type):
    from nooa.strategies.codeact import _ReturnResultSignal

    def return_result(*args, **kwargs):
        if len(args) > 1:
            raise ValueError("return_result() takes at most 1 positional argument")
        if args and kwargs:
            if "kind" not in getattr(return_type, "model_fields", {}):
                raise ValueError("Cannot mix positional and keyword arguments")
            if "kind" in kwargs:
                raise ValueError("return_result() got kind both positionally and by keyword")
            raise _ReturnResultSignal(result={**kwargs, "kind": args[0]})
        raise _ReturnResultSignal(result={"result": args[0]} if args else kwargs)

    return return_result


def cell_state_function(namespace, input_names):
    from nooa.strategies.codeact_v2 import CodeActV2

    initial = dict(namespace)
    inputs = {name: namespace.get(name) for name in input_names}

    def python_cell_state():
        # The guest combines globals and locals. Omit unchanged module and
        # framework bindings, while retaining inputs and cell-created values.
        live = {
            name: value
            for name, value in namespace.items()
            if name in inputs or name not in initial or value is not initial[name]
        }
        return CodeActV2._describe_cell_state(inputs, live)

    return python_cell_state


def apply_limits(config):
    import resource

    if config["max_memory_mb"]:
        baseline = config["baseline"]
        cap = baseline + config["max_memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    if config["max_cpu_seconds"]:
        import math

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cap = math.ceil(usage.ru_utime + usage.ru_stime) + config["max_cpu_seconds"]
        resource.setrlimit(resource.RLIMIT_CPU, (cap, cap))


async def _run_one(
    namespace,
    request,
    *,
    capture_config=None,
    max_attachments=20,
    max_error=None,
    error_tail=None,
    error_formatter=_FORMAT_ERROR_DEFAULT,
):
    result = await run_cell_source(
        request["code"],
        namespace,
        execution_count=request["execution_count"],
        capture_config=capture_config,
        max_attachments=max_attachments,
    )
    return result_to_dto(
        result, max_error=max_error, tail_chars=error_tail, error_formatter=error_formatter
    )


def main(*, protocol_fd=None):
    global _OUTPUT, _INPUT

    _OUTPUT = os.fdopen(os.dup(1) if protocol_fd is None else protocol_fd, "wb", buffering=0)
    if protocol_fd is None:
        os.dup2(2, 1)
    _INPUT = sys.stdin.buffer
    conn = Connection()
    lock = threading.RLock()
    try:
        init = conn.recv()
        if tuple(sys.version_info[:3]) != tuple(init["python_version"]):
            raise RuntimeError("host and sandbox Python versions must match")
        broker = ChildBroker(conn, lock)
        proxy = ParentAgentProxy(broker, None)
        builtins = init["builtins"]
        builtins["_call"] = init["call"]
        builtins["return_result"] = return_function(init["call"].return_type)
        namespace = build_namespace(
            None, builtins, proxy, init["restrictions"], module_globals=init["globals"]
        )
        if init.get("cell_state_inputs") is not None:
            namespace["python_cell_state"] = cell_state_function(
                namespace, init["cell_state_inputs"]
            )
        if init["cwd"]:
            os.chdir(init["cwd"])
            sys.path.insert(0, init["cwd"])
        from nooa.runtime.sandbox.guest_policy import apply_permissions

        with open("/proc/self/status") as status:
            memory = {
                line.split(":", 1)[0]: int(line.split()[1]) * 1024
                for line in status
                if line.startswith(("VmSize:", "VmRSS:"))
            }
            baseline = memory["VmSize"]
        degraded = apply_permissions(init["permissions"])
        apply_limits({**init["limits"], "baseline": baseline})
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        wire.send(
            conn,
            {
                "type": "ready",
                "python_version": list(sys.version_info[:3]),
                "pid": os.getpid(),
                "degraded": degraded,
                "rss_baseline": memory.get("VmRSS", 0),
                "cpu_baseline": usage.ru_utime + usage.ru_stime,
                "clock_ticks": os.sysconf("SC_CLK_TCK"),
            },
        )
    except BaseException as exc:
        wire.send(conn, {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    run_one = _run_one
    try:
        while True:
            request = conn.recv()
            if request["op"] == "shutdown":
                return
            dto = loop.run_until_complete(
                run_one(
                    namespace,
                    request,
                    capture_config=init["truncation"].capture,
                    max_attachments=init["truncation"].media_capture.max_attachments_per_execution,
                    max_error=init["max_error"],
                    error_tail=init["error_tail"],
                )
            )
            wire.send(conn, {"type": "response", "id": request["id"], "result": dto_to_wire(dto)})
    except EOFError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()

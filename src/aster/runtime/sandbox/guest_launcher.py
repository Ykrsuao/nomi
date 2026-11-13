# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run as a guest script so kernel kills retain their final resource usage.

The SDK reports signal exits as -1. This small parent stays outside the worker's
limits and sends its wait4 result before the exec stream closes. It runs only
inside the microVM and imports no framework code before starting the worker.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import msgpack


def main() -> None:
    with subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import os; fd=os.dup(1); os.dup2(2,1); "
            "from aster.runtime.sandbox.guest import main; main(protocol_fd=fd)",
        ]
    ) as worker:
        _, status, usage = os.wait4(worker.pid, 0)
        worker.returncode = os.waitstatus_to_exitcode(status)
        if os.WIFSIGNALED(status):
            payload = msgpack.packb(
                {
                    "type": "worker_exit",
                    "signal": os.WTERMSIG(status),
                    "cpu_seconds": usage.ru_utime + usage.ru_stime,
                }
            )
            assert isinstance(payload, bytes)
            # A fixed, tiny message in the same length-prefixed wire format.
            sys.stdout.buffer.write(struct.pack("!I", len(payload)) + payload)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise guest mount permissions in a disposable Linux root."""

import os
import stat
import subprocess
import sys

import pytest


@pytest.mark.sandbox
@pytest.mark.skipif(
    sys.platform != "linux" or getattr(os, "geteuid", lambda: -1)() != 0,
    reason="requires a Linux process with permission to create mount namespaces",
)
def test_nested_allow_does_not_change_workspace_permissions(tmp_path):
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    nested.mkdir(parents=True)
    workspace.chmod(0o777)
    nested.chmod(0o755)
    readonly = nested / "readonly.txt"
    readonly.write_text("allowed", encoding="utf-8")
    readonly.chmod(0o666)  # The read-only mount must deny writes, not the file mode.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes, os, sys
from pathlib import Path
from aster.runtime.sandbox.guest_policy import _filesystem_jail
libc = ctypes.CDLL(None, use_errno=True)
assert libc.unshare(0x00020000) == 0
assert libc.mount(None, b'/', None, (1 << 14) | (1 << 18), None) == 0
root = os.fsencode(sys.argv[1])
assert libc.mount(root, root, None, 1 << 12, None) == 0
os.chroot(sys.argv[1])
os.chdir('/')
_filesystem_jail({
    'workspace': '/workspace', 'system_paths': False,
    'allow': [{'path': '/workspace/nested/readonly.txt', 'access': 'read'}],
})
os.setgroups([])
os.setgid(1000)
os.setuid(1000)
assert 'nested' in os.listdir('/workspace')
Path('/workspace/created.txt').write_text('saved')
assert Path('/workspace/nested/readonly.txt').read_text() == 'allowed'
try:
    Path('/workspace/nested/readonly.txt').write_text('changed')
except PermissionError:
    pass
except OSError as error:
    import errno
    assert error.errno == errno.EROFS
else:
    raise AssertionError('read-only grant became writable')
""",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o777
    assert stat.S_IMODE(nested.stat().st_mode) == 0o755
    assert (workspace / "created.txt").read_text() == "saved"

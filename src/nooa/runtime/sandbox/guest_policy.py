# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fine-grained guest permissions; the microVM owns host isolation."""

from __future__ import annotations

import ctypes
import os
import platform
import socket
import struct

_NR_LANDLOCK_CREATE_RULESET = 444

_NR_LANDLOCK_ADD_RULE = 445

_NR_LANDLOCK_RESTRICT_SELF = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0

_LANDLOCK_RULE_PATH_BENEATH = 1

_PR_SET_NO_NEW_PRIVS = 38

_SECCOMP_SET_MODE_FILTER = 1

_ARCH = platform.machine()

if _ARCH == "x86_64":
    _NR_SECCOMP = 317
    _AUDIT_ARCH = 0xC000003E
    _NR_SOCKET = 41
elif _ARCH in ("aarch64", "arm64"):
    _NR_SECCOMP = 277
    _AUDIT_ARCH = 0xC00000B7
    _NR_SOCKET = 198
else:  # pragma: no cover - unsupported arch; probes below report unavailable
    _NR_SECCOMP = -1
    _AUDIT_ARCH = 0
    _NR_SOCKET = -1


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _install_no_new_privs() -> None:
    if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _build_no_inet_filter() -> bytes:
    """BPF program: deny ``socket(AF_INET/AF_INET6)`` with EACCES, allow the rest.

    ``AF_UNIX`` (and every other syscall) is untouched, so the parent broker pipe
    and local IPC keep working while all internet sockets are refused at creation.
    """
    BPF_LD = 0x00
    BPF_JMP = 0x05
    BPF_RET = 0x06
    BPF_W = 0x00
    BPF_ABS = 0x20
    BPF_JEQ = 0x10
    BPF_K = 0x00
    RET_ALLOW = 0x7FFF0000
    RET_ERRNO = 0x00050000
    EACCES = 13

    def ins(code: int, jt: int, jf: int, k: int) -> bytes:
        return struct.pack("HBBI", code, jt, jf, k)

    # seccomp_data offsets: nr@0, arch@4, args[0]@16 (little-endian low word).
    return b"".join(
        [
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 4),  # A = arch
            ins(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, _AUDIT_ARCH),  # native? -> continue
            ins(BPF_RET | BPF_K, 0, 0, RET_ERRNO | EACCES),  # foreign ABI -> deny
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),  # A = nr
            ins(BPF_JMP | BPF_JEQ | BPF_K, 0, 4, _NR_SOCKET),  # nr!=socket -> allow(+4)
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 16),  # A = args[0] (domain)
            ins(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, socket.AF_INET),  # INET -> errno
            ins(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, socket.AF_INET6),  # INET6 -> errno else allow
            ins(BPF_RET | BPF_K, 0, 0, RET_ERRNO | EACCES),
            ins(BPF_RET | BPF_K, 0, 0, RET_ALLOW),
        ]
    )


def _seccomp_install(prog: bytes) -> None:
    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = _SockFprog(len=len(prog) // 8, filter=ctypes.cast(buf, ctypes.c_void_p).value)
    if _libc().syscall(_NR_SECCOMP, _SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(fprog)) != 0:
        raise OSError(ctypes.get_errno(), "seccomp(SET_MODE_FILTER) failed")


def apply_seccomp_no_inet() -> None:
    """Block creation of any internet (AF_INET/AF_INET6) socket, irrevocably."""
    if _NR_SECCOMP < 0:
        raise OSError("seccomp is not available on this architecture")
    _install_no_new_privs()
    _seccomp_install(_build_no_inet_filter())


def _filesystem_jail(settings):
    from pathlib import Path

    from nooa.runtime.sandbox.config import FileRule, SandboxConfig, resolve_spec

    config = SandboxConfig(
        workspace=settings["workspace"],
        system_paths=settings["system_paths"],
        allow=tuple(FileRule(**rule) for rule in settings["allow"]),
    )
    rules = resolve_spec(config).landlock_rules
    libc = _libc()
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    libc.mount.restype = ctypes.c_int

    def mount(source, target, flags):
        if libc.mount(
            os.fsencode(source) if source else None, os.fsencode(target), None, flags, None
        ):
            raise OSError(ctypes.get_errno(), f"cannot mount sandbox path {target}")

    if libc.unshare(0x00020000):
        raise OSError(ctypes.get_errno(), "cannot create guest mount namespace")
    mount(None, "/", 1 << 14 | 1 << 18)  # MS_REC | MS_PRIVATE
    root = Path("/nooa-jail")
    root.mkdir(mode=0o111)
    for name in (
        "tmp",
        "var",
        "home",
        "root",
        "etc",
        "usr",
        "opt",
        "dev",
        "proc",
        "sys",
        "bin",
        "sbin",
        "lib",
        "lib64",
    ):
        (root / name).mkdir(mode=0o000)
    available = [rule for rule in rules if Path(rule.path).exists()]
    for rule in sorted(available, key=lambda item: len(Path(item.path).parts)):
        source = Path(rule.path)
        target = root / source.relative_to("/")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o111)
            if source.is_dir():
                target.mkdir(mode=0o111)
            else:
                target.touch(mode=0o000)
        mount(str(source), str(target), 1 << 12)
        if not rule.write:
            mount(None, str(target), 1 << 12 | 1 << 5 | 1)
    for rule in available:
        target = root / Path(rule.path).relative_to("/")
        for parent in target.parents:
            if parent == root:
                break
            try:
                os.chmod(parent, 0o111)
            except OSError:
                pass
    os.chroot(root)
    os.chdir(settings["workspace"] or "/")


def apply_permissions(settings):
    degraded = []
    if settings["filesystem"]:
        try:
            _filesystem_jail(settings)
        except OSError:
            if settings["require"]:
                raise
            degraded.append("filesystem")
    _install_no_new_privs()
    if not settings["network"]:
        try:
            apply_seccomp_no_inet()
        except OSError:
            if settings["require"]:
                raise
            degraded.append("socket creation (microVM network remains disabled)")
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    return degraded

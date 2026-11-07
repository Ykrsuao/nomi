# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded frames for the SDK's chunked exec stream."""

from __future__ import annotations

import struct
from collections.abc import Iterator

from nooa.runtime.sandbox.errors import CellSerializationError

MAX_FRAME_BYTES = 64 * 1024 * 1024


def frame(data: bytes) -> bytes:
    if not 0 < len(data) <= MAX_FRAME_BYTES:
        raise CellSerializationError("sandbox message exceeds the transport limit")
    return struct.pack("!I", len(data)) + data


class FrameReader:
    def __init__(self) -> None:
        self._data = bytearray()

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self._data.extend(chunk)
        while len(self._data) >= 4:
            size = struct.unpack("!I", self._data[:4])[0]
            if not 0 < size <= MAX_FRAME_BYTES:
                raise CellSerializationError("invalid sandbox frame length")
            if len(self._data) < size + 4:
                return
            body = bytes(self._data[4 : size + 4])
            del self._data[: size + 4]
            yield body

    def finish(self) -> None:
        if self._data:
            raise CellSerializationError("sandbox exited with an incomplete frame")

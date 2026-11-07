# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host implementation selection; all supported hosts start normally."""

import sys

from nooa.errors import NemoOOAgentsRuntimeError


class PosixOnlyFeatureError(NemoOOAgentsRuntimeError, NotImplementedError):
    """Legacy exception for third-party POSIX-only extensions."""

    def __init__(self, feature: str):
        self.feature = feature
        super().__init__(f"{feature} requires Linux or macOS. Use WSL 2 for this extension.")


def is_native_windows() -> bool:
    return sys.platform == "win32"

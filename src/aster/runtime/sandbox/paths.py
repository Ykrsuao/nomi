# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate authorized host paths at the guest data boundary."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import BaseModel


class PathMap:
    def __init__(self, entries: Any, *, windows: Any = False) -> None:
        self.entries = tuple(entries)
        self.windows = windows

    def to_guest(self, text: str) -> str:
        source_type = PureWindowsPath if self.windows else PurePosixPath
        path = source_type(text)
        if not path.is_absolute() or "\n" in text or "\0" in text:
            return text
        for host, guest in sorted(self.entries, key=lambda entry: len(entry[0]), reverse=True):
            try:
                suffix = path.relative_to(source_type(host))
            except ValueError:
                continue
            return str(PurePosixPath(guest, *suffix.parts))
        return text

    def to_host(self, text: str) -> str:
        path = PurePosixPath(text)
        if not path.is_absolute() or "\n" in text or "\0" in text:
            return text
        host_type = PureWindowsPath if self.windows else PurePosixPath
        for host, guest in sorted(self.entries, key=lambda entry: len(entry[1]), reverse=True):
            try:
                suffix = path.relative_to(PurePosixPath(guest))
            except ValueError:
                continue
            return str(host_type(host, *suffix.parts))
        return text

    def guest_values(self, value: Any) -> Any:
        memo: dict[int, Any] = {}

        def walk(item: Any) -> Any:
            result: Any
            if id(item) in memo:
                return memo[id(item)]
            if isinstance(item, PurePath):
                return PurePosixPath(self.to_guest(str(item)))
            if isinstance(item, str):
                mapped = self.to_guest(item)
                return item if mapped == item else mapped
            if isinstance(item, dict):
                result = {} if type(item) is dict else copy.copy(item)
                result.clear()
                memo[id(item)] = result
                result.update((walk(key), walk(value)) for key, value in item.items())
                return result
            if isinstance(item, list):
                result = []
                memo[id(item)] = result
                result.extend(walk(value) for value in item)
                return result
            if isinstance(item, (tuple, set, frozenset)):
                memo[id(item)] = item
                if isinstance(item, tuple) and hasattr(item, "_fields"):
                    result = type(item)(*(walk(value) for value in item))
                else:
                    result = type(item)(walk(value) for value in item)
                memo[id(item)] = result
                return result
            if isinstance(item, BaseModel):
                memo[id(item)] = item
                result = item.model_copy(
                    update={key: walk(getattr(item, key)) for key in type(item).model_fields}
                )
                memo[id(item)] = result
                return result
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                memo[id(item)] = item
                updates = {
                    field.name: walk(getattr(item, field.name))
                    for field in dataclasses.fields(item)
                }
                if all(value is getattr(item, key) for key, value in updates.items()):
                    return item
                result = copy.copy(item)
                memo[id(item)] = result
                for key, value in updates.items():
                    object.__setattr__(result, key, value)
                return result
            return item

        return walk(value)


def guest_path(text: str) -> Path:
    return Path(text)

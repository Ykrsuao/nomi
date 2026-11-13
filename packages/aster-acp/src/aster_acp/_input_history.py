# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded read-only bootstrap for the terminal's submitted-input history."""

import heapq
import json

from aster_cli.sessions import SessionStore


def recent_inputs(
    store: SessionStore, *, limit: int = 1_000, max_bytes: int = 4 * 1024 * 1024
) -> list[str]:
    # SessionStore's readers do not acquire writer ownership or construct agents.
    candidates = (
        (turn.timestamp, session_index, turn_index, turn.content)
        for session_index, session in enumerate(store.list(limit=100))
        for turn_index, turn in enumerate(store.load_turns(session.id))
        if turn.role == "user" and turn.content.strip()
    )
    selected: list[str] = []
    used = 0
    for _, _, _, text in heapq.nlargest(limit, candidates):
        size = len(json.dumps(text, ensure_ascii=False).encode("utf-8")) + 1
        if size > max_bytes - used:
            continue
        selected.append(text)
        used += size
    return list(reversed(selected))

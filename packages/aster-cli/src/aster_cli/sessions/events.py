# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata for interactive coding-agent sessions."""

from typing import Any, ClassVar, Literal

from aster.context_blocks import Metadata
from aster.context_blocks.roles import Role

type SteeringState = Literal["accepted", "applied", "cancelled", "interrupted", "turn_finished"]


class SessionStarted(Metadata):
    """Identity and environment recorded once when a session is created."""

    _role: ClassVar[Role] = Role.METADATA

    origin: str = ""
    model: str = ""
    agent: str = ""
    working_directory: str = ""
    provisional: bool = False


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by the agent runtime as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""
    turn_id: str | None = None


class SessionTurnFinished(Metadata):
    """Bind a foreground turn to its completed reply, or mark it incomplete."""

    user_message_id: str
    completed_message_id: str | None


class SessionSteering(Metadata):
    """A durable state change for input addressed to an existing foreground turn."""

    message_id: str
    turn_id: str
    text: str
    state: SteeringState
    reason: str | None = None


class SessionActivity(Metadata):
    """A host-visible tool activity snapshot, excluded from model context."""

    activity: dict[str, Any]


SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
    SessionTurnFinished,
    SessionSteering,
    SessionActivity,
)

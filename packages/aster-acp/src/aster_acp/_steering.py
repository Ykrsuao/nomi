# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable live input owned by one ACP session, addressed to one original turn."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from acp.schema import SessionInfoUpdate
from aster_cli.coding.steering import current_steering
from aster_cli.sessions import SessionHandle, SessionSteering, SteeringState
from pydantic import ValidationError

from aster.agents.summarization import summary_fork_active
from aster.events import Notification
from aster.interactive import Done, NeedInput, Waiting

if TYPE_CHECKING:
    from aster.agent import Agent
    from aster.runtime.middleware import LLMCallContext, LLMCallNext
    from aster.strategies.base import RuntimeServices

_SOURCE = "New message from the user while you were working."


def steering_update(event: SessionSteering) -> SessionInfoUpdate:
    payload = {
        "messageId": event.message_id,
        "turnId": event.turn_id,
        "text": event.text,
        "state": event.state,
    }
    if event.reason:
        payload["reason"] = event.reason
    return SessionInfoUpdate(
        session_update="session_info_update", field_meta={"aster": {"steering": payload}}
    )


class SteeringInbox:
    """Accept durably, apply at the next model request, and terminate leftovers."""

    def __init__(self, agent: Agent, handle: SessionHandle, publish: Callable[[Any], None]) -> None:
        self.agent = agent
        self.handle = handle
        self.publish = publish
        self.wakeup = asyncio.Event()
        self.turn_id: str | None = None
        self._pending: list[str] = []
        self._messages: dict[str, SessionSteering] = {}
        self._turns: set[str] = set()
        for record in handle.transcript():
            if record.event_type in {"SessionUserMessage", "TUIUserInput"}:
                turn_id = record.data.get("turn_id")
                if isinstance(turn_id, str):
                    self._turns.add(turn_id)
            elif record.event_type == "SessionSteering":
                try:
                    event = SessionSteering.model_validate(record.data)
                except ValidationError:
                    continue
                self._messages[event.message_id] = event
        # A recovered inbox never requeues input from an interrupted process.
        interrupted = {
            event.message_id for event in self._messages.values() if event.state == "accepted"
        }
        for tag, event in agent.event_manager.items():
            if (
                isinstance(event, Notification)
                and event.metadata.get("steering_message_id") in interrupted
            ):
                # A crash can fall between storing context and its applied
                # record. Keep the accepted metadata, not unacknowledged input
                # in a later turn's model context.
                agent.event_manager.remove(tag)
        for event in tuple(self._messages.values()):
            if event.state == "accepted":
                self._record(
                    event, "interrupted", "The original turn was interrupted.", notify=False
                )
        self._unsubscribe = agent.event_manager.intercept("llm_call", self._include_input)

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def begin(self, turn_id: str) -> None:
        if self.turn_id is not None:
            raise ValueError("A prompt is already running")
        if turn_id in self._turns:
            raise ValueError("turnId has already been used; start a new prompt with a new turnId")
        self._turns.add(turn_id)
        self.turn_id = turn_id

    def accept(self, turn_id: str, message_id: str, text: str) -> dict[str, Any]:
        if existing := self._messages.get(message_id):
            if existing.turn_id != turn_id or existing.text != text:
                raise ValueError("messageId was already used with different input")
            return self._ack(existing)
        if self.turn_id != turn_id:
            raise ValueError("The original turn is no longer accepting steering")
        event = SessionSteering(message_id=message_id, turn_id=turn_id, text=text, state="accepted")
        self.handle.record_steering(event)
        self._messages[message_id] = event
        self._pending.append(message_id)
        self.wakeup.set()
        self.publish(steering_update(event))
        return self._ack(event)

    @staticmethod
    def _ack(event: SessionSteering) -> dict[str, Any]:
        return {"messageId": event.message_id, "turnId": event.turn_id, "state": event.state}

    def _record(
        self,
        event: SessionSteering,
        state: SteeringState,
        reason: str | None = None,
        *,
        notify: bool = True,
    ) -> None:
        updated = SessionSteering(
            message_id=event.message_id,
            turn_id=event.turn_id,
            text=event.text,
            state=state,
            reason=reason,
        )
        self.handle.record_steering(updated)
        self._messages[event.message_id] = updated
        if notify:
            self.publish(steering_update(updated))

    def matches(self, runtime: RuntimeServices) -> bool:
        return (
            self.turn_id is not None
            and runtime.agent is self.agent
            and getattr(getattr(runtime, "_current_method", None), "__name__", "") == "handle"
            and runtime.get_parent_generation_id() is None
            and not summary_fork_active()
        )

    async def _include_input(self, ctx: LLMCallContext, nxt: LLMCallNext) -> LLMCallContext:
        if ctx.runtime is not None and current_steering(ctx.runtime) is self:
            # Message building may await dynamic context. Drain here so even
            # input received during that build is included in this exact call.
            for message_id in tuple(self._pending):
                event = self._messages[message_id]
                tag = self.agent.event_manager.add(
                    Notification(
                        source=_SOURCE,
                        description=event.text,
                        metadata={"steering_message_id": message_id, "turn_id": event.turn_id},
                    )
                )
                ctx.messages.append({"role": "user", "content": f"{_SOURCE}\n\n{event.text}"})
                try:
                    self._record(event, "applied")
                except BaseException:
                    # A failed durable transition must not leave an input in
                    # future context while its receipt still says unapplied.
                    ctx.messages.pop()
                    self.agent.event_manager.remove(tag)
                    raise
                self._pending.remove(message_id)
            self.wakeup.clear()
        return await nxt(ctx)

    def completed(self, result: Any) -> None:
        if isinstance(result, (Done, NeedInput)) or (
            isinstance(result, Waiting)
            and {name.strip() for name in result.on} == {"user_messages"}
        ):
            # Seal before CodeAct's asynchronous worker teardown. A late RPC
            # is rejected rather than accepted into a turn already committing.
            self.finish("turn_finished")

    def finish(self, state: SteeringState, reason: str | None = None) -> None:
        self.turn_id = None
        for message_id in tuple(self._pending):
            self._record(self._messages[message_id], state, reason)
            self._pending.remove(message_id)
        self.wakeup.clear()

    def close(self) -> None:
        self.finish("interrupted", "The session was closed.")
        self._unsubscribe()

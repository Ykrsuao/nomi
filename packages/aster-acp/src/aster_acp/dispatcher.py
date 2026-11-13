# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side dispatcher for a ASTER interactive agent."""

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import suppress
from contextvars import ContextVar
from typing import Any

from aster_cli.coding import CodingAgent, CodingSlashCommandRegistry
from aster_cli.coding.steering import SteeringControl, steering_scope
from pydantic import BaseModel

from aster.events import EventBase
from aster.interactive import AgentMessage, Done, NeedInput, Waiting
from aster.slash_dispatch import SlashCommandResult
from aster.strategies.base import GenerationStrategy

TurnResult = Done | NeedInput | Waiting

_current_handle: ContextVar[object | None] = ContextVar("aster_acp_handle", default=None)


class InteractiveSessionDispatcher:
    def __init__(self, agent: CodingAgent, *, strategy: GenerationStrategy | None = None) -> None:
        self.agent = agent
        self._strategy = strategy
        self._active_task: asyncio.Task[Any] | None = None
        self._cancel_requested = False
        self._cancelling = False
        self.steering: SteeringControl | None = None
        self.on_result: Callable[[TurnResult], None] | None = None
        self.on_turn_completed: Callable[[str | None], None] | None = None

    @property
    def active(self) -> bool:
        return self._cancelling or (self._active_task is not None and not self._active_task.done())

    async def submit(self, text: str) -> TurnResult | None:
        self._ensure_idle()
        self.agent.queue_manager.get_channel("user_messages").put(text)
        return await self._run_active(self._dispatch())

    async def invoke_slash(
        self,
        commands: CodingSlashCommandRegistry,
        name: str,
        raw_args: str,
    ) -> tuple[SlashCommandResult, TurnResult | None] | None:
        """Invoke and, when requested, dispatch a slash command as one cancellable turn."""

        async def _invoke() -> tuple[SlashCommandResult, TurnResult | None]:
            result = await commands.invoke(name, raw_args)
            if not result.output_to_agent:
                return result, None
            self.agent.queue_manager.get_channel("slash_commands").put(result)
            return result, await self._dispatch()

        return await self._run_active(_invoke())

    def _ensure_idle(self) -> None:
        if self.active:
            raise RuntimeError("A prompt is already running")

    async def _run_active(self, operation: Coroutine[Any, Any, Any]) -> Any:
        if self.active:
            operation.close()
            raise RuntimeError("A prompt is already running")

        self._cancel_requested = False
        with steering_scope(self.steering):
            task = asyncio.create_task(operation, name="aster-acp-dispatch")
        self._active_task = task
        try:
            return await task
        except asyncio.CancelledError:
            if self._cancel_requested:
                return None
            raise
        finally:
            if self._active_task is task:
                self._active_task = None

    async def _dispatch(self) -> TurnResult:
        while True:
            wins = await self._wait_for_input()
            notification: dict[str, list[Any]] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            for name, channel in self.agent.queue_manager.channels().items():
                if drained := channel.drain():
                    notification.setdefault(name, []).extend(drained)

            result, reply_id = await self._handle(notification)
            # ACP must finish this request before accepting the next user message.
            # Keep background waits open, including mixed user/background targets.
            if isinstance(result, Waiting) and set(result.on) != {"user_messages"}:
                continue
            # Complete synchronously before returning to the waiting ACP task:
            # a background job woken by the reply may otherwise change state
            # before the host has captured this foreground boundary.
            if self.on_turn_completed is not None:
                self.on_turn_completed(reply_id)
            return result

    async def _wait_for_input(self) -> list[tuple[str, Any]]:
        control = self.steering
        if control is None:
            return await self.agent.queue_manager.race()
        if control.pending:
            return []
        queued = asyncio.create_task(self.agent.queue_manager.race())
        steered = asyncio.create_task(control.wakeup.wait())
        try:
            await asyncio.wait((queued, steered), return_when=asyncio.FIRST_COMPLETED)
            if queued.done():
                return queued.result()
            # race() restores any unconsumed channel item when cancelled.
            return []
        finally:
            for task in (queued, steered):
                if not task.done():
                    task.cancel()
            await asyncio.gather(queued, steered, return_exceptions=True)

    async def _handle(self, notification: dict[str, list[Any]]) -> tuple[TurnResult, str | None]:
        last_message: AgentMessage | None = None
        last_public_message: AgentMessage | None = None
        invocation = object()

        def remember_message(event: EventBase) -> None:
            nonlocal last_message, last_public_message
            # Broker tasks inherit this context; background tasks from an
            # earlier handle keep theirs and cannot suppress this reply.
            if isinstance(event, AgentMessage):
                last_public_message = event
                if _current_handle.get() is invocation:
                    last_message = event

        unsubscribe = self.agent.event_manager.on("AgentMessage", remember_message)
        token = _current_handle.set(invocation)
        try:
            if self._strategy is None:
                result = await self.agent.handle(notification)
            else:
                result = await self.agent.handle(notification, _strategy=self._strategy)
            if self.steering is not None:
                self.steering.completed(result)
            if self.on_result is not None:
                self.on_result(result)
            self._show(
                result,
                last_message=last_message.content
                if last_message is not None and last_message is last_public_message
                else None,
            )
            reply_id = (
                last_message.id
                if last_message is not None and last_message is last_public_message
                else None
            )
            return result, reply_id
        finally:
            unsubscribe()
            _current_handle.reset(token)

    def _show(self, result: TurnResult, *, last_message: str | None = None) -> None:
        """Send the person the text a typed result carries, as an agent message.

        ``Done.message`` is the reply, ``Waiting.message`` the line shown while
        waiting, and a ``NeedInput`` shows its question, its reason, and its
        choices or the fields of its ``answer_type`` with their types and
        descriptions. This server has no forms, so a typed answer still
        arrives as the person's text reply. If the agent already sent the
        same complete text during this handle, keep that message once.
        """
        if isinstance(result, NeedInput):
            parts = [result.question]
            if result.reason:
                parts.append(result.reason)
            if result.options:
                parts.append("\n".join(f"- {option}" for option in result.options))
            if result.answer_type is not None:
                parts.append("Reply with these fields:\n" + _describe_fields(result.answer_type))
            text: str | None = "\n\n".join(parts)
        else:
            text = result.message
        if text and text != last_message:
            self.agent.message(text)

    async def cancel(self) -> bool:
        """Cancel the foreground turn and background jobs without closing the session."""
        task = self._active_task
        if task is None or task.done():
            return False

        self._cancelling = True
        self._cancel_requested = True
        try:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            for channel_name in ("user_messages", "slash_commands"):
                self.agent.queue_manager.get_channel(channel_name).flush()
            await self.agent.queue_manager.shutdown()
            return True
        finally:
            self._cancelling = False

    async def close(self) -> None:
        await self.cancel()
        await self.agent.close()


def _describe_fields(model: type[BaseModel]) -> str:
    """One ``- name (type): description`` line per field of ``model``."""
    lines = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        type_name = annotation.__name__ if isinstance(annotation, type) else str(annotation)
        line = f"- {name} ({type_name})"
        if field.description:
            line += f": {field.description}"
        lines.append(line)
    return "\n".join(lines)

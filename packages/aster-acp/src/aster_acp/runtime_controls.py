# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-session model selection and an approval gate for generated execution.

The gate is deliberately at the Python execution seam. A shell-only check
cannot control CodeAct, whose generated Python can open files and processes
directly. Planning disables generated execution entirely; it is not a claim
that the host Python interpreter has become a read-only sandbox.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from acp import text_block, tool_content
from acp.interfaces import Client
from acp.schema import (
    CurrentModeUpdate,
    PermissionOption,
    SessionMode,
    SessionModeState,
    ToolCallUpdate,
)

from aster.interactive import apply_model_limits
from aster_acp.approval_preview import approval_preview

if TYPE_CHECKING:
    from aster_cli.coding.agent import CodingAgent

    from aster.unifiedllm import UnifiedLLM

_MAX_APPROVAL_CODE = 128 * 1024
_MODES = (
    SessionMode(
        id="code",
        name="Code",
        description="Use tools and Python with the selected approval policy.",
    ),
    SessionMode(
        id="plan",
        name="Plan (no execution)",
        description="Reason from supplied context; generated code and tool execution are disabled.",
    ),
)


def configured_models() -> list[str]:
    """List chat aliases without exposing their routes, credentials or settings."""
    from aster.unifiedllm.registry import MODELS, ensure_loaded

    ensure_loaded()
    return sorted(
        name for name, config in MODELS.items() if config.get("client_type") != "decision"
    )


def _load_model(name: str) -> UnifiedLLM:
    from aster.unifiedllm import get_llm_client

    return get_llm_client(name)


class RuntimeControls:
    """Settings owned by one live ACP session, never restored as blanket consent.

    Hosts must hold the session's foreground-turn lock when calling configure.
    Attach this object as ``agent._execution_control``; CodingReplyStrategy
    consults it before any model-authored execution, native or sandbox.
    """

    def __init__(
        self,
        agent: CodingAgent,
        client: Client,
        session_id: str,
        *,
        publish: Callable[[Any], None],
        execution_backend: str = "native",
        model_factory: Callable[[str], UnifiedLLM] | None = None,
        model_names: Callable[[], Sequence[str]] = configured_models,
        waiting: Callable[[str], None] | None = None,
    ) -> None:
        self.agent = agent
        self.client = client
        self.session_id = session_id
        self.execution_backend = execution_backend
        self._publish = publish
        self._model_factory = model_factory or _load_model
        self._model_names = model_names
        self._waiting = waiting
        self._model_id = agent.llm.model
        self.mode = "code"
        self.approval = "ask"
        self._pending = 0
        self._approval_lock = asyncio.Lock()

    def instructions(self) -> str:
        if self.mode == "plan":
            return (
                "The user selected Plan mode. Generated Python and tool execution are disabled. "
                "Reason from the conversation and supplied file/image context. Explain any missing "
                "information and present the plan with the separate return_result JSON tool. "
                "Do not claim to have inspected or changed files that were not supplied."
            )
        if self.approval == "session":
            return "The user has allowed generated execution for this live session."
        return (
            "Generated Python requires approval before execution. Keep each cell focused and "
            "reviewable. A rejected cell did not run; do not retry it or bypass the user's choice."
        )

    def mode_state(self) -> SessionModeState:
        return SessionModeState(current_mode_id=self.mode, available_modes=list(_MODES))

    def status(self) -> dict[str, Any]:
        names = list(dict.fromkeys([self._model_id, *self._model_names()]))
        return {
            "model": self._model_id,
            "modelName": self.agent.llm.model,
            "models": [{"id": name, "name": name} for name in names],
            "mode": self.mode,
            "modes": [mode.model_dump(by_alias=True, exclude_none=True) for mode in _MODES],
            "approval": self.approval,
            "executionBackend": self.execution_backend,
            "executionEnabled": self.mode != "plan",
            "approvalPending": self._pending > 0,
        }

    def publish(self) -> None:
        self._publish(
            CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=self.mode,
                field_meta={"aster": {"runtime": self.status()}},
            )
        )

    async def configure(
        self,
        *,
        model: str | None = None,
        mode: str | None = None,
        approval: str | None = None,
    ) -> dict[str, Any]:
        """Validate before changing settings; the host serializes this with prompts."""
        if mode is not None and mode not in {"code", "plan"}:
            raise ValueError("Mode must be 'code' or 'plan'.")
        if approval is not None and approval not in {"ask", "session"}:
            raise ValueError("Approval must be 'ask' or 'session'.")
        if (mode or self.mode) == "plan" and approval == "session":
            raise ValueError("Switch to Code mode before allowing execution for the session.")
        if model is not None and model not in {self._model_id, *self._model_names()}:
            raise ValueError("Choose one of the configured models shown in this session.")
        if self._pending:
            raise ValueError("Resolve or cancel the pending approval before changing settings.")
        if self.agent.queue_manager.running_handles():
            raise ValueError("Stop or finish background jobs before changing runtime settings.")

        previous = None
        if model is not None and model != self._model_id:
            previous = self.agent.llm
            candidate = self._model_factory(model)
            try:
                await candidate.refresh_model_info()
                self.agent.set_llm(candidate)
                apply_model_limits(self.agent)
            except BaseException:
                self.agent.set_llm(previous)
                apply_model_limits(self.agent)
                await candidate.aclose()
                raise
            self._model_id = model
        if mode is not None:
            self.mode = mode
            if mode == "plan":
                self.approval = "ask"
        if approval is not None:
            self.approval = approval
        # Publish all effective settings before asynchronous cleanup. If that
        # await is cancelled, the UI still sees the configuration in use.
        self.publish()
        if previous is not None:
            # Cleanup failure cannot undo a model already committed above.
            try:
                await previous.aclose()
            except Exception:
                pass
        return self.status()

    async def authorize(self, code: str, *, tool_call_id: str | None = None) -> str | None:
        """Return a refusal reason or None; cancellation always propagates.

        Permission applies to the complete cell shown in the request. A session
        grant is scoped to this RuntimeControls instance and disappears when
        the session is closed/reconnected. Unknown client choices fail closed.
        """
        if self.mode == "plan":
            return (
                "Plan mode disables generated execution. This code was not run. "
                "Use the separate return_result JSON tool to present your plan."
            )
        if self.approval == "session":
            return None
        if len(code.encode("utf-8")) > _MAX_APPROVAL_CODE:
            return "This cell is too large to review. Split it into smaller, focused cells."
        async with self._approval_lock:
            # A concurrent cell can have obtained a live-session grant while
            # this one waited for the preceding dialog.
            if self.approval == "session":
                return None
            self._pending += 1
            self.publish()
            if self._waiting:
                self._waiting("Waiting for execution approval")
            try:
                preview = approval_preview(code)
                response = await self.client.request_permission(
                    self.session_id,
                    ToolCallUpdate(
                        tool_call_id=tool_call_id or f"approval-{uuid4().hex}",
                        title=f"{preview['summary']}?",
                        kind="other",
                        status="pending",
                        content=[tool_content(text_block(code))],
                        raw_input={"code": code},
                        field_meta={
                            "aster": {
                                "approvalScope": "python_execution",
                                "executionBackend": self.execution_backend,
                                "approvalPreview": preview,
                            }
                        },
                    ),
                    [
                        PermissionOption(
                            option_id="allow_once", name="Allow once", kind="allow_once"
                        ),
                        PermissionOption(
                            option_id="allow_session",
                            name="Allow all code for this session",
                            kind="allow_always",
                        ),
                        PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
                    ],
                )
                outcome = response.outcome
                option = getattr(outcome, "option_id", None)
                if getattr(outcome, "outcome", None) == "selected":
                    if option == "allow_once":
                        return None
                    if option == "allow_session":
                        self.approval = "session"
                        return None
                return "The user rejected this execution. The code was not run."
            except Exception:
                return "Execution approval could not be obtained. The code was not run."
            finally:
                self._pending -= 1
                self.publish()
                if self._waiting:
                    self._waiting("Processing response")

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep explicit messages and typed final replies from sending the same body twice."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_cli.coding import CodingAgent

from nooa.config.strategy_config import CodeActConfig
from nooa.errors import GenerationError
from nooa.interactive import AgentMessage, Done, NeedInput, SummarizationConfig
from nooa.skill_registry import SkillRegistry
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TIMEOUT = 15


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Optional installed skills are unrelated to reply delivery.
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    for name in (
        "NEMO_OO_SETTINGS",
        "NEMO_OO_RICH_URL",
        "NOOA_PROMPT_TIMING",
        "NOOA_STARTUP_TIMING",
    ):
        monkeypatch.delenv(name, raising=False)


def _scripted_llm(codes: tuple[str, ...]) -> FakeLLMClient:
    responses = [
        LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"message-test-{index}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
            finish_reason="tool_calls",
            reasoning=None,
            usage=None,
        )
        for index, code in enumerate(codes)
    ]
    return FakeLLMClient(responses, strict_exhaustion=True)


@dataclass
class _Session:
    agent: CodingAgent
    dispatcher: InteractiveSessionDispatcher
    llm: FakeLLMClient

    def messages(self) -> list[str]:
        return [
            event.content
            for event in self.agent.event_manager.values()
            if isinstance(event, AgentMessage)
        ]


@asynccontextmanager
async def _session(tmp_path: Path, *codes: str) -> AsyncIterator[_Session]:
    llm = _scripted_llm(codes)
    agent = CodingAgent(
        llm=llm,
        cwd=tmp_path,
        libs_dir=tmp_path / "libs",
        summarization=SummarizationConfig(policy="none"),
    )
    strategy = CodeActStrategy(
        config=CodeActConfig(execution_backend="inprocess", prefill=None, max_retries=1)
    )
    dispatcher = InteractiveSessionDispatcher(agent, strategy=strategy)
    try:
        async with asyncio.timeout(_TIMEOUT):
            yield _Session(agent, dispatcher, llm)
    finally:
        async with asyncio.timeout(_TIMEOUT):
            await dispatcher.close()


async def test_explicit_message_and_identical_done_body_are_sent_once(tmp_path):
    text = "NOOA-VERIFY-1 | sum=71 | system=Linux"
    code = (
        f"text = {text!r}\n"
        "self.message(text)\n"
        "return_result(Done(message=text, explanation='complete'))"
    )
    async with _session(tmp_path, code) as session:
        result = await session.dispatcher.submit("Return the synthetic verification result")

        assert isinstance(result, Done)
        assert result.message == text
        assert session.llm.call_count == 1
        assert session.messages() == [text]


async def test_same_text_from_an_earlier_turn_does_not_suppress_this_turn(tmp_path):
    text = "The requested check passed."
    explicit = f"self.message({text!r})\nreturn_result(Done(explanation='complete'))"
    final_only = f"return_result(Done(message={text!r}, explanation='complete'))"
    async with _session(tmp_path, explicit, final_only) as session:
        await session.dispatcher.submit("First check")
        assert session.messages() == [text]

        result = await session.dispatcher.submit("Repeat the check")

        assert isinstance(result, Done)
        assert session.messages() == [text, text]


async def test_different_progress_and_final_messages_are_both_kept(tmp_path):
    progress = "Checking the synthetic input."
    final = "Verification complete."
    code = (
        f"self.message({progress!r})\n"
        f"return_result(Done(message={final!r}, explanation='complete'))"
    )
    async with _session(tmp_path, code) as session:
        await session.dispatcher.submit("Run the check")

        assert session.messages() == [progress, final]


@pytest.mark.parametrize("explicit_wait_message", [False, True])
async def test_waiting_and_next_handle_can_each_send_the_same_body(tmp_path, explicit_wait_message):
    text = "The synthetic check completed."
    explicit = f"self.message({text!r})\n" if explicit_wait_message else ""
    waiting = (
        explicit
        + "self.queue_manager.get_channel('system_messages').put('continue')\n"
        + f"return_result(Waiting(message={text!r}, explanation='wait for continuation', "
        + "on=['system_messages']))"
    )
    final = f"return_result(Done(message={text!r}, explanation='complete'))"
    async with _session(tmp_path, waiting, final) as session:
        result = await session.dispatcher.submit("Wait for a synthetic continuation")

        assert isinstance(result, Done)
        assert session.llm.call_count == 2
        assert session.messages() == [text, text]


@pytest.mark.parametrize(
    ("setup", "answer_argument", "formatted_answer"),
    [
        ("", "options=['patch', 'minor']", "- patch\n- minor"),
        (
            "from pydantic import BaseModel, Field\n"
            "class ReleaseAnswer(BaseModel):\n"
            "    version: str = Field(description='The version number')\n"
            "    notes: str\n",
            "answer_type=ReleaseAnswer",
            "Reply with these fields:\n- version (str): The version number\n- notes (str)",
        ),
    ],
    ids=["options", "fields"],
)
async def test_existing_question_does_not_suppress_need_input_details(
    tmp_path, setup, answer_argument, formatted_answer
):
    question = "Which release?"
    reason = "The version controls the changelog."
    code = (
        setup
        + f"self.message({question!r})\n"
        + f"return_result(NeedInput(question={question!r}, reason={reason!r}, "
        + answer_argument
        + "))"
    )
    async with _session(tmp_path, code) as session:
        result = await session.dispatcher.submit("Ask for a synthetic release choice")

        assert isinstance(result, NeedInput)
        assert session.messages() == [question, f"{question}\n\n{reason}\n\n{formatted_answer}"]


async def test_previous_handles_background_message_does_not_suppress_next_final_reply(tmp_path):
    text = "An independently completed check."
    background = (
        "async def publish_later(agent=self):\n"
        "    await agent._message_test_release.wait()\n"
        f"    agent.message({text!r})\n"
        "    agent._message_test_sent.set()\n"
        "self._message_test_task = asyncio.create_task(publish_later())\n"
        "return_result(Done(explanation='background message scheduled'))"
    )
    final = (
        "self._message_test_release.set()\n"
        "await self._message_test_sent.wait()\n"
        f"return_result(Done(message={text!r}, explanation='foreground check complete'))"
    )
    async with _session(tmp_path, background, final) as session:
        session.agent._message_test_release = asyncio.Event()
        session.agent._message_test_sent = asyncio.Event()
        try:
            first = await session.dispatcher.submit("Schedule a synthetic background message")
            assert isinstance(first, Done)
            assert session.messages() == []

            second = await session.dispatcher.submit("Finish an independent foreground check")

            assert isinstance(second, Done)
            assert session.llm.call_count == 2
            assert session.messages() == [text, text]
        finally:
            task = getattr(session.agent, "_message_test_task", None)
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_two_explicit_identical_messages_are_kept_without_a_third_final_copy(tmp_path):
    text = "An intentionally repeated synthetic reply."
    code = (
        f"self.message({text!r})\n"
        f"self.message({text!r})\n"
        f"return_result(Done(message={text!r}, explanation='complete'))"
    )
    async with _session(tmp_path, code) as session:
        await session.dispatcher.submit("Repeat the synthetic reply explicitly")

        assert session.messages() == [text, text]


async def test_earlier_same_body_does_not_suppress_a_final_after_different_progress(tmp_path):
    text = "The synthetic check passed."
    progress = "Repeating the synthetic check."
    code = (
        f"self.message({text!r})\n"
        f"self.message({progress!r})\n"
        f"return_result(Done(message={text!r}, explanation='complete'))"
    )
    async with _session(tmp_path, code) as session:
        await session.dispatcher.submit("Check the synthetic input twice")

        assert session.messages() == [text, progress, text]


def _track_message_subscriptions(monkeypatch, agent: CodingAgent) -> set[object]:
    """Observe the public subscribe/unsubscribe contract without private handler state."""
    original_on = agent.event_manager.on
    active: set[object] = set()

    def on(event_type, handler):
        unsubscribe = original_on(event_type, handler)
        if event_type != "AgentMessage":
            return unsubscribe
        token = object()
        active.add(token)

        def remove():
            unsubscribe()
            active.discard(token)

        return remove

    monkeypatch.setattr(agent.event_manager, "on", on)
    return active


@pytest.mark.parametrize("interruption", ["cancel", "failure"])
async def test_interrupted_turn_releases_message_observers_and_next_reply_survives(
    tmp_path, monkeypatch, interruption
):
    text = "A reusable synthetic reply."
    first_code = f"self.message({text!r})"
    recovery_code = f"return_result(Done(message={text!r}, explanation='complete'))"
    async with _session(tmp_path, first_code, recovery_code) as session:
        active = _track_message_subscriptions(monkeypatch, session.agent)
        entered = asyncio.Event()
        never = asyncio.Event()
        original_acall = session.llm.acall
        calls = 0

        async def acall(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # The first real CodeAct cell has already emitted its message.
                # Interrupt its next model request, leaving the recovery response
                # unconsumed for the next dispatcher turn.
                entered.set()
                if interruption == "failure":
                    raise RuntimeError("synthetic provider failure")
                await never.wait()
            return await original_acall(*args, **kwargs)

        monkeypatch.setattr(session.llm, "acall", acall)
        first = asyncio.create_task(session.dispatcher.submit("Interrupt this turn"))
        try:
            await entered.wait()
            if interruption == "cancel":
                assert await session.dispatcher.cancel() is True
                assert await first is None
            else:
                with pytest.raises(GenerationError, match="synthetic provider failure"):
                    await first

            assert active == set()
            assert session.messages() == [text]

            result = await session.dispatcher.submit("Recover with the same reply")

            assert isinstance(result, Done)
            assert session.messages() == [text, text]
            assert active == set()
        finally:
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)

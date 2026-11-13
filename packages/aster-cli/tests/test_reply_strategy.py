# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coding reply policy rejects definite reply literals before Python execution."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aster_cli.coding.reply_strategy import CodingReplyStrategy

from aster import Agent, strategy
from aster.config import CodeActConfig
from aster.events import ExecutionResult, PythonOutput
from aster.interactive import Done, NeedInput, Waiting
from aster.strategies.codeact import CodeActStrategy
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def reply_call(name, arguments, identifier="reply"):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCall(id=identifier, name=name, arguments=json.dumps(arguments))],
    )


class ReplyAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.touched = False

    @strategy(CodingReplyStrategy(config=CodeActConfig(max_iterations=4)))
    async def handle(self) -> Done | NeedInput | Waiting:
        """Answer the user."""
        ...


async def test_literal_final_reply_is_corrected_before_side_effects_execute():
    llm = FakeLLMClient(
        scripted_responses=[
            reply_call(
                "execute_python",
                {
                    "code": "self.touched = True\nreturn_result(Done(message='buffered', explanation='done'))"
                },
                "wrong-path",
            ),
            reply_call(
                "return_result", {"result": {"message": "streamable", "explanation": "done"}}
            ),
        ]
    )
    agent = ReplyAgent(llm=llm)
    try:
        result = await agent.handle()
        assert result.message == "streamable"
        assert not agent.touched, "Rejected reply cells must not run their earlier side effects"
        errors = [
            e for e in agent.event_manager.values() if isinstance(e, PythonOutput) and e.error
        ]
        assert len(errors) == 1
        assert "separate return_result tool" in str(errors[0].error)
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "code",
    [
        "return_result(Done(message='answer', explanation='done'))",
        "return_result(Waiting(message='waiting', explanation='pending', on=['jobs']))",
        "return_result({'message': 'answer', 'explanation': 'done'})",
        "return_result(result={'message': 'answer', 'explanation': 'done'})",
        "return_result(message='answer', explanation='done')",
        "return Done(message='answer', explanation='done')",
        "text = 'answer'\nreturn_result(Done(message=text, explanation='done'))",
        "text = 'answer'\nreply = Done(message=text, explanation='done')\nreturn_result(reply)",
        "self.message('answer')\nreturn_result(Done(explanation='done'))",
        "self.message('answer')\nreturn_result(Done(message=None, explanation='done'))",
        "self.message('answer')\nreturn_result(Done(message='  ', explanation='done'))",
        "self.message(text='answer')\nreturn_result({'explanation': 'done'})",
        "return_result(Done(message='answer', explanation='done', evidence=checks))",
        "return_result(Waiting(message='waiting', explanation='pending', on=jobs))",
        "return_result({'message': 'answer', 'explanation': 'done', 'result': computed})",
        "return_result(message='answer', explanation='done', evidence=checks)",
    ],
)
async def test_definite_literal_replies_never_reach_executor(monkeypatch, code):
    execute = AsyncMock(return_value=ExecutionResult())
    monkeypatch.setattr(CodeActStrategy, "_execute_code", execute)
    runtime = SimpleNamespace(agent=SimpleNamespace())
    result = await CodingReplyStrategy()._execute_code(runtime, code, {}, None, "handle")
    assert result.error is not None
    assert "separate return_result tool" in str(result.error)
    execute.assert_not_awaited()


@pytest.mark.parametrize(
    "code",
    [
        "return_result(computed_result)",
        "return_result(Done(message=calculated_text, explanation='done'))",
        "answer = str(sum([2, 3]))\nreturn_result(Done(message=answer, explanation='done'))",
        "return_result(Done(message=f'{count} files', explanation='done'))",
        "self.message('Checking files')",
        "self.message('Sending computed report')\nreturn_result(Done(message=computed_report, explanation='done'))",
        "self.message('Sending computed report')\nreturn_result({'message': computed_report, 'explanation': 'done'})",
        "self.message('Sending computed report')\nreturn_result(message=computed_report, explanation='done')",
        "self.message('Checking files')\nanswer = await self.work()\nreturn_result(answer)",
        "return_result(NeedInput(question='Which folder?', answer_type=Answer))",
        "if ready:\n    return_result(Done(message='ready', explanation='done'))",
        "def respond():\n    return_result(Done(message='later', explanation='done'))",
        "text = 'stale'\ntext = await self.work()\nreturn_result(Done(message=text, explanation='done'))",
        "text = 'stale'\nif condition:\n    text = computed\nreturn_result(Done(message=text, explanation='done'))",
        "text = 'stale'\nother = await self.work()\nreturn_result(Done(message=text, explanation='done'))",
        "return_result(Done(message='answer', **unknown))",
        "return_result({'message': 'answer', **unknown})",
        "return_result(",
    ],
)
async def test_computation_progress_and_uncertain_source_keep_normal_execution(monkeypatch, code):
    expected = ExecutionResult(stdout="computed")
    execute = AsyncMock(return_value=expected)
    monkeypatch.setattr(CodeActStrategy, "_execute_code", execute)
    runtime = SimpleNamespace(agent=SimpleNamespace())
    result = await CodingReplyStrategy()._execute_code(runtime, code, {}, None, "handle")
    assert result is expected
    execute.assert_awaited_once()


@pytest.mark.parametrize(
    ("method", "metadata"),
    [("other_task", None), ("handle", {"prefill": True})],
)
async def test_other_methods_and_prefill_keep_generic_codeact_policy(monkeypatch, method, metadata):
    execute = AsyncMock(return_value=ExecutionResult())
    monkeypatch.setattr(CodeActStrategy, "_execute_code", execute)
    await CodingReplyStrategy()._execute_code(
        SimpleNamespace(agent=SimpleNamespace()),
        "return_result(Done(message='answer', explanation='done'))",
        {},
        None,
        method,
        output_metadata=metadata,
    )
    execute.assert_awaited_once()


@pytest.mark.parametrize("inline", [False, True])
async def test_correct_reply_and_computed_value_finish_without_rewriting(inline):
    response = (
        reply_call(
            "execute_python",
            {
                "code": "text = str(sum([2, 3]))\nreturn_result(Done(message=text, explanation='done'))"
            },
        )
        if inline
        else reply_call("return_result", {"result": {"message": "five", "explanation": "done"}})
    )
    agent = ReplyAgent(llm=FakeLLMClient(scripted_responses=[response]))
    try:
        result = await agent.handle()
        assert result.message == ("5" if inline else "five")
        assert len([e for e in agent.event_manager.values() if e.event_type == "LLMResponse"]) == 1
    finally:
        await agent.aclose()


async def test_coding_instructions_and_tool_description_require_reply_channel():
    policy = CodingReplyStrategy()
    instructions = await policy.strategy_instructions(
        SimpleNamespace(_current_method=SimpleNamespace(__name__="handle"))
    )
    assert "separate return_result tool" in instructions
    assert "ordinary" in instructions
    assert "computed" in instructions
    assert "prefer calling return_result(variable)" not in instructions
    tool = policy._build_return_result_tool(Done | NeedInput | Waiting, "handle")
    assert "separate return_result tool" in tool.description
    assert "prefer calling return_result(variable)" not in tool.description
    generic = policy._build_return_result_tool(int, "calculate")
    assert "prefer calling return_result(variable)" in generic.description


async def test_delegated_method_keeps_generic_strategy_instructions(monkeypatch):
    generic = AsyncMock(return_value="generic instructions")
    monkeypatch.setattr(CodeActStrategy, "_strategy_instructions", generic)
    runtime = SimpleNamespace(_current_method=SimpleNamespace(__name__="calculate"))
    assert await CodingReplyStrategy().strategy_instructions(runtime) == "generic instructions"
    generic.assert_awaited_once_with(runtime, restrictions=CodeActStrategy._restrictions_text())

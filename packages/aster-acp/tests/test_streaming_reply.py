# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User replies arrive during the provider request, with no internal text leakage."""

import asyncio
import json
from time import perf_counter
from types import SimpleNamespace

import pytest
from acp.schema import AgentMessageChunk
from aster_acp._streaming_reply import ReplyStream
from aster_acp.dispatcher import InteractiveSessionDispatcher
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import CodingAgent

from aster.interactive import AgentMessage, SummarizationConfig, install_summarizer
from aster.unifiedllm import CompletionClient, FakeLLMClient, ResponsesClient, RetryConfig
from aster.unifiedllm.streaming import emit_stream, observe_stream


def chat(arguments, *, name="", index=0):
    return {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": "PRIVATE_REASONING",
                    "content": "PRIVATE_CODEACT_PROSE",
                    "tool_calls": [
                        {"index": index, "function": {"name": name, "arguments": arguments}}
                    ],
                },
            }
        ]
    }


@pytest.mark.parametrize("step", [1, 2, 7, 30])
def test_only_explicit_reply_is_decoded_across_json_escape_and_unicode_boundaries(step):
    seen = []
    stream = ReplyStream(lambda *item: seen.append(item))
    text = '中文\nemoji 😀 and "quoted" \\ end'
    args = json.dumps({"result": {"message": text, "explanation": "PRIVATE_STATUS"}})
    stream.observe(
        "chat", chat('{"code":"self.message("PRIVATE_CODE")"}', name="execute_python", index=0)
    )
    stream.observe("chat", chat("", name="return_", index=1))
    assert not seen
    stream.observe("chat", chat("", name="result", index=1))
    for start in range(0, len(args), step):
        stream.observe("chat", chat(args[start : start + step], index=1))
    assert "".join(item[1] for item in seen) == text
    assert len({item[0] for item in seen}) == 1
    stream.finish(
        SimpleNamespace(
            finish_reason="tool_calls",
            tool_calls=[SimpleNamespace(name="return_result", arguments=args)],
        )
    )
    assert stream.accept(text) == seen[0][0]
    stream.interrupt()
    assert all(item[2] == "delta" for item in seen)


@pytest.mark.parametrize("finish", ["length", "error", "stop"])
def test_incomplete_provider_result_never_commits_preview(finish):
    seen = []
    stream = ReplyStream(lambda *item: seen.append(item))
    stream.observe("chat", chat('{"result":{"message":"partial', name="return_result"))
    stream.finish(SimpleNamespace(finish_reason=finish, tool_calls=[]))
    assert seen[-1][2] == "interrupted"
    assert stream.accept("partial") is None


def test_retry_resets_message_identity_and_invalid_json_cannot_commit():
    seen = []
    stream = ReplyStream(lambda *item: seen.append(item))
    stream.observe("chat", chat('{"result":{"message":"first', name="return_result"))
    first_id = seen[-1][0]
    stream.observe("reset", None)
    assert seen[-1][2] == "interrupted"
    stream.observe("chat", chat('{"result":{"message":"second', name="return_result"))
    assert seen[-1][0] != first_id
    stream.finish(SimpleNamespace(finish_reason="tool_calls", tool_calls=[]))
    assert seen[-1][2] == "interrupted"
    assert stream.accept("second") is None


@pytest.mark.parametrize(
    ("name", "arguments", "index"),
    [
        (7, "", 0),
        ("return_result", {"message": "bad"}, 0),
        ("return_result", "", []),
        ("return_result", "", 32),
    ],
)
def test_malformed_tool_fragments_do_not_break_observation(name, arguments, index):
    seen = []
    stream = ReplyStream(lambda *item: seen.append(item))
    stream.observe("chat", chat(arguments, name=name, index=index))
    assert not seen


def test_duplicate_message_key_interrupts_preview_without_committing_rewritten_text():
    seen = []
    stream = ReplyStream(lambda *item: seen.append(item))
    first = '{"result":{"message":"first"'
    rest = ',"message":"second","explanation":"done"}}'
    stream.observe("chat", chat(first, name="return_result"))
    stream.observe("chat", chat(rest))
    stream.finish(
        SimpleNamespace(
            finish_reason="tool_calls",
            tool_calls=[SimpleNamespace(name="return_result", arguments=first + rest)],
        )
    )
    assert [item[2] for item in seen] == ["delta", "interrupted"]
    assert stream.accept("second") is None


async def test_cancelled_scope_disables_inherited_provider_callbacks():
    seen = []
    release = asyncio.Event()

    async def late():
        await release.wait()
        emit_stream("chat", "late private value")

    with observe_stream(lambda *item: seen.append(item)):
        task = asyncio.create_task(late())
        emit_stream("reset")
    release.set()
    await task
    assert seen == [("reset", None)]


class RecordingClient:
    def __init__(self):
        self.updates = []
        self.first_reply = asyncio.Event()

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)
        if isinstance(update, AgentMessageChunk) and update.content.text:
            self.first_reply.set()


@pytest.mark.parametrize("api", ["completion", "responses"])
@pytest.mark.parametrize(
    "mode", ["complete", "retry", "invalid", "cancel", "legacy", "inline", "inline_cancel"]
)
async def test_real_provider_collection_exposes_reply_before_completion_and_commits_once(
    tmp_path, monkeypatch, api, mode
):
    import litellm
    from litellm.types.utils import ModelResponseStream

    release = asyncio.Event()
    entered = asyncio.Event()
    text = "First 中文\nThen complete."
    prefix = '{"result":{"message":"First '
    args = json.dumps(
        {"result": {"message": text, "explanation": "done"}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert args.startswith(prefix)
    requests = []

    class ChatStream:
        def __init__(self, attempt, arguments, name="return_result"):
            self.attempt = attempt
            self.arguments = arguments
            self.name = name

        async def __aiter__(self):
            yield ModelResponseStream(
                id="reply1",
                model="openai/test",
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "return1",
                                    "type": "function",
                                    "function": {
                                        "name": self.name,
                                        "arguments": prefix if self.name == "return_result" else "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            )
            if self.name == "return_result":
                await release.wait()
            if mode == "retry" and self.attempt == 1:
                raise litellm.BadGatewayError("temporary", model="test", llm_provider="openai")
            yield ModelResponseStream(
                id="reply1",
                model="openai/test",
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": self.arguments[len(prefix) :]
                                        if self.name == "return_result"
                                        else self.arguments,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            )

    async def responses(attempt, arguments, name="return_result"):
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": name, "arguments": ""},
        }
        opening = prefix if name == "return_result" else ""
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": opening,
        }
        if name == "return_result":
            await release.wait()
        if mode == "retry" and attempt == 1:
            raise litellm.BadGatewayError("temporary", model="test", llm_provider="openai")
        yield {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": arguments[len(opening) :],
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp1",
                "created_at": 1,
                "model": "test",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc1",
                        "call_id": "return1",
                        "name": name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
                "usage": None,
            },
        }

    async def provider(**kwargs):
        requests.append(kwargs)
        entered.set()
        if mode == "legacy":
            assert "stream" not in kwargs
        else:
            assert kwargs["stream"] is True
        arguments = args
        name = "return_result"
        if mode in {"inline", "inline_cancel"} and len(requests) == 1:
            name = "execute_python"
            arguments = json.dumps(
                {
                    "code": "await self.shell.write_file('should-not-run.txt', 'side effect')\n"
                    "return_result(Done(message='UNSENT_INLINE_REPLY', explanation='done'))"
                }
            )
        if mode == "invalid" and len(requests) == 1:
            arguments = args.replace(',"explanation":"done"', "")
        return (
            ChatStream(len(requests), arguments, name)
            if api == "completion"
            else responses(len(requests), arguments, name)
        )

    monkeypatch.setattr(litellm, "CustomStreamWrapper", ChatStream)
    monkeypatch.setattr(litellm, "acompletion" if api == "completion" else "aresponses", provider)
    cls = CompletionClient if api == "completion" else ResponsesClient
    llm = cls(
        model="openai/test",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=1, base_delay=0.001, max_delay=0.001),
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "test-session", stream_messages=mode != "legacy")
    dispatcher = InteractiveSessionDispatcher(agent)
    persisted = []
    agent.event_manager.on("AgentMessage", persisted.append)

    async def run():
        async with bridge.prompt(perf_counter(), "native"):
            return await dispatcher.submit("Answer the synthetic request")

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(
            entered.wait() if mode == "legacy" else client.first_reply.wait(), 10
        )
        assert not task.done(), "provider completed before the first reply was delivered"
        partial = [u for u in client.updates if isinstance(u, AgentMessageChunk)]
        assert [u.content.text for u in partial] == ([] if mode == "legacy" else ["First "])
        assert not persisted
        if mode in {"inline", "inline_cancel"}:
            assert len(requests) == 2, "literal Python answer was not redirected to the reply tool"
            assert not (tmp_path / "should-not-run.txt").exists()
        if mode in {"cancel", "inline_cancel"}:
            assert await dispatcher.cancel()
            assert await asyncio.wait_for(task, 10) is None
            partial = [u for u in client.updates if isinstance(u, AgentMessageChunk)]
            assert [u.field_meta["aster"]["messageState"] for u in partial] == [
                "delta",
                "interrupted",
            ]
            release.set()
            await asyncio.sleep(0.1)
            await bridge.flush()
            assert [u for u in client.updates if isinstance(u, AgentMessageChunk)] == partial
            assert not persisted
            return
        release.set()
        result = await asyncio.wait_for(task, 10)
        assert result.message == text
        messages = [u for u in client.updates if isinstance(u, AgentMessageChunk)]
        assert messages[-1].content.text == text
        metas = [u.field_meta["aster"] for u in messages]
        final_id = metas[-1]["messageId"]
        if mode == "legacy":
            assert len(messages) == 1
            assert "stream_message_id" not in persisted[0].metadata
        else:
            accepted_deltas = [
                u.content.text
                for u in messages
                if u.field_meta["aster"]["messageId"] == final_id
                and u.field_meta["aster"]["messageState"] == "delta"
            ]
            assert "".join(accepted_deltas) == text
            assert persisted[0].metadata["stream_message_id"] == final_id
            if mode in {"retry", "invalid"}:
                assert any(
                    m["messageState"] == "interrupted" and m["messageId"] != final_id for m in metas
                )
            else:
                assert {m["messageId"] for m in metas} == {final_id}
        assert metas[-1]["messageState"] == "complete" and metas[-1]["replace"] is True
        assert len(persisted) == 1 and isinstance(persisted[0], AgentMessage)
        assert len(requests) == (2 if mode in {"retry", "invalid", "inline"} else 1)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await bridge.close()
        await dispatcher.close()


@pytest.mark.parametrize("with_progress", [False, True])
async def test_computed_python_reply_is_delivered_without_another_model_request(
    tmp_path, with_progress
):
    (tmp_path / "computed.txt").write_text("computed result", encoding="utf-8")
    progress = "Read complete. Sending the computed result."
    code = "text = (await self.shell.read('computed.txt')).text\n"
    if with_progress:
        code += f"self.message({progress!r})\n"
    code += "return_result(Done(message=text, explanation='Read the saved result'))"
    llm = FakeLLMClient.with_tool_call(
        "execute_python",
        {"code": code},
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "computed-session", stream_messages=True)
    dispatcher = InteractiveSessionDispatcher(agent)
    try:
        async with bridge.prompt(perf_counter(), "native"):
            result = await dispatcher.submit("Return the computed file exactly")
        assert result.message == "computed result"
        assert llm.call_count == 1
        messages = [u for u in client.updates if isinstance(u, AgentMessageChunk)]
        assert [u.content.text for u in messages] == (
            [progress, result.message] if with_progress else [result.message]
        )
        assert messages[-1].field_meta["aster"]["messageState"] == "complete"
    finally:
        await bridge.close()
        await dispatcher.close()


@pytest.mark.parametrize("bridge_first", [False, True])
@pytest.mark.parametrize("summary_outcome", ["complete", "invalid", "failure"])
async def test_summary_fork_preserves_reply_identity_and_never_streams_private_content(
    tmp_path, monkeypatch, bridge_first, summary_outcome
):
    import litellm
    from litellm.types.utils import ModelResponseStream

    requests = []

    class ReplyProviderStream:
        def __init__(self, text):
            self.text = text

        async def __aiter__(self):
            arguments = json.dumps({"result": {"message": self.text, "explanation": "done"}})
            yield ModelResponseStream(
                id="reply",
                model="openai/test",
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "return1",
                                    "type": "function",
                                    "function": {"name": "return_result", "arguments": arguments},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            )
            yield ModelResponseStream(
                id="reply",
                model="openai/test",
                choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                usage={"prompt_tokens": 8000, "completion_tokens": 20, "total_tokens": 8020},
            )

    async def provider(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return ReplyProviderStream("VISIBLE_REPLY")
        assert len(requests) == 2
        assert "Background memory compaction" in kwargs["messages"][-1]["content"]
        if summary_outcome == "failure":
            raise litellm.BadRequestError("synthetic summary failure", "test", "openai")
        if summary_outcome == "invalid":
            # A summary that disobeys its instructions must not become a reply.
            return ReplyProviderStream("PRIVATE_SUMMARY")
        return litellm.ModelResponse(
            id="summary",
            model="openai/test",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "PRIVATE_SUMMARY"},
                    "finish_reason": "stop",
                }
            ],
        )

    monkeypatch.setattr(litellm, "CustomStreamWrapper", ReplyProviderStream)
    monkeypatch.setattr(litellm, "acompletion", provider)
    llm = CompletionClient(
        model="openai/test", api_key="synthetic", retry_config=RetryConfig(max_retries=0)
    )
    config = SummarizationConfig(max_tokens=1, preserve_recent=0)
    agent = CodingAgent(
        llm=llm,
        cwd=tmp_path,
        summarization=SummarizationConfig(policy="none") if bridge_first else config,
    )
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "test-session", stream_messages=True)
    if bridge_first:
        install_summarizer(config, agent)
    dispatcher = InteractiveSessionDispatcher(agent)
    persisted = []
    agent.event_manager.on("AgentMessage", persisted.append)
    try:
        async with bridge.prompt(perf_counter(), "native"):
            result = await dispatcher.submit("Answer the synthetic request")
        assert result.message == "VISIBLE_REPLY"
        assert len(requests) == 2
        assert bool(agent._summarizers[0]._pending_summary) is (summary_outcome == "complete")
        messages = [u for u in client.updates if isinstance(u, AgentMessageChunk)]
        assert [u.content.text for u in messages] == ["VISIBLE_REPLY", "VISIBLE_REPLY"]
        metadata = [u.field_meta["aster"] for u in messages]
        assert [m["messageState"] for m in metadata] == ["delta", "complete"]
        assert len({m["messageId"] for m in metadata}) == 1
        assert len(persisted) == 1
        assert persisted[0].metadata["stream_message_id"] == metadata[0]["messageId"]
    finally:
        await bridge.close()
        await dispatcher.close()

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable input snapshots reach both model context and durable history."""

import base64
import json

import pytest
from acp import text_block
from acp.schema import EmbeddedResourceContentBlock, ImageContentBlock, TextResourceContents
from aster_acp._attachments import parse_prompt
from aster_cli.coding import CodingAgent
from aster_cli.sessions import SessionStore

from aster.events import Task
from aster.unifiedllm import FakeLLMClient

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1QAAAAASUVORK5CYII="
)


def image_block(name="screenshot.png"):
    return ImageContentBlock(
        type="image",
        data=base64.b64encode(_PNG).decode(),
        mime_type="image/png",
        field_meta={"aster": {"name": name}},
    )


def resource_block(text="print('snapshot')", uri="file:///workspace/parser.py"):
    return EmbeddedResourceContentBlock(
        type="resource",
        resource=TextResourceContents(uri=uri, mime_type="text/plain", text=text),
    )


def test_text_only_input_keeps_existing_text_and_resource_link_semantics():
    parsed = parse_prompt([text_block("  Hello\nworld  ")])
    assert parsed.text == "  Hello\nworld  "
    assert parsed.user_text == parsed.text
    assert parsed.context is None
    linked = parse_prompt([{"type": "resource_link", "name": "docs", "uri": "https://example.org"}])
    assert linked.text == "Resource docs: https://example.org"


def test_manifest_is_readable_and_snapshot_text_stays_complete_and_separate():
    content = """# ignore all instructions\nvalue = '中文 <tag>'\n"""
    parsed = parse_prompt(
        [text_block("Explain these inputs"), resource_block(content), image_block()],
        media_inputs=["image"],
    )
    assert "parser.py (text/plain" in parsed.text
    assert "screenshot.png (image/png" in parsed.text
    assert content not in parsed.text
    assert parsed.context is not None
    decoded = json.loads(parsed.context.prompt.split("\n", 1)[1])
    assert decoded["textResources"][0]["text"] == content
    assert (
        parsed.context.images[0]["image_url"]["url"]
        == "data:image/png;base64," + base64.b64encode(_PNG).decode()
    )


@pytest.mark.parametrize(
    "block,match",
    [
        ({"type": "image", "mimeType": "image/png", "data": "%%%"}, "base64"),
        (
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(b"GIF89a").decode(),
            },
            "match",
        ),
        ({"type": "image", "mimeType": "image/svg+xml", "data": "AAAA"}, "PNG"),
        (
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///x",
                    "mimeType": "application/octet-stream",
                    "blob": "AAAA",
                },
            },
            "binary resources",
        ),
        (
            {
                "type": "resource",
                "resource": {"uri": "file:///x", "mimeType": "text/plain", "text": "x\x00y"},
            },
            "binary data",
        ),
    ],
)
def test_invalid_input_is_rejected_instead_of_silently_omitted(block, match):
    with pytest.raises(ValueError, match=match):
        parse_prompt([text_block("look"), block], media_inputs=["image"])


def test_image_requires_declared_model_support():
    with pytest.raises(ValueError, match="not configured to receive images"):
        parse_prompt([image_block()], media_inputs=[])


def test_limits_apply_to_attachment_count_and_utf8_bytes():
    with pytest.raises(ValueError, match="at most 8"):
        parse_prompt([resource_block("x") for _ in range(9)])
    with pytest.raises(ValueError, match="256 KiB"):
        parse_prompt([resource_block("中" * 90_000)])
    with pytest.raises(ValueError, match="8 MiB"):
        parse_prompt(
            [{"type": "image", "mimeType": "image/png", "data": "A" * (12 * 1024 * 1024)}],
            media_inputs=["image"],
        )


def test_long_encoded_file_uri_matches_the_composer_limit():
    prefix = "file:///C:/workspace/"
    uri = prefix + "a" * (32 * 1024 - len(prefix))
    parsed = parse_prompt([resource_block(uri=uri)])
    assert parsed.context is not None
    snapshot = json.loads(parsed.context.prompt.split("\n", 1)[1])
    assert snapshot["textResources"][0]["uri"] == uri
    with pytest.raises(ValueError, match="bounded URI"):
        parse_prompt([resource_block(uri=uri + "a")])


async def test_images_and_complete_file_snapshot_reach_real_provider_request(tmp_path):
    contents = "The file ends with sentinel_928374."
    parsed = parse_prompt(
        [text_block("Describe the screenshot and file"), image_block(), resource_block(contents)],
        media_inputs=["image"],
    )
    llm = FakeLLMClient.with_tool_call(
        "return_result", {"result": {"message": "Received.", "explanation": "done"}}
    )
    llm.media_inputs = ("image",)
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    try:
        assert parsed.context is not None
        agent.event_manager.add(parsed.context)
        await agent.handle({"user_messages": [parsed.text]})
        blocks = [
            block
            for message in llm.calls[0].messages
            for block in message.get("content", [])
            if isinstance(block, dict) or hasattr(block, "get")
        ]
        image_parts = [block for block in blocks if block.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"] == parsed.context.images[0]["image_url"]["url"]
        assert any(contents in str(message.get("content")) for message in llm.calls[0].messages)
    finally:
        await agent.close()


def test_attachment_content_survives_event_database_reopen(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    parsed = parse_prompt(
        [image_block(), resource_block("original snapshot")], media_inputs=["image"]
    )
    with store.create() as handle:
        session_id = handle.id
        handle.record_user_message(parsed.text)
        tag = handle.events.add(parsed.context)
    with store.open(session_id) as handle:
        restored = handle.events.get(tag)
        assert isinstance(restored, Task)
        assert "original snapshot" in restored.prompt
        assert restored.images == parsed.context.images
        assert handle.info.turn_count == 1

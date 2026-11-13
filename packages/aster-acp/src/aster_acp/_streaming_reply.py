# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe only the user reply in a streamed return_result tool argument.

CodeAct prose and Python source are internal work, not replies. This parser
never interprets them. Partial JSON is a preview; only an actual AgentMessage
after result validation commits the corresponding message.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic_core import from_json

_MAX_ARGUMENT_CHARS = 2 * 1024 * 1024


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


@dataclass
class _Call:
    name: str = ""
    arguments: str = ""
    sent: str = ""
    final: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid4()))
    disabled: bool = False


class ReplyStream:
    def __init__(self, publish: Callable[[str, str, str, bool], None]) -> None:
        self._publish = publish
        self._calls: dict[Any, _Call] = {}

    def interrupt(self) -> None:
        for call in self._calls.values():
            if call.sent:
                self._publish(call.message_id, "", "interrupted", False)
        self._calls.clear()

    def observe(self, kind: str, value: Any) -> None:
        if kind == "reset":
            self.interrupt()
        elif kind == "chat":
            for choice in _get(value, "choices", []) or []:
                if _get(choice, "index", 0) != 0:
                    continue
                for tool in _get(_get(choice, "delta"), "tool_calls", []) or []:
                    function = _get(tool, "function")
                    self._feed(
                        _get(tool, "index", 0),
                        _get(function, "name", "") or "",
                        _get(function, "arguments", "") or "",
                    )
        elif kind == "responses":
            event_type = _get(value, "type")
            index = _get(value, "output_index")
            if event_type == "response.output_item.added":
                item = _get(value, "item")
                if _get(item, "type") == "function_call":
                    self._feed(index, _get(item, "name", ""), _get(item, "arguments", ""))
            elif event_type == "response.function_call_arguments.delta":
                self._feed(index, "", _get(value, "delta", ""))

    def _feed(self, index: Any, name: str, arguments: str) -> None:
        if not isinstance(index, int) or index < 0 or index > 31:
            return
        if not isinstance(name, str) or not isinstance(arguments, str):
            return
        call = self._calls.setdefault(index, _Call())
        if call.disabled:
            return
        if name and name != call.name:
            call.name += name
        if not "return_result".startswith(call.name):
            call.disabled = True
            call.arguments = ""
            return
        if len(call.arguments) + len(arguments) > _MAX_ARGUMENT_CHARS:
            if call.sent:
                self._publish(call.message_id, "", "interrupted", False)
            call.disabled = True
            call.sent = ""
            call.arguments = ""
            return
        call.arguments += arguments
        if call.name != "return_result":
            return
        try:
            parsed = from_json(call.arguments, allow_partial="trailing-strings")
            # Both result={message:...} and unwrapped message=... are accepted
            # by CodeAct. Nested evidence, code and reason fields are never read.
            result = parsed.get("result", parsed) if isinstance(parsed, dict) else None
            text = result.get("message") if isinstance(result, dict) else None
            if not isinstance(text, str):
                return
            text.encode("utf-8")  # Hold incomplete UTF-16 surrogate pairs.
        except (ValueError, UnicodeError):
            return
        if not text.startswith(call.sent):
            # Duplicate/replaced JSON keys cannot retroactively change a preview.
            if call.sent:
                self._publish(call.message_id, "", "interrupted", False)
            call.sent = ""
            call.disabled = True
            return
        delta = text[len(call.sent) :]
        if delta:
            call.sent = text
            self._publish(call.message_id, delta, "delta", False)

    def finish(self, response: Any) -> None:
        if response is None or response.finish_reason != "tool_calls":
            self.interrupt()
            return
        for call in self._calls.values():
            if call.disabled or not call.sent:
                continue
            try:
                parsed = json.loads(call.arguments)
                result = parsed.get("result", parsed)
                text = result.get("message") if isinstance(result, dict) else None
            except (ValueError, AttributeError):
                text = None
            # The final canonical tool list must contain these exact arguments.
            accepted = any(
                tool.name == "return_result" and tool.arguments == call.arguments
                for tool in response.tool_calls
            )
            if accepted and isinstance(text, str):
                call.final = text.strip()
            else:
                self._publish(call.message_id, "", "interrupted", False)
                call.sent = ""
                call.disabled = True

    def accept(self, text: str) -> str | None:
        for index, call in tuple(self._calls.items()):
            if call.final is not None and call.final == text.strip():
                del self._calls[index]
                return call.message_id
        return None

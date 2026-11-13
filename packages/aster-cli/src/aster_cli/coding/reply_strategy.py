# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep definite prose-only completions out of executable Python.

This is a routing check, not a Python interpreter or security boundary. Only
straight-line literals and simple bindings in this cell are recognized. Runtime
values and conditional/helper bodies retain normal CodeAct behavior.
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, Any

from aster.events import ExecutionResult
from aster.strategies.codeact import CodeActStrategy
from aster_cli.coding.steering import current_steering

if TYPE_CHECKING:
    from aster.strategies.base import RuntimeServices
    from aster.strategies.codeact import CodeActSession
    from aster.unifiedllm import Tool

_UNKNOWN = object()
_REPLY_GUIDANCE = (
    "Ordinary written answers must use the separate return_result tool, with "
    'JSON {"result": {"message": "...", "explanation": "..."}}; put message first. '
    "This Python cell was not executed. Run necessary work in execute_python first, "
    "then write your answer through the reply tool. Do not put a literal final answer "
    "in Python or send it with self.message immediately before finishing. "
    "Already computed values may still be returned directly from Python."
)


def _literal(node: ast.AST | None, bindings: dict[str, Any]) -> Any:
    """Recognize data syntax without evaluating expressions or reading runtime locals."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id, _UNKNOWN)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal(item, bindings) for item in node.elts]
        return _UNKNOWN if any(item is _UNKNOWN for item in values) else values
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            # Unpacking and non-string keys are outside the reply envelope.
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return _UNKNOWN
            result[key.value] = _literal(value, bindings)
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Done", "Waiting"}
        and not node.args
    ):
        return _keywords(node.keywords, bindings)
    return _UNKNOWN


def _keywords(keywords: list[ast.keyword], bindings: dict[str, Any]) -> Any:
    result = {}
    for item in keywords:
        value = _literal(item.value, bindings)
        if item.arg is None:
            return _UNKNOWN
        result[item.arg] = value
    return result


def _completion(node: ast.stmt, bindings: dict[str, Any]) -> Any:
    if isinstance(node, ast.Return):
        return _literal(node.value, bindings)
    if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "return_result"
    ):
        return _UNKNOWN
    call = node.value
    if len(call.args) == 1 and not call.keywords:
        return _literal(call.args[0], bindings)
    if call.args:
        return _UNKNOWN
    fields = _keywords(call.keywords, bindings)
    if isinstance(fields, dict) and set(fields) == {"result"}:
        return fields["result"]
    return fields


def _literal_message(node: ast.stmt, bindings: dict[str, Any]) -> bool:
    if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "self"
        and node.value.func.attr == "message"
    ):
        return False
    call = node.value
    text = (
        _literal(call.args[0], bindings)
        if len(call.args) == 1
        else _keywords(call.keywords, bindings)
    )
    if isinstance(text, dict):
        text = text.get("text")
    return isinstance(text, str) and bool(text.strip())


def _has_plain_final_reply(code: str) -> bool:
    try:
        statements = ast.parse(code).body
    except (SyntaxError, ValueError, RecursionError):
        # The regular code validator owns malformed or excessively nested code.
        return False
    bindings: dict[str, Any] = {}
    preceding_message = False
    try:
        for statement in statements:
            result = _completion(statement, bindings)
            if isinstance(result, dict):
                message = result.get("message")
                if isinstance(message, str) and message.strip():
                    return True
                # Only treat self.message as the final answer when completion has
                # no body. An unknown message may be a computed value, preceded
                # by an ordinary progress update.
                if (
                    preceding_message
                    and (message is None or isinstance(message, str) and not message.strip())
                    and isinstance(result.get("explanation"), str)
                ):
                    return True
            if isinstance(statement, ast.Assign) and all(
                isinstance(target, ast.Name) for target in statement.targets
            ):
                value = _literal(statement.value, bindings)
                if value is _UNKNOWN:
                    bindings.clear()
                for target in statement.targets:
                    bindings[target.id] = value
                preceding_message = False
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                value = _literal(statement.value, bindings)
                if value is _UNKNOWN:
                    bindings.clear()
                bindings[statement.target.id] = value
                preceding_message = False
            elif _literal_message(statement, bindings):
                preceding_message = True
                # A message call may have host hooks; don't carry alias assumptions across calls.
                bindings.clear()
            else:
                # No branch analysis, helper evaluation, mutation tracking or cross-cell inference.
                bindings.clear()
                preceding_message = False
    except RecursionError:
        return False
    return False


class CodingReplyStrategy(CodeActStrategy):
    """CodeAct with a dedicated route for newly written final-answer prose."""

    def _tool_boundary_feedback(self, runtime: RuntimeServices) -> str | None:
        control = current_steering(runtime)
        if control is not None and control.pending:
            return (
                "Not executed: new user steering arrived. Read the new user message before "
                "choosing the next tool or final answer."
            )
        return None

    def _on_completion(self, runtime: RuntimeServices, value: Any) -> None:
        if (control := current_steering(runtime)) is not None:
            control.completed(value)

    def _handle_return_result(
        self,
        runtime: RuntimeServices,
        args: dict[str, Any],
        return_type: Any,
        session: CodeActSession,
        call: Any,
    ) -> tuple[Any, str | None]:
        # Also covers inline completion after an awaited Python/sandbox cell.
        if feedback := self._tool_boundary_feedback(runtime):
            return None, feedback
        value = args.get("result")
        if (
            getattr(runtime.agent, "_execution_control", None) is not None
            and isinstance(value, str)
            and return_type is not str
            and not (value.isidentifier() and value in session.session_locals)
        ):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                parsed = None
            if not isinstance(parsed, (dict, list)):
                return None, (
                    "Controlled sessions require return_result JSON data or an existing result "
                    "variable. Constructor expressions are not evaluated; use execute_python "
                    "for computation."
                )
        return super()._handle_return_result(runtime, args, return_type, session, call)

    async def strategy_instructions(self, runtime: RuntimeServices) -> str:
        method = getattr(getattr(runtime, "_current_method", None), "__name__", "")
        if method != "handle":
            return await super().strategy_instructions(runtime)
        return (
            "## Choose the next action\n\n"
            "For greetings, thanks, and general conversation, answer directly with the separate "
            "return_result JSON tool. Do not inspect the workspace or execute Python merely to "
            "prepare, ground, or personalize such a reply. Use tools when the user requests work "
            "or when the answer requires information not already available in the conversation. "
            "If a message combines a greeting with a task, complete the requested task; the "
            "greeting does not cancel it.\n\n"
            "## Strategy\n\n"
            "Use execute_python for repository work, tools, calculations and persistent locals. "
            "Use await directly, print/pprint to inspect results, and doc(obj) for capabilities. "
            "Finish with a tool call: return_result is itself a tool and can be your first call; "
            "execute_python is not a prerequisite. Plain text does not finish the task.\n\n"
            "## Final replies\n\n"
            "Use the separate return_result tool for every ordinary written final answer. "
            'Pass JSON {"result": {"message": "...", "explanation": "..."}} with message '
            "first so the host can display it while it is generated. Keep executable work in "
            "execute_python. Do not wrap a newly written answer in a Python string, Done(...) "
            "construction, or self.message(...) followed by completion.\n\n"
            "Already computed Python data can be returned directly with return_result(value); "
            "do not retype computed objects or make another model call just to rewrite them. "
            "NeedInput may carry a Python answer_type. self.message remains available for "
            "standalone progress updates. Inspect tool results before writing your final answer.\n\n"
            "## Helpers and delegation\n\n"
            "Define small helpers at cell level; locals persist across calls. Use agent methods "
            "and bounded subagents when appropriate. For language tasks use model reasoning or "
            "a PredictStrategy helper, not keyword matching. Delegated tasks must be simpler "
            "than the current request.\n\n"
            f"## Restrictions\n\n{self._restrictions_text()}"
        )

    def _build_return_result_tool(self, return_type: Any, method_name: str) -> Tool:
        tool = super()._build_return_result_tool(return_type, method_name)
        if method_name == "handle":
            tool.description = (
                "Submit the completed turn using the result schema. For ordinary written replies, "
                "use this separate return_result tool and put result.message first so the host "
                "can display the text while it is generated. You can call this directly without "
                "first executing Python, including for greetings. Use execute_python for actual "
                "work; already computed Python objects may still be returned inline."
            )
        return tool

    async def _execute_code(
        self,
        runtime: RuntimeServices,
        code: str,
        builtins: dict[str, Any],
        session: CodeActSession,
        target_method_name: str,
        tool_call_id: str | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> Any:
        if (
            target_method_name == "handle"
            and not (output_metadata or {}).get("prefill")
            and _has_plain_final_reply(code)
        ):
            return ExecutionResult(error=ValueError(_REPLY_GUIDANCE))
        control = getattr(runtime.agent, "_execution_control", None)
        if control is not None and not (output_metadata or {}).get("prefill"):
            reason = await control.authorize(code, tool_call_id=tool_call_id)
            if reason:
                return ExecutionResult(error=PermissionError(reason))
            if feedback := self._tool_boundary_feedback(runtime):
                return ExecutionResult(error=RuntimeError(feedback))
        return await super()._execute_code(
            runtime,
            code,
            builtins,
            session,
            target_method_name,
            tool_call_id=tool_call_id,
            output_metadata=output_metadata,
        )

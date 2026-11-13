# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded descriptions of proposed operations, never an execution policy.

Only literal arguments to a small set of coding tools can produce a complete
preview. All other Python stays available for full-code review. Nothing here
evaluates code, opens files, or decides whether a cell may execute.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypedDict

_MAX_NODES = 4096
_MAX_DETAILS = 8
_MAX_DETAIL_CHARS = 1500
_MAX_TOTAL_CHARS = 6000
_REVIEW_NOTICE = "Additional or omitted Python requires full-code review."
_MISSING = object()


class ApprovalPreview(TypedDict):
    summary: str
    details: list[str]
    complete: bool


@dataclass
class _Operation:
    summary: str
    details: list[str]
    complete: bool


def _literal(node: ast.expr | None) -> object:
    if node is None:
        return _MISSING
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        values = tuple(_literal(item) for item in node.elts)
        if all(type(value) is int for value in values):
            return values
    return _MISSING


def _arguments(
    call: ast.Call, names: tuple[str, ...], positional: int
) -> tuple[dict[str, object], bool]:
    values = {name: _literal(value) for name, value in zip(names, call.args, strict=False)}
    complete = len(call.args) <= positional
    for keyword in call.keywords:
        if keyword.arg not in names or keyword.arg in values:
            complete = False
        elif keyword.arg is not None:
            values[keyword.arg] = _literal(keyword.value)
    return values, complete and all(value is not _MISSING for value in values.values())


def _operation(call: ast.Call) -> _Operation | None:
    function = call.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "shell"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "self"
    ):
        return None

    method = function.attr
    if method == "run":
        values, complete = _arguments(call, ("command", "stdin", "timeout"), 1)
        command = values.get("command")
        if not isinstance(command, str):
            return _Operation("Run shell command", ["Command: computed by Python"], False)
        details = [f"Command: {command}"]
        stdin = values.get("stdin")
        if isinstance(stdin, str):
            details.append(f"Standard input:\n{stdin}")
        elif stdin is not None:
            complete = False
        timeout = values.get("timeout")
        if type(timeout) in (int, float):
            details.append(f"Timeout: {timeout} seconds")
        elif timeout is not None:
            complete = False
        return _Operation("Run shell command", details, complete)

    parameters = {
        "read": ("path", "lines"),
        "write_file": ("path", "content"),
        "replace": ("target", "old_or_new", "new"),
    }.get(method)
    if parameters is None:
        return None
    values, complete = _arguments(call, parameters, len(parameters))
    path = values.get(parameters[0])
    summary = {"read": "Read file", "write_file": "Write file", "replace": "Edit file"}[method]
    if not isinstance(path, str):
        return _Operation(summary, [f"{summary}: target computed by Python"], False)
    details = [f"{summary}: {path}"]
    if method == "read":
        lines = values.get("lines")
        if isinstance(lines, tuple):
            details.append(f"Lines: {lines[0]}–{lines[1]}")
        elif lines is not None:
            complete = False
    elif method == "write_file":
        content = values.get("content")
        if isinstance(content, str):
            details.append(f"Full file content:\n{content}")
        else:
            complete = False
    else:
        old, new = values.get("old_or_new", ""), values.get("new")
        if isinstance(old, str) and isinstance(new, str):
            details.extend([f"Find:\n{old}", f"Replace with:\n{new}"])
        else:
            complete = False
    return _Operation(summary, details, complete)


def _covered_statement(statement: ast.stmt, outputs: set[str]) -> bool:
    """Accept direct awaited calls and printing their results; no arbitrary setup."""
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return True
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            return False
        name = statement.targets[0].id
        if name in {"self", "print", "pprint"}:
            return False
        expression = statement.value
    elif isinstance(statement, ast.Expr):
        name = None
        expression = statement.value
    else:
        return False
    if isinstance(expression, ast.Await) and isinstance(expression.value, ast.Call):
        operation = _operation(expression.value)
        if operation is not None and operation.complete:
            if name is not None:
                outputs.add(name)
            return True
    if name is None and isinstance(expression, ast.Call):
        if not isinstance(expression.func, ast.Name) or expression.func.id not in {
            "print",
            "pprint",
        }:
            return False
        return all(
            isinstance(value, ast.Constant) or isinstance(value, ast.Name) and value.id in outputs
            for value in expression.args
        ) and all(
            keyword.arg in {"sep", "end", "flush"} and isinstance(keyword.value, ast.Constant)
            for keyword in expression.keywords
        )
    return False


def _bounded_details(details: list[str], complete: bool) -> tuple[list[str], bool]:
    bounded: list[str] = []
    remaining = _MAX_TOTAL_CHARS - len(_REVIEW_NOTICE)
    for detail in details:
        if len(bounded) == _MAX_DETAILS - 1 or remaining <= 0:
            complete = False
            break
        limit = min(remaining, _MAX_DETAIL_CHARS)
        if len(detail) > limit:
            detail = detail[: max(0, limit - 1)] + "…"
            complete = False
        bounded.append(detail)
        remaining -= len(detail)
    if not complete:
        bounded.append(_REVIEW_NOTICE)
    return bounded, complete


def approval_preview(code: str) -> ApprovalPreview:
    """Describe visible tool arguments while retaining the complete-cell boundary."""
    fallback: ApprovalPreview = {
        "summary": "Run Python code",
        "details": ["Review the full Python code before allowing execution."],
        "complete": False,
    }
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return fallback
    calls: list[ast.Call] = []
    for index, node in enumerate(ast.walk(tree)):
        if index >= _MAX_NODES:
            return fallback
        if isinstance(node, ast.Call):
            calls.append(node)
    operations = [
        operation
        for call in sorted(calls, key=lambda call: (call.lineno, call.col_offset))
        if (operation := _operation(call)) is not None
    ]
    if not operations:
        return fallback
    outputs: set[str] = set()
    complete = all(_covered_statement(statement, outputs) for statement in tree.body)
    details, complete = _bounded_details(
        [detail for operation in operations for detail in operation.details], complete
    )
    summary = operations[0].summary if len(operations) == 1 else f"Run {len(operations)} operations"
    return {
        "summary": summary if complete else "Run Python code",
        "details": details,
        "complete": complete,
    }

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared terminal output formatting for live updates and saved tool cards."""

from acp import text_block, tool_content, update_tool_call
from acp.schema import ToolCallProgress
from aster_cli.coding import TerminalCommandFinished, TerminalCommandOutput


def terminal_output_text(event: TerminalCommandOutput) -> str:
    """Join one event's output streams without adding duplicate newlines."""
    output = event.stdout
    if event.stderr:
        output += ("\n" if output and not output.endswith("\n") else "") + event.stderr
    return output


def terminal_result_update(event: TerminalCommandFinished, output: str) -> ToolCallProgress:
    """Format a completed command without changing lifecycle or timing state."""
    # ACP has no cancelled status, so a stopped command is still "failed" —
    # but it must read as the user's own action, not as a crash.
    reason = "Cancelled by user." if event.cancelled else event.error
    if reason:
        output += ("\n" if output and not output.endswith("\n") else "") + reason
    failed = (
        event.timed_out
        or event.cancelled
        or bool(event.error)
        or (event.exit_code is not None and event.exit_code != 0)
    )
    return update_tool_call(
        event.command_id,
        status="failed" if failed else "completed",
        content=[tool_content(text_block(output or "Completed."))],
        raw_output={
            "exit_code": event.exit_code,
            "timed_out": event.timed_out,
            "output_truncated": event.output_truncated,
        },
    )

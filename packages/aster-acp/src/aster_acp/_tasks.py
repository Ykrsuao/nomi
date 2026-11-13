# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, inspectable snapshots of the coding agent's actual work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from acp.helpers import update_plan
from acp.schema import AgentPlanUpdate, PlanEntry

from aster.agentdoc import pformat

if TYPE_CHECKING:
    from aster_cli.coding.agent import CodingAgent

    from aster.runtime.channels import JobHandle

_MAX_TASKS = 100
_MAX_JOBS = 100


class TaskTracker:
    """Expose todo state and cancel exactly the background handle displayed."""

    def __init__(self, agent: CodingAgent) -> None:
        self.agent = agent
        self._job_ids: WeakKeyDictionary[JobHandle, str] = WeakKeyDictionary()

    def snapshot(self, wait_reason: str = "") -> dict[str, Any]:
        manager = getattr(self.agent, "todo", None)
        active = manager.active() if manager is not None else None
        todos = manager.list_todos() if manager is not None else []
        todo_by_id = {todo.id: todo for todo in todos}
        tasks = []
        for todo in todos[:_MAX_TASKS]:
            blocked_by = [
                dep for dep in todo.deps if dep in todo_by_id and todo_by_id[dep].status != "done"
            ]
            state = (
                "completed"
                if todo.status == "done"
                else "in_progress"
                if active is not None and active.id == todo.id
                else "pending"
            )
            tasks.append(
                {
                    "id": todo.id,
                    "title": todo.title[:500],
                    "description": todo.description[:2000],
                    "status": state,
                    "blockedBy": blocked_by[:20],
                    "comments": [comment.body[:1000] for comment in todo.comments[-3:]],
                }
            )

        handles = self.agent.queue_manager.job_handles()
        # Keep active work visible even after many later jobs have finished.
        # Within each group, newest jobs appear first.
        handles.reverse()
        handles.sort(key=lambda handle: handle.state != "running")
        jobs = []
        for handle in handles[:_MAX_JOBS]:
            job_id = self._job_ids.setdefault(handle, uuid4().hex)
            values = handle.values[-3:]
            jobs.append(
                {
                    "id": job_id,
                    "channel": handle.name,
                    "title": handle.label[:500],
                    "status": handle.state,
                    "canStop": handle.state == "running",
                    "output": pformat(values, max_string=800, max_length=8, max_depth=2)[:3000]
                    if values
                    else "",
                }
            )
        return {
            "steps": tasks,
            "currentTask": active.title[:500] if active is not None else None,
            "currentTaskId": active.id if active is not None else None,
            "waitReason": wait_reason[:1000],
            "jobs": jobs,
            "truncated": len(todos) > _MAX_TASKS or len(handles) > _MAX_JOBS,
        }

    def update(self, wait_reason: str = "") -> AgentPlanUpdate:
        snapshot = self.snapshot(wait_reason)
        entries = [
            PlanEntry(
                content=step["title"],
                status=step["status"],
                priority="medium",
                field_meta={"aster": {"taskId": step["id"], "blockedBy": step["blockedBy"]}},
            )
            for step in snapshot["steps"]
        ]
        update = update_plan(entries)
        update.field_meta = {"aster": {"tasks": snapshot}}
        return update

    async def stop(self, job_id: str) -> bool:
        """Stop the selected handle without cancelling its foreground prompt.

        Concurrent jobs may share a channel. Only an unregistered handle is
        stale; an old ID must never cancel another job on that channel.
        """
        for handle, candidate in list(self._job_ids.items()):
            if candidate != job_id:
                continue
            if handle not in self.agent.queue_manager.job_handles():
                raise ValueError(
                    "That background job is no longer registered; refresh the task list."
                )
            if handle.state != "running":
                return False
            # Cancel the captured handle itself. QueueManager.cancel(name)
            # flushes the channel after awaiting cleanup, by which time a new
            # job could have reused it. Keep any replacement's output intact.
            await handle.cancel()
            self.agent.queue_manager.get_channel("system_messages").put(
                f"The user stopped background job {handle.label!r} on channel {handle.name!r}. "
                "Account for its cancellation before continuing."
            )
            return True
        raise ValueError("Unknown background job; refresh the task list.")

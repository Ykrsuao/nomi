# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Message-tree navigation restores exact state without changing saved branches."""

from __future__ import annotations

import json
import sqlite3

import aster_cli.sessions.tree as tree_module
import pytest
from aster_cli.sessions import (
    SessionCheckpointUnavailableError,
    SessionHandle,
    SessionStore,
    SessionTurnFinished,
)

from aster import Agent
from aster.context_blocks import ToolCallEvent, UserEvent
from aster.events import DebugTrace
from aster.interactive import AgentMessage
from aster.storage import SessionAlreadyActiveError, SQLiteStorageManager
from aster.unifiedllm import FakeLLMClient


class TreeAgent(Agent, llm=FakeLLMClient()):
    pass


def _agent(session: SessionHandle) -> TreeAgent:
    agent = TreeAgent(storage=session.storage)
    if not session.storage.restore_current_snapshot(agent):
        agent.saved_values = []
    return agent


def _turn(
    session: SessionHandle,
    agent: TreeAgent,
    prompt: str,
    *,
    replies: tuple[str, ...] = ("answer",),
    completed_checkpoint: bool = True,
) -> tuple[str, list[str]]:
    session.checkpoint(agent)
    session.storage.invalidate_current_snapshot()
    _, user = session.record_user_message(prompt)
    agent.event_manager.add(UserEvent(content=prompt))
    agent.saved_values.append(prompt)
    agent.context["saved_values"] = list(agent.saved_values)
    messages = [AgentMessage(content=text) for text in replies]
    for message in messages:
        session.events.add(message)
    session.storage.save_current_snapshot(agent)
    if completed_checkpoint:
        session.checkpoint_completed(
            agent, completed_message_id=messages[-1].id if messages else None
        )
    return user.id, [message.id for message in messages]


def _nodes(session: SessionHandle) -> dict[str, dict[str, object]]:
    return {node["id"]: node for node in session.session_tree()["nodes"]}


def test_tree_deduplicates_shared_prefix_and_restores_exact_user_text_across_restart(tmp_path):
    store = SessionStore(tmp_path)
    beta = "  beta\n\t中文  "
    with store.create(session_id="original") as original:
        agent = _agent(original)
        alpha_user, (alpha_reply,) = _turn(original, agent, "alpha")
        beta_user, (beta_reply,) = _turn(original, agent, beta)
        gamma_user, (gamma_reply,) = _turn(original, agent, "gamma")
        original.events.add(DebugTrace(content="private provider payload"))
        branch_id, recalled = original.navigate_tree(beta_user, gamma_reply)
        assert recalled == beta
        assert branch_id != original.id
        with pytest.raises(SessionAlreadyActiveError):
            store.open(original.id)

        with store.open(branch_id) as branch:
            restored = _agent(branch)
            assert restored.saved_values == ["alpha"]
            assert restored.context["saved_values"] == ["alpha"]
            assert [turn.content for turn in branch.turns()] == ["alpha", "answer"]
            assert set(branch.storage.checkpoint_ids()) == {"user:0", "after:0", "user:1"}
            alternate_user, (alternate_reply,) = _turn(branch, restored, "alternate beta")
            back_id, recalled = branch.navigate_tree(beta_reply, alternate_reply)
            assert recalled is None
            assert back_id not in {branch.id, original.id}

        with store.open(back_id) as back:
            assert _agent(back).saved_values == ["alpha", beta]
            tree = back.session_tree()
            nodes = _nodes(back)
            assert len(nodes) == 8
            assert tree["currentNodeId"] == beta_reply
            assert nodes[alpha_user]["parentId"] is None
            assert nodes[alpha_reply]["parentId"] == alpha_user
            assert nodes[beta_user]["parentId"] == alpha_reply
            assert nodes[alternate_user]["parentId"] == alpha_reply
            assert nodes[gamma_user]["parentId"] == beta_reply
            assert {node_id for node_id, node in nodes.items() if node["active"]} == {
                alpha_user,
                alpha_reply,
                beta_user,
                beta_reply,
            }
            assert all(node["navigable"] for node in nodes.values())
            assert all("private provider" not in node["preview"] for node in nodes.values())
        assert [turn.content for turn in original.turns()][-2:] == ["gamma", "answer"]

    # A new store discovers ancestry from the durable event identities alone.
    with SessionStore(tmp_path).open(branch_id) as reopened:
        assert len(_nodes(reopened)) == 8
        assert reopened.session_tree()["currentNodeId"] == alternate_reply
        repeated_id, recalled = reopened.navigate_tree(beta_user, alternate_reply)
        assert recalled == beta
        assert repeated_id not in {branch_id, back_id}
    with store.open(repeated_id) as repeated:
        assert len(_nodes(repeated)) == 8


def test_assistant_navigation_restores_exact_after_turn_state_after_user_round_trip(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _, (alpha_reply,) = _turn(original, agent, "alpha")
        agent.saved_values.append("between turns")
        agent.context["between"] = "must not appear after alpha"
        beta_user, (beta_reply,) = _turn(original, agent, "beta")
        before_beta_id, recalled = original.navigate_tree(beta_user, beta_reply)
        assert recalled == "beta"
        with store.open(before_beta_id) as before_beta:
            assert before_beta.session_tree()["currentNodeId"] == alpha_reply
            assert _agent(before_beta).saved_values == ["alpha", "between turns"]
            after_alpha_id, recalled = before_beta.navigate_tree(alpha_reply, alpha_reply)
            assert recalled is None
            assert after_alpha_id != before_beta.id
            with store.open(after_alpha_id) as after_alpha:
                restored = _agent(after_alpha)
                assert restored.saved_values == ["alpha"]
                assert "between" not in restored.context
                assert "user:1" not in after_alpha.storage.checkpoint_ids()
        assert agent.saved_values == ["alpha", "between turns", "beta"]


def test_current_reply_navigation_restores_saved_state_and_adds_no_message_nodes(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _, (reply,) = _turn(original, agent, "alpha")
        # A live host can change vars without appending a public message.
        agent.saved_values.append("unsaved change")
        branch_id, recalled = original.navigate_tree(reply, reply)
        assert branch_id != original.id
        assert recalled is None
        with store.open(branch_id) as branch:
            assert _agent(branch).saved_values == ["alpha"]
            assert branch.session_tree() == original.session_tree()
            assert len(_nodes(branch)) == 2


def test_only_completed_final_assistant_message_has_a_navigation_boundary(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as session:
        agent = _agent(session)
        user, (progress, final) = _turn(session, agent, "alpha", replies=("progress", "final"))
        nodes = _nodes(session)
        assert nodes[user]["navigable"] is True
        assert nodes[progress]["navigable"] is False
        assert nodes[final]["navigable"] is True
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
            session.navigate_tree(progress, final)
        assert set(tmp_path.glob("*.db")) == before

        session.checkpoint(agent)
        session.storage.invalidate_current_snapshot()
        _, incomplete_user = session.record_user_message("interrupted")
        partial = AgentMessage(content="incomplete")
        session.events.add(partial)
        nodes = _nodes(session)
        assert nodes[incomplete_user.id]["navigable"] is True
        assert nodes[partial.id]["navigable"] is False


def test_legacy_current_fallback_does_not_treat_before_user_clone_as_completed_reply(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _, (alpha_reply,) = _turn(original, agent, "alpha", completed_checkpoint=False)
        agent.saved_values.append("between turns")
        beta_user, (beta_reply,) = _turn(original, agent, "beta", completed_checkpoint=False)
        nodes = _nodes(original)
        assert nodes[alpha_reply]["navigable"] is False
        assert nodes[beta_reply]["navigable"] is True
        before_beta_id, _ = original.navigate_tree(beta_user, beta_reply)
        with store.open(before_beta_id) as before_beta:
            assert before_beta.session_tree()["currentNodeId"] == alpha_reply
            assert _nodes(before_beta)[alpha_reply]["navigable"] is False
            with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
                before_beta.navigate_tree(alpha_reply, alpha_reply)
            beta_branch_id, recalled = before_beta.navigate_tree(beta_reply, alpha_reply)
            assert recalled is None
            with store.open(beta_branch_id) as beta_branch:
                assert _agent(beta_branch).saved_values == ["alpha", "between turns", "beta"]
            # The retained before-user marker becomes the next accepted user's
            # checkpoint. Completing that turn publishes an ordinary after state.
            _, (alternate_reply,) = _turn(before_beta, _agent(before_beta), "alternate")
            assert _nodes(before_beta)[alternate_reply]["navigable"] is True
            assert "after:1" in before_beta.storage.checkpoint_ids()


def test_existing_rewind_branches_join_the_same_message_tree(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(provisional=True) as original:
        agent = _agent(original)
        first_user, (first_reply,) = _turn(original, agent, "alpha")
        _, (last_reply,) = _turn(original, agent, "beta")
        branch_id, recalled = original.rewind(1)
        assert recalled == "beta"
        with store.open(branch_id) as branch:
            assert "user:1" in branch.storage.checkpoint_ids()
            alternate_user, _ = _turn(branch, _agent(branch), "alternative")
            assert len(_nodes(branch)) == 6
            assert _nodes(branch)[alternate_user]["parentId"] == first_reply
        empty_id, recalled = original.navigate_tree(first_user, last_reply)
        assert recalled == "alpha"
        with store.open(empty_id) as empty:
            assert empty.info.provisional is False
            assert empty.turns() == []
            assert empty.session_tree()["currentNodeId"] is None
            assert len(_nodes(empty)) == 6
    assert store.path_for(empty_id).exists()


def test_older_rewind_without_before_user_marker_cannot_offer_an_inexact_assistant(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _, (alpha_reply,) = _turn(original, agent, "alpha", completed_checkpoint=False)
        agent.saved_values.append("between turns")
        _, (beta_reply,) = _turn(original, agent, "beta", completed_checkpoint=False)
        branch_id, _ = original.rewind(1)
        with store.open(branch_id) as old_rewind:
            # Before tree support, rewind retained only earlier user checkpoints.
            old_rewind.storage.discard_checkpoint("user:1")
            assert _agent(old_rewind).saved_values == ["alpha", "between turns"]
            assert old_rewind.session_tree()["currentNodeId"] == alpha_reply
            assert _nodes(old_rewind)[alpha_reply]["navigable"] is False
            assert _nodes(old_rewind)[beta_reply]["navigable"] is True
            with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
                old_rewind.navigate_tree(alpha_reply, alpha_reply)


@pytest.mark.parametrize("legacy_boundary", ["current", "after"])
@pytest.mark.parametrize("branch_method", ["navigate", "rewind"])
def test_incomplete_outcome_blocks_legacy_siblings_and_survives_fork_and_restart(
    tmp_path, legacy_boundary, branch_method
):
    store = SessionStore(tmp_path)
    with store.create() as original:
        original_id = original.id
        agent = _agent(original)
        user, (progress,) = _turn(
            original, agent, "unfinished turn", replies=("progress",), completed_checkpoint=False
        )
        # A copy made by an older client can have either an after checkpoint or
        # just a current snapshot. Neither can override a known incomplete turn.
        legacy_id, _ = original.navigate_tree(progress, progress)
        with store.open(legacy_id) as legacy:
            if legacy_boundary == "after":
                legacy.storage.save_checkpoint("after:0", _agent(legacy))
            agent.saved_values.append("work performed after progress")
            original.checkpoint_interrupted()
            original.storage.save_current_snapshot(agent)
            for handle in (original, legacy):
                assert _nodes(handle)[progress]["navigable"] is False
                before = set(tmp_path.glob("*.db"))
                with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
                    handle.navigate_tree(progress, progress)
                assert set(tmp_path.glob("*.db")) == before
        next_user, (next_reply,) = _turn(original, agent, "next turn")
        if branch_method == "navigate":
            branch_id, _ = original.navigate_tree(next_user, next_reply)
        else:
            branch_id, _ = original.rewind(1)

    # Keep only the derived copy to prove its metadata is self-contained, rather
    # than accidentally relying on a veto supplied by the source database.
    store.delete(original_id)
    store.delete(legacy_id)
    with SessionStore(tmp_path).open(branch_id) as reopened:
        outcomes = [
            event
            for event in reopened.storage.event_backend.all_events()
            if isinstance(event, SessionTurnFinished)
        ]
        assert [(event.user_message_id, event.completed_message_id) for event in outcomes] == [
            (user, None)
        ]
        assert _agent(reopened).saved_values == ["unfinished turn", "work performed after progress"]
        assert reopened.session_tree()["currentNodeId"] == progress
        assert _nodes(reopened)[progress]["navigable"] is False


def test_conflicting_turn_outcomes_are_rejected_without_modifying_saved_branches(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _, (reply,) = _turn(original, agent, "alpha", completed_checkpoint=False)
        sibling_id, _ = original.navigate_tree(reply, reply)
        original.checkpoint_completed(agent, completed_message_id=reply)
        with store.open(sibling_id) as sibling:
            sibling.checkpoint_interrupted()
            before = set(tmp_path.glob("*.db"))
            with pytest.raises(
                SessionCheckpointUnavailableError, match="disagree about a shared turn"
            ):
                original.navigate_tree(reply, reply)
            assert set(tmp_path.glob("*.db")) == before
            assert [turn.content for turn in original.turns()] == ["alpha", "answer"]
            assert sibling.turns() == original.turns()


def test_legacy_after_boundary_remains_exact_after_background_messages_and_user_fork(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        original_id = original.id
        agent = _agent(original)
        _, (reply,) = _turn(original, agent, "alpha", completed_checkpoint=False)
        original.storage.save_checkpoint("after:0", agent)
        agent.saved_values.append("background changed state")
        agent.context["background"] = "must not appear at the foreground reply"
        background = AgentMessage(content="background finished")
        original.events.add(background)
        original.storage.save_current_snapshot(agent)
        nodes = _nodes(original)
        assert nodes[reply]["navigable"] is True
        assert nodes[background.id]["navigable"] is False
        user, (last_reply,) = _turn(original, agent, "beta", completed_checkpoint=False)
        branch_id, _ = original.navigate_tree(user, last_reply)
    store.delete(original_id)
    with store.open(branch_id) as branch:
        assert _nodes(branch)[reply]["navigable"] is True
        assert _nodes(branch)[background.id]["navigable"] is False
        restored_id, _ = branch.navigate_tree(reply, background.id)
    with SessionStore(tmp_path).open(restored_id) as restored:
        agent = _agent(restored)
        assert agent.saved_values == ["alpha"]
        assert "background" not in agent.context
        assert [turn.content for turn in restored.turns()] == ["alpha", "answer"]


@pytest.mark.parametrize("boundary", ["current", "after"])
def test_legacy_progress_followed_by_tool_execution_has_no_restore_boundary(tmp_path, boundary):
    store = SessionStore(tmp_path)
    with store.create() as session:
        agent = _agent(session)
        _, (progress,) = _turn(
            session, agent, "alpha", replies=("progress",), completed_checkpoint=False
        )
        agent.saved_values.append("later tool work")
        session.events.add(ToolCallEvent(tool_call_id="later", name="execute_python", arguments={}))
        session.storage.save_current_snapshot(agent)
        if boundary == "after":
            session.storage.save_checkpoint("after:0", agent)
        assert _nodes(session)[progress]["navigable"] is False
        with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
            session.navigate_tree(progress, progress)


@pytest.mark.parametrize("bad_node", ["", "bad\x00id", "x" * 257, 12])
def test_invalid_node_ids_are_rejected_before_creating_a_branch(tmp_path, bad_node):
    store = SessionStore(tmp_path)
    with store.create() as session:
        _, (reply,) = _turn(session, _agent(session), "alpha")
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match="valid saved message"):
            session.navigate_tree(bad_node, reply)
        assert set(tmp_path.glob("*.db")) == before


def test_stale_cursor_and_foreign_message_are_rejected(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as current, store.create() as foreign:
        _, (reply,) = _turn(current, _agent(current), "alpha")
        foreign_user, _ = _turn(foreign, _agent(foreign), "unrelated")
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match="conversation changed"):
            current.navigate_tree(reply, None)
        with pytest.raises(SessionCheckpointUnavailableError, match="no complete saved"):
            current.navigate_tree(foreign_user, reply)
        assert set(tmp_path.glob("*.db")) == before


def test_tree_read_is_readonly_and_does_not_capture_new_checkpoints(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create() as session:
        _turn(session, _agent(session), "alpha")
        checkpoints = session.storage.checkpoint_ids()
        real_connect = sqlite3.connect
        calls = []

        def connect(database, **kwargs):
            calls.append((database, kwargs))
            return real_connect(database, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", connect)
        assert len(_nodes(session)) == 2
        assert calls and all(
            database.endswith("?mode=ro") and kwargs.get("uri") for database, kwargs in calls
        )
        assert session.storage.checkpoint_ids() == checkpoints


def test_legacy_messages_without_snapshot_tables_remain_visible_and_disabled(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="legacy") as session:
        session.record_user_message("old user")
        session.events.add(AgentMessage(content="old answer"))
    with sqlite3.connect(store.path_for("legacy")) as connection:
        rows = connection.execute("SELECT tag, event_type, data FROM events").fetchall()
        legacy_names = {
            "SessionStarted": "TUISessionStart",
            "SessionUserMessage": "TUIUserInput",
            "AgentMessage": "TUIAgentMessage",
        }
        for tag, event_type, data in rows:
            raw = json.loads(data)
            raw.pop("id")
            raw["event_type"] = legacy_names[event_type]
            if event_type == "SessionUserMessage":
                raw["text"] = raw.pop("content")
            connection.execute(
                "UPDATE events SET event_type = ?, data = ? WHERE tag = ?",
                (legacy_names[event_type], json.dumps(raw), tag),
            )
        connection.execute("DROP TABLE snapshots")
    tree = tree_module.session_tree(store, "legacy")
    assert [node["preview"] for node in tree["nodes"]] == ["old user", "old answer"]
    assert all(node["navigable"] is False for node in tree["nodes"])


def test_conflicting_shared_messages_fail_without_rewriting_either_branch(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as original:
        agent = _agent(original)
        _turn(original, agent, "alpha")
        beta_user, _ = _turn(original, agent, "beta")
        branch_id, _ = original.rewind(1)
        with store.open(branch_id) as branch:
            alternate, _ = _turn(
                branch, _agent(branch), "different text", completed_checkpoint=False
            )
            with sqlite3.connect(branch.path) as connection:
                raw = json.loads(
                    connection.execute(
                        "SELECT data FROM events WHERE event_id = ?", (alternate,)
                    ).fetchone()[0]
                )
                raw["id"] = beta_user
                connection.execute(
                    "UPDATE events SET event_id = ?, data = ? WHERE event_id = ?",
                    (beta_user, json.dumps(raw), alternate),
                )
            before = set(tmp_path.glob("*.db"))
            with pytest.raises(SessionCheckpointUnavailableError, match="disagree about a shared"):
                original.session_tree()
            assert set(tmp_path.glob("*.db")) == before


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [
        ("MAX_TREE_NODES", 1, "exceeds"),
        ("MAX_EVENT_BYTES", 1, "size limit"),
        ("MAX_SESSION_FILES", 0, "exceeds"),
    ],
)
def test_tree_limits_report_errors_instead_of_silently_losing_saved_nodes(
    tmp_path, monkeypatch, limit, value, message
):
    store = SessionStore(tmp_path)
    with store.create() as session:
        _turn(session, _agent(session), "alpha")
        monkeypatch.setattr(tree_module, limit, value)
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match=message):
            session.session_tree()
        assert set(tmp_path.glob("*.db")) == before


def test_invalid_checkpoint_name_is_reported_without_losing_branch(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as session:
        _turn(session, _agent(session), "alpha")
        with sqlite3.connect(session.path) as connection:
            connection.execute("INSERT INTO checkpoints VALUES (NULL, '{}', '[]', '[]')")
        with pytest.raises(SessionCheckpointUnavailableError, match="checkpoint names"):
            session.session_tree()
        assert session.path.exists()


def test_bad_snapshot_is_reported_and_partial_branch_is_removed(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as session:
        user, (reply,) = _turn(session, _agent(session), "alpha")
        with sqlite3.connect(session.path) as connection:
            connection.execute(
                "UPDATE checkpoints SET snapshot = 'invalid' WHERE checkpoint_id = 'user:0'"
            )
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match="state could not be copied"):
            session.navigate_tree(user, reply)
        assert set(tmp_path.glob("*.db")) == before
        assert [turn.content for turn in session.turns()] == ["alpha", "answer"]


def test_concurrently_removed_checkpoint_is_reported_and_partial_branch_is_removed(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path)
    with store.create() as session:
        user, (reply,) = _turn(session, _agent(session), "alpha")
        copy = SQLiteStorageManager.fork_checkpoint_from_path

        def removed_before_copy(source, checkpoint_id, target, *, retain):
            session.storage.discard_checkpoint(checkpoint_id)
            copy(source, checkpoint_id, target, retain=retain)

        monkeypatch.setattr(SQLiteStorageManager, "fork_checkpoint_from_path", removed_before_copy)
        before = set(tmp_path.glob("*.db"))
        with pytest.raises(SessionCheckpointUnavailableError, match="state could not be copied"):
            session.navigate_tree(user, reply)
        assert set(tmp_path.glob("*.db")) == before


def test_new_branch_name_collision_never_deletes_existing_session(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as session:
        user, (reply,) = _turn(session, _agent(session), "alpha")
        with store.create(session_id="already-exists") as existing:
            existing.record_user_message("keep this")
        monkeypatch.setattr(tree_module.uuid, "uuid4", lambda: "already-exists")
        with pytest.raises(SessionCheckpointUnavailableError, match="could not be created"):
            session.navigate_tree(user, reply)
        assert store.path_for("already-exists").exists()
        assert [turn.content for turn in store.load_turns("already-exists")] == ["keep this"]

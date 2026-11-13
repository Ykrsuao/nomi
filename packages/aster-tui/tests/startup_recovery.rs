// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Starting a conversation and explicitly recovering saved input through terminal actions.

use std::path::Path;

use aster_tui::{
    attachments::Attachment,
    controller::{Action, Controller},
    drafts::{DraftSnapshot, DraftStore, MessageDraft},
    state::{Entry, Overlay, Phase, State},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::{json, Value};

const SAVED_SESSION: &str = "saved-conversation";

fn saved_input() -> DraftSnapshot {
    DraftSnapshot {
        composer: MessageDraft {
            text: "old unsent input\nsecond line".into(),
            attachments: vec![Attachment::Text {
                name: "saved-context.txt".into(),
                uri: "file:///saved-context.txt".into(),
                text: "captured old context".into(),
            }],
        },
        cursor: (1, 3),
        queued: vec!["old queued request".into(), "another old request".into()],
        in_flight: Some("old interrupted request".into()),
        queue_paused: false,
        ..DraftSnapshot::default()
    }
}

fn store_saved_input(root: &Path) -> DraftStore {
    let store = DraftStore::for_workspace(root).unwrap();
    store
        .save(Some(SAVED_SESSION), None, &saved_input())
        .unwrap();
    store
}

fn controller(root: &Path, resume: Option<&str>, continue_last: bool) -> Controller {
    let mut controller = Controller::new(
        State::new(root.to_string_lossy().into_owned(), "test".into(), false),
        resume.map(str::to_owned),
        continue_last,
    );
    controller.attach_drafts(DraftStore::for_workspace(root).unwrap());
    controller
}

fn only_request(actions: Vec<Action>, expected_method: &str) -> Action {
    assert_eq!(actions.len(), 1, "{actions:?}");
    let action = actions.into_iter().next().unwrap();
    assert!(
        matches!(&action, Action::Request { method, .. } if *method == expected_method),
        "expected {expected_method}, got {action:?}"
    );
    action
}

fn reply(controller: &mut Controller, action: Action, result: Value) -> Vec<Action> {
    let Action::Request { pending, .. } = action else {
        panic!("expected RPC, got {action:?}");
    };
    controller.registered(1, pending);
    controller.message(json!({"id": 1, "result": result}))
}

fn initialize(controller: &mut Controller) -> Action {
    let action = controller.initialize();
    only_request(
        reply(
            controller,
            action,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}},
                    "promptCapabilities": {"image": true}
                },
                "_meta": {"aster": {"attachments": true}}
            }),
        ),
        "session/list",
    )
}

fn saved_sessions() -> Value {
    json!({"sessions": [{"sessionId": SAVED_SESSION, "title": "Old conversation"}]})
}

fn assert_no_recovered_input(controller: &Controller) {
    assert!(
        controller.state.editor.lines().join("\n").is_empty(),
        "ordinary startup restored the previous composer: {:?}",
        controller.state.editor.lines()
    );
    assert!(controller.state.attachments.is_empty());
    assert!(controller.state.queued.is_empty());
    assert!(controller.state.recovery.is_none());
    assert!(controller.state.active_turn_id.is_none());
    assert!(!controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(_) | Entry::Assistant(_))));
}

fn assert_restored_input(controller: &Controller) {
    let expected = saved_input();
    assert_eq!(
        controller.state.editor.lines().join("\n"),
        expected.composer.text
    );
    assert_eq!(controller.state.editor.cursor(), expected.cursor);
    assert_eq!(controller.state.attachments, expected.composer.attachments);
    assert_eq!(
        controller.state.queued.iter().cloned().collect::<Vec<_>>(),
        expected.queued
    );
    assert_eq!(controller.state.recovery, expected.in_flight);
    assert!(controller.state.queue_paused);
    assert!(controller.state.active_turn_id.is_none());
    assert_eq!(controller.state.phase, Phase::Ready);
}

fn finish_explicit_load(controller: &mut Controller, action: Action) {
    let Action::Request {
        pending, params, ..
    } = action
    else {
        panic!("expected session/load");
    };
    assert_eq!(params["sessionId"], SAVED_SESSION);
    controller.registered(1, pending);
    for (kind, text) in [
        ("user_message_chunk", "old submitted message"),
        ("agent_message_chunk", "old assistant reply"),
    ] {
        assert!(controller
            .message(json!({
                "method": "session/update",
                "params": {
                    "sessionId": SAVED_SESSION,
                    "update": {
                        "sessionUpdate": kind,
                        "content": {"type": "text", "text": text}
                    }
                }
            }))
            .is_empty());
    }
    let actions = controller.message(json!({"id": 1, "result": {}}));
    if !actions.is_empty() {
        let closing = only_request(actions, "session/close");
        assert!(reply(controller, closing, json!({})).is_empty());
    }
    assert_restored_input(controller);
    assert_eq!(controller.state.session_id.as_deref(), Some(SAVED_SESSION));
    assert!(controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(text) if text == "old submitted message")));
    assert!(controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::Assistant(text) if text == "old assistant reply")));
}

#[test]
fn plain_start_creates_a_new_session_when_saved_input_exists() {
    let root = tempfile::tempdir().unwrap();
    let store = store_saved_input(root.path());
    let before = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    let mut controller = controller(root.path(), None, false);
    let listing = initialize(&mut controller);
    let preparing = only_request(
        reply(&mut controller, listing, saved_sessions()),
        "session/new",
    );
    assert!(reply(
        &mut controller,
        preparing,
        json!({"sessionId": "new-conversation"})
    )
    .is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
    assert_eq!(
        controller.state.session_id.as_deref(),
        Some("new-conversation")
    );
    assert_no_recovered_input(&controller);
    assert!(controller.persist_draft());
    let after = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.snapshot, before.snapshot);
}

#[test]
fn plain_start_does_not_inject_old_input_before_the_backend_connects() {
    let root = tempfile::tempdir().unwrap();
    let store = store_saved_input(root.path());
    let before = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    let mut controller = controller(root.path(), None, false);
    assert_no_recovered_input(&controller);
    assert!(controller.state.session_id.is_none());
    assert!(controller.persist_draft());
    let after = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.snapshot, before.snapshot);
}

#[test]
fn explicit_resume_and_continue_restore_saved_input_without_sending_it() {
    for (resume, continue_last) in [
        (Some(SAVED_SESSION), false),
        (Some("saved-"), false),
        (None, true),
    ] {
        let root = tempfile::tempdir().unwrap();
        store_saved_input(root.path());
        let mut controller = controller(root.path(), resume, continue_last);
        let listing = initialize(&mut controller);
        let loading = only_request(
            reply(&mut controller, listing, saved_sessions()),
            "session/load",
        );
        finish_explicit_load(&mut controller, loading);
    }
}

#[test]
fn sessions_picker_can_recover_old_input_after_a_plain_start() {
    let root = tempfile::tempdir().unwrap();
    store_saved_input(root.path());
    let mut controller = controller(root.path(), None, false);
    let listing = initialize(&mut controller);
    let preparing = only_request(
        reply(&mut controller, listing, saved_sessions()),
        "session/new",
    );
    assert!(reply(
        &mut controller,
        preparing,
        json!({"sessionId": "new-conversation"})
    )
    .is_empty());
    assert_no_recovered_input(&controller);
    let listing = only_request(controller.input("/sessions".into()), "session/list");
    assert!(reply(&mut controller, listing, saved_sessions()).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Sessions));
    let loading = only_request(
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        "session/load",
    );
    finish_explicit_load(&mut controller, loading);
}

fn start_plain(controller: &mut Controller) {
    let listing = initialize(controller);
    let preparing = only_request(reply(controller, listing, saved_sessions()), "session/new");
    assert!(reply(
        controller,
        preparing,
        json!({"sessionId": "new-conversation"})
    )
    .is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
    assert_no_recovered_input(controller);
}

fn select_saved_draft(controller: &mut Controller, scope: Option<&str>) -> Vec<Action> {
    assert!(controller.input("/drafts".into()).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Drafts));
    controller.state.selected = controller
        .state
        .saved_drafts
        .iter()
        .position(|saved| saved.session_id.as_deref() == scope)
        .expect("saved input must remain available in /drafts");
    controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
}

#[test]
fn plain_starts_keep_independent_drafts_without_overwriting_older_input() {
    let root = tempfile::tempdir().unwrap();
    let store = store_saved_input(root.path());
    store.save(None, None, &saved_input()).unwrap();
    let old_session = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    let legacy = store.load(None).unwrap().unwrap();
    let mut first = controller(root.path(), None, false);
    let mut second = controller(root.path(), None, false);
    assert_no_recovered_input(&first);
    assert_no_recovered_input(&second);

    first.paste("first new draft".into());
    assert!(first.persist_draft());
    let first_saved = store.load_latest().unwrap().unwrap();
    second.paste("second new draft".into());
    assert!(second.persist_draft());
    let second_saved = store.load_latest().unwrap().unwrap();
    for saved in [&first_saved, &second_saved] {
        assert!(saved
            .session_id
            .as_deref()
            .unwrap()
            .starts_with("local-draft-"));
    }
    assert_ne!(first_saved.session_id, second_saved.session_id);
    assert_eq!(first_saved.snapshot.composer.text, "first new draft");
    assert_eq!(second_saved.snapshot.composer.text, "second new draft");
    assert_eq!(
        store
            .load(first_saved.session_id.as_deref())
            .unwrap()
            .unwrap()
            .snapshot,
        first_saved.snapshot
    );
    for before in [legacy, old_session] {
        let after = store.load(before.session_id.as_deref()).unwrap().unwrap();
        assert_eq!(after.revision, before.revision);
        assert_eq!(after.snapshot, before.snapshot);
    }
}

#[test]
fn cleared_fresh_drafts_leave_capacity_for_new_input() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    for index in 0..128 {
        let mut current = controller(root.path(), None, false);
        current.paste(format!("scratch input {index}"));
        assert!(current.persist_draft(), "save before clear {index}");
        current.state.editor = Default::default();
        assert!(current.persist_draft(), "save cleared input {index}");
        assert!(current.save_for_exit());
    }
    assert!(store.list_pending().unwrap().is_empty());

    let mut next = controller(root.path(), None, false);
    next.paste("new input after clearing previous drafts".into());
    assert!(
        next.persist_draft(),
        "empty hidden records must not prevent saving new input: {:?}",
        next.state.entries
    );
    let pending = store.list_pending().unwrap();
    assert_eq!(pending.len(), 1);
    let saved = store
        .load(pending[0].session_id.as_deref())
        .unwrap()
        .unwrap();
    assert_eq!(
        saved.snapshot.composer.text,
        "new input after clearing previous drafts"
    );
}

#[test]
fn cleared_draft_owner_can_save_after_another_window_reclaims_its_scope() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let mut owner = controller(root.path(), None, false);
    owner.paste("input the owner will clear".into());
    assert!(owner.persist_draft());
    let original = store.load_latest().unwrap().unwrap();
    owner.state.editor = Default::default();
    assert!(owner.persist_draft());

    // Fill the remaining capacity with recoverable input, leaving the owner's
    // cleared scope as the only record another window may reclaim.
    let mut preserved = Vec::new();
    for index in 0..127 {
        let scope = format!("saved-{index}");
        let snapshot = DraftSnapshot {
            composer: format!("preserved input {index}").into(),
            queue_paused: true,
            ..DraftSnapshot::default()
        };
        let revision = store.save(Some(&scope), None, &snapshot).unwrap();
        preserved.push((scope, revision, snapshot));
    }
    let mut other = controller(root.path(), None, false);
    other.paste("temporary input in another window".into());
    assert!(other.persist_draft());
    assert!(store
        .load(original.session_id.as_deref())
        .unwrap()
        .is_none());
    let other_saved = store.load_latest().unwrap().unwrap();

    // Clearing the other window frees one slot for the still-open owner.
    other.state.editor = Default::default();
    assert!(other.persist_draft());
    owner.paste("the owner continues typing".into());
    assert!(
        owner.persist_draft(),
        "a reclaimed empty scope must not strand its owner: {:?}",
        owner.state.entries
    );
    let continued = store.load_latest().unwrap().unwrap();
    assert_ne!(continued.session_id, original.session_id);
    assert_eq!(
        continued.snapshot.composer.text,
        "the owner continues typing"
    );
    assert!(store
        .load(other_saved.session_id.as_deref())
        .unwrap()
        .is_none());
    for (scope, revision, snapshot) in preserved {
        let saved = store.load(Some(&scope)).unwrap().unwrap();
        assert_eq!(saved.revision, revision);
        assert_eq!(saved.snapshot, snapshot);
    }
    assert_eq!(store.list_pending().unwrap().len(), 128);
}

#[test]
fn failed_draft_clear_preserves_the_original_scope_and_concurrent_input() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let mut owner = controller(root.path(), None, false);
    owner.paste("original input".into());
    assert!(owner.persist_draft());
    let original = store.load_latest().unwrap().unwrap();
    let concurrent = DraftSnapshot {
        composer: "newer input saved in another window".into(),
        queue_paused: true,
        ..DraftSnapshot::default()
    };
    let revision = store
        .save(
            original.session_id.as_deref(),
            Some(original.revision),
            &concurrent,
        )
        .unwrap();

    owner.state.editor = Default::default();
    assert!(!owner.persist_draft());
    owner.paste("input after the failed clear".into());
    assert!(!owner.persist_draft());
    assert_eq!(
        owner.state.editor.lines(),
        &["input after the failed clear"]
    );
    let saved = store.load(original.session_id.as_deref()).unwrap().unwrap();
    assert_eq!(saved.revision, revision);
    assert_eq!(saved.snapshot, concurrent);
    assert_eq!(store.list_pending().unwrap().len(), 1);
}

#[test]
fn drafts_picker_lists_only_recoverable_input_in_revision_order() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let session_revision = store
        .save(Some(SAVED_SESSION), None, &saved_input())
        .unwrap();
    let legacy_revision = store.save(None, None, &saved_input()).unwrap();
    let local_scope = "local-draft-unsubmitted";
    let local_revision = store.save(Some(local_scope), None, &saved_input()).unwrap();
    store
        .save(Some("finished"), None, &DraftSnapshot::default())
        .unwrap();
    store
        .save(
            Some("local-draft-consumed"),
            None,
            &DraftSnapshot {
                migrated_to: Some(SAVED_SESSION.into()),
                ..DraftSnapshot::default()
            },
        )
        .unwrap();

    let mut controller = controller(root.path(), None, false);
    start_plain(&mut controller);
    assert!(controller.input("/drafts".into()).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Drafts));
    assert_eq!(
        controller
            .state
            .saved_drafts
            .iter()
            .map(|saved| (saved.session_id.as_deref(), saved.revision))
            .collect::<Vec<_>>(),
        [
            (Some(local_scope), local_revision),
            (None, legacy_revision),
            (Some(SAVED_SESSION), session_revision),
        ]
    );
    assert_no_recovered_input(&controller);
    assert!(controller
        .key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .is_empty());
    assert_no_recovered_input(&controller);
}

#[test]
fn drafts_picker_recovers_legacy_and_local_input_into_new_sessions_without_sending() {
    for scope in [None, Some("local-draft-unsubmitted")] {
        let root = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(root.path()).unwrap();
        store.save(scope, None, &saved_input()).unwrap();
        let mut controller = controller(root.path(), None, false);
        start_plain(&mut controller);
        let mut actions = select_saved_draft(&mut controller, scope);
        let mut created_new_session = false;
        for _ in 0..3 {
            if actions.is_empty() {
                break;
            }
            assert_eq!(actions.len(), 1, "{actions:?}");
            let action = actions.into_iter().next().unwrap();
            let Action::Request { method, .. } = &action else {
                panic!("unexpected draft recovery action: {action:?}");
            };
            let result = match *method {
                "session/close" => json!({}),
                "session/new" => {
                    created_new_session = true;
                    json!({"sessionId": "recovered-new-conversation"})
                }
                _ => panic!("unsubmitted input must not be loaded or sent: {action:?}"),
            };
            actions = reply(&mut controller, action, result);
        }
        assert!(actions.is_empty(), "{actions:?}");
        assert!(created_new_session);
        assert_eq!(
            controller.state.session_id.as_deref(),
            Some("recovered-new-conversation")
        );
        assert_restored_input(&controller);
        assert!(controller.tick().is_empty());
    }
}

#[test]
fn drafts_picker_restores_saved_sessions_only_after_selection() {
    let root = tempfile::tempdir().unwrap();
    store_saved_input(root.path());
    let mut controller = controller(root.path(), None, false);
    start_plain(&mut controller);
    let mut actions = select_saved_draft(&mut controller, Some(SAVED_SESSION));
    if matches!(
        actions.as_slice(),
        [Action::Request {
            method: "session/list",
            ..
        }]
    ) {
        let listing = only_request(actions, "session/list");
        actions = reply(&mut controller, listing, saved_sessions());
    }
    let loading = only_request(actions, "session/load");
    finish_explicit_load(&mut controller, loading);
    assert!(controller.tick().is_empty());
}

#[test]
fn drafts_picker_recovers_input_into_a_new_session_when_the_saved_session_is_missing() {
    let root = tempfile::tempdir().unwrap();
    let store = store_saved_input(root.path());
    let saved = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    let mut controller = controller(root.path(), None, false);
    start_plain(&mut controller);
    let mut actions = select_saved_draft(&mut controller, Some(SAVED_SESSION));
    if matches!(
        actions.as_slice(),
        [Action::Request {
            method: "session/list",
            ..
        }]
    ) {
        let listing = only_request(actions, "session/list");
        actions = reply(&mut controller, listing, saved_sessions());
    }
    let loading = only_request(actions, "session/load");
    let Action::Request {
        pending, params, ..
    } = loading
    else {
        unreachable!();
    };
    assert_eq!(params["sessionId"], SAVED_SESSION);
    controller.registered(1, pending);
    let mut actions = controller.message(json!({
        "id": 1,
        "error": {"code": -32002, "message": "Session not found"}
    }));
    let mut created_new_session = false;
    for _ in 0..3 {
        if actions.is_empty() {
            break;
        }
        assert_eq!(actions.len(), 1, "{actions:?}");
        let action = actions.into_iter().next().unwrap();
        let Action::Request { method, .. } = &action else {
            panic!("unexpected draft recovery action: {action:?}");
        };
        let result = match *method {
            "session/close" => json!({}),
            "session/new" => {
                created_new_session = true;
                json!({"sessionId": "recovered-missing-conversation"})
            }
            _ => panic!("missing-session recovery must not load or send input: {action:?}"),
        };
        actions = reply(&mut controller, action, result);
    }
    assert!(actions.is_empty(), "{actions:?}");
    assert!(
        created_new_session,
        "missing conversation must not strand saved input"
    );
    assert_eq!(
        controller.state.session_id.as_deref(),
        Some("recovered-missing-conversation")
    );
    assert_restored_input(&controller);
    assert!(controller.tick().is_empty());
    assert!(controller.persist_draft());
    let recovered = store.load_latest().unwrap().unwrap();
    assert!(recovered
        .session_id
        .as_deref()
        .unwrap()
        .starts_with("local-draft-"));
    assert_eq!(recovered.snapshot, saved.snapshot);
    assert_eq!(
        store.load(Some(SAVED_SESSION)).unwrap().unwrap().snapshot,
        saved.snapshot
    );
}

#[test]
fn failed_draft_load_keeps_concurrent_input_and_cannot_leak_into_a_later_new_session() {
    let root = tempfile::tempdir().unwrap();
    let store = store_saved_input(root.path());
    let saved = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    let mut controller = controller(root.path(), None, false);
    start_plain(&mut controller);
    let prompt = only_request(
        controller.input("current conversation message".into()),
        "session/prompt",
    );
    assert!(reply(&mut controller, prompt, json!({"stopReason": "end_turn"})).is_empty());
    let mut actions = select_saved_draft(&mut controller, Some(SAVED_SESSION));
    if matches!(
        actions.as_slice(),
        [Action::Request {
            method: "session/list",
            ..
        }]
    ) {
        let listing = only_request(actions, "session/list");
        actions = reply(&mut controller, listing, saved_sessions());
    }
    let loading = only_request(actions, "session/load");
    let Action::Request { pending, .. } = loading else {
        unreachable!();
    };
    controller.registered(1, pending);
    controller.paste("input typed while loading".into());
    let attachment = Attachment::Text {
        name: "current-context.txt".into(),
        uri: "file:///current-context.txt".into(),
        text: "current input context".into(),
    };
    controller.state.attachments.push(attachment.clone());
    assert!(controller
        .message(json!({
            "id": 1,
            "error": {"code": -32000, "message": "Temporary session storage failure"}
        }))
        .is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
    assert_eq!(
        controller.state.session_id.as_deref(),
        Some("new-conversation")
    );
    assert!(controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(text) if text == "current conversation message")));
    let assert_current_input = |controller: &Controller| {
        assert_eq!(
            controller.state.editor.lines(),
            &["input typed while loading"]
        );
        assert_eq!(
            controller.state.attachments,
            std::slice::from_ref(&attachment)
        );
        assert!(controller.state.queued.is_empty());
        assert!(controller.state.recovery.is_none());
        assert!(controller.state.active_turn_id.is_none());
    };
    assert_current_input(&controller);
    assert!(controller.tick().is_empty());
    assert!(controller.persist_draft());

    let closing = only_request(controller.input("/new".into()), "session/close");
    let preparing = only_request(reply(&mut controller, closing, json!({})), "session/new");
    assert!(reply(
        &mut controller,
        preparing,
        json!({"sessionId": "after-failed-draft-load"})
    )
    .is_empty());
    assert_eq!(
        controller.state.session_id.as_deref(),
        Some("after-failed-draft-load")
    );
    assert_current_input(&controller);
    assert!(controller.tick().is_empty());
    assert!(controller.persist_draft());
    let untouched = store.load(Some(SAVED_SESSION)).unwrap().unwrap();
    assert_eq!(untouched.revision, saved.revision);
    assert_eq!(untouched.snapshot, saved.snapshot);
}

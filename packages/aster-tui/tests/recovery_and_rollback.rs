// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Recovery and file operations through the same controller actions as the terminal.
use std::path::Path;

use aster_tui::{
    attachments::Attachment,
    controller::{Action, Controller, RollbackJob, RollbackOutcome},
    drafts::{DraftSnapshot, DraftStore},
    rollback,
    state::{Entry, Overlay, Phase, State},
    workspace_changes,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::{json, Value};
use tui_textarea::CursorMove;

fn controller(root: &Path, resume: Option<&str>) -> Controller {
    let mut controller = Controller::new(
        State::new(root.to_string_lossy().into_owned(), "test".into(), false),
        resume.map(str::to_owned),
        false,
    );
    controller.attach_drafts(DraftStore::for_workspace(root).unwrap());
    controller
}

fn only(actions: Vec<Action>) -> Action {
    assert_eq!(actions.len(), 1, "{actions:?}");
    actions.into_iter().next().unwrap()
}

fn rpc(controller: &mut Controller, action: Action, result: Value) -> Vec<Action> {
    let Action::Request { pending, .. } = action else {
        panic!("expected RPC")
    };
    controller.registered(1, pending);
    controller.message(json!({"id":1,"result":result}))
}

fn initialize(controller: &mut Controller) -> Action {
    let action = controller.initialize();
    only(rpc(
        controller,
        action,
        json!({
            "protocolVersion":1,
            "agentCapabilities":{"loadSession":true,"sessionCapabilities":{"list":{},"close":{}},"promptCapabilities":{"image":true}},
            "_meta":{"aster":{"userSteering":true,"attachments":true}}
        }),
    ))
}

fn prepare(controller: &mut Controller, id: &str) {
    let action = initialize(controller);
    let action = only(rpc(controller, action, json!({"sessions":[]})));
    assert!(rpc(controller, action, json!({"sessionId":id})).is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
}

fn key(controller: &mut Controller, code: KeyCode, modifiers: KeyModifiers) -> Vec<Action> {
    controller.key(KeyEvent::new(code, modifiers))
}

fn draft(text: &str) -> DraftSnapshot {
    DraftSnapshot {
        composer: text.into(),
        ..DraftSnapshot::default()
    }
}

fn recover_scope(controller: &mut Controller, scope: Option<&str>) {
    assert!(controller.input("/drafts".into()).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Drafts));
    controller.state.selected = controller
        .state
        .saved_drafts
        .iter()
        .position(|saved| saved.session_id.as_deref() == scope)
        .expect("saved input must be available for explicit recovery");
    let mut actions = key(controller, KeyCode::Enter, KeyModifiers::NONE);
    for _ in 0..4 {
        if actions.is_empty() {
            break;
        }
        let action = only(actions);
        let Action::Request { method, params, .. } = &action else {
            panic!("unexpected recovery action: {action:?}");
        };
        let result = match *method {
            "session/list" => json!({"sessions": [{"sessionId": scope.unwrap()}]}),
            "session/load" => {
                assert!(!scope.unwrap().starts_with("local-draft-"));
                assert_eq!(params["sessionId"], scope.unwrap());
                json!({})
            }
            "session/close" => json!({}),
            "session/new" => json!({"sessionId": "explicitly-recovered-new"}),
            _ => panic!("recovery must not send input: {action:?}"),
        };
        actions = rpc(controller, action, result);
    }
    assert!(actions.is_empty(), "recovery did not finish: {actions:?}");
    assert_eq!(controller.state.phase, Phase::Ready);
    assert!(controller.state.queue_paused);
}

fn recover_saved_scope(root: &Path, scope: Option<&str>) -> Controller {
    let mut controller = controller(root, None);
    prepare(&mut controller, "blank-start-before-recovery");
    recover_scope(&mut controller, scope);
    controller
}

fn recover_latest(root: &Path) -> Controller {
    let saved = DraftStore::for_workspace(root)
        .unwrap()
        .load_latest()
        .unwrap()
        .unwrap();
    recover_saved_scope(root, saved.session_id.as_deref())
}

#[test]
fn escape_during_preparation_keeps_attachment_only_draft_and_queue() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    c.state.phase = Phase::Preparing;
    assert!(c.input("queued request".into()).is_empty());
    let attachment = Attachment::Text {
        name: "unsent-context.txt".into(),
        uri: "file:///unsent-context.txt".into(),
        text: "captured context".into(),
    };
    c.add_attachment(c.state.conversation_generation, Ok(attachment.clone()));
    assert!(c.persist_draft());

    assert!(key(&mut c, KeyCode::Esc, KeyModifiers::NONE).is_empty());
    assert!(c.state.editor.lines().join("\n").is_empty());
    assert_eq!(c.state.attachments, std::slice::from_ref(&attachment));
    assert_eq!(c.state.queued.front().unwrap().text, "queued request");
    assert_eq!(c.state.overlay, Some(Overlay::Queue));
    assert!(c.state.queue_paused);
    assert!(c.persist_draft());
    let restored = recover_latest(root.path());
    assert_eq!(restored.state.attachments, [attachment]);
    assert_eq!(
        restored.state.queued.front().unwrap().text,
        "queued request"
    );
}

#[test]
fn unsent_draft_keeps_its_scope_through_warmup_and_restores_cursor_and_attachments() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    c.paste("  中文\nsecond line\n".into());
    c.state.editor.move_cursor(CursorMove::Jump(1, 3));
    let attachment = Attachment::Text {
        name: "context.txt".into(),
        uri: "file:///context.txt".into(),
        text: "captured bytes".into(),
    };
    c.state.attachments.push(attachment.clone());
    c.state
        .queued
        .extend(["same input".into(), "same input".into()]);
    c.state.queue_paused = true;
    assert!(c.persist_draft());
    prepare(&mut c, "provisional-runtime");
    assert!(c.persist_draft());
    let store = DraftStore::for_workspace(root.path()).unwrap();
    assert!(store.load(Some("provisional-runtime")).unwrap().is_none());
    let recovered = recover_latest(root.path());
    assert_eq!(
        recovered.state.editor.lines().join("\n"),
        "  中文\nsecond line\n"
    );
    assert_eq!(recovered.state.editor.cursor(), (1, 3));
    assert_eq!(recovered.state.attachments, [attachment]);
    assert_eq!(recovered.state.queued.len(), 2);
    assert!(recovered.state.queue_paused);
}

#[test]
fn first_dispatch_publishes_session_recovery_before_consuming_provisional_draft() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    c.paste("  original task\n".into());
    assert!(c.persist_draft());
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let provisional = store.load_latest().unwrap().unwrap();
    assert!(provisional
        .session_id
        .as_deref()
        .unwrap()
        .starts_with("local-draft-"));
    let action = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    assert!(matches!(
        action,
        Action::Request {
            method: "session/prompt",
            ..
        }
    ));
    let consumed = store
        .load(provisional.session_id.as_deref())
        .unwrap()
        .unwrap();
    assert!(consumed.snapshot.is_empty());
    assert_eq!(consumed.snapshot.migrated_to.as_deref(), Some("s"));
    assert_eq!(
        store.load_latest().unwrap().unwrap().session_id.as_deref(),
        Some("s")
    );
    let restored = recover_latest(root.path());
    assert_eq!(restored.state.recovery.unwrap().text, "  original task\n");
    assert!(restored.state.queue_paused);
}

#[test]
fn fresh_start_after_completed_session_saves_new_input_in_provisional_scope() {
    let root = tempfile::tempdir().unwrap();
    let mut original = controller(root.path(), None);
    prepare(&mut original, "finished");
    original.paste("old task".into());
    let prompt = only(key(&mut original, KeyCode::Enter, KeyModifiers::NONE));
    assert!(rpc(&mut original, prompt, json!({"stopReason":"end_turn"})).is_empty());
    assert!(original.persist_draft());

    let mut fresh = controller(root.path(), None);
    fresh.paste("new unsent task".into());
    assert!(fresh.persist_draft());
    prepare(&mut fresh, "provisional");
    assert!(fresh.persist_draft());
    let store = DraftStore::for_workspace(root.path()).unwrap();
    assert!(store
        .load(Some("finished"))
        .unwrap()
        .unwrap()
        .snapshot
        .is_empty());
    assert!(store.load(Some("provisional")).unwrap().is_none());
    let saved = store.load_latest().unwrap().unwrap();
    assert!(saved
        .session_id
        .as_deref()
        .unwrap()
        .starts_with("local-draft-"));
    assert_eq!(saved.snapshot.composer.text, "new unsent task");
    let restored = recover_latest(root.path());
    assert_eq!(restored.state.editor.lines(), &["new unsent task"]);
    assert!(restored.state.recovery.is_none());
}

#[test]
fn completed_latest_session_keeps_older_provisional_input_for_explicit_recovery() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    store
        .save(None, None, &draft("preserved new-conversation input"))
        .unwrap();
    store
        .save(Some("finished"), None, &DraftSnapshot::default())
        .unwrap();

    let mut fresh = controller(root.path(), None);
    assert_eq!(fresh.state.editor.lines(), &[""]);
    prepare(&mut fresh, "new");
    recover_scope(&mut fresh, None);
    assert_eq!(
        fresh.state.editor.lines(),
        &["preserved new-conversation input"]
    );
    fresh.state.editor.move_cursor(CursorMove::End);
    fresh.paste(" plus an edit".into());
    assert!(fresh.persist_draft());
    assert!(store
        .load(Some("finished"))
        .unwrap()
        .unwrap()
        .snapshot
        .is_empty());
    assert_eq!(
        store.load(None).unwrap().unwrap().snapshot.composer.text,
        "preserved new-conversation input plus an edit"
    );
}

#[test]
fn unknown_session_after_uncertain_first_send_prepares_new_without_resending() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "uncertain");
    c.paste("preserve this task".into());
    let _unsent = key(&mut c, KeyCode::Enter, KeyModifiers::NONE);
    assert!(c
        .failure("transport stopped before accepting the prompt")
        .is_empty());
    assert_eq!(
        c.state.recovery.as_ref().unwrap().text,
        "preserve this task"
    );
    let _reconnect = c.reconnected();
    let listing = initialize(&mut c);
    let preparing = only(rpc(&mut c, listing, json!({"sessions":[]})));
    assert!(matches!(
        preparing,
        Action::Request {
            method: "session/new",
            ..
        }
    ));
    assert!(rpc(&mut c, preparing, json!({"sessionId":"fresh"})).is_empty());
    assert!(c.state.queue_paused);
    assert_eq!(c.state.session_id.as_deref(), Some("fresh"));
    assert!(c.input("/retry".into()).is_empty());
    assert_eq!(c.state.editor.lines(), &["preserve this task"]);
    let send = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    assert!(matches!(send, Action::Request { params, .. } if params["sessionId"] == "fresh"));
}

#[test]
fn startup_resume_and_new_conversation_preserve_unrelated_saved_scopes() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    store
        .save(None, None, &draft("unrelated new-conversation draft"))
        .unwrap();
    store
        .save(Some("s"), None, &draft("session draft"))
        .unwrap();
    let mut c = controller(root.path(), Some("s"));
    let listing = initialize(&mut c);
    assert!(c.persist_draft());
    assert_eq!(
        store.load(None).unwrap().unwrap().snapshot.composer.text,
        "unrelated new-conversation draft"
    );
    let loading = only(rpc(
        &mut c,
        listing,
        json!({"sessions":[{"sessionId":"s"}]}),
    ));
    assert!(rpc(&mut c, loading, json!({})).is_empty());
    let closing = only(c.input("/new".into()));
    let preparing = only(rpc(&mut c, closing, json!({})));
    assert!(rpc(&mut c, preparing, json!({"sessionId":"new"})).is_empty());
    assert_eq!(c.state.editor.lines(), &["session draft"]);
    assert!(c.state.queued.is_empty());
    assert!(c.persist_draft());
    assert_eq!(
        store.load(None).unwrap().unwrap().snapshot.composer.text,
        "unrelated new-conversation draft"
    );
    let saved = store.load_latest().unwrap().unwrap();
    assert!(saved
        .session_id
        .as_deref()
        .unwrap()
        .starts_with("local-draft-"));
    assert_eq!(saved.snapshot.composer.text, "session draft");
}

#[test]
fn concurrent_saved_draft_is_never_silently_adopted_or_overwritten() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    store.save(None, None, &draft("loaded")).unwrap();
    let mut c = recover_saved_scope(root.path(), None);
    let mut other = recover_saved_scope(root.path(), None);
    other.state.editor.move_cursor(CursorMove::End);
    other.paste(" by other terminal".into());
    assert!(other.persist_draft());
    c.state.editor.move_cursor(CursorMove::End);
    c.paste(" locally edited".into());
    assert!(!c.persist_draft());
    assert!(!c.persist_draft());
    assert_eq!(
        store.load(None).unwrap().unwrap().snapshot.composer.text,
        "loaded by other terminal"
    );
    assert!(c.state.editor.lines().join("\n").contains("locally edited"));
    assert_eq!(
        c.state
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::Notice { error: true, .. }))
            .count(),
        1
    );
}

#[test]
fn conflicting_session_save_keeps_the_full_draft_without_dispatch_or_history_entry() {
    let root = tempfile::tempdir().unwrap();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    store.save(Some("s"), None, &draft("local input")).unwrap();
    let mut c = recover_saved_scope(root.path(), Some("s"));
    let mut other = recover_saved_scope(root.path(), Some("s"));
    other.state.editor.select_all();
    other.paste("other terminal input".into());
    assert!(other.persist_draft());
    c.state.editor.move_cursor(CursorMove::End);
    c.paste(" with edits".into());
    let attachment = Attachment::Text {
        name: "context.txt".into(),
        uri: "file:///context.txt".into(),
        text: "local snapshot".into(),
    };
    c.state.attachments.push(attachment.clone());
    assert!(!c.persist_draft());

    assert!(key(&mut c, KeyCode::Enter, KeyModifiers::NONE).is_empty());
    assert_eq!(c.state.phase, Phase::Ready);
    assert_eq!(c.state.editor.lines(), &["local input with edits"]);
    assert_eq!(c.state.attachments, [attachment]);
    assert!(c.state.queue_paused);
    assert!(!c
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(_))));
    assert_eq!(
        store
            .load(Some("s"))
            .unwrap()
            .unwrap()
            .snapshot
            .composer
            .text,
        "other terminal input"
    );
}

#[test]
fn unchanged_input_retries_after_a_transient_save_failure() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    c.paste("first save".into());
    assert!(c.persist_draft());
    let saved_file = std::fs::read_dir(root.path().join(".aster/tui-drafts"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .unwrap();
    let blocker = saved_file.with_extension("tmp");
    std::fs::create_dir(&blocker).unwrap();
    c.paste(" and the final edit".into());
    assert!(!c.persist_draft());
    assert!(!c.autosave_draft());
    assert!(!c.persist_draft());
    assert_eq!(
        c.state
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::Notice { error: true, .. }))
            .count(),
        1
    );
    std::fs::remove_dir(&blocker).unwrap();
    // Explicit flush (also used before exit) retries the exact same snapshot.
    assert!(c.persist_draft());
    let restored = recover_latest(root.path());
    assert_eq!(
        restored.state.editor.lines(),
        &["first save and the final edit"]
    );
}

fn accepted_correction(c: &mut Controller, text: &str) -> Value {
    c.paste(text.to_owned());
    let action = only(key(c, KeyCode::Char('s'), KeyModifiers::ALT));
    let Action::Request {
        pending, params, ..
    } = action
    else {
        panic!("steering RPC")
    };
    c.registered(22, pending);
    c.message(json!({"id":22,"result":{"messageId":params["messageId"],"turnId":params["turnId"],"state":"accepted"}}));
    params
}

#[test]
fn acknowledged_unapplied_correction_survives_disconnect_once_with_raw_whitespace() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    c.input("first task".into());
    let text = "  correction 中文\n    keep indentation\n";
    accepted_correction(&mut c, text);
    assert_eq!(c.state.editor.lines(), &[""]);
    c.failure("connection lost");
    assert_eq!(c.state.editor.lines().join("\n"), text);
    assert!(c.state.queued.is_empty());
    let recovered = recover_latest(root.path());
    assert_eq!(recovered.state.editor.lines().join("\n"), text);
    assert!(recovered.state.queued.is_empty());
    assert_eq!(recovered.state.recovery.unwrap().text, "first task");
}

#[test]
fn terminal_correction_notification_before_ack_cannot_erase_its_recovered_draft() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    c.input("first task".into());
    c.paste("/steer retain me".into());
    let action = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    let Action::Request {
        pending, params, ..
    } = action
    else {
        panic!("steering RPC")
    };
    c.registered(22, pending);
    c.message(
        json!({"method":"session/update","params":{"sessionId":"s","update":{
            "sessionUpdate":"session_info_update","_meta":{"aster":{"steering":{
                "messageId":params["messageId"],"turnId":params["turnId"],"state":"turn_finished"
            }}}
        }}}),
    );
    c.message(json!({"id":22,"result":{"messageId":params["messageId"],"turnId":params["turnId"],"state":"accepted"}}));
    assert_eq!(c.state.editor.lines(), &["/steer retain me"]);
    assert_eq!(
        c.state.steering_messages[0].state,
        aster_tui::state::SteeringState::TurnFinished
    );
    assert!(c.state.queue_paused);
}

#[test]
fn reconnecting_same_session_does_not_close_the_loaded_runtime() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    let action = only(c.input("completed task".into()));
    rpc(&mut c, action, json!({"stopReason":"end_turn"}));
    c.failure("disconnect while idle");
    c.reconnected();
    let listing = initialize(&mut c);
    let loading = only(rpc(
        &mut c,
        listing,
        json!({"sessions":[{"sessionId":"s"}]}),
    ));
    assert!(rpc(&mut c, loading, json!({})).is_empty());
    assert_eq!(c.state.session_id.as_deref(), Some("s"));
    assert_eq!(c.state.phase, Phase::Ready);
}

#[test]
fn browsing_input_history_preserves_the_hidden_unsent_editor_on_restart() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    for text in ["old submission 1", "old submission 2"] {
        let action = only(c.input(text.into()));
        rpc(&mut c, action, json!({"stopReason":"end_turn"}));
    }
    c.paste("new unsent input 中文".into());
    c.state.editor.move_cursor(CursorMove::Jump(0, 4));
    let cursor = c.state.editor.cursor();
    key(&mut c, KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(c.state.editor.lines(), &["old submission 2"]);
    assert!(c.persist_draft());
    key(&mut c, KeyCode::Up, KeyModifiers::NONE);
    assert_eq!(c.state.editor.lines(), &["old submission 1"]);
    assert!(c.persist_draft());
    let recovered = recover_latest(root.path());
    assert_eq!(recovered.state.editor.lines(), &["new unsent input 中文"]);
    assert_eq!(recovered.state.editor.cursor(), cursor);
}

fn git(root: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

async fn run_rollback(c: &mut Controller, action: Action, root: &Path) -> Vec<Action> {
    let Action::Rollback { job, generation } = action else {
        panic!("file operation")
    };
    let outcome = match job {
        RollbackJob::Prepare { file, hunk } => rollback::prepare(root, &file, hunk)
            .await
            .map(RollbackOutcome::Prepared),
        RollbackJob::Apply(plan) => rollback::apply(plan).await.map(RollbackOutcome::Applied),
        RollbackJob::FindUndo => rollback::latest(root).await.map(RollbackOutcome::UndoReady),
        RollbackJob::Undo(receipt) => rollback::undo(&receipt)
            .await
            .map(|()| RollbackOutcome::Undone(receipt.path)),
    };
    assert!(outcome.is_ok(), "file operation failed");
    c.rollback_result(generation, outcome)
}

#[tokio::test]
async fn file_hunk_rollback_requires_review_and_supports_durable_undo() {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "--quiet"]);
    git(root.path(), &["config", "core.autocrlf", "false"]);
    let path = root.path().join("example.txt");
    let original = (1..=30).map(|n| format!("line {n}\n")).collect::<String>();
    std::fs::write(&path, &original).unwrap();
    git(root.path(), &["add", "example.txt"]);
    git(
        root.path(),
        &[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    );
    let modified = original
        .replace("line 2\n", "first edit\n")
        .replace("line 25\n", "second edit\n");
    std::fs::write(&path, &modified).unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    c.state.workspace_changes = Some(workspace_changes::load(root.path()).await.unwrap());
    c.state.open(Overlay::Changes);
    assert!(key(&mut c, KeyCode::Char('u'), KeyModifiers::CONTROL).is_empty());
    assert_eq!(c.state.overlay, Some(Overlay::Rollback));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), modified);
    // First choice is whole-file scope, but the final dialog defaults to Cancel.
    let preview = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    run_rollback(&mut c, preview, root.path()).await;
    assert_eq!(c.state.choices()[c.state.selected].value, "cancel");
    assert!(key(&mut c, KeyCode::Enter, KeyModifiers::NONE).is_empty());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), modified);
    // Choose only the second hunk, review its inverse patch, then explicitly apply.
    c.state.open(Overlay::Changes);
    key(&mut c, KeyCode::Char('u'), KeyModifiers::CONTROL);
    c.state.selected = 2;
    let preview = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    run_rollback(&mut c, preview, root.path()).await;
    assert!(c
        .state
        .rollback
        .plan
        .as_ref()
        .unwrap()
        .preview
        .contains("-second edit"));
    c.state.selected = 1;
    let apply = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    run_rollback(&mut c, apply, root.path()).await;
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        original.replace("line 2\n", "first edit\n")
    );
    let latest = only(c.input("/undo".into()));
    run_rollback(&mut c, latest, root.path()).await;
    assert_eq!(c.state.choices()[c.state.selected].value, "cancel");
    c.state.selected = 1;
    let undo = only(key(&mut c, KeyCode::Enter, KeyModifiers::NONE));
    run_rollback(&mut c, undo, root.path()).await;
    assert_eq!(std::fs::read_to_string(&path).unwrap(), modified);
    assert!(rollback::latest(root.path()).await.unwrap().is_none());
    assert!(c.state.queue_paused);
}

#[test]
fn rollback_and_undo_are_blocked_while_foreground_or_background_work_runs() {
    let root = tempfile::tempdir().unwrap();
    let mut c = controller(root.path(), None);
    prepare(&mut c, "s");
    let action = only(c.input("running task".into()));
    assert!(c.input("/undo".into()).is_empty());
    rpc(&mut c, action, json!({"stopReason":"end_turn"}));
    c.state.tasks = json!({"jobs":[{"id":"job-1","status":"running","canStop":true}]});
    assert!(c.input("/undo".into()).is_empty());
    assert_eq!(c.state.phase, Phase::Ready);
    assert!(!c.state.rollback.loading);
}

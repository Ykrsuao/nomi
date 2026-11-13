// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exercise session controls through the same controller and wire boundaries as the frontend.

use std::path::Path;

use aster_tui::attachments::{load_file, Attachment};
use aster_tui::controller::{Action, Controller};
use aster_tui::state::{Overlay, Phase, State};
use base64::Engine;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use serde_json::{json, Value};

struct Client {
    controller: Controller,
    next_id: u64,
}

impl Client {
    fn ready() -> Self {
        let mut client = Self {
            controller: Controller::new(
                State::new("C:/test-workspace".into(), "model-a".into(), false),
                None,
                false,
            ),
            next_id: 1,
        };
        let action = client.controller.initialize();
        let (id, _) = client.register(vec![action], "initialize");
        let actions = client.answer(
            id,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}},
                    "promptCapabilities": {"image": true}
                },
                "_meta": {"aster": {
                    "attachments": true, "runtimeControls": true, "structuredProgress": true
                }}
            }),
        );
        let (id, _) = client.register(actions, "session/list");
        let actions = client.answer(id, json!({"sessions": []}));
        let (id, _) = client.register(actions, "session/new");
        let actions = client.answer(id, json!({"sessionId": "session-a"}));
        let (id, _) = client.register(actions, "_aster/session/runtime");
        assert!(client
            .answer(id, runtime("model-a", "code", "ask"))
            .is_empty());
        assert_eq!(client.controller.state.phase, Phase::Ready);
        client
    }

    fn register(&mut self, actions: Vec<Action>, expected_method: &str) -> (u64, Value) {
        assert_eq!(
            actions.len(),
            1,
            "expected one {expected_method} request: {actions:?}"
        );
        let Action::Request {
            method,
            params,
            pending,
        } = actions.into_iter().next().unwrap()
        else {
            panic!("expected a {expected_method} request");
        };
        assert_eq!(method, expected_method);
        let id = self.next_id;
        self.next_id += 1;
        self.controller.registered(id, pending);
        (id, params)
    }

    fn answer(&mut self, id: u64, result: Value) -> Vec<Action> {
        self.controller.message(json!({"id": id, "result": result}))
    }

    fn start_prompt(&mut self) -> u64 {
        let actions = self.controller.input("Inspect the supplied context".into());
        self.register(actions, "session/prompt").0
    }

    fn update(&mut self, update: Value) {
        assert!(self
            .controller
            .message(json!({
                "method": "session/update",
                "params": {"sessionId": "session-a", "update": update}
            }))
            .is_empty());
    }
}

fn key(controller: &mut Controller, code: KeyCode) -> Vec<Action> {
    controller.key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn text_attachment(uri: &str, text: &str) -> Attachment {
    Attachment::Text {
        name: "README.md".into(),
        uri: uri.into(),
        text: text.into(),
    }
}

fn clipboard_image() -> Attachment {
    let bytes = base64::engine::general_purpose::STANDARD.decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1QAAAAASUVORK5CYII="
    ).unwrap();
    Attachment::from_image("clipboard.png", &bytes).unwrap()
}

fn runtime(model: &str, mode: &str, approval: &str) -> Value {
    json!({
        "model": model,
        "mode": mode,
        "approval": approval,
        "executionBackend": "native",
        "models": [{"id": "model-a", "name": "Model A"}, {"id": "model-b", "name": "Model B"}],
        "modes": [{"id": "code", "name": "Code"}, {"id": "plan", "name": "Plan"}]
    })
}

fn permission(id: &str, session: &str) -> Value {
    json!({
        "id": id,
        "method": "session/request_permission",
        "params": {
            "sessionId": session,
            "toolCall": {"toolCallId": "cell-1", "title": "Execute Python", "rawInput": {"code": "print('checked')"}},
            "options": [
                {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "allow_session", "name": "Allow this session", "kind": "allow_always"},
                {"optionId": "reject", "name": "Reject", "kind": "reject_once"}
            ]
        }
    })
}

fn assert_cancelled_permission(actions: &[Action], expected_id: &str) {
    assert!(
        actions.iter().any(|action| matches!(action,
            Action::Reply { id, result }
            if id == expected_id && result == &json!({"outcome": {"outcome": "cancelled"}})
        )),
        "expected cancellation for {expected_id}: {actions:?}"
    );
}

#[test]
fn completing_a_prompt_dismisses_its_pending_execution_approval() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    assert!(client
        .controller
        .permission_request(&permission("approval-1", "session-a"))
        .is_empty());
    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));

    let actions = client.answer(prompt, json!({"stopReason": "end_turn"}));

    assert_cancelled_permission(&actions, "approval-1");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[tokio::test]
async fn same_named_files_keep_both_reviewed_snapshots_when_sent() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("one")).unwrap();
    std::fs::create_dir(directory.path().join("two")).unwrap();
    std::fs::write(directory.path().join("one/README.md"), "first snapshot").unwrap();
    std::fs::write(directory.path().join("two/README.md"), "second snapshot").unwrap();
    let mut first = load_file(directory.path(), Path::new("one/README.md"))
        .await
        .unwrap();
    let mut second = load_file(directory.path(), Path::new("two/README.md"))
        .await
        .unwrap();
    assert_eq!(first.name(), "one/README.md");
    assert_eq!(second.name(), "two/README.md");
    // Relative labels distinguish these files, but display labels are not identity.
    // Keep the real file snapshots and make the labels coincide to cover deduplication.
    for attachment in [&mut first, &mut second] {
        let Attachment::Text { name, .. } = attachment else {
            panic!("expected a text snapshot");
        };
        *name = "README.md".into();
    }

    let mut client = Client::ready();
    let generation = client.controller.state.conversation_generation;
    client.controller.add_attachment(generation, Ok(first));
    client.controller.add_attachment(generation, Ok(second));
    // Sending must use the content the user attached, even if its source changes.
    std::fs::write(directory.path().join("one/README.md"), "later edit").unwrap();
    client.controller.paste("Compare both files".into());
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");

    assert_eq!(params["prompt"].as_array().unwrap().len(), 3);
    assert_eq!(params["prompt"][1]["resource"]["text"], "first snapshot");
    assert_eq!(params["prompt"][2]["resource"]["text"], "second snapshot");
    assert_ne!(
        params["prompt"][1]["resource"]["uri"],
        params["prompt"][2]["resource"]["uri"]
    );
    assert!(client.controller.state.attachments.is_empty());
}

#[test]
fn pasted_image_cannot_release_the_pending_file_send_guard() {
    let mut client = Client::ready();
    let actions = client.controller.input("/attach pending.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read: {actions:?}");
    };
    let generation = *generation;
    client.controller.paste("Review both attachments".into());
    client
        .controller
        .add_attachment(generation, Ok(clipboard_image()));

    assert!(client.controller.state.attachment_loading);
    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["Review both attachments"]
    );
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/pending.txt",
            "complete file",
        )),
    );
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");

    assert!(!client.controller.state.attachment_loading);
    assert_eq!(params["prompt"].as_array().unwrap().len(), 3);
    assert_eq!(params["prompt"][1]["type"], "image");
    assert_eq!(params["prompt"][1]["mimeType"], "image/png");
    assert_eq!(params["prompt"][2]["resource"]["text"], "complete file");
}

#[test]
fn backend_slash_prompts_wait_for_the_selected_file_before_sending() {
    let mut client = Client::ready();
    client.update(json!({
        "sessionUpdate": "available_commands_update",
        "availableCommands": [{"name": "review", "description": "Review supplied context"}]
    }));
    assert!(client
        .controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/review"));
    let actions = client.controller.input("/attach pending.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read");
    };
    let generation = *generation;
    client.controller.paste("/review attached context".into());

    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["/review attached context"]
    );
    assert!(client.controller.state.attachment_loading);
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/pending.txt",
            "complete file",
        )),
    );
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");
    assert_eq!(params["prompt"][0]["text"], "/review attached context");
    assert_eq!(params["prompt"][1]["resource"]["text"], "complete file");
}

#[test]
fn file_completion_from_another_conversation_does_not_attach_to_the_new_one() {
    let mut client = Client::ready();
    let actions = client.controller.input("/attach old.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read");
    };
    let generation = *generation;
    let actions = client.controller.resume("session-b".into());
    client.register(actions, "session/load");
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/old.txt",
            "old conversation",
        )),
    );

    assert!(client.controller.state.attachments.is_empty());
    assert!(!client.controller.state.attachment_loading);
}

#[test]
fn approval_choices_reply_to_the_request_and_keep_the_running_prompt_and_draft() {
    for (choice, expected) in [
        (KeyCode::Char('1'), "allow_once"),
        (KeyCode::Char('2'), "allow_session"),
        (KeyCode::Char('3'), "reject"),
        (KeyCode::Enter, "allow_once"),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        let turn = client.controller.state.active_turn_id.clone();
        client.controller.paste("unfinished message".into());
        client
            .controller
            .permission_request(&permission("approval-1", "session-a"));
        assert_eq!(
            client.controller.state.approvals[0].code,
            "print('checked')"
        );

        let actions = key(&mut client.controller, choice);

        assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
            if id == "approval-1" && result == &json!({"outcome": {"outcome": "selected", "optionId": expected}})
        ));
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(
            client.controller.state.editor.lines(),
            &["unfinished message"]
        );
        assert!(client.controller.state.approvals.is_empty());
        assert_eq!(client.controller.state.overlay, None);
    }
}

#[test]
fn approval_default_uses_option_kind_and_falls_back_to_reject() {
    for (kinds, expected) in [
        (
            vec!["allow_always", "reject_once", "allow_once"],
            "option-2",
        ),
        (vec!["allow_always", "reject_once"], "option-1"),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        let mut request = permission("approval-1", "session-a");
        request["params"]["options"] = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                json!({"optionId": format!("option-{index}"), "name": kind, "kind": kind})
            })
            .collect();
        assert!(client.controller.permission_request(&request).is_empty());

        let actions = key(&mut client.controller, KeyCode::Enter);

        assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
            if id == "approval-1" && result == &json!({"outcome": {"outcome": "selected", "optionId": expected}})
        ));
        assert!(client.controller.state.approvals.is_empty());
    }
}

#[test]
fn approval_key_repeats_do_not_confirm_the_next_request() {
    for code in [
        KeyCode::Enter,
        KeyCode::Char('1'),
        KeyCode::Char('2'),
        KeyCode::Char('3'),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        for id in ["first", "second"] {
            assert!(client
                .controller
                .permission_request(&permission(id, "session-a"))
                .is_empty());
        }
        let actions = key(&mut client.controller, code);
        assert!(matches!(actions.as_slice(), [Action::Reply { id, .. }] if id == "first"));

        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            assert!(client
                .controller
                .key(KeyEvent::new_with_kind(code, KeyModifiers::NONE, kind))
                .is_empty());
        }
        assert_eq!(client.controller.state.approvals.len(), 1);
        assert_eq!(client.controller.state.approvals[0].id, "second");

        let actions = key(&mut client.controller, code);
        assert!(matches!(actions.as_slice(), [Action::Reply { id, .. }] if id == "second"));
        assert!(client.controller.state.approvals.is_empty());
    }
}

#[test]
fn escape_rejects_the_execution_request_without_cancelling_the_whole_prompt() {
    let mut client = Client::ready();
    client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    let actions = key(&mut client.controller, KeyCode::Esc);

    assert_eq!(actions.len(), 1);
    assert_cancelled_permission(&actions, "approval-1");
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.phase, Phase::Busy);
}

#[test]
fn permissions_for_another_session_or_a_finished_turn_are_rejected() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let actions = client
        .controller
        .permission_request(&permission("wrong-session", "session-b"));
    assert_cancelled_permission(&actions, "wrong-session");
    assert!(client.controller.state.approvals.is_empty());
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());

    let actions = client
        .controller
        .permission_request(&permission("late-request", "session-a"));

    assert_cancelled_permission(&actions, "late-request");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn failed_or_cancelled_prompts_dismiss_pending_approvals_and_pause_queued_work() {
    for result in [
        json!({"stopReason": "cancelled"}),
        json!({"stopReason": "max_tokens"}),
    ] {
        let mut client = Client::ready();
        let prompt = client.start_prompt();
        client.controller.paste("keep this draft".into());
        client
            .controller
            .permission_request(&permission("approval-1", "session-a"));

        let actions = client.answer(prompt, result);

        assert_cancelled_permission(&actions, "approval-1");
        assert!(client.controller.state.approvals.is_empty());
        assert_eq!(client.controller.state.overlay, None);
        assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
        assert!(client.controller.state.queue_paused);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn a_prompt_error_dismisses_its_permission_instead_of_leaving_a_stale_dialog() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    let actions = client
        .controller
        .message(json!({"id": prompt, "error": {"code": -32603, "message": "provider stopped"}}));

    assert_cancelled_permission(&actions, "approval-1");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn a_disconnected_backend_cannot_leave_or_open_an_execution_dialog() {
    let mut client = Client::ready();
    client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    client.controller.failure("Protocol stream closed");
    let actions = client
        .controller
        .permission_request(&permission("after-close", "session-a"));

    assert_cancelled_permission(&actions, "after-close");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Failed);
}

#[test]
fn changing_runtime_settings_uses_the_backend_result_before_showing_the_new_value() {
    for (command, field, value, model, mode, approval) in [
        (
            "/model model-b",
            "model",
            "model-b",
            "model-b",
            "code",
            "ask",
        ),
        ("/mode plan", "mode", "plan", "model-a", "plan", "ask"),
        (
            "/permissions session",
            "approval",
            "session",
            "model-a",
            "code",
            "session",
        ),
    ] {
        let mut client = Client::ready();
        let actions = client.controller.input(command.into());
        let (id, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params["sessionId"], "session-a");
        assert_eq!(params[field], value);
        assert_eq!(client.controller.state.model, "model-a");
        assert_eq!(client.controller.state.mode, "code");
        assert_eq!(client.controller.state.approval_policy, "ask");

        assert!(client.answer(id, runtime(model, mode, approval)).is_empty());

        assert_eq!(client.controller.state.model, model);
        assert_eq!(client.controller.state.mode, mode);
        assert_eq!(client.controller.state.approval_policy, approval);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn an_active_prompt_rejects_runtime_changes_without_losing_the_composer() {
    let mut client = Client::ready();
    client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    client.controller.paste("next instruction".into());

    for command in ["/model model-b", "/mode plan", "/permissions session"] {
        assert!(client.controller.input(command.into()).is_empty());
        assert_eq!(client.controller.state.model, "model-a");
        assert_eq!(client.controller.state.mode, "code");
        assert_eq!(client.controller.state.approval_policy, "ask");
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(
            client.controller.state.editor.lines(),
            &["next instruction"]
        );
    }
}

#[test]
fn a_runtime_inspection_response_does_not_finish_or_replace_the_active_prompt() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    let started = client.controller.state.started;
    client.controller.paste("keep editing".into());
    let actions = client.controller.input("/model".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");

    assert!(client
        .answer(inspection, runtime("model-a", "code", "ask"))
        .is_empty());

    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.started, started);
    assert_eq!(client.controller.state.editor.lines(), &["keep editing"]);
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn failed_runtime_configuration_retains_the_effective_settings() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model model-b".into());
    let (configuration, _) = client.register(actions, "_aster/session/configure");

    assert!(client
        .controller
        .message(json!({
            "id": configuration,
            "error": {"code": -32602, "message": "Model is unavailable"}
        }))
        .is_empty());

    assert_eq!(client.controller.state.model, "model-a");
    assert_eq!(client.controller.state.mode, "code");
    assert_eq!(client.controller.state.approval_policy, "ask");
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn runtime_replies_from_before_a_conversation_switch_are_ignored() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");
    key(&mut client.controller, KeyCode::Esc);
    let actions = client.controller.resume("session-b".into());
    client.register(actions, "session/load");
    let settings = client.controller.state.runtime.clone();
    let mode = client.controller.state.mode.clone();
    let approval = client.controller.state.approval_policy.clone();

    assert!(client
        .answer(inspection, runtime("model-b", "plan", "session"))
        .is_empty());

    assert_eq!(client.controller.state.runtime, settings);
    assert_eq!(client.controller.state.mode, mode);
    assert_eq!(client.controller.state.approval_policy, approval);
    assert_eq!(client.controller.state.phase, Phase::Busy);
}

fn task_snapshot(jobs: Value) -> Value {
    json!({
        "steps": [{"id": "review", "title": "Review parser", "status": "in_progress", "blockedBy": []}],
        "currentTask": "Review parser", "currentTaskId": "review",
        "waitReason": "Waiting for checks", "jobs": jobs, "truncated": false
    })
}

fn publish_tasks(client: &mut Client, snapshot: Value) {
    client.update(json!({
        "sessionUpdate": "plan",
        "entries": [{"content": "Review parser", "priority": "medium", "status": "in_progress"}],
        "_meta": {"aster": {"tasks": snapshot}}
    }));
}

#[test]
fn stopping_a_selected_background_job_keeps_other_jobs_and_the_prompt_running() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    let snapshot = task_snapshot(json!([
        {"id": "check-job", "channel": "checks", "title": "Check parser", "status": "running", "canStop": true},
        {"id": "preview-job", "channel": "preview", "title": "Render preview", "status": "running", "canStop": true}
    ]));
    publish_tasks(&mut client, snapshot.clone());
    assert!(client.controller.state.status.contains("Review parser"));
    assert!(client
        .controller
        .state
        .status
        .contains("Waiting for checks"));
    let actions = client.controller.input("/tasks".into());
    let (inspection, _) = client.register(actions, "_aster/session/tasks");
    assert!(client.answer(inspection, snapshot).is_empty());
    assert_eq!(
        client
            .controller
            .state
            .choices()
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        ["step:review", "job:check-job", "job:preview-job"]
    );
    key(&mut client.controller, KeyCode::Down);

    let actions = client
        .controller
        .key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
    let (stop, params) = client.register(actions, "_aster/session/stop_task");
    assert_eq!(
        params,
        json!({"sessionId": "session-a", "taskId": "check-job"})
    );
    let after = task_snapshot(json!([
        {"id": "check-job", "channel": "checks", "title": "Check parser", "status": "cancelled", "canStop": false},
        {"id": "preview-job", "channel": "preview", "title": "Render preview", "status": "running", "canStop": true}
    ]));
    assert!(client
        .answer(
            stop,
            json!({"id": "check-job", "stopped": true, "tasks": after})
        )
        .is_empty());

    assert_eq!(
        client.controller.state.tasks["jobs"][0]["status"],
        "cancelled"
    );
    assert_eq!(
        client.controller.state.tasks["jobs"][1]["id"],
        "preview-job"
    );
    assert_eq!(client.controller.state.tasks["jobs"][1]["canStop"], true);
    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn finished_or_replaced_job_ids_cannot_stop_a_replacement_with_the_same_channel() {
    let mut client = Client::ready();
    client.start_prompt();
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "old-job", "channel": "checks", "title": "Old check", "status": "running", "canStop": true}
        ])),
    );
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "new-job", "channel": "checks", "title": "New check", "status": "running", "canStop": true},
            {"id": "done-job", "channel": "preview", "title": "Preview", "status": "done", "canStop": false}
        ])),
    );

    for command in ["/stop old-job", "/stop done-job", "/stop checks"] {
        assert!(client.controller.input(command.into()).is_empty());
    }
    assert_eq!(client.controller.state.phase, Phase::Busy);
    let actions = client.controller.input("/stop new-job".into());
    let (_, params) = client.register(actions, "_aster/session/stop_task");
    assert_eq!(params["taskId"], "new-job");
}

#[test]
fn a_running_background_job_can_request_approval_after_the_prompt_finishes() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "background", "channel": "checks", "title": "Check parser", "status": "running", "canStop": true}
        ])),
    );
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());

    assert!(client
        .controller
        .permission_request(&permission("background-approval", "session-a"))
        .is_empty());

    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));
    let actions = key(&mut client.controller, KeyCode::Char('1'));
    assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
        if id == "background-approval" && result["outcome"]["optionId"] == "allow_once"
    ));
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn task_progress_does_not_hide_a_pending_execution_decision() {
    let mut client = Client::ready();
    client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    publish_tasks(&mut client, task_snapshot(json!([])));

    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));
    assert_eq!(
        client.controller.state.status,
        "Waiting for execution approval"
    );
    assert_eq!(client.controller.state.approvals[0].id, "approval-1");
    assert_eq!(client.controller.state.choices()[2].value, "reject");
}

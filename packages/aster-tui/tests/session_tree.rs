// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Browse and navigate branches through the same keys and RPC boundary as the TUI.

use std::path::Path;
use std::time::Duration;

use aster_tui::{
    attachments::Attachment,
    controller::{Action, Controller},
    drafts::{DraftSnapshot, DraftStore},
    input_history::InputHistory,
    session_tree::SessionTree,
    state::{Entry, Overlay, Phase, State},
    ui,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::{json, Value};

struct Client {
    controller: Controller,
    next_id: u64,
}

impl Client {
    fn ready(root: &Path, supported: bool) -> Self {
        Self::ready_with_search(root, supported, false)
    }

    fn ready_with_search(root: &Path, supported: bool, search: bool) -> Self {
        let mut controller = Controller::new(
            State::new(root.to_string_lossy().into_owned(), "test".into(), false),
            None,
            false,
        );
        controller.attach_drafts(DraftStore::for_workspace(root).unwrap());
        let mut client = Self {
            controller,
            next_id: 1,
        };
        let action = client.controller.initialize();
        let (id, _) = client.register(vec![action], "initialize");
        let actions = client.answer(
            id,
            json!({
                "protocolVersion":1,
                "agentCapabilities":{"loadSession":true,"sessionCapabilities":{"list":{},"close":{}}},
                "_meta":{"aster":{"sessionTree":supported,"sessionTreeSearch":search}}
            }),
        );
        let (id, _) = client.register(actions, "session/list");
        let actions = client.answer(id, json!({"sessions":[]}));
        let (id, _) = client.register(actions, "session/new");
        assert!(client.answer(id, json!({"sessionId":"current"})).is_empty());
        client.controller.state.entries = vec![
            Entry::User("root request".into()),
            Entry::Assistant("original response".into()),
        ];
        assert_eq!(client.controller.state.phase, Phase::Ready);
        client
    }

    fn register(&mut self, actions: Vec<Action>, method: &str) -> (u64, Value) {
        assert_eq!(actions.len(), 1, "expected {method}: {actions:?}");
        let Action::Request {
            method: actual,
            params,
            pending,
        } = actions.into_iter().next().unwrap()
        else {
            panic!("expected request {method}");
        };
        assert_eq!(actual, method);
        let id = self.next_id;
        self.next_id += 1;
        self.controller.registered(id, pending);
        (id, params)
    }

    fn answer(&mut self, id: u64, result: Value) -> Vec<Action> {
        self.controller.message(json!({"id":id,"result":result}))
    }

    fn fail(&mut self, id: u64) -> Vec<Action> {
        self.controller
            .message(json!({"id":id,"error":{"code":-32603,"message":"fixture failure"}}))
    }

    fn key(&mut self, code: KeyCode) -> Vec<Action> {
        self.controller.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn open_tree(&mut self) {
        let actions = self.controller.input("/tree".into());
        let (id, params) = self.register(actions, "_aster/session/tree");
        assert_eq!(params["sessionId"], "current");
        assert!(self.controller.state.tree_loading);
        assert!(self.answer(id, tree()).is_empty());
    }

    fn select(&mut self, node_id: &str) -> Vec<Action> {
        self.controller.state.selected = self
            .controller
            .state
            .choices()
            .iter()
            .position(|choice| choice.value == node_id)
            .unwrap();
        self.key(KeyCode::Enter)
    }

    fn search(&mut self, query: &str) -> (u64, Value) {
        self.controller.paste(query.into());
        assert!(self.controller.state.choices().is_empty());
        assert!(self.key(KeyCode::Enter).is_empty());
        std::thread::sleep(Duration::from_millis(200));
        let actions = self.controller.tick();
        self.register(actions, "_aster/session/tree")
    }

    fn replay_partial(&mut self) {
        assert!(self.controller.message(json!({"method":"session/update","params":{
            "sessionId":"fork",
            "update":{"sessionUpdate":"agent_message_chunk","content":{"text":"partial failed replay"}}
        }})).is_empty());
    }

    fn finish_load(&mut self, actions: Vec<Action>, target: &str) -> Vec<Action> {
        let (load, params) = self.register(actions, "session/load");
        assert_eq!(params["sessionId"], target);
        self.controller
            .message(json!({"method":"session/update","params":{
                "sessionId":target,
                "update":{"sessionUpdate":"user_message_chunk","content":{"text":"root request"}}
            }}));
        self.controller.message(json!({"method":"session/update","params":{
            "sessionId":target,
            "update":{"sessionUpdate":"agent_message_chunk","content":{"text":"restored response"}}
        }}));
        let actions = self.answer(load, json!({}));
        let (close, params) = self.register(actions, "session/close");
        assert_eq!(params["sessionId"], "current");
        self.answer(close, json!({}))
    }
}

fn tree() -> Value {
    json!({
        "nodes":[
            {"id":"u1","parentId":null,"role":"user","preview":"root request","navigable":true},
            {"id":"a1","parentId":"u1","role":"assistant","preview":"shared reply","navigable":true},
            {"id":"u2","parentId":"a1","role":"user","preview":"small patch 中文","navigable":true},
            {"id":"a2","parentId":"u2","role":"assistant","preview":"old branch reply","navigable":true},
            {"id":"u3","parentId":"a1","role":"user","preview":"alternative plan","navigable":true},
            {"id":"a3","parentId":"u3","role":"assistant","preview":"current branch reply","navigable":true}
        ],
        "currentNodeId":"a3"
    })
}

fn search_result(query: &str, matches: &[&str]) -> Value {
    let mut result = tree();
    result["query"] = json!(query);
    result["matchedNodeIds"] = json!(matches);
    result
}

fn assert_original_transcript(client: &Client) {
    assert_eq!(
        client.controller.state.session_id.as_deref(),
        Some("current")
    );
    assert!(client
        .controller
        .state
        .entries
        .iter()
        .any(|entry| { matches!(entry, Entry::Assistant(text) if text == "original response") }));
    assert!(!client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Assistant(text) if text == "partial failed replay")
    }));
}

#[test]
fn tree_search_keeps_ancestry_order_and_navigates_by_stable_message_id() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.open_tree();
    let choices = client.controller.state.choices();
    assert_eq!(choices[client.controller.state.selected].value, "a3");
    assert!(choices[5].label.contains('●'));
    assert!(choices[2].label.contains('├'));
    assert!(choices[4].label.contains('└'));
    client.controller.paste("small patch".into());
    let choices = client.controller.state.choices();
    assert_eq!(
        choices
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        vec!["u1", "a1", "u2"]
    );
    assert!(client.key(KeyCode::Down).is_empty());
    assert!(client.key(KeyCode::Down).is_empty());
    let actions = client.key(KeyCode::Enter);
    let (_, params) = client.register(actions, "_aster/session/navigate");
    assert_eq!(
        params,
        json!({"sessionId":"current","nodeId":"u2","expectedCurrentNodeId":"a3"})
    );
}

#[test]
fn full_message_search_finds_an_inactive_sibling_and_keeps_its_ancestors() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready_with_search(root.path(), true, true);
    client.open_tree();
    let query = "Beyond Preview 中文";
    let (search, params) = client.search(query);
    assert_eq!(params, json!({"sessionId":"current","query":query}));
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert!(render(&mut client.controller.state, 80, 24).contains("Searching complete messages"));
    assert!(client
        .answer(search, search_result(query, &["a2"]))
        .is_empty());
    assert_eq!(
        client
            .controller
            .state
            .choices()
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        vec!["u1", "a1", "u2", "a2"]
    );
    assert_eq!(
        client
            .controller
            .state
            .session_tree
            .current_node_id
            .as_deref(),
        Some("a3")
    );
    let actions = client.select("a2");
    let (_, params) = client.register(actions, "_aster/session/navigate");
    assert_eq!(params["nodeId"], "a2");
    assert_eq!(params["expectedCurrentNodeId"], "a3");
}

#[test]
fn full_message_search_serializes_requests_and_ignores_replaced_queries() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready_with_search(root.path(), true, true);
    client.open_tree();
    let (first, _) = client.search("old");
    client.controller.paste(" newer".into());
    std::thread::sleep(Duration::from_millis(200));
    assert!(client.controller.tick().is_empty());
    assert!(client
        .answer(first, search_result("old", &["a2"]))
        .is_empty());
    assert!(client.controller.state.choices().is_empty());
    assert!(client.key(KeyCode::Enter).is_empty());
    let actions = client.controller.tick();
    let (second, params) = client.register(actions, "_aster/session/tree");
    assert_eq!(params["query"], "old newer");
    assert!(client
        .answer(second, search_result("old newer", &["u3"]))
        .is_empty());
    let choices = client.controller.state.choices();
    assert_eq!(
        choices
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        vec!["u1", "a1", "u3"]
    );
    assert_eq!(client.controller.state.selected, 0);
    for _ in 0.."old newer".len() {
        assert!(client.key(KeyCode::Backspace).is_empty());
    }
    assert_eq!(client.controller.state.choices().len(), 6);
    assert!(client.controller.tick().is_empty());
}

#[test]
fn clearing_or_reopening_a_tree_invalidates_in_flight_search_results() {
    for reopen in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready_with_search(root.path(), true, true);
        client.open_tree();
        let (search, _) = client.search("old");
        if reopen {
            assert!(client.key(KeyCode::Esc).is_empty());
            client.open_tree();
            client.controller.paste("old".into());
        } else {
            for _ in 0..3 {
                assert!(client.key(KeyCode::Backspace).is_empty());
            }
        }
        assert!(client
            .answer(search, search_result("old", &["a2"]))
            .is_empty());
        if reopen {
            assert!(client.controller.state.choices().is_empty());
            assert!(client.key(KeyCode::Enter).is_empty());
            std::thread::sleep(Duration::from_millis(200));
            let actions = client.controller.tick();
            let (fresh, _) = client.register(actions, "_aster/session/tree");
            assert!(client
                .answer(fresh, search_result("old", &["a3"]))
                .is_empty());
            assert_eq!(
                client.controller.state.choices().last().unwrap().value,
                "a3"
            );
        } else {
            assert_eq!(client.controller.state.choices().len(), 6);
            assert!(client.controller.tick().is_empty());
        }
    }
}

#[test]
fn search_errors_and_stale_sessions_do_not_change_foreground_state() {
    for stale in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready_with_search(root.path(), true, true);
        client.open_tree();
        let (search, _) = client.search("keyword");
        if stale {
            client.controller.state.session_id = Some("newer".into());
            client.controller.state.conversation_generation += 1;
            client.controller.state.phase = Phase::Busy;
            client.controller.state.status = "Another operation".into();
            client.controller.state.session_tree = SessionTree::default();
        }
        let phase = client.controller.state.phase;
        let status = client.controller.state.status.clone();
        let entries = client.controller.state.entries.len();
        let paused = client.controller.state.queue_paused;
        assert!(client.fail(search).is_empty());
        assert_eq!(client.controller.state.phase, phase);
        assert_eq!(client.controller.state.status, status);
        assert_eq!(client.controller.state.entries.len(), entries);
        assert_eq!(client.controller.state.queue_paused, paused);
        assert!(client.controller.state.choices().is_empty());
        if !stale {
            assert!(render(&mut client.controller.state, 80, 24).contains("fixture failure"));
        }
    }
}

#[test]
fn full_message_search_rejects_mismatched_and_invalid_match_results() {
    for response in [
        search_result("different query", &["a2"]),
        search_result("query", &["missing"]),
        tree(),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready_with_search(root.path(), true, true);
        client.open_tree();
        let (search, _) = client.search("query");
        assert!(client.answer(search, response).is_empty());
        assert!(client.controller.state.choices().is_empty());
        assert!(client.key(KeyCode::Enter).is_empty());
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn full_message_search_respects_query_bounds_and_empty_server_matches() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready_with_search(root.path(), true, true);
    client.open_tree();
    client.controller.paste("中".repeat(4097));
    assert!(client.controller.state.choices().is_empty());
    std::thread::sleep(Duration::from_millis(200));
    assert!(client.controller.tick().is_empty());
    assert!(client.key(KeyCode::Enter).is_empty());
    assert!(render(&mut client.controller.state, 80, 24).contains("4096 characters"));
    assert!(client.key(KeyCode::Backspace).is_empty());
    std::thread::sleep(Duration::from_millis(200));
    let actions = client.controller.tick();
    let (search, params) = client.register(actions, "_aster/session/tree");
    assert_eq!(params["query"].as_str().unwrap().chars().count(), 4096);
    assert!(client
        .answer(search, search_result(&"中".repeat(4096), &[]))
        .is_empty());
    assert!(client.controller.state.choices().is_empty());
    assert!(render(&mut client.controller.state, 80, 24).contains("No matches"));
    assert!(client.key(KeyCode::Esc).is_empty());
    client.open_tree();
    for character in "old".chars() {
        assert!(client.key(KeyCode::Char(character)).is_empty());
    }
    std::thread::sleep(Duration::from_millis(200));
    let actions = client.controller.tick();
    let (search, _) = client.register(actions, "_aster/session/tree");
    assert!(client.answer(search, search_result("old", &[])).is_empty());
    // A preview match must not leak into an authoritative empty result.
    assert!(client.controller.state.choices().is_empty());
    for (width, height) in [(24, 6), (40, 12), (80, 24)] {
        assert!(!render(&mut client.controller.state, width, height).contains("Enter continue"));
    }
}

#[test]
fn legacy_tree_search_explicitly_describes_its_preview_limit() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.open_tree();
    assert!(render(&mut client.controller.state, 80, 24).contains("previews only"));
    client.controller.paste("small patch".into());
    assert!(client.controller.tick().is_empty());
    assert_eq!(
        client.controller.state.choices().last().unwrap().value,
        "u2"
    );
}

#[test]
fn user_node_replays_parent_then_recalls_exact_text_without_sending() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.open_tree();
    let actions = client.select("u2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let prompt = "  selected 中文\nsecond line  \n";
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":prompt}));
    assert!(client.controller.state.editor.lines().join("\n").is_empty());
    assert!(client.finish_load(actions, "fork").is_empty());
    assert_eq!(client.controller.state.session_id.as_deref(), Some("fork"));
    assert_eq!(client.controller.state.editor.lines().join("\n"), prompt);
    assert!(client.controller.state.queued.is_empty());
    assert!(client.controller.state.queue_paused);
    assert!(client.controller.state.recovery.is_none());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn inactive_assistant_branch_restores_after_reply_and_keeps_draft() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("my next question".into());
    client.open_tree();
    assert_eq!(
        client.controller.state.editor.lines(),
        &["my next question"]
    );
    let actions = client.select("a2");
    let (navigate, params) = client.register(actions, "_aster/session/navigate");
    assert_eq!(params["nodeId"], "a2");
    let actions = client.answer(navigate, json!({"sessionId":"other-branch","prompt":null}));
    assert!(client.finish_load(actions, "other-branch").is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["my next question"]
    );
    assert!(client.controller.state.queued.is_empty());
    assert!(client.controller.state.queue_paused);
}

#[test]
fn navigation_recovers_the_unsent_editor_hidden_behind_persistent_history() {
    for (node, prompt) in [("u2", Some("selected input")), ("a2", None)] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        let history = InputHistory::for_workspace(root.path());
        history.append("previously submitted input").unwrap();
        client.controller.attach_history(history);
        client.controller.paste("unsent 中文 question".into());
        client.key(KeyCode::Left);
        let cursor = client.controller.state.editor.cursor();
        assert!(client
            .controller
            .key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
            .is_empty());
        assert_eq!(
            client.controller.state.editor.lines(),
            &["previously submitted input"]
        );
        client.open_tree();
        let actions = client.select(node);
        let (navigate, _) = client.register(actions, "_aster/session/navigate");
        let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":prompt}));
        assert!(client.finish_load(actions, "fork").is_empty());
        assert_eq!(
            client.controller.state.editor.lines(),
            &["unsent 中文 question"]
        );
        assert_eq!(client.controller.state.editor.cursor(), cursor);
        assert_eq!(
            client
                .controller
                .state
                .queued
                .front()
                .map(|draft| draft.text.as_str()),
            prompt
        );
        assert!(client.controller.state.queue_paused);
        let saved = DraftStore::for_workspace(root.path())
            .unwrap()
            .list_pending()
            .unwrap();
        assert!(saved
            .iter()
            .any(|draft| draft.snapshot.composer.text == "unsent 中文 question"));
    }
}

#[test]
fn navigating_keeps_attachments_and_new_typing_with_recalled_input_paused() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("draft before navigation".into());
    let attachment = Attachment::Text {
        name: "context.txt".into(),
        uri: "file:///fixture/context.txt".into(),
        text: "saved context".into(),
    };
    client.controller.state.attachments.push(attachment.clone());
    client.open_tree();
    let actions = client.select("u2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    client.controller.paste(" and new typing".into());
    let actions = client.answer(
        navigate,
        json!({"sessionId":"fork","prompt":"selected input"}),
    );
    assert!(client.finish_load(actions, "fork").is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["draft before navigation and new typing"]
    );
    assert_eq!(client.controller.state.attachments, vec![attachment]);
    assert_eq!(
        client.controller.state.queued.front().unwrap().text,
        "selected input"
    );
    assert!(client.controller.state.queue_paused);
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let saved = store.list_pending().unwrap();
    assert!(saved
        .iter()
        .any(|draft| { draft.snapshot.composer.text == "draft before navigation and new typing" }));
}

#[test]
fn failed_tree_requests_and_load_keep_the_original_conversation() {
    for stage in ["tree", "navigate", "load"] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.controller.paste("draft kept".into());
        let actions = client.controller.input("/tree".into());
        let (list, _) = client.register(actions, "_aster/session/tree");
        let failure = if stage == "tree" {
            list
        } else {
            client.answer(list, tree());
            let actions = client.select("u2");
            let (navigate, _) = client.register(actions, "_aster/session/navigate");
            if stage == "navigate" {
                navigate
            } else {
                let actions =
                    client.answer(navigate, json!({"sessionId":"fork","prompt":"recalled"}));
                let (load, _) = client.register(actions, "session/load");
                client.controller.message(json!({"method":"session/update","params":{
                    "sessionId":"fork","update":{"sessionUpdate":"agent_message_chunk","content":{"text":"partial failed replay"}}
                }}));
                load
            }
        };
        assert!(client.fail(failure).is_empty());
        assert_original_transcript(&client);
        assert_eq!(client.controller.state.editor.lines(), &["draft kept"]);
        assert!(!client.controller.state.tree_loading);
        assert!(client.controller.state.queue_paused);
        assert!(client.controller.state.queued.is_empty());
    }
}

#[test]
fn tree_failures_show_the_backend_recovery_reason_and_keep_input() {
    for navigate in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.controller.paste("draft kept".into());
        let actions = client.controller.input("/tree".into());
        let (list, _) = client.register(actions, "_aster/session/tree");
        let failure = if navigate {
            client.answer(list, tree());
            let actions = client.select("u2");
            client.register(actions, "_aster/session/navigate").0
        } else {
            list
        };
        let reason = "The conversation changed; open the tree again.";
        assert!(client
            .controller
            .message(json!({
                "id":failure,
                "error":{"code":-32602,"message":"Invalid params","data":{"reason":reason}}
            }))
            .is_empty());
        assert_original_transcript(&client);
        assert_eq!(client.controller.state.editor.lines(), &["draft kept"]);
        assert!(client.controller.state.queue_paused);
        assert!(client.controller.state.entries.iter().any(|entry| {
            matches!(entry, Entry::Notice { text, error: true } if text == reason)
        }));
    }
}

#[test]
fn current_assistant_is_a_noop_and_invalid_navigation_responses_do_not_switch() {
    for (node, response, no_op) in [
        ("a3", json!({"sessionId":"current","prompt":null}), true),
        ("a2", json!({"sessionId":"current","prompt":null}), false),
        ("u2", json!({"sessionId":"fork","prompt":null}), false),
        (
            "a2",
            json!({"sessionId":"fork","prompt":"unexpected"}),
            false,
        ),
        ("u2", json!({"sessionId":"","prompt":"selected"}), false),
        ("u2", json!({"sessionId":"fork"}), false),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.open_tree();
        let actions = client.select(node);
        let (navigate, _) = client.register(actions, "_aster/session/navigate");
        assert!(client.answer(navigate, response).is_empty());
        assert_original_transcript(&client);
        assert_eq!(
            client.controller.state.status,
            if no_op {
                "Already at this conversation position"
            } else {
                "Request failed"
            }
        );
    }
}

#[test]
fn stale_tree_responses_cannot_replace_another_conversation() {
    for navigate in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        let actions = client.controller.input("/tree".into());
        let (list, _) = client.register(actions, "_aster/session/tree");
        let pending = if navigate {
            client.answer(list, tree());
            let actions = client.select("u2");
            client.register(actions, "_aster/session/navigate").0
        } else {
            list
        };
        client.controller.state.session_id = Some("newer-conversation".into());
        client.controller.state.conversation_generation += 1;
        client.controller.state.phase = Phase::Ready;
        client.controller.state.session_tree = SessionTree::default();
        let response = if navigate {
            json!({"sessionId":"stale-fork","prompt":"stale input"})
        } else {
            tree()
        };
        assert!(client.answer(pending, response).is_empty());
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("newer-conversation")
        );
        assert!(client.controller.state.session_tree.nodes.is_empty());
        assert!(client.controller.state.editor.lines().join("\n").is_empty());
    }
}

#[test]
fn tree_navigation_blocks_pending_queue_and_concurrent_draft_conflicts() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("my saved draft".into());
    assert!(client.controller.persist_draft());
    client.open_tree();
    client
        .controller
        .state
        .queued
        .push_back("queued input".into());
    assert!(client.select("u2").is_empty());
    assert_eq!(client.controller.state.overlay, Some(Overlay::Queue));
    assert!(client.controller.state.queue_paused);
    client.controller.state.queued.clear();
    client.open_tree();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let old = store.load_latest().unwrap().unwrap();
    let concurrent = DraftSnapshot {
        composer: "another window's input".into(),
        ..DraftSnapshot::default()
    };
    let revision = store
        .save(old.session_id.as_deref(), Some(old.revision), &concurrent)
        .unwrap();
    client.controller.state.overlay = None;
    client.controller.paste(" changed".into());
    client.controller.state.open(Overlay::Tree);
    assert!(client.select("u2").is_empty());
    assert_original_transcript(&client);
    let preserved = store.load(old.session_id.as_deref()).unwrap().unwrap();
    assert_eq!(preserved.revision, revision);
    assert_eq!(preserved.snapshot.composer, concurrent.composer);
}

#[test]
fn background_work_blocks_navigation_before_request_and_before_loading() {
    for waiting_for_response in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.controller.paste("keep my draft".into());
        client.open_tree();
        let pending = if waiting_for_response {
            let actions = client.select("u2");
            Some(client.register(actions, "_aster/session/navigate").0)
        } else {
            None
        };
        let jobs = json!({"jobs":[{"id":"check","status":"running","canStop":true}]});
        client.controller.state.tasks = jobs.clone();
        let actions = if let Some(id) = pending {
            client.answer(id, json!({"sessionId":"fork","prompt":"recalled"}))
        } else {
            client.select("u2")
        };
        assert!(actions.is_empty());
        assert_original_transcript(&client);
        assert_eq!(client.controller.state.editor.lines(), &["keep my draft"]);
        assert_eq!(client.controller.state.tasks, jobs);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn saving_new_input_after_navigation_response_must_succeed_before_loading() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("initial draft".into());
    client.open_tree();
    let actions = client.select("u2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let old = store.load_latest().unwrap().unwrap();
    store
        .save(
            old.session_id.as_deref(),
            Some(old.revision),
            &DraftSnapshot {
                composer: "concurrent update".into(),
                ..DraftSnapshot::default()
            },
        )
        .unwrap();
    client.controller.paste(" typed while waiting".into());
    assert!(client
        .answer(navigate, json!({"sessionId":"fork","prompt":"recalled"}))
        .is_empty());
    assert_original_transcript(&client);
    assert_eq!(
        client.controller.state.editor.lines(),
        &["initial draft typed while waiting"]
    );
}

#[test]
fn tree_load_rechecks_messages_queued_while_replaying() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("unsent original draft".into());
    client.controller.state.recovery = Some("interrupted original request".into());
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    client.replay_partial();
    assert!(client
        .controller
        .key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL))
        .is_empty());
    client.controller.paste("retry".into());
    assert_eq!(client.controller.state.choices()[0].value, "/retry");
    assert!(client.key(KeyCode::Enter).is_empty());
    assert_eq!(
        client.controller.state.queued.front().unwrap().text,
        "interrupted original request"
    );
    let actions = client.answer(load, json!({}));
    assert_original_transcript(&client);
    let (close, params) = client.register(actions, "session/close");
    assert_eq!(params["sessionId"], "fork");
    assert!(client.answer(close, json!({})).is_empty());
    assert_original_transcript(&client);
    assert_eq!(
        client.controller.state.editor.lines(),
        &["unsent original draft"]
    );
    assert_eq!(
        client.controller.state.queued.front().unwrap().text,
        "interrupted original request"
    );
    assert!(client.controller.state.queue_paused);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn tree_load_rechecks_draft_conflicts_before_committing() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.paste("unsent original draft".into());
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    client.replay_partial();
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let old = store.load_latest().unwrap().unwrap();
    let concurrent = DraftSnapshot {
        composer: "concurrent other window's update".into(),
        ..DraftSnapshot::default()
    };
    let revision = store
        .save(old.session_id.as_deref(), Some(old.revision), &concurrent)
        .unwrap();
    client.controller.paste(" typed during load".into());
    assert!(!client.controller.persist_draft());
    let actions = client.answer(load, json!({}));
    assert_original_transcript(&client);
    let (close, params) = client.register(actions, "session/close");
    assert_eq!(params["sessionId"], "fork");
    assert!(client.answer(close, json!({})).is_empty());
    assert_original_transcript(&client);
    assert_eq!(
        client.controller.state.editor.lines(),
        &["unsent original draft typed during load"]
    );
    let saved = store.load(old.session_id.as_deref()).unwrap().unwrap();
    assert_eq!(saved.revision, revision);
    assert_eq!(saved.snapshot.composer, concurrent.composer);
    assert!(store.load(Some("fork")).unwrap().is_none());
    assert!(client.controller.state.queue_paused);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn rejected_tree_load_restores_background_updates_and_hidden_input() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    let history = InputHistory::for_workspace(root.path());
    history.append("previously submitted input").unwrap();
    client.controller.attach_history(history);
    client.controller.paste("unsent 中文 question".into());
    client.key(KeyCode::Left);
    let cursor = client.controller.state.editor.cursor();
    let attachment = Attachment::Text {
        name: "context.txt".into(),
        uri: "file:///fixture/context.txt".into(),
        text: "kept attachment bytes".into(),
    };
    client.controller.state.attachments.push(attachment.clone());
    client.controller.state.recovery = Some("kept interrupted input".into());
    assert!(client
        .controller
        .key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["previously submitted input"]
    );
    assert!(client
        .controller
        .message(json!({"method":"session/update","params":{
            "sessionId":"current", "update":{
                "sessionUpdate":"agent_message_chunk", "content":{"text":"background "},
                "_meta":{"aster":{"messageId":"job-message","messageState":"delta"}}
            }
        }}))
        .is_empty());
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    client.replay_partial();
    let jobs = json!({"jobs":[{"id":"check","status":"running","canStop":true}]});
    assert!(client
        .controller
        .message(json!({"method":"session/update","params":{
            "sessionId":"current", "update":{
                "sessionUpdate":"agent_message_chunk", "content":{"text":"job started"},
                "_meta":{"aster":{"messageId":"job-message","messageState":"complete","tasks":jobs}}
            }
        }}))
        .is_empty());
    assert!(client.controller.message(json!({"method":"session/update","params":{
        "sessionId":"current", "update":{
            "sessionUpdate":"tool_call", "toolCallId":"source-job", "title":"Source background job", "status":"in_progress"
        }
    }})).is_empty());
    let store = DraftStore::for_workspace(root.path()).unwrap();
    let scope = store.load_latest().unwrap().unwrap().session_id;
    let actions = client.answer(load, json!({}));
    assert_original_transcript(&client);
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert_eq!(client.controller.state.tasks, jobs);
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Assistant(text) if text == "background job started")
    }));
    assert_eq!(
        client
            .controller
            .state
            .entries
            .iter()
            .filter(|entry| {
                matches!(entry, Entry::Assistant(text) if text.starts_with("background"))
            })
            .count(),
        1
    );
    let (close, params) = client.register(actions, "session/close");
    assert_eq!(params["sessionId"], "fork");
    assert!(client.fail(close).is_empty());
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Tool(tool) if tool.id == "source-job" && tool.status == "in_progress")
    }));
    assert_eq!(
        client.controller.state.editor.lines(),
        &["previously submitted input"]
    );
    assert!(client
        .controller
        .key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL))
        .is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["unsent 中文 question"]
    );
    assert_eq!(client.controller.state.editor.cursor(), cursor);
    assert_eq!(
        client.controller.state.attachments,
        vec![attachment.clone()]
    );
    assert_eq!(
        client.controller.state.recovery.as_deref(),
        Some("kept interrupted input")
    );
    assert!(client.controller.state.queued.is_empty());
    assert!(client.controller.state.queue_paused);
    assert!(client.controller.persist_draft());
    let saved = store.load(scope.as_deref()).unwrap().unwrap();
    assert_eq!(saved.snapshot.composer.text, "unsent 中文 question");
    assert_eq!(saved.snapshot.composer.attachments, vec![attachment]);
    assert_eq!(saved.snapshot.cursor, cursor);
    assert!(store.load(Some("fork")).unwrap().is_none());
}

#[test]
fn tree_load_rechecks_source_tasks_fetched_during_replay() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.state.tasks_supported = true;
    client.controller.paste("kept draft".into());
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    client.replay_partial();
    let actions = client.controller.input("/tasks".into());
    let (tasks, params) = client.register(actions, "_aster/session/tasks");
    assert_eq!(params["sessionId"], "current");
    let jobs = json!({"jobs":[{"id":"check","status":"running","canStop":true}]});
    assert!(client.answer(tasks, jobs.clone()).is_empty());
    let actions = client.answer(load, json!({}));
    assert_original_transcript(&client);
    assert_eq!(client.controller.state.tasks, jobs);
    assert_eq!(client.controller.state.editor.lines(), &["kept draft"]);
    assert_eq!(client.controller.state.phase, Phase::Ready);
    let (close, params) = client.register(actions, "session/close");
    assert_eq!(params["sessionId"], "fork");
    assert!(client.answer(close, json!({})).is_empty());
    assert!(client.controller.state.queue_paused);
}

#[test]
fn tree_load_accepts_source_tasks_requested_before_replay() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.controller.state.tasks_supported = true;
    client.controller.paste("kept source draft".into());
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let source_generation = client.controller.state.conversation_generation;
    let actions = client
        .controller
        .key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::ALT));
    let (tasks, params) = client.register(actions, "_aster/session/tasks");
    assert_eq!(params["sessionId"], "current");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    assert_eq!(
        client.controller.state.conversation_generation,
        source_generation.wrapping_add(1)
    );
    client.replay_partial();
    let jobs = json!({"jobs":[{"id":"check","status":"running","canStop":true}]});
    assert!(client.answer(tasks, jobs.clone()).is_empty());
    let actions = client.answer(load, json!({}));
    assert_original_transcript(&client);
    assert_eq!(client.controller.state.tasks, jobs);
    assert_eq!(
        client.controller.state.editor.lines(),
        &["kept source draft"]
    );
    assert_eq!(client.controller.state.phase, Phase::Ready);
    let (close, params) = client.register(actions, "session/close");
    assert_eq!(params["sessionId"], "fork");
    assert!(client.answer(close, json!({})).is_empty());
    assert!(client.controller.state.queue_paused);
}

#[test]
fn tree_load_rejects_runtime_replies_from_older_loads_and_other_sessions() {
    for other_session in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.controller.state.tasks_supported = true;
        if other_session {
            let actions = client.controller.resume("other".into());
            let (load, _) = client.register(actions, "session/load");
            let actions = client.answer(load, json!({}));
            let (close, params) = client.register(actions, "session/close");
            assert_eq!(params["sessionId"], "current");
            assert!(client.answer(close, json!({})).is_empty());
        }
        let stale_generation = client.controller.state.conversation_generation;
        let actions = client
            .controller
            .key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::ALT));
        let (tasks, params) = client.register(actions, "_aster/session/tasks");
        assert_eq!(
            params["sessionId"],
            if other_session { "other" } else { "current" }
        );
        assert!(client.key(KeyCode::Esc).is_empty());
        let actions = client.controller.resume(
            if other_session {
                "current"
            } else {
                "discarded"
            }
            .into(),
        );
        let (load, _) = client.register(actions, "session/load");
        if other_session {
            let actions = client.answer(load, json!({}));
            let (close, params) = client.register(actions, "session/close");
            assert_eq!(params["sessionId"], "other");
            assert!(client.answer(close, json!({})).is_empty());
        } else {
            assert!(client.fail(load).is_empty());
        }
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("current")
        );
        assert_eq!(
            client.controller.state.conversation_generation,
            stale_generation.wrapping_add(1)
        );
        client.open_tree();
        let actions = client.select("a2");
        let (navigate, _) = client.register(actions, "_aster/session/navigate");
        let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
        let (load, _) = client.register(actions, "session/load");
        let jobs = json!({"jobs":[{"id":"stale","status":"running","canStop":true}]});
        assert!(client.answer(tasks, jobs).is_empty());
        let actions = client.answer(load, json!({}));
        assert_eq!(client.controller.state.session_id.as_deref(), Some("fork"));
        assert!(client.controller.state.tasks.is_null());
        let (close, params) = client.register(actions, "session/close");
        assert_eq!(params["sessionId"], "current");
        assert!(client.answer(close, json!({})).is_empty());
        assert!(client.controller.state.queue_paused);
    }
}

#[test]
fn failed_target_load_keeps_source_background_updates_and_tools() {
    let root = tempfile::tempdir().unwrap();
    let mut client = Client::ready(root.path(), true);
    client.open_tree();
    let actions = client.select("a2");
    let (navigate, _) = client.register(actions, "_aster/session/navigate");
    let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
    let (load, _) = client.register(actions, "session/load");
    client.replay_partial();
    assert!(client.controller.message(json!({"method":"session/update","params":{
        "sessionId":"current", "update":{
            "sessionUpdate":"tool_call", "toolCallId":"source-job", "title":"Source job", "status":"in_progress"
        }
    }})).is_empty());
    assert!(client
        .controller
        .message(json!({"method":"session/update","params":{
            "sessionId":"fork", "update":{
                "sessionUpdate":"current_mode_update", "currentModeId":"plan",
                "_meta":{"aster":{"runtime":{"model":"target-model","executionBackend":"sandbox"}}}
            }
        }}))
        .is_empty());
    assert!(client.fail(load).is_empty());
    assert_original_transcript(&client);
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Tool(tool) if tool.id == "source-job" && tool.status == "in_progress")
    }));
    assert_eq!(client.controller.state.model, "test");
    assert!(!client.controller.state.sandbox);
    assert_eq!(client.controller.state.mode, "code");
}

#[test]
fn unused_tree_cleanup_cannot_complete_a_new_operation_or_change_sessions() {
    for (error, stale) in [(false, false), (true, false), (false, true), (true, true)] {
        let root = tempfile::tempdir().unwrap();
        let mut client = Client::ready(root.path(), true);
        client.open_tree();
        let actions = client.select("a2");
        let (navigate, _) = client.register(actions, "_aster/session/navigate");
        let actions = client.answer(navigate, json!({"sessionId":"fork","prompt":null}));
        let (load, _) = client.register(actions, "session/load");
        client.replay_partial();
        client
            .controller
            .state
            .queued
            .push_back("kept queue".into());
        let actions = client.answer(load, json!({}));
        let (cleanup, params) = client.register(actions, "session/close");
        assert_eq!(params["sessionId"], "fork");
        assert_eq!(client.controller.state.phase, Phase::Ready);
        let actions = client.controller.input("/tree".into());
        let (next_tree, _) = client.register(actions, "_aster/session/tree");
        if stale {
            client.controller.state.session_id = Some("newer".into());
            client.controller.state.conversation_generation += 1;
        }
        let status = client.controller.state.status.clone();
        let session = client.controller.state.session_id.clone();
        let actions = if error {
            client.fail(cleanup)
        } else {
            client.answer(cleanup, json!({}))
        };
        assert!(actions.is_empty());
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.status, status);
        assert_eq!(client.controller.state.session_id, session);
        assert_eq!(
            client.controller.state.queued.front().unwrap().text,
            "kept queue"
        );
        assert!(client.controller.state.queue_paused);
        assert!(client.answer(next_tree, tree()).is_empty());
        assert!(client.answer(cleanup, json!({})).is_empty());
    }
}

#[test]
fn unsupported_unavailable_and_malformed_trees_do_not_fake_navigation() {
    let root = tempfile::tempdir().unwrap();
    let mut unsupported = Client::ready(root.path(), false);
    assert!(unsupported.controller.input("/tree".into()).is_empty());
    assert_original_transcript(&unsupported);

    let mut client = Client::ready(root.path(), true);
    let actions = client.controller.input("/tree".into());
    let (list, _) = client.register(actions, "_aster/session/tree");
    let mut response = tree();
    response["nodes"][2]["navigable"] = json!(false);
    assert!(client.answer(list, response).is_empty());
    assert!(client.select("u2").is_empty());
    assert_original_transcript(&client);
    let actions = client.controller.input("/tree".into());
    let (list, _) = client.register(actions, "_aster/session/tree");
    let mut cyclic = tree();
    cyclic["nodes"][0]["parentId"] = json!("a3");
    assert!(client.answer(list, cyclic).is_empty());
    assert!(client.controller.state.session_tree.nodes.is_empty());
    assert!(!client.controller.state.tree_loading);
    assert_original_transcript(&client);
}

#[test]
fn tree_model_rejects_invalid_graphs_and_handles_deep_ancestry_without_recursion() {
    for damaged in [
        json!({"nodes":[],"currentNodeId":"missing"}),
        json!({"nodes":[{"id":"x","parentId":"absent","role":"user","preview":"x","navigable":true}],"currentNodeId":null}),
        json!({"nodes":[{"id":"x","parentId":"x","role":"user","preview":"x","navigable":true}],"currentNodeId":null}),
        json!({"nodes":[{"id":"x","parentId":null,"role":"tool","preview":"x","navigable":true}],"currentNodeId":null}),
    ] {
        assert!(SessionTree::from_response(&damaged).is_err());
    }
    let mut duplicate = tree();
    duplicate["nodes"][1]["id"] = json!("u1");
    assert!(SessionTree::from_response(&duplicate).is_err());
    let nodes: Vec<_> = (0..2_000)
        .map(|index| {
            json!({
                "id":index.to_string(),
                "parentId":if index == 0 { None } else { Some((index - 1).to_string()) },
                "role":"assistant","preview":format!("message {index}"),"navigable":true
            })
        })
        .collect();
    let deep = SessionTree::from_response(&json!({"nodes":nodes,"currentNodeId":"1999"})).unwrap();
    let choices = deep.choices("message 1999");
    assert_eq!(choices.len(), 2_000);
    assert!(choices.last().unwrap().label.contains('●'));
    assert!(choices.last().unwrap().label.chars().count() < 100);
}

#[test]
fn tree_picker_renders_branches_current_position_and_empty_states_at_compact_sizes() {
    for (width, height) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.session_tree = SessionTree::from_response(&tree()).unwrap();
        state.open(Overlay::Tree);
        state.selected = 5;
        let text = render(&mut state, width, height);
        assert!(
            text.contains("Conversation tree"),
            "{width}x{height}\n{text}"
        );
        if width >= 40 {
            assert!(text.contains("● current"), "{width}x{height}\n{text}");
            assert!(
                text.contains('├') && text.contains('└'),
                "{width}x{height}\n{text}"
            );
            assert!(text.contains("● Assistant"), "{width}x{height}\n{text}");
            assert!(text.contains("Enter continue"), "{width}x{height}\n{text}");
        } else {
            assert!(text.contains("● AI: current"), "{width}x{height}\n{text}");
            assert!(text.contains("Enter resume"), "{width}x{height}\n{text}");
        }
    }
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.open(Overlay::Tree);
    state.tree_loading = true;
    assert!(render(&mut state, 80, 24).contains("Loading conversation tree"));
    state.tree_loading = false;
    assert!(render(&mut state, 80, 24).contains("No messages in this conversation tree"));
    state.session_tree = SessionTree::from_response(&tree()).unwrap();
    state.query = "not present".into();
    assert!(render(&mut state, 80, 24).contains("No matches"));
    state.query.clear();
    state.selected = 1;
    let mut response = tree();
    response["nodes"][1]["navigable"] = json!(false);
    state.session_tree = SessionTree::from_response(&response).unwrap();
    assert!(render(&mut state, 80, 24).contains("History only · no restore point"));
}

fn render(state: &mut State, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal
        .draw(|frame| {
            ui::draw(frame, state);
        })
        .unwrap();
    let buffer = terminal.backend().buffer();
    buffer
        .content
        .chunks(width as usize)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

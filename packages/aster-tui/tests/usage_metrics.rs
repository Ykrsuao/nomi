// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Usage indicators through the same session messages and renderer as the terminal.

use aster_tui::{
    controller::{Action, Controller},
    state::{Phase, State},
    ui,
};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::{json, Value};

struct Client {
    controller: Controller,
    next_id: u64,
}

impl Client {
    fn new() -> Self {
        Self {
            controller: Controller::new(
                State::new("C:/test-workspace".into(), "model-a".into(), false),
                None,
                false,
            ),
            next_id: 1,
        }
    }

    fn ready() -> Self {
        let mut client = Self::new();
        let action = client.controller.initialize();
        let id = client.register(vec![action], "initialize");
        let actions = client.answer(
            id,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}}
                },
                "_meta": {"aster": {"runtimeControls": true}}
            }),
        );
        let id = client.register(actions, "session/list");
        let actions = client.answer(id, json!({"sessions": []}));
        let id = client.register(actions, "session/new");
        let actions = client.answer(id, json!({"sessionId": "session-a"}));
        client.finish_runtime(actions, "model-a");
        assert_eq!(client.controller.state.phase, Phase::Ready);
        client
    }

    fn register(&mut self, actions: Vec<Action>, expected: &str) -> u64 {
        assert_eq!(actions.len(), 1, "expected {expected}: {actions:?}");
        let Action::Request {
            method, pending, ..
        } = actions.into_iter().next().unwrap()
        else {
            panic!("expected {expected} request");
        };
        assert_eq!(method, expected);
        let id = self.next_id;
        self.next_id += 1;
        self.controller.registered(id, pending);
        id
    }

    fn answer(&mut self, id: u64, result: Value) -> Vec<Action> {
        self.controller.message(json!({"id": id, "result": result}))
    }

    fn fail(&mut self, id: u64) {
        assert!(self
            .controller
            .message(json!({
                "id": id,
                "error": {"code": -32603, "message": "fixture failure"}
            }))
            .is_empty());
    }

    fn finish_runtime(&mut self, actions: Vec<Action>, model: &str) {
        let id = self.register(actions, "_aster/session/runtime");
        assert!(self.answer(id, runtime(model, "code", "ask")).is_empty());
    }

    fn update(&mut self, session: &str, update: Value) {
        assert!(self
            .controller
            .message(json!({
                "method": "session/update",
                "params": {"sessionId": session, "update": update}
            }))
            .is_empty());
    }

    fn header(&mut self, columns: u16, rows: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        terminal
            .draw(|frame| {
                ui::draw(frame, &mut self.controller.state);
            })
            .unwrap();
        let start = if rows >= 18 { columns as usize } else { 0 };
        terminal.backend().buffer().content[start..start + columns as usize]
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn seed_usage(&mut self) {
        self.update("session-a", usage(18_000, 100_000, Some(0.0024)));
    }
}

fn runtime(model: &str, mode: &str, approval: &str) -> Value {
    json!({
        "model": model,
        "mode": mode,
        "approval": approval,
        "executionBackend": "native"
    })
}

fn usage(used: u64, size: u64, cost: Option<f64>) -> Value {
    let mut value = json!({"sessionUpdate": "usage_update", "used": used, "size": size});
    if let Some(cost) = cost {
        value["cost"] = json!({"amount": cost, "currency": "USD"});
    }
    value
}

fn assert_context_label(header: &str) {
    assert!(
        header.contains("context") || header.contains("ctx"),
        "{header}"
    );
}

fn assert_pending(header: &str) {
    assert_context_label(header);
    assert!(header.contains("cost"), "{header}");
    assert!(header.matches('—').count() >= 2, "{header}");
    assert!(!header.contains('%') && !header.contains('$'), "{header}");
    assert!(!header.contains("n/a"), "{header}");
}

fn assert_known(header: &str, percent: &str, cost: &str) {
    assert_context_label(header);
    assert!(header.contains(percent), "{header}");
    assert!(header.contains(cost), "{header}");
}

#[test]
fn startup_and_ready_headers_show_pending_indicators_at_supported_widths() {
    for mut client in [Client::new(), Client::ready()] {
        for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
            assert_pending(&client.header(columns, rows));
        }
    }
}

#[test]
fn usage_notifications_render_real_context_and_cost_even_on_narrow_screens() {
    let mut client = Client::ready();
    client.seed_usage();
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        assert_known(&client.header(columns, rows), "18%", "$0.0024");
    }

    client.update("session-a", usage(0, 100_000, Some(0.0)));
    assert_known(&client.header(80, 24), "0%", "$0.0000");
}

#[test]
fn unknown_context_window_preserves_real_tokens_and_cost() {
    let mut client = Client::ready();
    client.update("session-a", usage(18_000, 0, Some(0.0024)));
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        let header = client.header(columns, rows);
        assert!(header.contains("18000"), "{header}");
        assert!(header.contains("tok"), "{header}");
        assert!(header.contains("$0.0024"), "{header}");
        assert!(!header.contains('%') && !header.contains("/ 0"), "{header}");
    }
    assert!(client.controller.state.context_ratio.is_none());
}

#[test]
fn missing_or_invalid_cost_replaces_the_previous_amount_with_unavailable() {
    for cost in [
        None,
        Some(Value::Null),
        Some(json!({"amount": -0.25, "currency": "USD"})),
        Some(json!({"amount": "unknown", "currency": "USD"})),
        Some(json!({"amount": null, "currency": "USD"})),
    ] {
        let mut client = Client::ready();
        client.seed_usage();
        let mut update = usage(25_000, 100_000, None);
        if let Some(cost) = cost {
            update["cost"] = cost;
        }
        client.update("session-a", update);
        let header = client.header(80, 24);
        assert!(header.contains("25%"), "{header}");
        assert!(header.contains("cost n/a"), "{header}");
        assert!(!header.contains('$'), "{header}");
    }
}

#[test]
fn responses_without_usage_show_unavailable_instead_of_fake_zero_or_stale_values() {
    let mut client = Client::ready();
    client.seed_usage();
    let mut update = usage(0, 0, None);
    update["_meta"] = json!({"aster": {"usageAvailable": false}});
    client.update("session-a", update);
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        let header = client.header(columns, rows);
        assert_context_label(&header);
        assert!(header.matches("n/a").count() >= 2, "{header}");
        assert!(!header.contains('%') && !header.contains('$'), "{header}");
        assert!(!header.contains("tok"), "{header}");
    }
    assert!(client.controller.state.context_ratio.is_none());
}

#[test]
fn malformed_token_counts_do_not_become_a_zero_percent_context() {
    for used in [Value::Null, json!(-1), json!("unknown")] {
        let mut client = Client::ready();
        client.seed_usage();
        let mut update = usage(0, 100_000, Some(0.003));
        update["used"] = used;
        client.update("session-a", update);
        let header = client.header(80, 24);
        assert_context_label(&header);
        assert!(header.contains("n/a"), "{header}");
        assert!(header.contains("$0.0030"), "{header}");
        assert!(!header.contains('%'), "{header}");
        assert!(client.controller.state.context_ratio.is_none());
    }
}

#[test]
fn new_conversation_clears_usage_and_ignores_notifications_from_the_closed_session() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/new".into());
    let close = client.register(actions, "session/close");
    assert_known(&client.header(80, 24), "18%", "$0.0024");

    let actions = client.answer(close, json!({}));
    let new = client.register(actions, "session/new");
    assert_pending(&client.header(80, 24));
    client.update("session-a", usage(99_000, 100_000, Some(9.0)));
    assert_pending(&client.header(80, 24));

    let actions = client.answer(new, json!({"sessionId": "session-b"}));
    client.finish_runtime(actions, "model-a");
    assert_pending(&client.header(80, 24));
    client.update("session-a", usage(99_000, 100_000, Some(9.0)));
    assert_pending(&client.header(80, 24));
    client.update("session-b", usage(5_000, 100_000, Some(0.001)));
    assert_known(&client.header(80, 24), "5%", "$0.0010");
}

#[test]
fn successful_load_keeps_only_target_usage_and_never_the_sources_buffered_values() {
    for target_has_usage in [false, true] {
        let mut client = Client::ready();
        client.seed_usage();
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        assert_pending(&client.header(80, 24));
        client.update("session-a", usage(25_000, 100_000, Some(0.5)));
        assert_pending(&client.header(80, 24));
        if target_has_usage {
            client.update("session-b", usage(5_000, 100_000, Some(0.75)));
        }

        let actions = client.answer(load, json!({}));
        let close = client.register(actions, "session/close");
        let actions = client.answer(close, json!({}));
        client.finish_runtime(actions, "model-a");
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("session-b")
        );
        client.update("session-a", usage(99_000, 100_000, Some(9.0)));
        client.update("unrelated-session", usage(99_000, 100_000, Some(9.0)));
        let header = client.header(80, 24);
        if target_has_usage {
            assert_known(&header, "5%", "$0.7500");
        } else {
            assert_pending(&header);
        }
    }
}

#[test]
fn failed_load_restores_source_usage_including_updates_received_during_replay() {
    for source_updates in [false, true] {
        let mut client = Client::ready();
        client.seed_usage();
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        client.update("session-b", usage(5_000, 100_000, Some(0.75)));
        assert_known(&client.header(80, 24), "5%", "$0.7500");
        if source_updates {
            client.update("session-a", usage(25_000, 100_000, Some(0.5)));
            assert_known(&client.header(80, 24), "5%", "$0.7500");
        }

        client.fail(load);
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("session-a")
        );
        let (percent, cost) = if source_updates {
            ("25%", "$0.5000")
        } else {
            ("18%", "$0.0024")
        };
        assert_known(&client.header(80, 24), percent, cost);
        client.update("session-b", usage(99_000, 100_000, Some(9.0)));
        assert_known(&client.header(80, 24), percent, cost);
    }
}

#[test]
fn successful_model_change_clears_context_but_keeps_session_cost() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/model model-b".into());
    let configure = client.register(actions, "_aster/session/configure");
    assert_known(&client.header(80, 24), "18%", "$0.0024");
    assert!(client
        .answer(configure, runtime("model-b", "code", "ask"))
        .is_empty());

    let header = client.header(80, 24);
    assert_context_label(&header);
    assert!(header.contains('—'), "{header}");
    assert!(header.contains("$0.0024"), "{header}");
    assert!(!header.contains('%'), "{header}");
    assert!(client.controller.state.context_ratio.is_none());

    client.update("session-a", usage(10_000, 200_000, Some(0.004)));
    assert_known(&client.header(80, 24), "5%", "$0.0040");
}

#[test]
fn failed_model_change_and_non_model_settings_preserve_current_metrics() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/model unavailable".into());
    let configure = client.register(actions, "_aster/session/configure");
    client.fail(configure);
    assert_eq!(client.controller.state.model, "model-a");
    assert_known(&client.header(80, 24), "18%", "$0.0024");

    for (command, mode, approval) in [
        ("/mode plan", "plan", "ask"),
        ("/permissions session", "plan", "session"),
        ("/model model-a", "plan", "session"),
    ] {
        let actions = client.controller.input(command.into());
        let configure = client.register(actions, "_aster/session/configure");
        assert!(client
            .answer(configure, runtime("model-a", mode, approval))
            .is_empty());
        assert_known(&client.header(80, 24), "18%", "$0.0024");
    }
}

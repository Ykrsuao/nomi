// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Capture actual Ratatui cells for visual review, without starting a backend.
//! Conversation content and usage notifications are synthetic offline fixtures.
use aster_tui::controller::Controller;
use aster_tui::state::{Entry, Overlay, Phase, Session, State, Tool};
use aster_tui::ui;
use ratatui::{backend::TestBackend, style::Color, Terminal};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthStr;

fn color(value: Color, fallback: &str) -> String {
    match value {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        _ => fallback.into(),
    }
}

fn preview_image() -> aster_tui::attachments::Attachment {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 96, 48);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let mut pixels = Vec::new();
        for y in 0..48 {
            for x in 0..96 {
                let color = if (12..84).contains(&x) && (10..38).contains(&y) {
                    if y < 16 {
                        [147, 186, 133]
                    } else if x < 40 {
                        [64, 64, 64]
                    } else {
                        [200, 200, 200]
                    }
                } else {
                    [20 + x as u8, 20 + y as u8, 32]
                };
                pixels.extend_from_slice(&color);
            }
        }
        writer.write_image_data(&pixels).unwrap();
    }
    aster_tui::attachments::Attachment::from_image("terminal-layout.png", &bytes).unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let width: u16 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(120);
    let height: u16 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(36);
    let mode = args.get(3).map(String::as_str).unwrap_or("work");
    let mut state = State::new(
        "workspace/aster".into(),
        "global:deepseek-v4.1-flash".into(),
        false,
    );
    state.phase = Phase::Ready;
    state.monochrome = false;
    state.status = "Ready".into();
    if !matches!(
        mode,
        "welcome" | "welcome-drafts" | "drafts" | "drafts-empty"
    ) {
        state.title = "Polish the terminal workspace".into();
        state.entries=vec![
            Entry::User("把终端界面升级成完整的工作台，保留中文输入和会话恢复。".into()),
            Entry::Assistant("I'll inspect the existing layout, then update the conversation view and run the interaction tests.".into()),
            Entry::Tool(Tool{id:"read".into(), title:"Read src/ui.rs".into(),status:"completed".into(),
                content:"Loaded 248 lines from src/ui.rs".into(),diff:String::new(),expanded:false,
                started:Some(Instant::now()),elapsed:Some(0.4)}),
            Entry::Tool(Tool{id:"edit".into(),title:"Update conversation layout".into(),status:"completed".into(),
                content:"Applied patch to src/ui.rs".into(),diff:"--- a/src/ui.rs\n+++ b/src/ui.rs\n@@ -18,2 +18,3 @@\n-    render_prompt(frame);\n+    render_conversation(frame, &state);\n+    render_composer(frame, &state);\n     render_status(frame);".into(),
                expanded:true,started:Some(Instant::now()),elapsed:Some(1.2)}),
            Entry::Tool(Tool{id:"test".into(),title:"Run interaction tests".into(),status:"in_progress".into(),
                content:"running 20 tests\ntest controller::chinese_input ... ok\ntest controller::resume_session ... ok\ntest transport::cancel_request ... ok".into(),diff:String::new(),
                expanded:false,started:Some(Instant::now()-Duration::from_secs(4)),elapsed:None}),
        ];
        state.phase = Phase::Busy;
        state.status = "Running interaction tests…".into();
        state.started = Some(Instant::now() - Duration::from_secs(12));
        state.editor.insert_str("完成后检查窄窗口下的布局");
        state.sessions = vec![
            Session {
                id: "session-current".into(),
                title: "Polish the terminal workspace".into(),
                updated: "2026-11-09".into(),
            },
            Session {
                id: "session-previous".into(),
                title: "Fix ACP protocol output".into(),
                updated: "2026-11-08".into(),
            },
        ];
        state.session_id = Some("session-current".into());
    }
    match mode {
        "tree" | "tree-search" | "tree-empty" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.editor = Default::default();
            let nodes = if mode == "tree-empty" {
                vec![]
            } else {
                vec![
                    json!({"id":"u1","parentId":null,"role":"user","preview":"Review the parser","navigable":true}),
                    json!({"id":"a1","parentId":"u1","role":"assistant","preview":"Two approaches are available","navigable":true}),
                    json!({"id":"u2","parentId":"a1","role":"user","preview":"Try a small patch","navigable":true}),
                    json!({"id":"a2","parentId":"u2","role":"assistant","preview":"The patch passes its tests","navigable":true}),
                    json!({"id":"u3","parentId":"a1","role":"user","preview":"尝试另一条分支","navigable":true}),
                    json!({"id":"a3","parentId":"u3","role":"assistant","preview":"Alternative plan ready","navigable":true}),
                ]
            };
            let mut response = json!({
                "nodes":nodes,
                "currentNodeId":if mode == "tree-empty" { Value::Null } else { json!("a3") }
            });
            if mode == "tree-search" {
                response["query"] = json!("small patch");
                response["matchedNodeIds"] = json!(["u2"]);
            }
            state.session_tree =
                aster_tui::session_tree::SessionTree::from_response(&response).unwrap();
            state.tree_search_supported = true;
            state.open(Overlay::Tree);
            if mode == "tree-search" {
                state.query = "small patch".into();
                state.selected = 2;
            } else {
                state.selected = state.choices().len().saturating_sub(1);
            }
        }
        "welcome-drafts" | "drafts" | "drafts-empty" => {
            use aster_tui::drafts::{provisional_scope, DraftSnapshot, LoadedDraft};

            if mode != "drafts-empty" {
                state.saved_drafts = vec![
                    LoadedDraft {
                        session_id: Some(provisional_scope("preview")),
                        revision: 2,
                        snapshot: DraftSnapshot {
                            composer: "Review 中文 parser".into(),
                            queued: vec!["Check the tests".into()],
                            queue_paused: true,
                            ..DraftSnapshot::default()
                        },
                        warning: None,
                    },
                    LoadedDraft {
                        session_id: Some("saved-preview-session".into()),
                        revision: 1,
                        snapshot: DraftSnapshot {
                            composer: "Update documentation".into(),
                            in_flight: Some("Inspect the startup path".into()),
                            queue_paused: true,
                            ..DraftSnapshot::default()
                        },
                        warning: None,
                    },
                ];
            }
            if mode != "welcome-drafts" {
                state.open(Overlay::Drafts);
            }
        }
        "steering" => {
            state.steering_supported = true;
            state.entries.push(Entry::Notice { text: "Correction received · waiting for the next model request\n先检查取消和重连，不要改动其他模块。".into(), error:false });
            state.queued.push_back("下一轮再补文档和截图".into());
        }
        "files" => {
            state.file_candidates = vec![
                "packages/aster-tui/src/controller.rs".into(),
                "packages/aster-tui/tests/session_controls.rs".into(),
                "docs/contributing.md".into(),
            ];
            state.open(Overlay::Files);
        }
        "attachments" | "image" => {
            state.attachments.push(aster_tui::attachments::Attachment::Text {
                name:"src/controller.rs".into(), uri:"file:///workspace/src/controller.rs".into(),
                text:"pub fn reconnect(&mut self) {\n    self.queue.pause();\n    self.restore_saved_session();\n}\n".into(),
            });
            state.attachments.push(preview_image());
            state.open(Overlay::Attachments);
            if mode == "image" {
                state.selected = 1;
            }
        }
        "approval" | "approval-expanded" | "approval-fallback" => {
            use aster_tui::state::{ApprovalOption, ApprovalPreview, ApprovalRequest};
            state.mode = "code".into();
            state.approval_policy = "ask".into();
            state.status = "Waiting for execution approval".into();
            state.approvals.push_back(ApprovalRequest {
                id:json!(7), title:"Run shell command?".into(),
                code:"result = await self.shell.run(\n    'cargo test --locked --test session_controls'\n)\nprint(result)\n".into(),
                preview: (mode != "approval-fallback").then(|| ApprovalPreview {
                    summary: "Run shell command".into(),
                    details: vec!["Command: cargo test --locked --test session_controls".into()],
                    complete: true,
                }),
                show_code: mode != "approval",
                options:vec![
                    ApprovalOption { id:"allow_once".into(), name:"Allow once".into(), kind:"allow_once".into() },
                    ApprovalOption { id:"allow_session".into(), name:"Allow all code for this session".into(), kind:"allow_always".into() },
                    ApprovalOption { id:"reject".into(), name:"Reject execution".into(), kind:"reject_once".into() },
                ],
            });
            state.open(Overlay::Approval);
            state.selected = state.approvals[0].default_selection();
        }
        "tasks" | "progress" => {
            state.tasks_supported = true;
            state.tasks = json!({
                "steps":[
                    {"id":"step-1","title":"Wire steering into the active turn","status":"completed"},
                    {"id":"step-2","title":"Verify draft recovery and approvals","description":"Run deterministic tests against the local backend.","status":"in_progress","blockedBy":[],"comments":["The composer remains editable while tests run."]},
                    {"id":"step-3","title":"Build the native release executable","status":"pending","blockedBy":["step-2"]}
                ],
                "currentTask":"Verify draft recovery and approvals",
                "waitReason":"Waiting for the test process",
                "jobs":[{"id":"job-tests-3","channel":"tests","title":"Interaction tests","status":"running","canStop":true,"output":"running 20 tests\nattachments ... ok\napproval_cancel ... ok\nreconnect ... ok"}]
            });
            state.status = "Verify recovery · waiting for test process".into();
            if mode == "tasks" {
                state.open(Overlay::Tasks);
                state.selected = 3;
            }
        }
        "recovery" => {
            state.phase = Phase::Failed;
            state.started = None;
            state.status = "Backend disconnected · /reconnect · /retry".into();
            state.recovery = Some("继续检查取消和重连的边界条件".into());
            state.queued.push_back("补充文档和使用示例".into());
            state.queue_paused = true;
            state.entries.push(Entry::Notice {
                text: "Backend disconnected. Input was saved; /retry restores it for review."
                    .into(),
                error: true,
            });
        }
        "rollback" => {
            use aster_tui::workspace_changes::{ChangeStage, ChangedFile};
            state.phase = Phase::Ready;
            state.started = None;
            let file = ChangedFile { path:"src/controller.rs".into(), original_path:None,
                status:"Modified".into(), stage:ChangeStage::Unstaged,
                patch:"diff --git a/src/controller.rs b/src/controller.rs\n--- a/src/controller.rs\n+++ b/src/controller.rs\n@@ -1 +1 @@\n-let queue_paused = false;\n+let queue_paused = true;\n".into() };
            state.rollback.hunks = aster_tui::rollback::hunk_choices(&file).unwrap();
            state.rollback.file = Some(file);
            state.open(Overlay::Rollback);
            state.selected = 1;
        }
        "python-error" => {
            use aster_tui::controller::Controller;

            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.title = "Inspect a failed Python call".into();
            state.editor = tui_textarea::TextArea::default();
            state.entries.clear();
            let mut controller = Controller::new(state, None, false);
            controller.message(json!({
                "method":"session/update","params":{"sessionId":"session-current","update":{
                    "sessionUpdate":"tool_call_update","toolCallId":"python-error",
                    "title":"Python failed","status":"failed","_meta":{"aster":{"elapsedSeconds":0.01}},
                    "content":[
                        {"type":"content","content":{"type":"text","text":"```python\nimport subprocess\nresult = subprocess.run(['git', 'status', '--short'])\n```","_meta":{"aster":{"format":"fenced-code"}}}},
                        {"type":"content","content":{"type":"text","text":"```text\nCell In[1], line 1\n    import subprocess\n    ^\nimport of 'subprocess' is blocked. This module can freeze the event loop and is not allowed in agent code.\n```","_meta":{"aster":{"format":"fenced-code"}}}}
                    ]
                }}
            }));
            controller.toggle_tool("python-error");
            state = controller.state;
        }
        "tools" => state.open(Overlay::Tools),
        "commands" => state.open(Overlay::Commands),
        "slash" => {
            state.editor = tui_textarea::TextArea::default();
            state.editor.insert_str("/");
        }
        "changes" => {
            use aster_tui::workspace_changes::{ChangeStage, ChangedFile, WorkspaceChanges};
            state.workspace_changes = Some(WorkspaceChanges {
                root: state.workspace.clone().into(),
                files: vec![
                    ChangedFile { path:"src/ui.rs".into(), original_path:None, status:"Modified".into(), stage:ChangeStage::Staged,
                        patch:"--- a/src/ui.rs\n+++ b/src/ui.rs\n@@ -18,2 +18,3 @@\n-    render_prompt(frame);\n+    render_conversation(frame, &state);\n+    render_composer(frame, &state);\n     render_status(frame);".into() },
                    ChangedFile { path:"src/controller.rs".into(), original_path:None, status:"Modified".into(), stage:ChangeStage::Unstaged,patch:"--- a/src/controller.rs\n+++ b/src/controller.rs\n@@ -1 +1 @@\n-old\n+new".into() },
                    ChangedFile { path:"docs/使用说明.md".into(), original_path:None, status:"Added".into(), stage:ChangeStage::Untracked,patch:"@@ -0,0 +1 @@\n+使用右键粘贴，按 Enter 发送。".into() },
                ], notice:None,
            });
            state.open(Overlay::Changes);
        }
        "sessions" => state.open(Overlay::Sessions),
        "sidebar" => state.show_sidebar = true,
        "search" => {
            state.open(Overlay::Search);
            state.query = "layout".into();
        }
        "queue" => {
            state.queued.push_back("检查中文输入和长回答滚动".into());
            state
                .queued
                .push_back("完成后运行回归测试，并总结这轮改动".into());
            state.open(Overlay::Queue);
        }
        "markdown" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.entries = vec![Entry::Assistant(
                "## Terminal workspace\n\nRead **clearer answers**, keep `code` readable, and find earlier output.\n\n> Your draft stays intact while you browse.\n\n- [x] Search messages and tool output\n- [x] Send follow-up messages in order\n- [ ] Review the final result\n\n```rust\nfn main() {\n    println!(\"你好，Aster\");\n}\n```\n\n| Feature | Shortcut |\n| --- | --- |\n| 搜索对话 | Ctrl+F |\n| 消息队列 | Ctrl+E |\n| 工具输出 | Ctrl+T |".into())];
        }
        "compose" => {
            state.phase = Phase::Ready;
            state.started = None;
            state.editor = tui_textarea::TextArea::default();
            state.editor.insert_str("请检查 src/main.rs 中的会话恢复逻辑，保留现有草稿和中文输入，完成后运行回归测试并说明改动。长输入现在会自动折行，方向键按屏幕上的行移动。");
        }
        "compact" => {
            state.phase = Phase::Ready;
            state.started = None;
            state.editor = tui_textarea::TextArea::default();
            state.entries = vec![
                Entry::User("请检查终端的输入、会话和工具执行流程。\n要求：\n- 保留中文和多行草稿\n- 长日志支持紧凑预览\n- 搜索能定位隐藏的内容\n- 运行测试并汇报结果".into()),
                Entry::Assistant("I'll check the input flow and run the regression tests.".into()),
                Entry::Tool(Tool {
                    id: "compact-test".into(), title: "Run cargo test".into(), status: "completed".into(),
                    content: format!("running 24 tests\n{}\n\ntest result: ok. 24 passed; 0 failed", (1..=24).map(|i| format!("test interaction_{i:02} ... ok")).collect::<Vec<_>>().join("\n")),
                    diff: String::new(), expanded: true, started: Some(Instant::now()), elapsed: Some(1.2),
                }),
                Entry::Assistant("The input flow is ready. Text stays unchanged while the composer wraps it on screen.\n\n```rust\nfn visible_rows(text: &str, width: usize) -> usize {\n    text.chars().count().div_ceil(width.max(1))\n}\n```".into()),
            ];
        }
        _ => {}
    }
    let has_sample_usage = state.session_id.is_some();
    if has_sample_usage {
        let mut update = json!({
            "sessionUpdate": "usage_update", "used": 18000, "size": 100000,
            "cost": {"amount": 0.0024, "currency": "USD"}
        });
        match mode {
            "usage-unknown-window" => update["size"] = json!(0),
            "usage-no-price" => update["cost"] = Value::Null,
            "usage-unavailable" => {
                update["used"] = json!(0);
                update["size"] = json!(0);
                update["cost"] = Value::Null;
                update["_meta"] = json!({"aster": {"usageAvailable": false}});
            }
            "usage-free" => update["cost"]["amount"] = json!(0),
            _ => {}
        }
        // Use the production notification parser, not direct display strings.
        let mut controller = Controller::new(state, None, false);
        controller.message(json!({
            "method": "session/update",
            "params": {"sessionId": "session-current", "update": update}
        }));
        state = controller.state;
    }
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal
        .draw(|frame| {
            ui::draw(frame, &mut state);
        })
        .unwrap();
    let buffer = terminal.backend().buffer();
    let mut cells = Vec::new();
    for row in buffer.content.chunks(width as usize) {
        let mut continuation = 0;
        let mut previous_fg = String::new();
        let mut previous_bg = String::new();
        for cell in row {
            if continuation > 0 {
                cells.push(json!({"text":"", "fg":previous_fg, "bg":previous_bg, "bold":false}));
                continuation -= 1;
                continue;
            }
            previous_fg = color(cell.fg, "#e1e1e1");
            previous_bg = color(cell.bg, "#141414");
            continuation = cell.symbol().width().saturating_sub(1);
            cells.push(
                json!({"text":cell.symbol(),"fg":previous_fg,"bg":previous_bg,
                "bold":cell.modifier.contains(ratatui::style::Modifier::BOLD)}),
            );
        }
    }
    println!(
        "{}",
        json!({
            "width":width,"height":height,"cells":cells,
            "source":"offline preview", "mode":mode,
            "usageSource":if has_sample_usage { "synthetic ACP notification" } else { "none" }
        })
    );
}

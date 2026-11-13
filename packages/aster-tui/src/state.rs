// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use tui_textarea::TextArea;

#[derive(Clone, Debug)]
pub struct Tool {
    pub id: String,
    pub title: String,
    pub status: String,
    pub content: String,
    pub diff: String,
    pub expanded: bool,
    pub started: Option<Instant>,
    pub elapsed: Option<f64>,
}

impl Tool {
    pub fn elapsed_seconds(&self) -> Option<f64> {
        self.elapsed.or_else(|| {
            if matches!(self.status.as_str(), "pending" | "in_progress") {
                self.started.map(|start| start.elapsed().as_secs_f64())
            } else {
                None
            }
        })
    }
}

#[derive(Clone, Debug)]
pub enum Entry {
    User(String),
    Assistant(String),
    Tool(Tool),
    Notice { text: String, error: bool },
}

#[derive(Clone, Debug, Default)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub updated: String,
}

#[derive(Clone, Debug)]
pub struct Command {
    pub name: String,
    pub description: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Connecting,
    Preparing,
    Ready,
    Busy,
    Cancelling,
    Closing,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    Commands,
    Sessions,
    Drafts,
    Tools,
    Changes,
    Help,
    Diagnostics,
    Queue,
    Search,
    Rewind,
    Tree,
    Files,
    Attachments,
    Models,
    Modes,
    Permissions,
    Approval,
    Tasks,
    Rollback,
    RollbackConfirm,
    UndoConfirm,
}

#[derive(Clone, Debug)]
pub struct RewindPoint {
    pub user_message_index: usize,
    pub prompt: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringState {
    Sending,
    Accepted,
    Applied,
    Cancelled,
    Interrupted,
    TurnFinished,
    Rejected,
}

impl SteeringState {
    pub fn from_wire(value: &Value) -> Option<Self> {
        // The protocol uses string states. Serde also accepts externally tagged
        // objects for unit variants, which are not valid steering receipts.
        value.as_str()?;
        serde_json::from_value(value.clone()).ok()
    }

    pub fn is_pending(self) -> bool {
        match self {
            Self::Sending | Self::Accepted => true,
            Self::Applied
            | Self::Cancelled
            | Self::Interrupted
            | Self::TurnFinished
            | Self::Rejected => false,
        }
    }

    pub fn is_terminal(self) -> bool {
        !self.is_pending()
    }

    pub fn is_acknowledged(self) -> bool {
        matches!(self, Self::Accepted | Self::Applied)
    }

    /// Notifications and replies may arrive in either order. A terminal receipt
    /// cannot become pending again; an applied receipt cannot be downgraded.
    pub fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::Applied => next == Self::Applied,
            _ if self.is_terminal() => next.is_terminal(),
            _ => true,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sending => "Sending correction…",
            Self::Accepted => "Correction received · waiting for the next model request",
            Self::Applied => "Correction included in the model request",
            Self::Cancelled => "Correction not applied · task cancelled",
            Self::Interrupted => "Correction interrupted · review before resending",
            Self::TurnFinished => "Correction not applied · task already finished",
            Self::Rejected => "Correction rejected · draft kept",
        }
    }

    pub fn is_error(self) -> bool {
        matches!(
            self,
            Self::Rejected | Self::Interrupted | Self::TurnFinished
        )
    }
}

#[derive(Clone, Debug)]
pub struct SteeringMessage {
    pub id: String,
    pub turn_id: String,
    pub text: String,
    pub state: SteeringState,
    pub reason: String,
    pub entry: usize,
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub label: String,
    pub detail: String,
    pub value: String,
}

#[derive(Clone, Debug)]
pub struct ApprovalOption {
    pub id: String,
    pub name: String,
    pub kind: String,
}

#[derive(Clone, Debug)]
pub struct ApprovalPreview {
    pub summary: String,
    pub details: Vec<String>,
    /// Display completeness only; this never changes execution permissions.
    pub complete: bool,
}

#[derive(Clone, Debug)]
pub struct ApprovalRequest {
    pub id: Value,
    pub title: String,
    pub code: String,
    pub preview: Option<ApprovalPreview>,
    pub show_code: bool,
    pub options: Vec<ApprovalOption>,
}

impl ApprovalRequest {
    pub fn default_selection(&self) -> usize {
        self.options
            .iter()
            .position(|option| option.kind == "allow_once")
            .or_else(|| {
                self.options
                    .iter()
                    .position(|option| option.kind.starts_with("reject"))
            })
            .unwrap_or(0)
    }
}

#[derive(Default)]
pub struct RollbackView {
    pub file: Option<crate::workspace_changes::ChangedFile>,
    pub hunks: Vec<String>,
    pub plan: Option<crate::rollback::RollbackPlan>,
    pub receipt: Option<crate::rollback::RollbackReceipt>,
    pub loading: bool,
}

pub(crate) struct RemovedAttachment {
    pub attachment: crate::attachments::Attachment,
    pub index: usize,
    pub generation: u64,
}

pub struct State {
    pub workspace: String,
    pub model: String,
    pub sandbox: bool,
    pub session_id: Option<String>,
    /// Changes on a conversation switch, not when background preparation returns its ID.
    pub conversation_generation: u64,
    pub title: String,
    pub phase: Phase,
    pub status: String,
    /// A second Ctrl+C before this instant confirms the user's exit request.
    pub exit_confirmation: Option<Instant>,
    /// Only an idle, unconsumed Esc press can start the double-Esc gesture.
    pub escape_confirmation: Option<Instant>,
    pub rewind_points: Vec<RewindPoint>,
    pub rewind_loading: bool,
    pub session_tree: crate::session_tree::SessionTree,
    pub tree_loading: bool,
    pub tree_search_supported: bool,
    pub tree_search_error: Option<String>,
    pub steering_supported: bool,
    pub active_turn_id: Option<String>,
    pub steering_messages: Vec<SteeringMessage>,
    pub entries: Vec<Entry>,
    /// Full transcript details explicitly opened by the user or a search jump.
    pub expanded_entries: HashSet<usize>,
    pub sessions: Vec<Session>,
    pub saved_drafts: Vec<crate::drafts::LoadedDraft>,
    pub commands: Vec<Command>,
    pub editor: TextArea<'static>,
    pub(crate) composer: crate::composer::ComposerState,
    /// Invalidates asynchronous edits even when a later draft has identical text.
    pub draft_revision: u64,
    pub overlay: Option<Overlay>,
    pub query: String,
    pub selected: usize,
    pub slash_selected: usize,
    pub slash_dismissed: bool,
    /// Unsent messages belong to the current conversation and remain in FIFO order.
    pub queued: VecDeque<crate::drafts::MessageDraft>,
    pub attachments: Vec<crate::attachments::Attachment>,
    pub(crate) removed_attachment: Option<RemovedAttachment>,
    pub(crate) image_preview: crate::image_preview::PreviewCache,
    pub terminal_graphics: crate::terminal_input::GraphicsCapabilities,
    pub terminal_image: crate::terminal_image::PreviewCache,
    /// Image-viewer status remains visible while the attachment inspector is open.
    pub original_image_notice: Option<(String, bool)>,
    pub attachments_supported: bool,
    pub image_input_supported: bool,
    pub attachment_loading: bool,
    pub file_candidates: Vec<String>,
    pub files_loading: bool,
    pub files_notice: String,
    pub recovery: Option<crate::drafts::MessageDraft>,
    pub runtime_supported: bool,
    pub runtime: Value,
    pub mode: String,
    pub approval_policy: String,
    pub approvals: VecDeque<ApprovalRequest>,
    pub tasks_supported: bool,
    pub tasks: Value,
    pub queue_paused: bool,
    /// The next render should reveal this transcript entry after a search jump.
    pub focus_entry: Option<usize>,
    pub search_highlight: String,
    /// Number of wrapped transcript rows above the newest output.
    pub scroll: usize,
    pub(crate) rendered_rows: Option<(usize, u16)>,
    pub(crate) transcript_cache: crate::ui::TranscriptCache,
    pub detail_scroll: u16,
    /// Actual approval viewport rows, so paging cannot skip unreviewed code.
    pub(crate) approval_page_rows: u16,
    pub show_sidebar: bool,
    pub usage: String,
    pub context_ratio: Option<f64>,
    pub cost: String,
    pub started: Option<Instant>,
    pub last_elapsed: Option<f64>,
    pub diagnostics: VecDeque<String>,
    pub workspace_changes: Option<crate::workspace_changes::WorkspaceChanges>,
    pub changes_loading: bool,
    pub changes_error: Option<String>,
    pub conversation_changes: bool,
    pub rollback: RollbackView,
    pub monochrome: bool,
}

impl State {
    pub fn new(workspace: String, model: String, sandbox: bool) -> Self {
        Self {
            workspace,
            model,
            sandbox,
            session_id: None,
            conversation_generation: 0,
            title: String::new(),
            phase: Phase::Connecting,
            status: "Starting agent…".into(),
            exit_confirmation: None,
            escape_confirmation: None,
            rewind_points: Vec::new(),
            rewind_loading: false,
            session_tree: crate::session_tree::SessionTree::default(),
            tree_loading: false,
            tree_search_supported: false,
            tree_search_error: None,
            steering_supported: false,
            active_turn_id: None,
            steering_messages: Vec::new(),
            entries: Vec::new(),
            expanded_entries: HashSet::new(),
            sessions: Vec::new(),
            saved_drafts: Vec::new(),
            commands: default_commands(),
            editor: TextArea::default(),
            composer: crate::composer::ComposerState::default(),
            draft_revision: 0,
            overlay: None,
            query: String::new(),
            selected: 0,
            slash_selected: 0,
            slash_dismissed: false,
            queued: VecDeque::new(),
            attachments: Vec::new(),
            removed_attachment: None,
            image_preview: crate::image_preview::PreviewCache::default(),
            terminal_graphics: crate::terminal_input::GraphicsCapabilities::default(),
            terminal_image: crate::terminal_image::PreviewCache::default(),
            original_image_notice: None,
            attachments_supported: false,
            image_input_supported: false,
            attachment_loading: false,
            file_candidates: Vec::new(),
            files_loading: false,
            files_notice: String::new(),
            recovery: None,
            runtime_supported: false,
            runtime: Value::Null,
            mode: "code".into(),
            approval_policy: "backend".into(),
            approvals: VecDeque::new(),
            tasks_supported: false,
            tasks: Value::Null,
            queue_paused: false,
            focus_entry: None,
            search_highlight: String::new(),
            scroll: 0,
            rendered_rows: None,
            transcript_cache: crate::ui::TranscriptCache::default(),
            detail_scroll: 0,
            approval_page_rows: 1,
            show_sidebar: false,
            usage: String::new(),
            context_ratio: None,
            cost: String::new(),
            started: None,
            last_elapsed: None,
            diagnostics: VecDeque::new(),
            workspace_changes: None,
            changes_loading: false,
            changes_error: None,
            conversation_changes: false,
            rollback: RollbackView::default(),
            monochrome: std::env::var_os("NO_COLOR").is_some(),
        }
    }

    pub fn busy(&self) -> bool {
        matches!(
            self.phase,
            Phase::Connecting | Phase::Preparing | Phase::Busy | Phase::Cancelling | Phase::Closing
        )
    }

    pub fn preparing(&self) -> bool {
        matches!(self.phase, Phase::Connecting | Phase::Preparing)
    }

    pub fn can_undo_attachment(&self) -> bool {
        self.removed_attachment
            .as_ref()
            .is_some_and(|removed| removed.generation == self.conversation_generation)
    }

    pub(crate) fn draft_changed(&mut self) {
        self.draft_revision = self.draft_revision.wrapping_add(1);
        self.removed_attachment = None;
        self.composer.reset_navigation();
    }

    pub fn open(&mut self, overlay: Overlay) {
        self.escape_confirmation = None;
        self.original_image_notice = None;
        self.composer.cancel_pointer();
        self.overlay = Some(overlay);
        self.query.clear();
        self.selected = 0;
        self.detail_scroll = 0;
        self.approval_page_rows = 1;
    }

    pub fn notice(&mut self, text: impl Into<String>, error: bool) {
        self.entries.push(Entry::Notice {
            text: clean(&text.into()),
            error,
        });
    }

    pub fn tree_search_pending(&self) -> bool {
        self.tree_search_supported
            && !self.query.trim().is_empty()
            && self.tree_search_error.is_none()
            && !self.session_tree.has_search_results(&self.query)
    }

    /// Command suggestions belong to the draft, independently of modal pickers.
    pub fn slash_choices(&self) -> Vec<Choice> {
        if self.overlay.is_some() || self.slash_dismissed || self.editor.lines().len() != 1 {
            return Vec::new();
        }
        let text = &self.editor.lines()[0];
        if !text.starts_with('/') || text.chars().any(char::is_whitespace) {
            return Vec::new();
        }
        let query = text.to_lowercase();
        self.commands
            .iter()
            .filter(|command| command.name.to_lowercase().starts_with(&query))
            .map(|command| Choice {
                label: command.name.clone(),
                detail: command.description.clone(),
                value: command.name.clone(),
            })
            .collect()
    }

    pub fn choices(&self) -> Vec<Choice> {
        if self.overlay == Some(Overlay::Tree) {
            return if self.tree_loading
                || self.tree_search_pending()
                || self.tree_search_error.is_some()
            {
                Vec::new()
            } else {
                self.session_tree.choices(&self.query)
            };
        }
        if self.overlay == Some(Overlay::Search) {
            return crate::search::choices(&self.entries, &self.query);
        }
        if self.overlay == Some(Overlay::Queue) {
            let query = self.query.to_lowercase();
            return self
                .queued
                .iter()
                .enumerate()
                .filter(|(_, text)| text.to_lowercase().contains(&query))
                .map(|(index, text)| Choice {
                    label: clean(
                        text.lines()
                            .find(|line| !line.trim().is_empty())
                            .unwrap_or(""),
                    ),
                    detail: format!(
                        "#{} · {} lines · {} characters · {} attachments",
                        index + 1,
                        text.lines().count(),
                        text.chars().count(),
                        text.attachments.len()
                    ),
                    value: index.to_string(),
                })
                .collect();
        }
        let query = self.query.to_lowercase();
        let choices: Vec<Choice> = match self.overlay {
            Some(Overlay::Rollback) => {
                let mut scopes = vec![Choice {
                    label: "All unstaged hunks in this file".into(),
                    detail: "Review the complete rollback before applying".into(),
                    value: "all".into(),
                }];
                scopes.extend(
                    self.rollback
                        .hunks
                        .iter()
                        .enumerate()
                        .map(|(index, header)| Choice {
                            label: format!("Hunk {} · {}", index + 1, clean(header)),
                            detail: "Keep every other hunk".into(),
                            value: index.to_string(),
                        }),
                );
                scopes
            }
            Some(Overlay::RollbackConfirm | Overlay::UndoConfirm) if !self.rollback.loading => {
                vec![
                    Choice {
                        label: "Cancel".into(),
                        detail: "Keep the current file".into(),
                        value: "cancel".into(),
                    },
                    Choice {
                        label: if self.overlay == Some(Overlay::UndoConfirm) {
                            "Undo the previous rollback"
                        } else {
                            "Apply the reviewed rollback"
                        }
                        .into(),
                        detail: "File and index are checked again before writing".into(),
                        value: "apply".into(),
                    },
                ]
            }
            Some(Overlay::Models) => self.runtime["models"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|model| {
                    let id = model["id"].as_str()?;
                    Some(Choice {
                        label: clean(model["name"].as_str().unwrap_or(id)),
                        detail: if id == self.model {
                            "Current model"
                        } else {
                            "Use this configured model for the session"
                        }
                        .into(),
                        value: id.into(),
                    })
                })
                .collect(),
            Some(Overlay::Modes) => self.runtime["modes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|mode| {
                    let id = mode["id"].as_str()?;
                    Some(Choice {
                        label: clean(mode["name"].as_str().unwrap_or(id)),
                        detail: clean(mode["description"].as_str().unwrap_or("")),
                        value: id.into(),
                    })
                })
                .collect(),
            Some(Overlay::Permissions) => vec![
                Choice {
                    label: "Ask before executing code".into(),
                    detail:
                        "Review each generated cell; allow once, allow for this session, or reject"
                            .into(),
                    value: "ask".into(),
                },
                Choice {
                    label: "Allow code for this live session".into(),
                    detail:
                        "All generated code may execute until this session closes or reconnects"
                            .into(),
                    value: "session".into(),
                },
            ],
            Some(Overlay::Approval) => self
                .approvals
                .front()
                .map(|approval| {
                    approval
                        .options
                        .iter()
                        .map(|option| Choice {
                            label: option.name.clone(),
                            detail: option.kind.replace('_', " "),
                            value: option.id.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Some(Overlay::Tasks) => {
                let mut choices = Vec::new();
                if let Some(steps) = self.tasks["steps"].as_array() {
                    for step in steps {
                        let id = step["id"].as_str().unwrap_or("");
                        choices.push(Choice {
                            label: clean(step["title"].as_str().unwrap_or("Plan step")),
                            detail: clean(step["status"].as_str().unwrap_or("pending")),
                            value: format!("step:{id}"),
                        });
                    }
                }
                if let Some(jobs) = self.tasks["jobs"].as_array() {
                    for job in jobs {
                        let id = job["id"].as_str().unwrap_or("");
                        choices.push(Choice {
                            label: clean(job["title"].as_str().unwrap_or(id)),
                            detail: format!(
                                "Background · {}{}",
                                clean(job["status"].as_str().unwrap_or("unknown")),
                                if job["canStop"] == true {
                                    " · Ctrl+X stop"
                                } else {
                                    ""
                                }
                            ),
                            value: format!("job:{id}"),
                        });
                    }
                }
                choices
            }
            Some(Overlay::Files) => self
                .file_candidates
                .iter()
                .map(|path| Choice {
                    label: clean(path),
                    detail: "Enter attaches a snapshot of this file".into(),
                    value: path.clone(),
                })
                .collect(),
            Some(Overlay::Attachments) => self
                .attachments
                .iter()
                .enumerate()
                .map(|(index, item)| Choice {
                    label: clean(item.name()),
                    detail: format!(
                        "{} · {} bytes · included on send",
                        item.mime_type(),
                        item.byte_len()
                    ),
                    value: index.to_string(),
                })
                .collect(),
            Some(Overlay::Commands) => self
                .commands
                .iter()
                .map(|c| Choice {
                    label: c.name.clone(),
                    detail: c.description.clone(),
                    value: c.name.clone(),
                })
                .collect(),
            Some(Overlay::Drafts) => self
                .saved_drafts
                .iter()
                .enumerate()
                .map(|(index, draft)| {
                    let snapshot = &draft.snapshot;
                    let input = std::iter::once(&snapshot.composer)
                        .chain(snapshot.in_flight.iter())
                        .chain(snapshot.queued.iter())
                        .find(|input| !input.is_empty());
                    let label = input
                        .and_then(|input| input.text.lines().find(|line| !line.trim().is_empty()))
                        .map(|line| clean(&line.chars().take(160).collect::<String>()))
                        .unwrap_or_else(|| "Attached context".into());
                    let conversation =
                        if crate::drafts::is_provisional_scope(draft.session_id.as_deref()) {
                            "New conversation"
                        } else {
                            "Saved conversation"
                        };
                    Choice {
                        label,
                        detail: format!(
                            "{conversation} · {} queued{}",
                            snapshot.queued.len(),
                            if snapshot.in_flight.is_some() {
                                " · interrupted input"
                            } else {
                                ""
                            }
                        ),
                        value: index.to_string(),
                    }
                })
                .collect(),
            Some(Overlay::Sessions) => self
                .sessions
                .iter()
                .map(|s| Choice {
                    label: if s.title.is_empty() {
                        "Untitled conversation".into()
                    } else {
                        s.title.clone()
                    },
                    detail: format!("{}  {}", s.id, s.updated),
                    value: s.id.clone(),
                })
                .collect(),
            Some(Overlay::Rewind) => {
                return self
                    .rewind_points
                    .iter()
                    .rev()
                    .filter(|point| clean(&point.prompt).to_lowercase().contains(&query))
                    .map(|point| Choice {
                        label: clean(&point.prompt)
                            .lines()
                            .find(|line| !line.trim().is_empty())
                            .unwrap_or("")
                            .to_owned(),
                        detail: format!(
                            "Message {} · edit and continue from before this message",
                            point.user_message_index + 1
                        ),
                        value: point.user_message_index.to_string(),
                    })
                    .collect()
            }
            Some(Overlay::Tree) => {
                unreachable!("tree choices are handled before query normalization")
            }
            Some(Overlay::Tools) => self
                .entries
                .iter()
                .rev()
                .filter_map(|e| match e {
                    Entry::Tool(t) => Some(Choice {
                        label: t.title.clone(),
                        detail: t.status.clone(),
                        value: t.id.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            Some(Overlay::Changes) if !self.conversation_changes => self
                .workspace_changes
                .as_ref()
                .map(|snapshot| {
                    snapshot
                        .files
                        .iter()
                        .enumerate()
                        .map(|(index, file)| Choice {
                            label: clean(&file.path),
                            detail: format!("{} · {}", file.stage.label(), clean(&file.status)),
                            value: index.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        choices
            .into_iter()
            .filter(|c| {
                format!("{} {}", c.label, c.detail)
                    .to_lowercase()
                    .contains(&query)
            })
            .collect()
    }

    pub fn selected_tool(&self) -> Option<&Tool> {
        let choices = self.choices();
        let id = &choices.get(self.selected)?.value;
        self.entries.iter().find_map(|e| match e {
            Entry::Tool(t) if &t.id == id => Some(t),
            _ => None,
        })
    }

    pub fn selected_change(&self) -> Option<&crate::workspace_changes::ChangedFile> {
        let choices = self.choices();
        let index = choices.get(self.selected)?.value.parse::<usize>().ok()?;
        self.workspace_changes.as_ref()?.files.get(index)
    }

    pub fn selected_attachment(&self) -> Option<&crate::attachments::Attachment> {
        if self.overlay != Some(Overlay::Attachments) {
            return None;
        }
        let choices = self.choices();
        let index = choices.get(self.selected)?.value.parse::<usize>().ok()?;
        self.attachments.get(index)
    }
}

pub fn default_commands() -> Vec<Command> {
    [
        ("/new", "Save and start a new conversation"),
        ("/sessions", "Find and resume a saved conversation"),
        ("/drafts", "Choose saved input to restore"),
        ("/resume", "Resume a session by ID or prefix"),
        ("/rewind", "Return to an earlier message and edit it"),
        ("/tree", "Explore and continue any conversation branch"),
        ("/tools", "Browse tool calls and their full output"),
        (
            "/changes",
            "Inspect Git workspace changes and conversation edits",
        ),
        ("/queue", "Edit or remove messages waiting to be sent"),
        ("/steer", "Send a correction to the running task (Alt+S)"),
        ("/attach", "Search workspace files or attach a file path"),
        (
            "/attachments",
            "Review or remove the current draft's attachments",
        ),
        (
            "/reconnect",
            "Reconnect the backend and restore this conversation",
        ),
        (
            "/retry",
            "Restore interrupted input for editing before resending",
        ),
        ("/undo", "Review and undo the last file rollback"),
        ("/model", "Choose a configured model for this session"),
        ("/mode", "Choose Code or Plan mode"),
        ("/plan", "Enter planning mode with execution disabled"),
        ("/code", "Return to Code mode"),
        ("/permissions", "Choose how execution approval is requested"),
        ("/tasks", "Inspect plan steps and background jobs (Alt+T)"),
        ("/stop", "Stop a selected background job by its task ID"),
        (
            "/search",
            "Find a message or tool output in this conversation",
        ),
        ("/status", "Show workspace, model and session"),
        ("/logs", "Inspect backend diagnostics"),
        ("/help", "Keyboard shortcuts and commands"),
        ("/cancel", "Stop the current request"),
        ("/exit", "Save and exit"),
    ]
    .into_iter()
    .map(|(name, description)| Command {
        name: name.into(),
        description: description.into(),
    })
    .collect()
}

/// Keep untrusted output from injecting terminal control sequences.
pub fn clean(text: &str) -> String {
    TerminalTextFilter::default().feed(text)
}

#[derive(Default)]
pub(crate) struct TerminalTextFilter {
    state: u8,
    osc: bool,
}

impl TerminalTextFilter {
    pub(crate) fn feed(&mut self, text: &str) -> String {
        let mut visible = String::new();
        for character in text.chars() {
            if matches!(self.state, 3 | 4) {
                if matches!(character, '\u{18}' | '\u{1a}' | '\u{9c}')
                    || (self.osc && character == '\u{7}')
                    || (self.state == 4 && character == '\\')
                {
                    self.state = 0;
                } else {
                    self.state = if character == '\u{1b}' { 4 } else { 3 };
                }
                continue;
            }
            match character {
                '\u{1b}' => {
                    self.state = 1;
                    continue;
                }
                '\u{9b}' => {
                    self.state = 2;
                    continue;
                }
                '\u{90}' | '\u{98}' | '\u{9d}' | '\u{9e}' | '\u{9f}' => {
                    self.state = 3;
                    self.osc = character == '\u{9d}';
                    continue;
                }
                '\u{18}' | '\u{1a}' | '\u{9c}' => {
                    self.state = 0;
                    continue;
                }
                _ => {}
            }
            if self.state == 5 {
                if ('0'..='~').contains(&character) {
                    self.state = 0;
                }
                continue;
            }
            if self.state == 1 {
                if character == '[' {
                    self.state = 2;
                    continue;
                }
                if matches!(character, 'P' | 'X' | ']' | '^' | '_') {
                    self.state = 3;
                    self.osc = character == ']';
                    continue;
                }
                if (' '..='/').contains(&character) {
                    self.state = 5;
                    continue;
                }
                if ('0'..='~').contains(&character) {
                    self.state = 0;
                    continue;
                }
            } else if self.state == 2 {
                if ('@'..='~').contains(&character) {
                    self.state = 0;
                    continue;
                }
                if (' '..='?').contains(&character) {
                    continue;
                }
            }
            if character == '\n'
                || character == '\t'
                || (!character.is_control()
                    && !matches!(character, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'))
            {
                self.state = 0;
                visible.push(character);
            }
        }
        visible
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_control_payloads_and_handles_split_sequences() {
        let examples = [
            ("你好\tthere\nnext\r\0\x07\x08\x7f", "你好\tthere\nnext"),
            ("\x1b[2J\x1b[1;1H\x1b[31mred\x1b[0m", "red"),
            ("\x1b]52;c;c2VjcmV0\x07visible", "visible"),
            (
                "\x1b]8;;https://example.com\x1b\\label\x1b]8;;\x1b\\",
                "label",
            ),
            ("a\x1bPprivate payload\x1b\\b\x1b^hidden\x1b\\c", "abc"),
            ("\u{9d}52;c;hidden\u{9c}\u{9b}31mred\u{9b}0m", "red"),
            ("\x1b(Bnormal\x1b(P text\x1b7 saved", "normal text saved"),
            ("visible\x1b]52;c;unterminated", "visible"),
        ];
        for (source, expected) in examples {
            assert_eq!(clean(source), expected);
            let mut filter = TerminalTextFilter::default();
            let actual: String = source
                .chars()
                .map(|c| filter.feed(&c.to_string()))
                .collect();
            assert_eq!(actual, expected);
        }
    }
}

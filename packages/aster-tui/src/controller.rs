// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Conversation transitions are independent of terminal drawing and process IO.
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent};
use ratatui::layout::Rect;
use serde_json::{json, Value};
use tui_textarea::{CursorMove, TextArea};

use crate::drafts::{
    is_provisional_scope, provisional_scope, DraftSnapshot, DraftStore, LoadedDraft, MessageDraft,
};
use crate::input_history::InputHistory;
use crate::state::{
    clean, default_commands, ApprovalOption, ApprovalPreview, ApprovalRequest, Command, Entry,
    Overlay, Phase, RemovedAttachment, RewindPoint, Session, State, SteeringMessage, SteeringState,
    Tool,
};

const CANCEL_TIMEOUT: Duration = Duration::from_secs(20);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(20);
const EXIT_CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(2);
const DOUBLE_ESCAPE_TIMEOUT: Duration = Duration::from_millis(600);
const TREE_SEARCH_DEBOUNCE: Duration = Duration::from_millis(150);
const DRAFT_CLEARED_STATUS: &str = "Draft cleared · Ctrl+Z to undo";

#[derive(Clone, Debug)]
pub enum ListPurpose {
    StartNew,
    Browse,
    Resume(String),
    Continue,
}

#[derive(Clone, Debug)]
pub enum Pending {
    Initialize,
    Prepare,
    Prompt,
    Steer {
        session: String,
        generation: u64,
        turn_id: String,
        message_id: String,
        text: String,
        draft: String,
    },
    Runtime {
        session: String,
        generation: u64,
        operation: RuntimeRequest,
    },
    List(ListPurpose),
    Checkpoints {
        source: String,
        generation: u64,
    },
    Rewind {
        source: String,
        generation: u64,
    },
    Tree {
        source: String,
        generation: u64,
    },
    TreeSearch {
        source: String,
        generation: u64,
        view: u64,
        query: String,
    },
    NavigateTree {
        source: String,
        generation: u64,
        node_id: String,
    },
    Load {
        id: String,
        previous: Option<String>,
    },
    TreeCleanup {
        source: Option<String>,
        generation: u64,
    },
    NewClose,
    OldClose,
    ExitClose,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeRequest {
    Inspect,
    Configure,
    Tasks,
    StopTask,
}

#[derive(Debug)]
pub enum RollbackJob {
    Prepare {
        file: crate::workspace_changes::ChangedFile,
        hunk: Option<usize>,
    },
    Apply(crate::rollback::RollbackPlan),
    FindUndo,
    Undo(crate::rollback::RollbackReceipt),
}

pub enum RollbackOutcome {
    Prepared(crate::rollback::RollbackPlan),
    Applied(crate::rollback::RollbackReceipt),
    UndoReady(Option<crate::rollback::RollbackReceipt>),
    Undone(String),
}

#[derive(Debug)]
pub enum Action {
    PasteClipboard,
    CopySelection {
        text: String,
        cut: bool,
    },
    SearchFiles {
        query: String,
        generation: u64,
    },
    AttachFile {
        path: String,
        generation: u64,
    },
    OpenOriginalImage {
        attachment: crate::attachments::Attachment,
        generation: u64,
    },
    Reconnect,
    Reply {
        id: Value,
        result: Value,
    },
    LoadWorkspaceChanges,
    Rollback {
        job: RollbackJob,
        generation: u64,
    },
    Request {
        method: &'static str,
        params: Value,
        pending: Pending,
    },
    Notify {
        method: &'static str,
        params: Value,
    },
    Exit,
}

struct Backup {
    source_session: Option<String>,
    source_generation: u64,
    entries: Vec<Entry>,
    expanded_entries: HashSet<usize>,
    title: String,
    usage: String,
    ratio: Option<f64>,
    cost: String,
    commands: Vec<Command>,
    scroll: usize,
    elapsed: Option<f64>,
    runtime: Value,
    tasks: Value,
    model: String,
    sandbox: bool,
    mode: String,
    approval: String,
    steering: Vec<SteeringMessage>,
    message_streams: HashMap<(bool, String), MessageStream>,
    updates: Vec<SourceUpdate>,
}

enum SourceUpdate {
    Notification(Value),
    Runtime {
        operation: RuntimeRequest,
        response: Value,
    },
}

struct MessageStream {
    index: usize,
    filter: crate::state::TerminalTextFilter,
    complete: bool,
}

pub struct Controller {
    pub state: State,
    pending: HashMap<u64, Pending>,
    startup: Option<ListPurpose>,
    starting: bool,
    opening: Option<String>,
    backup: Option<Backup>,
    operation: &'static str,
    cancelled: bool,
    cancel_timed_out: bool,
    exiting: bool,
    deadline: Option<Instant>,
    cursors: HashSet<String>,
    history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<TextArea<'static>>,
    history_end: usize,
    history_store: Option<InputHistory>,
    history_import_requested: bool,
    message_streams: HashMap<(bool, String), MessageStream>,
    rewind_supported: bool,
    /// Committed to the editor only after the fork's replay succeeds.
    rewind_prompt: Option<(String, String)>,
    tree_supported: bool,
    tree_view: u64,
    tree_search_query: String,
    tree_search_due: Option<Instant>,
    tree_search_request: Option<(u64, String)>,
    /// The selected node is committed only after its conversation replay succeeds.
    tree_prompt: Option<(String, Option<String>)>,
    id_prefix: String,
    next_local_id: u64,
    draft_store: Option<DraftStore>,
    draft_revisions: HashMap<Option<String>, u64>,
    last_draft: Option<(Option<String>, DraftSnapshot)>,
    in_flight: Option<MessageDraft>,
    draft_scope: Option<Option<String>>,
    recovering_session: bool,
    draft_to_restore: Option<LoadedDraft>,
    draft_save_failed: bool,
    draft_retry_at: Option<Instant>,
    steering_drafts: HashMap<String, String>,
    pub fatal_error: Option<String>,
}

fn request(method: &'static str, params: Value, pending: Pending) -> Action {
    Action::Request {
        method,
        params,
        pending,
    }
}

fn tool_content_text(item: &Value) -> Option<&str> {
    let text = item["content"]["text"].as_str()?;
    if item["content"]["_meta"]["aster"]["format"] != "fenced-code" {
        return Some(text);
    }
    // Strip only the envelope explicitly marked by the backend. Literal
    // terminal output and any inner backticks must remain byte-for-byte text.
    let Some((opening, rest)) = text.split_once('\n') else {
        return Some(text);
    };
    let opening = opening.strip_suffix('\r').unwrap_or(opening);
    let fence_len = opening
        .chars()
        .take_while(|character| *character == '`')
        .count();
    if fence_len < 3 {
        return Some(text);
    }
    let fence = &opening[..fence_len];
    let Some((body, closing)) = rest.rsplit_once('\n') else {
        return Some(text);
    };
    if closing.strip_suffix('\r').unwrap_or(closing) == fence {
        Some(body)
    } else {
        Some(text)
    }
}

impl Controller {
    pub fn new(state: State, resume: Option<String>, continue_last: bool) -> Self {
        Self {
            state,
            pending: HashMap::new(),
            startup: resume
                .map(ListPurpose::Resume)
                .or_else(|| continue_last.then_some(ListPurpose::Continue)),
            starting: true,
            opening: None,
            backup: None,
            operation: "initialize",
            cancelled: false,
            cancel_timed_out: false,
            exiting: false,
            deadline: Some(Instant::now() + Duration::from_secs(60)),
            cursors: HashSet::new(),
            history: Vec::new(),
            history_index: None,
            history_draft: None,
            history_end: 0,
            history_store: None,
            history_import_requested: false,
            message_streams: HashMap::new(),
            rewind_supported: false,
            rewind_prompt: None,
            tree_supported: false,
            tree_view: 0,
            tree_search_query: String::new(),
            tree_search_due: None,
            tree_search_request: None,
            tree_prompt: None,
            id_prefix: format!(
                "tui-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            next_local_id: 0,
            draft_store: None,
            draft_revisions: HashMap::new(),
            last_draft: None,
            in_flight: None,
            draft_scope: None,
            recovering_session: false,
            draft_to_restore: None,
            draft_save_failed: false,
            draft_retry_at: None,
            steering_drafts: HashMap::new(),
            fatal_error: None,
        }
    }

    /// Input recall is workspace-local and does not resume an agent conversation.
    pub fn attach_history(&mut self, store: InputHistory) {
        match store.load() {
            Ok(history) => {
                self.history_import_requested = history.is_empty();
                self.history = history;
            }
            Err(error) => self
                .state
                .notice(format!("Could not load input history: {error}"), true),
        }
        self.history_store = Some(store);
    }

    pub fn initialize(&self) -> Action {
        request(
            "initialize",
            json!({"protocolVersion": 1, "clientCapabilities": {},
            "_meta":{"aster":{"messageStreaming":true,"runtimeControls":true,"structuredProgress":true}},
            "clientInfo": {"name":"aster", "version":env!("CARGO_PKG_VERSION")}}),
            Pending::Initialize,
        )
    }

    pub fn refresh_changes(&mut self) -> Vec<Action> {
        if self.state.changes_loading {
            return vec![];
        }
        self.state.changes_loading = true;
        self.state.changes_error = None;
        vec![Action::LoadWorkspaceChanges]
    }

    fn open_files(&mut self) -> Vec<Action> {
        self.state.open(Overlay::Files);
        self.search_files()
    }

    pub fn search_files(&mut self) -> Vec<Action> {
        self.state.files_loading = true;
        self.state.files_notice.clear();
        vec![Action::SearchFiles {
            query: self.state.query.clone(),
            generation: self.state.conversation_generation,
        }]
    }

    pub fn files_result(
        &mut self,
        generation: u64,
        query: &str,
        result: anyhow::Result<crate::attachments::FileSearch>,
    ) {
        if generation != self.state.conversation_generation || self.state.query != query {
            return;
        }
        self.state.files_loading = false;
        match result {
            Ok(result) => {
                self.state.file_candidates = result.files;
                self.state.files_notice = if result.truncated {
                    "Showing a bounded file list; type more of the path to narrow it.".into()
                } else {
                    String::new()
                };
            }
            Err(error) => {
                self.state.file_candidates.clear();
                self.state.files_notice = clean(&format!("Could not search files: {error}"));
            }
        }
        self.state.selected = self
            .state
            .selected
            .min(self.state.file_candidates.len().saturating_sub(1));
    }

    fn attach_file(&mut self, path: String) -> Vec<Action> {
        if self.state.attachment_loading {
            self.state
                .notice("Another attachment is still being read.", false);
            return vec![];
        }
        self.state.attachment_loading = true;
        self.state.overlay = None;
        vec![Action::AttachFile {
            path,
            generation: self.state.conversation_generation,
        }]
    }

    pub fn attachment_result(
        &mut self,
        generation: u64,
        result: anyhow::Result<crate::attachments::Attachment>,
    ) {
        self.state.attachment_loading = false;
        self.add_attachment(generation, result);
    }

    pub fn add_attachment(
        &mut self,
        generation: u64,
        result: anyhow::Result<crate::attachments::Attachment>,
    ) {
        if generation != self.state.conversation_generation {
            self.state.notice("The conversation changed before the attachment was ready. Attach it again in this conversation.", false);
            return;
        }
        match result {
            Ok(attachment) => {
                if self.state.attachments.contains(&attachment) {
                    return;
                }
                let mut attachments = self.state.attachments.clone();
                attachments.push(attachment);
                match crate::attachments::prompt(
                    &self.state.editor.lines().join("\n"),
                    &attachments,
                ) {
                    Ok(_) => {
                        self.state.draft_changed();
                        self.state.attachments = attachments;
                    }
                    Err(error) => self
                        .state
                        .notice(format!("Could not attach file: {error}"), true),
                }
            }
            Err(error) => self
                .state
                .notice(format!("Could not attach file: {error}"), true),
        }
    }

    pub fn remove_attachment(&mut self, index: usize) -> bool {
        if index >= self.state.attachments.len() {
            return false;
        }
        self.state.draft_changed();
        self.state.removed_attachment = Some(RemovedAttachment {
            attachment: self.state.attachments.remove(index),
            index,
            generation: self.state.conversation_generation,
        });
        if self.state.overlay == Some(Overlay::Attachments) {
            self.state.selected = self
                .state
                .selected
                .min(self.state.choices().len().saturating_sub(1));
            self.state.detail_scroll = 0;
        }
        true
    }

    pub fn open_attachment(&mut self, index: usize) -> bool {
        if index >= self.state.attachments.len() {
            return false;
        }
        self.state.open(Overlay::Attachments);
        self.state.selected = index;
        true
    }

    pub fn open_original_image(&self) -> Vec<Action> {
        if self.state.overlay != Some(Overlay::Attachments) {
            return vec![];
        }
        let Some(attachment) = self
            .state
            .selected_attachment()
            .filter(|attachment| attachment.is_image())
        else {
            return vec![];
        };
        vec![Action::OpenOriginalImage {
            attachment: attachment.clone(),
            generation: self.state.conversation_generation,
        }]
    }

    pub fn undo_attachment(&mut self) -> bool {
        if !self.state.can_undo_attachment() {
            return false;
        }
        let removed = self.state.removed_attachment.take().expect("checked undo");
        let index = removed.index.min(self.state.attachments.len());
        self.state.attachments.insert(index, removed.attachment);
        if let Err(error) = crate::attachments::validate(&self.state.attachments) {
            self.state.removed_attachment = Some(RemovedAttachment {
                attachment: self.state.attachments.remove(index),
                index: removed.index,
                generation: removed.generation,
            });
            self.state
                .notice(format!("Could not restore attachment: {error}"), true);
            return false;
        }
        self.state.draft_changed();
        if self.state.overlay == Some(Overlay::Attachments) {
            if let Some(selected) = self
                .state
                .choices()
                .iter()
                .position(|choice| choice.value == index.to_string())
            {
                self.state.selected = selected;
            }
            self.state.detail_scroll = 0;
        }
        true
    }

    fn open_changes(&mut self) -> Vec<Action> {
        self.state.open(Overlay::Changes);
        self.state.conversation_changes = false;
        self.refresh_changes()
    }

    pub fn changes_result(
        &mut self,
        result: anyhow::Result<crate::workspace_changes::WorkspaceChanges>,
    ) {
        self.state.changes_loading = false;
        match result {
            Ok(snapshot) => {
                self.state.workspace_changes = Some(snapshot);
                self.state.changes_error = None;
            }
            Err(error) => {
                self.state.workspace_changes = None;
                self.state.changes_error = Some(clean(&format!("{error:#}")));
            }
        }
        if self.state.overlay == Some(Overlay::Changes) && !self.state.conversation_changes {
            self.state.selected = self
                .state
                .selected
                .min(self.state.choices().len().saturating_sub(1));
            self.state.detail_scroll = 0;
        }
    }

    fn rollback_available(&mut self) -> bool {
        if self.state.phase != Phase::Ready
            || self.state.rollback.loading
            || self.background_running()
        {
            self.state.notice(
                "Finish or stop foreground and background work before rolling back files.",
                false,
            );
            return false;
        }
        true
    }

    fn open_rollback(&mut self) -> Vec<Action> {
        if !self.rollback_available() {
            return vec![];
        }
        if self.state.changes_loading || self.state.conversation_changes {
            self.state.notice(
                "Wait for the Git workspace snapshot, then choose an unstaged file.",
                false,
            );
            return vec![];
        }
        let Some(file) = self.state.selected_change().cloned() else {
            return vec![];
        };
        match crate::rollback::hunk_choices(&file) {
            Ok(hunks) => {
                self.state.rollback.file = Some(file);
                self.state.rollback.hunks = hunks;
                self.state.rollback.plan = None;
                self.state.open(Overlay::Rollback);
            }
            Err(error) => self
                .state
                .notice(format!("Cannot roll back this entry: {error}"), true),
        }
        vec![]
    }

    fn start_rollback_job(&mut self, job: RollbackJob, status: &str) -> Vec<Action> {
        self.state.rollback.loading = true;
        self.state.queue_paused = true;
        self.begin("rollback", status);
        vec![Action::Rollback {
            job,
            generation: self.state.conversation_generation,
        }]
    }

    fn prepare_rollback(&mut self, value: &str) -> Vec<Action> {
        if !self.rollback_available() {
            return vec![];
        }
        let Some(file) = self.state.rollback.file.clone() else {
            return vec![];
        };
        let hunk = if value == "all" {
            None
        } else {
            let Ok(index) = value.parse::<usize>() else {
                return vec![];
            };
            Some(index)
        };
        self.state.rollback.plan = None;
        self.state.open(Overlay::RollbackConfirm);
        self.start_rollback_job(
            RollbackJob::Prepare { file, hunk },
            "Preparing rollback preview…",
        )
    }

    fn open_undo(&mut self) -> Vec<Action> {
        if !self.rollback_available() {
            return vec![];
        }
        self.state.rollback.receipt = None;
        self.state.open(Overlay::UndoConfirm);
        self.start_rollback_job(RollbackJob::FindUndo, "Reading the last rollback backup…")
    }

    fn confirm_rollback(&mut self, undo: bool) -> Vec<Action> {
        if !self.rollback_available() {
            return vec![];
        }
        let job = if undo {
            let Some(receipt) = self.state.rollback.receipt.clone() else {
                return vec![];
            };
            RollbackJob::Undo(receipt)
        } else {
            let Some(plan) = self.state.rollback.plan.clone() else {
                return vec![];
            };
            RollbackJob::Apply(plan)
        };
        self.start_rollback_job(
            job,
            if undo {
                "Restoring the rollback backup…"
            } else {
                "Applying the reviewed rollback…"
            },
        )
    }

    pub fn rollback_result(
        &mut self,
        generation: u64,
        result: anyhow::Result<RollbackOutcome>,
    ) -> Vec<Action> {
        self.state.rollback.loading = false;
        if generation != self.state.conversation_generation {
            return vec![];
        }
        let mut refresh = false;
        match result {
            Ok(RollbackOutcome::Prepared(plan)) => self.state.rollback.plan = Some(plan),
            Ok(RollbackOutcome::UndoReady(Some(receipt))) => {
                if let Some(warning) = &receipt.warning {
                    self.state.notice(warning, true);
                }
                self.state.rollback.receipt = Some(receipt)
            }
            Ok(RollbackOutcome::UndoReady(None)) => {
                self.state.overlay = None;
                self.state.notice(
                    "No file rollback is available to undo in this workspace.",
                    false,
                );
            }
            Ok(RollbackOutcome::Applied(receipt)) => {
                if let Some(warning) = &receipt.warning {
                    self.state.notice(warning, true);
                }
                self.state.notice(
                    format!(
                        "Rolled back {}. Use /undo to restore it.\nBackup: {}",
                        receipt.path,
                        receipt.backup_path.display()
                    ),
                    false,
                );
                self.state.rollback.receipt = Some(receipt);
                self.state.rollback.plan = None;
                self.state.overlay = None;
                refresh = true;
            }
            Ok(RollbackOutcome::Undone(path)) => {
                self.state
                    .notice(format!("Restored {path} from the rollback backup."), false);
                self.state.rollback.receipt = None;
                self.state.overlay = None;
                refresh = true;
            }
            Err(error) => {
                self.state.rollback.plan = None;
                self.state.overlay = None;
                self.state.notice(format!("File operation was not completed: {error:#}. Refresh Changes and review again."), true);
                refresh = true;
            }
        }
        let mut actions = if self.operation == "rollback" {
            self.ready("Ready")
        } else {
            vec![]
        };
        if refresh && !self.exiting {
            actions.extend(self.refresh_changes());
        }
        actions
    }

    pub fn registered(&mut self, id: u64, pending: Pending) {
        self.pending.insert(id, pending);
    }

    pub fn failure(&mut self, message: impl Into<String>) -> Vec<Action> {
        let message = clean(&message.into());
        self.fatal_error = Some(message.clone());
        self.state.notice(&message, true);
        self.state.phase = Phase::Failed;
        self.state.queue_paused = true;
        self.state.status = "Backend disconnected · /reconnect · /retry".into();
        self.deadline = None;
        self.finish_tools("failed");
        self.state.active_turn_id = None;
        self.operation = "";
        if let Some(draft) = self.in_flight.take() {
            self.state.recovery = Some(draft);
        }
        self.finish_steering(SteeringState::Interrupted);
        self.pending.clear();
        self.tree_search_request = None;
        self.tree_search_due = None;
        if self.state.overlay == Some(Overlay::Tree) {
            self.state.tree_search_error =
                Some("Backend disconnected. Reconnect to search again.".into());
        }
        self.draft_to_restore = None;
        self.state.approvals.clear();
        if self.state.overlay == Some(Overlay::Approval) {
            self.state.overlay = None;
        }
        self.persist_draft();
        if self.exiting {
            vec![Action::Exit]
        } else {
            vec![]
        }
    }

    pub fn reconnect(&mut self) -> Vec<Action> {
        if self.state.busy() && !self.state.preparing() {
            self.state
                .notice("Stop the running task before reconnecting.", false);
            return vec![];
        }
        self.state.queue_paused = true;
        self.state.phase = Phase::Connecting;
        self.state.status = "Reconnecting… recovered messages remain paused".into();
        self.deadline = None;
        self.persist_draft();
        vec![Action::Reconnect]
    }

    pub fn reconnected(&mut self) -> Vec<Action> {
        self.pending.clear();
        self.starting = true;
        self.exiting = false;
        self.cancelled = false;
        self.cancel_timed_out = false;
        self.fatal_error = None;
        let durable = self.durable_session();
        self.recovering_session = durable.is_some();
        self.startup = durable.clone().map(ListPurpose::Resume);
        self.state.session_id = durable;
        self.state.conversation_generation = self.state.conversation_generation.wrapping_add(1);
        self.state.approvals.clear();
        self.state.tasks = Value::Null;
        self.state.runtime = Value::Null;
        self.state.approval_policy = "backend".into();
        self.operation = "initialize";
        self.state.phase = Phase::Connecting;
        self.deadline = Some(Instant::now() + Duration::from_secs(60));
        vec![self.initialize()]
    }

    /// A plain launch owns a new draft. Saved input is loaded only on request.
    pub fn attach_drafts(&mut self, store: DraftStore) {
        let loaded = match &self.startup {
            Some(ListPurpose::Resume(id)) => store.load(Some(id)),
            _ => Ok(None),
        };
        self.draft_store = Some(store);
        self.start_new_draft();
        match loaded {
            Ok(Some(loaded)) => {
                if !loaded.snapshot.is_empty() {
                    self.recovering_session = true;
                    self.restore_draft(loaded);
                } else {
                    if let Some(warning) = loaded.warning {
                        self.state.notice(warning, true);
                    }
                }
            }
            Ok(None) => {}
            Err(error) => self
                .state
                .notice(format!("Could not restore drafts: {error}"), true),
        }
        self.refresh_saved_drafts();
    }

    fn start_new_draft(&mut self) {
        self.draft_scope = Some(Some(provisional_scope(&self.local_id())));
        self.last_draft = None;
        self.draft_save_failed = false;
        self.draft_retry_at = None;
    }

    fn refresh_saved_drafts(&mut self) {
        let Some(store) = &self.draft_store else {
            return;
        };
        match store.list_pending() {
            Ok(drafts) => {
                self.state.saved_drafts = drafts
                    .into_iter()
                    .filter(|draft| self.draft_scope.as_ref() != Some(&draft.session_id))
                    .collect();
                if let Some(warning) = self
                    .state
                    .saved_drafts
                    .iter()
                    .find_map(|draft| draft.warning.clone())
                {
                    self.state.notice(warning, true);
                }
            }
            Err(error) => {
                self.state.saved_drafts.clear();
                self.state
                    .notice(format!("Could not list saved drafts: {error}"), true);
            }
        }
    }

    fn restore_saved_draft(&mut self, index: usize) -> Vec<Action> {
        if self.state.busy() || self.queue_blocks_switch() {
            return vec![];
        }
        let Some(scope) = self
            .state
            .saved_drafts
            .get(index)
            .map(|draft| draft.session_id.clone())
        else {
            return vec![];
        };
        let loaded = self
            .draft_store
            .as_ref()
            .expect("saved drafts have a store")
            .load(scope.as_deref());
        let loaded = match loaded {
            Ok(Some(loaded)) if !loaded.snapshot.is_empty() => loaded,
            Ok(_) => {
                self.refresh_saved_drafts();
                self.state
                    .notice("This draft is no longer available.", false);
                return vec![];
            }
            Err(error) => {
                self.state
                    .notice(format!("Could not restore draft: {error}"), true);
                return vec![];
            }
        };
        if !self.persist_draft() {
            return vec![];
        }
        if !is_provisional_scope(scope.as_deref()) {
            let id = scope.expect("saved session has an id");
            if self.state.session_id.as_deref() == Some(&id) {
                self.restore_draft(loaded);
                self.state.overlay = None;
                self.refresh_saved_drafts();
                return vec![];
            }
            let actions = self.resume(id);
            if !actions.is_empty() {
                self.draft_to_restore = Some(loaded);
            }
            return actions;
        }
        self.draft_to_restore = Some(loaded);
        self.state.overlay = None;
        if let Some(id) = &self.state.session_id {
            let action = request("session/close", json!({"sessionId":id}), Pending::NewClose);
            self.begin("close", "Saving conversation…");
            vec![action]
        } else {
            self.reset_conversation();
            self.finish_new_draft();
            self.prepare_session()
        }
    }

    fn finish_new_draft(&mut self) {
        self.in_flight = None;
        self.state.recovery = None;
        if let Some(loaded) = self.draft_to_restore.take() {
            let missing_session = !is_provisional_scope(loaded.session_id.as_deref());
            self.restore_draft(loaded);
            if missing_session {
                // Only a failed explicit recovery brings a session draft here.
                // Keep its original snapshot and save recovered edits separately.
                self.start_new_draft();
                self.state.notice("The saved conversation is unavailable. Recovered input is ready in a new conversation and remains paused for review.", false);
            }
        } else {
            self.start_new_draft();
        }
        self.refresh_saved_drafts();
    }

    fn restore_draft(&mut self, loaded: LoadedDraft) {
        self.draft_scope = Some(loaded.session_id.clone());
        self.draft_revisions
            .insert(loaded.session_id, loaded.revision);
        let snapshot = loaded.snapshot;
        if self.state.editor.lines().join("\n").is_empty() && self.state.attachments.is_empty() {
            self.set_draft(&snapshot.composer.text);
            self.state.attachments = snapshot.composer.attachments;
            self.state.editor.move_cursor(CursorMove::Jump(
                snapshot.cursor.0.min(u16::MAX as usize) as u16,
                snapshot.cursor.1.min(u16::MAX as usize) as u16,
            ));
        } else if (!snapshot.composer.text.is_empty() || !snapshot.composer.attachments.is_empty())
            && (snapshot.composer.text != self.state.editor.lines().join("\n")
                || snapshot.composer.attachments != self.state.attachments)
        {
            self.state.queued.push_back(snapshot.composer);
        }
        self.state.queued.extend(snapshot.queued);
        if let Some(in_flight) = snapshot.in_flight {
            if let Some(previous) = self.state.recovery.replace(in_flight.clone()) {
                if previous != in_flight && !self.state.queued.contains(&previous) {
                    self.state.queued.push_back(previous);
                }
            }
        }
        self.state.queue_paused = true;
        self.state.notice("Draft restored. Queued messages are paused; /retry recalls interrupted input for review.", false);
        if let Some(warning) = loaded.warning {
            self.state.notice(warning, true);
        }
    }

    /// Retry background storage failures at most once per second. Explicit
    /// dispatch and exit still call persist_draft to make one immediate attempt.
    pub fn autosave_draft(&mut self) -> bool {
        if self
            .draft_retry_at
            .is_some_and(|retry| Instant::now() < retry)
        {
            return false;
        }
        self.persist_draft()
    }

    fn draft_snapshot(&self) -> DraftSnapshot {
        let mut queued: Vec<_> = self.state.queued.iter().cloned().collect();
        // History browsing temporarily hides the unsent editor. Preserve that
        // original (including its cursor); recalled submissions remain in history.
        let editor = self
            .history_draft
            .as_ref()
            .filter(|_| self.history_index.is_some())
            .unwrap_or(&self.state.editor);
        let composer = editor.lines().join("\n");
        for steering in self
            .state
            .steering_messages
            .iter()
            .filter(|item| item.state.is_pending())
        {
            if composer != steering.text
                && composer.strip_prefix("/steer ") != Some(steering.text.as_str())
                && !queued.iter().any(|draft: &MessageDraft| {
                    draft.text == steering.text && draft.attachments.is_empty()
                })
            {
                queued.push(steering.text.clone().into());
            }
        }
        DraftSnapshot {
            composer: MessageDraft {
                text: composer,
                attachments: self.state.attachments.clone(),
            },
            cursor: editor.cursor(),
            queued,
            queue_paused: self.state.queue_paused,
            in_flight: self
                .in_flight
                .clone()
                .or_else(|| self.state.recovery.clone()),
            migrated_to: None,
        }
    }

    pub fn persist_draft(&mut self) -> bool {
        if self.draft_store.is_none() {
            return false;
        }
        // A prepared runtime ID is provisional, and startup list/load RPCs do
        // not choose where input is saved. Keep the owned scope until input is
        // committed to a conversation or a saved provisional draft is cleared.
        let Some(scope) = self.draft_scope.clone() else {
            return false;
        };
        let snapshot = self.draft_snapshot();
        // Opening an empty window must not publish a record or supersede saved input.
        if snapshot.is_empty() && !self.draft_revisions.contains_key(&scope) {
            return true;
        }
        if !self.draft_save_failed
            && self
                .last_draft
                .as_ref()
                .is_some_and(|(saved_scope, saved)| saved_scope == &scope && saved == &snapshot)
        {
            return true;
        }
        let store = self.draft_store.as_ref().expect("draft store exists");
        let revision = self.draft_revisions.get(&scope).copied();
        match store.save(scope.as_deref(), revision, &snapshot) {
            Ok(revision) => {
                self.draft_save_failed = false;
                self.draft_retry_at = None;
                if snapshot.is_empty() && scope.is_some() && is_provisional_scope(scope.as_deref())
                {
                    // The cleared record is durable and may now be reclaimed.
                    // Future input must not retain a revision for a removed scope.
                    self.draft_revisions.remove(&scope);
                    self.start_new_draft();
                } else {
                    self.draft_revisions.insert(scope.clone(), revision);
                    self.draft_scope = Some(scope.clone());
                    self.last_draft = Some((scope, snapshot));
                }
                true
            }
            Err(error) => {
                if !self.draft_save_failed {
                    self.state.notice(
                        format!(
                            "Could not save drafts: {error}. Your input is still available here."
                        ),
                        true,
                    );
                }
                self.draft_save_failed = true;
                self.draft_retry_at = Some(Instant::now() + Duration::from_secs(1));
                // Failed snapshots still retry even if the user stops typing.
                // The failure flag deduplicates notices until storage recovers.
                self.last_draft = Some((scope, snapshot));
                false
            }
        }
    }

    fn durable_session(&self) -> Option<String> {
        let id = self.state.session_id.as_ref()?;
        (self.in_flight.is_some()
            || self.state.sessions.iter().any(|session| &session.id == id)
            || self
                .state
                .entries
                .iter()
                .any(|entry| matches!(entry, Entry::User(_))))
        .then(|| id.clone())
    }

    fn load_draft_scope(&mut self, scope: Option<String>) -> bool {
        if self.draft_scope.as_ref() == Some(&scope) {
            return true;
        }
        let Some(store) = &self.draft_store else {
            return false;
        };
        match store.load(scope.as_deref()) {
            Ok(Some(loaded)) if !loaded.snapshot.is_empty() => self.restore_draft(loaded),
            Ok(Some(loaded)) => {
                self.draft_revisions.insert(scope.clone(), loaded.revision);
                self.draft_scope = Some(scope);
            }
            Ok(None) => self.draft_scope = Some(scope),
            Err(error) => {
                // Never adopt a revision without consuming its saved input.
                self.draft_scope = None;
                self.state.notice(
                    format!("Could not restore this session's draft: {error}"),
                    true,
                );
                return false;
            }
        }
        self.last_draft = None;
        self.draft_save_failed = false;
        self.draft_retry_at = None;
        true
    }

    fn bind_prompt_draft(&mut self, draft: &MessageDraft) -> bool {
        let previous = self.draft_scope.clone();
        if self.draft_store.is_some() && !self.load_draft_scope(self.state.session_id.clone()) {
            return false;
        }
        // Loading a scope can restore an earlier interrupted input. Keep it
        // alongside the new request before publishing the in-flight snapshot.
        if let Some(recovered) = self.state.recovery.take() {
            self.state.queued.push_back(recovered);
            self.state.queue_paused = true;
        }
        self.in_flight = Some(draft.clone());
        if self.draft_store.is_none() {
            return true;
        }
        if !self.persist_draft() {
            self.in_flight = None;
            return false;
        }
        if let Some(previous) = previous.filter(|scope| {
            is_provisional_scope(scope.as_deref())
                && self.draft_scope.as_ref() != Some(scope)
                && self.draft_revisions.contains_key(scope)
        }) {
            // Publish the complete session snapshot before consuming the
            // provisional draft. A failed CAS keeps the other writer's input.
            let store = self.draft_store.as_ref().expect("draft store exists");
            match store.save(
                previous.as_deref(),
                self.draft_revisions.get(&previous).copied(),
                &DraftSnapshot {
                    migrated_to: self.state.session_id.clone(),
                    ..Default::default()
                },
            ) {
                Ok(revision) => {
                    self.draft_revisions.insert(previous, revision);
                }
                Err(error) => self.state.notice(
                    format!("The earlier draft was kept because it changed elsewhere: {error}"),
                    true,
                ),
            }
        }
        true
    }

    fn prepare_recovered_session(&mut self) -> Vec<Action> {
        if self.draft_to_restore.is_some() {
            self.recovering_session = false;
            if !self.persist_draft() {
                self.draft_to_restore = None;
                return self.ready("Draft recovery paused");
            }
            if let Some(id) = &self.state.session_id {
                let action = request("session/close", json!({"sessionId":id}), Pending::NewClose);
                self.begin("close", "Preparing to recover input in a new conversation…");
                return vec![action];
            }
            self.reset_conversation();
            self.finish_new_draft();
            return self.prepare_session();
        }
        self.persist_draft();
        self.state.session_id = None;
        self.recovering_session = false;
        self.reset_conversation();
        self.start_new_draft();
        self.state.queue_paused = true;
        self.state.notice("The saved conversation is unavailable. Preparing a new session; recovered input remains paused for review.", false);
        self.prepare_session()
    }

    fn retry_input(&mut self) -> Vec<Action> {
        let Some(draft) = self.state.recovery.take() else {
            self.state.notice("No interrupted input to restore.", false);
            return vec![];
        };
        self.restore_message(draft);
        if self.state.phase == Phase::Failed {
            self.reconnect()
        } else {
            self.state.notice(
                "Interrupted input restored for review. Enter sends it when ready.",
                false,
            );
            vec![]
        }
    }

    fn restore_message(&mut self, draft: MessageDraft) {
        if self.state.editor.lines().join("\n").is_empty() && self.state.attachments.is_empty() {
            self.set_draft(&draft.text);
            self.state.attachments = draft.attachments;
        } else {
            self.state.queued.push_front(draft);
            self.state.queue_paused = true;
            self.state
                .notice("Recovered input is paused in the queue (Ctrl+E).", false);
        }
    }

    pub fn diagnostic(&mut self, message: String) {
        let text = clean(&message);
        if !text.trim().is_empty() {
            self.state.diagnostics.push_back(text);
            while self.state.diagnostics.len() > 200 {
                self.state.diagnostics.pop_front();
            }
        }
    }

    pub fn tick(&mut self) -> Vec<Action> {
        let now = Instant::now();
        if self.state.exit_confirmation.is_some_and(|time| now >= time) {
            self.state.exit_confirmation = None;
        }
        if self
            .state
            .escape_confirmation
            .is_some_and(|time| now >= time)
        {
            self.state.escape_confirmation = None;
        }
        if self.deadline.is_some_and(|time| now >= time) {
            let message = if self.exiting {
                "Closing timed out; stopping the backend.".to_owned()
            } else if self.state.preparing() {
                self.deadline = None;
                self.state.queue_paused = true;
                self.state.status = "Preparation is taking longer than expected. Your messages are kept; press Ctrl+C twice to exit.".into();
                self.state.notice(self.state.status.clone(), true);
                return vec![];
            } else if self.cancelled {
                self.deadline = None;
                self.cancel_timed_out = true;
                self.state.queue_paused = true;
                self.state.status = "The task is still stopping. You can keep waiting or press Ctrl+C twice to exit.".into();
                self.state.notice(self.state.status.clone(), false);
                return vec![];
            } else {
                "The backend operation timed out. Restart Aster to reconnect.".to_owned()
            };
            return self.failure(message);
        }
        self.search_tree(now)
    }

    fn limit_deadline(&mut self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        self.deadline = Some(
            self.deadline
                .map_or(deadline, |current| current.min(deadline)),
        );
    }

    fn begin(&mut self, operation: &'static str, status: &str) {
        self.state.escape_confirmation = None;
        self.operation = operation;
        self.state.phase = if self.starting || operation == "prepare" {
            Phase::Preparing
        } else {
            Phase::Busy
        };
        self.state.status = status.into();
        self.cancelled = false;
        self.cancel_timed_out = false;
        self.deadline = if operation == "prompt" {
            None
        } else {
            Some(Instant::now() + Duration::from_secs(60))
        };
    }

    fn ready(&mut self, status: &str) -> Vec<Action> {
        self.starting = false;
        self.operation = "";
        self.state.phase = Phase::Ready;
        self.state.status = status.into();
        self.deadline = None;
        self.cancel_timed_out = false;
        if self.exiting {
            self.close_for_exit()
        } else {
            Vec::new()
        }
    }

    fn reset_conversation(&mut self) {
        self.state.escape_confirmation = None;
        self.state.rewind_points.clear();
        self.state.rewind_loading = false;
        self.state.session_tree = crate::session_tree::SessionTree::default();
        self.state.tree_loading = false;
        self.state.tree_search_error = None;
        self.tree_view = self.tree_view.wrapping_add(1);
        self.tree_search_query.clear();
        self.tree_search_due = None;
        self.state.active_turn_id = None;
        self.state.steering_messages.clear();
        self.steering_drafts.clear();
        self.state.tasks = Value::Null;
        self.state.runtime = Value::Null;
        self.state.mode = "code".into();
        self.state.approval_policy = "backend".into();
        self.state.approvals.clear();
        if !self.starting {
            self.state.draft_changed();
            self.state.conversation_generation = self.state.conversation_generation.wrapping_add(1);
        }
        self.message_streams.clear();
        self.state.entries.clear();
        self.state.expanded_entries.clear();
        self.state.title.clear();
        self.state.usage.clear();
        self.state.cost.clear();
        self.state.context_ratio = None;
        self.state.last_elapsed = None;
        self.state.started = None;
        self.state.scroll = 0;
        self.state.rendered_rows = None;
        self.state.focus_entry = None;
        self.state.search_highlight.clear();
        self.state.queue_paused = false;
        self.state.commands = default_commands();
    }

    fn list(&mut self, purpose: ListPurpose) -> Vec<Action> {
        self.begin("sessions", "Loading conversations…");
        self.state.sessions.clear();
        self.cursors.clear();
        let mut params = json!({"cwd":self.state.workspace});
        if self.history_import_requested {
            params["_meta"] = json!({"aster":{"includeInputHistory":true}});
        }
        vec![request("session/list", params, Pending::List(purpose))]
    }

    pub fn resume(&mut self, id: String) -> Vec<Action> {
        if self.state.busy() {
            self.state
                .notice("Finish or cancel the current operation first.", false);
            return vec![];
        }
        if !self.starting && self.state.session_id.as_deref() == Some(&id) {
            self.state.overlay = None;
            return vec![];
        }
        if !self.starting && self.queue_blocks_switch() {
            return vec![];
        }
        self.draft_to_restore = None;
        self.persist_draft();
        self.backup = Some(Backup {
            source_session: self.state.session_id.clone(),
            source_generation: self.state.conversation_generation,
            entries: std::mem::take(&mut self.state.entries),
            expanded_entries: self.state.expanded_entries.clone(),
            title: self.state.title.clone(),
            usage: self.state.usage.clone(),
            ratio: self.state.context_ratio,
            cost: self.state.cost.clone(),
            commands: self.state.commands.clone(),
            scroll: self.state.scroll,
            elapsed: self.state.last_elapsed,
            runtime: self.state.runtime.clone(),
            tasks: self.state.tasks.clone(),
            model: self.state.model.clone(),
            sandbox: self.state.sandbox,
            mode: self.state.mode.clone(),
            approval: self.state.approval_policy.clone(),
            steering: self.state.steering_messages.clone(),
            message_streams: std::mem::take(&mut self.message_streams),
            updates: Vec::new(),
        });
        self.message_streams.clear();
        let queue_paused = self.state.queue_paused;
        self.reset_conversation();
        if self.starting {
            self.state.queue_paused = queue_paused;
        }
        self.state.title = self
            .state
            .sessions
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.title.clone())
            .unwrap_or_default();
        self.opening = Some(id.clone());
        self.state.overlay = None;
        self.begin("load", "Restoring conversation…");
        vec![request(
            "session/load",
            json!({"cwd":self.state.workspace,"sessionId":id,"mcpServers":[]}),
            Pending::Load {
                id,
                previous: self.state.session_id.clone(),
            },
        )]
    }

    fn restore_load_backup(&mut self) {
        self.rewind_prompt = None;
        self.tree_prompt = None;
        self.message_streams.clear();
        self.opening = None;
        if let Some(backup) = self.backup.take() {
            self.state.entries = backup.entries;
            self.state.expanded_entries = backup.expanded_entries;
            self.state.title = backup.title;
            self.state.usage = backup.usage;
            self.state.context_ratio = backup.ratio;
            self.state.cost = backup.cost;
            self.state.commands = backup.commands;
            self.state.scroll = backup.scroll;
            self.state.last_elapsed = backup.elapsed;
            self.state.runtime = backup.runtime;
            self.state.tasks = backup.tasks;
            self.state.model = backup.model;
            self.state.sandbox = backup.sandbox;
            self.state.mode = backup.mode;
            self.state.approval_policy = backup.approval;
            self.state.steering_messages = backup.steering;
            self.message_streams = backup.message_streams;
            for update in backup.updates {
                match update {
                    SourceUpdate::Notification(params) => self.update(&params),
                    SourceUpdate::Runtime {
                        operation,
                        response,
                    } => {
                        self.apply_runtime_response(operation, &response);
                    }
                }
            }
        }
        self.state.rendered_rows = None;
    }

    fn reject_loaded_tree(&mut self, target: String, status: &'static str) -> Vec<Action> {
        // The original draft scope and input have not been consumed yet. Only
        // the replayed conversation display needs rolling back.
        self.restore_load_backup();
        self.draft_to_restore = None;
        self.state.queue_paused = true;
        self.state.notice(status, false);
        // Cleanup is independent: its acknowledgement must not finish a newer
        // operation, fail the source's tools, or dispatch the retained queue.
        let mut actions = self.ready(status);
        actions.push(request(
            "session/close",
            json!({"sessionId":target}),
            Pending::TreeCleanup {
                source: self.state.session_id.clone(),
                generation: self.state.conversation_generation,
            },
        ));
        actions
    }

    pub fn input(&mut self, text: String) -> Vec<Action> {
        let trimmed = text.trim();
        if trimmed.is_empty() && self.state.attachments.is_empty() {
            return vec![];
        }
        let (command, argument) = trimmed
            .split_once(char::is_whitespace)
            .unwrap_or((trimmed, ""));
        let argument = argument.trim();
        match command {
            "/model" => {
                return if argument.is_empty() {
                    self.open_runtime(Overlay::Models)
                } else {
                    self.configure(json!({"model":argument}))
                }
            }
            "/mode" => {
                return if argument.is_empty() {
                    self.open_runtime(Overlay::Modes)
                } else {
                    self.configure(json!({"mode":argument}))
                }
            }
            "/plan" => return self.configure(json!({"mode":"plan"})),
            "/code" => return self.configure(json!({"mode":"code"})),
            "/permissions" => {
                return if argument.is_empty() {
                    self.open_runtime(Overlay::Permissions)
                } else {
                    self.configure(json!({"approval":argument}))
                }
            }
            "/tasks" => return self.open_tasks(),
            "/stop" => {
                return if argument.is_empty() {
                    self.open_tasks()
                } else {
                    self.stop_task(argument)
                }
            }
            "/attach" => {
                if argument.is_empty() {
                    return self.open_files();
                }
                return self.attach_file(argument.trim_matches('"').to_owned());
            }
            "/attachments" => {
                self.state.open(Overlay::Attachments);
                return vec![];
            }
            "/reconnect" => return self.reconnect(),
            "/retry" => return self.retry_input(),
            "/undo" => return self.open_undo(),
            "/steer" => {
                let payload = trimmed
                    .strip_prefix("/steer")
                    .unwrap_or("")
                    .trim_start_matches(char::is_whitespace);
                if payload.is_empty() {
                    self.insert_command("/steer");
                    return vec![];
                }
                return self.steer(payload.to_owned());
            }
            "/exit" | "/quit" => return self.quit(),
            "/help" | "/" => {
                self.state.open(Overlay::Help);
                return vec![];
            }
            "/tools" => {
                self.state.open(Overlay::Tools);
                return vec![];
            }
            "/changes" => {
                return self.open_changes();
            }
            "/queue" => {
                self.state.open(Overlay::Queue);
                return vec![];
            }
            "/drafts" => {
                self.refresh_saved_drafts();
                self.state.open(Overlay::Drafts);
                return vec![];
            }
            "/search" => {
                self.state.open(Overlay::Search);
                self.state.query = argument.into();
                return vec![];
            }
            "/logs" => {
                self.state.open(Overlay::Diagnostics);
                return vec![];
            }
            "/cancel" => return self.cancel(),
            "/status" => {
                self.state.notice(
                    format!(
                        "Workspace: {}\nModel: {}\nExecution: {}\nMode: {}\nApproval: {}\nSession: {}",
                        self.state.workspace,
                        self.state.model,
                        if self.state.sandbox {
                            "sandbox"
                        } else {
                            "native"
                        },
                        self.state.mode,
                        self.state.approval_policy,
                        self.state
                            .session_id
                            .as_deref()
                            .unwrap_or("new (saved on first message)")
                    ),
                    false,
                );
                return vec![];
            }
            _ => {}
        }
        if self.state.busy() {
            if !command.starts_with('/') && self.can_enqueue() {
                self.enqueue(text);
                return vec![];
            }
            self.state.notice(
                if self.state.preparing() {
                    "Still preparing. Your draft is kept; you can keep editing."
                } else {
                    "Still working. Your draft is kept; Esc stops the request."
                },
                false,
            );
            return vec![];
        }
        if self.state.phase == Phase::Failed {
            self.state.draft_changed();
            let attachments = std::mem::take(&mut self.state.attachments);
            self.restore_message(MessageDraft { text, attachments });
            self.state.notice(
                "The backend is disconnected. Use /reconnect; your input is kept.",
                false,
            );
            return vec![];
        }
        match command {
            "/new" => {
                if self.queue_blocks_switch() {
                    return vec![];
                }
                if self.draft_store.is_some() && !self.persist_draft() {
                    return vec![];
                }
                self.draft_to_restore = None;
                if let Some(id) = &self.state.session_id {
                    let action =
                        request("session/close", json!({"sessionId":id}), Pending::NewClose);
                    self.begin("close", "Saving conversation…");
                    vec![action]
                } else {
                    self.reset_conversation();
                    self.finish_new_draft();
                    self.prepare_session()
                }
            }
            "/sessions" | "/resume" if argument.is_empty() => {
                self.state.open(Overlay::Sessions);
                self.list(ListPurpose::Browse)
            }
            "/resume" => self.list(ListPurpose::Resume(argument.into())),
            "/rewind" => self.open_rewind(),
            "/tree" => self.open_tree(),
            _ if command.starts_with('/')
                && !self.state.commands.iter().any(|c| c.name == command) =>
            {
                self.state.notice(
                    format!("Unknown command: {command}. Use /help or Ctrl+K."),
                    true,
                );
                vec![]
            }
            _ => {
                if !self.state.queued.is_empty() {
                    if command.starts_with('/') {
                        self.queue_blocks_switch();
                        return vec![];
                    }
                    self.enqueue(text);
                    return self.send_queued();
                }
                self.state.queue_paused = false;
                self.state.draft_changed();
                let draft = MessageDraft {
                    text,
                    attachments: std::mem::take(&mut self.state.attachments),
                };
                self.submit(draft)
            }
        }
    }

    fn open_rewind(&mut self) -> Vec<Action> {
        if self.state.busy() {
            self.state
                .notice("Finish or cancel the current operation first.", false);
            return vec![];
        }
        if !self.rewind_supported {
            self.state.notice(
                "This backend does not support conversation rewind; update aster-acp.",
                true,
            );
            return vec![];
        }
        if self.queue_blocks_switch() || !self.rewind_draft_is_empty() {
            return vec![];
        }
        let Some(source) = self.state.session_id.clone() else {
            self.state
                .notice("Send a message before rewinding a conversation.", false);
            return vec![];
        };
        self.state.open(Overlay::Rewind);
        self.state.rewind_points.clear();
        self.state.rewind_loading = true;
        self.begin("checkpoints", "Loading rewind points…");
        vec![request(
            "_aster/session/checkpoints",
            json!({"sessionId":source}),
            Pending::Checkpoints {
                source,
                generation: self.state.conversation_generation,
            },
        )]
    }

    fn open_tree(&mut self) -> Vec<Action> {
        if self.state.busy() {
            self.state
                .notice("Finish or cancel the current operation first.", false);
            return vec![];
        }
        if !self.tree_supported {
            self.state.notice(
                "This backend does not support conversation trees; update aster-acp.",
                true,
            );
            return vec![];
        }
        let Some(source) = self.state.session_id.clone() else {
            self.state
                .notice("Send a message to start a conversation tree.", false);
            return vec![];
        };
        self.state.open(Overlay::Tree);
        self.tree_view = self.tree_view.wrapping_add(1);
        self.tree_search_query.clear();
        self.tree_search_due = None;
        self.state.tree_search_error = None;
        self.state.session_tree = crate::session_tree::SessionTree::default();
        self.state.tree_loading = true;
        self.begin("tree", "Loading conversation tree…");
        vec![request(
            "_aster/session/tree",
            json!({"sessionId":source}),
            Pending::Tree {
                source,
                generation: self.state.conversation_generation,
            },
        )]
    }

    fn navigate_tree(&mut self, node_id: String) -> Vec<Action> {
        if self.state.busy() || self.queue_blocks_switch() {
            return vec![];
        }
        if self.background_running() {
            self.state.notice(
                "Stop or finish background jobs before switching branches.",
                false,
            );
            return vec![];
        }
        let Some(node) = self.state.session_tree.node(&node_id) else {
            return vec![];
        };
        if !node.navigable {
            self.state.notice(
                "This message has no saved restore point. Choose another message.",
                false,
            );
            return vec![];
        }
        if self.draft_store.is_some() && !self.persist_draft() {
            return vec![];
        }
        let Some(source) = self.state.session_id.clone() else {
            return vec![];
        };
        self.state.overlay = None;
        self.state.queue_paused = true;
        self.begin("navigate", "Restoring selected conversation branch…");
        vec![request(
            "_aster/session/navigate",
            json!({"sessionId":source,"nodeId":node_id,
                "expectedCurrentNodeId":self.state.session_tree.current_node_id}),
            Pending::NavigateTree {
                source,
                generation: self.state.conversation_generation,
                node_id,
            },
        )]
    }

    fn tree_query_changed(&mut self) {
        self.tree_search_query = self.state.query.clone();
        self.state.tree_search_error = None;
        self.tree_search_due = None;
        if !self.state.tree_search_supported || self.state.query.trim().is_empty() {
            return;
        }
        if self.state.query.chars().count() > crate::session_tree::MAX_QUERY_CHARS {
            self.state.tree_search_error = Some("Search is limited to 4096 characters.".into());
        } else {
            self.tree_search_due = Some(Instant::now() + TREE_SEARCH_DEBOUNCE);
        }
    }

    fn search_tree(&mut self, now: Instant) -> Vec<Action> {
        if self.state.overlay != Some(Overlay::Tree) || !self.state.tree_search_supported {
            return vec![];
        }
        if self.tree_search_query != self.state.query {
            self.tree_query_changed();
        }
        if self.state.phase != Phase::Ready
            || !self.state.tree_search_pending()
            || self.tree_search_request.is_some()
            || self.tree_search_due.is_none_or(|due| now < due)
        {
            return vec![];
        }
        let Some(source) = self.state.session_id.clone() else {
            return vec![];
        };
        let query = self.state.query.clone();
        self.tree_search_request = Some((self.tree_view, query.clone()));
        vec![request(
            "_aster/session/tree",
            json!({"sessionId":source,"query":query}),
            Pending::TreeSearch {
                source,
                generation: self.state.conversation_generation,
                view: self.tree_view,
                query,
            },
        )]
    }

    fn tree_search_response(&mut self, pending: Pending, value: &Value) -> Vec<Action> {
        let Pending::TreeSearch {
            source,
            generation,
            view,
            query,
        } = pending
        else {
            return vec![];
        };
        if self.tree_search_request.as_ref() == Some(&(view, query.clone())) {
            self.tree_search_request = None;
        }
        if self.state.overlay != Some(Overlay::Tree)
            || self.state.session_id.as_ref() != Some(&source)
            || self.state.conversation_generation != generation
            || self.tree_view != view
            || self.state.query != query
            || self.state.tree_loading
        {
            return vec![];
        }
        if let Some(error) = value.get("error") {
            let reason = error["data"]["reason"]
                .as_str()
                .or_else(|| error["message"].as_str())
                .unwrap_or("The search request failed.");
            self.state.tree_search_error = Some(format!("Search failed: {}", clean(reason)));
            return vec![];
        }
        match crate::session_tree::SessionTree::from_response(&value["result"]) {
            Ok(tree) if tree.has_search_results(&query) => {
                self.state.session_tree = tree;
                self.state.tree_search_error = None;
                self.state.selected = 0;
            }
            Ok(_) => {
                self.state.tree_search_error =
                    Some("The backend returned results for a different search.".into());
            }
            Err(error) => {
                self.state.tree_search_error =
                    Some(format!("Could not display search results: {error}"));
            }
        }
        vec![]
    }

    fn rewind_draft_is_empty(&mut self) -> bool {
        if self.state.editor.lines().join("\n").is_empty() && self.state.attachments.is_empty() {
            return true;
        }
        self.state.status = "Keep or clear your current draft before rewinding.".into();
        self.state.notice(self.state.status.clone(), false);
        false
    }

    fn rewind(&mut self, user_message_index: usize) -> Vec<Action> {
        if self.state.busy() || self.queue_blocks_switch() || !self.rewind_draft_is_empty() {
            return vec![];
        }
        if !self
            .state
            .rewind_points
            .iter()
            .any(|point| point.user_message_index == user_message_index)
        {
            return vec![];
        }
        let Some(source) = self.state.session_id.clone() else {
            return vec![];
        };
        self.state.overlay = None;
        self.state.queue_paused = true;
        self.begin(
            "rewind",
            "Restoring conversation before the selected message…",
        );
        vec![request(
            "_aster/session/rewind",
            json!({"sessionId":source,"userMessageIndex":user_message_index}),
            Pending::Rewind {
                source,
                generation: self.state.conversation_generation,
            },
        )]
    }

    fn can_enqueue(&self) -> bool {
        !self.exiting && (self.operation == "prompt" || self.state.preparing())
    }

    fn enqueue(&mut self, text: String) {
        self.state.draft_changed();
        if self.state.preparing() && self.state.queued.is_empty() {
            self.state.queue_paused = false;
        }
        self.state.queued.push_back(MessageDraft {
            text,
            attachments: std::mem::take(&mut self.state.attachments),
        });
    }

    fn queue_blocks_switch(&mut self) -> bool {
        if self.state.queued.is_empty() {
            return false;
        }
        self.state.queue_paused = true;
        self.state.status = "Send or remove queued messages before switching conversations.".into();
        self.state.notice(self.state.status.clone(), false);
        self.state.open(Overlay::Queue);
        true
    }

    fn send_queued(&mut self) -> Vec<Action> {
        if self.exiting || self.state.phase != Phase::Ready || self.state.queue_paused {
            return vec![];
        }
        if self.state.session_id.is_none() && !self.state.queued.is_empty() {
            return self.prepare_session();
        }
        let selected_queue_index = if self.state.overlay == Some(Overlay::Queue) {
            self.state
                .choices()
                .get(self.state.selected)
                .and_then(|choice| choice.value.parse::<usize>().ok())
        } else {
            None
        };
        match self.state.queued.pop_front() {
            Some(text) => {
                let choices = self.state.choices();
                // Preserve the actual queue item across automatic dispatch, even
                // when the picker is filtered or multiple messages have equal text.
                self.state.selected = selected_queue_index
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| {
                        choices
                            .iter()
                            .position(|choice| choice.value.parse::<usize>().ok() == Some(index))
                    })
                    .unwrap_or_else(|| self.state.selected.min(choices.len().saturating_sub(1)));
                self.submit(text)
            }
            None => vec![],
        }
    }

    fn continue_queue(&mut self) -> Vec<Action> {
        if self.exiting || matches!(self.state.phase, Phase::Failed | Phase::Cancelling) {
            return vec![];
        }
        self.state.queue_paused = false;
        self.send_queued()
    }

    fn submit(&mut self, draft: MessageDraft) -> Vec<Action> {
        if self.state.attachment_loading {
            self.restore_message(draft);
            self.state.queue_paused = true;
            self.state.notice(
                "Reading the selected attachment. Review the complete draft before sending it.",
                false,
            );
            return vec![];
        }
        if self.state.session_id.is_none() {
            self.state.queued.push_front(draft);
            return self.prepare_session();
        }
        if !draft.attachments.is_empty() && !self.state.attachments_supported {
            self.restore_message(draft);
            self.state.queue_paused = true;
            self.state.notice("This backend does not accept file/image attachments; update aster-acp before sending.", true);
            return vec![];
        }
        if !self.state.image_input_supported
            && draft
                .attachments
                .iter()
                .any(|item| item.mime_type().starts_with("image/"))
        {
            self.restore_message(draft);
            self.state.queue_paused = true;
            self.state.notice(
                "This backend does not accept image input. The attachment is kept for review.",
                true,
            );
            return vec![];
        }
        let prompt = match crate::attachments::prompt(&draft.text, &draft.attachments) {
            Ok(prompt) => prompt,
            Err(error) => {
                self.restore_message(draft);
                self.state.queue_paused = true;
                self.state
                    .notice(format!("Could not prepare input: {error}"), true);
                return vec![];
            }
        };
        if !self.bind_prompt_draft(&draft) {
            self.restore_message(draft);
            self.state.queue_paused = true;
            self.state.notice(
                "Message not sent: its recovery copy could not be saved. Your input is kept for review.",
                true,
            );
            return vec![];
        }
        let text = &draft.text;
        if self.state.title.is_empty() {
            self.state.title = clean(text)
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(80)
                .collect();
        }
        if let Some(id) = &self.state.session_id {
            if !self.state.sessions.iter().any(|session| &session.id == id) {
                self.state.sessions.insert(
                    0,
                    Session {
                        id: id.clone(),
                        title: self.state.title.clone(),
                        updated: String::new(),
                    },
                );
            }
        }
        if let Some(store) = &self.history_store {
            if let Err(error) = store.append(text) {
                self.state
                    .notice(format!("Could not save input history: {error}"), true);
            }
        }
        self.history.push(text.clone());
        let mut shown = clean(text);
        if !draft.attachments.is_empty() {
            shown.push_str("\n\nAttached context:");
            for attachment in &draft.attachments {
                shown.push_str(&format!(
                    "\n@{} · {} · {} bytes",
                    clean(attachment.name()),
                    attachment.mime_type(),
                    attachment.byte_len()
                ));
            }
        }
        self.state.entries.push(Entry::User(shown));
        self.state.scroll = 0;
        self.state.focus_entry = None;
        self.state.started = Some(Instant::now());
        self.prompt(prompt)
    }

    fn prepare_session(&mut self) -> Vec<Action> {
        self.begin("prepare", "Preparing…");
        vec![request(
            "session/new",
            json!({"cwd":self.state.workspace,"mcpServers":[],
                "_meta":{"aster":{"prepare":true}}}),
            Pending::Prepare,
        )]
    }

    fn prepared(&mut self) -> Vec<Action> {
        self.recovering_session = false;
        let mut actions = self.ready("Ready");
        if actions.is_empty() {
            if self.state.runtime_supported {
                actions.extend(self.runtime_request(
                    "_aster/session/runtime",
                    json!({}),
                    RuntimeRequest::Inspect,
                ));
            }
            actions.extend(self.send_queued());
        }
        actions
    }

    fn recall_waiting_message(&mut self) {
        self.state.queue_paused = true;
        if self.state.queued.is_empty() {
            return;
        }
        if !self.state.editor.lines().join("\n").is_empty() || !self.state.attachments.is_empty() {
            self.state.open(Overlay::Queue);
            self.state.notice(
                "Waiting messages paused. Your current draft is kept; edit the queue with Ctrl+E.",
                false,
            );
            return;
        }
        if let Some(draft) = self.state.queued.pop_front() {
            self.history_index = None;
            self.history_draft = None;
            self.set_draft(&draft.text);
            self.state.attachments = draft.attachments;
        }
    }

    fn prompt(&mut self, prompt: Vec<Value>) -> Vec<Action> {
        self.begin("prompt", "Working…");
        let turn_id = self.local_id();
        self.state.active_turn_id = Some(turn_id.clone());
        vec![request(
            "session/prompt",
            json!({"sessionId":self.state.session_id,
            "prompt":prompt,
            "_meta":{"aster":{"turnId":turn_id}}}),
            Pending::Prompt,
        )]
    }

    fn local_id(&mut self) -> String {
        self.next_local_id = self.next_local_id.wrapping_add(1);
        format!("{}-{}", self.id_prefix, self.next_local_id)
    }

    fn runtime_request(
        &self,
        method: &'static str,
        mut params: Value,
        operation: RuntimeRequest,
    ) -> Vec<Action> {
        let Some(session) = self.state.session_id.clone() else {
            return vec![];
        };
        params["sessionId"] = json!(session);
        vec![request(
            method,
            params,
            Pending::Runtime {
                session,
                generation: self.state.conversation_generation,
                operation,
            },
        )]
    }

    fn open_runtime(&mut self, overlay: Overlay) -> Vec<Action> {
        if !self.state.runtime_supported {
            self.state.notice("This backend does not expose session settings. Update aster-acp to use this control.", false);
            return vec![];
        }
        self.state.open(overlay);
        self.runtime_request("_aster/session/runtime", json!({}), RuntimeRequest::Inspect)
    }

    fn configure(&mut self, params: Value) -> Vec<Action> {
        if !self.state.runtime_supported || self.state.session_id.is_none() {
            self.state.notice("Session settings are not available until preparation finishes with a supported backend.", false);
            return vec![];
        }
        if self.state.busy() || self.state.phase == Phase::Failed {
            self.state.notice(
                "Finish or stop the active task before changing its model, mode, or permissions.",
                false,
            );
            return vec![];
        }
        self.state.overlay = None;
        self.begin("configure", "Applying session settings…");
        self.runtime_request(
            "_aster/session/configure",
            params,
            RuntimeRequest::Configure,
        )
    }

    fn runtime_update(&mut self, value: &Value) {
        if let Some(model) = value["model"].as_str() {
            let model = clean(model);
            if !model.is_empty() && model != self.state.model {
                self.state.model = model;
                self.state.usage.clear();
                self.state.context_ratio = None;
            }
        }
        if let Some(mode) = value["mode"].as_str() {
            self.state.mode = clean(mode);
        }
        if let Some(approval) = value["approval"].as_str() {
            self.state.approval_policy = clean(approval);
        }
        if let Some(backend) = value["executionBackend"].as_str() {
            self.state.sandbox = backend == "sandbox";
        }
        self.state.runtime = value.clone();
    }

    fn runtime_response(&mut self, pending: Pending, value: &Value) -> Vec<Action> {
        let Pending::Runtime {
            session,
            generation,
            operation,
        } = pending
        else {
            return vec![];
        };
        if self.state.session_id.as_ref() != Some(&session) {
            return vec![];
        }
        if self
            .opening
            .as_deref()
            .is_some_and(|target| target != session)
        {
            if let Some(backup) = self.backup.as_mut() {
                // Replay increments the visible generation while the source
                // remains open. Its requests issued just before load still
                // belong to this backup, as do source requests during replay.
                if backup.source_session.as_ref() != Some(&session)
                    || (generation != backup.source_generation
                        && generation != self.state.conversation_generation)
                {
                    return vec![];
                }
                if value.get("error").is_none() {
                    let tasks = match operation {
                        RuntimeRequest::Tasks => &value["result"],
                        RuntimeRequest::StopTask => &value["result"]["tasks"],
                        _ => &Value::Null,
                    };
                    if tasks.is_object() {
                        backup.tasks = tasks.clone();
                    }
                }
                backup.updates.push(SourceUpdate::Runtime {
                    operation,
                    response: value.clone(),
                });
                return vec![];
            }
        }
        if self.state.conversation_generation != generation {
            return vec![];
        }
        self.apply_runtime_response(operation, value)
    }

    fn apply_runtime_response(&mut self, operation: RuntimeRequest, value: &Value) -> Vec<Action> {
        if let Some(error) = value.get("error") {
            self.state.notice(
                error["message"]
                    .as_str()
                    .unwrap_or("Session control request failed."),
                true,
            );
            return if operation == RuntimeRequest::Configure && self.operation == "configure" {
                self.ready("Settings unchanged")
            } else {
                vec![]
            };
        }
        let result = &value["result"];
        match operation {
            RuntimeRequest::Inspect | RuntimeRequest::Configure => self.runtime_update(result),
            RuntimeRequest::Tasks => self.tasks_update(result),
            RuntimeRequest::StopTask => {
                if result["tasks"].is_object() {
                    self.tasks_update(&result["tasks"]);
                }
                self.state.notice(
                    if result["stopped"] == true {
                        "Selected background job stopped."
                    } else {
                        "That background job has already finished."
                    },
                    false,
                );
            }
        }
        if operation == RuntimeRequest::Configure && self.operation == "configure" {
            let mut actions = self.ready("Session settings updated");
            if actions.is_empty() {
                actions.extend(self.send_queued());
            }
            actions
        } else {
            vec![]
        }
    }

    fn tasks_update(&mut self, value: &Value) {
        self.state.tasks = value.clone();
        if self.operation == "prompt"
            && self.state.phase == Phase::Busy
            && self.state.approvals.is_empty()
        {
            let current = clean(value["currentTask"].as_str().unwrap_or(""));
            let waiting = clean(value["waitReason"].as_str().unwrap_or(""));
            if !current.is_empty() || !waiting.is_empty() {
                self.state.status = [current, waiting]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" · ");
            }
        }
    }

    fn open_tasks(&mut self) -> Vec<Action> {
        if !self.state.tasks_supported {
            self.state.notice(
                "This backend does not publish structured task progress.",
                false,
            );
            return vec![];
        }
        self.state.open(Overlay::Tasks);
        self.runtime_request("_aster/session/tasks", json!({}), RuntimeRequest::Tasks)
    }

    fn stop_task(&mut self, id: &str) -> Vec<Action> {
        if !self.state.tasks_supported {
            return vec![];
        }
        let found = self.state.tasks["jobs"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|job| job["id"] == id && job["canStop"] == true);
        if !found {
            self.state.notice(
                "Choose a running background job in /tasks before stopping it.",
                false,
            );
            return vec![];
        }
        self.runtime_request(
            "_aster/session/stop_task",
            json!({"taskId":id}),
            RuntimeRequest::StopTask,
        )
    }

    pub fn permission_request(&mut self, value: &Value) -> Vec<Action> {
        let id = value["id"].clone();
        let params = &value["params"];
        let session = params["sessionId"].as_str();
        if session != self.state.session_id.as_deref()
            || self.exiting
            || self.cancelled
            || self.state.phase == Phase::Failed
            || (self.operation != "prompt" && !self.background_running())
            || self.state.approvals.len() >= 16
        {
            return vec![Action::Reply {
                id,
                result: json!({"outcome":{"outcome":"cancelled"}}),
            }];
        }
        if self
            .state
            .approvals
            .iter()
            .any(|approval| approval.id == id)
        {
            return vec![];
        }
        let options: Vec<ApprovalOption> = params["options"]
            .as_array()
            .into_iter()
            .flatten()
            .take(16)
            .filter_map(|option| {
                let id = option["optionId"].as_str()?;
                let kind = option["kind"].as_str()?;
                if !matches!(
                    kind,
                    "allow_once" | "allow_always" | "reject_once" | "reject_always"
                ) {
                    return None;
                }
                Some(ApprovalOption {
                    id: id.into(),
                    name: clean(option["name"].as_str().unwrap_or(id)),
                    kind: kind.into(),
                })
            })
            .collect();
        if options.is_empty() {
            return vec![Action::Reply {
                id,
                result: json!({"outcome":{"outcome":"cancelled"}}),
            }];
        }
        let code = params["toolCall"]["rawInput"]["code"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                params["toolCall"]["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(tool_content_text)
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        if code.len() > 256 * 1024 {
            self.state.notice(
                "Approval request exceeded the review limit; execution was cancelled.",
                true,
            );
            return vec![Action::Reply {
                id,
                result: json!({"outcome":{"outcome":"cancelled"}}),
            }];
        }
        let preview = approval_preview(&params["toolCall"]["_meta"]["aster"]["approvalPreview"]);
        let show_code = !preview.as_ref().is_some_and(|preview| preview.complete);
        self.state.approvals.push_back(ApprovalRequest {
            id,
            title: clean(
                params["toolCall"]["title"]
                    .as_str()
                    .unwrap_or("Execution approval"),
            ),
            code: clean(&code),
            preview,
            show_code,
            options,
        });
        // A queued request must not reset the review or selection in progress.
        if self.state.approvals.len() > 1 {
            return vec![];
        }
        self.state.open(Overlay::Approval);
        self.state.selected = self
            .state
            .approvals
            .front()
            .map(ApprovalRequest::default_selection)
            .unwrap_or(0);
        self.state.status = "Waiting for execution approval".into();
        vec![]
    }

    fn decide_permission(&mut self, option: Option<String>) -> Vec<Action> {
        let Some(approval) = self.state.approvals.pop_front() else {
            return vec![];
        };
        let outcome = if let Some(option) =
            option.filter(|id| approval.options.iter().any(|item| &item.id == id))
        {
            json!({"outcome":"selected","optionId":option})
        } else {
            json!({"outcome":"cancelled"})
        };
        self.state.overlay = if self.state.approvals.is_empty() {
            None
        } else {
            Some(Overlay::Approval)
        };
        self.state.selected = self
            .state
            .approvals
            .front()
            .map(ApprovalRequest::default_selection)
            .unwrap_or(0);
        self.state.detail_scroll = 0;
        self.state.approval_page_rows = 1;
        if self.state.approvals.is_empty() && self.operation == "prompt" {
            self.state.status = "Working…".into();
        }
        vec![Action::Reply {
            id: approval.id,
            result: json!({"outcome":outcome}),
        }]
    }

    fn background_running(&self) -> bool {
        let tasks = if self.tree_prompt.is_some() && self.opening.is_some() {
            self.backup.as_ref().map(|backup| &backup.tasks)
        } else {
            None
        }
        .unwrap_or(&self.state.tasks);
        tasks["jobs"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|job| job["status"] == "running" || job["canStop"] == true)
    }

    fn dismiss_approvals(&mut self) -> Vec<Action> {
        let actions = self
            .state
            .approvals
            .drain(..)
            .map(|approval| Action::Reply {
                id: approval.id,
                result: json!({"outcome":{"outcome":"cancelled"}}),
            })
            .collect();
        if self.state.overlay == Some(Overlay::Approval) {
            self.state.overlay = None;
        }
        actions
    }

    fn approval_key(&mut self, key: KeyEvent) -> Vec<Action> {
        match key.code {
            KeyCode::Esc => return self.decide_permission(None),
            KeyCode::Tab if key.modifiers == KeyModifiers::NONE => {
                if let Some(approval) = self.state.approvals.front_mut() {
                    if approval
                        .preview
                        .as_ref()
                        .is_some_and(|preview| preview.complete)
                    {
                        approval.show_code = !approval.show_code;
                        self.state.detail_scroll = 0;
                        self.state.approval_page_rows = 1;
                    }
                }
            }
            KeyCode::Up => self.state.selected = self.state.selected.saturating_sub(1),
            KeyCode::Down => {
                self.state.selected =
                    (self.state.selected + 1).min(self.state.choices().len().saturating_sub(1))
            }
            KeyCode::PageUp => {
                self.state.detail_scroll = self
                    .state
                    .detail_scroll
                    .saturating_sub(self.state.approval_page_rows.max(1))
            }
            KeyCode::PageDown => {
                self.state.detail_scroll = self
                    .state
                    .detail_scroll
                    .saturating_add(self.state.approval_page_rows.max(1))
            }
            KeyCode::Enter if key.kind == KeyEventKind::Press => {
                let option = self
                    .state
                    .choices()
                    .get(self.state.selected)
                    .map(|choice| choice.value.clone());
                return self.decide_permission(option);
            }
            KeyCode::Char(digit @ '1'..='9')
                if key.modifiers == KeyModifiers::NONE && key.kind == KeyEventKind::Press =>
            {
                let option = self
                    .state
                    .choices()
                    .get(digit as usize - '1' as usize)
                    .map(|choice| choice.value.clone());
                if option.is_some() {
                    return self.decide_permission(option);
                }
            }
            _ => {}
        }
        vec![]
    }

    /// Steering is an independent acknowledged request; it never owns the
    /// foreground prompt's deadline, cancellation state or completion.
    fn steer(&mut self, text: String) -> Vec<Action> {
        if text.trim().is_empty() {
            return vec![];
        }
        if !self.state.attachments.is_empty() || self.state.attachment_loading {
            self.state.notice("Corrections accept text. Enter queues this draft with its attachments for the next turn.", false);
            return vec![];
        }
        if !self.state.steering_supported {
            self.state.notice(
                "This backend does not support steering. Enter queues a message for the next turn.",
                false,
            );
            return vec![];
        }
        if self.operation != "prompt"
            || self.state.phase != Phase::Busy
            || self.cancelled
            || self.exiting
        {
            self.state.notice(
                "Steering needs a running task. Your draft is kept; Enter sends or queues it.",
                false,
            );
            return vec![];
        }
        let (Some(session), Some(turn_id)) = (
            self.state.session_id.clone(),
            self.state.active_turn_id.clone(),
        ) else {
            return vec![];
        };
        if self.pending.values().any(|pending| {
            matches!(pending, Pending::Steer { text: pending_text, turn_id: pending_turn, .. }
                if pending_text == &text && pending_turn == &turn_id)
        }) {
            self.state.notice(
                "Waiting for the backend to acknowledge this correction.",
                false,
            );
            return vec![];
        }
        let message_id = self.local_id();
        self.steering_drafts
            .insert(message_id.clone(), self.state.editor.lines().join("\n"));
        self.steering_update(&json!({
            "messageId":message_id,"turnId":turn_id,"text":text,"state":SteeringState::Sending
        }));
        vec![request(
            "_aster/session/steer",
            json!({"sessionId":session,"turnId":turn_id,"messageId":message_id,"text":text}),
            Pending::Steer {
                session,
                generation: self.state.conversation_generation,
                turn_id,
                message_id,
                text,
                draft: self.state.editor.lines().join("\n"),
            },
        )]
    }

    fn steering_update(&mut self, value: &Value) {
        let (Some(id), Some(turn_id), Some(status)) = (
            value["messageId"].as_str(),
            value["turnId"].as_str(),
            SteeringState::from_wire(&value["state"]),
        ) else {
            return;
        };
        let position = self
            .state
            .steering_messages
            .iter()
            .position(|item| item.id == id);
        let index = match position {
            Some(index) => index,
            None => {
                let Some(text) = value["text"].as_str() else {
                    return;
                };
                let entry = self.state.entries.len();
                self.state.entries.push(Entry::Notice {
                    text: String::new(),
                    error: false,
                });
                self.state.steering_messages.push(SteeringMessage {
                    id: id.into(),
                    turn_id: turn_id.into(),
                    text: text.to_owned(),
                    state: SteeringState::Sending,
                    reason: String::new(),
                    entry,
                });
                self.state.steering_messages.len() - 1
            }
        };
        let item = &mut self.state.steering_messages[index];
        if item.turn_id != turn_id {
            return;
        }
        if !item.state.can_transition_to(status) {
            return;
        }
        item.state = status;
        if let Some(reason) = value["reason"].as_str() {
            item.reason = clean(reason);
        }
        let label = status.label();
        if let Some(Entry::Notice { text, error }) = self.state.entries.get_mut(item.entry) {
            *text = format!("{label}\n{}", clean(&item.text));
            if !item.reason.is_empty() {
                text.push_str(&format!("\n{}", item.reason));
            }
            *error = status.is_error();
        }
        let raw = item.text.clone();
        if status == SteeringState::Applied {
            self.steering_drafts.remove(id);
        } else if status.is_terminal() {
            if let Some(original) = self.steering_drafts.remove(id) {
                let current = self.state.editor.lines().join("\n");
                if current.is_empty() && self.state.attachments.is_empty() {
                    self.set_draft(&raw);
                } else if current != original
                    && current != raw
                    && !self
                        .state
                        .queued
                        .iter()
                        .any(|draft| draft.text == raw && draft.attachments.is_empty())
                {
                    self.state.queued.push_back(raw.into());
                }
                self.state.queue_paused = true;
            }
        }
    }

    fn finish_steering(&mut self, status: SteeringState) {
        let unresolved: Vec<_> = self
            .state
            .steering_messages
            .iter()
            .filter(|item| item.state.is_pending())
            .map(|item| (item.id.clone(), item.turn_id.clone()))
            .collect();
        for (id, turn) in unresolved {
            self.steering_update(&json!({"messageId":id,"turnId":turn,"state":status}));
        }
    }

    fn steering_response(&mut self, pending: Pending, value: &Value) -> Vec<Action> {
        let Pending::Steer {
            session,
            generation,
            turn_id,
            message_id,
            text,
            draft,
        } = pending
        else {
            return vec![];
        };
        if self.state.session_id.as_ref() != Some(&session)
            || self.state.conversation_generation != generation
        {
            return vec![];
        }
        let result = &value["result"];
        let status = SteeringState::from_wire(&result["state"]);
        let accepted = value.get("error").is_none()
            && result["messageId"] == message_id
            && result["turnId"] == turn_id
            && status.is_some_and(SteeringState::is_acknowledged);
        if accepted {
            self.steering_update(&json!({"messageId":message_id,"turnId":turn_id,"state":status}));
            // Only clear the exact draft that produced this request. Typing
            // during the round trip creates a newer draft that must survive.
            let applied_or_waiting = self
                .state
                .steering_messages
                .iter()
                .any(|item| item.id == message_id && item.state.is_acknowledged());
            if applied_or_waiting
                && self.state.editor.lines().join("\n") == draft
                && !draft.is_empty()
            {
                self.state.draft_changed();
                self.state.editor = TextArea::default();
                self.history_index = None;
                self.history_draft = None;
                self.state.composer.reset_navigation();
            }
            if let Some(store) = &self.history_store {
                if let Err(error) = store.append(&text) {
                    self.state
                        .notice(format!("Could not save input history: {error}"), true);
                }
            }
            self.history.push(text);
        } else {
            let reason = value["error"]["message"]
                .as_str()
                .unwrap_or("The correction was not accepted; review it before resending.");
            self.steering_update(&json!({"messageId":message_id,"turnId":turn_id,"state":SteeringState::Rejected,"reason":reason}));
        }
        vec![]
    }

    pub fn cancel(&mut self) -> Vec<Action> {
        self.state.queue_paused = true;
        if self.state.preparing() {
            self.recall_waiting_message();
            return vec![];
        }
        if self.cancelled {
            return vec![];
        }
        if self.operation == "prompt" {
            self.cancelled = true;
            // Wait for the terminal response before permitting another prompt.
            // The deadline shows a notice; only an explicit exit closes the UI.
            self.limit_deadline(CANCEL_TIMEOUT);
            self.state.phase = Phase::Cancelling;
            self.state.status = "Cancelling…".into();
            let mut actions = self.dismiss_approvals();
            actions.push(Action::Notify {
                method: "session/cancel",
                params: json!({"sessionId":self.state.session_id}),
            });
            return actions;
        } else if !self.state.busy() {
            self.state.notice(
                if self.state.queued.is_empty() {
                    "No request is running."
                } else {
                    "Message queue paused. Ctrl+S continues; Ctrl+E edits the queue."
                },
                false,
            );
        }
        vec![]
    }

    pub fn is_exiting(&self) -> bool {
        self.exiting
    }

    /// Save at both ends of shutdown: input can change while session/close runs.
    /// A failed save cancels the exit without consuming any composer or queue.
    pub fn save_for_exit(&mut self) -> bool {
        // Controllers embedded without persistence retain their original behavior.
        // The terminal always attaches a store before accepting any input.
        if self.draft_store.is_none() || self.persist_draft() {
            return true;
        }
        let snapshot = self.draft_snapshot();
        if snapshot.composer.text.is_empty()
            && snapshot.composer.attachments.is_empty()
            && snapshot.queued.is_empty()
            && snapshot.in_flight.is_none()
        {
            // Loading a corrupt or unavailable store must not trap an empty
            // terminal. Leave the original records untouched and report the
            // storage error during final shutdown.
            return true;
        }
        self.exiting = false;
        self.state.queue_paused = true;
        if self.state.phase == Phase::Closing {
            // session/close may already have succeeded. Reconnect must be available
            // instead of treating that closed session as ready to receive prompts.
            self.state.phase = Phase::Failed;
            self.operation = "";
            self.deadline = None;
        }
        let status = "Exit paused: draft save failed; input and queue are kept here.";
        if self.state.status != status {
            self.state.notice(
                "Exit paused. Resolve the draft storage error and retry exit; your input and queue remain available.",
                true,
            );
        }
        self.state.status = status.into();
        false
    }

    pub fn quit(&mut self) -> Vec<Action> {
        self.state.exit_confirmation = None;
        self.state.escape_confirmation = None;
        if self.exiting {
            return vec![];
        }
        self.state.queue_paused = true;
        if !self.save_for_exit() {
            return vec![];
        }
        self.exiting = true;
        if self.state.preparing() || self.state.phase == Phase::Failed || self.cancel_timed_out {
            return vec![Action::Exit];
        }
        if self.state.busy() {
            let actions = self.cancel();
            self.limit_deadline(CLOSE_TIMEOUT);
            self.state.status = "Saving and closing…".into();
            return actions;
        }
        self.close_for_exit()
    }

    fn close_for_exit(&mut self) -> Vec<Action> {
        self.state.phase = Phase::Closing;
        self.state.status = "Saving and closing…".into();
        self.deadline = Some(Instant::now() + CLOSE_TIMEOUT);
        match &self.state.session_id {
            Some(id) => vec![request(
                "session/close",
                json!({"sessionId":id}),
                Pending::ExitClose,
            )],
            None => vec![Action::Exit],
        }
    }

    pub fn message(&mut self, value: Value) -> Vec<Action> {
        if value["method"] == "session/update" {
            self.update(&value["params"]);
            return vec![];
        }
        let Some(id) = value["id"].as_u64() else {
            return vec![];
        };
        let Some(pending) = self.pending.remove(&id) else {
            return vec![];
        };
        if matches!(pending, Pending::Steer { .. }) {
            return self.steering_response(pending, &value);
        }
        if matches!(pending, Pending::Runtime { .. }) {
            return self.runtime_response(pending, &value);
        }
        if matches!(pending, Pending::TreeSearch { .. }) {
            return self.tree_search_response(pending, &value);
        }
        if let Pending::TreeCleanup { source, generation } = pending {
            if self.state.session_id != source || self.state.conversation_generation != generation {
                return vec![];
            }
            if let Some(error) = value.get("error") {
                let message = error["message"].as_str().unwrap_or("The request failed.");
                self.state.notice(
                    format!(
                        "Original conversation kept; could not close the unused branch: {message}"
                    ),
                    true,
                );
            }
            return vec![];
        }
        if let Pending::Checkpoints { source, generation }
        | Pending::Rewind { source, generation }
        | Pending::Tree { source, generation }
        | Pending::NavigateTree {
            source, generation, ..
        } = &pending
        {
            if self.state.session_id.as_ref() != Some(source)
                || self.state.conversation_generation != *generation
            {
                return vec![];
            }
        }
        if let Some(error) = value.get("error") {
            if matches!(pending, Pending::Checkpoints { .. }) {
                self.state.rewind_loading = false;
                self.state.overlay = None;
            }
            if matches!(pending, Pending::Tree { .. }) {
                self.state.tree_loading = false;
                self.state.overlay = None;
            }
            self.state.queue_paused = true;
            let tree_reason =
                if matches!(pending, Pending::Tree { .. } | Pending::NavigateTree { .. }) {
                    error["data"]["reason"]
                        .as_str()
                        .filter(|reason| !reason.trim().is_empty())
                } else {
                    None
                };
            let text = tree_reason
                .or_else(|| error["message"].as_str())
                .unwrap_or("The backend request failed.");
            if matches!(pending, Pending::Initialize) {
                return self.failure(text);
            }
            if matches!(pending, Pending::Prompt) {
                self.state.active_turn_id = None;
                self.state.recovery = self.in_flight.take();
                self.finish_steering(SteeringState::Interrupted);
                self.state.notice(
                    "Input saved for retry. Use /retry to edit it before resending.",
                    false,
                );
            }
            if matches!(pending, Pending::NewClose) {
                self.draft_to_restore = None;
            }
            if matches!(pending, Pending::Load { .. }) {
                self.restore_load_backup();
                if (self.recovering_session || self.draft_to_restore.is_some())
                    && error["code"] == -32002
                {
                    return self.prepare_recovered_session();
                }
                self.draft_to_restore = None;
            }
            self.state.notice(text, true);
            if !matches!(pending, Pending::Load { .. }) {
                self.finish_tools("failed");
            }
            if matches!(pending, Pending::ExitClose) {
                return self.failure(format!("Could not close session: {text}"));
            }
            let mut actions = if matches!(pending, Pending::Prompt) {
                self.dismiss_approvals()
            } else {
                vec![]
            };
            actions.extend(self.ready("Request failed"));
            return actions;
        }
        let result = &value["result"];
        match pending {
            Pending::Initialize => {
                if result["protocolVersion"] != 1 {
                    return self.failure("Unsupported ACP protocol version.");
                }
                let caps = &result["agentCapabilities"];
                if caps["loadSession"] != true
                    || !caps["sessionCapabilities"]["list"].is_object()
                    || !caps["sessionCapabilities"]["close"].is_object()
                {
                    return self.failure(
                        "Backend does not support required session capabilities; update aster-acp.",
                    );
                }
                self.rewind_supported = result["_meta"]["aster"]["conversationRewind"] == true;
                self.tree_supported = result["_meta"]["aster"]["sessionTree"] == true;
                self.state.tree_search_supported =
                    self.tree_supported && result["_meta"]["aster"]["sessionTreeSearch"] == true;
                self.state.steering_supported = result["_meta"]["aster"]["userSteering"] == true;
                self.state.attachments_supported = result["_meta"]["aster"]["attachments"] == true;
                self.state.image_input_supported = caps["promptCapabilities"]["image"] == true;
                self.state.runtime_supported = result["_meta"]["aster"]["runtimeControls"] == true;
                self.state.tasks_supported = result["_meta"]["aster"]["structuredProgress"] == true;
                self.state.phase = Phase::Ready;
                if let Some(purpose) = self.startup.take() {
                    self.list(purpose)
                } else {
                    self.list(ListPurpose::StartNew)
                }
            }
            Pending::Prepare => {
                let Some(id) = result["sessionId"].as_str() else {
                    return self.failure("Backend returned no session ID.");
                };
                self.state.session_id = Some(id.into());
                self.prepared()
            }
            Pending::Prompt => {
                self.state.active_turn_id = None;
                let reason = result["stopReason"].as_str().unwrap_or("unknown");
                let cancelled = self.cancelled || reason == "cancelled";
                if cancelled || reason != "end_turn" {
                    self.state.recovery = self.in_flight.take();
                } else {
                    self.in_flight = None;
                }
                self.finish_tools(if cancelled { "cancelled" } else { "completed" });
                if !matches!(reason, "end_turn" | "cancelled") {
                    self.state
                        .notice(format!("Request stopped: {reason}"), true);
                }
                if cancelled || reason != "end_turn" {
                    self.state.queue_paused = true;
                }
                self.finish_steering(if cancelled {
                    SteeringState::Cancelled
                } else {
                    SteeringState::TurnFinished
                });
                let mut actions = self.dismiss_approvals();
                let ready_actions = self.ready(if cancelled { "Cancelled" } else { "Ready" });
                if ready_actions.is_empty() {
                    actions.extend(self.send_queued());
                } else {
                    actions.extend(ready_actions);
                }
                actions
            }
            Pending::List(purpose) => {
                if self.history_import_requested {
                    self.history_import_requested = false;
                    if let Some(entries) = result["_meta"]["aster"]["inputHistory"].as_array() {
                        let imported: Vec<String> = entries
                            .iter()
                            .filter_map(Value::as_str)
                            .filter(|text| !text.trim().is_empty())
                            .map(str::to_owned)
                            .collect();
                        if let Some(store) = &self.history_store {
                            match store.seed_if_empty(&imported) {
                                Ok(saved) if self.history.is_empty() => self.history = saved,
                                Ok(_) => {}
                                Err(error) => {
                                    if self.history.is_empty() {
                                        self.history = imported;
                                    }
                                    self.state.notice(
                                        format!("Could not save restored input history: {error}"),
                                        true,
                                    );
                                }
                            }
                        }
                    }
                }
                if let Some(sessions) = result["sessions"].as_array() {
                    for session in sessions {
                        if let Some(id) = session["sessionId"].as_str() {
                            self.state.sessions.push(Session {
                                id: id.into(),
                                title: clean(session["title"].as_str().unwrap_or("")),
                                updated: clean(session["updatedAt"].as_str().unwrap_or("")),
                            });
                        }
                    }
                }
                if let Some(cursor) = result["nextCursor"].as_str() {
                    if !self.cursors.insert(cursor.into()) {
                        self.state.queue_paused = true;
                        self.state
                            .notice("Backend repeated a session page cursor.", true);
                        return self.ready("Request failed");
                    }
                    return vec![request(
                        "session/list",
                        json!({"cwd":self.state.workspace,"cursor":cursor}),
                        Pending::List(purpose),
                    )];
                }
                if self.exiting {
                    return self.ready("Ready");
                }
                let continue_last = matches!(purpose, ListPurpose::Continue);
                let target = match purpose {
                    ListPurpose::StartNew => return self.prepare_session(),
                    ListPurpose::Browse => return self.ready("Ready"),
                    ListPurpose::Continue => self.state.sessions.first().map(|s| s.id.clone()),
                    ListPurpose::Resume(prefix) => {
                        let exact = self
                            .state
                            .sessions
                            .iter()
                            .find(|s| s.id == prefix)
                            .map(|s| s.id.clone());
                        if exact.is_some() {
                            exact
                        } else {
                            let found: Vec<_> = self
                                .state
                                .sessions
                                .iter()
                                .filter(|s| s.id.starts_with(&prefix))
                                .collect();
                            if found.len() != 1 {
                                if found.is_empty() && self.recovering_session {
                                    return self.prepare_recovered_session();
                                }
                                self.state.notice(format!("{} conversations match {prefix:?}. Use /sessions to choose.", found.len()), true);
                                None
                            } else {
                                Some(found[0].id.clone())
                            }
                        }
                    }
                };
                match target {
                    Some(id) => {
                        // Startup input belongs to the requested restored session.
                        self.state.phase = Phase::Ready;
                        self.resume(id)
                    }
                    None if continue_last => self.prepare_session(),
                    None => {
                        self.state.queue_paused = true;
                        if self.starting {
                            self.state.session_id = None;
                        }
                        self.ready("Ready")
                    }
                }
            }
            Pending::Tree { .. } => {
                self.state.tree_loading = false;
                match crate::session_tree::SessionTree::from_response(result) {
                    Ok(tree) => {
                        self.state.session_tree = tree;
                        self.state.selected = self
                            .state
                            .choices()
                            .iter()
                            .position(|choice| {
                                Some(choice.value.as_str())
                                    == self.state.session_tree.current_node_id.as_deref()
                            })
                            .unwrap_or(0);
                        self.ready("Ready")
                    }
                    Err(error) => {
                        self.state.overlay = None;
                        self.state.notice(
                            format!("Could not display conversation tree: {error}"),
                            true,
                        );
                        self.ready("Request failed")
                    }
                }
            }
            Pending::NavigateTree {
                source, node_id, ..
            } => {
                if self.exiting {
                    return self.ready("Ready");
                }
                let Some(target) = result["sessionId"].as_str().filter(|id| {
                    !id.is_empty() && id.len() <= 256 && !id.chars().any(char::is_control)
                }) else {
                    self.state.notice(
                        "Backend returned no valid conversation for this branch.",
                        true,
                    );
                    return self.ready("Request failed");
                };
                let prompt = match result.get("prompt") {
                    Some(Value::Null) => None,
                    Some(Value::String(prompt)) => Some(prompt.clone()),
                    _ => {
                        self.state
                            .notice("Backend returned an invalid message for this branch.", true);
                        return self.ready("Request failed");
                    }
                };
                if !self.state.session_tree.node(&node_id).is_some_and(|node| {
                    (node.role == crate::session_tree::TreeRole::User) == prompt.is_some()
                }) {
                    self.state.notice(
                        "Backend returned an unexpected message for this branch.",
                        true,
                    );
                    return self.ready("Request failed");
                }
                if target == source {
                    if prompt.is_none()
                        && self.state.session_tree.current_node_id.as_deref()
                            == Some(node_id.as_str())
                        && self.state.session_tree.node(&node_id).is_some_and(|node| {
                            node.role == crate::session_tree::TreeRole::Assistant
                        })
                    {
                        return self.ready("Already at this conversation position");
                    }
                    self.state.notice(
                        "Backend did not restore the selected conversation branch.",
                        true,
                    );
                    return self.ready("Request failed");
                }
                if self.background_running() {
                    self.state.notice(
                        "Stop or finish background jobs before switching branches.",
                        false,
                    );
                    return self.ready("Branch switch paused; background work is still running");
                }
                // Input can change while the navigation RPC is pending. It must
                // be durable before the existing staged session switch begins.
                if self.draft_store.is_some() && !self.persist_draft() {
                    return self.ready("Branch switch paused; save your draft first");
                }
                if self.queue_blocks_switch() {
                    return self.ready("Branch switch paused; review queued messages");
                }
                self.tree_prompt = Some((target.to_owned(), prompt));
                self.state.phase = Phase::Ready;
                self.resume(target.to_owned())
            }
            Pending::Checkpoints { .. } => {
                self.state.rewind_loading = false;
                let Some(points) = result["checkpoints"].as_array() else {
                    self.state.overlay = None;
                    self.state
                        .notice("Backend returned invalid rewind points.", true);
                    return self.ready("Request failed");
                };
                self.state.rewind_points = points
                    .iter()
                    .filter_map(|point| {
                        Some(RewindPoint {
                            user_message_index: usize::try_from(
                                point["userMessageIndex"].as_u64()?,
                            )
                            .ok()?,
                            prompt: point["prompt"].as_str()?.to_owned(),
                        })
                    })
                    .collect();
                self.state
                    .rewind_points
                    .sort_by_key(|point| point.user_message_index);
                self.state
                    .rewind_points
                    .dedup_by_key(|point| point.user_message_index);
                self.state.selected = 0;
                self.ready("Ready")
            }
            Pending::Rewind { source, .. } => {
                if self.exiting {
                    return self.ready("Ready");
                }
                let Some(new_id) = result["sessionId"]
                    .as_str()
                    .filter(|id| !id.is_empty() && *id != source)
                else {
                    self.state
                        .notice("Backend returned no new conversation for rewind.", true);
                    return self.ready("Request failed");
                };
                let Some(prompt) = result["prompt"].as_str() else {
                    self.state
                        .notice("Backend returned no message to edit.", true);
                    return self.ready("Request failed");
                };
                self.rewind_prompt = Some((new_id.to_owned(), prompt.to_owned()));
                self.state.phase = Phase::Ready;
                self.resume(new_id.to_owned())
            }
            Pending::Load { id, previous } => {
                if self
                    .tree_prompt
                    .as_ref()
                    .is_some_and(|(target, _)| target == &id)
                {
                    if self.background_running() {
                        return self.reject_loaded_tree(
                            id,
                            "Branch switch paused; background work is still running.",
                        );
                    }
                    if self.draft_store.is_some() && !self.persist_draft() {
                        return self.reject_loaded_tree(
                            id,
                            "Branch switch paused; save your draft first.",
                        );
                    }
                    if self.queue_blocks_switch() {
                        return self.reject_loaded_tree(
                            id,
                            "Branch switch paused; review queued messages.",
                        );
                    }
                }
                self.opening = None;
                self.backup = None;
                self.draft_to_restore = None;
                self.state.session_id = Some(id.clone());
                self.load_draft_scope(Some(id.clone()));
                // Replay is committed only after load succeeds. A failed switch
                // keeps the previous conversation's submitted-input history.
                let replay_history: Vec<String> = self
                    .state
                    .entries
                    .iter()
                    .filter_map(|entry| {
                        if let Entry::User(text) = entry {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if self.history_store.is_none() {
                    self.history = replay_history;
                    // Conversation-local recall now refers to the loaded history.
                    // Restore any draft hidden behind recall before ending it.
                    if let Some(draft) = self.history_draft.take() {
                        self.state.draft_changed();
                        self.state.editor = draft;
                        self.state.composer.reset_navigation();
                        self.state.slash_selected = 0;
                        self.state.slash_dismissed = false;
                    }
                    self.history_index = None;
                }
                // Persistent history is workspace-wide, so its navigation and
                // saved draft remain valid across a conversation switch.
                if let Some((target, prompt)) = self.rewind_prompt.take() {
                    if target == id {
                        self.state.queue_paused = true;
                        self.history_index = None;
                        self.history_draft = None;
                        if self.state.editor.lines().join("\n").is_empty() {
                            self.set_draft(&prompt);
                        } else {
                            // Typing during a restore remains meaningful. Keep
                            // that newer draft and make the recalled text
                            // explicitly paused, never an automatic prompt.
                            self.state.queued.push_back(prompt.into());
                            self.state.notice("Your new draft was kept. The selected message is paused in the queue (Ctrl+E).", false);
                        }
                        self.state.notice("Conversation restored. Edit the message and press Enter when ready. Original conversation kept; files unchanged.", false);
                    }
                }
                if let Some((target, prompt)) = self.tree_prompt.take() {
                    if target == id {
                        self.state.queue_paused = true;
                        // Tree navigation ends recall. Recover the unsent editor
                        // hidden behind workspace history before merging input.
                        if self.history_index.is_some() {
                            if let Some(draft) = self.history_draft.take() {
                                self.state.draft_changed();
                                self.state.editor = draft;
                                self.state.composer.reset_navigation();
                                self.state.slash_selected = 0;
                                self.state.slash_dismissed = false;
                            }
                        }
                        self.history_index = None;
                        self.history_draft = None;
                        if let Some(prompt) = prompt {
                            if self.state.editor.lines().join("\n").is_empty()
                                && self.state.attachments.is_empty()
                            {
                                self.set_draft(&prompt);
                            } else {
                                self.state.queued.push_back(prompt.into());
                                self.state.notice("Your draft was kept. The selected message is paused in the queue (Ctrl+E).", false);
                            }
                        }
                        self.state.notice("Conversation branch restored. Continue when ready; all branches are kept.", false);
                    }
                }
                if let Some(previous) = previous.filter(|previous| previous != &id) {
                    self.begin("close", "Closing previous conversation…");
                    vec![request(
                        "session/close",
                        json!({"sessionId":previous}),
                        Pending::OldClose,
                    )]
                } else {
                    self.prepared()
                }
            }
            Pending::NewClose => {
                self.state.session_id = None;
                self.reset_conversation();
                self.finish_new_draft();
                if self.exiting {
                    self.ready("Ready")
                } else {
                    self.prepare_session()
                }
            }
            Pending::OldClose => self.prepared(),
            Pending::ExitClose => vec![Action::Exit],
            Pending::Steer { .. } => unreachable!("steering responses are handled independently"),
            Pending::Runtime { .. } => unreachable!("runtime responses are handled independently"),
            Pending::TreeCleanup { .. } => unreachable!("tree cleanup is handled independently"),
            Pending::TreeSearch { .. } => unreachable!("tree searches are handled independently"),
        }
    }

    fn finish_tools(&mut self, status: &str) {
        if let Some(start) = self.state.started.take() {
            self.state.last_elapsed = Some(start.elapsed().as_secs_f64());
        }
        for entry in &mut self.state.entries {
            if let Entry::Tool(tool) = entry {
                if matches!(tool.status.as_str(), "pending" | "in_progress") {
                    tool.status = status.into();
                    tool.elapsed = tool.started.map(|start| start.elapsed().as_secs_f64());
                }
            }
        }
    }

    fn message_chunk(&mut self, update: &Value) {
        let Some(text) = update["content"]["text"].as_str() else {
            return;
        };
        let user = update["sessionUpdate"] == "user_message_chunk";
        if user && self.state.title.is_empty() {
            self.state.title = clean(text)
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(80)
                .collect();
        }
        let meta = &update["_meta"]["aster"];
        let id = meta["messageId"].as_str().filter(|id| !id.is_empty());
        let status = meta["messageState"].as_str().unwrap_or("");
        let Some(id) = id.filter(|_| matches!(status, "delta" | "complete" | "interrupted")) else {
            self.state.entries.push(if user {
                Entry::User(clean(text))
            } else {
                Entry::Assistant(clean(text))
            });
            return;
        };
        let key = (user, id.to_owned());
        if !self.message_streams.contains_key(&key) {
            if status == "interrupted" && text.is_empty() {
                return;
            }
            let index = self.state.entries.len();
            self.state.entries.push(if user {
                Entry::User(String::new())
            } else {
                Entry::Assistant(String::new())
            });
            self.message_streams.insert(
                key.clone(),
                MessageStream {
                    index,
                    filter: crate::state::TerminalTextFilter::default(),
                    complete: false,
                },
            );
        }
        let stream = self
            .message_streams
            .get_mut(&key)
            .expect("message stream exists");
        let replace = meta["replace"].as_bool() == Some(true);
        if stream.complete && !replace {
            return;
        }
        if replace {
            stream.filter = crate::state::TerminalTextFilter::default();
        }
        let visible = stream.filter.feed(text);
        if let Some(Entry::User(content) | Entry::Assistant(content)) =
            self.state.entries.get_mut(stream.index)
        {
            if replace {
                *content = visible;
            } else {
                content.push_str(&visible);
            }
        }
        stream.complete = matches!(status, "complete" | "interrupted");
        if status == "interrupted" {
            self.state
                .notice("Response interrupted; the text above is incomplete.", false);
        }
    }

    fn update(&mut self, params: &Value) {
        let id = params["sessionId"].as_str().unwrap_or("");
        if self.opening.as_deref().is_some_and(|target| target != id)
            && self.state.session_id.as_deref() == Some(id)
        {
            // Original-session jobs may change while the target is replayed.
            // Retain their latest state for the commit guard and rollback.
            if let Some(backup) = self.backup.as_mut() {
                let tasks = &params["update"]["_meta"]["aster"]["tasks"];
                if tasks.is_object() {
                    backup.tasks = tasks.clone();
                }
                backup
                    .updates
                    .push(SourceUpdate::Notification(params.clone()));
            }
            return;
        }
        let expected = self.opening.as_ref().or(self.state.session_id.as_ref());
        if expected.is_some_and(|expected| expected != id) {
            return;
        }
        let update = &params["update"];
        if expected.is_none() && update["sessionUpdate"] == "usage_update" {
            // A closed session may send a final update while its replacement
            // is being prepared. It must not populate the new session header.
            return;
        }
        if update["_meta"]["aster"]["steering"].is_object() {
            self.steering_update(&update["_meta"]["aster"]["steering"]);
        }
        if update["_meta"]["aster"]["runtime"].is_object() {
            self.runtime_update(&update["_meta"]["aster"]["runtime"]);
        }
        if update["_meta"]["aster"]["tasks"].is_object() {
            self.tasks_update(&update["_meta"]["aster"]["tasks"]);
        }
        match update["sessionUpdate"].as_str().unwrap_or("") {
            "current_mode_update" => {
                if let Some(mode) = update["currentModeId"].as_str() {
                    self.state.mode = clean(mode);
                }
            }
            "agent_message_chunk" | "user_message_chunk" => {
                self.message_chunk(update);
            }
            "tool_call" | "tool_call_update" => {
                let tool_id = update["toolCallId"].as_str().unwrap_or("");
                if tool_id.starts_with("aster-progress-") {
                    if self.state.busy()
                        && !self.cancelled
                        && !self.exiting
                        && self.state.approvals.is_empty()
                    {
                        if let Some(title) = update["title"].as_str() {
                            self.state.status = clean(title);
                        }
                    }
                    return;
                }
                if tool_id.is_empty() {
                    return;
                }
                let index = self
                    .state
                    .entries
                    .iter()
                    .position(|e| matches!(e, Entry::Tool(t) if t.id==tool_id));
                let index = index.unwrap_or_else(|| {
                    self.state.entries.push(Entry::Tool(Tool {
                        id: tool_id.into(),
                        title: "Tool".into(),
                        status: "pending".into(),
                        content: String::new(),
                        diff: String::new(),
                        expanded: false,
                        started: None,
                        elapsed: None,
                    }));
                    self.state.entries.len() - 1
                });
                if let Entry::Tool(tool) = &mut self.state.entries[index] {
                    if let Some(title) = update["title"].as_str() {
                        tool.title = clean(title);
                    }
                    if let Some(status) = update["status"].as_str() {
                        tool.status = status.into();
                    }
                    if let Some(content) = update["content"].as_array() {
                        // ASTER sends cumulative output snapshots, not deltas.
                        tool.content.clear();
                        tool.diff.clear();
                        for item in content {
                            if item["type"] == "content" {
                                if let Some(text) = tool_content_text(item) {
                                    if !tool.content.is_empty() {
                                        tool.content.push('\n');
                                    }
                                    tool.content.push_str(&clean(text));
                                }
                            } else if item["type"] == "diff" {
                                tool.diff.push_str(&diff(item));
                            }
                        }
                    }
                    if matches!(tool.status.as_str(), "pending" | "in_progress") {
                        tool.started.get_or_insert_with(Instant::now);
                    } else if let Some(seconds) = update["_meta"]["aster"]["elapsedSeconds"]
                        .as_f64()
                        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                    {
                        tool.elapsed = Some(seconds);
                    } else if tool.elapsed.is_none() {
                        tool.elapsed = tool.started.map(|start| start.elapsed().as_secs_f64());
                    }
                }
            }
            "usage_update" => {
                let available = update["_meta"]["aster"]["usageAvailable"] != false;
                let used = update["used"].as_u64().filter(|_| available);
                let size = update["size"].as_u64().filter(|size| *size > 0);
                self.state.context_ratio = used
                    .zip(size)
                    .map(|(used, size)| (used as f64 / size as f64).clamp(0.0, 1.0));
                self.state.usage = match (used, size) {
                    (Some(used), Some(size)) => format!("{used} / {size}"),
                    (Some(used), None) => format!("{used} tokens"),
                    (None, _) => "context n/a".into(),
                };
                let amount = update["cost"]["amount"]
                    .as_f64()
                    .filter(|amount| amount.is_finite() && *amount >= 0.0);
                let currency = update["cost"]["currency"]
                    .as_str()
                    .map(clean)
                    .filter(|currency| !currency.trim().is_empty());
                self.state.cost = match (amount, currency) {
                    (Some(amount), Some(currency)) if currency == "USD" => {
                        format!("${amount:.4}")
                    }
                    (Some(amount), Some(currency)) => format!("{amount:.4} {currency}"),
                    _ => "cost n/a".into(),
                };
            }
            "session_info_update" => {
                if update.get("title").is_some() {
                    self.state.title = clean(update["title"].as_str().unwrap_or(""));
                    if let Some(session) = self.state.sessions.iter_mut().find(|s| s.id == id) {
                        session.title = self.state.title.clone();
                    }
                }
            }
            "available_commands_update" => {
                self.state.commands = default_commands();
                if let Some(commands) = update["availableCommands"].as_array() {
                    for command in commands {
                        if let Some(name) = command["name"].as_str() {
                            let name = format!("/{}", clean(name));
                            if !self.state.commands.iter().any(|c| c.name == name) {
                                self.state.commands.push(Command {
                                    name,
                                    description: clean(
                                        command["description"].as_str().unwrap_or(""),
                                    ),
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn toggle_tool(&mut self, id: &str) {
        for (index, entry) in self.state.entries.iter_mut().enumerate() {
            if let Entry::Tool(t) = entry {
                if t.id == id {
                    t.expanded = !t.expanded;
                    if !t.expanded {
                        self.state.expanded_entries.remove(&index);
                    }
                    self.state.rendered_rows = None;
                    self.state.focus_entry = Some(index);
                    self.state.search_highlight.clear();
                }
            }
        }
    }

    pub fn toggle_entry(&mut self, index: usize) {
        match self.state.entries.get_mut(index) {
            Some(Entry::User(_)) => {}
            Some(Entry::Tool(tool)) => tool.expanded = true,
            _ => return,
        }
        if !self.state.expanded_entries.remove(&index) {
            self.state.expanded_entries.insert(index);
        }
        self.state.rendered_rows = None;
        self.state.focus_entry = Some(index);
        self.state.search_highlight.clear();
    }

    pub fn composer_mouse(&mut self, area: Rect, mouse: MouseEvent) -> bool {
        if self.state.overlay.is_some() {
            self.state.composer.cancel_pointer();
            return false;
        }
        crate::composer::handle_mouse(
            &mut self.state.editor,
            &mut self.state.composer,
            area,
            mouse,
        )
    }

    fn selected_text(&self) -> Option<String> {
        let (start, end) = self.state.editor.selection_range()?;
        if start == end {
            return None;
        }
        Some(
            self.state.editor.lines()[start.0..=end.0]
                .iter()
                .enumerate()
                .map(|(offset, line)| {
                    let row = start.0 + offset;
                    let from = if row == start.0 { start.1 } else { 0 };
                    let to = if row == end.0 {
                        end.1
                    } else {
                        line.chars().count()
                    };
                    line.chars().skip(from).take(to - from).collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    /// Called only after clipboard writing succeeds and its input snapshot still matches.
    pub fn cut_selection(&mut self) -> bool {
        if self.state.overlay.is_some() || self.selected_text().is_none() {
            return false;
        }
        self.state.editor.delete_char();
        self.state.draft_changed();
        self.clear_draft_notice();
        self.history_index = None;
        self.history_draft = None;
        self.state.slash_selected = 0;
        self.state.slash_dismissed = false;
        true
    }

    pub fn paste(&mut self, text: String) {
        self.state.composer.cancel_pointer();
        self.state.exit_confirmation = None;
        self.state.escape_confirmation = None;
        let text = clean(&text.replace("\r\n", "\n").replace('\r', "\n"));
        if let Some(overlay) = self.state.overlay {
            if matches!(
                overlay,
                Overlay::Commands
                    | Overlay::Sessions
                    | Overlay::Drafts
                    | Overlay::Tools
                    | Overlay::Search
                    | Overlay::Queue
                    | Overlay::Rewind
                    | Overlay::Tree
                    | Overlay::Files
                    | Overlay::Attachments
                    | Overlay::Models
                    | Overlay::Modes
                    | Overlay::Permissions
                    | Overlay::Tasks
            ) || (overlay == Overlay::Changes && !self.state.conversation_changes)
            {
                self.state.query.push_str(&text.replace('\n', " "));
                self.state.selected = 0;
                if overlay == Overlay::Tree {
                    self.tree_query_changed();
                }
            }
        } else {
            self.state.composer.reset_navigation();
            let before = self.state.editor.lines().to_vec();
            self.state.editor.insert_str(text);
            if self.state.editor.lines() != before {
                self.state.draft_changed();
                self.clear_draft_notice();
                self.history_index = None;
                self.history_draft = None;
                self.state.slash_selected = 0;
                self.state.slash_dismissed = false;
            }
        }
    }

    pub fn key(&mut self, key: KeyEvent) -> Vec<Action> {
        if key.kind == KeyEventKind::Release {
            return vec![];
        }
        if key.code == KeyCode::Esc && key.kind == KeyEventKind::Repeat {
            return vec![];
        }
        self.state.composer.cancel_pointer();
        if key.code != KeyCode::Esc || key.modifiers != KeyModifiers::NONE {
            self.state.escape_confirmation = None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        if ctrl
            && !alt
            && self.state.overlay.is_none()
            && matches!(key.code, KeyCode::Char('c' | 'C' | 'x' | 'X'))
        {
            if let Some(text) = self.selected_text() {
                self.state.exit_confirmation = None;
                return if key.kind == KeyEventKind::Repeat {
                    vec![]
                } else {
                    vec![Action::CopySelection {
                        text,
                        cut: matches!(key.code, KeyCode::Char('x' | 'X')),
                    }]
                };
            }
        }
        if ctrl && !alt && matches!(key.code, KeyCode::Char('c' | 'C')) {
            if key.kind == KeyEventKind::Repeat {
                return vec![];
            }
            let now = Instant::now();
            if self.state.exit_confirmation.is_some_and(|time| now < time) {
                return self.quit();
            }
            self.state.exit_confirmation = Some(now + EXIT_CONFIRMATION_TIMEOUT);
            return vec![];
        }
        self.state.exit_confirmation = None;
        if !matches!(key.code, KeyCode::Up | KeyCode::Down) || ctrl || alt {
            self.state.composer.reset_navigation();
        }
        if ctrl && key.code == KeyCode::Char('q') {
            return self.quit();
        }
        if !alt
            && ((ctrl && matches!(key.code, KeyCode::Char('v' | 'V')))
                || (key.code == KeyCode::Insert && key.modifiers == KeyModifiers::SHIFT))
        {
            return vec![Action::PasteClipboard];
        }
        if let Some(overlay) = self.state.overlay {
            self.state.escape_confirmation = None;
            return self.overlay_key(overlay, key);
        }
        if key.code == KeyCode::Backspace
            && (key.modifiers == KeyModifiers::ALT
                || (key.modifiers == KeyModifiers::NONE
                    && self.state.editor.lines() == [""]
                    && !self.state.attachments.is_empty()))
        {
            if key.kind != KeyEventKind::Repeat {
                if let Some(index) = self.state.attachments.len().checked_sub(1) {
                    self.remove_attachment(index);
                }
            }
            return vec![];
        }
        let choices = self.state.slash_choices();
        if !choices.is_empty() && key.modifiers == KeyModifiers::NONE {
            self.state.slash_selected = self.state.slash_selected.min(choices.len() - 1);
            match key.code {
                KeyCode::Up => {
                    self.state.slash_selected = self.state.slash_selected.saturating_sub(1);
                    return vec![];
                }
                KeyCode::Down => {
                    self.state.slash_selected =
                        (self.state.slash_selected + 1).min(choices.len() - 1);
                    return vec![];
                }
                KeyCode::Tab => {
                    self.set_draft(&format!("{} ", choices[self.state.slash_selected].value));
                    return vec![];
                }
                KeyCode::Enter => return self.choose_slash(self.state.slash_selected),
                KeyCode::Esc => {
                    self.state.escape_confirmation = None;
                    self.state.slash_dismissed = true;
                    return vec![];
                }
                _ => {}
            }
        }
        if key.modifiers == KeyModifiers::NONE && matches!(key.code, KeyCode::Up | KeyCode::Down) {
            let previous = key.code == KeyCode::Up;
            if (previous || self.history_index.is_some())
                && crate::composer::at_vertical_edge(
                    &self.state.editor,
                    &self.state.composer,
                    previous,
                )
            {
                self.history_move(previous);
                return vec![];
            }
        }
        let before = self.state.editor.lines().to_vec();
        match key.code {
            KeyCode::Char('@')
                if !ctrl && !alt && {
                    let (row, column) = self.state.editor.cursor();
                    column == 0
                        || self.state.editor.lines()[row]
                            .chars()
                            .nth(column - 1)
                            .is_some_and(char::is_whitespace)
                } =>
            {
                return self.open_files()
            }
            KeyCode::Char('a' | 'A') if alt && !ctrl => return self.open_files(),
            KeyCode::Char('t' | 'T') if alt && !ctrl => return self.open_tasks(),
            KeyCode::Char('d') if ctrl && self.state.editor.lines().join("\n").is_empty() => {
                return self.quit()
            }
            KeyCode::Char('k') if ctrl => self.state.open(Overlay::Commands),
            KeyCode::Char('x') if ctrl => self.state.open(Overlay::Help),
            KeyCode::Char('e') if ctrl => self.state.open(Overlay::Queue),
            KeyCode::Char('f') if ctrl => self.state.open(Overlay::Search),
            KeyCode::Char('s') if ctrl => return self.continue_queue(),
            KeyCode::Char('s' | 'S') if alt => {
                return self.steer(self.state.editor.lines().join("\n"));
            }
            KeyCode::Char('r') if ctrl => {
                self.state.open(Overlay::Sessions);
                if !self.state.busy() {
                    return self.list(ListPurpose::Browse);
                }
            }
            KeyCode::Char('t') if ctrl => self.state.open(Overlay::Tools),
            KeyCode::Char('g') if ctrl => return self.open_changes(),
            KeyCode::Char('b') if alt => self.state.show_sidebar = !self.state.show_sidebar,
            KeyCode::Char('o') if ctrl => {
                let expand = self.state.expanded_entries.is_empty()
                    && !self
                        .state
                        .entries
                        .iter()
                        .any(|e| matches!(e, Entry::Tool(t) if t.expanded));
                self.state.expanded_entries.clear();
                for (index, entry) in self.state.entries.iter_mut().enumerate() {
                    if let Entry::Tool(t) = entry {
                        t.expanded = expand;
                    }
                    if expand && matches!(entry, Entry::User(_) | Entry::Tool(_)) {
                        self.state.expanded_entries.insert(index);
                    }
                }
                self.state.rendered_rows = None;
            }
            KeyCode::Char('p') if ctrl => {
                self.history_move(true);
                return vec![];
            }
            KeyCode::Char('n') if ctrl => {
                self.history_move(false);
                return vec![];
            }
            KeyCode::Char('z') if ctrl && !alt => {
                if self.state.can_undo_attachment() {
                    if key.kind != KeyEventKind::Repeat {
                        self.undo_attachment();
                    }
                } else {
                    self.state.editor.undo();
                }
            }
            KeyCode::F(1) => self.state.open(Overlay::Help),
            KeyCode::PageUp => self.state.scroll = self.state.scroll.saturating_add(10),
            KeyCode::PageDown => self.state.scroll = self.state.scroll.saturating_sub(10),
            KeyCode::End if ctrl => {
                self.state.scroll = 0;
                self.state.search_highlight.clear();
                self.state.focus_entry = None;
            }
            KeyCode::Esc if key.modifiers == KeyModifiers::NONE => return self.escape(),
            KeyCode::Esc => {}
            KeyCode::Enter if alt || key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.state.editor.insert_newline();
            }
            KeyCode::Enter => {
                let text = self.state.editor.lines().join("\n");
                return self.send_editor(text);
            }
            _ => {
                if !crate::composer::handle_key(
                    &mut self.state.editor,
                    &mut self.state.composer,
                    key,
                ) {
                    self.state.editor.input(key);
                }
            }
        }
        if self.state.editor.lines() != before {
            self.state.draft_changed();
            self.clear_draft_notice();
            self.history_index = None;
            self.history_draft = None;
            self.state.slash_selected = 0;
            self.state.slash_dismissed = false;
        }
        vec![]
    }

    fn escape(&mut self) -> Vec<Action> {
        if self.state.busy() {
            self.state.escape_confirmation = None;
            return self.cancel();
        }
        if !self.state.search_highlight.is_empty() {
            self.state.escape_confirmation = None;
            self.state.search_highlight.clear();
            return vec![];
        }
        let now = Instant::now();
        if self
            .state
            .escape_confirmation
            .take()
            .is_none_or(|until| now >= until)
        {
            self.state.escape_confirmation = Some(now + DOUBLE_ESCAPE_TIMEOUT);
            return vec![];
        }
        if self.state.editor.lines().join("\n").is_empty() {
            return self.open_rewind();
        }
        // One editor transaction keeps the complete draft in the existing
        // undo history, including leading/trailing whitespace and line breaks.
        self.state.editor.select_all();
        self.state.editor.delete_char();
        self.state.draft_changed();
        self.history_index = None;
        self.history_draft = None;
        self.state.composer.reset_navigation();
        self.state.slash_selected = 0;
        self.state.slash_dismissed = false;
        self.state.status = DRAFT_CLEARED_STATUS.into();
        vec![]
    }

    fn clear_draft_notice(&mut self) {
        if self.state.status == DRAFT_CLEARED_STATUS {
            self.state.status = "Ready".into();
        }
    }

    /// Select and execute an inline command without routing through a modal query.
    pub fn choose_slash(&mut self, index: usize) -> Vec<Action> {
        let Some(choice) = self.state.slash_choices().get(index).cloned() else {
            return vec![];
        };
        self.state.slash_selected = index;
        self.send_editor(choice.value)
    }

    fn send_editor(&mut self, text: String) -> Vec<Action> {
        let command = text.split_whitespace().next().unwrap_or("");
        if self.state.attachment_loading && !command.starts_with('/') {
            self.state.notice(
                "Reading the selected attachment. Your draft is kept until it is ready.",
                false,
            );
            return vec![];
        }
        if command == "/steer" {
            let payload = text
                .trim_start()
                .strip_prefix("/steer")
                .unwrap_or("")
                .trim_start_matches(char::is_whitespace);
            if payload.trim().is_empty() {
                self.set_draft("/steer ");
                return vec![];
            }
            return self.steer(payload.to_owned());
        }
        if self
            .pending
            .values()
            .any(|pending| matches!(pending, Pending::Steer { draft, .. } if draft == &text))
        {
            self.state.notice("This draft is being sent as a correction. Wait for its acknowledgement or edit a new message.", false);
            return vec![];
        }
        if self.state.busy()
            && (command.starts_with('/') || !self.can_enqueue())
            && !matches!(
                command,
                "/exit"
                    | "/quit"
                    | "/help"
                    | "/"
                    | "/tools"
                    | "/changes"
                    | "/queue"
                    | "/search"
                    | "/logs"
                    | "/cancel"
                    | "/status"
                    | "/attach"
                    | "/attachments"
                    | "/reconnect"
                    | "/retry"
                    | "/model"
                    | "/mode"
                    | "/permissions"
                    | "/tasks"
                    | "/stop"
            )
        {
            self.state.notice(
                if self.state.preparing() {
                    "Still preparing. Your draft is kept; you can keep editing."
                } else {
                    "Still working. Your draft is kept; Esc stops the request."
                },
                false,
            );
            return vec![];
        }
        self.state.draft_changed();
        self.state.editor = TextArea::default();
        self.history_index = None;
        self.history_draft = None;
        self.state.slash_selected = 0;
        self.state.slash_dismissed = false;
        self.input(text)
    }

    fn history_move(&mut self, previous: bool) {
        if self.history.is_empty() {
            return;
        }
        let index = match (previous, self.history_index) {
            (true, None) => {
                self.history_draft = Some(self.state.editor.clone());
                // Automatic queue submissions may append while the user is
                // browsing. Keep the return-to-draft boundary stable.
                self.history_end = self.history.len();
                self.history.len() - 1
            }
            (true, Some(i)) => i.saturating_sub(1),
            (false, Some(i)) if i + 1 < self.history_end => i + 1,
            (false, Some(_)) => {
                if let Some(draft) = self.history_draft.take() {
                    self.state.draft_changed();
                    self.state.editor = draft;
                }
                self.state.composer.reset_navigation();
                self.state.slash_selected = 0;
                self.state.slash_dismissed = false;
                self.history_index = None;
                return;
            }
            (false, None) => return,
        };
        self.history_index = Some(index);
        self.set_draft(&self.history[index].clone());
        // Recalling a slash command must not steal Up/Down for suggestions.
        self.state.slash_dismissed = true;
    }

    fn set_draft(&mut self, text: &str) {
        self.state.draft_changed();
        self.clear_draft_notice();
        self.state.escape_confirmation = None;
        self.state.composer.reset_navigation();
        let text = clean(text);
        self.state.editor = TextArea::from(text.split('\n').map(String::from).collect::<Vec<_>>());
        self.state.editor.move_cursor(CursorMove::Bottom);
        self.state.editor.move_cursor(CursorMove::End);
        self.state.slash_selected = 0;
        self.state.slash_dismissed = false;
    }

    fn insert_command(&mut self, command: &str) {
        if !self.state.editor.lines().join("\n").is_empty() {
            self.state.status =
                "Keep or clear your current draft before inserting a command.".into();
            self.state.notice(self.state.status.clone(), false);
            return;
        }
        self.state.overlay = None;
        self.set_draft(&format!("{command} "));
    }

    fn overlay_key(&mut self, overlay: Overlay, key: KeyEvent) -> Vec<Action> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if overlay == Overlay::Attachments
            && key.code == KeyCode::Char('z')
            && key.modifiers == KeyModifiers::CONTROL
        {
            if key.kind != KeyEventKind::Repeat {
                self.undo_attachment();
            }
            return vec![];
        }
        if overlay == Overlay::Approval {
            return self.approval_key(key);
        }
        let query_before = self.state.query.clone();
        match key.code {
            KeyCode::Char('u') if ctrl && overlay == Overlay::Changes => {
                return self.open_rollback()
            }
            KeyCode::Char('x') if ctrl && overlay == Overlay::Tasks => {
                if let Some(choice) = self.state.choices().get(self.state.selected) {
                    if let Some(id) = choice.value.strip_prefix("job:").map(str::to_owned) {
                        return self.stop_task(&id);
                    }
                }
            }
            KeyCode::Char('r') if ctrl && overlay == Overlay::Tasks => {
                return self.runtime_request(
                    "_aster/session/tasks",
                    json!({}),
                    RuntimeRequest::Tasks,
                );
            }
            KeyCode::Tab if overlay == Overlay::Changes => {
                self.state.conversation_changes = !self.state.conversation_changes;
                self.state.query.clear();
                self.state.selected = 0;
                self.state.detail_scroll = 0;
                if !self.state.conversation_changes {
                    return self.refresh_changes();
                }
            }
            KeyCode::Char('r') if ctrl && overlay == Overlay::Changes => {
                if !self.state.conversation_changes {
                    return self.refresh_changes();
                }
            }
            KeyCode::Esc => {
                self.state.overlay = None;
            }
            KeyCode::Char('s') if ctrl => return self.continue_queue(),
            KeyCode::Delete if overlay == Overlay::Queue => {
                if let Some(index) = self
                    .state
                    .choices()
                    .get(self.state.selected)
                    .and_then(|choice| choice.value.parse::<usize>().ok())
                {
                    self.state.queued.remove(index);
                    self.state.selected = self
                        .state
                        .selected
                        .min(self.state.choices().len().saturating_sub(1));
                    if self.state.queued.is_empty() && !self.state.busy() {
                        self.state.queue_paused = false;
                    }
                }
            }
            KeyCode::Delete if overlay == Overlay::Attachments => {
                if let Some(index) = self
                    .state
                    .choices()
                    .get(self.state.selected)
                    .and_then(|choice| choice.value.parse::<usize>().ok())
                {
                    self.remove_attachment(index);
                }
            }
            KeyCode::PageUp => {
                self.state.detail_scroll = self.state.detail_scroll.saturating_sub(10)
            }
            KeyCode::PageDown => {
                self.state.detail_scroll = self.state.detail_scroll.saturating_add(10)
            }
            KeyCode::Up => {
                self.state.selected = self.state.selected.saturating_sub(1);
                self.state.detail_scroll = 0;
            }
            KeyCode::Down => {
                self.state.selected =
                    (self.state.selected + 1).min(self.state.choices().len().saturating_sub(1));
                self.state.detail_scroll = 0;
            }
            KeyCode::Enter if overlay == Overlay::Attachments => {
                if key.kind != KeyEventKind::Repeat {
                    return self.open_original_image();
                }
            }
            KeyCode::Enter => {
                if let Some(choice) = self.state.choices().get(self.state.selected).cloned() {
                    match overlay {
                        Overlay::Rollback => return self.prepare_rollback(&choice.value),
                        Overlay::RollbackConfirm | Overlay::UndoConfirm => {
                            if choice.value == "apply" {
                                return self.confirm_rollback(overlay == Overlay::UndoConfirm);
                            }
                            self.state.overlay = None;
                            self.state.rollback.plan = None;
                        }
                        Overlay::Commands => {
                            if choice.value == "/resume" {
                                self.state.open(Overlay::Sessions);
                                if !self.state.busy() {
                                    return self.list(ListPurpose::Browse);
                                }
                            } else if choice.value.starts_with('/')
                                && !default_commands().iter().any(|c| c.name == choice.value)
                            {
                                self.insert_command(&choice.value);
                            } else {
                                self.state.overlay = None;
                                return self.input(choice.value);
                            }
                        }
                        Overlay::Sessions => return self.resume(choice.value),
                        Overlay::Drafts => {
                            if let Ok(index) = choice.value.parse::<usize>() {
                                return self.restore_saved_draft(index);
                            }
                        }
                        Overlay::Files => return self.attach_file(choice.value),
                        Overlay::Models => return self.configure(json!({"model":choice.value})),
                        Overlay::Modes => return self.configure(json!({"mode":choice.value})),
                        Overlay::Permissions => {
                            return self.configure(json!({"approval":choice.value}))
                        }
                        Overlay::Rewind => {
                            if let Ok(index) = choice.value.parse::<usize>() {
                                return self.rewind(index);
                            }
                        }
                        Overlay::Tree => return self.navigate_tree(choice.value),
                        Overlay::Tools => self.toggle_tool(&choice.value),
                        Overlay::Queue => {
                            if !self.state.editor.lines().join("\n").is_empty()
                                || !self.state.attachments.is_empty()
                            {
                                self.state.status = "Keep or clear your current draft before editing a queued message.".into();
                                self.state.notice(self.state.status.clone(), false);
                            } else if let Ok(index) = choice.value.parse::<usize>() {
                                if let Some(draft) = self.state.queued.remove(index) {
                                    self.state.queue_paused = true;
                                    self.set_draft(&draft.text);
                                    self.state.attachments = draft.attachments;
                                    self.state.overlay = None;
                                }
                            }
                        }
                        Overlay::Search => {
                            if let Ok(index) = choice.value.parse::<usize>() {
                                if index < self.state.entries.len() {
                                    if matches!(
                                        self.state.entries[index],
                                        Entry::User(_) | Entry::Tool(_)
                                    ) {
                                        self.state.expanded_entries.insert(index);
                                    }
                                    self.state.focus_entry = Some(index);
                                    self.state.search_highlight =
                                        self.state.query.trim().to_owned();
                                    if let Entry::Tool(tool) = &mut self.state.entries[index] {
                                        tool.expanded = true;
                                    }
                                    self.state.overlay = None;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            KeyCode::Backspace => {
                self.state.query.pop();
                self.state.selected = 0;
            }
            KeyCode::Char(' ') if overlay == Overlay::Commands && !ctrl => {
                // Slash commands with arguments remain directly typeable, e.g.
                // /resume <id>, without sending the draft or losing its cursor.
                let prefix = self.state.query.trim_start_matches('/');
                if let Some(command) = self
                    .state
                    .commands
                    .iter()
                    .find(|c| c.name.trim_start_matches('/') == prefix)
                    .cloned()
                {
                    self.insert_command(&command.name);
                } else {
                    self.state.query.push(' ');
                    self.state.selected = 0;
                }
            }
            KeyCode::Char(c)
                if !ctrl
                    && !key.modifiers.contains(KeyModifiers::ALT)
                    && (overlay != Overlay::Changes || !self.state.conversation_changes)
                    && matches!(
                        overlay,
                        Overlay::Commands
                            | Overlay::Sessions
                            | Overlay::Drafts
                            | Overlay::Tools
                            | Overlay::Queue
                            | Overlay::Search
                            | Overlay::Rewind
                            | Overlay::Tree
                            | Overlay::Changes
                    ) =>
            {
                self.state.query.push(c);
                self.state.selected = 0;
                self.state.detail_scroll = 0;
            }
            _ => {}
        }
        if overlay == Overlay::Files && query_before != self.state.query {
            return self.search_files();
        }
        if overlay == Overlay::Tree && query_before != self.state.query {
            self.tree_query_changed();
        }
        vec![]
    }
}

fn approval_preview(value: &Value) -> Option<ApprovalPreview> {
    let summary = value["summary"].as_str()?;
    let details = value["details"].as_array()?;
    let mut complete = value["complete"].as_bool()?;
    if summary.trim().is_empty()
        || summary.len() > 160
        || summary.lines().count() != 1
        || details.is_empty()
        || details.len() > 16
    {
        return None;
    }
    let mut size = 0usize;
    let details = details
        .iter()
        .map(|detail| {
            let raw = detail.as_str()?;
            size = size.saturating_add(raw.len());
            if size > 8 * 1024 || raw.trim().is_empty() {
                return None;
            }
            let visible = clean(raw);
            complete &= visible == raw;
            Some(visible)
        })
        .collect::<Option<Vec<_>>>()?;
    let visible_summary = clean(summary);
    complete &= visible_summary == summary;
    Some(ApprovalPreview {
        summary: visible_summary,
        details,
        complete,
    })
}

fn diff(item: &Value) -> String {
    let path = clean(item["path"].as_str().unwrap_or("file"));
    let old = clean(item["oldText"].as_str().unwrap_or(""));
    let new = clean(item["newText"].as_str().unwrap_or(""));
    let old_lines: Vec<_> = old.lines().collect();
    let new_lines: Vec<_> = new.lines().collect();
    let prefix = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = old_lines[prefix..]
        .iter()
        .rev()
        .zip(new_lines[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    if old == new {
        out.push_str(" (no changes)\n");
        return out;
    }
    let start = prefix.saturating_sub(3);
    let old_end = (old_lines.len() - suffix + 3).min(old_lines.len());
    let new_end = (new_lines.len() - suffix + 3).min(new_lines.len());
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        if old_lines.is_empty() { 0 } else { start + 1 },
        old_end - start,
        if new_lines.is_empty() { 0 } else { start + 1 },
        new_end - start
    ));
    for line in &old_lines[start..prefix] {
        out.push_str(&format!(" {line}\n"));
    }
    for line in &old_lines[prefix..old_lines.len() - suffix] {
        out.push_str(&format!("-{line}\n"));
    }
    for line in &new_lines[prefix..new_lines.len() - suffix] {
        out.push_str(&format!("+{line}\n"));
    }
    for line in &new_lines[new_lines.len() - suffix..new_end] {
        out.push_str(&format!(" {line}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready() -> Controller {
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            false,
        );
        c.ready("Ready");
        c
    }
    fn reply(c: &mut Controller, pending: Pending, result: Value) -> Vec<Action> {
        c.registered(1, pending);
        c.message(json!({"id":1,"result":result}))
    }
    fn update(c: &mut Controller, update: Value) {
        c.message(json!({"method":"session/update","params":{"sessionId":"s","update":update}}));
    }

    fn initialize_reply(c: &mut Controller) -> Vec<Action> {
        reply(
            c,
            Pending::Initialize,
            json!({"protocolVersion":1,
            "agentCapabilities":{"loadSession":true,"sessionCapabilities":{"list":{},"close":{}}}}),
        )
    }

    fn running_with_steering() -> Controller {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.state.steering_supported = true;
        let action = c.input("initial task".into()).pop().unwrap();
        let Action::Request { pending, .. } = action else {
            panic!("prompt request")
        };
        c.registered(90, pending);
        c
    }

    fn preview_permission(id: &str, preview: Value) -> Value {
        json!({
            "id": id,
            "params": {
                "sessionId": "s",
                "toolCall": {
                    "title": "Run shell command?",
                    "rawInput": {"code": "result = await self.shell.run('Get-Location')\nprint(result)"},
                    "_meta": {"aster": {"approvalPreview": preview}}
                },
                "options": [
                    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "allow_session", "name": "Allow all code for this session", "kind": "allow_always"},
                    {"optionId": "reject", "name": "Reject", "kind": "reject_once"}
                ]
            }
        })
    }

    #[test]
    fn approval_preview_toggle_scroll_and_queue_do_not_grant_execution() {
        let mut c = running_with_steering();
        let preview = json!({"summary": "Run shell command", "details": ["Command: Get-Location"], "complete": true});
        assert!(c
            .permission_request(&preview_permission("first", preview))
            .is_empty());
        assert_eq!(c.state.selected, 0);
        assert!(!c.state.approvals[0].show_code);
        assert!(c
            .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .is_empty());
        assert!(c.state.approvals[0].show_code);
        assert_eq!(c.state.selected, 0);
        assert!(c
            .key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.detail_scroll, 1);
        c.state.selected = 2;
        assert!(c
            .permission_request(&preview_permission("second", Value::Null))
            .is_empty());
        assert_eq!(c.state.selected, 2);
        assert_eq!(c.state.detail_scroll, 1);
        assert!(c.state.approvals[0].show_code);
        let actions = c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(&actions[..], [Action::Reply { id, result }]
            if id == "first" && result["outcome"]["outcome"] == "cancelled"));
        assert_eq!(c.state.selected, 0);
        assert_eq!(c.state.detail_scroll, 0);
        assert!(c.state.approvals[0].show_code);
        assert!(c
            .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .is_empty());
        assert!(
            c.state.approvals[0].show_code,
            "unknown code cannot be hidden behind a summary"
        );
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(&actions[..], [Action::Reply { id, result }]
            if id == "second" && result["outcome"]["optionId"] == "allow_once"));
        assert!(c.state.approvals.is_empty());
    }

    #[test]
    fn approval_previews_show_full_code_when_missing_incomplete_or_malformed() {
        for preview in [
            Value::Null,
            json!({"summary": "Run Python code", "details": ["Additional code"], "complete": false}),
            json!({"summary": "Run shell command", "details": ["Command: Get-Location"], "complete": "true"}),
            json!({"summary": "Run shell command", "details": [], "complete": true}),
            json!({"summary": "Run shell command", "details": [7], "complete": true}),
            json!({"summary": "x".repeat(161), "details": ["command"], "complete": true}),
            json!({"summary": "Run shell command", "details": ["x".repeat(8193)], "complete": true}),
            json!({"summary": "Run shell command", "details": vec!["command"; 17], "complete": true}),
            json!({"summary": "Run shell command", "details": ["Command: \u{1b}[31mGet-Location"], "complete": true}),
        ] {
            let mut c = running_with_steering();
            assert!(c
                .permission_request(&preview_permission("first", preview))
                .is_empty());
            let approval = &c.state.approvals[0];
            assert!(approval.show_code);
            assert!(approval.code.contains("self.shell.run"));
            assert_eq!(c.state.selected, 0);
            if let Some(preview) = &approval.preview {
                assert!(!preview
                    .details
                    .iter()
                    .any(|detail| detail.contains('\u{1b}')));
            }
        }
    }

    fn polish_attachment(name: &str) -> crate::attachments::Attachment {
        crate::attachments::Attachment::from_image(name, b"\x89PNG\r\n\x1a\npixels").unwrap()
    }

    #[test]
    fn attachment_enter_opens_the_filtered_original_without_changing_the_draft() {
        let mut c = ready();
        c.paste("保留这段文字".into());
        for name in ["first.png", "chosen.png"] {
            c.add_attachment(0, Ok(polish_attachment(name)));
        }
        c.state.editor.select_all();
        let attachments = c.state.attachments.clone();
        let cursor = c.state.editor.cursor();
        let selection = c.state.editor.selection_range();
        let revision = c.state.draft_revision;
        c.state.open(Overlay::Attachments);
        c.state.query = "chosen".into();

        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            actions.len(),
            1,
            "Enter must open the selected original image"
        );
        let Action::OpenOriginalImage {
            attachment,
            generation,
        } = &actions[0]
        else {
            panic!("original-image action, not a model request");
        };
        assert_eq!(attachment, &attachments[1]);
        assert_eq!(*generation, c.state.conversation_generation);
        assert_eq!(c.state.overlay, Some(Overlay::Attachments));
        assert_eq!(c.state.query, "chosen");
        assert_eq!(c.state.attachments, attachments);
        assert_eq!(c.state.editor.lines(), &["保留这段文字"]);
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(c.state.editor.selection_range(), selection);
        assert_eq!(c.state.draft_revision, revision);
    }

    #[test]
    fn original_image_actions_require_an_image_selection_and_a_new_key_press() {
        let mut c = ready();
        c.add_attachment(0, Ok(polish_attachment("image.png")));
        c.add_attachment(
            0,
            Ok(crate::attachments::Attachment::Text {
                name: "notes.txt".into(),
                uri: "file:///notes.txt".into(),
                text: "context".into(),
            }),
        );
        assert!(c.open_original_image().is_empty());
        c.state.open(Overlay::Attachments);
        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            assert!(c
                .key(KeyEvent::new_with_kind(
                    KeyCode::Enter,
                    KeyModifiers::NONE,
                    kind
                ))
                .is_empty());
        }
        c.state.selected = 1;
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
        c.state.query = "no match".into();
        assert!(c.open_original_image().is_empty());
        c.state.query.clear();
        c.state.selected = usize::MAX;
        assert!(c.open_original_image().is_empty());
        c.state.open(Overlay::Commands);
        assert!(c.open_original_image().is_empty());
    }

    #[test]
    fn composer_attachment_undo_restores_original_position_and_snapshot() {
        let mut c = ready();
        c.paste("保留文字\nsecond line".into());
        for name in ["first.png", "middle.png", "last.png"] {
            c.add_attachment(0, Ok(polish_attachment(name)));
        }
        c.state.editor.select_all();
        let text = c.state.editor.lines().to_vec();
        let cursor = c.state.editor.cursor();
        let selection = c.state.editor.selection_range();
        let attachments = c.state.attachments.clone();
        c.state.queued.push_back(MessageDraft {
            text: "queued independently".into(),
            attachments: attachments.clone(),
        });
        assert!(c.remove_attachment(1));
        assert!(c.open_attachment(1));
        assert_eq!(c.state.selected, 1);
        c.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(c.state.attachments, attachments);
        assert_eq!(c.state.queued[0].attachments, attachments);
        assert_eq!(c.state.editor.lines(), text);
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(c.state.editor.selection_range(), selection);
        assert_eq!(c.state.selected, 1);
        assert!(!c.state.can_undo_attachment());
    }

    #[test]
    fn composer_attachment_undo_yields_to_the_next_actual_edit() {
        let mut c = ready();
        c.paste("draft".into());
        c.add_attachment(0, Ok(polish_attachment("kept.png")));
        c.add_attachment(0, Ok(polish_attachment("removed.png")));
        c.remove_attachment(1);
        c.key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        c.add_attachment(0, Err(anyhow::anyhow!("fixture read failure")));
        c.add_attachment(0, Ok(polish_attachment("kept.png")));
        c.paste(String::new());
        assert!(c.state.can_undo_attachment());
        c.key(KeyEvent::new(KeyCode::Char('!'), KeyModifiers::NONE));
        assert!(!c.state.can_undo_attachment());
        c.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(c.state.editor.lines(), &["draft"]);
        assert_eq!(c.state.attachments.len(), 1);
        assert!(!c.undo_attachment());
        c.remove_attachment(0);
        c.add_attachment(0, Ok(polish_attachment("new.png")));
        assert!(!c.state.can_undo_attachment());
    }

    #[test]
    fn composer_attachment_undo_stays_with_its_draft() {
        for boundary in [
            "send", "input", "queue", "recall", "switch", "clear", "history",
        ] {
            let mut c = ready();
            c.state.session_id = Some("s".into());
            c.paste("draft".into());
            c.add_attachment(0, Ok(polish_attachment("removed.png")));
            c.remove_attachment(0);
            match boundary {
                "send" => {
                    c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                }
                "input" => {
                    c.input("direct input".into());
                }
                "queue" => {
                    c.enqueue("queued".into());
                }
                "recall" => c.set_draft("another draft"),
                "switch" => c.reset_conversation(),
                "clear" => {
                    c.escape();
                    c.escape();
                }
                "history" => {
                    c.history.push("earlier input".into());
                    c.history_move(true);
                    c.history_move(false);
                }
                _ => unreachable!(),
            }
            assert!(!c.state.can_undo_attachment(), "{boundary}");
            assert!(!c.undo_attachment(), "{boundary}");
        }
    }

    #[test]
    fn composer_attachment_undo_survives_independent_queue_submission_and_warmup() {
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            false,
        );
        assert!(c.input("send this separately".into()).is_empty());
        c.paste("my draft".into());
        c.add_attachment(0, Ok(polish_attachment("removed.png")));
        c.remove_attachment(0);
        initialize_reply(&mut c);
        reply(
            &mut c,
            Pending::List(ListPurpose::StartNew),
            json!({"sessions":[]}),
        );
        assert!(c.state.can_undo_attachment());
        let actions = reply(
            &mut c,
            Pending::Prepare,
            json!({"sessionId":"prepared-session"}),
        );
        assert!(matches!(
            actions.as_slice(),
            [Action::Request {
                pending: Pending::Prompt,
                ..
            }]
        ));
        assert!(c.state.queued.is_empty());
        assert!(c.undo_attachment());
        assert_eq!(c.state.editor.lines(), &["my draft"]);
        assert_eq!(c.state.attachments[0].name(), "removed.png");
    }

    #[test]
    fn composer_attachment_undo_keeps_a_failed_restore_available() {
        let mut c = ready();
        c.add_attachment(0, Ok(polish_attachment("restore.png")));
        c.remove_attachment(0);
        for index in 0..crate::attachments::MAX_ATTACHMENTS {
            c.state
                .attachments
                .push(polish_attachment(&format!("{index}.png")));
        }
        assert!(!c.undo_attachment());
        assert_eq!(
            c.state.attachments.len(),
            crate::attachments::MAX_ATTACHMENTS
        );
        assert!(c.state.can_undo_attachment());
        c.state.attachments.pop();
        assert!(c.undo_attachment());
        assert_eq!(c.state.attachments[0].name(), "restore.png");
    }

    #[test]
    fn composer_copy_and_cut_require_a_nonempty_focused_selection() {
        let mut c = ready();
        c.paste("前中文\n👩‍💻 end".into());
        c.state.editor.move_cursor(CursorMove::Jump(0, 1));
        c.state.editor.start_selection();
        c.state.editor.move_cursor(CursorMove::Jump(1, 3));
        let original = c.state.editor.lines().to_vec();
        for (code, cut) in [('c', false), ('x', true)] {
            let actions = c.key(KeyEvent::new(KeyCode::Char(code), KeyModifiers::CONTROL));
            assert!(
                matches!(actions.as_slice(), [Action::CopySelection { text, cut: actual }]
                if text == "中文\n👩‍💻" && *actual == cut)
            );
            assert_eq!(c.state.editor.lines(), original);
            assert!(c.state.exit_confirmation.is_none());
            assert!(c.state.overlay.is_none());
        }
        c.state.open(Overlay::Help);
        assert!(c
            .key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .is_empty());
        assert!(c.state.exit_confirmation.is_some());
        c.state.overlay = None;
        c.state.editor.cancel_selection();
        c.state.exit_confirmation = None;
        c.key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
        assert_eq!(c.state.overlay, Some(Overlay::Help));
    }

    fn register_steer(c: &mut Controller, actions: Vec<Action>) -> Value {
        let Action::Request {
            method,
            params,
            pending,
        } = actions.into_iter().next().unwrap()
        else {
            panic!("steering request")
        };
        assert_eq!(method, "_aster/session/steer");
        c.registered(91, pending);
        params
    }

    #[test]
    fn steering_keeps_enter_as_queue_and_waits_for_ack_before_clearing() {
        let mut c = running_with_steering();
        c.paste("next turn".into());
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.queued.front().unwrap(), "next turn");
        c.paste("  改为只检查\nkeep indentation\n".into());
        let original = c.state.editor.lines().join("\n");
        let started = c.state.started;
        let actions = c.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::ALT));
        let params = register_steer(&mut c, actions);
        assert_eq!(params["text"], original);
        assert_eq!(c.state.editor.lines().join("\n"), original);
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Sending);
        assert!(c
            .message(json!({"id":91,"result":{
                "turnId":params["turnId"],"messageId":params["messageId"],"state":"accepted"
            }}))
            .is_empty());
        assert_eq!(c.state.editor.lines(), &[""]);
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Accepted);
        assert_eq!(c.operation, "prompt");
        assert_eq!(c.state.started, started);
        assert_eq!(c.state.phase, Phase::Busy);
        update(
            &mut c,
            json!({"sessionUpdate":"session_info_update","_meta":{"aster":{
                "steering":{"messageId":params["messageId"],"turnId":params["turnId"],"state":"applied"}
            }}}),
        );
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Applied);
        assert_eq!(c.state.queued.len(), 1);
    }

    #[test]
    fn steering_rejection_never_finishes_the_active_prompt_or_overwrites_new_input() {
        let mut c = running_with_steering();
        c.paste("/steer check only\nwith 中文".into());
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let params = register_steer(&mut c, actions);
        assert_eq!(params["text"], "check only\nwith 中文");
        c.set_draft("new draft");
        assert!(c
            .message(json!({"id":91,"error":{"code":-32000,"message":"Turn already finished"}}))
            .is_empty());
        assert_eq!(c.state.phase, Phase::Busy);
        assert_eq!(c.operation, "prompt");
        assert!(c.state.started.is_some());
        assert_eq!(c.state.editor.lines(), &["new draft"]);
        assert_eq!(c.state.queued.front().unwrap(), "check only\nwith 中文");
        assert!(c.state.queue_paused);
    }

    #[test]
    fn steering_ack_reordering_and_completion_preserve_current_turn_state() {
        let mut c = running_with_steering();
        c.paste("correction".into());
        let actions = c.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::ALT));
        let params = register_steer(&mut c, actions);
        update(
            &mut c,
            json!({"sessionUpdate":"session_info_update","_meta":{"aster":{
                "steering":{"messageId":params["messageId"],"turnId":params["turnId"],"state":"applied"}
            }}}),
        );
        c.message(json!({"id":90,"result":{"stopReason":"end_turn"}}));
        c.set_draft("a newer draft");
        c.message(json!({"id":91,"result":{"turnId":params["turnId"],"messageId":params["messageId"],"state":"accepted"}}));
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.state.active_turn_id.is_none());
        assert_eq!(c.state.steering_messages.len(), 1);
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Applied);
        assert_eq!(c.state.editor.lines(), &["a newer draft"]);
    }

    #[test]
    fn steering_unknown_notifications_keep_the_pending_draft() {
        let mut c = running_with_steering();
        c.paste("preserve this correction".into());
        let actions = c.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::ALT));
        let params = register_steer(&mut c, actions);
        let entry_count = c.state.entries.len();
        for status in [json!("future_state"), Value::Null, json!({"accepted":null})] {
            for message_id in [params["messageId"].clone(), json!("unknown-message")] {
                update(
                    &mut c,
                    json!({"sessionUpdate":"session_info_update","_meta":{"aster":{
                        "steering":{"messageId":message_id,"turnId":params["turnId"],
                            "text":"unknown update text","state":status,"reason":"ignore me"}
                    }}}),
                );
            }
        }
        assert_eq!(c.state.steering_messages.len(), 1);
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Sending);
        assert!(c.state.steering_messages[0].reason.is_empty());
        assert_eq!(c.state.entries.len(), entry_count);
        assert_eq!(c.state.editor.lines(), &["preserve this correction"]);
        assert!(c.state.queued.is_empty());
        c.message(json!({"id":91,"result":{
            "turnId":params["turnId"],"messageId":params["messageId"],"state":"accepted"
        }}));
        assert_eq!(c.state.editor.lines(), &[""]);
        assert_eq!(c.state.steering_messages[0].state, SteeringState::Accepted);
    }

    #[test]
    fn steering_unknown_ack_retains_input_and_active_prompt() {
        for status in [json!("future_state"), Value::Null, json!({"accepted":null})] {
            let mut c = running_with_steering();
            c.paste("preserve this correction".into());
            let actions = c.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::ALT));
            let params = register_steer(&mut c, actions);
            c.message(json!({"id":91,"result":{
                "turnId":params["turnId"],"messageId":params["messageId"],"state":status
            }}));
            assert_eq!(c.state.steering_messages[0].state, SteeringState::Rejected);
            assert_eq!(c.state.editor.lines(), &["preserve this correction"]);
            assert!(c.state.queued.is_empty());
            assert!(c.state.queue_paused);
            assert_eq!(c.state.phase, Phase::Busy);
            assert_eq!(c.operation, "prompt");
        }
    }

    #[test]
    fn unavailable_steering_preserves_drafts_and_cancel_deadline() {
        for phase in [Phase::Ready, Phase::Preparing, Phase::Cancelling] {
            let mut c = running_with_steering();
            c.state.phase = phase;
            c.paste("correction".into());
            let deadline = c.deadline;
            assert!(c
                .key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::ALT))
                .is_empty());
            assert_eq!(c.state.editor.lines(), &["correction"]);
            assert_eq!(c.deadline, deadline);
        }
        let mut c = running_with_steering();
        c.state.steering_supported = false;
        c.paste("/steer do less".into());
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.editor.lines(), &["/steer do less"]);
    }

    #[test]
    fn warmup_starts_without_a_message_and_keeps_draft_cursor_and_undo() {
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            false,
        );
        c.paste("draft 中文".into());
        c.state.editor.move_cursor(CursorMove::Back);
        let cursor = c.state.editor.cursor();
        assert!(matches!(
            initialize_reply(&mut c).as_slice(),
            [Action::Request {
                pending: Pending::List(ListPurpose::StartNew),
                ..
            }]
        ));
        let actions = reply(
            &mut c,
            Pending::List(ListPurpose::StartNew),
            json!({"sessions":[]}),
        );
        assert!(
            matches!(actions.as_slice(), [Action::Request { method:"session/new", pending:Pending::Prepare, params }]
            if params["_meta"]["aster"]["prepare"] == true)
        );
        assert!(c.state.preparing());
        assert!(reply(&mut c, Pending::Prepare, json!({"sessionId":"warmed"})).is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.state.entries.is_empty());
        assert!(c.state.sessions.is_empty());
        assert!(c.history.is_empty());
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(c.state.editor.lines(), &["draft 中文"]);
        assert_eq!(c.state.conversation_generation, 0);
        assert!(c.state.editor.undo());
        assert_eq!(c.state.editor.lines(), &[""]);
    }

    #[test]
    fn warmup_early_enter_and_escape_work_in_every_startup_stage() {
        for stage in 0..3 {
            for recall in [false, true] {
                let mut c = Controller::new(
                    State::new("C:/work".into(), "test".into(), false),
                    None,
                    false,
                );
                if stage >= 1 {
                    initialize_reply(&mut c);
                }
                if stage >= 2 {
                    reply(
                        &mut c,
                        Pending::List(ListPurpose::StartNew),
                        json!({"sessions":[]}),
                    );
                }
                c.paste("  early 中文\nnext\n".into());
                assert!(c
                    .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
                    .is_empty());
                assert_eq!(c.state.queued.len(), 1);
                assert!(c.history.is_empty());
                assert!(c.state.entries.is_empty());
                assert!(c.state.started.is_none());
                if recall {
                    assert!(c
                        .key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
                        .is_empty());
                    assert!(c.state.preparing());
                    assert!(!c.cancelled);
                    assert!(c.state.queued.is_empty());
                    assert_eq!(c.state.editor.lines(), &["  early 中文", "next", ""]);
                }
                if stage < 1 {
                    initialize_reply(&mut c);
                }
                if stage < 2 {
                    reply(
                        &mut c,
                        Pending::List(ListPurpose::StartNew),
                        json!({"sessions":[]}),
                    );
                }
                let actions = reply(&mut c, Pending::Prepare, json!({"sessionId":"s"}));
                if recall {
                    assert!(actions.is_empty());
                    assert!(c.state.entries.is_empty());
                    assert!(c.history.is_empty());
                    let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                    assert_eq!(prompt_text(&actions), "  early 中文\nnext\n");
                } else {
                    assert_eq!(prompt_text(&actions), "  early 中文\nnext\n");
                }
                assert_eq!(c.history, ["  early 中文\nnext\n"]);
                assert_eq!(c.state.entries.len(), 1);
            }
        }
    }

    #[test]
    fn warmup_recall_keeps_another_draft_and_pauses_all_waiting_messages() {
        let mut c = ready();
        c.input("first".into());
        c.input("second".into());
        c.paste("another draft".into());
        c.state.editor.move_cursor(CursorMove::Back);
        let cursor = c.state.editor.cursor();
        assert!(c.cancel().is_empty());
        assert_eq!(c.state.overlay, Some(Overlay::Queue));
        assert_eq!(c.state.editor.lines(), &["another draft"]);
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(
            c.state
                .queued
                .iter()
                .map(MessageDraft::as_str)
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(c.state.queue_paused);
        assert!(reply(&mut c, Pending::Prepare, json!({"sessionId":"s"})).is_empty());
        assert!(c.history.is_empty());
        assert_eq!(prompt_text(&c.continue_queue()), "first");
    }

    #[test]
    fn warmup_resume_preserves_early_input_and_does_not_create_an_extra_session() {
        for resume in [None, Some("old".to_owned())] {
            for recall in [false, true] {
                let purpose = if resume.is_some() {
                    ListPurpose::Resume("old".into())
                } else {
                    ListPurpose::Continue
                };
                let mut c = Controller::new(
                    State::new("C:/work".into(), "test".into(), false),
                    resume.clone(),
                    resume.is_none(),
                );
                c.input("follow up".into());
                if recall {
                    c.cancel();
                }
                initialize_reply(&mut c);
                let actions = reply(
                    &mut c,
                    Pending::List(purpose),
                    json!({"sessions":[{"sessionId":"old","title":"saved"}]}),
                );
                assert!(matches!(
                    actions.as_slice(),
                    [Action::Request {
                        method: "session/load",
                        ..
                    }]
                ));
                assert!(c.state.preparing());
                assert_eq!(c.state.conversation_generation, 0);
                let actions = reply(
                    &mut c,
                    Pending::Load {
                        id: "old".into(),
                        previous: None,
                    },
                    json!({}),
                );
                if recall {
                    assert!(actions.is_empty());
                    assert_eq!(c.state.editor.lines(), &["follow up"]);
                } else {
                    assert_eq!(prompt_text(&actions), "follow up");
                }
                assert_eq!(c.state.session_id.as_deref(), Some("old"));
            }
        }
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            true,
        );
        initialize_reply(&mut c);
        assert!(matches!(
            reply(
                &mut c,
                Pending::List(ListPurpose::Continue),
                json!({"sessions":[]})
            )
            .as_slice(),
            [Action::Request {
                pending: Pending::Prepare,
                ..
            }]
        ));
    }

    #[test]
    fn warmup_failure_keeps_waiting_input_until_explicit_retry() {
        let mut c = ready();
        c.input("waiting".into());
        c.paste("draft".into());
        c.registered(7, Pending::Prepare);
        assert!(c
            .message(json!({"id":7,"error":{"message":"invalid model configuration"}}))
            .is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.state.queue_paused);
        assert_eq!(c.state.editor.lines(), &["draft"]);
        assert_eq!(
            c.state.queued.front().map(MessageDraft::as_str),
            Some("waiting")
        );
        assert!(c.history.is_empty());
        assert!(matches!(
            c.continue_queue().as_slice(),
            [Action::Request {
                pending: Pending::Prepare,
                ..
            }]
        ));
        assert_eq!(
            prompt_text(&reply(&mut c, Pending::Prepare, json!({"sessionId":"s"}))),
            "waiting"
        );
    }

    #[test]
    fn warmup_timeout_keeps_input_and_late_success_does_not_send_it_unasked() {
        let mut c = ready();
        c.input("waiting".into());
        c.deadline = Some(Instant::now() - Duration::from_millis(1));
        assert!(c.tick().is_empty());
        assert!(c.state.preparing());
        assert!(c.fatal_error.is_none());
        assert!(c.state.queue_paused);
        let entries = c.state.entries.len();
        assert!(c.tick().is_empty());
        assert_eq!(c.state.entries.len(), entries);
        assert!(reply(&mut c, Pending::Prepare, json!({"sessionId":"s"})).is_empty());
        assert_eq!(prompt_text(&c.continue_queue()), "waiting");
    }

    #[test]
    fn warmup_new_conversation_prepares_again_and_exit_does_not_wait_for_it() {
        let mut c = ready();
        c.state.session_id = Some("old".into());
        c.input("/new".into());
        assert!(matches!(
            reply(&mut c, Pending::NewClose, json!({})).as_slice(),
            [Action::Request {
                pending: Pending::Prepare,
                ..
            }]
        ));
        assert_eq!(c.state.conversation_generation, 1);
        assert!(ctrl_c(&mut c).is_empty());
        assert!(matches!(ctrl_c(&mut c).as_slice(), [Action::Exit]));
        assert!(c.fatal_error.is_none());
    }

    #[test]
    fn warmup_repeated_session_page_pauses_input_after_failed_restore() {
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            Some("old".into()),
            false,
        );
        c.input("waiting for old".into());
        initialize_reply(&mut c);
        for _ in 0..2 {
            reply(
                &mut c,
                Pending::List(ListPurpose::Resume("old".into())),
                json!({"sessions":[],"nextCursor":"again"}),
            );
        }
        assert!(c.state.queue_paused);
        assert!(c.input("another message".into()).is_empty());
        assert_eq!(c.state.queued.len(), 2);
        assert!(c.state.session_id.is_none());
    }
    #[test]
    fn streaming_reply_renders_before_completion_and_reconciles_without_duplicates() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("task".into());
        c.paste("unsent draft".into());
        c.state.editor.move_cursor(CursorMove::Jump(0, 3));
        let notify = |id: &str, text: &str, state: &str, replace: bool| {
            json!({
                "sessionUpdate":"agent_message_chunk", "content":{"type":"text","text":text},
                "_meta":{"aster":{"messageId":id,"messageState":state,"replace":replace}}
            })
        };
        update(&mut c, notify("m", "First ", "delta", false));
        assert!(c.state.busy());
        assert!(matches!(c.state.entries.last(), Some(Entry::Assistant(text)) if text == "First "));
        update(
            &mut c,
            json!({"sessionUpdate":"tool_call_update", "toolCallId":"tool", "title":"Ran Python", "status":"completed"}),
        );
        update(&mut c, notify("m", "中文\nnext", "delta", false));
        update(&mut c, notify("m", "First 中文\nnext!", "complete", true));
        update(&mut c, notify("m", "late fragment", "delta", false));
        update(&mut c, notify("m", "First 中文\nnext!", "complete", true));
        let messages: Vec<_> = c
            .state
            .entries
            .iter()
            .filter_map(|e| {
                if let Entry::Assistant(text) = e {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(messages, ["First 中文\nnext!"]);
        assert_eq!(c.state.editor.lines(), &["unsent draft"]);
        assert_eq!(c.state.editor.cursor(), (0, 3));
        reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert_eq!(c.state.phase, Phase::Ready);
    }

    #[test]
    fn streaming_filters_split_controls_and_keeps_cancelled_text_distinct_from_replay() {
        let mut c = ready();
        let notify = |id: &str, text: &str, state: &str, replace: bool| {
            json!({
                "sessionUpdate":"agent_message_chunk", "content":{"text":text},
                "_meta":{"aster":{"messageId":id,"messageState":state,"replace":replace}}
            })
        };
        for text in ["visible\x1b]52;", "c;secret", "\x07 tail"] {
            update(&mut c, notify("one", text, "delta", false));
        }
        update(&mut c, notify("one", "", "interrupted", false));
        update(&mut c, notify("one", "late", "delta", false));
        assert!(matches!(&c.state.entries[0], Entry::Assistant(text) if text == "visible tail"));
        assert!(
            matches!(&c.state.entries[1], Entry::Notice{text,..} if text.contains("incomplete"))
        );
        c.reset_conversation();
        update(&mut c, notify("one", "replayed first", "complete", true));
        update(&mut c, notify("two", "replayed second", "complete", true));
        assert_eq!(c.state.entries.len(), 2);
        assert!(matches!(&c.state.entries[0], Entry::Assistant(text) if text == "replayed first"));
        assert!(matches!(&c.state.entries[1], Entry::Assistant(text) if text == "replayed second"));
    }

    #[test]
    fn workspace_changes_refresh_runs_independently_and_keeps_draft() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("active task".into());
        c.paste("draft stays".into());
        assert!(matches!(
            c.key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL))
                .as_slice(),
            [Action::LoadWorkspaceChanges]
        ));
        assert!(c.state.busy());
        assert!(c.state.changes_loading);
        assert!(c.refresh_changes().is_empty());
        c.key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(c.state.conversation_changes);
        c.state.detail_scroll = 50;
        c.changes_result(Err(anyhow::anyhow!("not a Git repository")));
        assert_eq!(c.state.detail_scroll, 50);
        assert!(!c.state.changes_loading);
        assert!(c.state.conversation_changes);
        assert!(c.state.busy());
        assert!(matches!(
            c.key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
                .as_slice(),
            [Action::LoadWorkspaceChanges]
        ));
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        c.changes_result(Ok(crate::workspace_changes::WorkspaceChanges {
            root: "C:/work".into(),
            files: vec![],
            notice: None,
        }));
        assert!(c.state.overlay.is_none());
        assert_eq!(c.state.editor.lines(), &["draft stays"]);
        assert!(c.state.queued.is_empty());
    }

    #[test]
    fn warmup_recall_retains_handle_and_skips_prompt() {
        let mut c = ready();
        assert!(c.state.session_id.is_none());
        let actions = c.input("中文任务".into());
        assert!(matches!(
            &actions[0],
            Action::Request {
                pending: Pending::Prepare,
                ..
            }
        ));
        assert!(c.cancel().is_empty());
        assert!(reply(&mut c, Pending::Prepare, json!({"sessionId":"s"})).is_empty());
        assert_eq!(c.state.session_id.as_deref(), Some("s"));
        assert!(!c.state.busy());
        assert_eq!(c.state.editor.lines(), &["中文任务"]);
        assert!(c.history.is_empty());
        assert!(matches!(
            c.input("next".into())[0],
            Action::Request {
                pending: Pending::Prompt,
                ..
            }
        ));
    }
    #[test]
    fn busy_draft_and_cursor_survive_events_and_picker() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("task".into());
        c.paste("中文\nsecond line".into());
        c.state.editor.move_cursor(CursorMove::Head);
        let cursor = c.state.editor.cursor();
        update(
            &mut c,
            json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"reply"}}),
        );
        c.key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["中文", "second line"]);
        assert_eq!(c.state.editor.cursor(), cursor);
    }
    #[test]
    fn snapshot_output_replaces_and_completed_tools_remain_browsable() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        for text in ["one", "one\ntwo"] {
            update(
                &mut c,
                json!({"sessionUpdate":"tool_call_update","toolCallId":"t","status":"in_progress","content":[{"type":"content","content":{"type":"text","text":text}}]}),
            );
        }
        update(
            &mut c,
            json!({"sessionUpdate":"tool_call_update","toolCallId":"t","status":"completed"}),
        );
        update(
            &mut c,
            json!({"sessionUpdate":"tool_call","toolCallId":"aster-progress-1","title":"Working"}),
        );
        c.state.open(Overlay::Tools);
        assert_eq!(c.state.choices().len(), 1);
        assert_eq!(c.state.selected_tool().unwrap().content, "one\ntwo");
        assert_eq!(c.state.selected_tool().unwrap().status, "completed");
    }

    #[test]
    fn backend_duration_survives_batched_tool_notifications() {
        let mut c = ready();
        for status in ["in_progress", "completed"] {
            update(
                &mut c,
                json!({"sessionUpdate":"tool_call_update", "toolCallId":"timed",
                "title":"Ran Python", "status":status,
                "_meta":{"aster":{"elapsedSeconds":1.237}}}),
            );
        }
        let Entry::Tool(tool) = &c.state.entries[0] else {
            panic!("tool missing")
        };
        assert_eq!(tool.elapsed, Some(1.237));
    }

    #[test]
    fn python_tool_content_hides_transport_fences_and_keeps_literal_output() {
        let mut c = ready();
        let source = "import subprocess\nprint('```literal```')\n";
        let output = "  import of 'subprocess' is blocked.\n\n```literal output```\n";
        // Live progress and restored final cards use the same content contract.
        for kind in ["tool_call_update", "tool_call"] {
            update(
                &mut c,
                json!({"sessionUpdate":kind,"toolCallId":"python","title":"Python failed","status":"failed",
                "content":[
                    {"type":"content","content":{"type":"text","text":format!("````python\n{source}\n````"),"_meta":{"aster":{"format":"fenced-code"}}}},
                    {"type":"content","content":{"type":"text","text":format!("````text\n{output}\n````"),"_meta":{"aster":{"format":"fenced-code"}}}}
                ]}),
            );
            c.state.open(Overlay::Tools);
            assert_eq!(
                c.state.selected_tool().unwrap().content,
                format!("{source}\n{output}")
            );
            c.state.open(Overlay::Search);
            c.state.query = "literal output".into();
            assert_eq!(c.state.choices().len(), 1);
            c.state.open(Overlay::Tools);
        }
        // Terminal output that happens to contain Markdown remains literal.
        for (text, meta) in [
            ("```python\nprint('terminal output')\n```", json!({})),
            (
                "```python\nunfinished",
                json!({"aster":{"format":"fenced-code"}}),
            ),
            ("ordinary output", json!({"aster":{"format":"fenced-code"}})),
        ] {
            update(
                &mut c,
                json!({"sessionUpdate":"tool_call_update","toolCallId":"python","content":[
                    {"type":"content","content":{"type":"text","text":text,"_meta":meta}}
                ]}),
            );
            assert_eq!(c.state.selected_tool().unwrap().content, text);
        }
    }

    #[test]
    fn completed_only_tool_has_no_fabricated_elapsed_time() {
        let mut c = ready();
        update(
            &mut c,
            json!({"sessionUpdate":"tool_call", "toolCallId":"file-edit",
            "title":"Edited file", "status":"completed"}),
        );
        let Entry::Tool(tool) = &c.state.entries[0] else {
            panic!("tool missing")
        };
        assert_eq!(tool.elapsed, None);
    }

    #[test]
    fn failed_python_card_renders_error_and_source_without_transport_markup() {
        use ratatui::{backend::TestBackend, Terminal};

        let source = format!(
            "import subprocess\n{}",
            (1..=11)
                .map(|n| format!("# source line {n}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let error = "Cell In[1], line 1\n    ^\nimport of 'subprocess' is blocked.";
        for columns in [48, 100] {
            let mut c = ready();
            update(
                &mut c,
                json!({
                    "sessionUpdate":"tool_call_update","toolCallId":"python","title":"Python failed","status":"failed",
                    "_meta":{"aster":{"elapsedSeconds":0.01}},
                    "content":[
                        {"type":"content","content":{"type":"text","text":format!("```python\n{source}\n```"),"_meta":{"aster":{"format":"fenced-code"}}}},
                        {"type":"content","content":{"type":"text","text":format!("```text\n{error}\n```"),"_meta":{"aster":{"format":"fenced-code"}}}}
                    ]
                }),
            );
            c.toggle_tool("python");
            for view in [0, 1, 2] {
                if view == 1 {
                    c.toggle_entry(0);
                } else if view == 2 {
                    c.state.open(Overlay::Tools);
                }
                let mut terminal = Terminal::new(TestBackend::new(columns, 40)).unwrap();
                terminal
                    .draw(|frame| {
                        crate::ui::draw(frame, &mut c.state);
                    })
                    .unwrap();
                let screen = terminal
                    .backend()
                    .buffer()
                    .content
                    .chunks(columns as usize)
                    .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(!screen.contains("```"), "{screen}");
                assert!(!screen.contains("failed · failed"), "{screen}");
                if view == 0 {
                    assert!(screen.contains("10 more lines"), "{screen}");
                    assert!(screen.contains("is blocked."), "{screen}");
                    assert!(!screen.contains("# source line 5"), "{screen}");
                } else {
                    assert!(screen.contains("# source line 5"), "{screen}");
                }
            }
        }
    }
    #[test]
    fn failed_resume_restores_conversation_and_does_not_close_previous() {
        let mut c = ready();
        c.state.session_id = Some("old".into());
        c.state.title = "original".into();
        c.state.entries.push(Entry::User("kept".into()));
        c.state.expanded_entries.insert(0);
        let actions = c.resume("other".into());
        assert!(c.state.expanded_entries.is_empty());
        assert!(matches!(
            &actions[0],
            Action::Request {
                pending: Pending::Load { .. },
                ..
            }
        ));
        c.registered(
            2,
            Pending::Load {
                id: "other".into(),
                previous: Some("old".into()),
            },
        );
        c.message(json!({"method":"session/update","params":{"sessionId":"other","update":{"sessionUpdate":"agent_message_chunk","content":{"text":"partial replay"}}}}));
        let actions = c.message(json!({"id":2,"error":{"message":"in use"}}));
        assert!(actions.is_empty());
        assert_eq!(c.state.title, "original");
        assert_eq!(c.state.session_id.as_deref(), Some("old"));
        assert!(matches!(&c.state.entries[0],Entry::User(s) if s=="kept"));
        assert_eq!(c.state.entries.len(), 2);
        assert_eq!(c.state.expanded_entries, HashSet::from([0]));
    }
    #[test]
    fn restore_accepts_replay_before_response_and_closes_previous_after_success() {
        let mut c = ready();
        c.state.session_id = Some("old".into());
        c.resume("new".into());
        c.message(json!({"method":"session/update","params":{"sessionId":"new","update":{"sessionUpdate":"user_message_chunk","content":{"text":"restored"}}}}));
        let actions = reply(
            &mut c,
            Pending::Load {
                id: "new".into(),
                previous: Some("old".into()),
            },
            json!({}),
        );
        assert_eq!(c.state.session_id.as_deref(), Some("new"));
        assert!(
            matches!(&actions[0],Action::Request{method:"session/close",params,..} if params["sessionId"]=="old")
        );
        assert!(matches!(&c.state.entries[0],Entry::User(s) if s=="restored"));
    }
    fn esc(c: &mut Controller) -> Vec<Action> {
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
    }

    fn rewind_ready() -> Controller {
        let mut c = ready();
        c.rewind_supported = true;
        c.state.session_id = Some("old".into());
        c.state.entries.push(Entry::User("first".into()));
        c.state.entries.push(Entry::Assistant("first reply".into()));
        c.state.entries.push(Entry::User("second".into()));
        c.state
            .entries
            .push(Entry::Assistant("second reply".into()));
        c
    }

    fn rewind_choices(c: &mut Controller) {
        c.input("/rewind".into());
        let generation = c.state.conversation_generation;
        reply(
            c,
            Pending::Checkpoints {
                source: "old".into(),
                generation,
            },
            json!({
                "checkpoints":[{"userMessageIndex":0,"prompt":"first"},
                    {"userMessageIndex":1,"prompt":"  second 中文\nnext  \n"}]
            }),
        );
    }

    fn rewind_response(c: &mut Controller) -> Vec<Action> {
        let generation = c.state.conversation_generation;
        reply(
            c,
            Pending::Rewind {
                source: "old".into(),
                generation,
            },
            json!({"sessionId":"fork","prompt":"  second 中文\nnext  \n"}),
        )
    }

    #[test]
    fn double_escape_clears_the_whole_draft_as_one_undoable_edit() {
        for undo in ['z', 'u'] {
            let mut c = ready();
            c.paste("  中文\n  next  \n".into());
            c.state.editor.move_cursor(CursorMove::Jump(1, 3));
            c.state.editor.start_selection();
            c.state.editor.move_cursor(CursorMove::Back);
            let cursor = c.state.editor.cursor();
            let selection = c.state.editor.selection_range();
            c.state.queued.push_back("queued stays".into());
            assert!(esc(&mut c).is_empty());
            assert_eq!(c.state.editor.lines(), &["  中文", "  next  ", ""]);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert_eq!(c.state.editor.selection_range(), selection);
            assert!(esc(&mut c).is_empty());
            assert_eq!(c.state.editor.lines(), &[""]);
            assert!(c.state.escape_confirmation.is_none());
            c.key(KeyEvent::new(KeyCode::Char(undo), KeyModifiers::CONTROL));
            assert_eq!(c.state.editor.lines(), &["  中文", "  next  ", ""]);
            assert_eq!(c.state.status, "Ready");
            assert!(c.state.editor.undo());
            assert_eq!(c.state.editor.lines(), &[""]);
            assert_eq!(c.state.queued.front().unwrap(), "queued stays");
            assert!(c.state.entries.is_empty());
        }
    }

    #[test]
    fn double_escape_requires_two_unmodified_presses_within_the_window() {
        let mut c = ready();
        c.paste("keep".into());
        esc(&mut c);
        let until = c.state.escape_confirmation.unwrap();
        for kind in [KeyEventKind::Release, KeyEventKind::Repeat] {
            c.key(KeyEvent::new_with_kind(
                KeyCode::Esc,
                KeyModifiers::NONE,
                kind,
            ));
            assert_eq!(c.state.escape_confirmation, Some(until));
            assert_eq!(c.state.editor.lines(), &["keep"]);
        }
        c.state.escape_confirmation = Some(Instant::now() - Duration::from_millis(1));
        esc(&mut c);
        assert_eq!(c.state.editor.lines(), &["keep"]);
        assert!(c.state.escape_confirmation.unwrap() > until);
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::ALT));
        assert!(c.state.escape_confirmation.is_none());
        esc(&mut c);
        assert_eq!(c.state.editor.lines(), &["keep"]);
        c.state.escape_confirmation = Some(Instant::now() - Duration::from_millis(1));
        c.tick();
        assert!(c.state.escape_confirmation.is_none());
    }

    #[test]
    fn other_input_and_conversation_transitions_disarm_double_escape() {
        for intervention in 0..5 {
            let mut c = rewind_ready();
            c.paste("keep".into());
            esc(&mut c);
            match intervention {
                0 => {
                    c.key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
                }
                1 => c.paste(String::new()),
                2 => {
                    ctrl_c(&mut c);
                }
                3 => {
                    c.state.open(Overlay::Help);
                    esc(&mut c);
                }
                _ => c.reset_conversation(),
            }
            assert!(c.state.escape_confirmation.is_none());
            esc(&mut c);
            assert_eq!(c.state.editor.lines(), &["keep"]);
        }
    }

    #[test]
    fn escape_consumed_by_cancel_or_dismissal_cannot_start_a_destructive_gesture() {
        let mut c = rewind_ready();
        c.input("working".into());
        c.paste("draft".into());
        assert!(matches!(
            esc(&mut c).as_slice(),
            [Action::Notify {
                method: "session/cancel",
                ..
            }]
        ));
        assert!(esc(&mut c).is_empty());
        assert_eq!(c.state.editor.lines(), &["draft"]);
        reply(&mut c, Pending::Prompt, json!({"stopReason":"cancelled"}));
        esc(&mut c);
        assert_eq!(c.state.editor.lines(), &["draft"]);

        for dismissal in 0..3 {
            let mut c = rewind_ready();
            if dismissal == 0 {
                c.paste("/help".into());
            } else {
                c.paste("keep".into());
            }
            if dismissal == 1 {
                c.state.open(Overlay::Help);
            }
            if dismissal == 2 {
                c.state.search_highlight = "search".into();
            }
            esc(&mut c);
            assert!(c.state.escape_confirmation.is_none());
            esc(&mut c);
            assert!(!c.state.editor.lines().join("\n").is_empty());
            esc(&mut c);
            assert_eq!(c.state.editor.lines(), &[""]);
        }
    }

    #[test]
    fn empty_double_escape_lists_only_real_checkpoints_and_selects_latest() {
        let mut c = rewind_ready();
        esc(&mut c);
        let actions = esc(&mut c);
        assert!(
            matches!(actions.as_slice(), [Action::Request { method:"_aster/session/checkpoints", params, .. }] if params["sessionId"] == "old")
        );
        assert!(c.state.rewind_loading);
        assert_eq!(c.state.overlay, Some(Overlay::Rewind));
        let generation = c.state.conversation_generation;
        reply(
            &mut c,
            Pending::Checkpoints {
                source: "old".into(),
                generation,
            },
            json!({"checkpoints":[{"userMessageIndex":1,"prompt":"second"}]}),
        );
        assert_eq!(c.state.phase, Phase::Ready);
        assert_eq!(c.state.choices().len(), 1);
        assert_eq!(c.state.choices()[0].value, "1");
        assert_eq!(c.state.selected, 0);
        assert_eq!(c.state.entries.len(), 4);
        assert_eq!(c.state.editor.lines(), &[""]);
    }

    #[test]
    fn rewind_success_replays_before_committing_draft_and_never_auto_sends() {
        let mut c = rewind_ready();
        rewind_choices(&mut c);
        assert_eq!(c.state.choices()[0].value, "1");
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(
            matches!(actions.as_slice(), [Action::Request { method:"_aster/session/rewind", params, .. }] if params["userMessageIndex"] == 1 && params["sessionId"] == "old")
        );
        assert_eq!(c.state.entries.len(), 4);
        assert!(
            matches!(rewind_response(&mut c).as_slice(), [Action::Request { method:"session/load", params, .. }] if params["sessionId"] == "fork")
        );
        assert_eq!(c.state.session_id.as_deref(), Some("old"));
        assert_eq!(c.state.editor.lines(), &[""]);
        c.message(json!({"method":"session/update","params":{"sessionId":"fork","update":{"sessionUpdate":"user_message_chunk","content":{"text":"first"}}}}));
        let actions = reply(
            &mut c,
            Pending::Load {
                id: "fork".into(),
                previous: Some("old".into()),
            },
            json!({}),
        );
        assert!(
            matches!(actions.as_slice(), [Action::Request { method:"session/close", params, .. }] if params["sessionId"] == "old")
        );
        assert_eq!(c.state.editor.lines(), &["  second 中文", "next  ", ""]);
        assert_eq!(c.state.session_id.as_deref(), Some("fork"));
        assert!(reply(&mut c, Pending::OldClose, json!({})).is_empty());
        assert!(c.state.queued.is_empty());
        assert_eq!(
            c.state
                .entries
                .iter()
                .filter(|e| matches!(e, Entry::User(_)))
                .count(),
            1
        );
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(
            matches!(actions.as_slice(), [Action::Request { method:"session/prompt", params, .. }] if params["sessionId"] == "fork" && params["prompt"][0]["text"] == "  second 中文\nnext  \n")
        );
    }

    #[test]
    fn failed_rewind_or_load_keeps_the_original_conversation_and_draft() {
        for during_load in [false, true] {
            let mut c = rewind_ready();
            c.state.title = "original title".into();
            c.state.expanded_entries.insert(0);
            rewind_choices(&mut c);
            c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            let pending = if during_load {
                rewind_response(&mut c);
                c.message(json!({"method":"session/update","params":{"sessionId":"fork","update":{"sessionUpdate":"agent_message_chunk","content":{"text":"partial replay"}}}}));
                Pending::Load {
                    id: "fork".into(),
                    previous: Some("old".into()),
                }
            } else {
                Pending::Rewind {
                    source: "old".into(),
                    generation: c.state.conversation_generation,
                }
            };
            c.paste("new draft".into());
            c.state.editor.move_cursor(CursorMove::Back);
            let cursor = c.state.editor.cursor();
            c.registered(42, pending);
            assert!(c
                .message(json!({"id":42,"error":{"message":"could not restore"}}))
                .is_empty());
            assert_eq!(c.state.session_id.as_deref(), Some("old"));
            assert_eq!(c.state.title, "original title");
            assert_eq!(c.state.editor.lines(), &["new draft"]);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert_eq!(
                c.state
                    .entries
                    .iter()
                    .filter(|e| matches!(e, Entry::User(_)))
                    .count(),
                2
            );
            assert!(c
                .state
                .entries
                .iter()
                .any(|e| matches!(e, Entry::Assistant(s) if s == "second reply")));
            assert!(!c
                .state
                .entries
                .iter()
                .any(|e| matches!(e, Entry::Assistant(s) if s == "partial replay")));
            assert_eq!(c.state.expanded_entries, HashSet::from([0]));
            assert!(c.rewind_prompt.is_none());
        }
    }

    #[test]
    fn new_typing_during_rewind_keeps_both_drafts_with_the_recalled_one_paused() {
        let mut c = rewind_ready();
        rewind_choices(&mut c);
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        rewind_response(&mut c);
        c.paste("typed during restore".into());
        c.state.editor.move_cursor(CursorMove::Back);
        let cursor = c.state.editor.cursor();
        reply(
            &mut c,
            Pending::Load {
                id: "fork".into(),
                previous: Some("old".into()),
            },
            json!({}),
        );
        assert!(reply(&mut c, Pending::OldClose, json!({})).is_empty());
        assert_eq!(c.state.editor.lines(), &["typed during restore"]);
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(c.state.queued.front().unwrap(), "  second 中文\nnext  \n");
        assert!(c.state.queue_paused);
        assert!(c.state.editor.undo());
        assert_eq!(c.state.editor.lines(), &[""]);
    }

    #[test]
    fn rewind_blocks_pending_queue_and_never_overwrites_an_existing_draft() {
        let mut c = rewind_ready();
        c.state.queued.push_back("pending".into());
        assert!(c.input("/rewind".into()).is_empty());
        assert_eq!(c.state.overlay, Some(Overlay::Queue));
        assert!(c.state.queue_paused);
        c.state.queued.clear();
        rewind_choices(&mut c);
        // Closing the picker and typing while its request is in flight can
        // leave a new composer draft when the user opens the picker again.
        c.state.editor.insert_str("keep draft");
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.editor.lines(), &["keep draft"]);
        assert_eq!(c.state.session_id.as_deref(), Some("old"));
    }

    #[test]
    fn stale_rewind_responses_cannot_switch_another_conversation() {
        for pending in [
            Pending::Checkpoints {
                source: "old".into(),
                generation: 0,
            },
            Pending::Rewind {
                source: "old".into(),
                generation: 0,
            },
        ] {
            let mut c = rewind_ready();
            c.state.conversation_generation = 1;
            c.state.session_id = Some("another".into());
            c.state.editor.insert_str("kept");
            assert!(reply(
                &mut c,
                pending,
                json!({"sessionId":"fork","prompt":"stale","checkpoints":[]})
            )
            .is_empty());
            assert_eq!(c.state.session_id.as_deref(), Some("another"));
            assert_eq!(c.state.editor.lines(), &["kept"]);
            assert_eq!(c.state.entries.len(), 4);
        }
    }

    #[test]
    fn unsupported_backend_and_missing_checkpoints_do_not_fake_a_rewind() {
        let mut c = ready();
        c.state.session_id = Some("old".into());
        esc(&mut c);
        assert!(esc(&mut c).is_empty());
        assert_eq!(c.state.overlay, None);
        assert!(c.state.entries.iter().any(
            |e| matches!(e, Entry::Notice { error:true, text } if text.contains("does not support"))
        ));
        c.rewind_supported = true;
        c.input("/rewind".into());
        let generation = c.state.conversation_generation;
        reply(
            &mut c,
            Pending::Checkpoints {
                source: "old".into(),
                generation,
            },
            json!({"checkpoints":[]}),
        );
        assert_eq!(c.state.overlay, Some(Overlay::Rewind));
        assert!(c.state.choices().is_empty());
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
    }

    #[test]
    fn exit_while_fork_is_pending_closes_source_without_loading_fork() {
        let mut c = rewind_ready();
        rewind_choices(&mut c);
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(c.quit().is_empty());
        assert!(
            matches!(rewind_response(&mut c).as_slice(), [Action::Request { method:"session/close", params, pending:Pending::ExitClose }] if params["sessionId"] == "old")
        );
        assert_eq!(c.state.session_id.as_deref(), Some("old"));
    }

    fn ctrl_c(c: &mut Controller) -> Vec<Action> {
        c.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
    }

    #[test]
    fn keymap_double_ctrl_c_confirms_exit_without_clearing_drafts_or_panels() {
        for overlay in [
            None,
            Some(Overlay::Commands),
            Some(Overlay::Queue),
            Some(Overlay::Search),
        ] {
            let mut c = ready();
            c.state.editor.insert_str("draft 中文");
            c.state.editor.move_cursor(CursorMove::Back);
            c.state.overlay = overlay;
            c.state.query = "filter".into();
            let cursor = c.state.editor.cursor();
            let status = c.state.status.clone();
            assert!(ctrl_c(&mut c).is_empty());
            assert!(!c.exiting);
            assert_eq!(c.state.editor.lines(), &["draft 中文"]);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert_eq!(c.state.overlay, overlay);
            assert_eq!(c.state.query, "filter");
            assert_eq!(c.state.status, status);
            assert!(c.state.exit_confirmation.is_some());
            assert!(matches!(ctrl_c(&mut c).as_slice(), [Action::Exit]));
            assert!(c.state.exit_confirmation.is_none());
        }
    }

    #[test]
    fn keymap_first_ctrl_c_does_not_interrupt_work_or_startup() {
        let mut c = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            false,
        );
        assert!(ctrl_c(&mut c).is_empty());
        assert!(!c.exiting);
        assert_eq!(c.state.phase, Phase::Connecting);
        assert!(matches!(ctrl_c(&mut c).as_slice(), [Action::Exit]));

        let mut c = ready();
        c.state.session_id = Some("saved".into());
        c.input("work".into());
        c.input("queued".into());
        assert!(ctrl_c(&mut c).is_empty());
        assert!(!c.cancelled);
        assert!(!c.exiting);
        assert!(!c.state.queue_paused);
        assert_eq!(c.state.phase, Phase::Busy);
        assert!(matches!(
            ctrl_c(&mut c).as_slice(),
            [Action::Notify {
                method: "session/cancel",
                ..
            }]
        ));
        assert!(c.exiting);
        assert!(c.state.queue_paused);
    }

    #[test]
    fn keymap_exit_confirmation_expires_and_ignores_releases_and_repeats() {
        let mut c = ready();
        assert!(ctrl_c(&mut c).is_empty());
        let until = c.state.exit_confirmation.unwrap();
        for kind in [KeyEventKind::Release, KeyEventKind::Repeat] {
            let mut key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
            key.kind = kind;
            assert!(c.key(key).is_empty());
            assert!(!c.exiting);
            assert_eq!(c.state.exit_confirmation, Some(until));
        }
        c.state.exit_confirmation = Some(Instant::now() - Duration::from_millis(1));
        assert!(c.tick().is_empty());
        assert!(c.state.exit_confirmation.is_none());
        assert!(ctrl_c(&mut c).is_empty());
        // Expiry is checked at the key boundary, even before another timer tick.
        c.state.exit_confirmation = Some(Instant::now() - Duration::from_millis(1));
        assert!(ctrl_c(&mut c).is_empty());
        assert!(!c.exiting);
        assert!(matches!(ctrl_c(&mut c).as_slice(), [Action::Exit]));
    }

    #[test]
    fn keymap_typing_pasting_and_escape_cancel_the_exit_confirmation() {
        for input in 0..4 {
            let mut c = ready();
            ctrl_c(&mut c);
            match input {
                0 => {
                    c.key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
                }
                1 => c.paste("pasted".into()),
                2 => {
                    c.key(KeyEvent::new(
                        KeyCode::Char('c'),
                        KeyModifiers::CONTROL | KeyModifiers::ALT,
                    ));
                }
                _ => {
                    c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
                }
            }
            assert!(c.state.exit_confirmation.is_none());
            assert!(ctrl_c(&mut c).is_empty());
            assert!(!c.exiting);
        }
    }

    #[test]
    fn keymap_backend_completion_does_not_reset_exit_confirmation() {
        let mut c = ready();
        c.state.session_id = Some("saved".into());
        c.input("work".into());
        ctrl_c(&mut c);
        let confirmation = c.state.exit_confirmation;
        assert!(reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"})).is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert_eq!(c.state.exit_confirmation, confirmation);
        assert!(matches!(
            ctrl_c(&mut c).as_slice(),
            [Action::Request {
                pending: Pending::ExitClose,
                ..
            }]
        ));
    }

    #[test]
    fn keymap_escape_cancels_work_and_keeps_draft_and_queue() {
        let mut c = ready();
        c.state.session_id = Some("saved".into());
        c.input("work".into());
        c.input("queued".into());
        c.state.editor.insert_str("keep this draft");
        assert!(matches!(
            c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
                .as_slice(),
            [Action::Notify {
                method: "session/cancel",
                ..
            }]
        ));
        assert!(!c.exiting);
        assert_eq!(c.state.phase, Phase::Cancelling);
        assert_eq!(c.state.editor.lines(), &["keep this draft"]);
        assert!(c.state.exit_confirmation.is_none());
        assert!(c
            .key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL))
            .is_empty());
        assert!(c.state.queue_paused);
        assert_eq!(c.state.queued.len(), 1);
    }

    #[test]
    fn keymap_stalled_cancellation_only_exits_when_the_user_confirms() {
        let mut c = ready();
        c.state.session_id = Some("saved".into());
        c.input("work".into());
        c.cancel();
        c.deadline = Some(Instant::now() - Duration::from_millis(1));
        assert!(c.tick().is_empty());
        assert!(!c.exiting);
        assert!(c.fatal_error.is_none());
        assert!(ctrl_c(&mut c).is_empty());
        assert!(!c.exiting);
        assert!(matches!(ctrl_c(&mut c).as_slice(), [Action::Exit]));
        assert!(c.exiting);
    }

    #[test]
    fn cancellation_deadline_keeps_the_ui_open_and_the_queue_paused() {
        let mut c = ready();
        c.state.session_id = Some("saved-session".into());
        c.input("first".into());
        c.input("second".into());
        assert!(c.deadline.is_none());

        let before = Instant::now();
        assert!(matches!(
            c.cancel().as_slice(),
            [Action::Notify {
                method: "session/cancel",
                ..
            }]
        ));
        let deadline = c.deadline.expect("cancelling must have a time bound");
        assert!(deadline >= before + Duration::from_secs(20));
        assert!(deadline <= Instant::now() + Duration::from_secs(20));
        assert_eq!(c.state.phase, Phase::Cancelling);
        assert!(c.cancel().is_empty());
        assert_eq!(c.deadline, Some(deadline));
        assert!(c.tick().is_empty());
        c.state.editor.insert_str("Unsent draft");

        // Advance the existing deadline instead of sleeping for the grace period.
        c.deadline = Some(Instant::now() - Duration::from_millis(1));
        assert!(c.tick().is_empty());
        assert_eq!(c.state.phase, Phase::Cancelling);
        assert!(c.fatal_error.is_none());
        assert!(c.deadline.is_none());
        assert_eq!(c.state.editor.lines(), &["Unsent draft"]);
        assert!(c.state.queue_paused);
        assert_eq!(
            c.state.queued.front().map(MessageDraft::as_str),
            Some("second")
        );
        assert!(c.continue_queue().is_empty());
        assert!(c.state.entries.iter().any(|entry| matches!(entry,
            Entry::Notice { text, .. } if text.contains("still stopping") && text.contains("Ctrl+C"))));
        let notice_count = c.state.entries.len();
        assert!(c.tick().is_empty());
        assert_eq!(c.state.entries.len(), notice_count);
        assert!(reply(&mut c, Pending::Prompt, json!({"stopReason":"cancelled"})).is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.state.queue_paused);
        // A late terminal result restores normal session closing, not forced exit.
        assert!(matches!(
            c.quit().as_slice(),
            [Action::Request {
                pending: Pending::ExitClose,
                ..
            }]
        ));
    }

    #[test]
    fn warmup_recall_does_not_cancel_or_extend_background_preparation() {
        for remaining in [Duration::from_secs(2), Duration::from_secs(60)] {
            let mut c = ready();
            c.input("first".into());
            let existing = Instant::now() + remaining;
            c.deadline = Some(existing);
            assert!(c.cancel().is_empty());
            let deadline = c.deadline.expect("preparation must have a time bound");
            assert_eq!(deadline, existing);
            assert_eq!(c.state.editor.lines(), &["first"]);
            assert!(c.state.queued.is_empty());
            assert!(!c.cancelled);
            assert!(c.cancel().is_empty());
            assert_eq!(c.deadline, Some(deadline));
            c.deadline = Some(Instant::now() - Duration::from_millis(1));
            assert!(c.tick().is_empty());
            assert_eq!(c.state.phase, Phase::Preparing);
            assert!(c.fatal_error.is_none());
            assert!(c.deadline.is_none());
        }
    }

    #[test]
    fn cancellation_deadline_clears_on_terminal_result_and_does_not_affect_the_next_prompt() {
        for response in [
            json!({"id":1,"result":{"stopReason":"cancelled"}}),
            json!({"id":1,"result":{"stopReason":"end_turn"}}),
            json!({"id":1,"error":{"message":"provider stopped"}}),
        ] {
            let mut c = ready();
            c.state.session_id = Some("s".into());
            c.input("first".into());
            c.registered(1, Pending::Prompt);
            c.cancel();
            assert!(c.deadline.is_some());
            assert!(c.message(response).is_empty());
            assert_eq!(c.state.phase, Phase::Ready);
            assert!(c.deadline.is_none());
            assert!(c.tick().is_empty());

            assert!(matches!(
                c.input("next".into()).as_slice(),
                [Action::Request {
                    pending: Pending::Prompt,
                    ..
                }]
            ));
            assert!(!c.cancelled);
            assert!(c.deadline.is_none());
            assert!(matches!(c.cancel().as_slice(), [Action::Notify { .. }]));
            assert!(c.deadline.is_some());
        }
    }

    #[test]
    fn warmup_recalled_message_is_not_sent_when_preparation_finishes() {
        let mut c = ready();
        c.input("first".into());
        c.cancel();
        assert!(c.deadline.is_some());
        assert!(reply(&mut c, Pending::Prepare, json!({"sessionId":"s"})).is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.deadline.is_none());
        assert_eq!(c.state.session_id.as_deref(), Some("s"));
        assert!(matches!(
            c.input("next".into()).as_slice(),
            [Action::Request {
                pending: Pending::Prompt,
                ..
            }]
        ));
        assert!(!c.cancelled);
        assert!(c.deadline.is_none());
    }

    #[test]
    fn cancellation_deadline_is_not_extended_by_quit_or_repeated_cancellation() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("first".into());
        c.cancel();
        let earlier = Instant::now() + Duration::from_secs(2);
        c.deadline = Some(earlier);
        assert!(c.quit().is_empty());
        assert_eq!(c.deadline, Some(earlier));
        assert!(c.cancel().is_empty());
        assert!(c.quit().is_empty());
        assert_eq!(c.deadline, Some(earlier));
        c.deadline = Some(Instant::now() - Duration::from_millis(1));
        assert!(matches!(c.tick().as_slice(), [Action::Exit]));
        assert!(c
            .fatal_error
            .as_deref()
            .unwrap()
            .contains("Closing timed out"));
    }

    #[test]
    fn shutdown_waits_for_cancelled_prompt_then_closes_session() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("work".into());
        assert!(matches!(
            &c.quit()[0],
            Action::Notify {
                method: "session/cancel",
                ..
            }
        ));
        assert!(c.state.busy());
        assert!(c.quit().is_empty());
        let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"cancelled"}));
        assert!(matches!(
            &actions[0],
            Action::Request {
                pending: Pending::ExitClose,
                ..
            }
        ));
        assert!(matches!(
            reply(&mut c, Pending::ExitClose, json!({}))[0],
            Action::Exit
        ));
    }
    #[test]
    fn pagination_resolves_exact_id_over_prefix_and_updates_commands() {
        let mut c = ready();
        c.list(ListPurpose::Resume("s".into()));
        let actions = reply(
            &mut c,
            Pending::List(ListPurpose::Resume("s".into())),
            json!({"sessions":[{"sessionId":"s-more"}],"nextCursor":"2"}),
        );
        assert!(matches!(&actions[0],Action::Request{params,..} if params["cursor"]=="2"));
        let actions = reply(
            &mut c,
            Pending::List(ListPurpose::Resume("s".into())),
            json!({"sessions":[{"sessionId":"s"}]}),
        );
        assert!(matches!(&actions[0],Action::Request{pending:Pending::Load{id,..},..} if id=="s"));
        update(
            &mut c,
            json!({"sessionUpdate":"available_commands_update","availableCommands":[{"name":"compact","description":"Compact context"}]}),
        );
        assert!(c.state.commands.iter().any(|v| v.name == "/compact"));
    }
    #[test]
    fn multiline_paste_is_not_submitted_and_cjk_backspace_is_safe() {
        let mut c = ready();
        c.paste("你好\r\n世界\x1b[2J".into());
        assert!(c.state.entries.is_empty());
        assert_eq!(c.state.editor.lines(), &["你好", "世界"]);
        c.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["你好", "世"]);
    }

    #[test]
    fn input_regression_arrows_recall_sent_messages_and_restore_draft() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        for text in ["first message", "第二条消息"] {
            c.paste(text.into());
            c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        }
        c.paste("unsent draft".into());
        c.state.editor.move_cursor(CursorMove::Jump(0, 3));
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["第二条消息"]);
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["first message"]);
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["第二条消息"]);
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["unsent draft"]);
        assert_eq!(c.state.editor.cursor(), (0, 3));
        assert!(c.state.editor.undo());
        assert_eq!(c.state.editor.lines(), &[""]);
    }

    #[test]
    fn fresh_controller_recalls_submitted_input_without_resuming_session() {
        let workspace = tempfile::tempdir().unwrap();
        let store = InputHistory::for_workspace(workspace.path());
        {
            let mut first = ready();
            first.attach_history(store.clone());
            first.paste("  中文\nprevious prompt  \n".into());
            first.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            reply(&mut first, Pending::Prepare, json!({"sessionId":"s"}));
            first.paste("do not persist this draft".into());
        }
        let mut restarted = ready();
        restarted.attach_history(store);
        restarted.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            restarted.state.editor.lines(),
            &["  中文", "previous prompt  ", ""]
        );
        assert!(restarted.state.entries.is_empty());
        assert!(restarted.state.session_id.is_none());
    }

    #[test]
    fn legacy_input_history_bootstrap_is_persisted_without_replaying_conversation() {
        let workspace = tempfile::tempdir().unwrap();
        let store = InputHistory::for_workspace(workspace.path());
        let mut c = ready();
        c.attach_history(store.clone());
        let actions = c.list(ListPurpose::Browse);
        assert!(matches!(&actions[0], Action::Request { params, .. }
            if params["_meta"]["aster"]["includeInputHistory"] == true));
        reply(
            &mut c,
            Pending::List(ListPurpose::Browse),
            json!({"sessions":[],
            "_meta":{"aster":{"inputHistory":["old input", "  latest\ninput  "]}}}),
        );
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["  latest", "input  "]);
        assert!(c.state.entries.is_empty());
        assert!(c.state.session_id.is_none());
        let mut next = ready();
        next.attach_history(store);
        let actions = next.list(ListPurpose::Browse);
        assert!(
            matches!(&actions[0], Action::Request { params, .. } if params.get("_meta").is_none())
        );
        next.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(next.state.editor.lines(), &["  latest", "input  "]);
    }

    #[test]
    fn history_storage_error_does_not_block_submitting_or_memory_recall() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join(".aster"), "not a directory").unwrap();
        let mut c = ready();
        c.attach_history(InputHistory::for_workspace(workspace.path()));
        let actions = c.input("still send this".into());
        assert!(matches!(
            &actions[0],
            Action::Request {
                method: "session/new",
                ..
            }
        ));
        assert_eq!(
            prompt_text(&reply(&mut c, Pending::Prepare, json!({"sessionId":"s"}))),
            "still send this"
        );
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["still send this"]);
        assert!(c
            .state
            .entries
            .iter()
            .any(|entry| matches!(entry, Entry::Notice { error: true, .. })));
    }

    #[test]
    fn input_regression_ctrl_v_requests_system_clipboard() {
        for overlay in [None, Some(Overlay::Search), Some(Overlay::Commands)] {
            for key in [
                KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL),
                KeyEvent::new(
                    KeyCode::Char('V'),
                    KeyModifiers::CONTROL | KeyModifiers::SHIFT,
                ),
                KeyEvent::new(KeyCode::Insert, KeyModifiers::SHIFT),
            ] {
                let mut c = ready();
                c.paste("draft".into());
                c.state.overlay = overlay;
                c.state.query = "query".into();
                assert!(matches!(c.key(key).as_slice(), [Action::PasteClipboard]));
                assert_eq!(c.state.editor.lines(), &["draft"]);
                assert_eq!(c.state.query, "query");
                assert!(c.state.entries.is_empty());
                let release = KeyEvent {
                    kind: KeyEventKind::Release,
                    ..key
                };
                assert!(c.key(release).is_empty());
            }
        }
    }

    #[test]
    fn input_regression_history_yields_to_visual_navigation_and_selection() {
        use ratatui::{backend::TestBackend, Terminal};
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("old".into());
        reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        c.paste("abcdefghi\nsecond".into());
        let mut terminal = Terminal::new(TestBackend::new(4, 6)).unwrap();
        terminal
            .draw(|frame| {
                crate::composer::draw(
                    frame,
                    frame.area(),
                    &mut c.state.editor,
                    &mut c.state.composer,
                    true,
                );
            })
            .unwrap();
        c.state.editor.move_cursor(CursorMove::Jump(0, 6));
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.cursor(), (0, 2));
        assert_eq!(c.state.editor.lines(), &["abcdefghi", "second"]);
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::SHIFT));
        assert_eq!(c.state.editor.lines(), &["abcdefghi", "second"]);
        assert!(c.state.editor.is_selecting());
        c.state.editor.cancel_selection();
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["old"]);
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["abcdefghi", "second"]);
        assert_eq!(c.state.editor.cursor(), (0, 2));
    }

    #[test]
    fn input_regression_editing_recalled_history_keeps_the_new_draft() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("old".into());
        reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        c.paste(" edited".into());
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["old edited"]);
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["old"]);
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["old edited"]);
        assert_eq!(c.history, ["old"]);
    }

    #[test]
    fn input_regression_history_includes_resumed_messages() {
        let mut c = ready();
        c.resume("s".into());
        update(
            &mut c,
            json!({"sessionUpdate":"user_message_chunk", "content":{"type":"text", "text":"restored input"}}),
        );
        update(
            &mut c,
            json!({"sessionUpdate":"agent_message_chunk", "content":{"type":"text", "text":"answer"}}),
        );
        reply(
            &mut c,
            Pending::Load {
                id: "s".into(),
                previous: None,
            },
            json!({}),
        );
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["restored input"]);
    }

    #[test]
    fn input_regression_queued_submission_preserves_history_draft() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("first".into());
        c.input("queued".into());
        c.paste("original draft".into());
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["first"]);
        reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert!(c.state.busy());
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["original draft"]);
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["queued"]);
    }
    #[test]
    fn diff_shows_changed_lines_with_context() {
        let value = diff(
            &json!({"path":"main.py","oldText":"first\nold\nlast\n","newText":"first\nnew\nlast\n"}),
        );
        assert!(value.contains("-old\n+new"));
        assert!(value.contains(" first"));
        assert!(value.contains(" last"));
    }

    #[test]
    fn sending_pasted_code_preserves_indentation_and_trailing_newlines() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        let text = "    return value  \n\n";
        let actions = c.input(text.into());
        assert!(
            matches!(&actions[0], Action::Request {params,..} if params["prompt"][0]["text"]==text)
        );
    }

    #[test]
    fn removed_monitor_commands_are_rejected_without_sending_a_prompt() {
        for (text, command) in [
            ("/dev", "/dev"),
            ("/dev on", "/dev"),
            ("/dev off", "/dev"),
            ("/perf", "/perf"),
        ] {
            let mut c = ready();
            c.state.session_id = Some("s".into());
            c.paste(text.into());
            assert!(c.state.slash_choices().is_empty());
            assert!(c
                .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
                .is_empty());
            assert!(c.pending.is_empty());
            assert!(c.state.queued.is_empty());
            assert!(c.state.overlay.is_none());
            assert_eq!(c.state.phase, Phase::Ready);
            assert!(matches!(
                c.state.entries.as_slice(),
                [Entry::Notice { text, error: true }]
                    if text == &format!("Unknown command: {command}. Use /help or Ctrl+K.")
            ));
        }
    }

    #[test]
    fn slash_picker_transitions_to_arguments_without_submitting() {
        let mut c = ready();
        for ch in "/resume ".chars() {
            c.key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert!(c.state.overlay.is_none());
        assert_eq!(c.state.editor.lines(), &["/resume "]);
        assert!(c.pending.is_empty());
    }

    #[test]
    fn inline_slash_suggestions_keep_draft_and_tab_adds_arguments() {
        let mut c = ready();
        for ch in "/res".chars() {
            c.key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert_eq!(c.state.editor.lines(), &["/res"]);
        assert_eq!(
            c.state
                .slash_choices()
                .iter()
                .map(|choice| choice.value.as_str())
                .collect::<Vec<_>>(),
            ["/resume"]
        );
        c.key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["/resume "]);
        assert!(c.state.slash_choices().is_empty());
        assert!(c.pending.is_empty());
    }

    #[test]
    fn inline_slash_navigation_and_escape_preserve_typed_command() {
        let mut c = ready();
        c.state.commands.push(Command {
            name: "/reset".into(),
            description: "Reset this view".into(),
        });
        c.key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        assert_eq!(c.state.slash_choices().len(), c.state.commands.len());
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.slash_selected, 1);
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(c.state.slash_choices().is_empty());
        assert_eq!(c.state.editor.lines(), &["/"]);
        c.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE));
        assert!(!c.state.slash_choices().is_empty());
    }

    #[test]
    fn inline_slash_enter_executes_selected_command_without_opening_modal() {
        let mut c = ready();
        c.state.commands.push(Command {
            name: "/compact".into(),
            description: "Compact context".into(),
        });
        c.key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        for ch in "compact".chars() {
            c.key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(c.state.overlay.is_none());
        assert!(matches!(
            &actions[0],
            Action::Request {
                method: "session/new",
                pending: Pending::Prepare,
                ..
            }
        ));
        assert_eq!(
            c.state.queued.front().map(MessageDraft::as_str),
            Some("/compact")
        );
        assert!(c.state.editor.lines().is_empty() || c.state.editor.lines() == [""]);
    }

    #[test]
    fn ctrl_x_opens_help_without_affecting_a_slash_draft() {
        let mut c = ready();
        c.key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        c.key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
        assert_eq!(c.state.overlay, Some(Overlay::Help));
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["/"]);
    }

    #[test]
    fn busy_slash_selection_preserves_draft_and_escape_does_not_cancel_request() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("working".into());
        c.paste("/ne".into());
        assert!(c.choose_slash(0).is_empty());
        assert_eq!(c.state.editor.lines(), &["/ne"]);
        assert!(c.state.queued.is_empty());
        assert_eq!(c.state.phase, Phase::Busy);
        assert!(c
            .key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.phase, Phase::Busy);
        assert!(!c.cancelled);
        assert!(c.state.slash_dismissed);
        c.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(!c.state.slash_choices().is_empty());
    }

    #[test]
    fn slash_draft_is_independent_of_search_and_modal_command_filter() {
        let mut c = ready();
        c.paste("/res".into());
        let cursor = c.state.editor.cursor();
        c.key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        c.paste("Chinese 中文".into());
        assert!(c.state.slash_choices().is_empty());
        assert_eq!(c.state.query, "Chinese 中文");
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["/res"]);
        assert_eq!(c.state.editor.cursor(), cursor);
        assert_eq!(c.state.slash_choices()[0].value, "/resume");
        c.key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
        c.paste("tool".into());
        assert_eq!(c.state.choices()[0].value, "/tools");
        c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["/res"]);
    }

    #[test]
    fn command_palette_preserves_nonempty_draft_cursor_and_undo() {
        for (query, key) in [("resume", KeyCode::Char(' ')), ("custom", KeyCode::Enter)] {
            let mut c = ready();
            c.state.commands.push(Command {
                name: "/custom".into(),
                description: "Custom command".into(),
            });
            c.paste("important unsent draft\n中文".into());
            c.state.editor.move_cursor(CursorMove::Head);
            c.paste("edited ".into());
            let draft = c.state.editor.lines().to_vec();
            let cursor = c.state.editor.cursor();

            c.key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
            c.paste(query.into());
            assert!(c.key(KeyEvent::new(key, KeyModifiers::NONE)).is_empty());
            assert_eq!(c.state.editor.lines(), draft);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert_eq!(c.state.overlay, Some(Overlay::Commands));
            assert!(c.state.status.contains("current draft"));

            c.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
            assert_eq!(c.state.editor.lines(), draft);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert!(c.state.editor.undo());
            assert_eq!(c.state.editor.lines(), &["important unsent draft", "中文"]);
        }
    }

    #[test]
    fn command_palette_fills_empty_composer() {
        for (query, key) in [("resume", KeyCode::Char(' ')), ("custom", KeyCode::Enter)] {
            let mut c = ready();
            c.state.commands.push(Command {
                name: "/custom".into(),
                description: "Custom command".into(),
            });
            c.key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
            c.paste(query.into());
            assert!(c.key(KeyEvent::new(key, KeyModifiers::NONE)).is_empty());
            assert_eq!(c.state.editor.lines(), &[format!("/{query} ")]);
            assert!(c.state.overlay.is_none());
        }
    }

    #[test]
    fn slash_arguments_and_multiline_paste_are_sent_without_completion_rewriting() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.state.commands.push(Command {
            name: "/compact".into(),
            description: "Compact context".into(),
        });
        let text = "/compact 保留中文\n    code = 1\n";
        c.paste(text.into());
        assert!(c.state.slash_choices().is_empty());
        let actions = c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(prompt_text(&actions), text);
        assert!(c.state.queued.is_empty());
    }

    fn prompt_text(actions: &[Action]) -> &str {
        match &actions[0] {
            Action::Request {
                method: "session/prompt",
                params,
                ..
            } => params["prompt"][0]["text"].as_str().unwrap(),
            action => panic!("expected a prompt, got {action:?}"),
        }
    }

    #[test]
    fn enter_queues_multiline_messages_and_sends_them_serially_in_order() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        assert_eq!(prompt_text(&c.input("first".into())), "first");
        c.paste("    第二条\n\n".into());
        assert!(c
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_empty());
        assert_eq!(c.state.editor.lines(), &[""]);
        assert_eq!(c.state.queued.front().unwrap(), "    第二条\n\n");
        assert!(c.input("third".into()).is_empty());
        assert_eq!(c.state.entries.len(), 1);

        let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert_eq!(prompt_text(&actions), "    第二条\n\n");
        assert_eq!(c.state.queued.len(), 1);
        assert!(c.state.busy());
        let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert_eq!(prompt_text(&actions), "third");
        assert!(c.state.queued.is_empty());
        assert!(reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"})).is_empty());
        assert_eq!(c.state.phase, Phase::Ready);
        assert_eq!(c.history, ["first", "    第二条\n\n", "third"]);
    }

    #[test]
    fn queue_waits_for_new_session_and_the_original_prompt_to_finish() {
        let mut c = ready();
        c.input("first".into());
        assert!(c.input("second".into()).is_empty());
        let actions = reply(&mut c, Pending::Prepare, json!({"sessionId":"s"}));
        assert_eq!(prompt_text(&actions), "first");
        assert_eq!(c.state.queued.len(), 1);
        let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert_eq!(prompt_text(&actions), "second");
    }

    #[test]
    fn cancelled_prompt_does_not_drain_queue_even_if_backend_reports_end_turn() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("first".into());
        c.input("second".into());
        c.cancel();
        assert!(reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"})).is_empty());
        assert!(c.state.queue_paused);
        assert_eq!(c.state.queued.front().unwrap(), "second");
        let actions = c.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL));
        assert_eq!(prompt_text(&actions), "second");
    }

    #[test]
    fn prompt_error_and_nonstandard_stop_reasons_preserve_and_pause_queue() {
        for result in [
            json!({"error":{"message":"provider failed"}}),
            json!({"result":{"stopReason":"max_tokens"}}),
            json!({"result":{"stopReason":"refusal"}}),
        ] {
            let mut c = ready();
            c.state.session_id = Some("s".into());
            c.input("first".into());
            c.input("second".into());
            c.registered(2, Pending::Prompt);
            let mut message = result;
            message["id"] = json!(2);
            assert!(c.message(message).is_empty());
            assert!(c.state.queue_paused);
            assert_eq!(c.state.queued.front().unwrap(), "second");
            assert_eq!(c.state.phase, Phase::Ready);
        }
    }

    #[test]
    fn continue_during_a_prompt_never_submits_concurrently() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("first".into());
        c.input("second".into());
        c.state.queue_paused = true;
        assert!(c
            .key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL))
            .is_empty());
        assert!(!c.state.queue_paused);
        assert_eq!(c.state.queued.len(), 1);
        assert_eq!(
            prompt_text(&reply(
                &mut c,
                Pending::Prompt,
                json!({"stopReason":"end_turn"})
            )),
            "second"
        );
    }

    #[test]
    fn pending_messages_cannot_cross_a_new_or_resumed_conversation() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.state.queued.push_back("belongs to s".into());
        assert!(c.input("/new".into()).is_empty());
        assert_eq!(c.state.session_id.as_deref(), Some("s"));
        assert_eq!(c.state.overlay, Some(Overlay::Queue));
        assert!(c.resume("other".into()).is_empty());
        assert_eq!(c.state.session_id.as_deref(), Some("s"));
        c.key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert!(c.state.queued.is_empty());
        assert!(matches!(
            &c.resume("other".into())[0],
            Action::Request {
                pending: Pending::Load { .. },
                ..
            }
        ));
    }

    #[test]
    fn editing_filtered_queue_items_preserves_existing_drafts_and_pauses_the_rest() {
        let mut c = ready();
        c.state
            .queued
            .extend(["first".into(), "    第二条\nkeep whitespace\n".into()]);
        c.paste("unsent draft".into());
        c.state.open(Overlay::Queue);
        c.state.query = "whitespace".into();
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(c.state.queued.len(), 2);
        assert_eq!(c.state.editor.lines(), &["unsent draft"]);
        assert_eq!(c.state.overlay, Some(Overlay::Queue));
        c.state.editor = TextArea::default();
        c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            c.state.editor.lines(),
            &["    第二条", "keep whitespace", ""]
        );
        assert_eq!(c.state.queued.front().unwrap(), "first");
        assert!(c.state.queue_paused);
        assert!(c.state.overlay.is_none());
    }

    #[test]
    fn automatic_queue_submission_keeps_the_selected_message_for_editing_or_deleting() {
        for (query, messages, selected) in [
            ("", vec!["first", "target", "last"], 1),
            (
                "keep",
                vec!["keep first", "hidden", "keep target", "keep last"],
                1,
            ),
            (
                "keep",
                vec!["hidden", "keep first", "keep target", "keep last"],
                1,
            ),
            (
                "keep",
                vec![
                    "keep first",
                    "keep duplicate",
                    "keep duplicate",
                    "keep last",
                ],
                2,
            ),
        ] {
            for key in [KeyCode::Enter, KeyCode::Delete] {
                let mut c = ready();
                c.state.session_id = Some("s".into());
                c.input("running".into());
                for text in &messages {
                    c.input((*text).into());
                }
                c.state.open(Overlay::Queue);
                c.state.query = query.into();
                c.state.selected = selected;
                let original = c.state.choices()[selected].clone();
                let original_index: usize = original.value.parse().unwrap();

                let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
                assert_eq!(prompt_text(&actions), messages[0]);
                let current = c.state.choices()[c.state.selected].clone();
                assert_eq!(
                    current.label, original.label,
                    "query {query:?}, key {key:?}"
                );
                assert_eq!(current.value, (original_index - 1).to_string());

                let mut remaining = c.state.queued.clone();
                let selected_text = remaining.remove(original_index - 1).unwrap();
                c.key(KeyEvent::new(key, KeyModifiers::NONE));
                assert_eq!(c.state.queued, remaining);
                if key == KeyCode::Enter {
                    assert_eq!(c.state.editor.lines().join("\n"), selected_text);
                    assert!(c.state.queue_paused);
                } else {
                    assert_eq!(c.state.editor.lines(), &[""]);
                }
            }
        }
    }

    #[test]
    fn successful_resume_keeps_history_navigation_and_the_original_draft() {
        for persistent in [true, false] {
            let workspace = tempfile::tempdir().unwrap();
            let mut c = ready();
            if persistent {
                c.attach_history(InputHistory::for_workspace(workspace.path()));
            }
            c.state.session_id = Some("old".into());
            c.input("previous input".into());
            reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
            c.paste("original 中文".into());
            c.paste("\nsecond line".into());
            c.state.editor.move_cursor(CursorMove::Jump(0, 3));
            let draft = c.state.editor.lines().to_vec();
            let cursor = c.state.editor.cursor();
            c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
            assert_eq!(c.state.editor.lines(), &["previous input"]);

            c.resume("s".into());
            update(
                &mut c,
                json!({"sessionUpdate":"user_message_chunk", "content":{"text":"restored input"}}),
            );
            reply(
                &mut c,
                Pending::Load {
                    id: "s".into(),
                    previous: Some("old".into()),
                },
                json!({}),
            );
            reply(&mut c, Pending::OldClose, json!({}));
            if persistent {
                assert_eq!(c.state.editor.lines(), &["previous input"]);
                c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
            }
            assert_eq!(
                c.state.editor.lines(),
                draft,
                "persistent history: {persistent}"
            );
            assert_eq!(c.state.editor.cursor(), cursor);
            assert!(c.state.editor.undo());
            assert_eq!(c.state.editor.lines(), &["original 中文"]);
            if !persistent {
                c.key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
                assert_eq!(c.state.editor.lines(), &["restored input"]);
            }
        }
    }

    #[test]
    fn failed_resume_keeps_history_navigation_and_the_original_draft() {
        for persistent in [true, false] {
            let workspace = tempfile::tempdir().unwrap();
            let mut c = ready();
            if persistent {
                c.attach_history(InputHistory::for_workspace(workspace.path()));
            }
            c.state.session_id = Some("old".into());
            c.input("previous input".into());
            reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
            c.paste("original 中文".into());
            c.paste("\nsecond line".into());
            c.state.editor.move_cursor(CursorMove::Jump(0, 3));
            let draft = c.state.editor.lines().to_vec();
            let cursor = c.state.editor.cursor();
            c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));

            c.resume("s".into());
            c.registered(
                2,
                Pending::Load {
                    id: "s".into(),
                    previous: Some("old".into()),
                },
            );
            c.message(json!({"id":2,"error":{"message":"session is in use"}}));
            assert_eq!(c.state.session_id.as_deref(), Some("old"));
            c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
            assert_eq!(c.state.editor.lines(), draft);
            assert_eq!(c.state.editor.cursor(), cursor);
            assert!(c.state.editor.undo());
            assert_eq!(c.state.editor.lines(), &["original 中文"]);
        }
    }

    #[test]
    fn quitting_with_queue_closes_session_without_sending_more_messages() {
        let mut c = ready();
        c.state.session_id = Some("s".into());
        c.input("first".into());
        c.input("second".into());
        c.quit();
        let actions = reply(&mut c, Pending::Prompt, json!({"stopReason":"end_turn"}));
        assert!(matches!(
            &actions[0],
            Action::Request {
                pending: Pending::ExitClose,
                ..
            }
        ));
        assert_eq!(c.state.queued.front().unwrap(), "second");
        assert!(c.continue_queue().is_empty());
    }

    #[test]
    fn searching_folded_messages_and_tools_reveals_full_content() {
        for tool in [false, true] {
            let mut c = ready();
            if tool {
                update(
                    &mut c,
                    json!({"sessionUpdate":"tool_call","toolCallId":"t","title":"Read file","content":[{"type":"content","content":{"type":"text","text":"before\nneedle\nafter"}}]}),
                );
            } else {
                c.state
                    .entries
                    .push(Entry::User("before\nneedle\nafter".into()));
            }
            c.input("/search needle".into());
            c.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert!(c.state.expanded_entries.contains(&0));
            assert_eq!(c.state.focus_entry, Some(0));
            if tool {
                assert!(matches!(&c.state.entries[0], Entry::Tool(t) if t.expanded));
            }
        }
    }

    #[test]
    fn folding_details_tracks_full_tool_output_and_resets_scroll_baseline() {
        let mut c = ready();
        c.state.entries.push(Entry::User("long prompt".into()));
        update(
            &mut c,
            json!({"sessionUpdate":"tool_call","toolCallId":"t","title":"Read file"}),
        );
        c.state.rendered_rows = Some((100, 80));
        c.toggle_entry(0);
        assert_eq!(c.state.expanded_entries, HashSet::from([0]));
        assert!(c.state.rendered_rows.is_none());
        c.toggle_entry(1);
        assert!(matches!(&c.state.entries[1], Entry::Tool(t) if t.expanded));
        assert_eq!(c.state.expanded_entries, HashSet::from([0, 1]));
        c.toggle_tool("t");
        assert_eq!(c.state.expanded_entries, HashSet::from([0]));
        c.key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(c.state.expanded_entries.is_empty());
        c.key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert_eq!(c.state.expanded_entries, HashSet::from([0, 1]));
        assert!(matches!(&c.state.entries[1], Entry::Tool(t) if t.expanded));
        c.reset_conversation();
        assert!(c.state.expanded_entries.is_empty());
    }

    #[test]
    fn wrapped_composer_navigation_preserves_editing_and_slash_priority() {
        use ratatui::{backend::TestBackend, Terminal};
        let mut c = ready();
        c.paste("abcdefgh".into());
        c.state.editor.move_cursor(CursorMove::Jump(0, 6));
        let mut terminal = Terminal::new(TestBackend::new(4, 3)).unwrap();
        terminal
            .draw(|frame| {
                crate::composer::draw(
                    frame,
                    frame.area(),
                    &mut c.state.editor,
                    &mut c.state.composer,
                    true,
                );
            })
            .unwrap();
        c.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(c.state.editor.cursor(), (0, 2));
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT));
        assert_eq!(c.state.editor.selection_range(), Some(((0, 2), (0, 6))));
        c.key(KeyEvent::new(KeyCode::Char('!'), KeyModifiers::NONE));
        assert_eq!(c.state.editor.lines(), &["ab!gh"]);
        assert!(c.state.editor.undo());
        assert_eq!(c.state.editor.lines(), &["abgh"]);
        assert!(c.state.editor.undo());
        assert_eq!(c.state.editor.lines(), &["abcdefgh"]);

        c.set_draft("/");
        c.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(c.state.slash_selected, 1);
        assert_eq!(c.state.editor.lines(), &["/"]);
        let cursor = c.state.editor.cursor();
        c.state.scroll = 10;
        c.key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        assert_eq!(c.state.scroll, 0);
        assert_eq!(c.state.editor.cursor(), cursor);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::VecDeque;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use aster_tui::{
    clipboard,
    controller::{Action, Controller, RollbackJob, RollbackOutcome},
    state::{Overlay, State},
    terminal_input::{TerminalEvent, TerminalInput},
    transport::{BackendConfig, Event as BackendEvent, Transport},
    ui,
};
use clap::Parser;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, layout::Position, Terminal};

mod terminal_title;

#[derive(Parser)]
#[command(
    name = "aster",
    version,
    about = "Aster — an AI coding workspace for your terminal"
)]
struct Args {
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    #[arg(long, default_value = "python")]
    python: PathBuf,
    #[arg(long, env = "ASTER_MODEL")]
    model: String,
    #[arg(long, value_parser = ["completion", "responses"])]
    client_type: Option<String>,
    #[arg(long)]
    sandbox: bool,
    #[arg(long, conflicts_with = "continue_last")]
    resume: Option<String>,
    #[arg(long = "continue")]
    continue_last: bool,
}

struct Screen {
    title: Option<terminal_title::Title>,
}

impl Screen {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut guard = Self { title: None };
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
        guard.title = Some(terminal_title::Title::enter());
        Ok(guard)
    }
}

fn restore() {
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    // Windows mouse capture restores the mode saved after raw mode was enabled.
    // Restore cooked input last so exiting leaves the shell usable.
    let _ = disable_raw_mode();
}

impl Drop for Screen {
    fn drop(&mut self) {
        restore();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    if let Some(result) = clipboard::worker() {
        return result;
    }
    let args = Args::parse();
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        bail!("TUI requires an interactive terminal for both input and output.");
    }
    if args.model.trim().is_empty() {
        bail!("Model cannot be blank.");
    }
    if args.resume.as_ref().is_some_and(|s| s.trim().is_empty()) {
        bail!("Session ID cannot be blank.");
    }
    let workspace = args
        .workspace
        .canonicalize()
        .context("Workspace does not exist")?;
    if !workspace.is_dir() {
        bail!("Workspace must be a directory.");
    }
    let config = BackendConfig {
        python: args.python,
        workspace: workspace.clone(),
        model: args.model.clone(),
        client_type: args.client_type,
        sandbox: args.sandbox,
    };
    let workspace_text = workspace.to_string_lossy();
    let workspace_text = if let Some(unc) = workspace_text.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{unc}")
    } else {
        workspace_text
            .strip_prefix("\\\\?\\")
            .unwrap_or(&workspace_text)
            .to_owned()
    };
    let state = State::new(workspace_text, args.model, args.sandbox);
    let mut controller = Controller::new(state, args.resume, args.continue_last);
    controller.attach_history(aster_tui::input_history::InputHistory::for_workspace(
        &workspace,
    ));
    controller.attach_drafts(
        aster_tui::drafts::DraftStore::for_workspace(&workspace)
            .context("Could not initialize draft recovery")?,
    );
    let mut transport = match Transport::spawn(&config).await {
        Ok(transport) => Some(transport),
        Err(error) => {
            controller.failure(format!("Could not start the backend: {error:#}"));
            None
        }
    };
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore();
        hook(info);
    }));
    let result = run(&mut controller, &mut transport, &config).await;
    let result = if controller.persist_draft() {
        result
    } else {
        let message = "Drafts could not be saved before the terminal closed.";
        match result {
            Ok(()) => Err(anyhow::anyhow!(message)),
            Err(error) => Err(error.context(message)),
        }
    };
    if let Some(mut transport) = transport {
        let cleanup = transport.shutdown().await;
        finish_run(result, cleanup, controller.fatal_error, &transport)
    } else {
        result.and_then(|()| match controller.fatal_error {
            Some(error) => Err(anyhow::anyhow!(error)),
            None => Ok(()),
        })
    }
}

fn finish_run(
    result: Result<()>,
    cleanup: Result<()>,
    fatal_error: Option<String>,
    transport: &Transport,
) -> Result<()> {
    let outcome = result.and(cleanup).and_then(|()| match fatal_error {
        Some(error) => Err(anyhow::anyhow!(error)),
        None => Ok(()),
    });
    outcome.map_err(|error| {
        let diagnostics = transport.diagnostic_tail();
        if diagnostics.trim().is_empty() {
            error
        } else {
            anyhow::anyhow!("{error:#}\n\nBackend diagnostics (last 16 KiB):\n{diagnostics}")
        }
    })
}

#[derive(Clone, PartialEq, Eq)]
struct PasteTarget {
    conversation: u64,
    overlay: Option<Overlay>,
    draft: Vec<String>,
    cursor: (usize, usize),
    selection: Option<((usize, usize), (usize, usize))>,
    query: String,
}

impl PasteTarget {
    fn capture(controller: &Controller) -> Self {
        Self {
            conversation: controller.state.conversation_generation,
            overlay: controller.state.overlay,
            draft: controller.state.editor.lines().to_vec(),
            cursor: controller.state.editor.cursor(),
            selection: controller.state.editor.selection_range(),
            query: controller.state.query.clone(),
        }
    }
}

struct PendingPaste {
    target: PasteTarget,
    task: tokio::task::JoinHandle<Result<clipboard::ClipboardContent>>,
}

#[derive(Clone, PartialEq, Eq)]
struct CopyTarget {
    input: PasteTarget,
    draft_revision: u64,
}

impl CopyTarget {
    fn capture(controller: &Controller) -> Self {
        Self {
            input: PasteTarget::capture(controller),
            draft_revision: controller.state.draft_revision,
        }
    }
}

struct PendingCopy {
    target: CopyTarget,
    cut: bool,
    task: tokio::task::JoinHandle<Result<()>>,
}

impl Drop for PendingCopy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

enum ClipboardIntent {
    Copy {
        target: CopyTarget,
        text: String,
        cut: bool,
    },
    Paste(PasteTarget),
}

struct PendingChanges(
    tokio::task::JoinHandle<Result<aster_tui::workspace_changes::WorkspaceChanges>>,
);

struct PendingFiles {
    generation: u64,
    query: String,
    task: tokio::task::JoinHandle<Result<aster_tui::attachments::FileSearch>>,
}

struct PendingAttachment {
    generation: u64,
    task: tokio::task::JoinHandle<Result<aster_tui::attachments::Attachment>>,
}

struct PendingOriginalImage {
    generation: u64,
    name: String,
    task: tokio::task::JoinHandle<Result<aster_tui::original_image::OriginalImage>>,
}

struct PendingTerminalImage {
    key: aster_tui::terminal_image::RenderKey,
    task: tokio::task::JoinHandle<aster_tui::terminal_image::RenderResult>,
}

struct PendingRollback {
    generation: u64,
    task: tokio::task::JoinHandle<Result<RollbackOutcome>>,
}

#[derive(Default)]
struct IoTasks {
    paste: Option<PendingPaste>,
    copy: Option<PendingCopy>,
    clipboard_requests: VecDeque<ClipboardIntent>,
    changes: Option<PendingChanges>,
    files: Option<PendingFiles>,
    attachment: Option<PendingAttachment>,
    original_image: Option<PendingOriginalImage>,
    terminal_image: Option<PendingTerminalImage>,
    // Keep files available after the launcher returns, within a bounded session cache.
    original_images: VecDeque<aster_tui::original_image::OriginalImage>,
    restart: Option<tokio::task::JoinHandle<Result<Transport>>>,
    rollback: Option<PendingRollback>,
}

impl Drop for IoTasks {
    fn drop(&mut self) {
        if let Some(pending) = &self.files {
            pending.task.abort();
        }
        if let Some(pending) = &self.attachment {
            pending.task.abort();
        }
        if let Some(pending) = &self.original_image {
            pending.task.abort();
        }
        if let Some(pending) = &self.terminal_image {
            pending.task.abort();
        }
        if let Some(pending) = &self.restart {
            pending.abort();
        }
        if let Some(pending) = &self.rollback {
            pending.task.abort();
        }
    }
}

impl Drop for PendingChanges {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn mouse_paste(controller: &Controller, hits: &ui::HitAreas, mouse: MouseEvent) -> Option<Action> {
    let target = if controller.state.overlay.is_some() {
        hits.query?
    } else {
        hits.composer
    };
    (mouse.kind == MouseEventKind::Down(MouseButton::Right)
        && target.contains(Position::new(mouse.column, mouse.row)))
    .then_some(Action::PasteClipboard)
}

fn mouse_remove_attachment(
    controller: &mut Controller,
    hits: &ui::HitAreas,
    mouse: MouseEvent,
) -> bool {
    if controller.state.overlay.is_some() || mouse.kind != MouseEventKind::Down(MouseButton::Left) {
        return false;
    }
    hits.attachment_remove
        .iter()
        .find(|(rect, _)| rect.contains(Position::new(mouse.column, mouse.row)))
        .is_some_and(|(_, index)| controller.remove_attachment(*index))
}

fn mouse_attachment(controller: &mut Controller, hits: &ui::HitAreas, mouse: MouseEvent) -> bool {
    if controller.state.overlay.is_some() || mouse.kind != MouseEventKind::Down(MouseButton::Left) {
        return false;
    }
    if mouse_remove_attachment(controller, hits, mouse) {
        return true;
    }
    let position = Position::new(mouse.column, mouse.row);
    if let Some((_, index)) = hits
        .attachment_preview
        .iter()
        .find(|(rect, _)| rect.contains(position))
    {
        return controller.open_attachment(*index);
    }
    if hits
        .attachment_overflow
        .is_some_and(|rect| rect.contains(position))
    {
        return controller.open_attachment(0);
    }
    if hits
        .attachment_undo
        .is_some_and(|rect| rect.contains(position))
    {
        controller.undo_attachment();
        return true;
    }
    false
}

fn mouse_original_image(
    controller: &Controller,
    hits: &ui::HitAreas,
    mouse: MouseEvent,
) -> Vec<Action> {
    if mouse.kind == MouseEventKind::Down(MouseButton::Left)
        && hits
            .attachment_original
            .iter()
            .any(|rect| rect.contains(Position::new(mouse.column, mouse.row)))
    {
        controller.open_original_image()
    } else {
        vec![]
    }
}

fn original_image_result(
    controller: &mut Controller,
    tasks: &mut IoTasks,
    generation: u64,
    name: &str,
    result: Result<aster_tui::original_image::OriginalImage>,
) {
    let notice = match result {
        Ok(image) => {
            const MAX_OPEN_ORIGINALS: usize = 16;
            tasks.original_images.push_back(image);
            while tasks.original_images.len() > MAX_OPEN_ORIGINALS {
                tasks.original_images.pop_front();
            }
            (format!("Opened {name} in the image viewer."), false)
        }
        Err(error) => (format!("Could not open {name}: {error:#}"), true),
    };
    if generation == controller.state.conversation_generation {
        controller.state.original_image_notice =
            Some((aster_tui::state::clean(&notice.0), notice.1));
        if notice.1 {
            controller.state.notice(notice.0, true);
        }
    }
}

impl Drop for PendingPaste {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn paste_result(controller: &mut Controller, target: &PasteTarget, result: Result<String>) {
    if *target != PasteTarget::capture(controller) {
        controller.state.notice(
            "Paste cancelled because the input changed. Press Ctrl+V again.",
            false,
        );
        return;
    }
    match result {
        Ok(text) => controller.paste(text),
        Err(error) => {
            controller.state.notice(
                format!(
                    "Could not paste: {error}. Use your terminal's Paste command to try again."
                ),
                true,
            );
            if !controller.state.busy() {
                controller.state.status = "Clipboard unavailable".into();
            }
        }
    }
}

fn clipboard_result(
    controller: &mut Controller,
    target: &PasteTarget,
    result: Result<clipboard::ClipboardContent>,
) {
    match result {
        Ok(clipboard::ClipboardContent::Text(text)) => paste_result(controller, target, Ok(text)),
        Ok(clipboard::ClipboardContent::Image(image)) => {
            if *target != PasteTarget::capture(controller) || controller.state.overlay.is_some() {
                controller.state.notice("Image paste cancelled because the input changed. Paste into the message composer again.", false);
            } else {
                controller.add_attachment(target.conversation, Ok(image));
            }
        }
        Err(error) => paste_result(controller, target, Err(error)),
    }
}

fn copy_result(
    controller: &mut Controller,
    target: &CopyTarget,
    cut: bool,
    result: Result<()>,
) -> bool {
    match result {
        Ok(()) if cut => {
            if *target == CopyTarget::capture(controller) {
                return controller.cut_selection();
            } else {
                controller.state.notice(
                    "Copied to clipboard. The input changed, so the text was kept.",
                    false,
                );
            }
        }
        Ok(()) => {}
        Err(error) => controller.state.notice(
            format!("Could not copy: {error}. Your text was kept."),
            true,
        ),
    }
    false
}

fn queue_clipboard(controller: &mut Controller, tasks: &mut IoTasks, intent: ClipboardIntent) {
    if tasks.clipboard_requests.len() >= 8 {
        controller.state.notice(
            "Clipboard is busy. Wait for the pending operations and try again.",
            false,
        );
    } else {
        tasks.clipboard_requests.push_back(intent);
    }
}

fn next_clipboard(controller: &mut Controller, tasks: &mut IoTasks) -> Option<ClipboardIntent> {
    if tasks.copy.is_some() || tasks.paste.is_some() {
        return None;
    }
    while let Some(intent) = tasks.clipboard_requests.pop_front() {
        if let ClipboardIntent::Paste(target) = &intent {
            if *target != PasteTarget::capture(controller) {
                controller.state.notice(
                    "Paste cancelled because the input changed. Press Ctrl+V again.",
                    false,
                );
                continue;
            }
        }
        return Some(intent);
    }
    None
}

fn start_next_clipboard(controller: &mut Controller, tasks: &mut IoTasks) {
    match next_clipboard(controller, tasks) {
        Some(ClipboardIntent::Copy { target, text, cut }) => {
            tasks.copy = Some(PendingCopy {
                target,
                cut,
                task: tokio::spawn(async move { clipboard::write_text(&text).await }),
            });
        }
        Some(ClipboardIntent::Paste(target)) => {
            tasks.paste = Some(PendingPaste {
                target,
                task: tokio::spawn(clipboard::read()),
            });
        }
        None => {}
    }
}

fn start_terminal_image(state: &mut State, tasks: &mut IoTasks) {
    // Cancelling a blocking worker does not stop its encoder. Finish the one
    // running job, then take only the latest requested attachment/geometry.
    if tasks.terminal_image.is_none() {
        if let Some(request) = state.terminal_image.take_request() {
            tasks.terminal_image = Some(PendingTerminalImage {
                key: request.key(),
                task: tokio::task::spawn_blocking(move || request.render()),
            });
        }
    }
}

/// Clipboard edits precede already-requested pastes at the same input position.
/// Other edits keep their original target and are rejected if it is now stale.
fn rebase_clipboard_pastes(tasks: &mut IoTasks, before: &PasteTarget, after: &PasteTarget) {
    for intent in &mut tasks.clipboard_requests {
        if let ClipboardIntent::Paste(target) = intent {
            if target == before {
                *target = after.clone();
            }
        }
    }
}

fn complete_copy(
    controller: &mut Controller,
    tasks: &mut IoTasks,
    target: &CopyTarget,
    cut: bool,
    result: Result<()>,
) {
    let copied = result.is_ok();
    let before = PasteTarget::capture(controller);
    if copy_result(controller, target, cut, result) {
        rebase_clipboard_pastes(tasks, &before, &PasteTarget::capture(controller));
    }
    if !copied {
        // A paste requested after this copy must not use older clipboard data.
        let mut cancelled = false;
        while matches!(
            tasks.clipboard_requests.front(),
            Some(ClipboardIntent::Paste(_))
        ) {
            tasks.clipboard_requests.pop_front();
            cancelled = true;
        }
        if cancelled {
            controller
                .state
                .notice("Paste cancelled because the preceding copy failed.", false);
        }
    }
}

fn complete_paste(
    controller: &mut Controller,
    tasks: &mut IoTasks,
    target: &PasteTarget,
    result: Result<clipboard::ClipboardContent>,
) {
    let before = PasteTarget::capture(controller);
    clipboard_result(controller, target, result);
    let after = PasteTarget::capture(controller);
    if before != after {
        rebase_clipboard_pastes(tasks, &before, &after);
    }
}

fn cancel_pastes(
    controller: &mut Controller,
    tasks: &mut IoTasks,
    should_cancel: impl Fn(&PasteTarget) -> bool,
) -> bool {
    let active = tasks
        .paste
        .as_ref()
        .is_some_and(|pending| should_cancel(&pending.target));
    if active {
        tasks.paste.take();
    }
    let queued = tasks.clipboard_requests.len();
    tasks.clipboard_requests.retain(
        |intent| !matches!(intent, ClipboardIntent::Paste(target) if should_cancel(target)),
    );
    let cancelled = active || tasks.clipboard_requests.len() != queued;
    if cancelled {
        controller.state.notice(
            "Paste cancelled because the input changed. Press Ctrl+V again.",
            false,
        );
    }
    cancelled
}

fn cancel_changed_pastes(controller: &mut Controller, tasks: &mut IoTasks) -> bool {
    if tasks.paste.is_none()
        && !tasks
            .clipboard_requests
            .iter()
            .any(|intent| matches!(intent, ClipboardIntent::Paste(_)))
    {
        return false;
    }
    let current = PasteTarget::capture(controller);
    cancel_pastes(controller, tasks, |target| target != &current)
}

async fn dispatch(
    actions: Vec<Action>,
    controller: &mut Controller,
    transport: &mut Option<Transport>,
    tasks: &mut IoTasks,
    config: &BackendConfig,
) -> Result<bool> {
    for action in actions {
        match action {
            Action::Rollback { job, generation } => {
                if tasks.rollback.is_none() {
                    let workspace = PathBuf::from(&controller.state.workspace);
                    tasks.rollback = Some(PendingRollback {
                        generation,
                        task: tokio::spawn(async move {
                            match job {
                                RollbackJob::Prepare { file, hunk } => {
                                    aster_tui::rollback::prepare(&workspace, &file, hunk)
                                        .await
                                        .map(RollbackOutcome::Prepared)
                                }
                                RollbackJob::Apply(plan) => aster_tui::rollback::apply(plan)
                                    .await
                                    .map(RollbackOutcome::Applied),
                                RollbackJob::FindUndo => aster_tui::rollback::latest(&workspace)
                                    .await
                                    .map(RollbackOutcome::UndoReady),
                                RollbackJob::Undo(receipt) => {
                                    aster_tui::rollback::undo(&receipt).await?;
                                    Ok(RollbackOutcome::Undone(receipt.path))
                                }
                            }
                        }),
                    });
                }
            }
            Action::LoadWorkspaceChanges => {
                if tasks.changes.is_none() {
                    let workspace = PathBuf::from(&controller.state.workspace);
                    tasks.changes = Some(PendingChanges(tokio::spawn(async move {
                        aster_tui::workspace_changes::load(workspace).await
                    })));
                }
            }
            Action::SearchFiles { query, generation } => {
                if let Some(previous) = tasks.files.take() {
                    previous.task.abort();
                }
                let workspace = PathBuf::from(&controller.state.workspace);
                let search = query.clone();
                tasks.files = Some(PendingFiles {
                    query,
                    generation,
                    task: tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        aster_tui::attachments::search_files(&workspace, &search).await
                    }),
                });
            }
            Action::AttachFile { path, generation } => {
                if tasks.attachment.is_none() {
                    let workspace = PathBuf::from(&controller.state.workspace);
                    tasks.attachment = Some(PendingAttachment {
                        generation,
                        task: tokio::spawn(async move {
                            aster_tui::attachments::load_file(&workspace, &PathBuf::from(path))
                                .await
                        }),
                    });
                }
            }
            Action::OpenOriginalImage {
                attachment,
                generation,
            } => {
                if tasks.original_image.is_none()
                    && generation == controller.state.conversation_generation
                {
                    let name = aster_tui::state::clean(attachment.name());
                    controller.state.original_image_notice =
                        Some((format!("Opening {name}…"), false));
                    tasks.original_image = Some(PendingOriginalImage {
                        generation,
                        name,
                        task: tokio::spawn(aster_tui::original_image::open(attachment)),
                    });
                }
            }
            Action::Reconnect => {
                cancel_pastes(controller, tasks, |_| true);
                if tasks.restart.is_none() {
                    let previous = transport.take();
                    let config = config.clone();
                    tasks.restart = Some(tokio::spawn(async move {
                        if let Some(mut previous) = previous {
                            previous.shutdown().await?;
                        }
                        Transport::spawn(&config).await
                    }));
                }
            }
            Action::Request {
                method,
                params,
                pending,
            } => {
                // A load keeps the previous session ID until its response, so
                // cancel at the request boundary as well as checking snapshots.
                if matches!(method, "session/load" | "session/close") {
                    cancel_pastes(controller, tasks, |_| true);
                }
                controller.persist_draft();
                let Some(transport) = transport.as_mut() else {
                    controller.failure("The backend is disconnected. Use /reconnect.");
                    return Ok(false);
                };
                match transport.request(method, params).await {
                    Ok(id) => controller.registered(id, pending),
                    Err(error) => {
                        controller.failure(format!("Could not send request: {error}"));
                        return Ok(false);
                    }
                }
            }
            Action::Notify { method, params } => {
                if let Some(transport) = transport.as_mut() {
                    if let Err(error) = transport.notify(method, params).await {
                        controller.failure(format!("Could not notify backend: {error}"));
                    }
                }
            }
            Action::Reply { id, result } => {
                if let Some(transport) = transport.as_mut() {
                    if let Err(error) = transport.reply(id, result).await {
                        controller.failure(format!("Could not send approval: {error}"));
                    }
                }
            }
            Action::PasteClipboard => {
                let target = PasteTarget::capture(controller);
                queue_clipboard(controller, tasks, ClipboardIntent::Paste(target));
            }
            Action::CopySelection { text, cut } => {
                let target = CopyTarget::capture(controller);
                queue_clipboard(
                    controller,
                    tasks,
                    ClipboardIntent::Copy { target, text, cut },
                );
            }
            Action::Exit => {
                return Ok(controller.save_for_exit());
            }
        }
    }
    start_next_clipboard(controller, tasks);
    Ok(false)
}

async fn run(
    controller: &mut Controller,
    transport: &mut Option<Transport>,
    config: &BackendConfig,
) -> Result<()> {
    let _screen = Screen::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;
    let mut input = TerminalInput::new()?;
    let mut input_closed = false;
    let mut ticker = tokio::time::interval(Duration::from_millis(60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut queue = if transport.is_some() {
        VecDeque::from([controller.initialize()])
    } else {
        VecDeque::new()
    };
    let mut hits = ui::HitAreas::default();
    let mut dirty = true;
    let mut last_draw = Instant::now();
    let mut tasks = IoTasks::default();
    let mut last_draft_save = Instant::now();
    loop {
        dirty |= cancel_changed_pastes(controller, &mut tasks);
        dirty |= !queue.is_empty();
        if dispatch(
            queue.drain(..).collect(),
            controller,
            transport,
            &mut tasks,
            config,
        )
        .await?
        {
            break;
        }
        if input_closed && !controller.is_exiting() {
            bail!("Terminal input closed while drafts could not be saved.");
        }
        // A noisy subprocess must not starve keyboard input and redraws.
        for _ in 0..256 {
            let Some(event) = transport.as_mut().and_then(Transport::try_recv) else {
                break;
            };
            dirty = true;
            match event {
                BackendEvent::Message(value) => {
                    if value.get("id").is_some() && value.get("method").is_some() {
                        if value["method"] == "session/request_permission" {
                            queue.extend(controller.permission_request(&value));
                        } else {
                            transport
                                .as_mut()
                                .expect("backend event has a transport")
                                .reject(
                                    value["id"].clone(),
                                    -32601,
                                    "Client method is not supported",
                                )
                                .await?;
                        }
                    } else {
                        queue.extend(controller.message(value));
                    }
                }
                BackendEvent::Diagnostic(text) => controller.diagnostic(text),
                BackendEvent::Closed(reason) => {
                    queue.extend(controller.failure(reason));
                    break;
                }
            }
        }
        if !queue.is_empty() {
            continue;
        }
        if dirty {
            hits = ui::render(&mut terminal, &mut controller.state)?;
            dirty = false;
            last_draw = Instant::now();
        }
        start_terminal_image(&mut controller.state, &mut tasks);
        tokio::select! {
            result = async { (&mut tasks.terminal_image.as_mut().expect("pending terminal image").task).await }, if tasks.terminal_image.is_some() => {
                let pending = tasks.terminal_image.take().expect("pending terminal image");
                let completed = result.unwrap_or_else(|_| pending.key.failed("Image preview worker stopped.".into()));
                dirty |= controller.state.terminal_image.complete(completed);
            },
            result = async { (&mut tasks.rollback.as_mut().expect("pending rollback").task).await }, if tasks.rollback.is_some() => {
                let pending = tasks.rollback.take().expect("pending rollback");
                queue.extend(controller.rollback_result(pending.generation, result.unwrap_or_else(|_| Err(anyhow::anyhow!("File operation worker stopped")))));
                dirty = true;
            },
            result = async { tasks.restart.as_mut().expect("pending restart").await }, if tasks.restart.is_some() => {
                tasks.restart.take();
                match result {
                    Ok(Ok(restarted)) => {
                        *transport = Some(restarted);
                        queue.extend(controller.reconnected());
                    }
                    Ok(Err(error)) => { queue.extend(controller.failure(format!("Could not reconnect: {error}"))); }
                    Err(error) => { queue.extend(controller.failure(format!("Reconnect stopped: {error}"))); }
                }
                dirty = true;
            },
            result = async { (&mut tasks.files.as_mut().expect("pending files").task).await }, if tasks.files.is_some() => {
                let pending = tasks.files.take().expect("pending files");
                controller.files_result(pending.generation, &pending.query, result.unwrap_or_else(|_| Err(anyhow::anyhow!("File search stopped"))));
                dirty = true;
            },
            result = async { (&mut tasks.attachment.as_mut().expect("pending attachment").task).await }, if tasks.attachment.is_some() => {
                let pending = tasks.attachment.take().expect("pending attachment");
                controller.attachment_result(pending.generation, result.unwrap_or_else(|_| Err(anyhow::anyhow!("File reader stopped"))));
                controller.persist_draft();
                dirty = true;
            },
            result = async { (&mut tasks.original_image.as_mut().expect("pending original image").task).await }, if tasks.original_image.is_some() => {
                let pending = tasks.original_image.take().expect("pending original image");
                original_image_result(controller, &mut tasks, pending.generation, &pending.name, result.unwrap_or_else(|_| Err(anyhow::anyhow!("Image viewer worker stopped"))));
                dirty = true;
            },
            result = async { (&mut tasks.changes.as_mut().expect("pending changes").0).await }, if tasks.changes.is_some() => {
                tasks.changes.take();
                controller.changes_result(result.unwrap_or_else(|_| Err(anyhow::anyhow!("Git reader stopped"))));
                dirty = true;
            },
            result = async { (&mut tasks.paste.as_mut().expect("pending paste").task).await }, if tasks.paste.is_some() => {
                let pending = tasks.paste.take().expect("pending paste");
                complete_paste(controller, &mut tasks, &pending.target, result.unwrap_or_else(|_| Err(anyhow::anyhow!("Clipboard reader stopped"))));
                start_next_clipboard(controller, &mut tasks);
                if controller.state.overlay == Some(Overlay::Files) { queue.extend(controller.search_files()); }
                dirty = true;
            },
            result = async { (&mut tasks.copy.as_mut().expect("pending copy").task).await }, if tasks.copy.is_some() => {
                let pending = tasks.copy.take().expect("pending copy");
                complete_copy(controller, &mut tasks, &pending.target, pending.cut, result.unwrap_or_else(|_| Err(anyhow::anyhow!("Clipboard writer stopped"))));
                start_next_clipboard(controller, &mut tasks);
                dirty = true;
            },
            _ = ticker.tick() => {
                if last_draft_save.elapsed() >= Duration::from_millis(300) {
                    let previous_entries = controller.state.entries.len();
                    controller.autosave_draft();
                    dirty |= controller.state.entries.len() != previous_entries;
                    last_draft_save = Instant::now();
                }
                let exit_confirmation = controller.state.exit_confirmation;
                let escape_confirmation = controller.state.escape_confirmation;
                queue.extend(controller.tick());
                if exit_confirmation != controller.state.exit_confirmation
                    || escape_confirmation != controller.state.escape_confirmation
                    || (controller.state.busy() && last_draw.elapsed() >= Duration::from_millis(100)) {
                    dirty = true;
                }
            },
            event = input.next(), if !input_closed => {
                match event {
                    Some(Ok(TerminalEvent::Graphics(capabilities))) => {
                        controller.state.terminal_graphics = capabilities;
                        dirty = true;
                    },
                    Some(Ok(TerminalEvent::Input(Event::Key(key)))) => { dirty = true; queue.extend(controller.key(key)); },
                    Some(Ok(TerminalEvent::Input(Event::Paste(text)))) => {
                        dirty = true; controller.paste(text);
                        if controller.state.overlay == Some(Overlay::Files) { queue.extend(controller.search_files()); }
                    },
                    Some(Ok(TerminalEvent::Input(Event::Mouse(mouse)))) => {
                        if mouse.kind != MouseEventKind::Moved {
                            let exit = controller.state.exit_confirmation.take().is_some();
                            let escape = controller.state.escape_confirmation.take().is_some();
                            dirty |= exit || escape;
                        }
                        let position = Position::new(mouse.column,mouse.row);
                        // A captured right-button press pastes once into the visible input.
                        // Reuse the asynchronous clipboard path, preserving cursor/selection.
                        queue.extend(mouse_paste(controller, &hits, mouse));
                        let composer_handled = controller.composer_mouse(hits.composer_text, mouse);
                        dirty |= composer_handled;
                        match mouse.kind {
                            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                                dirty = true;
                                let up = mouse.kind == MouseEventKind::ScrollUp;
                                if let Some(overlay) = controller.state.overlay {
                                    let list = matches!(overlay, Overlay::Commands | Overlay::Sessions | Overlay::Drafts | Overlay::Search | Overlay::Queue | Overlay::Rewind | Overlay::Files | Overlay::Models | Overlay::Modes | Overlay::Permissions)
                                        || (matches!(overlay, Overlay::Tools | Overlay::Changes | Overlay::Attachments | Overlay::Tasks | Overlay::Approval | Overlay::Rollback | Overlay::RollbackConfirm | Overlay::UndoConfirm) && hits.choice_rows.iter().any(|(rect, _)| rect.contains(position)));
                                    if list {
                                        let max = controller.state.choices().len().saturating_sub(1);
                                        controller.state.selected = if up { controller.state.selected.saturating_sub(1) }
                                            else { controller.state.selected.saturating_add(1).min(max) };
                                        controller.state.detail_scroll = 0;
                                    } else {
                                        controller.state.detail_scroll = if up { controller.state.detail_scroll.saturating_sub(3) }
                                            else { controller.state.detail_scroll.saturating_add(3) };
                                    }
                                } else if hits.slash_rows.iter().any(|(rect, _)| rect.contains(position)) {
                                    let max = controller.state.slash_choices().len().saturating_sub(1);
                                    controller.state.slash_selected = if up {
                                        controller.state.slash_selected.saturating_sub(1)
                                    } else { controller.state.slash_selected.saturating_add(1).min(max) };
                                } else if hits.transcript.contains(position) {
                                    controller.state.scroll = if up { controller.state.scroll.saturating_add(3) }
                                        else { controller.state.scroll.saturating_sub(3) };
                                }
                            }
                            MouseEventKind::Down(MouseButton::Left) if controller.state.overlay.is_some() => {
                                let actions = mouse_original_image(controller, &hits, mouse);
                                if !actions.is_empty() {
                                    queue.extend(actions);
                                    dirty = true;
                                } else if let Some((_, index)) = hits.choice_rows.iter().find(|(rect, _)| rect.contains(position)) {
                                    dirty = true;
                                    controller.state.selected = *index;
                                    controller.state.detail_scroll = 0;
                                }
                            }
                            MouseEventKind::Down(MouseButton::Left) if composer_handled => {}
                            MouseEventKind::Down(MouseButton::Left) if mouse_attachment(controller, &hits, mouse) => {
                                dirty = true;
                            }
                            MouseEventKind::Down(MouseButton::Left) if controller.state.overlay.is_none() => {
                                dirty = true;
                                if let Some((_,index)) = hits.slash_rows.iter().find(|(rect,_)|rect.contains(position)) {
                                    queue.extend(controller.choose_slash(*index));
                                } else if let Some((_,command)) = hits.welcome_rows.iter().find(|(rect,_)|rect.contains(position)) {
                                    if *command == "/" {
                                        controller.state.open(Overlay::Commands);
                                    } else {
                                        queue.extend(controller.input((*command).into()));
                                    }
                                } else if let Some((_,index)) = hits.detail_rows.iter().find(|(rect,_)|rect.contains(position)) {
                                    controller.toggle_entry(*index);
                                } else if let Some((_,id)) = hits.tool_rows.iter().find(|(rect,_)|rect.contains(position)) {
                                    controller.toggle_tool(id);
                                } else if let Some((_,id)) = hits.session_rows.iter().find(|(rect,_)|rect.contains(position)) {
                                    queue.extend(controller.resume(id.clone()));
                                }
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(TerminalEvent::Input(Event::Resize(..)))) => {
                        input.refresh_graphics()?;
                        dirty = true;
                    },
                    Some(Ok(_)) => {},
                    Some(Err(error)) => return Err(error.into()),
                    None => {
                        input_closed = true;
                        queue.extend(controller.quit());
                        if !controller.is_exiting() {
                            bail!("Terminal input closed while drafts could not be saved.");
                        }
                    },
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft_controller(workspace: &std::path::Path) -> Controller {
        let mut controller = Controller::new(
            State::new(
                workspace.to_string_lossy().into_owned(),
                "test".into(),
                false,
            ),
            None,
            false,
        );
        controller.attach_drafts(aster_tui::drafts::DraftStore::for_workspace(workspace).unwrap());
        controller
    }

    fn recover_draft_controller(workspace: &std::path::Path) -> Controller {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut controller = draft_controller(workspace);
        assert_eq!(controller.state.editor.lines(), &[""]);
        controller.state.phase = aster_tui::state::Phase::Ready;
        assert!(controller.input("/drafts".into()).is_empty());
        assert_eq!(controller.state.overlay, Some(Overlay::Drafts));
        assert!(!controller.state.saved_drafts.is_empty());
        let mut actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(actions.len(), 1);
        let Action::Request { method, .. } = actions.pop().unwrap() else {
            panic!("expected explicit recovery to prepare a session");
        };
        assert_eq!(method, "session/new");
        // Input is recovered while backend preparation is still in progress.
        assert_eq!(controller.state.phase, aster_tui::state::Phase::Preparing);
        assert!(controller.state.queue_paused);
        controller
    }

    fn saved_draft_path(workspace: &std::path::Path) -> PathBuf {
        let paths: Vec<_> = std::fs::read_dir(workspace.join(".aster/tui-drafts"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect();
        assert_eq!(paths.len(), 1);
        paths.into_iter().next().unwrap()
    }

    async fn dispatch_exit(actions: Vec<Action>, controller: &mut Controller) -> bool {
        let config = BackendConfig {
            python: PathBuf::from("unused"),
            workspace: PathBuf::from(&controller.state.workspace),
            model: "test".into(),
            client_type: None,
            sandbox: false,
        };
        dispatch(
            actions,
            controller,
            &mut None,
            &mut IoTasks::default(),
            &config,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn draft_conflict_keeps_exit_open_and_input_editable() {
        let root = tempfile::tempdir().unwrap();
        let mut seed = draft_controller(root.path());
        seed.paste("shared saved input".into());
        assert!(seed.persist_draft());
        let mut first = recover_draft_controller(root.path());
        let mut stale = recover_draft_controller(root.path());
        first.state.editor.select_all();
        first.paste("other terminal draft".into());
        assert!(first.persist_draft());
        stale.state.editor.select_all();
        stale.paste("local unsaved draft".into());
        assert!(!stale.persist_draft());

        let actions = stale.quit();
        assert!(!dispatch_exit(actions, &mut stale).await);
        assert_eq!(stale.state.editor.lines(), &["local unsaved draft"]);
        assert!(stale.state.queue_paused);
        assert!(stale.input("another queued message".into()).is_empty());
        assert_eq!(
            stale.state.queued.back().unwrap().text,
            "another queued message"
        );
        let restored = recover_draft_controller(root.path());
        assert_eq!(restored.state.editor.lines(), &["other terminal draft"]);
    }

    #[tokio::test]
    async fn unreadable_old_draft_requires_a_valid_allocator_to_save_new_input() {
        for case in [
            "empty",
            "whitespace",
            "text",
            "attachment",
            "queue",
            "recovery",
        ] {
            for has_sequence in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let directory = root.path().join(".aster/tui-drafts");
                std::fs::create_dir_all(&directory).unwrap();
                let saved = directory.join("new.json");
                std::fs::write(&saved, b"unreadable recovery record").unwrap();
                if has_sequence {
                    std::fs::write(directory.join(".sequence"), b"42").unwrap();
                }
                let mut controller = draft_controller(root.path());
                match case {
                    "empty" => {}
                    "whitespace" => controller.paste(" ".into()),
                    "text" => controller.paste("local input".into()),
                    "attachment" => controller.state.attachments.push(
                        aster_tui::attachments::Attachment::Text {
                            name: "context.txt".into(),
                            uri: "file:///context.txt".into(),
                            text: "captured context".into(),
                        },
                    ),
                    "queue" => controller.state.queued.push_back("queued input".into()),
                    "recovery" => controller.state.recovery = Some("interrupted input".into()),
                    _ => unreachable!(),
                }
                let composer = controller.state.editor.lines().to_vec();
                let attachments = controller.state.attachments.clone();
                let queued = controller.state.queued.clone();
                let recovery = controller.state.recovery.clone();

                let actions = controller.quit();
                let exited = dispatch_exit(actions, &mut controller).await;
                assert_eq!(
                    exited,
                    case == "empty" || has_sequence,
                    "{case}, allocator: {has_sequence}"
                );
                assert_eq!(controller.state.editor.lines(), composer);
                assert_eq!(controller.state.attachments, attachments);
                assert_eq!(controller.state.queued, queued);
                assert_eq!(controller.state.recovery, recovery);
                assert_eq!(
                    std::fs::read(&saved).unwrap(),
                    b"unreadable recovery record"
                );
                if has_sequence && case != "empty" {
                    let restored = recover_draft_controller(root.path());
                    assert_eq!(restored.state.editor.lines(), composer);
                    assert_eq!(restored.state.attachments, attachments);
                    assert_eq!(restored.state.queued, queued);
                    assert_eq!(restored.state.recovery, recovery);
                }
            }
        }
    }

    #[tokio::test]
    async fn unreadable_current_draft_keeps_exit_open_and_the_final_edit() {
        let root = tempfile::tempdir().unwrap();
        let mut controller = draft_controller(root.path());
        controller.paste("saved input".into());
        assert!(controller.persist_draft());
        let saved = saved_draft_path(root.path());
        std::fs::write(&saved, b"unreadable current record").unwrap();
        controller.paste(" and the final edit".into());

        let actions = controller.quit();
        assert!(!dispatch_exit(actions, &mut controller).await);
        assert_eq!(
            controller.state.editor.lines(),
            &["saved input and the final edit"]
        );
        assert_eq!(std::fs::read(&saved).unwrap(), b"unreadable current record");
    }

    #[tokio::test]
    async fn draft_save_failure_can_retry_exit_without_losing_the_final_edit() {
        let root = tempfile::tempdir().unwrap();
        let mut controller = draft_controller(root.path());
        controller.paste("saved input".into());
        assert!(controller.persist_draft());
        let blocker = saved_draft_path(root.path()).with_extension("tmp");
        std::fs::create_dir(&blocker).unwrap();
        controller.paste(" and the final edit".into());

        let actions = controller.quit();
        assert!(!dispatch_exit(actions, &mut controller).await);
        assert_eq!(
            controller.state.editor.lines(),
            &["saved input and the final edit"]
        );
        std::fs::remove_dir(&blocker).unwrap();
        let actions = controller.quit();
        assert!(dispatch_exit(actions, &mut controller).await);
        let restored = recover_draft_controller(root.path());
        assert_eq!(
            restored.state.editor.lines(),
            &["saved input and the final edit"]
        );
    }

    #[tokio::test]
    async fn late_draft_failure_after_backend_close_keeps_reconnect_available() {
        let root = tempfile::tempdir().unwrap();
        let mut controller = draft_controller(root.path());
        controller.state.phase = aster_tui::state::Phase::Ready;
        controller.state.session_id = Some("closed-session".into());
        controller.paste("saved input".into());
        assert!(controller.persist_draft());
        let mut actions = controller.quit();
        let Some(Action::Request { pending, .. }) = actions.pop() else {
            panic!("expected session/close");
        };
        controller.registered(1, pending);
        let actions = controller.message(serde_json::json!({"id":1,"result":{}}));
        assert!(matches!(actions.as_slice(), [Action::Exit]));
        let blocker = saved_draft_path(root.path()).with_extension("tmp");
        std::fs::create_dir(&blocker).unwrap();
        controller.paste(" edited while closing".into());

        assert!(!dispatch_exit(actions, &mut controller).await);
        assert_eq!(
            controller.state.editor.lines(),
            &["saved input edited while closing"]
        );
        assert!(matches!(
            controller.reconnect().as_slice(),
            [Action::Reconnect]
        ));
        std::fs::remove_dir(&blocker).unwrap();
        assert!(controller.persist_draft());
        let restored = recover_draft_controller(root.path());
        assert_eq!(
            restored.state.editor.lines(),
            &["saved input edited while closing"]
        );
    }

    fn paste_controller() -> Controller {
        let mut controller = Controller::new(
            State::new("C:/work".into(), "test".into(), false),
            None,
            false,
        );
        controller.state.phase = aster_tui::state::Phase::Ready;
        controller
    }

    fn rendered_hits(controller: &mut Controller) -> ui::HitAreas {
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(110, 32)).unwrap();
        let mut hits = ui::HitAreas::default();
        terminal
            .draw(|frame| hits = ui::draw(frame, &mut controller.state))
            .unwrap();
        hits
    }

    fn right_mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: crossterm::event::KeyModifiers::NONE,
        }
    }

    fn paste_test_image(controller: &mut Controller, name: &str) {
        let image =
            aster_tui::attachments::Attachment::from_image(name, b"\x89PNG\r\n\x1a\npixels")
                .unwrap();
        let target = PasteTarget::capture(controller);
        clipboard_result(
            controller,
            &target,
            Ok(clipboard::ClipboardContent::Image(image)),
        );
    }

    #[test]
    fn attachment_shortcut_keeps_text_and_queued_snapshots_after_restart() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let root = tempfile::tempdir().unwrap();
        let mut controller = draft_controller(root.path());
        controller.paste("保留这段文字\nsecond line".into());
        controller.state.editor.select_all();
        paste_test_image(&mut controller, "kept.png");
        paste_test_image(&mut controller, "wrong.png");
        controller
            .state
            .queued
            .push_back(aster_tui::drafts::MessageDraft {
                text: "already queued".into(),
                attachments: controller.state.attachments.clone(),
            });
        controller.state.phase = aster_tui::state::Phase::Busy;
        let draft = PasteTarget::capture(&controller);
        assert!(controller
            .key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::ALT))
            .is_empty());
        assert!(draft == PasteTarget::capture(&controller));
        assert_eq!(controller.state.attachments.len(), 1);
        assert_eq!(controller.state.attachments[0].name(), "kept.png");
        assert_eq!(controller.state.queued[0].attachments.len(), 2);
        assert_eq!(controller.state.phase, aster_tui::state::Phase::Busy);
        assert!(controller.persist_draft());

        let restored = recover_draft_controller(root.path());
        assert_eq!(
            restored.state.editor.lines(),
            controller.state.editor.lines()
        );
        assert_eq!(restored.state.attachments, controller.state.attachments);
        assert_eq!(restored.state.queued[0].attachments.len(), 2);
    }

    #[test]
    fn attachment_backspace_respects_text_and_overlay_focus() {
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

        let mut controller = paste_controller();
        controller.paste("x".into());
        paste_test_image(&mut controller, "kept.png");
        paste_test_image(&mut controller, "wrong.png");
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(controller.state.editor.lines(), &[""]);
        assert_eq!(controller.state.attachments.len(), 2);
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(controller.state.attachments.len(), 1);

        controller.state.open(Overlay::Commands);
        controller.state.query = "x".into();
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::ALT));
        assert_eq!(controller.state.attachments.len(), 1);
        assert!(controller.state.query.is_empty());
        controller.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            controller.key(KeyEvent::new_with_kind(
                KeyCode::Backspace,
                KeyModifiers::NONE,
                kind,
            ));
            assert_eq!(controller.state.attachments.len(), 1);
        }
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(controller.state.attachments.is_empty());
        controller.paste("two words".into());
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::ALT));
        assert_eq!(controller.state.editor.lines(), &["two words"]);
    }

    #[test]
    fn composer_attachment_names_overflow_and_undo_work_from_visible_hits() {
        let mut controller = paste_controller();
        controller.paste("keep the draft".into());
        for index in 0..8 {
            paste_test_image(&mut controller, &format!("long-image-name-{index}.png"));
        }
        let original = controller.state.attachments.clone();
        let hits = rendered_hits(&mut controller);
        let (preview, index) = *hits.attachment_preview.last().unwrap();
        assert!(mouse_attachment(
            &mut controller,
            &hits,
            right_mouse(
                MouseEventKind::Down(MouseButton::Left),
                preview.x,
                preview.y,
            )
        ));
        assert_eq!(controller.state.overlay, Some(Overlay::Attachments));
        assert_eq!(controller.state.selected, index);
        assert!(!mouse_attachment(
            &mut controller,
            &hits,
            right_mouse(
                MouseEventKind::Down(MouseButton::Left),
                preview.x,
                preview.y,
            )
        ));
        controller.state.overlay = None;
        let overflow = hits.attachment_overflow.expect("hidden attachment count");
        assert!(mouse_attachment(
            &mut controller,
            &hits,
            right_mouse(
                MouseEventKind::Down(MouseButton::Left),
                overflow.x,
                overflow.y,
            )
        ));
        assert_eq!(controller.state.selected, 0);
        controller.state.overlay = None;
        let (remove, _) = *hits.attachment_remove.last().unwrap();
        assert!(mouse_attachment(
            &mut controller,
            &hits,
            right_mouse(MouseEventKind::Down(MouseButton::Left), remove.x, remove.y,)
        ));
        let hits = rendered_hits(&mut controller);
        let undo = hits.attachment_undo.expect("undo available after removal");
        assert!(mouse_attachment(
            &mut controller,
            &hits,
            right_mouse(MouseEventKind::Down(MouseButton::Left), undo.x, undo.y,)
        ));
        assert_eq!(controller.state.attachments, original);
        assert_eq!(controller.state.editor.lines(), &["keep the draft"]);
    }

    #[test]
    fn attachment_original_click_opens_the_filtered_snapshot_once() {
        let mut controller = paste_controller();
        controller.paste("keep the draft".into());
        paste_test_image(&mut controller, "first.png");
        paste_test_image(&mut controller, "chosen.png");
        controller.state.editor.select_all();
        controller.state.open(Overlay::Attachments);
        controller.state.query = "chosen".into();
        let before = PasteTarget::capture(&controller);
        let attachments = controller.state.attachments.clone();
        let hits = rendered_hits(&mut controller);
        assert!(!hits.attachment_original.is_empty());
        for rect in &hits.attachment_original {
            let actions = mouse_original_image(
                &controller,
                &hits,
                right_mouse(MouseEventKind::Down(MouseButton::Left), rect.x, rect.y),
            );
            let [Action::OpenOriginalImage {
                attachment,
                generation,
            }] = &actions[..]
            else {
                panic!("one local image-viewer action");
            };
            assert_eq!(attachment, &attachments[1]);
            assert_eq!(*generation, controller.state.conversation_generation);
            for kind in [
                MouseEventKind::Up(MouseButton::Left),
                MouseEventKind::Drag(MouseButton::Left),
                MouseEventKind::Down(MouseButton::Right),
            ] {
                assert!(mouse_original_image(
                    &controller,
                    &hits,
                    right_mouse(kind, rect.x, rect.y)
                )
                .is_empty());
            }
        }
        assert!(before == PasteTarget::capture(&controller));
        assert_eq!(controller.state.attachments, attachments);
        let rect = hits.attachment_original[0];
        controller.state.open(Overlay::Commands);
        assert!(mouse_original_image(
            &controller,
            &hits,
            right_mouse(MouseEventKind::Down(MouseButton::Left), rect.x, rect.y),
        )
        .is_empty());
    }

    #[tokio::test]
    async fn terminal_image_resizes_keep_one_worker_and_only_the_latest_result() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 24, 24);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[80, 140, 200].repeat(24 * 24))
                .unwrap();
            writer.finish().unwrap();
        }
        let image = aster_tui::attachments::Attachment::from_image("preview.png", &bytes).unwrap();
        let pixels = aster_tui::terminal_image::CellPixels::new(10, 20).unwrap();
        let mut controller = paste_controller();
        controller.paste("keep the draft".into());
        let before = PasteTarget::capture(&controller);
        controller
            .state
            .terminal_image
            .get_or_request(&image, 40, 20, pixels, [32, 32, 32])
            .unwrap();
        let first = controller.state.terminal_image.take_request().unwrap();
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let mut tasks = IoTasks::default();
        tasks.terminal_image = Some(PendingTerminalImage {
            key: first.key(),
            task: tokio::spawn(async move {
                gate.await.unwrap();
                first.render()
            }),
        });
        let first_worker = tasks.terminal_image.as_ref().unwrap().task.id();
        for columns in [50, 60, 75] {
            controller
                .state
                .terminal_image
                .get_or_request(&image, columns, 20, pixels, [32, 32, 32])
                .unwrap();
            start_terminal_image(&mut controller.state, &mut tasks);
            assert_eq!(
                tasks.terminal_image.as_ref().unwrap().task.id(),
                first_worker
            );
        }
        release.send(()).unwrap();
        let first = tasks.terminal_image.take().unwrap();
        assert!(!controller
            .state
            .terminal_image
            .complete(first.task.await.unwrap()));
        start_terminal_image(&mut controller.state, &mut tasks);
        let latest = tasks
            .terminal_image
            .take()
            .expect("latest resize is queued");
        assert!(controller
            .state
            .terminal_image
            .complete(latest.task.await.unwrap()));
        assert!(controller.state.terminal_image.ready().is_some());
        assert!(controller.state.terminal_image.take_request().is_none());
        assert!(before == PasteTarget::capture(&controller));
    }

    #[test]
    fn original_image_failure_preserves_input_and_does_not_leak_into_another_session() {
        let mut controller = paste_controller();
        controller.paste("keep the draft".into());
        paste_test_image(&mut controller, "image.png");
        controller.state.open(Overlay::Attachments);
        let before = PasteTarget::capture(&controller);
        let attachments = controller.state.attachments.clone();
        let generation = controller.state.conversation_generation;
        let mut tasks = IoTasks::default();
        original_image_result(
            &mut controller,
            &mut tasks,
            generation,
            "image.png",
            Err(anyhow::anyhow!("fixture: no image viewer")),
        );
        assert!(before == PasteTarget::capture(&controller));
        assert_eq!(controller.state.attachments, attachments);
        let notice = controller.state.original_image_notice.clone().unwrap();
        assert!(notice.0.contains("no image viewer"));
        assert!(notice.1);
        controller.state.conversation_generation += 1;
        controller.state.original_image_notice = None;
        let entries = controller.state.entries.len();
        original_image_result(
            &mut controller,
            &mut tasks,
            generation,
            "image.png",
            Err(anyhow::anyhow!("stale failure")),
        );
        assert!(controller.state.original_image_notice.is_none());
        assert_eq!(controller.state.entries.len(), entries);
    }

    #[test]
    fn opened_original_files_survive_completion_and_have_a_bounded_lifetime() {
        let mut controller = paste_controller();
        let image = aster_tui::attachments::Attachment::from_image(
            "image.png",
            b"\x89PNG\r\n\x1a\nsnapshot",
        )
        .unwrap();
        let mut tasks = IoTasks::default();
        let mut paths = Vec::new();
        let generation = controller.state.conversation_generation;
        for _ in 0..18 {
            let original = aster_tui::original_image::prepare(&image).unwrap();
            paths.push(original.path().to_path_buf());
            original_image_result(
                &mut controller,
                &mut tasks,
                generation,
                "image.png",
                Ok(original),
            );
        }
        assert_eq!(tasks.original_images.len(), 16);
        assert!(paths[..2].iter().all(|path| !path.exists()));
        assert!(paths[2..].iter().all(|path| path.exists()));
        drop(tasks);
        assert!(paths.iter().all(|path| !path.exists()));
    }

    #[tokio::test]
    async fn original_image_dispatch_keeps_one_launch_in_flight() {
        let mut controller = paste_controller();
        paste_test_image(&mut controller, "image.png");
        controller.state.open(Overlay::Attachments);
        let mut tasks = IoTasks::default();
        let task = tokio::spawn(std::future::pending::<
            Result<aster_tui::original_image::OriginalImage>,
        >());
        let task_id = task.id();
        tasks.original_image = Some(PendingOriginalImage {
            generation: controller.state.conversation_generation,
            name: "already opening.png".into(),
            task,
        });
        let config = BackendConfig {
            python: PathBuf::from("unused"),
            workspace: PathBuf::from(&controller.state.workspace),
            model: "test".into(),
            client_type: None,
            sandbox: false,
        };
        let actions = controller.open_original_image();
        assert!(
            !dispatch(actions, &mut controller, &mut None, &mut tasks, &config)
                .await
                .unwrap()
        );
        assert_eq!(tasks.original_image.as_ref().unwrap().task.id(), task_id);
        assert_eq!(
            tasks.original_image.as_ref().unwrap().name,
            "already opening.png"
        );
        assert!(controller.state.original_image_notice.is_none());
        tasks.original_image.take().unwrap().task.abort();
    }

    #[test]
    fn composer_pointer_routes_only_to_the_visible_text_area() {
        let mut controller = paste_controller();
        controller.paste("ab中文cd".into());
        let hits = rendered_hits(&mut controller);
        let area = hits.composer_text;
        assert!(controller.composer_mouse(
            area,
            right_mouse(MouseEventKind::Down(MouseButton::Left), area.x + 3, area.y,)
        ));
        assert_eq!(controller.state.editor.cursor(), (0, 2));
        assert!(controller.composer_mouse(
            area,
            right_mouse(MouseEventKind::Drag(MouseButton::Left), area.x + 6, area.y,)
        ));
        assert_eq!(
            controller.state.editor.selection_range(),
            Some(((0, 2), (0, 4)))
        );
        let target = PasteTarget::capture(&controller);
        controller.state.open(Overlay::Commands);
        assert!(!controller.composer_mouse(
            area,
            right_mouse(MouseEventKind::Drag(MouseButton::Left), area.x, area.y,)
        ));
        controller.state.overlay = None;
        assert!(!controller.composer_mouse(
            area,
            right_mouse(MouseEventKind::Drag(MouseButton::Left), area.x, area.y,)
        ));
        assert!(target == PasteTarget::capture(&controller));
        controller.paste("replacement".into());
        assert_eq!(controller.state.editor.lines(), &["abreplacementcd"]);
    }

    #[test]
    fn composer_cut_waits_for_copy_success_and_remains_undoable() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut controller = paste_controller();
        controller.paste("中文\nselected text".into());
        paste_test_image(&mut controller, "keep.png");
        controller.state.editor.select_all();
        let target = CopyTarget::capture(&controller);
        copy_result(
            &mut controller,
            &target,
            true,
            Err(anyhow::anyhow!("fixture unavailable")),
        );
        assert_eq!(controller.state.editor.lines(), &["中文", "selected text"]);
        assert!(controller.state.editor.is_selecting());
        copy_result(&mut controller, &target, false, Ok(()));
        assert!(target == CopyTarget::capture(&controller));
        copy_result(&mut controller, &target, true, Ok(()));
        assert_eq!(controller.state.editor.lines(), &[""]);
        assert_eq!(controller.state.attachments[0].name(), "keep.png");
        controller.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(controller.state.editor.lines(), &["中文", "selected text"]);
        assert!(controller.state.exit_confirmation.is_none());
    }

    #[test]
    fn composer_cut_cannot_modify_a_later_identical_draft_or_a_modal() {
        let mut controller = paste_controller();
        controller.paste("same draft".into());
        controller.state.editor.select_all();
        let target = CopyTarget::capture(&controller);
        controller.paste("temporary draft".into());
        controller.state.editor.select_all();
        controller.paste("same draft".into());
        controller.state.editor.select_all();
        assert!(target.input == PasteTarget::capture(&controller));
        copy_result(&mut controller, &target, true, Ok(()));
        assert_eq!(controller.state.editor.lines(), &["same draft"]);
        let target = CopyTarget::capture(&controller);
        controller.state.open(Overlay::Search);
        copy_result(&mut controller, &target, true, Ok(()));
        assert_eq!(controller.state.editor.lines(), &["same draft"]);
    }

    fn clipboard_test_config() -> BackendConfig {
        BackendConfig {
            python: PathBuf::from("unused"),
            workspace: PathBuf::from("unused"),
            model: "test".into(),
            client_type: None,
            sandbox: false,
        }
    }

    fn delayed_copy(
        controller: &Controller,
        cut: bool,
    ) -> (PendingCopy, tokio::sync::oneshot::Sender<()>) {
        let (release, ready) = tokio::sync::oneshot::channel();
        (
            PendingCopy {
                target: CopyTarget::capture(controller),
                cut,
                task: tokio::spawn(async move { ready.await.context("fixture writer cancelled") }),
            },
            release,
        )
    }

    #[tokio::test]
    async fn clipboard_preserves_new_copy_intents_while_a_writer_is_delayed() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use tui_textarea::CursorMove;

        let mut controller = paste_controller();
        controller.paste("first second".into());
        controller.state.editor.move_cursor(CursorMove::Jump(0, 0));
        controller.state.editor.start_selection();
        controller.state.editor.move_cursor(CursorMove::Jump(0, 5));
        let (pending, release) = delayed_copy(&controller, false);
        let mut tasks = IoTasks::default();
        assert!(tasks.copy.replace(pending).is_none());
        controller.state.editor.cancel_selection();
        controller.state.editor.move_cursor(CursorMove::Jump(0, 6));
        controller.state.editor.start_selection();
        controller.state.editor.move_cursor(CursorMove::End);
        let actions = controller.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        dispatch(
            actions,
            &mut controller,
            &mut None,
            &mut tasks,
            &clipboard_test_config(),
        )
        .await
        .unwrap();
        dispatch(
            vec![Action::PasteClipboard],
            &mut controller,
            &mut None,
            &mut tasks,
            &clipboard_test_config(),
        )
        .await
        .unwrap();
        assert!(next_clipboard(&mut controller, &mut tasks).is_none());
        assert!(tasks.paste.is_none());
        release.send(()).unwrap();
        let mut pending = tasks.copy.take().unwrap();
        let result = (&mut pending.task).await.unwrap();
        complete_copy(&mut controller, &mut tasks, &pending.target, false, result);
        assert!(matches!(next_clipboard(&mut controller, &mut tasks),
            Some(ClipboardIntent::Copy { text, cut: false, .. }) if text == "second"));
        assert!(matches!(
            next_clipboard(&mut controller, &mut tasks),
            Some(ClipboardIntent::Paste(_))
        ));
        assert_eq!(controller.state.editor.lines(), &["first second"]);
    }

    #[tokio::test]
    async fn clipboard_paste_waits_for_cut_and_uses_its_resulting_cursor() {
        use tui_textarea::CursorMove;

        let mut controller = paste_controller();
        controller.paste("before remove after".into());
        controller.state.editor.move_cursor(CursorMove::Jump(0, 7));
        controller.state.editor.start_selection();
        controller.state.editor.move_cursor(CursorMove::Jump(0, 13));
        let (pending, release) = delayed_copy(&controller, true);
        let mut tasks = IoTasks::default();
        assert!(tasks.copy.replace(pending).is_none());
        dispatch(
            vec![Action::PasteClipboard, Action::PasteClipboard],
            &mut controller,
            &mut None,
            &mut tasks,
            &clipboard_test_config(),
        )
        .await
        .unwrap();
        assert!(tasks.paste.is_none());
        assert_eq!(controller.state.editor.lines(), &["before remove after"]);
        release.send(()).unwrap();
        let mut pending = tasks.copy.take().unwrap();
        let result = (&mut pending.task).await.unwrap();
        complete_copy(&mut controller, &mut tasks, &pending.target, true, result);
        assert_eq!(controller.state.editor.lines(), &["before  after"]);
        assert!(!cancel_changed_pastes(&mut controller, &mut tasks));
        let Some(ClipboardIntent::Paste(target)) = next_clipboard(&mut controller, &mut tasks)
        else {
            panic!("paste follows the completed cut");
        };
        assert!(target == PasteTarget::capture(&controller));
        complete_paste(
            &mut controller,
            &mut tasks,
            &target,
            Ok(clipboard::ClipboardContent::Text("remove".into())),
        );
        assert_eq!(controller.state.editor.lines(), &["before remove after"]);
        assert!(!cancel_changed_pastes(&mut controller, &mut tasks));
        let Some(ClipboardIntent::Paste(target)) = next_clipboard(&mut controller, &mut tasks)
        else {
            panic!("the second paste follows the first paste's cursor");
        };
        complete_paste(
            &mut controller,
            &mut tasks,
            &target,
            Ok(clipboard::ClipboardContent::Text("remove".into())),
        );
        assert_eq!(
            controller.state.editor.lines(),
            &["before removeremove after"]
        );
    }

    #[test]
    fn clipboard_failed_copy_cancels_dependent_paste_without_using_old_contents() {
        let mut controller = paste_controller();
        controller.paste("keep selected text".into());
        controller.state.editor.select_all();
        let target = CopyTarget::capture(&controller);
        let mut tasks = IoTasks::default();
        let paste = ClipboardIntent::Paste(PasteTarget::capture(&controller));
        queue_clipboard(&mut controller, &mut tasks, paste);
        complete_copy(
            &mut controller,
            &mut tasks,
            &target,
            true,
            Err(anyhow::anyhow!("fixture writer failed")),
        );
        assert!(next_clipboard(&mut controller, &mut tasks).is_none());
        assert_eq!(controller.state.editor.lines(), &["keep selected text"]);
        assert!(controller.state.editor.is_selecting());
    }

    #[tokio::test]
    async fn clipboard_switch_boundaries_discard_queued_pastes_and_preserve_copies() {
        use aster_tui::controller::Pending;

        for boundary in ["reconnect", "session/load", "session/close"] {
            let mut controller = paste_controller();
            controller.paste("keep selected text".into());
            controller.state.editor.select_all();
            let target = PasteTarget::capture(&controller);
            let (pending, _release) = delayed_copy(&controller, false);
            let mut tasks = IoTasks::default();
            assert!(tasks.copy.replace(pending).is_none());
            for text in ["first copy", "second copy"] {
                tasks
                    .clipboard_requests
                    .push_back(ClipboardIntent::Paste(target.clone()));
                tasks.clipboard_requests.push_back(ClipboardIntent::Copy {
                    target: CopyTarget::capture(&controller),
                    text: text.into(),
                    cut: false,
                });
            }
            let action = if boundary == "reconnect" {
                // Keep backend startup synthetic while exercising its clipboard boundary.
                tasks.restart = Some(tokio::spawn(std::future::pending()));
                Action::Reconnect
            } else {
                Action::Request {
                    method: boundary,
                    params: serde_json::json!({}),
                    pending: if boundary == "session/load" {
                        Pending::Load {
                            id: "another".into(),
                            previous: None,
                        }
                    } else {
                        Pending::NewClose
                    },
                }
            };
            dispatch(
                vec![action],
                &mut controller,
                &mut None,
                &mut tasks,
                &clipboard_test_config(),
            )
            .await
            .unwrap();
            assert!(tasks.copy.is_some(), "{boundary}");
            assert!(tasks.paste.is_none(), "{boundary}");
            for expected in ["first copy", "second copy"] {
                assert!(
                    matches!(tasks.clipboard_requests.pop_front(),
                    Some(ClipboardIntent::Copy { text, .. }) if text == expected),
                    "{boundary}"
                );
            }
            assert!(tasks.clipboard_requests.is_empty(), "{boundary}");
            assert!(target == PasteTarget::capture(&controller), "{boundary}");
        }
    }

    #[tokio::test]
    async fn clipboard_explicit_cancellation_removes_active_and_queued_pastes() {
        let mut controller = paste_controller();
        controller.paste("keep draft".into());
        let target = PasteTarget::capture(&controller);
        let mut tasks = IoTasks::default();
        assert!(tasks
            .paste
            .replace(PendingPaste {
                target: target.clone(),
                task: tokio::spawn(std::future::pending()),
            })
            .is_none());
        tasks.clipboard_requests = VecDeque::from([
            ClipboardIntent::Paste(target.clone()),
            ClipboardIntent::Copy {
                target: CopyTarget::capture(&controller),
                text: "keep copy".into(),
                cut: false,
            },
            ClipboardIntent::Paste(target.clone()),
        ]);
        assert!(cancel_pastes(&mut controller, &mut tasks, |_| true));
        assert!(tasks.paste.is_none());
        assert!(matches!(tasks.clipboard_requests.pop_front(),
            Some(ClipboardIntent::Copy { text, .. }) if text == "keep copy"));
        assert!(tasks.clipboard_requests.is_empty());
        assert!(target == PasteTarget::capture(&controller));
    }

    #[tokio::test]
    async fn clipboard_queued_paste_stays_cancelled_when_input_is_restored() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        for change in ["cursor", "draft"] {
            let mut controller = paste_controller();
            controller.paste("keep selected text".into());
            controller.state.editor.select_all();
            let original = PasteTarget::capture(&controller);
            let (pending, release) = delayed_copy(&controller, false);
            let mut tasks = IoTasks::default();
            assert!(tasks.copy.replace(pending).is_none());
            tasks
                .clipboard_requests
                .push_back(ClipboardIntent::Paste(original.clone()));
            assert!(!cancel_changed_pastes(&mut controller, &mut tasks));
            if change == "cursor" {
                controller.key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
            } else {
                controller.paste("temporary text".into());
            }
            assert!(original != PasteTarget::capture(&controller), "{change}");
            assert!(
                cancel_changed_pastes(&mut controller, &mut tasks),
                "{change}"
            );
            assert!(tasks.copy.is_some());
            if change == "draft" {
                controller.state.editor.select_all();
                controller.paste("keep selected text".into());
            }
            controller.state.editor.select_all();
            assert!(original == PasteTarget::capture(&controller), "{change}");
            assert!(!cancel_changed_pastes(&mut controller, &mut tasks));
            release.send(()).unwrap();
            let mut pending = tasks.copy.take().unwrap();
            let result = (&mut pending.task).await.unwrap();
            complete_copy(&mut controller, &mut tasks, &pending.target, false, result);
            assert!(
                next_clipboard(&mut controller, &mut tasks).is_none(),
                "{change}"
            );
            assert_eq!(controller.state.editor.lines(), &["keep selected text"]);
        }
    }

    #[test]
    fn composer_restored_attachment_is_saved_but_undo_buffer_is_not() {
        let root = tempfile::tempdir().unwrap();
        let mut controller = draft_controller(root.path());
        controller.paste("saved draft".into());
        paste_test_image(&mut controller, "restore.png");
        let original = controller.state.attachments.clone();
        controller.remove_attachment(0);
        assert!(controller.persist_draft());
        let removed = recover_draft_controller(root.path());
        assert!(removed.state.attachments.is_empty());
        assert!(!removed.state.can_undo_attachment());
        assert!(controller.undo_attachment());
        assert!(controller.persist_draft());
        let restored = recover_draft_controller(root.path());
        assert_eq!(restored.state.attachments, original);
        assert_eq!(restored.state.editor.lines(), &["saved draft"]);
        assert!(!restored.state.can_undo_attachment());
    }

    #[test]
    fn attachment_click_removes_only_the_visible_target_and_ignores_covered_controls() {
        let mut controller = paste_controller();
        controller.paste("文字草稿".into());
        paste_test_image(&mut controller, "first.png");
        paste_test_image(&mut controller, "second.png");
        let draft = PasteTarget::capture(&controller);
        let hits = rendered_hits(&mut controller);
        let (remove, _) = hits
            .attachment_remove
            .iter()
            .find(|(_, index)| *index == 0)
            .unwrap();
        for kind in [
            MouseEventKind::Down(MouseButton::Right),
            MouseEventKind::Up(MouseButton::Left),
            MouseEventKind::Drag(MouseButton::Left),
            MouseEventKind::Moved,
        ] {
            assert!(!mouse_remove_attachment(
                &mut controller,
                &hits,
                right_mouse(kind, remove.x, remove.y)
            ));
        }
        let click = right_mouse(MouseEventKind::Down(MouseButton::Left), remove.x, remove.y);
        assert!(!mouse_remove_attachment(
            &mut controller,
            &hits,
            right_mouse(
                MouseEventKind::Down(MouseButton::Left),
                remove.x - 1,
                remove.y
            ),
        ));
        controller.state.open(Overlay::Search);
        assert!(!mouse_remove_attachment(&mut controller, &hits, click));
        controller.state.overlay = None;
        assert!(mouse_remove_attachment(&mut controller, &hits, click));
        assert_eq!(controller.state.attachments.len(), 1);
        assert_eq!(controller.state.attachments[0].name(), "second.png");
        assert!(draft == PasteTarget::capture(&controller));
        assert!(controller.state.queued.is_empty());
        assert!(controller.state.entries.is_empty());
    }

    #[test]
    fn right_click_pastes_once_at_input_cursor_without_submitting() {
        let mut controller = paste_controller();
        controller.paste("beforeafter".into());
        controller
            .state
            .editor
            .move_cursor(tui_textarea::CursorMove::Jump(0, 6));
        let hits = rendered_hits(&mut controller);
        let x = hits.composer.x + 1;
        let y = hits.composer.y + 1;
        assert!(matches!(
            mouse_paste(
                &controller,
                &hits,
                right_mouse(MouseEventKind::Down(MouseButton::Right), x, y)
            ),
            Some(Action::PasteClipboard)
        ));
        let target = PasteTarget::capture(&controller);
        paste_result(&mut controller, &target, Ok("  中文\r\nnext  \n".into()));
        assert_eq!(
            controller.state.editor.lines(),
            &["before  中文", "next  ", "after"]
        );
        for kind in [
            MouseEventKind::Up(MouseButton::Right),
            MouseEventKind::Drag(MouseButton::Right),
            MouseEventKind::Moved,
        ] {
            assert!(mouse_paste(&controller, &hits, right_mouse(kind, x, y)).is_none());
        }
        assert!(mouse_paste(
            &controller,
            &hits,
            right_mouse(
                MouseEventKind::Down(MouseButton::Right),
                hits.transcript.x,
                hits.transcript.y
            )
        )
        .is_none());
        assert!(controller.state.entries.is_empty());
        assert!(controller.state.queued.is_empty());
        assert!(controller.state.session_id.is_none());
    }

    #[test]
    fn right_click_targets_visible_query_and_ignores_obscured_composer() {
        let mut controller = paste_controller();
        controller.paste("kept draft".into());
        for overlay in [
            Overlay::Commands,
            Overlay::Sessions,
            Overlay::Drafts,
            Overlay::Search,
            Overlay::Tools,
            Overlay::Queue,
            Overlay::Rewind,
            Overlay::Help,
            Overlay::Changes,
            Overlay::Diagnostics,
        ] {
            controller.state.open(overlay);
            let hits = rendered_hits(&mut controller);
            assert!(mouse_paste(
                &controller,
                &hits,
                right_mouse(
                    MouseEventKind::Down(MouseButton::Right),
                    hits.composer.x + 1,
                    hits.composer.y + 1
                )
            )
            .is_none());
            if matches!(overlay, Overlay::Help | Overlay::Diagnostics) {
                assert!(hits.query.is_none());
                continue;
            }
            let query = hits.query.expect("visible search field");
            assert!(matches!(
                mouse_paste(
                    &controller,
                    &hits,
                    right_mouse(
                        MouseEventKind::Down(MouseButton::Right),
                        query.x + 1,
                        query.y
                    )
                ),
                Some(Action::PasteClipboard)
            ));
            let target = PasteTarget::capture(&controller);
            paste_result(&mut controller, &target, Ok("中文\nquery".into()));
            assert_eq!(controller.state.query, "中文 query");
            assert_eq!(controller.state.editor.lines(), &["kept draft"]);
        }
        assert!(controller.state.entries.is_empty());
        assert!(controller.state.queued.is_empty());
    }

    #[test]
    fn clipboard_result_replaces_selection_and_does_not_submit() {
        let mut controller = paste_controller();
        controller.paste("replace".into());
        controller.state.editor.select_all();
        let target = PasteTarget::capture(&controller);
        paste_result(&mut controller, &target, Ok("  中文\r\nnext  \n".into()));
        assert_eq!(controller.state.editor.lines(), &["  中文", "next  ", ""]);
        assert!(controller.state.entries.is_empty());
        assert!(controller.state.session_id.is_none());
    }

    #[test]
    fn clipboard_result_updates_overlay_query_and_failure_keeps_inputs() {
        let mut controller = paste_controller();
        controller.paste("kept draft".into());
        controller.state.open(Overlay::Search);
        let target = PasteTarget::capture(&controller);
        paste_result(&mut controller, &target, Ok("中文\nquery".into()));
        assert_eq!(controller.state.query, "中文 query");
        assert_eq!(controller.state.editor.lines(), &["kept draft"]);
        let target = PasteTarget::capture(&controller);
        paste_result(
            &mut controller,
            &target,
            Err(anyhow::anyhow!("Clipboard unavailable")),
        );
        assert_eq!(controller.state.query, "中文 query");
        assert_eq!(controller.state.editor.lines(), &["kept draft"]);
        assert_eq!(controller.state.status, "Clipboard unavailable");
    }

    #[test]
    fn late_clipboard_result_cannot_enter_a_different_input_or_session() {
        for change in [0, 1, 2, 3] {
            let mut controller = paste_controller();
            controller.paste("kept".into());
            let target = PasteTarget::capture(&controller);
            match change {
                0 => controller.state.open(Overlay::Search),
                1 => controller.state.conversation_generation += 1,
                2 => {
                    controller
                        .state
                        .editor
                        .move_cursor(tui_textarea::CursorMove::Head);
                }
                _ => controller.paste(" edited".into()),
            }
            let draft = controller.state.editor.lines().to_vec();
            paste_result(&mut controller, &target, Ok("LATE_CLIPBOARD".into()));
            assert_eq!(controller.state.editor.lines(), draft);
            assert!(controller.state.query.is_empty());
        }
    }

    #[test]
    fn clipboard_result_survives_background_session_preparation() {
        let mut controller = paste_controller();
        controller.state.phase = aster_tui::state::Phase::Preparing;
        controller.paste("beforeafter".into());
        controller
            .state
            .editor
            .move_cursor(tui_textarea::CursorMove::Jump(0, 6));
        let target = PasteTarget::capture(&controller);
        controller.state.session_id = Some("prepared session".into());
        controller.state.phase = aster_tui::state::Phase::Ready;
        paste_result(&mut controller, &target, Ok("中文".into()));
        assert_eq!(controller.state.editor.lines(), &["before中文after"]);
        assert!(controller.state.entries.is_empty());
        assert!(controller.state.queued.is_empty());
    }

    #[test]
    fn clipboard_result_cannot_cross_a_new_conversation_without_a_session_id() {
        let mut controller = paste_controller();
        controller.state.phase = aster_tui::state::Phase::Preparing;
        controller.paste("kept draft".into());
        let target = PasteTarget::capture(&controller);
        controller.state.conversation_generation += 1;
        assert!(controller.state.session_id.is_none());
        paste_result(&mut controller, &target, Ok("WRONG_CONVERSATION".into()));
        assert_eq!(controller.state.editor.lines(), &["kept draft"]);
    }

    #[tokio::test]
    async fn fatal_startup_reports_captured_stderr_after_terminal_run() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("aster_acp.py"),
            "import sys\nprint('\\x1b]52;c;hidden\\x07startup 中文', file=sys.stderr)\nraise ModuleNotFoundError('REVIEW_MISSING_DEPENDENCY_SENTINEL')\n").unwrap();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let python = std::env::var_os("ASTER_TEST_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                root.join(if cfg!(windows) {
                    ".venv/Scripts/python.exe"
                } else {
                    ".venv/bin/python"
                })
            });
        let mut transport = Transport::spawn(&BackendConfig {
            python,
            workspace: directory.path().to_path_buf(),
            model: "no-api-needed".into(),
            client_type: None,
            sandbox: false,
        })
        .await
        .unwrap();
        let reason = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(BackendEvent::Closed(reason)) = transport.try_recv() {
                    break reason;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let cleanup = transport.shutdown().await;
        let error = finish_run(Ok(()), cleanup, Some(reason), &transport).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("ModuleNotFoundError: REVIEW_MISSING_DEPENDENCY_SENTINEL"),
            "{message}"
        );
        assert!(message.contains("startup 中文"));
        assert!(!message.contains('\u{1b}'));
        assert!(!message.contains("hidden"));
    }
}

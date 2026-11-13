// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Reviewable, byte-checked rollback of tracked unstaged text, with durable undo.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions, Permissions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::file_io::{self, is_link, ReadError};
use crate::process_tree::ProcessTree;
use crate::workspace_changes::{ChangeStage, ChangedFile};

const MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_PATCH_BYTES: usize = 256 * 1024;
const MAX_PREVIEW_BYTES: usize = MAX_PATCH_BYTES * 2;
const MAX_INDEX_BYTES: usize = 32 * 1024 * 1024;
const MAX_BACKUP_BYTES: usize = 12 * 1024 * 1024;
const MAX_BACKUPS: usize = 64;
const MAX_STORAGE_FILES: usize = MAX_BACKUPS * 8 + 16;
const MAX_STORAGE_BYTES: u64 = 256 * 1024 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(5);
const BACKUP_SCHEMA: u32 = 1;

#[derive(Clone, Debug)]
pub struct RollbackPlan {
    pub path: String,
    pub scope: String,
    pub preview: String,
    root: PathBuf,
    selected_path: PathBuf,
    before: Vec<u8>,
    after: Vec<u8>,
    permissions: Permissions,
    index_path: PathBuf,
    index: Vec<u8>,
    undo_preview: String,
}

#[derive(Clone, Debug)]
pub struct RollbackReceipt {
    pub path: String,
    pub backup_path: PathBuf,
    pub scope: String,
    pub preview: String,
    pub warning: Option<String>,
    root: PathBuf,
    reviewed_before: Vec<u8>,
    reviewed_after: Vec<u8>,
    selected_backup: PathBuf,
}

#[derive(Clone, Debug)]
struct Hunk {
    header: String,
    old_start: usize,
    old_count: usize,
    new_start: usize,
    new_count: usize,
    lines: Vec<PatchLine>,
}

#[derive(Clone, Debug)]
struct PatchLine {
    kind: u8,
    bytes: Vec<u8>,
}

pub fn hunk_choices(file: &ChangedFile) -> Result<Vec<String>> {
    selectable(file)?;
    Ok(parse_patch(&file.patch)?
        .into_iter()
        .map(|hunk| hunk.header)
        .collect())
}

/// Preparation is read-only. The reviewed patch must be the exact current
/// unstaged patch, and both the worktree bytes and index are captured twice.
pub async fn prepare(
    workspace: &Path,
    file: &ChangedFile,
    hunk: Option<usize>,
) -> Result<RollbackPlan> {
    selectable(file)?;
    let root = repository_root(workspace).await?;
    let index_path = index_path(&root).await?;
    let index = read_regular(&index_path, MAX_INDEX_BYTES)?;
    let path = checked_file(&root, &file.path)?;
    let before = read_regular(&path, MAX_FILE_BYTES)?;
    text_file(&before)?;
    let permissions = fs::metadata(&path)?.permissions();
    let checkout_crlf = check_attributes(&root, &file.path).await?;
    let status = git(
        &root,
        &[
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
            "--renames",
            "--",
            &file.path,
        ],
        64 * 1024,
    )
    .await?;
    let expected_status = format!(" M {}\0", file.path.replace('\\', "/"));
    if status != expected_status.as_bytes() {
        bail!("Rollback requires a tracked, unstaged-only modified file; refresh Changes after resolving staged, renamed, deleted, or conflicted changes");
    }
    let entry = git(
        &root,
        &["ls-files", "--stage", "-z", "--", &file.path],
        64 * 1024,
    )
    .await?;
    let (mode, object) = index_entry(&entry, &file.path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if (permissions.mode() & 0o111 != 0) != (mode == "100755") {
            bail!("Rollback does not handle executable-mode changes");
        }
    }
    #[cfg(not(unix))]
    let _ = mode;
    let current = current_patch(&root, &file.path).await?;
    if current != file.patch.as_bytes() {
        bail!("The selected patch changed; refresh Changes and review it again");
    }
    let patch =
        std::str::from_utf8(&current).context("Rollback supports UTF-8 text patches only")?;
    let hunks = parse_patch(patch)?;
    if hunk.is_some_and(|selected| selected >= hunks.len()) {
        bail!("Selected hunk is no longer available");
    }
    let baseline = git(&root, &["cat-file", "blob", &object], MAX_FILE_BYTES).await?;
    text_file(&baseline)?;
    let after = reverse_hunks(&before, &hunks, hunk, checkout_crlf)?;
    if hunk.is_none() && normalized(&after) != normalized(&baseline) {
        bail!(
            "The displayed patch does not describe the complete text change; rollback was refused"
        );
    }
    if before == after {
        bail!("The selected patch has no rollback changes");
    }
    if read_regular(&index_path, MAX_INDEX_BYTES)? != index
        || read_regular(&checked_file(&root, &file.path)?, MAX_FILE_BYTES)? != before
    {
        bail!("File or Git index changed during preparation; refresh and review again");
    }
    let scope = hunk.map_or_else(
        || format!("All {} unstaged hunks", hunks.len()),
        |selected| format!("Hunk {} of {}", selected + 1, hunks.len()),
    );
    let (preview, undo_preview) = change_previews(&file.path, &hunks, hunk, &before, &after)?;
    Ok(RollbackPlan {
        path: file.path.clone(),
        scope,
        preview,
        root,
        selected_path: path,
        before,
        after,
        permissions,
        index_path,
        index,
        undo_preview,
    })
}

pub async fn apply(plan: RollbackPlan) -> Result<RollbackReceipt> {
    tokio::task::spawn_blocking(move || apply_sync(plan))
        .await
        .context("Rollback worker stopped")?
}

pub async fn undo(receipt: &RollbackReceipt) -> Result<()> {
    let receipt = receipt.clone();
    let index_path = index_path(&receipt.root).await?;
    tokio::task::spawn_blocking(move || undo_sync(&receipt, &index_path))
        .await
        .context("Undo worker stopped")?
}

pub async fn latest(workspace: &Path) -> Result<Option<RollbackReceipt>> {
    let root = repository_root(workspace).await?;
    tokio::task::spawn_blocking(move || {
        let Some(directory) = backup_directory(&root, false)? else {
            return Ok(None);
        };
        let storage = inventory(&directory)?;
        let mut warnings = Vec::new();
        for (_, path) in storage.records.into_iter().rev() {
            let backup = match read_backup(&path) {
                Ok(backup) => backup,
                Err(error) => {
                    warnings.push(format!("Skipped {}: {error:#}", path.display()));
                    continue;
                }
            };
            let current = match checked_file(&root, &backup.path)
                .and_then(|file| read_regular(&file, MAX_FILE_BYTES))
            {
                Ok(current) => current,
                Err(error) => {
                    if backup.state != BackupState::Undone {
                        warnings.push(format!("Skipped {}: {error:#}", backup.path));
                    }
                    continue;
                }
            };
            if current == backup.before()? {
                continue;
            }
            if current != backup.after()? {
                if backup.state != BackupState::Undone {
                    warnings.push(format!(
                        "Skipped {} because it has newer edits; its backup was preserved",
                        backup.path
                    ));
                }
                continue;
            }
            // Metadata publication can fail after the atomic rename. Matching
            // worktree bytes are authoritative, including an Undone journal
            // whose failed finalization was followed by file recovery.
            if backup.state == BackupState::Undone {
                warnings.push(format!(
                    "Recovered undo for {}: its journal says undone but the current bytes still match the rollback; review before retrying",
                    backup.path
                ));
            }
            return Ok(Some(receipt(&root, path, &backup, warning(&warnings))?));
        }
        if !warnings.is_empty() {
            bail!(
                "No applicable undo was found. {}",
                warning(&warnings).unwrap()
            );
        }
        Ok(None)
    })
    .await
    .context("Rollback recovery worker stopped")?
}

fn selectable(file: &ChangedFile) -> Result<()> {
    if file.stage != ChangeStage::Unstaged
        || file.status != "Modified"
        || file.original_path.is_some()
    {
        bail!("Rollback supports only tracked, unstaged modified text files; staged, new, deleted, renamed, conflicted, and type changes are excluded");
    }
    safe_relative(&file.path)?;
    if file.patch.len() > MAX_PATCH_BYTES {
        bail!("Rollback patch exceeds the 256 KiB limit");
    }
    Ok(())
}

fn safe_relative(path: &str) -> Result<&Path> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.contains(['\0', '\n', '\r'])
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        || path
            .replace('\\', "/")
            .split('/')
            .any(|part| part.eq_ignore_ascii_case(".git"))
        || path.replace('\\', "/").starts_with(".aster/tui-")
        || path.contains(':')
    {
        bail!(
            "Rollback path must be a regular workspace-relative file outside Git and TUI storage"
        );
    }
    Ok(parsed)
}

fn checked_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = safe_relative(relative)?;
    let mut selected = root.to_path_buf();
    for (index, part) in relative.components().enumerate() {
        selected.push(part);
        let metadata =
            fs::symlink_metadata(&selected).context("Could not inspect rollback path")?;
        if is_link(&metadata) {
            bail!("Rollback does not follow symbolic links or junctions");
        }
        if index + 1 == relative.components().count() {
            if !metadata.is_file() {
                bail!("Rollback requires an existing regular file");
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() != 1 {
                    bail!("Rollback does not modify files with multiple hard links");
                }
            }
            #[cfg(windows)]
            {
                use std::os::windows::io::AsRawHandle;
                use windows_sys::Win32::Storage::FileSystem::{
                    GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
                };
                let file = File::open(&selected)?;
                let mut info = BY_HANDLE_FILE_INFORMATION::default();
                // SAFETY: the live file handle writes into a correctly sized buffer.
                if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
                    return Err(io::Error::last_os_error())
                        .context("Could not inspect rollback file links");
                }
                if info.nNumberOfLinks != 1 {
                    bail!("Rollback does not modify files with multiple hard links");
                }
            }
        } else if !metadata.is_dir() {
            bail!("Rollback parent path must be a directory");
        }
    }
    if !selected.canonicalize()?.starts_with(root) {
        bail!("Rollback path resolves outside the workspace");
    }
    Ok(selected)
}

fn read_regular(path: &Path, limit: usize) -> Result<Vec<u8>> {
    match file_io::read_regular(path, limit) {
        Ok(bytes) => Ok(bytes),
        Err(ReadError::Inspect(error)) => Err(error).context("Could not inspect rollback snapshot"),
        Err(ReadError::NotRegular | ReadError::TooLarge) => {
            bail!("Rollback snapshot must be a regular file within its size limit");
        }
        Err(ReadError::Open(error)) => Err(error).context("Could not read rollback snapshot"),
        Err(ReadError::OpenedMetadata(error) | ReadError::Read(error)) => Err(error.into()),
        Err(ReadError::ChangedType) => bail!("Rollback file changed while opening it"),
        Err(ReadError::GrewTooLarge) => bail!("Rollback snapshot exceeds its size limit"),
    }
}

fn text_file(bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_FILE_BYTES || bytes.contains(&0) || std::str::from_utf8(bytes).is_err() {
        bail!("Rollback supports UTF-8 text files up to 4 MiB; binary files are excluded");
    }
    Ok(())
}

fn index_entry(bytes: &[u8], path: &str) -> Result<(String, String)> {
    let text = std::str::from_utf8(bytes).context("Git index entry is not UTF-8")?;
    let entry = text
        .strip_suffix('\0')
        .context("Incomplete Git index entry")?;
    let (metadata, selected) = entry.split_once('\t').context("Invalid Git index entry")?;
    let fields: Vec<_> = metadata.split(' ').collect();
    if selected != path.replace('\\', "/")
        || fields.len() != 3
        || fields[2] != "0"
        || !matches!(fields[0], "100644" | "100755")
        || !matches!(fields[1].len(), 40 | 64)
        || !fields[1].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("Rollback requires one regular, tracked, nonconflicted index entry");
    }
    Ok((fields[0].into(), fields[1].into()))
}

async fn repository_root(workspace: &Path) -> Result<PathBuf> {
    let bytes = git(workspace, &["rev-parse", "--show-toplevel"], 64 * 1024).await?;
    let text = std::str::from_utf8(&bytes)?.trim_end_matches(['\r', '\n']);
    PathBuf::from(text)
        .canonicalize()
        .context("Could not locate Git workspace")
}

async fn index_path(root: &Path) -> Result<PathBuf> {
    let bytes = git(
        root,
        &["rev-parse", "--path-format=absolute", "--git-path", "index"],
        64 * 1024,
    )
    .await?;
    let path = PathBuf::from(std::str::from_utf8(&bytes)?.trim_end_matches(['\r', '\n']));
    if !path.is_absolute() {
        bail!("Git returned a nonabsolute index path");
    }
    let parent = path
        .parent()
        .context("Git index has no parent")?
        .canonicalize()?;
    if is_link(&fs::symlink_metadata(&path)?) {
        bail!("Rollback does not use a symbolic-link Git index");
    }
    Ok(parent.join(path.file_name().context("Git index path is invalid")?))
}

async fn check_attributes(root: &Path, path: &str) -> Result<bool> {
    let bytes = git(
        root,
        &[
            "check-attr",
            "-z",
            "filter",
            "working-tree-encoding",
            "text",
            "eol",
            "--",
            path,
        ],
        64 * 1024,
    )
    .await?;
    let fields: Vec<_> = bytes.split(|byte| *byte == 0).collect();
    let mut attrs = BTreeMap::new();
    for record in fields[..fields.len().saturating_sub(1)].chunks(3) {
        if record.len() != 3 {
            bail!("Git returned incomplete file attributes");
        }
        attrs.insert(
            std::str::from_utf8(record[1])?,
            std::str::from_utf8(record[2])?,
        );
    }
    for key in ["filter", "working-tree-encoding"] {
        if !matches!(attrs.get(key).copied(), Some("unspecified" | "unset")) {
            bail!("Rollback excludes files with custom filters or working-tree encodings");
        }
    }
    if attrs.get("text") == Some(&"unset") || attrs.get("eol") == Some(&"lf") {
        return Ok(false);
    }
    if attrs.get("eol") == Some(&"crlf") {
        return Ok(true);
    }
    let autocrlf = git(root, &["config", "--get", "core.autocrlf"], 1024)
        .await
        .unwrap_or_default();
    if autocrlf.eq_ignore_ascii_case(b"true\n") {
        return Ok(true);
    }
    let eol = git(root, &["config", "--get", "core.eol"], 1024)
        .await
        .unwrap_or_default();
    Ok(eol.eq_ignore_ascii_case(b"crlf\n")
        && matches!(attrs.get("text").copied(), Some("set" | "auto")))
}

async fn current_patch(root: &Path, path: &str) -> Result<Vec<u8>> {
    git(
        root,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-relative",
            "--find-renames",
            "--submodule=short",
            "--unified=3",
            "--",
            path,
        ],
        MAX_PATCH_BYTES,
    )
    .await
}

async fn git(root: &Path, arguments: &[&str], limit: usize) -> Result<Vec<u8>> {
    let mut command = Command::new("git");
    command
        .current_dir(root)
        .args([
            "--no-pager",
            "--literal-pathspecs",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "color.ui=false",
            "-c",
            "core.quotePath=false",
        ])
        .args(arguments);
    command
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C");
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_EXTERNAL_DIFF",
        "GIT_DIFF_OPTS",
    ] {
        command.env_remove(name);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let (mut child, mut tree) =
        ProcessTree::spawn(&mut command).context("Could not start rollback Git inspection")?;
    let stdout = child
        .stdout
        .take()
        .context("Git inspection output is unavailable")?;
    let mut bytes = Vec::new();
    let result = timeout(PROCESS_TIMEOUT, async {
        stdout
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .await?;
        if bytes.len() > limit {
            bail!("Git inspection exceeded the rollback size limit");
        }
        if !child.wait().await?.success() {
            bail!("Git could not inspect this rollback selection");
        }
        Ok(())
    })
    .await;
    tree.terminate_and_wait().await?;
    timeout(PROCESS_TIMEOUT, child.wait())
        .await
        .context("Git inspection did not stop")??;
    result.context("Git rollback inspection timed out")??;
    Ok(bytes)
}

fn parse_range(range: &str, prefix: char) -> Result<(usize, usize)> {
    let range = range.strip_prefix(prefix).context("Invalid patch range")?;
    let (start, count) = range.split_once(',').unwrap_or((range, "1"));
    let start: usize = start.parse().context("Invalid patch start")?;
    let count: usize = count.parse().context("Invalid patch count")?;
    if (start == 0 && count != 0) || start > MAX_FILE_BYTES || count > MAX_FILE_BYTES {
        bail!("Patch range exceeds text-file limits");
    }
    Ok((start, count))
}

fn parse_patch(patch: &str) -> Result<Vec<Hunk>> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut headers = 0;
    for line in patch.split_inclusive('\n') {
        if line.starts_with("@@ ") {
            let parts: Vec<_> = line.splitn(5, ' ').collect();
            if parts.len() < 4 || parts[0] != "@@" || parts[3].trim_end() != "@@" {
                bail!("Malformed or combined patch hunk is unsupported");
            }
            let (old_start, old_count) = parse_range(parts[1], '-')?;
            let (new_start, new_count) = parse_range(parts[2], '+')?;
            hunks.push(Hunk {
                header: line.trim_end_matches('\n').into(),
                old_start,
                old_count,
                new_start,
                new_count,
                lines: Vec::new(),
            });
        } else if let Some(hunk) = hunks.last_mut() {
            if line == "\\ No newline at end of file\n" || line == "\\ No newline at end of file" {
                let previous = hunk
                    .lines
                    .last_mut()
                    .context("Orphan patch newline marker")?;
                if previous.bytes.pop() != Some(b'\n') {
                    bail!("Repeated or invalid patch newline marker");
                }
            } else if matches!(line.as_bytes().first(), Some(b' ' | b'+' | b'-')) {
                hunk.lines.push(PatchLine {
                    kind: line.as_bytes()[0],
                    bytes: line.as_bytes()[1..].into(),
                });
            } else {
                bail!("Truncated, binary, or unsupported text patch cannot be rolled back");
            }
        } else if line.starts_with("diff --git ") {
            headers += 1;
        } else if !(line.starts_with("index ")
            || line.starts_with("--- ")
            || line.starts_with("+++ "))
        {
            bail!("Rollback excludes binary, mode, rename, creation, and deletion patches");
        }
    }
    if headers != 1 || hunks.is_empty() {
        bail!("This selection has no complete text hunks to roll back");
    }
    for hunk in &hunks {
        let old = hunk.lines.iter().filter(|line| line.kind != b'+').count();
        let new = hunk.lines.iter().filter(|line| line.kind != b'-').count();
        if old != hunk.old_count
            || new != hunk.new_count
            || !hunk.lines.iter().any(|line| line.kind != b' ')
        {
            bail!("Patch hunk counts do not match the displayed text");
        }
    }
    Ok(hunks)
}

fn normalized(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte != b'\r' || bytes.get(index + 1) != Some(&b'\n') {
            output.push(byte);
        }
    }
    output
}

fn reverse_hunks(
    before: &[u8],
    hunks: &[Hunk],
    selected: Option<usize>,
    checkout_crlf: bool,
) -> Result<Vec<u8>> {
    let lines: Vec<_> = before.split_inclusive(|byte| *byte == b'\n').collect();
    let mut output = Vec::with_capacity(before.len());
    let mut cursor = 0;
    for (number, hunk) in hunks.iter().enumerate() {
        if selected.is_some_and(|selected| selected != number) {
            continue;
        }
        let start = if hunk.new_count == 0 {
            hunk.new_start
        } else {
            hunk.new_start - 1
        };
        if start < cursor
            || start
                .checked_add(hunk.new_count)
                .is_none_or(|end| end > lines.len())
        {
            bail!("Selected hunk no longer fits the current file");
        }
        let mut normalize = checkout_crlf && before.is_empty();
        let mut next = start;
        for patch_line in hunk.lines.iter().filter(|line| line.kind != b'-') {
            if lines[next] != patch_line.bytes {
                if normalized(lines[next]) != normalized(&patch_line.bytes) {
                    bail!("Selected hunk text changed; refresh and review again");
                }
                normalize |= lines[next].ends_with(b"\r\n") && !patch_line.bytes.ends_with(b"\r\n");
            }
            next += 1;
        }
        for line in &lines[cursor..start] {
            output.extend_from_slice(line);
        }
        cursor = start;
        for patch_line in &hunk.lines {
            match patch_line.kind {
                b' ' => {
                    output.extend_from_slice(lines[cursor]);
                    cursor += 1;
                }
                b'+' => cursor += 1,
                b'-' => {
                    if normalize
                        && patch_line.bytes.ends_with(b"\n")
                        && !patch_line.bytes.ends_with(b"\r\n")
                    {
                        output.extend_from_slice(&patch_line.bytes[..patch_line.bytes.len() - 1]);
                        output.extend_from_slice(b"\r\n");
                    } else {
                        output.extend_from_slice(&patch_line.bytes);
                    }
                }
                _ => unreachable!(),
            }
        }
        if output.len() > MAX_FILE_BYTES {
            bail!("Rolled-back file would exceed the 4 MiB limit");
        }
    }
    for line in &lines[cursor..] {
        output.extend_from_slice(line);
    }
    text_file(&output)?;
    Ok(output)
}

fn change_previews(
    path: &str,
    hunks: &[Hunk],
    selected: Option<usize>,
    before: &[u8],
    after: &[u8],
) -> Result<(String, String)> {
    use std::fmt::Write;
    let label = serde_json::to_string(path).expect("strings serialize");
    let mut preview = format!("--- Current {label}\n+++ After rollback {label}\n");
    let mut undo = format!("--- Current {label}\n+++ After undo {label}\n");
    let before_lines: Vec<_> = before.split_inclusive(|byte| *byte == b'\n').collect();
    let after_lines: Vec<_> = after.split_inclusive(|byte| *byte == b'\n').collect();
    let mut delta = 0_isize;
    for (index, hunk) in hunks.iter().enumerate() {
        if selected.is_some_and(|selected| selected != index) {
            continue;
        }
        let current_start = hunk.new_start - usize::from(hunk.new_count != 0);
        let result_start = current_start
            .checked_add_signed(delta)
            .context("Rollback preview has an invalid hunk position")?;
        let result_range = result_start + usize::from(hunk.old_count != 0);
        if selected.is_none() && result_range != hunk.old_start {
            bail!("Rollback preview does not match the complete patch ranges");
        }
        writeln!(
            preview,
            "@@ -{},{} +{},{} @@",
            hunk.new_start, hunk.new_count, result_range, hunk.old_count
        )
        .unwrap();
        writeln!(
            undo,
            "@@ -{},{} +{},{} @@",
            result_range, hunk.old_count, hunk.new_start, hunk.new_count
        )
        .unwrap();
        let mut current = current_start;
        let mut result = result_start;
        for line in &hunk.lines {
            let bytes = if line.kind == b'-' {
                *after_lines
                    .get(result)
                    .context("Incomplete rollback preview")?
            } else {
                *before_lines
                    .get(current)
                    .context("Incomplete rollback preview")?
            };
            let reverse_kind = match line.kind {
                b'+' => b'-',
                b'-' => b'+',
                other => other,
            };
            preview_line(&mut preview, reverse_kind, bytes);
            preview_line(&mut undo, line.kind, bytes);
            if line.kind != b'-' {
                current += 1;
            }
            if line.kind != b'+' {
                result += 1;
            }
        }
        delta += hunk.old_count as isize - hunk.new_count as isize;
    }
    if preview.len() > MAX_PREVIEW_BYTES || undo.len() > MAX_PREVIEW_BYTES {
        bail!("Rollback preview exceeds its size limit");
    }
    Ok((preview, undo))
}

fn preview_line(preview: &mut String, kind: u8, bytes: &[u8]) {
    preview.push(char::from(kind));
    preview.push_str(std::str::from_utf8(bytes).expect("snapshots were UTF-8"));
    if !bytes.ends_with(b"\n") {
        preview.push_str("\n\\ No newline at end of file\n");
    }
}

fn legacy_undo_preview(path: &str, before: &[u8], after: &[u8]) -> Result<String> {
    let old: Vec<_> = after.split_inclusive(|byte| *byte == b'\n').collect();
    let new: Vec<_> = before.split_inclusive(|byte| *byte == b'\n').collect();
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let start = prefix.saturating_sub(3);
    let old_end = (old.len() - suffix + 3).min(old.len());
    let new_end = (new.len() - suffix + 3).min(new.len());
    let label = serde_json::to_string(path)?;
    let old_count = old_end - start;
    let new_count = new_end - start;
    let mut preview = format!(
        "--- Current {label}\n+++ After undo {label}\n@@ -{},{} +{},{} @@\n",
        start + usize::from(old_count != 0),
        old_count,
        start + usize::from(new_count != 0),
        new_count
    );
    for line in &old[start..prefix] {
        preview_line(&mut preview, b' ', line);
    }
    for line in &old[prefix..old.len() - suffix] {
        preview_line(&mut preview, b'-', line);
    }
    for line in &new[prefix..new.len() - suffix] {
        preview_line(&mut preview, b'+', line);
    }
    for line in &old[old.len() - suffix..old_end] {
        preview_line(&mut preview, b' ', line);
    }
    if preview.len() > MAX_PREVIEW_BYTES {
        bail!("Legacy undo needs manual review because its preview exceeds the size limit; the backup was preserved");
    }
    Ok(preview)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum BackupState {
    Prepared,
    Applied,
    Undone,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Backup {
    schema_version: u32,
    path: String,
    before: String,
    after: String,
    readonly: bool,
    mode: Option<u32>,
    state: BackupState,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    undo_preview: String,
}

impl Backup {
    fn before(&self) -> Result<Vec<u8>> {
        decode_snapshot(&self.before)
    }
    fn after(&self) -> Result<Vec<u8>> {
        decode_snapshot(&self.after)
    }
}

fn decode_snapshot(data: &str) -> Result<Vec<u8>> {
    if data.len() > MAX_FILE_BYTES.div_ceil(3) * 4 {
        bail!("Rollback backup exceeds the file size limit");
    }
    let bytes = STANDARD
        .decode(data)
        .context("Rollback backup has invalid encoding")?;
    text_file(&bytes)?;
    Ok(bytes)
}

fn read_backup(path: &Path) -> Result<Backup> {
    let mut backup: Backup = serde_json::from_slice(&read_regular(path, MAX_BACKUP_BYTES)?)
        .context("Rollback backup is corrupt; its file was preserved")?;
    if backup.schema_version != BACKUP_SCHEMA {
        bail!("Rollback backup uses an unsupported schema; its file was preserved");
    }
    safe_relative(&backup.path)?;
    let before = backup.before()?;
    let after = backup.after()?;
    if before == after {
        bail!("Rollback backup contains no changes");
    }
    if backup.scope.is_empty() {
        backup.scope = "Saved rollback".into();
    }
    if backup.undo_preview.is_empty() {
        backup.undo_preview = legacy_undo_preview(&backup.path, &before, &after)?;
    }
    if backup.scope.len() > 128 || backup.undo_preview.len() > MAX_PREVIEW_BYTES {
        bail!("Rollback backup review exceeds its size limit");
    }
    let patch = format!("diff --git a/backup b/backup\n{}", backup.undo_preview);
    let hunks = parse_patch(&patch).context("Rollback backup preview is invalid")?;
    if reverse_hunks(&before, &hunks, None, false)? != after {
        bail!("Rollback backup preview does not match its saved bytes");
    }
    if backup.mode.is_some_and(|mode| mode & !0o7777 != 0) {
        bail!("Rollback backup has invalid permission data");
    }
    Ok(backup)
}

fn warning(warnings: &[String]) -> Option<String> {
    if warnings.is_empty() {
        return None;
    }
    let mut message = warnings
        .iter()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    if warnings.len() > 4 {
        message.push_str(&format!(
            "; {} more backups were skipped",
            warnings.len() - 4
        ));
    }
    Some(message)
}

fn receipt(
    root: &Path,
    backup_path: PathBuf,
    backup: &Backup,
    warning: Option<String>,
) -> Result<RollbackReceipt> {
    Ok(RollbackReceipt {
        path: backup.path.clone(),
        scope: backup.scope.clone(),
        preview: backup.undo_preview.clone(),
        warning,
        root: root.into(),
        reviewed_before: backup.before()?,
        reviewed_after: backup.after()?,
        selected_backup: backup_path.clone(),
        backup_path,
    })
}

fn backup_directory(root: &Path, create: bool) -> Result<Option<PathBuf>> {
    let directory = root.join(".aster").join("tui-rollbacks");
    for path in [root.join(".aster"), directory.clone()] {
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_dir() && !is_link(&metadata) => {}
            Ok(_) => bail!("Rollback storage must not be a symbolic link or file"),
            Err(error) if error.kind() == io::ErrorKind::NotFound && !create => return Ok(None),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let builder = fs::DirBuilder::new();
                #[cfg(unix)]
                let builder = {
                    use std::os::unix::fs::DirBuilderExt;
                    let mut builder = builder;
                    builder.mode(0o700);
                    builder
                };
                builder
                    .create(&path)
                    .context("Could not create rollback storage")?;
            }
            Err(error) => return Err(error).context("Could not inspect rollback storage"),
        }
    }
    Ok(Some(directory))
}

struct Inventory {
    records: BTreeMap<u128, PathBuf>,
    bytes: u64,
    files: usize,
}

fn inventory(directory: &Path) -> Result<Inventory> {
    let mut records = BTreeMap::new();
    let mut bytes = 0_u64;
    let mut files = 0;
    for (number, entry) in fs::read_dir(directory)?.enumerate() {
        if number >= MAX_STORAGE_FILES {
            bail!("Rollback storage contains too many files; no undo copies were removed");
        }
        files += 1;
        let path = entry?.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.is_file() {
            bytes = bytes.saturating_add(metadata.len());
        }
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Some(key) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("rollback-"))
            {
                if key.len() == 32 {
                    if let Ok(sequence) = u128::from_str_radix(key, 16) {
                        records.insert(sequence, path);
                    }
                }
            }
        }
    }
    Ok(Inventory {
        records,
        bytes,
        files,
    })
}

fn reserve_storage(storage: &Inventory, bytes: u64, files: usize) -> Result<()> {
    if storage.bytes.saturating_add(bytes) > MAX_STORAGE_BYTES
        || storage.files.saturating_add(files) > MAX_STORAGE_FILES
    {
        bail!("Rollback storage would exceed its size limit; no undo copies were removed");
    }
    Ok(())
}

fn apply_sync(plan: RollbackPlan) -> Result<RollbackReceipt> {
    apply_with_backup(plan, publish_backup)
}

fn apply_with_backup(
    plan: RollbackPlan,
    mut publish: impl FnMut(&Path, &Backup) -> Result<()>,
) -> Result<RollbackReceipt> {
    let _lock = IndexLock::acquire(&plan.index_path)?;
    let path = checked_file(&plan.root, &plan.path)?;
    if path != plan.selected_path {
        bail!("Rollback selection changed after preparation");
    }
    revalidate(
        &path,
        &plan.before,
        &plan.permissions,
        &plan.index_path,
        &plan.index,
    )?;
    let directory = backup_directory(&plan.root, true)?.expect("created");
    let storage = inventory(&directory)?;
    if storage.records.len() >= MAX_BACKUPS {
        bail!("Rollback storage already has 64 undo copies; no older copy was removed");
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let sequence = now
        .max(storage.records.last_key_value().map_or(0, |(key, _)| *key))
        .checked_add(1)
        .context("Rollback sequence is exhausted")?;
    let backup_path = directory.join(format!("rollback-{sequence:032x}.json"));
    #[cfg(unix)]
    let mode = {
        use std::os::unix::fs::PermissionsExt;
        Some(plan.permissions.mode() & 0o7777)
    };
    #[cfg(not(unix))]
    let mode = None;
    let mut backup = Backup {
        schema_version: BACKUP_SCHEMA,
        path: plan.path.clone(),
        before: STANDARD.encode(&plan.before),
        after: STANDARD.encode(&plan.after),
        readonly: plan.permissions.readonly(),
        mode,
        state: BackupState::Prepared,
        scope: plan.scope,
        undo_preview: plan.undo_preview,
    };
    let bytes = serde_json::to_vec(&backup)?;
    if bytes.len() > MAX_BACKUP_BYTES {
        bail!("Rollback backup would exceed storage limits; no file was changed");
    }
    reserve_storage(&storage, (bytes.len() * 2 + MAX_FILE_BYTES * 4) as u64, 5)?;
    publish(&backup_path, &backup)?;
    let write = FileWrite {
        path: &path,
        permissions: &plan.permissions,
        index_path: &plan.index_path,
        index: &plan.index,
        backup_path: &backup_path,
    };
    let published = write.replace(&plan.before, &plan.after)?;
    backup.state = BackupState::Applied;
    if let Err(error) = publish(&backup_path, &backup) {
        write.recover(&published, &plan.after)
            .with_context(|| format!("Rollback failed and automatic recovery could not complete; saved versions remain beside {}", backup_path.display()))?;
        return Err(error).with_context(|| {
            format!(
                "Rollback did not complete; original bytes remain in {}",
                backup_path.display()
            )
        });
    }
    receipt(&plan.root, backup_path, &backup, None)
}

fn undo_sync(receipt: &RollbackReceipt, index_path: &Path) -> Result<()> {
    undo_with_backup(receipt, index_path, publish_backup)
}

fn undo_with_backup(
    receipt: &RollbackReceipt,
    index_path: &Path,
    mut publish: impl FnMut(&Path, &Backup) -> Result<()>,
) -> Result<()> {
    let _lock = IndexLock::acquire(index_path)?;
    let directory =
        backup_directory(&receipt.root, false)?.context("Rollback backup directory is missing")?;
    if receipt.backup_path.parent() != Some(directory.as_path())
        || receipt.backup_path != receipt.selected_backup
    {
        bail!("Undo receipt does not belong to this workspace");
    }
    let mut backup = read_backup(&receipt.backup_path)?;
    if backup.path != receipt.path {
        bail!("Undo receipt is invalid");
    }
    let path = checked_file(&receipt.root, &receipt.path)?;
    let before = backup.before()?;
    let after = backup.after()?;
    if before != receipt.reviewed_before || after != receipt.reviewed_after {
        bail!("Undo backup changed after review; review the saved change again");
    }
    let permissions = fs::metadata(&path)?.permissions();
    if permissions.readonly() != backup.readonly {
        bail!("File permissions changed after rollback; undo will not overwrite newer work");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if backup.mode != Some(permissions.mode() & 0o7777) {
            bail!("File permissions changed after rollback; undo will not overwrite newer work");
        }
    }
    let index = read_regular(index_path, MAX_INDEX_BYTES)?;
    if read_regular(&path, MAX_FILE_BYTES)? != after {
        bail!("File changed after rollback; undo will not overwrite newer work");
    }
    let storage = inventory(&directory)?;
    reserve_storage(&storage, (MAX_FILE_BYTES * 4 + MAX_BACKUP_BYTES) as u64, 4)?;
    let write = FileWrite {
        path: &path,
        permissions: &permissions,
        index_path,
        index: &index,
        backup_path: &receipt.backup_path,
    };
    let published = write.replace(&after, &before)?;
    backup.state = BackupState::Undone;
    if let Err(error) = publish(&receipt.backup_path, &backup) {
        write.recover(&published, &before)
            .context("Undo failed and automatic recovery could not complete; saved versions remain beside its backup")?;
        return Err(error)
            .context("Undo did not complete; the previous file was retained or restored");
    }
    Ok(())
}

fn revalidate(
    path: &Path,
    expected: &[u8],
    permissions: &Permissions,
    index_path: &Path,
    index: &[u8],
) -> Result<()> {
    if read_regular(index_path, MAX_INDEX_BYTES)? != index {
        bail!("Git index changed after review; refresh and prepare rollback again");
    }
    if read_regular(path, MAX_FILE_BYTES)? != expected {
        bail!("File changed after review; newer work was preserved");
    }
    let current = fs::metadata(path)?.permissions();
    if current.readonly() != permissions.readonly() {
        bail!("File permissions changed after review");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if current.mode() != permissions.mode() {
            bail!("File permissions changed after review");
        }
    }
    Ok(())
}

struct FileWrite<'a> {
    path: &'a Path,
    permissions: &'a Permissions,
    index_path: &'a Path,
    index: &'a [u8],
    backup_path: &'a Path,
}

#[derive(Debug)]
struct PublishedFile {
    captured: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublicationPoint {
    BeforePublication,
    AfterPublication,
    BeforeRecovery,
    AfterRecovery,
}

impl FileWrite<'_> {
    fn replace(&self, expected: &[u8], replacement: &[u8]) -> Result<PublishedFile> {
        self.replace_with(expected, replacement, &mut |_, _| Ok(()))
    }

    fn replace_with(
        &self,
        expected: &[u8],
        replacement: &[u8],
        hook: &mut impl FnMut(PublicationPoint, &Path) -> Result<()>,
    ) -> Result<PublishedFile> {
        let directory = self
            .backup_path
            .parent()
            .context("Rollback backup has no parent")?;
        let parent = self.path.parent().context("Rollback file has no parent")?;
        let _parents = ParentGuards::acquire(parent, directory)?;
        reject_link_ancestors(parent)?;
        reject_link_ancestors(directory)?;
        let temporary = retained_path(self.backup_path, "replacement")?;
        let cleanup = TemporaryFile::new(temporary.clone());
        let mut file = private_options()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(replacement)
            .context("Could not write rollback replacement")?;
        file.set_permissions(self.permissions.clone())?;
        file.sync_all()
            .context("Could not flush rollback replacement")?;
        drop(file);
        sync_directory(directory)?;
        revalidate(
            self.path,
            expected,
            self.permissions,
            self.index_path,
            self.index,
        )?;
        let captured = capture_path(self.backup_path, &temporary)?;
        // From here onward, either path may contain the actual displaced file,
        // even if the OS reports a partial failure. Never give either to a
        // temporary-file guard: an editor may still be writing an open handle.
        cleanup.preserve();
        hook(PublicationPoint::BeforePublication, &captured)?;
        capture_replace(&temporary, self.path, &captured).with_context(|| {
            format!(
                "Could not publish rollback; inspect the current file and retained versions {} and {}",
                temporary.display(), captured.display()
            )
        })?;
        let published = PublishedFile { captured };
        let result = (|| {
            hook(PublicationPoint::AfterPublication, &published.captured)?;
            sync_directory(directory)?;
            sync_directory(parent)?;
            revalidate(
                &published.captured,
                expected,
                self.permissions,
                self.index_path,
                self.index,
            )?;
            if read_regular(self.path, MAX_FILE_BYTES)? != replacement {
                bail!("File changed while rollback was being published");
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.recover_with(&published, replacement, hook).with_context(|| {
                format!("Publication failed ({error:#}) and automatic recovery could not complete; all retained versions are beside {}", self.backup_path.display())
            })?;
            return Err(error).with_context(|| {
                format!("Rollback publication did not complete; concurrent bytes were retained beside {}", self.backup_path.display())
            });
        }
        Ok(published)
    }

    fn recover(&self, published: &PublishedFile, written: &[u8]) -> Result<()> {
        self.recover_with(published, written, &mut |_, _| Ok(()))
    }

    fn recover_with(
        &self,
        published: &PublishedFile,
        written: &[u8],
        hook: &mut impl FnMut(PublicationPoint, &Path) -> Result<()>,
    ) -> Result<()> {
        // Preserve a newer current version instead of overwriting it. The
        // displaced original remains at its retained path for manual recovery.
        if read_regular(self.path, MAX_FILE_BYTES)? != written {
            return Ok(());
        }
        let parent = self.path.parent().context("Rollback file has no parent")?;
        let directory = self
            .backup_path
            .parent()
            .context("Rollback backup has no parent")?;
        let _parents = ParentGuards::acquire(parent, directory)?;
        reject_link_ancestors(directory)?;
        reject_link_ancestors(parent)?;
        let metadata = fs::symlink_metadata(&published.captured)?;
        if !metadata.is_file() || is_link(&metadata) {
            bail!(
                "Retained file changed type; inspect {}",
                published.captured.display()
            );
        }
        revalidate(
            self.path,
            written,
            self.permissions,
            self.index_path,
            self.index,
        )?;
        let displaced = capture_path(self.backup_path, &published.captured)?;
        hook(PublicationPoint::BeforeRecovery, &displaced)?;
        // Move the retained file itself back, including writes through any
        // editor handle that survived publication. Copying the reviewed bytes
        // here would lose those writes. The destination is retained again.
        capture_replace(&published.captured, self.path, &displaced).with_context(|| {
            format!(
                "Recovery could not finish; inspect retained files {} and {}",
                published.captured.display(),
                displaced.display()
            )
        })?;
        hook(PublicationPoint::AfterRecovery, &displaced)?;
        sync_directory(directory)?;
        sync_directory(parent)?;
        if read_regular(&displaced, MAX_FILE_BYTES)? != written {
            bail!("Another save occurred during recovery; the prior file is restored and the newer displaced version is retained at {}", displaced.display());
        }
        Ok(())
    }
}

fn retained_path(backup: &Path, label: &str) -> Result<PathBuf> {
    let directory = backup.parent().context("Rollback backup has no parent")?;
    let stem = backup
        .file_stem()
        .and_then(|name| name.to_str())
        .context("Invalid rollback backup name")?;
    Ok(unique_temporary(directory, &format!("{stem}-{label}"))?.with_extension("capture"))
}

fn capture_path(backup: &Path, replacement: &Path) -> Result<PathBuf> {
    #[cfg(windows)]
    {
        let _ = replacement;
        retained_path(backup, "displaced")
    }
    #[cfg(not(windows))]
    {
        let _ = backup;
        Ok(replacement.into())
    }
}

// Windows directory handles deny deletion until publication/recovery finishes,
// preventing a parent from being renamed or replaced by a junction mid-write.
struct ParentGuards {
    _files: Vec<File>,
}

impl ParentGuards {
    fn acquire(target_parent: &Path, storage: &Path) -> Result<Self> {
        let mut files = Vec::new();
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ,
                FILE_SHARE_WRITE,
            };
            let mut paths = Vec::new();
            for path in [target_parent, storage] {
                for ancestor in path.ancestors().collect::<Vec<_>>().into_iter().rev() {
                    if paths.contains(&ancestor) {
                        continue;
                    }
                    let file = OpenOptions::new()
                        .read(true)
                        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
                        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
                        .open(ancestor)
                        .context("Could not hold rollback parent directories stable")?;
                    let metadata = file.metadata()?;
                    if !metadata.is_dir() || is_link(&metadata) {
                        bail!("Rollback parent became a symbolic link or non-directory");
                    }
                    paths.push(ancestor);
                    files.push(file);
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (target_parent, storage, &mut files);
        }
        Ok(Self { _files: files })
    }
}

fn reject_link_ancestors(path: &Path) -> Result<()> {
    for parent in path.ancestors() {
        if is_link(&fs::symlink_metadata(parent)?) {
            bail!("Rollback path acquired a symbolic-link parent");
        }
    }
    Ok(())
}

fn private_options() -> OpenOptions {
    let options = OpenOptions::new();
    #[cfg(unix)]
    let options = {
        use std::os::unix::fs::OpenOptionsExt;
        let mut options = options;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        options
    };
    options
}

fn unique_temporary(directory: &Path, label: &str) -> Result<PathBuf> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(directory.join(format!(
        ".aster-rollback-{label}-{}-{time:x}-{count}.tmp",
        std::process::id()
    )))
}

fn publish_backup(path: &Path, backup: &Backup) -> Result<()> {
    let bytes = serde_json::to_vec(backup)?;
    if bytes.len() > MAX_BACKUP_BYTES {
        bail!("Rollback backup exceeds its storage limit");
    }
    let directory = path.parent().context("Rollback backup has no parent")?;
    reject_link_ancestors(directory)?;
    let temporary = unique_temporary(directory, "backup")?;
    let cleanup = TemporaryFile::new(temporary.clone());
    let mut file = private_options()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    if path.exists() {
        read_regular(path, MAX_BACKUP_BYTES)?;
        atomic_replace(&temporary, path)?;
    } else {
        fs::rename(&temporary, path)?;
    }
    drop(cleanup);
    sync_directory(directory)?;
    Ok(())
}

fn atomic_replace(temporary: &Path, target: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;
        let temporary: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
        let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: both terminated paths stay live; null optional parameters mean
        // no extra backup. ReplaceFile preserves the original Windows ACLs.
        if unsafe {
            ReplaceFileW(
                target.as_ptr(),
                temporary.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        fs::rename(temporary, target)
    }
}

/// A successful exchange always retains the actual displaced file. No fallback
/// may replace this with a plain rename: it would reopen the editor-save race.
fn capture_replace(replacement: &Path, target: &Path, captured: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;
        let replacement: Vec<u16> = replacement
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        let captured: Vec<u16> = captured.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: all three terminated paths remain live; the backup argument
        // retains the replaced file. Windows preserves the target ACLs.
        if unsafe {
            ReplaceFileW(
                target.as_ptr(),
                replacement.as_ptr(),
                captured.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use std::ffi::CString;
        use std::os::fd::AsRawFd;
        use std::os::unix::ffi::OsStrExt;
        if captured != replacement {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Exchange capture must be the replacement path",
            ));
        }
        let source_parent = anchored_directory(replacement.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Replacement has no parent")
        })?)?;
        let target_parent =
            anchored_directory(target.parent().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Target has no parent")
            })?)?;
        let source = CString::new(
            replacement
                .file_name()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Replacement has no name")
                })?
                .as_bytes(),
        )?;
        let target = CString::new(
            target
                .file_name()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Target has no name"))?
                .as_bytes(),
        )?;
        // SAFETY: names and directory descriptors remain live. Both operations
        // atomically exchange entries; they refuse cross-filesystem moves.
        #[cfg(target_os = "linux")]
        let result = unsafe {
            libc::renameat2(
                source_parent.as_raw_fd(),
                source.as_ptr(),
                target_parent.as_raw_fd(),
                target.as_ptr(),
                libc::RENAME_EXCHANGE,
            )
        };
        #[cfg(target_os = "macos")]
        let result = unsafe {
            libc::renameatx_np(
                source_parent.as_raw_fd(),
                source.as_ptr(),
                target_parent.as_raw_fd(),
                target.as_ptr(),
                libc::RENAME_SWAP,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        source_parent.sync_all()?;
        target_parent.sync_all()?;
        Ok(())
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = (replacement, target, captured);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "This platform does not support rollback with atomic file retention",
        ))
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn anchored_directory(path: &Path) -> io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Rollback directory is not absolute",
        ));
    }
    let mut directory = File::open("/")?;
    for component in path.components() {
        let Component::Normal(name) = component else {
            if component == Component::RootDir {
                continue;
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid rollback directory",
            ));
        };
        let name = CString::new(name.as_bytes())?;
        // SAFETY: the live descriptor and terminated component are valid. No
        // symlink is followed, including intermediate parent components.
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat returned a fresh owned descriptor.
        directory = unsafe { File::from_raw_fd(fd) };
    }
    Ok(directory)
}

fn sync_directory(directory: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(directory)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = directory;
    Ok(())
}

struct TemporaryFile(Option<PathBuf>);
impl TemporaryFile {
    fn new(path: PathBuf) -> Self {
        Self(Some(path))
    }
    fn preserve(mut self) {
        self.0 = None;
    }
}
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = fs::remove_file(path);
        }
    }
}

struct IndexLock {
    path: PathBuf,
    file: Option<File>,
}

impl IndexLock {
    fn acquire(index_path: &Path) -> Result<Self> {
        let mut name: OsString = index_path.as_os_str().into();
        name.push(".lock");
        let path = PathBuf::from(name);
        reject_link_ancestors(path.parent().context("Git index has no parent")?)?;
        let file = private_options()
            .create_new(true)
            .write(true)
            .open(&path)
            .context("Git index is busy or cannot be locked; no file was changed")?;
        Ok(Self {
            path,
            file: Some(file),
        })
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_reader_keeps_required_files_and_rollback_diagnostics() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot");
        assert_eq!(
            read_regular(&path, 1).unwrap_err().to_string(),
            "Could not inspect rollback snapshot"
        );
        fs::write(&path, b"too large").unwrap();
        for path in [directory.path(), path.as_path()] {
            assert_eq!(
                read_regular(path, 1).unwrap_err().to_string(),
                "Rollback snapshot must be a regular file within its size limit"
            );
        }
    }

    fn git_ok(root: &Path, args: &[&str]) -> Vec<u8> {
        let mut command = std::process::Command::new("git");
        command
            .current_dir(root)
            .args(["-c", "core.hooksPath=", "-c", "commit.gpgsign=false"])
            .args(args);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    fn fixture(files: &[(&str, &str)]) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        git_ok(root, &["init", "--quiet"]);
        git_ok(root, &["config", "user.name", "Rollback Test"]);
        git_ok(
            root,
            &["config", "user.email", "rollback-test@example.invalid"],
        );
        git_ok(root, &["config", "core.autocrlf", "false"]);
        for (path, contents) in files {
            let absolute = root.join(path);
            fs::create_dir_all(absolute.parent().unwrap()).unwrap();
            fs::write(absolute, contents).unwrap();
        }
        git_ok(root, &["add", "--all"]);
        git_ok(root, &["commit", "--quiet", "--no-verify", "-m", "fixture"]);
        directory
    }

    async fn selection(root: &Path, path: &str) -> ChangedFile {
        ChangedFile {
            path: path.into(),
            original_path: None,
            status: "Modified".into(),
            stage: ChangeStage::Unstaged,
            patch: String::from_utf8(current_patch(root, path).await.unwrap()).unwrap(),
        }
    }

    struct WriteFixture {
        _directory: tempfile::TempDir,
        path: PathBuf,
        backup_path: PathBuf,
        index_path: PathBuf,
        index: Vec<u8>,
        permissions: Permissions,
    }

    impl WriteFixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().canonicalize().unwrap();
            let storage = backup_directory(&root, true).unwrap().unwrap();
            let path = root.join("tracked.txt");
            fs::write(&path, b"reviewed edit\n").unwrap();
            let permissions = fs::metadata(&path).unwrap().permissions();
            let index_path = root.join("test-index");
            let index = b"unchanged test index".to_vec();
            fs::write(&index_path, &index).unwrap();
            Self {
                _directory: directory,
                path,
                backup_path: storage.join("rollback-00000000000000000000000000000001.json"),
                index_path,
                index,
                permissions,
            }
        }

        fn writer(&self) -> FileWrite<'_> {
            FileWrite {
                path: &self.path,
                permissions: &self.permissions,
                index_path: &self.index_path,
                index: &self.index,
                backup_path: &self.backup_path,
            }
        }

        fn retained(&self) -> Vec<Vec<u8>> {
            fs::read_dir(self.backup_path.parent().unwrap())
                .unwrap()
                .map(|entry| fs::read(entry.unwrap().path()).unwrap())
                .collect()
        }
    }

    #[test]
    fn final_publication_gap_save_is_restored_from_the_actual_displaced_file() {
        let fixture = WriteFixture::new();
        let result =
            fixture
                .writer()
                .replace_with(b"reviewed edit\n", b"rolled back\n", &mut |point, _| {
                    if point == PublicationPoint::BeforePublication {
                        fs::write(&fixture.path, b"saved after validation\n")?;
                    }
                    Ok(())
                });
        assert!(result.is_err());
        assert_eq!(
            fs::read(&fixture.path).unwrap(),
            b"saved after validation\n"
        );
        assert!(fixture.retained().contains(&b"rolled back\n".to_vec()));
        assert_eq!(fs::read(&fixture.index_path).unwrap(), fixture.index);
    }

    #[test]
    fn recovery_retains_another_save_in_its_own_final_validation_gap() {
        let fixture = WriteFixture::new();
        let error = fixture
            .writer()
            .replace_with(b"reviewed edit\n", b"rolled back\n", &mut |point, _| {
                match point {
                    PublicationPoint::BeforePublication => {
                        fs::write(&fixture.path, b"first concurrent save\n")?
                    }
                    PublicationPoint::BeforeRecovery => {
                        fs::write(&fixture.path, b"second concurrent save\n")?
                    }
                    _ => {}
                }
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("Another save occurred during recovery"));
        assert_eq!(fs::read(&fixture.path).unwrap(), b"first concurrent save\n");
        assert!(fixture
            .retained()
            .contains(&b"second concurrent save\n".to_vec()));
    }

    #[test]
    fn failure_after_exchange_leaves_newer_current_and_displaced_versions_intact() {
        let fixture = WriteFixture::new();
        let error = fixture
            .writer()
            .replace_with(b"reviewed edit\n", b"rolled back\n", &mut |point, _| {
                if point == PublicationPoint::AfterPublication {
                    fs::write(&fixture.path, b"saved after exchange\n")?;
                    bail!("Injected failure after exchange");
                }
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{error:#}").contains("Injected failure"));
        assert_eq!(fs::read(&fixture.path).unwrap(), b"saved after exchange\n");
        assert!(fixture.retained().contains(&b"reviewed edit\n".to_vec()));
    }

    #[test]
    fn open_editor_handle_remains_durable_and_recovery_moves_its_latest_bytes() {
        use std::io::{Seek, SeekFrom};
        let fixture = WriteFixture::new();
        let mut editor = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fixture.path)
            .unwrap();
        let published = fixture
            .writer()
            .replace(b"reviewed edit\n", b"rolled back\n")
            .unwrap();
        editor.seek(SeekFrom::Start(0)).unwrap();
        let late_save = b"late save through the original editor handle\n";
        editor.write_all(late_save).unwrap();
        editor.set_len(late_save.len() as u64).unwrap();
        editor.sync_all().unwrap();
        assert_eq!(fs::read(&published.captured).unwrap(), late_save);
        assert_eq!(fs::read(&fixture.path).unwrap(), b"rolled back\n");
        #[cfg(windows)]
        {
            // ReplaceFile refuses to consume a replacement still open for
            // writing. Refusal must preserve both versions for later recovery.
            assert!(fixture
                .writer()
                .recover(&published, b"rolled back\n")
                .is_err());
            assert_eq!(fs::read(&published.captured).unwrap(), late_save);
            assert_eq!(fs::read(&fixture.path).unwrap(), b"rolled back\n");
        }
        drop(editor);
        fixture
            .writer()
            .recover(&published, b"rolled back\n")
            .unwrap();
        assert_eq!(fs::read(&fixture.path).unwrap(), late_save);
        assert!(fixture.retained().contains(&b"rolled back\n".to_vec()));
    }

    #[test]
    fn worker_unwind_after_exchange_never_deletes_the_displaced_capture() {
        let fixture = WriteFixture::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture
                .writer()
                .replace_with(b"reviewed edit\n", b"rolled back\n", &mut |point, _| {
                    if point == PublicationPoint::AfterPublication {
                        panic!("Injected worker interruption");
                    }
                    Ok(())
                })
        }));
        assert!(result.is_err());
        assert_eq!(fs::read(&fixture.path).unwrap(), b"rolled back\n");
        assert!(fixture.retained().contains(&b"reviewed edit\n".to_vec()));
    }

    #[cfg(windows)]
    #[test]
    fn parent_directory_handles_prevent_a_rename_during_publication() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let parent = root.join("parent");
        let renamed = root.join("renamed-parent");
        fs::create_dir(&parent).unwrap();
        let guards = ParentGuards::acquire(&parent, &root).unwrap();
        assert!(fs::rename(&parent, &renamed).is_err());
        drop(guards);
        fs::rename(&parent, &renamed).unwrap();
    }

    #[test]
    fn later_selected_hunk_preview_accounts_for_retained_insertions_and_deletions() {
        for (patch, current, expected, header, undo_header) in [
            (
                "diff --git a/f b/f\n@@ -0,0 +1,1 @@\n+extra\n@@ -3,1 +3,0 @@\n-three\n",
                "extra\none\ntwo\n",
                "extra\none\ntwo\nthree\n",
                "@@ -3,0 +4,1 @@",
                "@@ -4,1 +3,0 @@",
            ),
            (
                "diff --git a/f b/f\n@@ -1,1 +0,0 @@\n-extra\n@@ -3,0 +3,1 @@\n+three\n",
                "one\ntwo\nthree\n",
                "one\ntwo\n",
                "@@ -3,1 +2,0 @@",
                "@@ -2,0 +3,1 @@",
            ),
        ] {
            let hunks = parse_patch(patch).unwrap();
            let after = reverse_hunks(current.as_bytes(), &hunks, Some(1), false).unwrap();
            assert_eq!(after, expected.as_bytes());
            let (preview, undo) =
                change_previews("f", &hunks, Some(1), current.as_bytes(), &after).unwrap();
            assert!(preview.contains(header), "{preview}");
            assert!(undo.contains(undo_header), "{undo}");
            let undo_hunks = parse_patch(&format!("diff --git a/f b/f\n{undo}")).unwrap();
            assert_eq!(
                reverse_hunks(current.as_bytes(), &undo_hunks, None, false).unwrap(),
                after
            );
        }
    }

    #[test]
    fn legacy_preview_is_exact_for_empty_crlf_and_missing_final_newline_snapshots() {
        for (before, after) in [
            ("edit\r\n", "baseline\r\n"),
            ("edit", "baseline\n"),
            ("new content\n", ""),
            ("", "old content\n"),
            ("a\nb\nc\nd\nx\nf\ng\nh\ni\n", "a\nb\nc\nd\ne\nf\ng\nh\ni\n"),
        ] {
            let preview = legacy_undo_preview("file", before.as_bytes(), after.as_bytes()).unwrap();
            let hunks = parse_patch(&format!("diff --git a/file b/file\n{preview}")).unwrap();
            assert_eq!(
                reverse_hunks(before.as_bytes(), &hunks, None, false).unwrap(),
                after.as_bytes()
            );
        }
    }

    #[tokio::test]
    async fn selected_hunk_preserves_other_hunks_other_files_and_exact_index() {
        let original: String = (0..24).map(|index| format!("line {index}\n")).collect();
        let directory = fixture(&[("tracked.txt", &original), ("other.txt", "other\n")]);
        let root = directory.path();
        let changed = original
            .replace("line 2\n", "agent change\n")
            .replace("line 20\n", "user change\n");
        fs::write(root.join("tracked.txt"), &changed).unwrap();
        fs::write(root.join("other.txt"), "unrelated dirty file\n").unwrap();
        let selected = selection(root, "tracked.txt").await;
        assert_eq!(hunk_choices(&selected).unwrap().len(), 2);
        let plan = prepare(root, &selected, Some(0)).await.unwrap();
        assert!(plan.preview.contains("-agent change"));
        assert!(plan.preview.contains("+line 2"));
        let index_path = plan.index_path.clone();
        let index = plan.index.clone();
        let receipt = apply(plan).await.unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            original.replace("line 20\n", "user change\n")
        );
        assert_eq!(
            fs::read_to_string(root.join("other.txt")).unwrap(),
            "unrelated dirty file\n"
        );
        assert_eq!(fs::read(&index_path).unwrap(), index);
        assert!(receipt.backup_path.is_file());
        assert!(!root.join(".git/index.lock").exists());
        undo(&receipt).await.unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            changed
        );
        assert_eq!(fs::read(&index_path).unwrap(), index);
    }

    #[tokio::test]
    async fn complete_file_rollback_has_durable_undo_and_preserves_later_unrelated_staging() {
        let directory = fixture(&[("tracked.txt", "baseline\n"), ("other.txt", "other\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "unsent changes\n").unwrap();
        let selected = selection(root, "tracked.txt").await;
        let receipt = apply(prepare(root, &selected, None).await.unwrap())
            .await
            .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "baseline\n"
        );
        let loaded = latest(root).await.unwrap().unwrap();
        assert_eq!(loaded.backup_path, receipt.backup_path);
        fs::write(root.join("other.txt"), "later staged work\n").unwrap();
        git_ok(root, &["add", "--", "other.txt"]);
        let index = fs::read(root.join(".git/index")).unwrap();
        undo(&loaded).await.unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "unsent changes\n"
        );
        assert_eq!(fs::read(root.join(".git/index")).unwrap(), index);
        assert!(latest(root).await.unwrap().is_none());
        assert!(undo(&loaded).await.is_err());
        let backup = read_backup(&loaded.backup_path).unwrap();
        assert_eq!(backup.before().unwrap(), b"unsent changes\n");
        assert_eq!(backup.after().unwrap(), b"baseline\n");
    }

    #[tokio::test]
    async fn stale_display_or_prepared_file_is_refused_before_changing_any_bytes() {
        let directory = fixture(&[("tracked.txt", "baseline\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "first edit\n").unwrap();
        let old = selection(root, "tracked.txt").await;
        fs::write(root.join("tracked.txt"), "second edit\n").unwrap();
        assert!(prepare(root, &old, None)
            .await
            .unwrap_err()
            .to_string()
            .contains("patch changed"));
        let selected = selection(root, "tracked.txt").await;
        let plan = prepare(root, &selected, None).await.unwrap();
        fs::write(root.join("tracked.txt"), "newer user work\n").unwrap();
        assert!(apply(plan)
            .await
            .unwrap_err()
            .to_string()
            .contains("changed after review"));
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "newer user work\n"
        );
        assert!(!root.join(".aster/tui-rollbacks").exists());
        assert!(!root.join(".git/index.lock").exists());
    }

    #[tokio::test]
    async fn changed_index_and_busy_index_lock_refuse_application() {
        let directory = fixture(&[("tracked.txt", "baseline\n"), ("other.txt", "other\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "edit\n").unwrap();
        let selected = selection(root, "tracked.txt").await;
        let plan = prepare(root, &selected, None).await.unwrap();
        let lock = IndexLock::acquire(&plan.index_path).unwrap();
        assert!(apply(plan.clone())
            .await
            .unwrap_err()
            .to_string()
            .contains("locked"));
        assert!(root.join(".git/index.lock").exists());
        drop(lock);
        fs::write(root.join("other.txt"), "stage this\n").unwrap();
        git_ok(root, &["add", "--", "other.txt"]);
        let index = fs::read(root.join(".git/index")).unwrap();
        assert!(apply(plan)
            .await
            .unwrap_err()
            .to_string()
            .contains("index changed"));
        assert_eq!(fs::read(root.join(".git/index")).unwrap(), index);
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "edit\n"
        );
    }

    #[tokio::test]
    async fn undo_never_overwrites_newer_file_edits() {
        let directory = fixture(&[("tracked.txt", "baseline\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "original edit\n").unwrap();
        let receipt = apply(
            prepare(root, &selection(root, "tracked.txt").await, None)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        fs::write(root.join("tracked.txt"), "newer edit after rollback\n").unwrap();
        assert!(undo(&receipt)
            .await
            .unwrap_err()
            .to_string()
            .contains("File changed after rollback"));
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "newer edit after rollback\n"
        );
        assert_eq!(
            read_backup(&receipt.backup_path).unwrap().before().unwrap(),
            b"original edit\n"
        );
    }

    #[tokio::test]
    async fn preserves_crlf_missing_final_newlines_and_complete_content_deletion() {
        for (baseline, changed, autocrlf) in [
            ("first\r\nsecond", "changed\r\nsecond", false),
            ("no final newline", "updated\n", false),
            ("baseline\n", "updated without newline", false),
            ("first\nsecond\n", "", false),
            ("first\nsecond\n", "changed\r\nsecond\r\n", true),
        ] {
            let directory = fixture(&[("tracked.txt", baseline)]);
            let root = directory.path();
            if autocrlf {
                git_ok(root, &["config", "core.autocrlf", "true"]);
            }
            fs::write(root.join("tracked.txt"), changed).unwrap();
            let selected = selection(root, "tracked.txt").await;
            let receipt = apply(prepare(root, &selected, None).await.unwrap())
                .await
                .unwrap();
            let expected = if autocrlf {
                baseline.replace('\n', "\r\n")
            } else {
                baseline.into()
            };
            assert_eq!(
                fs::read_to_string(root.join("tracked.txt")).unwrap(),
                expected
            );
            undo(&receipt).await.unwrap();
            assert_eq!(
                fs::read_to_string(root.join("tracked.txt")).unwrap(),
                changed
            );
        }
    }

    #[tokio::test]
    async fn prepared_backup_recovers_after_interruption_but_never_auto_applies() {
        let directory = fixture(&[("tracked.txt", "baseline\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "original edit\n").unwrap();
        let receipt = apply(
            prepare(root, &selection(root, "tracked.txt").await, None)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let mut backup = read_backup(&receipt.backup_path).unwrap();
        backup.state = BackupState::Prepared; // interruption after file replacement
        publish_backup(&receipt.backup_path, &backup).unwrap();
        let recovered = latest(root).await.unwrap().unwrap();
        undo(&recovered).await.unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "original edit\n"
        );
        backup.state = BackupState::Prepared; // interruption before file replacement
        publish_backup(&receipt.backup_path, &backup).unwrap();
        assert!(latest(root).await.unwrap().is_none());
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            "original edit\n"
        );
    }

    #[tokio::test]
    async fn failed_backup_finalization_restores_original_bytes_and_keeps_durable_copies() {
        for failure_after_publication in [false, true] {
            let directory = fixture(&[("tracked.txt", "baseline\n")]);
            let root = directory.path();
            fs::write(root.join("tracked.txt"), "original work\n").unwrap();
            let plan = prepare(root, &selection(root, "tracked.txt").await, None)
                .await
                .unwrap();
            let index = plan.index.clone();
            let index_path = plan.index_path.clone();
            let error = apply_with_backup(plan, |path, backup| {
                if backup.state == BackupState::Applied {
                    if failure_after_publication {
                        publish_backup(path, backup)?;
                    }
                    bail!("Injected final publication failure");
                }
                publish_backup(path, backup)
            })
            .unwrap_err();
            assert!(error.to_string().contains("Rollback did not complete"));
            assert_eq!(
                fs::read_to_string(root.join("tracked.txt")).unwrap(),
                "original work\n"
            );
            assert_eq!(fs::read(&index_path).unwrap(), index);
            assert!(!root.join(".git/index.lock").exists());
            assert!(latest(root).await.unwrap().is_none());
            let directory = backup_directory(&root.canonicalize().unwrap(), false)
                .unwrap()
                .unwrap();
            let records = inventory(&directory).unwrap().records;
            assert_eq!(records.len(), 1);
            let backup = read_backup(records.values().next().unwrap()).unwrap();
            assert_eq!(backup.before().unwrap(), b"original work\n");
            assert_eq!(backup.after().unwrap(), b"baseline\n");
        }
    }

    #[tokio::test]
    async fn failed_undo_finalization_stays_reviewable_before_and_after_metadata_publication() {
        for failure_after_publication in [false, true] {
            let directory = fixture(&[("tracked.txt", "baseline\n")]);
            let root = directory.path();
            fs::write(root.join("tracked.txt"), "original edit\n").unwrap();
            let receipt = apply(
                prepare(root, &selection(root, "tracked.txt").await, None)
                    .await
                    .unwrap(),
            )
            .await
            .unwrap();
            let index_path = index_path(&receipt.root).await.unwrap();
            let index = fs::read(&index_path).unwrap();
            let error = undo_with_backup(&receipt, &index_path, |path, backup| {
                if backup.state == BackupState::Undone {
                    if failure_after_publication {
                        publish_backup(path, backup)?;
                    }
                    bail!("Injected undo finalization failure");
                }
                publish_backup(path, backup)
            })
            .unwrap_err();
            assert!(format!("{error:#}").contains("Injected undo finalization failure"));
            assert_eq!(fs::read(root.join("tracked.txt")).unwrap(), b"baseline\n");
            assert_eq!(fs::read(&index_path).unwrap(), index);
            let recovered = latest(root).await.unwrap().unwrap();
            assert_eq!(recovered.backup_path, receipt.backup_path);
            assert_eq!(recovered.preview, receipt.preview);
            assert!(recovered.preview.contains("-baseline"));
            assert!(recovered.preview.contains("+original edit"));
            if failure_after_publication {
                assert!(recovered
                    .warning
                    .as_deref()
                    .unwrap()
                    .contains("Recovered undo"));
            }
            undo(&recovered).await.unwrap();
            assert_eq!(
                fs::read(root.join("tracked.txt")).unwrap(),
                b"original edit\n"
            );
            assert!(latest(root).await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn unavailable_or_newly_edited_newest_target_does_not_hide_an_older_undo() {
        let directory = fixture(&[
            ("older.txt", "older baseline\n"),
            ("newer.txt", "newer baseline\n"),
        ]);
        let root = directory.path();
        let mut receipts = Vec::new();
        for path in ["older.txt", "newer.txt"] {
            fs::write(root.join(path), "saved edit\n").unwrap();
            receipts.push(
                apply(
                    prepare(root, &selection(root, path).await, None)
                        .await
                        .unwrap(),
                )
                .await
                .unwrap(),
            );
        }
        fs::rename(root.join("newer.txt"), root.join("moved.txt")).unwrap();
        let recovered = latest(root).await.unwrap().unwrap();
        assert_eq!(recovered.backup_path, receipts[0].backup_path);
        assert!(recovered
            .warning
            .as_deref()
            .unwrap()
            .contains("Skipped newer.txt"));
        fs::rename(root.join("moved.txt"), root.join("newer.txt")).unwrap();
        fs::write(root.join("newer.txt"), "newer work after rollback\n").unwrap();
        let recovered = latest(root).await.unwrap().unwrap();
        assert_eq!(recovered.backup_path, receipts[0].backup_path);
        assert!(recovered
            .warning
            .as_deref()
            .unwrap()
            .contains("newer edits"));
        undo(&recovered).await.unwrap();
        assert_eq!(fs::read(root.join("older.txt")).unwrap(), b"saved edit\n");
        assert_eq!(
            fs::read(root.join("newer.txt")).unwrap(),
            b"newer work after rollback\n"
        );
        assert!(latest(root)
            .await
            .unwrap_err()
            .to_string()
            .contains("Skipped newer.txt"));
    }

    #[tokio::test]
    async fn already_undone_history_does_not_report_conflicts_after_unwinding_multiple_rollbacks() {
        let original: String = (0..24).map(|number| format!("line {number}\n")).collect();
        let directory = fixture(&[("tracked.txt", &original)]);
        let root = directory.path();
        let edited = original
            .replace("line 2\n", "first edit\n")
            .replace("line 20\n", "second edit\n");
        fs::write(root.join("tracked.txt"), &edited).unwrap();
        let first = apply(
            prepare(root, &selection(root, "tracked.txt").await, Some(0))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let second = apply(
            prepare(root, &selection(root, "tracked.txt").await, None)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        undo(&second).await.unwrap();
        let pending = latest(root).await.unwrap().unwrap();
        assert_eq!(pending.backup_path, first.backup_path);
        undo(&pending).await.unwrap();
        assert_eq!(
            fs::read_to_string(root.join("tracked.txt")).unwrap(),
            edited
        );
        assert!(latest(root).await.unwrap().is_none());
        fs::rename(root.join("tracked.txt"), root.join("moved.txt")).unwrap();
        assert!(latest(root).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn undo_is_bound_to_the_saved_bytes_that_were_reviewed() {
        let directory = fixture(&[("tracked.txt", "baseline\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "reviewed edit\n").unwrap();
        let receipt = apply(
            prepare(root, &selection(root, "tracked.txt").await, None)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let mut backup = read_backup(&receipt.backup_path).unwrap();
        backup.before = STANDARD.encode("different unreviewed contents\n");
        backup.undo_preview = legacy_undo_preview(
            &backup.path,
            &backup.before().unwrap(),
            &backup.after().unwrap(),
        )
        .unwrap();
        publish_backup(&receipt.backup_path, &backup).unwrap();
        assert!(undo(&receipt)
            .await
            .unwrap_err()
            .to_string()
            .contains("changed after review"));
        assert_eq!(fs::read(root.join("tracked.txt")).unwrap(), b"baseline\n");
    }

    #[tokio::test]
    async fn preparation_binds_the_selection_even_if_a_display_field_is_changed() {
        let directory = fixture(&[("one.txt", "baseline\n"), ("two.txt", "baseline\n")]);
        let root = directory.path();
        for path in ["one.txt", "two.txt"] {
            fs::write(root.join(path), "same edited bytes\n").unwrap();
        }
        let mut plan = prepare(root, &selection(root, "one.txt").await, None)
            .await
            .unwrap();
        plan.path = "two.txt".into();
        assert!(apply(plan)
            .await
            .unwrap_err()
            .to_string()
            .contains("selection changed"));
        assert_eq!(
            fs::read_to_string(root.join("one.txt")).unwrap(),
            "same edited bytes\n"
        );
        assert_eq!(
            fs::read_to_string(root.join("two.txt")).unwrap(),
            "same edited bytes\n"
        );
    }

    #[tokio::test]
    async fn rejects_staged_files_binary_data_hard_links_and_unsafe_paths() {
        let directory = fixture(&[("tracked.txt", "baseline\n")]);
        let root = directory.path();
        fs::write(root.join("tracked.txt"), "staged\n").unwrap();
        git_ok(root, &["add", "--", "tracked.txt"]);
        fs::write(root.join("tracked.txt"), "unstaged on top\n").unwrap();
        assert!(prepare(root, &selection(root, "tracked.txt").await, None)
            .await
            .is_err());
        let mut selected = selection(root, "tracked.txt").await;
        for stage in [ChangeStage::Staged, ChangeStage::Untracked] {
            selected.stage = stage;
            assert!(hunk_choices(&selected).is_err());
        }
        for path in [
            "../escape",
            ".git/config",
            "a/.git/index",
            ".",
            "/absolute",
            "file:stream",
            ".aster/tui-drafts/new.json",
        ] {
            assert!(safe_relative(path).is_err());
        }
        fs::hard_link(root.join("tracked.txt"), root.join("other-link.txt")).unwrap();
        assert!(checked_file(&root.canonicalize().unwrap(), "tracked.txt")
            .unwrap_err()
            .to_string()
            .contains("hard links"));
        assert!(text_file(&[0, 1, 2]).is_err());
        assert!(text_file(&[255]).is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlinks_in_selected_file_or_parent_and_preserves_executable_permissions() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let directory = fixture(&[("nested/tracked.txt", "baseline\n")]);
        let root = directory.path();
        let external = tempfile::tempdir().unwrap();
        fs::write(external.path().join("outside.txt"), "outside").unwrap();
        symlink(external.path().join("outside.txt"), root.join("linked.txt")).unwrap();
        symlink(external.path(), root.join("linked-dir")).unwrap();
        let canonical = root.canonicalize().unwrap();
        assert!(checked_file(&canonical, "linked.txt").is_err());
        assert!(checked_file(&canonical, "linked-dir/outside.txt").is_err());
        fs::set_permissions(
            root.join("nested/tracked.txt"),
            Permissions::from_mode(0o755),
        )
        .unwrap();
        git_ok(root, &["add", "--", "nested/tracked.txt"]);
        git_ok(
            root,
            &["commit", "--quiet", "--no-verify", "-m", "mode fixture"],
        );
        fs::write(root.join("nested/tracked.txt"), "updated\n").unwrap();
        let receipt = apply(
            prepare(root, &selection(root, "nested/tracked.txt").await, None)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(
            fs::metadata(root.join("nested/tracked.txt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_eq!(
            fs::metadata(&receipt.backup_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        undo(&receipt).await.unwrap();
        assert_eq!(
            fs::metadata(root.join("nested/tracked.txt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }

    #[test]
    fn malformed_binary_mode_and_truncated_patches_have_no_hunk_choices() {
        for patch in [
            "diff --git a/f b/f\nBinary files a/f and b/f differ\n",
            "diff --git a/f b/f\nold mode 100644\nnew mode 100755\n",
            "diff --git a/f b/f\n@@ -1,2 +1,1 @@\n-old\n+new\n",
            "diff --git a/f b/f\n@@ -1 +1 @@\n-old\n+new\n[Patch truncated]\n",
        ] {
            assert!(parse_patch(patch).is_err());
        }
    }
}

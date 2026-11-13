// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bounded, immutable file snapshots and workspace-only file discovery.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::process_tree::ProcessTree;

pub const MAX_ATTACHMENTS: usize = 8;
pub const MAX_TEXT_FILE_BYTES: usize = 256 * 1024;
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const MAX_NAME_BYTES: usize = 1024;
const MAX_URI_BYTES: usize = 32 * 1024;
const MAX_LIST_BYTES: usize = 2 * 1024 * 1024;
const MAX_CANDIDATES: usize = 20_000;
const MAX_RESULTS: usize = 100;
const SEARCH_TIMEOUT: Duration = Duration::from_secs(3);

/// Owns the selected bytes, so queuing, restarting, and sending never reread a path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attachment {
    Text {
        name: String,
        uri: String,
        text: String,
    },
    Image {
        name: String,
        mime_type: String,
        data: String,
    },
}

impl Attachment {
    pub fn name(&self) -> &str {
        match self {
            Self::Text { name, .. } | Self::Image { name, .. } => name,
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    pub fn mime_type(&self) -> &str {
        match self {
            Self::Text { .. } => "text/plain",
            Self::Image { mime_type, .. } => mime_type,
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len(),
            Self::Image { data, .. } => (data.len() / 4 * 3)
                .saturating_sub(data.bytes().rev().take_while(|b| *b == b'=').count().min(2)),
        }
    }

    /// Validate restored data as well as new selections before building ACP blocks.
    pub fn validate(&self) -> Result<()> {
        if self.name().is_empty() || self.name().len() > MAX_NAME_BYTES {
            bail!("Attachment name is empty or too long");
        }
        match self {
            Self::Text { uri, text, .. } => {
                if !uri.starts_with("file://") || uri.len() > MAX_URI_BYTES {
                    bail!("Text attachment must have a bounded file URI");
                }
                if text.len() > MAX_TEXT_FILE_BYTES || text.contains('\0') {
                    bail!("Text attachments must be UTF-8 text no larger than 256 KiB");
                }
            }
            Self::Image {
                mime_type, data, ..
            } => {
                if data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
                    bail!("Image attachment exceeds the 8 MiB limit");
                }
                let bytes = STANDARD
                    .decode(data)
                    .context("Invalid saved image encoding")?;
                if bytes.len() > MAX_IMAGE_BYTES || image_mime(&bytes) != Some(mime_type.as_str()) {
                    bail!("Attachment is not a supported PNG, JPEG, GIF, or WebP image");
                }
            }
        }
        Ok(())
    }

    pub fn from_image(name: impl Into<String>, bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_IMAGE_BYTES {
            bail!("Image attachment exceeds the 8 MiB limit");
        }
        let mime_type = image_mime(bytes)
            .context("Clipboard or file does not contain a PNG, JPEG, GIF, or WebP image")?;
        let attachment = Self::Image {
            name: name.into(),
            mime_type: mime_type.into(),
            data: STANDARD.encode(bytes),
        };
        attachment.validate()?;
        Ok(attachment)
    }

    pub fn content_block(&self) -> Value {
        match self {
            Self::Text { uri, text, .. } => {
                json!({"type":"resource", "resource":{"uri":uri, "mimeType":"text/plain", "text":text}})
            }
            Self::Image {
                name,
                mime_type,
                data,
            } => {
                json!({"type":"image", "mimeType":mime_type, "data":data, "_meta":{"aster":{"name":name}}})
            }
        }
    }
}

pub fn validate(attachments: &[Attachment]) -> Result<()> {
    if attachments.len() > MAX_ATTACHMENTS {
        bail!("A message can contain at most 8 attachments");
    }
    let mut total = 0_usize;
    for attachment in attachments {
        attachment.validate()?;
        total = total.saturating_add(attachment.byte_len());
        if total > MAX_TOTAL_BYTES {
            bail!("Message attachments exceed the 16 MiB total limit");
        }
    }
    Ok(())
}

pub fn prompt(text: &str, attachments: &[Attachment]) -> Result<Vec<Value>> {
    validate(attachments)?;
    let mut blocks = Vec::with_capacity(attachments.len() + 1);
    if !text.is_empty() {
        blocks.push(json!({"type":"text", "text":text}));
    }
    blocks.extend(attachments.iter().map(Attachment::content_block));
    Ok(blocks)
}

/// Relative selections must stay in the workspace; an explicit absolute path
/// can intentionally select a regular file elsewhere.
pub async fn load_file(workspace: &Path, path: &Path) -> Result<Attachment> {
    let workspace = workspace.to_path_buf();
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || snapshot_file(&workspace, &path))
        .await
        .context("Attachment reader stopped")?
}

fn snapshot_file(workspace: &Path, path: &Path) -> Result<Attachment> {
    let root = workspace
        .canonicalize()
        .context("Could not locate workspace")?;
    let selected = if path.is_absolute() {
        path.to_path_buf()
    } else {
        if path
            .components()
            .any(|c| matches!(c, Component::Prefix(_) | Component::RootDir))
        {
            bail!("Use a workspace-relative path or a complete absolute path");
        }
        root.join(path)
    };
    if !fs::symlink_metadata(&selected)
        .context("Could not inspect attachment")?
        .file_type()
        .is_file()
    {
        bail!("Attachments must be regular files, not directories or symbolic links");
    }
    let resolved = selected
        .canonicalize()
        .context("Could not locate attachment")?;
    if !path.is_absolute() && !resolved.starts_with(&root) {
        bail!("Relative attachment path leaves the workspace; use an explicit absolute path");
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Do not follow a replaced symlink or block on a replaced FIFO.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file: File = options
        .open(&resolved)
        .context("Could not open attachment")?;
    if !file
        .metadata()
        .context("Could not inspect opened attachment")?
        .is_file()
    {
        bail!("Attachment is no longer a regular file");
    }
    let mut bytes = Vec::new();
    file.take(MAX_IMAGE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .context("Could not read attachment")?;
    if bytes.len() > MAX_IMAGE_BYTES {
        bail!("Attachment exceeds the 8 MiB limit");
    }
    let display_path = resolved
        .strip_prefix(&root)
        .unwrap_or_else(|_| selected.file_name().map(Path::new).unwrap_or(&selected));
    let name = display_path
        .to_str()
        .context("Attachment filename is not valid UTF-8")?
        .replace('\\', "/");
    if image_mime(&bytes).is_some() {
        return Attachment::from_image(name, &bytes);
    }
    if bytes.len() > MAX_TEXT_FILE_BYTES {
        bail!("Text attachments cannot exceed 256 KiB");
    }
    let text =
        String::from_utf8(bytes).context("Attachment is not UTF-8 text or a supported image")?;
    let attachment = Attachment::Text {
        name,
        uri: file_uri(&resolved)?,
        text,
    };
    attachment.validate()?;
    Ok(attachment)
}

fn image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn file_uri(path: &Path) -> Result<String> {
    let path = path
        .to_str()
        .context("Attachment path is not valid UTF-8")?;
    #[cfg(windows)]
    let path = path
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!("//{rest}"))
        .unwrap_or_else(|| path.strip_prefix(r"\\?\").unwrap_or(path).to_owned())
        .replace('\\', "/");
    #[cfg(not(windows))]
    let path = path.to_owned();
    let mut uri = if path.starts_with("//") {
        "file:".to_owned()
    } else if path.starts_with('/') {
        "file://".to_owned()
    } else {
        "file:///".to_owned()
    };
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || b"/:-._~".contains(&byte) {
            uri.push(char::from(byte));
        } else {
            use std::fmt::Write;
            write!(uri, "%{byte:02X}").expect("writing to String cannot fail");
        }
    }
    Ok(uri)
}

#[derive(Clone, Debug, Default)]
pub struct FileSearch {
    pub files: Vec<String>,
    pub truncated: bool,
}

/// Git's ignore-aware listing includes tracked and untracked files. Ripgrep is
/// the fallback for workspaces without Git; neither command reads file contents.
pub async fn search_files(workspace: &Path, query: &str) -> Result<FileSearch> {
    let mut git = Command::new("git");
    git.current_dir(workspace).args([
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        ".",
    ]);
    git.env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0");
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
    ] {
        git.env_remove(name);
    }
    let (bytes, truncated) = match list_files(&mut git).await {
        Ok(output) => output,
        Err(_) => {
            let mut rg = Command::new("rg");
            rg.current_dir(workspace).args([
                "--files",
                "--hidden",
                "-0",
                "-g",
                "!.git",
                "-g",
                "!.aster/tui-drafts/**",
                "-g",
                "!.aster/sessions/**",
            ]);
            list_files(&mut rg)
                .await
                .context("Could not search workspace files; Git or ripgrep must be available")?
        }
    };
    let workspace = workspace.to_path_buf();
    let query = query.to_owned();
    tokio::task::spawn_blocking(move || select_files(&workspace, &query, &bytes, truncated))
        .await
        .context("Workspace file search stopped")?
}

async fn list_files(command: &mut Command) -> Result<(Vec<u8>, bool)> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let (mut child, mut tree) =
        ProcessTree::spawn(command).context("Could not start file search")?;
    let stdout = child
        .stdout
        .take()
        .context("File search output is unavailable")?;
    let mut bytes = Vec::new();
    let result = timeout(SEARCH_TIMEOUT, async {
        stdout
            .take(MAX_LIST_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .await?;
        if bytes.len() > MAX_LIST_BYTES {
            return Ok(true);
        }
        let status = child.wait().await?;
        // Ripgrep uses exit 1 for an empty file set, including an empty
        // nongit workspace. Its actual invocation failures use exit 2.
        if !(status.success() || status.code() == Some(1) && bytes.is_empty()) {
            bail!("File search command failed");
        }
        Ok(false)
    })
    .await;
    tree.terminate_and_wait()
        .await
        .context("Could not clean up file search")?;
    timeout(SEARCH_TIMEOUT, child.wait())
        .await
        .context("File search did not stop")??;
    let truncated = result.context("Workspace file search timed out")??;
    if truncated {
        bytes.truncate(MAX_LIST_BYTES);
        // Only complete NUL-delimited paths can be selected from bounded output.
        bytes.truncate(bytes.iter().rposition(|b| *b == 0).map_or(0, |i| i + 1));
    }
    Ok((bytes, truncated))
}

fn select_files(
    workspace: &Path,
    query: &str,
    bytes: &[u8],
    mut truncated: bool,
) -> Result<FileSearch> {
    let root = workspace
        .canonicalize()
        .context("Could not locate workspace")?;
    let query = query.trim().replace('\\', "/").to_lowercase();
    let mut files = BTreeSet::new();
    let mut candidates = bytes.split(|b| *b == 0).filter(|b| !b.is_empty());
    for bytes in candidates.by_ref().take(MAX_CANDIDATES) {
        let Ok(name) = std::str::from_utf8(bytes) else {
            continue;
        };
        let name = name.replace('\\', "/");
        let path = PathBuf::from(&name);
        if path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::Prefix(_)))
            || name.starts_with(".git/")
            || name.starts_with(".aster/tui-drafts/")
            || name.starts_with(".aster/sessions/")
            || !name.to_lowercase().contains(&query)
        {
            continue;
        }
        let selected = root.join(&path);
        if !fs::symlink_metadata(&selected).is_ok_and(|m| m.file_type().is_file())
            || !selected.canonicalize().is_ok_and(|p| p.starts_with(&root))
        {
            continue;
        }
        files.insert(name);
    }
    truncated |= candidates.next().is_some();
    let mut files: Vec<_> = files.into_iter().collect();
    files.sort_by_key(|path| {
        let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();
        (!filename.starts_with(&query), path.len(), path.clone())
    });
    truncated |= files.len() > MAX_RESULTS;
    files.truncate(MAX_RESULTS);
    Ok(FileSearch { files, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn text_is_an_immutable_unicode_snapshot_with_an_encoded_uri() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("hello # 中文.txt");
        fs::write(&path, "  old 中文\r\n").unwrap();
        let attachment = load_file(workspace.path(), Path::new("hello # 中文.txt"))
            .await
            .unwrap();
        fs::write(&path, "new contents").unwrap();
        let blocks = prompt("read this", &[attachment]).unwrap();
        assert_eq!(blocks[0], json!({"type":"text","text":"read this"}));
        assert_eq!(blocks[1]["type"], "resource");
        assert_eq!(blocks[1]["resource"]["text"], "  old 中文\r\n");
        let uri = blocks[1]["resource"]["uri"].as_str().unwrap();
        assert!(uri.starts_with("file:///"));
        assert!(uri.contains("%20%23%20%E4%B8%AD%E6%96%87.txt"));
    }

    #[tokio::test]
    async fn rejects_binary_directories_oversized_text_and_relative_escape() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("binary"), [0, 1, 2]).unwrap();
        fs::write(
            workspace.path().join("large"),
            vec![b'a'; MAX_TEXT_FILE_BYTES + 1],
        )
        .unwrap();
        for name in ["binary", "large", "."] {
            assert!(load_file(workspace.path(), Path::new(name)).await.is_err());
        }
        fs::write(outside.path().join("selected.txt"), b"explicit selection").unwrap();
        let absolute = outside.path().join("selected.txt");
        assert!(load_file(workspace.path(), &absolute).await.is_ok());
        let relative = PathBuf::from("..")
            .join(outside.path().file_name().unwrap())
            .join("selected.txt");
        assert!(load_file(workspace.path(), &relative).await.is_err());
    }

    #[test]
    fn restored_images_are_validated_and_form_standard_acp_blocks() {
        let bytes = b"\x89PNG\r\n\x1a\nimage bytes";
        let attachment = Attachment::from_image("clipboard.png", bytes).unwrap();
        let encoded = serde_json::to_vec(&attachment).unwrap();
        let restored: Attachment = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(restored.byte_len(), bytes.len());
        assert_eq!(
            prompt("", &[restored]).unwrap(),
            vec![json!({
                "type":"image", "mimeType":"image/png", "data":STANDARD.encode(bytes),
                "_meta":{"aster":{"name":"clipboard.png"}},
            })]
        );
        let invalid = Attachment::Image {
            name: "bad.png".into(),
            mime_type: "image/png".into(),
            data: "invalid".into(),
        };
        assert!(prompt("", &[invalid]).is_err());
        assert!(validate(&vec![attachment; MAX_ATTACHMENTS + 1]).is_err());
    }

    #[test]
    fn file_search_is_bounded_deduplicated_and_scoped_to_regular_workspace_files() {
        let workspace = tempfile::tempdir().unwrap();
        fs::create_dir(workspace.path().join("src")).unwrap();
        fs::write(workspace.path().join("src/中文.rs"), "").unwrap();
        fs::write(workspace.path().join("README.md"), "").unwrap();
        fs::create_dir_all(workspace.path().join(".aster/tui-drafts")).unwrap();
        fs::write(
            workspace.path().join(".aster/tui-drafts/new.json"),
            "private",
        )
        .unwrap();
        let search = select_files(
            workspace.path(),
            "RS",
            "README.md\0src/中文.rs\0src/中文.rs\0../other.rs\0.aster/tui-drafts/new.json\0"
                .as_bytes(),
            false,
        )
        .unwrap();
        assert_eq!(search.files, ["src/中文.rs"]);
        assert!(!search.truncated);
        let mut listing = String::new();
        for i in 0..MAX_RESULTS + 5 {
            let name = format!("{i}.txt");
            fs::write(workspace.path().join(&name), "").unwrap();
            listing.push_str(&name);
            listing.push('\0');
        }
        let search = select_files(workspace.path(), "", listing.as_bytes(), false).unwrap();
        assert_eq!(search.files.len(), MAX_RESULTS);
        assert!(search.truncated);
    }

    #[tokio::test]
    async fn git_search_respects_ignore_rules_without_needing_a_commit() {
        let workspace = tempfile::tempdir().unwrap();
        let status = std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(workspace.path())
            .status()
            .unwrap();
        assert!(status.success());
        fs::write(workspace.path().join(".gitignore"), "ignored/\n").unwrap();
        fs::create_dir(workspace.path().join("ignored")).unwrap();
        fs::write(workspace.path().join("ignored/secret.rs"), "private").unwrap();
        fs::write(workspace.path().join("visible.rs"), "fn main() {}").unwrap();
        let result = search_files(workspace.path(), ".rs").await.unwrap();
        assert_eq!(result.files, ["visible.rs"]);
        assert!(!result.truncated);
    }

    #[tokio::test]
    async fn empty_nongit_workspace_is_a_successful_empty_search() {
        let workspace = tempfile::tempdir().unwrap();
        let result = search_files(workspace.path(), "anything").await.unwrap();
        assert!(result.files.is_empty());
        assert!(!result.truncated);
    }
}

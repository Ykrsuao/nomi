// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Read-only, bounded snapshots of Git's index and working tree.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::future::join_all;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::process_tree::ProcessTree;

const MAX_FILES: usize = 500;
const MAX_STATUS_BYTES: usize = 1024 * 1024;
const MAX_PATCH_BYTES: usize = 256 * 1024;
const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(5);
const LOAD_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeStage {
    Staged,
    Unstaged,
    Untracked,
}

impl ChangeStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Staged => "Staged",
            Self::Unstaged => "Unstaged",
            Self::Untracked => "Untracked",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChangedFile {
    /// Repository-relative display name. Raw filename bytes are retained internally
    /// for Git arguments; sanitize control characters when rendering this label.
    pub path: String,
    pub original_path: Option<String>,
    pub status: String,
    pub stage: ChangeStage,
    pub patch: String,
}

#[derive(Clone, Debug)]
pub struct WorkspaceChanges {
    pub root: PathBuf,
    pub files: Vec<ChangedFile>,
    /// A visible explanation if the snapshot was bounded or has unavailable patches.
    pub notice: Option<String>,
}

struct StatusEntry {
    file: ChangedFile,
    path: PathBuf,
    original_path: Option<PathBuf>,
}

/// Runs in the background without taking the index lock or modifying repository state.
/// Staged and unstaged changes to one path are separate entries. The snapshot is
/// refreshed explicitly by callers; files may change between status and diff.
pub async fn load(workspace: impl AsRef<Path>) -> Result<WorkspaceChanges> {
    timeout(LOAD_TIMEOUT, load_inner(workspace.as_ref()))
        .await
        .map_err(|_| anyhow!("Reading Git changes timed out; try refreshing the changes view"))?
}

async fn load_inner(workspace: &Path) -> Result<WorkspaceChanges> {
    let mut root_command = git(workspace);
    root_command.args(["rev-parse", "--show-toplevel"]);
    let root_output = run(&mut root_command, 64 * 1024, PROCESS_TIMEOUT).await?;
    if !root_output.success() {
        let diagnostic = String::from_utf8_lossy(&root_output.stderr);
        if diagnostic.contains("not a git repository")
            || diagnostic.contains("must be run in a work tree")
        {
            bail!("This workspace is not inside a Git working tree");
        }
        bail!("Could not open Git working tree: {}", diagnostic.trim());
    }
    let root_bytes = root_output
        .stdout
        .strip_suffix(b"\n")
        .unwrap_or(&root_output.stdout);
    #[cfg(windows)]
    let root_bytes = root_bytes.strip_suffix(b"\r").unwrap_or(root_bytes);
    let root = path_from_bytes(root_bytes)?;
    let mut status_command = git(&root);
    status_command.args([
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--renames",
        "--ignore-submodules=none",
    ]);
    let status_output = run(&mut status_command, MAX_STATUS_BYTES, PROCESS_TIMEOUT).await?;
    if !status_output.success() && !status_output.truncated {
        bail!(
            "Could not read Git status: {}",
            String::from_utf8_lossy(&status_output.stderr).trim()
        );
    }
    let (mut files, mut bounded) = parse_status(&status_output.stdout)?;
    bounded |= status_output.truncated;
    let mut remaining = MAX_TOTAL_BYTES;
    let mut unavailable = false;
    for batch in files.chunks_mut(4) {
        if remaining == 0 {
            for entry in batch {
                entry.file.patch =
                    "Patch omitted: the changes view reached its 8 MiB preview limit.".into();
            }
            bounded = true;
            continue;
        }
        let patches = join_all(batch.iter().map(|entry| read_patch(&root, entry))).await;
        for (entry, result) in batch.iter_mut().zip(patches) {
            if remaining == 0 {
                entry.file.patch =
                    "Patch omitted: the changes view reached its 8 MiB preview limit.".into();
                bounded = true;
                continue;
            }
            let text = match result {
                Ok((text, truncated)) => {
                    bounded |= truncated;
                    text
                }
                Err(error) => {
                    unavailable = true;
                    format!("Could not read this patch: {error}. Refresh to try again.")
                }
            };
            let limit = MAX_PATCH_BYTES.min(remaining);
            let (patch, truncated) = bounded_text(text, limit);
            remaining = remaining.saturating_sub(patch.len());
            bounded |= truncated;
            entry.file.patch = patch;
        }
    }
    let notice = match (bounded, unavailable) {
        (true, true) => Some("Large changes are bounded to 500 entries, 256 KiB per patch and 8 MiB total. Some patches could not be read; refresh to retry.".into()),
        (true, false) => Some("Large changes are bounded to 500 entries, 256 KiB per patch and 8 MiB total. Omitted content is marked in the preview.".into()),
        (false, true) => Some("Some patches could not be read. Files may have changed during the scan; refresh to retry.".into()),
        (false, false) => None,
    };
    Ok(WorkspaceChanges {
        root,
        files: files.into_iter().map(|entry| entry.file).collect(),
        notice,
    })
}

fn git(workspace: &Path) -> Command {
    let mut command = Command::new("git");
    command.current_dir(workspace).args([
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
    ]);
    command
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C");
    // Scope reads to the requested workspace even when launched from a Git hook.
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
}

struct Output {
    status: Option<ExitStatus>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    truncated: bool,
}

impl Output {
    fn success(&self) -> bool {
        self.status.is_some_and(|status| status.success())
    }
}

async fn read_limited(
    reader: impl tokio::io::AsyncRead + Unpin,
    bytes: &mut Vec<u8>,
    limit: usize,
) -> Result<()> {
    reader.take(limit as u64 + 1).read_to_end(bytes).await?;
    if bytes.len() > limit {
        bail!("Git output exceeded the preview limit");
    }
    Ok(())
}

async fn run(command: &mut Command, limit: usize, deadline: Duration) -> Result<Output> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Git for Windows may launch its real worker from cmd/git.exe. Contain
    // descendants before any code runs so truncated reads and cancellation
    // cannot leave a worker holding mappings of files in the working tree.
    let (mut child, mut process_tree) = ProcessTree::spawn(command)
        .context("Could not start Git; ensure Git is installed and on PATH")?;
    let stdout = child.stdout.take().context("Git stdout is unavailable")?;
    let stderr = child.stderr.take().context("Git stderr is unavailable")?;
    let mut out = Vec::new();
    let mut err = Vec::new();
    let result = timeout(deadline, async {
        tokio::try_join!(
            read_limited(stdout, &mut out, limit),
            read_limited(stderr, &mut err, 16 * 1024)
        )?;
        child.wait().await.context("Could not wait for Git")
    })
    .await;
    // Releasing only the direct child is insufficient, even if it has already
    // exited. Wait for the contained tree before returning a usable snapshot.
    let cleanup = process_tree.terminate_and_wait().await;
    let reaped = timeout(PROCESS_TIMEOUT, child.wait()).await;
    cleanup.context("Could not clean up the Git process tree")?;
    reaped
        .context("Git did not exit after its process tree was terminated")?
        .context("Could not reap Git after cleanup")?;
    match result {
        Ok(Ok(status)) => Ok(Output {
            status: Some(status),
            stdout: out,
            stderr: err,
            truncated: false,
        }),
        other => {
            if out.len() > limit {
                out.truncate(limit);
                return Ok(Output {
                    status: None,
                    stdout: out,
                    stderr: err,
                    truncated: true,
                });
            }
            match other {
                Err(_) => bail!("Git command timed out"),
                Ok(Err(error)) => Err(error),
                Ok(Ok(_)) => unreachable!(),
            }
        }
    }
}

fn path_from_bytes(bytes: &[u8]) -> Result<PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Ok(std::ffi::OsString::from_vec(bytes.to_vec()).into())
    }
    #[cfg(not(unix))]
    Ok(PathBuf::from(std::str::from_utf8(bytes).context(
        "Git returned a filename that is not valid UTF-8",
    )?))
}

fn parse_status(bytes: &[u8]) -> Result<(Vec<StatusEntry>, bool)> {
    let mut fields = bytes.split_inclusive(|byte| *byte == 0);
    let mut files = Vec::new();
    let mut bounded = false;
    while let Some(field) = fields.next() {
        if field.last() != Some(&0) {
            bounded = true;
            break;
        }
        let field = &field[..field.len() - 1];
        if field.len() < 4 || field[2] != b' ' {
            bail!("Git returned malformed status data");
        }
        let path = path_from_bytes(&field[3..])?;
        let original_path = if matches!(field[0], b'R' | b'C') || matches!(field[1], b'R' | b'C') {
            let Some(original) = fields.next().filter(|part| part.last() == Some(&0)) else {
                bounded = true;
                break;
            };
            Some(path_from_bytes(&original[..original.len() - 1])?)
        } else {
            None
        };
        let mut add = |stage, status: &str| {
            if files.len() < MAX_FILES {
                files.push(StatusEntry {
                    path: path.clone(),
                    original_path: original_path.clone(),
                    file: ChangedFile {
                        path: path.to_string_lossy().into_owned(),
                        original_path: original_path
                            .as_ref()
                            .map(|path| path.to_string_lossy().into_owned()),
                        stage,
                        status: status.into(),
                        patch: String::new(),
                    },
                });
            } else {
                bounded = true;
            }
        };
        if &field[..2] == b"??" {
            add(ChangeStage::Untracked, "Added");
        } else if field[..2].contains(&b'U') || matches!(&field[..2], b"AA" | b"DD") {
            add(ChangeStage::Unstaged, "Conflicted");
        } else {
            for (code, stage) in [
                (field[0], ChangeStage::Staged),
                (field[1], ChangeStage::Unstaged),
            ] {
                let status = match code {
                    b' ' | b'!' => continue,
                    b'A' => "Added",
                    b'D' => "Deleted",
                    b'R' => "Renamed",
                    b'C' => "Copied",
                    b'T' => "Type changed",
                    _ => "Modified",
                };
                add(stage, status);
            }
        }
        if files.len() == MAX_FILES && fields.clone().next().is_some() {
            bounded = true;
            break;
        }
    }
    Ok((files, bounded))
}

async fn read_patch(root: &Path, entry: &StatusEntry) -> Result<(String, bool)> {
    if entry.file.stage == ChangeStage::Untracked {
        let root = root.to_path_buf();
        let path = entry.path.clone();
        return tokio::task::spawn_blocking(move || untracked_patch(&root, &path))
            .await
            .context("File preview worker stopped")?;
    }
    let mut command = git(root);
    command.args([
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-relative",
        "--find-renames",
        "--submodule=short",
        "--unified=3",
    ]);
    if entry.file.stage == ChangeStage::Staged {
        command.arg("--cached");
    }
    command.arg("--").arg(&entry.path);
    if let Some(original) = &entry.original_path {
        command.arg(original);
    }
    let output = run(&mut command, MAX_PATCH_BYTES, PROCESS_TIMEOUT).await?;
    if !output.success() && !output.truncated {
        bail!(
            "Git diff failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    if output.truncated {
        text.push_str("\n[Patch truncated at 256 KiB.]\n");
    }
    if text.is_empty() {
        text =
            "No text patch available. The file may have changed since the scan; refresh to update."
                .into();
    }
    Ok((text, output.truncated))
}

fn untracked_patch(root: &Path, path: &Path) -> Result<(String, bool)> {
    let absolute = root.join(path);
    // Inspect link contents, never their target. Also reject a parent directory
    // that was replaced by a link outside the repository between status and read.
    let parent = absolute
        .parent()
        .context("Untracked file has no parent")?
        .canonicalize()?;
    if !parent.starts_with(root.canonicalize()?) {
        bail!("Untracked path resolves outside the Git working tree");
    }
    let metadata =
        std::fs::symlink_metadata(&absolute).context("Could not inspect untracked file")?;
    let (bytes, mode) = if metadata.file_type().is_symlink() {
        (
            std::fs::read_link(&absolute)?
                .to_string_lossy()
                .as_bytes()
                .to_vec(),
            "120000",
        )
    } else if metadata.is_file() {
        let mut bytes = Vec::new();
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
        }
        let file = options.open(&absolute)?;
        let opened = file.metadata()?;
        if !opened.is_file() || opened.file_type().is_symlink() {
            bail!("Untracked file changed during the scan");
        }
        file.take(MAX_PATCH_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        #[cfg(unix)]
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            if opened.permissions().mode() & 0o111 != 0 {
                "100755"
            } else {
                "100644"
            }
        };
        #[cfg(not(unix))]
        let mode = "100644";
        (bytes, mode)
    } else {
        return Ok((
            "Untracked directory or special file. Its contents are not expanded in this preview."
                .into(),
            false,
        ));
    };
    if bytes.len() > MAX_PATCH_BYTES {
        return Ok(("Untracked file exceeds the 256 KiB preview limit. Open the file directly to inspect it.".into(), true));
    }
    if bytes.contains(&0) || std::str::from_utf8(&bytes).is_err() {
        return Ok((
            format!(
                "Binary or non-UTF-8 file: {}\nText preview unavailable.",
                path.display()
            ),
            false,
        ));
    }
    let text = std::str::from_utf8(&bytes).expect("UTF-8 checked");
    let a_path =
        serde_json::to_string(&format!("a/{}", path.display())).expect("strings serialize");
    let b_path =
        serde_json::to_string(&format!("b/{}", path.display())).expect("strings serialize");
    let mut patch = format!(
        "diff --git {a_path} {b_path}\nnew file mode {mode}\n--- /dev/null\n+++ {b_path}\n"
    );
    let count = text.lines().count();
    if count > 0 {
        patch.push_str(&format!("@@ -0,0 +1,{count} @@\n"));
    }
    for line in text.split_inclusive('\n') {
        patch.push('+');
        patch.push_str(line);
    }
    if !text.is_empty() && !text.ends_with('\n') {
        patch.push_str("\n\\ No newline at end of file\n");
    }
    Ok((patch, false))
}

fn bounded_text(mut text: String, limit: usize) -> (String, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    let marker = "\n[Patch preview truncated.]\n";
    let mut end = limit.saturating_sub(marker.len()).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    if limit >= marker.len() {
        text.push_str(marker);
    }
    (text, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        git_ok(directory.path(), &["init", "--quiet"]);
        git_ok(directory.path(), &["config", "user.name", "Test"]);
        git_ok(
            directory.path(),
            &["config", "user.email", "test@example.invalid"],
        );
        git_ok(directory.path(), &["config", "core.autocrlf", "false"]);
        directory
    }

    fn git_ok(root: &Path, args: &[&str]) {
        let result = std::process::Command::new("git")
            .args(["-c", "core.hooksPath=", "-c", "commit.gpgsign=false"])
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    fn write(root: &Path, path: &str, text: impl AsRef<[u8]>) {
        std::fs::write(root.join(path), text).unwrap();
    }

    fn test_python() -> PathBuf {
        let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        std::env::var_os("ASTER_TEST_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                repository_root.join(if cfg!(windows) {
                    ".venv/Scripts/python.exe"
                } else {
                    ".venv/bin/python"
                })
            })
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn truncated_reader_releases_descendant_file_mappings_before_returning() {
        let repository = tempfile::tempdir().unwrap();
        let path = repository.path().join("mapped.txt");
        std::fs::write(&path, b"mapping content").unwrap();
        let child_script = concat!(
            "import mmap,os,sys,time\n",
            "f=open(sys.argv[1], 'r+b')\n",
            "m=mmap.mmap(f.fileno(), 0)\n",
            "try:\n",
            "    os.write(1, b'x' * 1000000)\n",
            "except BrokenPipeError:\n",
            "    pass\n",
            "time.sleep(60)\n",
        );
        let parent_script = "import subprocess,sys; p=subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); print(p.pid, file=sys.stderr, flush=True); p.wait()";
        let mut command = Command::new(test_python());
        command
            .arg("-c")
            .arg(parent_script)
            .arg(&path)
            .arg(child_script);
        let result = run(&mut command, 100, Duration::from_secs(3))
            .await
            .unwrap();
        assert!(result.truncated);
        let replaced = std::fs::write(&path, "replaced");
        // Only this test's fixture child is cleaned up if the regression occurs.
        // Keep the assertion after cleanup so a red test leaves no worker behind.
        if let Some(pid) = replaced.as_ref().err().and_then(|_| {
            String::from_utf8_lossy(&result.stderr)
                .trim()
                .parse::<u32>()
                .ok()
        }) {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Threading::{
                OpenProcess, TerminateProcess, WaitForSingleObject, PROCESS_TERMINATE,
            };
            // SAFETY: this is the still-live fixture worker reported by our
            // wrapper; close the owned process handle after bounded cleanup.
            unsafe {
                let handle = OpenProcess(PROCESS_TERMINATE | 0x0010_0000, 0, pid);
                if !handle.is_null() {
                    TerminateProcess(handle, 1);
                    WaitForSingleObject(handle, 3000);
                    CloseHandle(handle);
                }
            }
        }
        assert!(
            replaced.is_ok(),
            "Mapped descendant survived truncated read: {replaced:?}"
        );
    }

    #[tokio::test]
    async fn separates_index_and_worktree_and_reads_literal_unicode_paths_without_writing_index() {
        let repository = fixture();
        let root = repository.path();
        write(root, "tracked [1] 中文.txt", "before\n");
        write(root, "old name.txt", "rename me\n");
        write(root, "deleted.txt", "remove me\n");
        write(root, ".gitignore", "ignored.txt\n");
        write(root, ".gitattributes", "*.txt diff=blocked\n");
        git_ok(root, &["add", "."]);
        git_ok(root, &["commit", "--quiet", "-m", "baseline"]);
        git_ok(
            root,
            &["config", "diff.external", "ASTER_FORBIDDEN_EXTERNAL_DIFF"],
        );
        git_ok(
            root,
            &[
                "config",
                "diff.blocked.textconv",
                "ASTER_FORBIDDEN_TEXTCONV",
            ],
        );
        git_ok(
            root,
            &["config", "core.fsmonitor", "ASTER_FORBIDDEN_FSMONITOR"],
        );
        write(root, "tracked [1] 中文.txt", "staged\n");
        git_ok(
            root,
            &["-c", "core.fsmonitor=false", "add", "tracked [1] 中文.txt"],
        );
        write(root, "tracked [1] 中文.txt", "working tree\n");
        git_ok(
            root,
            &[
                "-c",
                "core.fsmonitor=false",
                "mv",
                "old name.txt",
                "renamed 中文.txt",
            ],
        );
        std::fs::remove_file(root.join("deleted.txt")).unwrap();
        write(root, "untracked 中文.txt", "  新行\nline two");
        write(root, "binary.bin", b"\x00\x01\x02");
        write(root, "ignored.txt", "ignored sentinel");
        let index_before = std::fs::read(root.join(".git/index")).unwrap();
        let snapshot = load(root).await.unwrap();
        assert!(snapshot.notice.is_none(), "{:?}", snapshot.notice);
        assert_eq!(snapshot.files.len(), 6);
        let staged = snapshot
            .files
            .iter()
            .find(|file| file.path == "tracked [1] 中文.txt" && file.stage == ChangeStage::Staged)
            .unwrap();
        assert!(staged.patch.contains("-before") && staged.patch.contains("+staged"));
        let unstaged = snapshot
            .files
            .iter()
            .find(|file| file.path == staged.path && file.stage == ChangeStage::Unstaged)
            .unwrap();
        assert!(unstaged.patch.contains("-staged") && unstaged.patch.contains("+working tree"));
        let renamed = snapshot
            .files
            .iter()
            .find(|file| file.path == "renamed 中文.txt")
            .unwrap();
        assert_eq!(renamed.status, "Renamed");
        assert_eq!(renamed.original_path.as_deref(), Some("old name.txt"));
        assert!(renamed.patch.contains("rename from old name.txt"));
        let deleted = snapshot
            .files
            .iter()
            .find(|file| file.path == "deleted.txt")
            .unwrap();
        assert_eq!(deleted.status, "Deleted");
        assert!(deleted.patch.contains("-remove me"));
        let untracked = snapshot
            .files
            .iter()
            .find(|file| file.path == "untracked 中文.txt")
            .unwrap();
        assert!(untracked
            .patch
            .contains("+  新行\n+line two\n\\ No newline"));
        assert!(snapshot
            .files
            .iter()
            .any(|file| file.path == "binary.bin" && file.patch.contains("Binary")));
        assert!(!snapshot.files.iter().any(|file| file.path == "ignored.txt"));
        assert_eq!(
            std::fs::read(root.join(".git/index")).unwrap(),
            index_before
        );
    }

    #[tokio::test]
    async fn supports_unborn_index_clean_tree_nested_workspace_and_non_repository() {
        let plain = tempfile::tempdir().unwrap();
        assert!(load(plain.path())
            .await
            .unwrap_err()
            .to_string()
            .contains("not inside a Git"));
        let repository = fixture();
        assert!(load(repository.path()).await.unwrap().files.is_empty());
        write(repository.path(), "first.txt", "first staged content\n");
        git_ok(repository.path(), &["add", "first.txt"]);
        std::fs::create_dir(repository.path().join("nested")).unwrap();
        let snapshot = load(repository.path().join("nested")).await.unwrap();
        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].stage, ChangeStage::Staged);
        assert!(snapshot.files[0].patch.contains("+first staged content"));
        assert_eq!(
            snapshot.root.canonicalize().unwrap(),
            repository.path().canonicalize().unwrap()
        );
    }

    #[tokio::test]
    async fn bounds_patch_output_and_untracked_content() {
        let repository = fixture();
        write(repository.path(), "large.txt", "baseline\n");
        git_ok(repository.path(), &["add", "large.txt"]);
        git_ok(repository.path(), &["commit", "--quiet", "-m", "baseline"]);
        write(
            repository.path(),
            "large.txt",
            "modified line\n".repeat(MAX_PATCH_BYTES / 5),
        );
        write(
            repository.path(),
            "large new.txt",
            "new line\n".repeat(MAX_PATCH_BYTES / 5),
        );
        let snapshot = load(repository.path()).await.unwrap();
        assert!(snapshot.notice.is_some());
        assert!(snapshot
            .files
            .iter()
            .all(|file| file.patch.len() <= MAX_PATCH_BYTES));
        assert!(snapshot
            .files
            .iter()
            .find(|file| file.path == "large.txt")
            .unwrap()
            .patch
            .contains("truncated"));
        assert!(snapshot
            .files
            .iter()
            .find(|file| file.path == "large new.txt")
            .unwrap()
            .patch
            .contains("preview limit"));
        write(repository.path(), "large.txt", "baseline\n");
        let only_untracked = load(repository.path()).await.unwrap();
        assert_eq!(only_untracked.files.len(), 1);
        assert!(only_untracked.notice.is_some());
        assert_eq!(only_untracked.files[0].stage, ChangeStage::Untracked);
    }

    #[test]
    fn porcelain_paths_keep_newlines_and_conflicts_do_not_duplicate_entries() {
        let (files, bounded) =
            parse_status(b"R  to\nname\0from\tname\0UU conflicted\0?? next\0").unwrap();
        assert!(!bounded);
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].file.path, "to\nname");
        assert_eq!(files[0].file.original_path.as_deref(), Some("from\tname"));
        assert_eq!(files[1].file.status, "Conflicted");
        assert_eq!(files[1].file.stage, ChangeStage::Unstaged);
        let (files, bounded) = parse_status(b" M complete\0R  incomplete\0original cut").unwrap();
        assert!(bounded);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn untracked_preview_rejects_paths_outside_the_working_tree() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("repository");
        std::fs::create_dir(&root).unwrap();
        write(directory.path(), "outside.txt", "OUTSIDE_CONTENT_SENTINEL");
        let error = untracked_patch(&root, Path::new("../outside.txt")).unwrap_err();
        assert!(error.to_string().contains("outside the Git working tree"));
        assert!(!error.to_string().contains("OUTSIDE_CONTENT_SENTINEL"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn raw_filename_bytes_and_symlinks_do_not_read_external_contents() {
        use std::os::unix::{ffi::OsStringExt, fs::symlink};
        let repository = fixture();
        let external = tempfile::tempdir().unwrap();
        write(external.path(), "private.txt", "OUTSIDE_CONTENT_SENTINEL");
        symlink(
            external.path().join("private.txt"),
            repository.path().join("link.txt"),
        )
        .unwrap();
        let raw_name = std::ffi::OsString::from_vec(b"raw\xff.txt".to_vec());
        std::fs::write(repository.path().join(raw_name), "raw filename text\n").unwrap();
        let snapshot = load(repository.path()).await.unwrap();
        assert_eq!(snapshot.files.len(), 2);
        assert!(snapshot
            .files
            .iter()
            .any(|entry| entry.patch.contains("+raw filename text")));
        let link = snapshot
            .files
            .iter()
            .find(|entry| entry.path == "link.txt")
            .unwrap();
        assert!(link.patch.contains("new file mode 120000"));
        assert!(!link.patch.contains("OUTSIDE_CONTENT_SENTINEL"));
        symlink(external.path(), repository.path().join("outside-dir")).unwrap();
        assert!(
            untracked_patch(repository.path(), Path::new("outside-dir/private.txt"))
                .unwrap_err()
                .to_string()
                .contains("outside the Git working tree")
        );
    }

    #[tokio::test]
    async fn bounded_process_reader_does_not_deadlock_on_full_stderr_and_timeout() {
        let python = test_python();
        let mut oversized = Command::new(&python);
        oversized.args(["-c", "import sys; sys.stdout.buffer.write(b'x' * 1000000); sys.stdout.flush(); sys.stderr.write('late')"]);
        let output = run(&mut oversized, 100, Duration::from_secs(2))
            .await
            .unwrap();
        assert!(output.truncated);
        assert_eq!(output.stdout.len(), 100);
        let mut noisy_error = Command::new(&python);
        noisy_error.args(["-c", "import sys; sys.stderr.buffer.write(b'x' * 1000000); sys.stderr.flush(); sys.stdout.write('late')"]);
        assert!(run(&mut noisy_error, 100, Duration::from_secs(2))
            .await
            .is_err());
        let mut stalled = Command::new(&python);
        stalled.args(["-c", "import time; time.sleep(60)"]);
        let started = std::time::Instant::now();
        assert!(run(&mut stalled, 100, Duration::from_millis(100))
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}

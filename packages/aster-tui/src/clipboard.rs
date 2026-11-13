// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Explicit clipboard access, isolated from the terminal and bounded in time and size.

use std::process::Stdio;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::timeout;

use crate::attachments::Attachment;

#[cfg(windows)]
mod windows;

const MAX_BYTES: usize = 8 * 1024 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(windows)]
const WORKER_FLAG: &str = "--aster-clipboard-worker";
#[cfg(windows)]
const CONTENT_WORKER_FLAG: &str = "--aster-clipboard-content-worker";
#[cfg(windows)]
const WRITE_WORKER_FLAG: &str = "--aster-clipboard-write-worker";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClipboardContent {
    Text(String),
    Image(Attachment),
}

/// A private single-purpose process keeps delayed Windows clipboard rendering
/// out of the UI process. It runs before normal CLI validation or terminal setup.
pub fn worker() -> Option<Result<()>> {
    #[cfg(windows)]
    if std::env::args_os()
        .skip(1)
        .eq([std::ffi::OsString::from(WORKER_FLAG)])
    {
        use std::io::Write;
        return Some(windows::read_text().and_then(|text| {
            std::io::stdout()
                .write_all(text.as_bytes())
                .context("Could not deliver clipboard text")
        }));
    }
    #[cfg(windows)]
    if std::env::args_os()
        .skip(1)
        .eq([std::ffi::OsString::from(CONTENT_WORKER_FLAG)])
    {
        return Some(
            windows::read().and_then(|content| write_content(&mut std::io::stdout(), content)),
        );
    }
    #[cfg(windows)]
    if std::env::args_os()
        .skip(1)
        .eq([std::ffi::OsString::from(WRITE_WORKER_FLAG)])
    {
        return Some(
            read_write_payload(std::io::stdin()).and_then(|text| windows::write_text(&text)),
        );
    }
    None
}

fn validate_write_text(text: &str) -> Result<()> {
    if text.len() > MAX_BYTES {
        bail!("Clipboard text exceeds the 8 MiB limit");
    }
    if text.contains('\0') {
        bail!("Clipboard text cannot contain a NUL character");
    }
    #[cfg(windows)]
    if text.encode_utf16().count() + 1 > MAX_BYTES / 2 {
        bail!("Windows clipboard text exceeds its 8 MiB memory limit");
    }
    Ok(())
}

#[cfg(any(windows, test))]
fn read_write_payload(reader: impl std::io::Read) -> Result<String> {
    use std::io::Read;

    let mut bytes = Vec::new();
    reader
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .context("Could not read clipboard input")?;
    if bytes.len() > MAX_BYTES {
        bail!("Clipboard text exceeds the 8 MiB limit");
    }
    let text = String::from_utf8(bytes).context("Clipboard input is not valid UTF-8")?;
    validate_write_text(&text)?;
    Ok(text)
}

/// Copied text travels through stdin, never shell commands or process arguments.
pub async fn write_text(text: &str) -> Result<()> {
    validate_write_text(text)?;
    #[cfg(windows)]
    {
        let mut command =
            Command::new(std::env::current_exe().context("Could not locate clipboard helper")?);
        command.arg(WRITE_WORKER_FLAG).creation_flags(0x0800_0000);
        write_process(&mut command, text, READ_TIMEOUT).await
    }
    #[cfg(target_os = "macos")]
    {
        write_process(&mut Command::new("pbcopy"), text, READ_TIMEOUT).await
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let started = tokio::time::Instant::now();
        let mut candidates: Vec<(&str, &[&str])> = Vec::new();
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            candidates.push(("wl-copy", &["--type", "text/plain;charset=utf-8"]));
        }
        if std::env::var_os("DISPLAY").is_some() {
            candidates.push(("xclip", &["-selection", "clipboard", "-in"]));
            candidates.push(("xsel", &["--clipboard", "--input"]));
        }
        for (program, arguments) in candidates {
            let remaining = READ_TIMEOUT.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                bail!("Clipboard write timed out");
            }
            let mut command = Command::new(program);
            command.args(arguments);
            if write_process(&mut command, text, remaining).await.is_ok() {
                return Ok(());
            }
        }
        bail!("Clipboard unavailable. Install wl-clipboard for Wayland or xclip/xsel for X11")
    }
    #[cfg(not(any(windows, unix)))]
    {
        bail!("System clipboard is unavailable")
    }
}

async fn write_process(command: &mut Command, text: &str, deadline: Duration) -> Result<()> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("Could not start clipboard writer")?;
    let mut stdin = child
        .stdin
        .take()
        .context("Clipboard input is unavailable")?;
    let operation = async {
        stdin
            .write_all(text.as_bytes())
            .await
            .context("Could not write clipboard input")?;
        stdin
            .shutdown()
            .await
            .context("Could not finish clipboard input")?;
        drop(stdin);
        if !child
            .wait()
            .await
            .context("Could not wait for clipboard writer")?
            .success()
        {
            bail!("Clipboard is temporarily unavailable");
        }
        Ok(())
    };
    let result = timeout(deadline, operation)
        .await
        .unwrap_or_else(|_| Err(anyhow!("Clipboard write timed out")));
    if result.is_err() {
        // Explicit failure/timeout reaps the worker; cancellation uses kill_on_drop.
        let _ = timeout(Duration::from_secs(1), child.kill()).await;
    }
    result
}

/// Prefer a copied image to text supplied alongside it by browsers or editors.
/// Workers return a one-byte content tag followed by bounded raw bytes, avoiding
/// JSON/base64 expansion of large clipboard text.
pub async fn read() -> Result<ClipboardContent> {
    #[cfg(windows)]
    {
        let mut command =
            Command::new(std::env::current_exe().context("Could not locate clipboard helper")?);
        command.arg(CONTENT_WORKER_FLAG).creation_flags(0x0800_0000);
        let bytes = read_process_bytes(&mut command, READ_TIMEOUT, MAX_BYTES + 1).await?;
        decode_content(&bytes)
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let started = tokio::time::Instant::now();
        let mut candidates: Vec<(&str, &[&str])> = Vec::new();
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            candidates.push(("wl-paste", &["--no-newline", "--type", "image/png"]));
        }
        if std::env::var_os("DISPLAY").is_some() {
            candidates.push((
                "xclip",
                &["-selection", "clipboard", "-t", "image/png", "-out"],
            ));
        }
        for (program, arguments) in candidates {
            let remaining = READ_TIMEOUT.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                bail!("Clipboard read timed out");
            }
            let mut command = Command::new(program);
            command.args(arguments);
            if let Ok(bytes) = read_process_bytes(&mut command, remaining, MAX_BYTES).await {
                if !bytes.is_empty() {
                    return Ok(ClipboardContent::Image(Attachment::from_image(
                        "clipboard.png",
                        &bytes,
                    )?));
                }
            }
        }
        let remaining = READ_TIMEOUT.saturating_sub(started.elapsed());
        timeout(remaining, read_text())
            .await
            .context("Clipboard read timed out")?
            .map(ClipboardContent::Text)
    }
    #[cfg(any(target_os = "macos", not(any(windows, unix))))]
    {
        read_text().await.map(ClipboardContent::Text)
    }
}

#[cfg(any(windows, test))]
fn write_content(writer: &mut impl std::io::Write, content: ClipboardContent) -> Result<()> {
    use base64::Engine;
    match content {
        ClipboardContent::Text(text) => {
            if text.len() > MAX_BYTES {
                bail!("Clipboard text exceeds the 8 MiB limit");
            }
            writer.write_all(b"T")?;
            writer.write_all(text.as_bytes())?;
        }
        ClipboardContent::Image(attachment) => {
            attachment.validate()?;
            let Attachment::Image { data, .. } = attachment else {
                bail!("Clipboard image worker returned text content");
            };
            let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
            writer.write_all(b"I")?;
            writer.write_all(&bytes)?;
        }
    }
    Ok(())
}

#[cfg(any(windows, test))]
fn decode_content(bytes: &[u8]) -> Result<ClipboardContent> {
    let Some((kind, bytes)) = bytes.split_first() else {
        bail!("Clipboard worker returned no content");
    };
    if bytes.len() > MAX_BYTES {
        bail!("Clipboard content exceeds the 8 MiB limit");
    }
    match kind {
        b'T' => Ok(ClipboardContent::Text(
            String::from_utf8(bytes.to_vec()).context("Clipboard text is not valid UTF-8")?,
        )),
        b'I' => Ok(ClipboardContent::Image(Attachment::from_image(
            "clipboard.png",
            bytes,
        )?)),
        _ => bail!("Clipboard worker returned an unknown content format"),
    }
}

pub async fn read_text() -> Result<String> {
    #[cfg(windows)]
    {
        let mut command =
            Command::new(std::env::current_exe().context("Could not locate clipboard helper")?);
        command.arg(WORKER_FLAG).creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        read_process(&mut command, READ_TIMEOUT, MAX_BYTES).await
    }
    #[cfg(target_os = "macos")]
    {
        read_process(&mut Command::new("pbpaste"), READ_TIMEOUT, MAX_BYTES).await
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let started = tokio::time::Instant::now();
        let mut candidates: Vec<(&str, &[&str])> = Vec::new();
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            candidates.push(("wl-paste", &["--no-newline", "--type", "text"]));
        }
        if std::env::var_os("DISPLAY").is_some() {
            candidates.push(("xclip", &["-selection", "clipboard", "-out"]));
            candidates.push(("xsel", &["--clipboard", "--output"]));
        }
        for (program, arguments) in candidates {
            let remaining = READ_TIMEOUT.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                bail!("Clipboard read timed out");
            }
            let mut command = Command::new(program);
            command.args(arguments);
            if let Ok(text) = read_process(&mut command, remaining, MAX_BYTES).await {
                return Ok(text);
            }
        }
        bail!("Clipboard unavailable. Install wl-clipboard for Wayland or xclip/xsel for X11, or use your terminal's Paste command")
    }
    #[cfg(not(any(windows, unix)))]
    {
        bail!("System clipboard is unavailable; use your terminal's Paste command")
    }
}

async fn read_process(command: &mut Command, deadline: Duration, limit: usize) -> Result<String> {
    String::from_utf8(read_process_bytes(command, deadline, limit).await?)
        .context("Clipboard text is not valid UTF-8")
}

async fn read_process_bytes(
    command: &mut Command,
    deadline: Duration,
    limit: usize,
) -> Result<Vec<u8>> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("Could not start clipboard reader")?;
    let stdout = child
        .stdout
        .take()
        .context("Clipboard output is unavailable")?;
    let operation = async {
        let mut bytes = Vec::new();
        stdout
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .await
            .context("Could not read clipboard content")?;
        if bytes.len() > limit {
            bail!("Clipboard content exceeds its size limit");
        }
        if !child
            .wait()
            .await
            .context("Could not wait for clipboard reader")?
            .success()
        {
            // Helpers can print copied content in diagnostics. Never include their stderr.
            bail!("Clipboard contains no readable content or is temporarily unavailable");
        }
        Ok(bytes)
    };
    let result = timeout(deadline, operation)
        .await
        .unwrap_or_else(|_| Err(anyhow!("Clipboard read timed out")));
    if result.is_err() {
        // Reap failed or timed-out helpers. kill_on_drop is the cancellation fallback.
        let _ = timeout(Duration::from_secs(1), child.kill()).await;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn tagged_worker_protocol_preserves_text_and_image_snapshots() {
        for content in [
            ClipboardContent::Text("  中文\r\nsecond  \n".into()),
            ClipboardContent::Text(String::new()),
            ClipboardContent::Image(
                Attachment::from_image("clipboard.png", b"\x89PNG\r\n\x1a\npixels").unwrap(),
            ),
        ] {
            let mut bytes = Vec::new();
            write_content(&mut bytes, content.clone()).unwrap();
            assert_eq!(decode_content(&bytes).unwrap(), content);
        }
        for bytes in [b"".as_slice(), b"unknown", b"T\xff", b"Inot an image"] {
            assert!(decode_content(bytes).is_err());
        }
    }

    fn python(script: &str) -> Command {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let executable = std::env::var_os("ASTER_TEST_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                root.join(if cfg!(windows) {
                    ".venv/Scripts/python.exe"
                } else {
                    ".venv/bin/python"
                })
            });
        let mut command = Command::new(executable);
        command.args(["-c", script]);
        #[cfg(windows)]
        command.creation_flags(0x0800_0000);
        command
    }

    #[test]
    fn write_payload_is_bounded_utf8_without_nul() {
        let text = "  中文👩‍💻\nsecond\t \r\n";
        assert_eq!(read_write_payload(text.as_bytes()).unwrap(), text);
        assert_eq!(read_write_payload(&b""[..]).unwrap(), "");
        for bytes in [&b"invalid\xff"[..], &b"embedded\0nul"[..]] {
            assert!(read_write_payload(bytes).is_err());
        }
        assert!(read_write_payload(vec![b'x'; MAX_BYTES + 1].as_slice()).is_err());
    }

    #[tokio::test]
    async fn clipboard_writer_sends_exact_text_through_stdin() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("copied.txt");
        let mut command = python(
            "import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
        );
        command.arg(&path);
        let text = "  中文\n👩‍💻 literal $(whoami) `quotes`  \n";
        write_process(&mut command, text, READ_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), text);
    }

    #[tokio::test]
    async fn clipboard_writer_errors_do_not_expose_copied_text() {
        let mut command = python(
            "import sys; text = sys.stdin.read(); print(text); print(text, file=sys.stderr); sys.exit(1)",
        );
        let error = write_process(&mut command, "PRIVATE_COPY_SENTINEL", READ_TIMEOUT)
            .await
            .unwrap_err();
        assert!(!format!("{error:#}").contains("PRIVATE_COPY_SENTINEL"));
    }

    #[tokio::test]
    async fn clipboard_writer_times_out_while_stdin_is_blocked() {
        let started = tokio::time::Instant::now();
        let mut command = python("import time; time.sleep(60)");
        let result = write_process(
            &mut command,
            &"x".repeat(256 * 1024),
            Duration::from_millis(100),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn helper_preserves_unicode_newlines_and_trailing_spaces() {
        let text = read_process(
            &mut python(
                "import sys; sys.stdout.buffer.write('  中文\\r\\nsecond  \\n'.encode('utf-8'))",
            ),
            READ_TIMEOUT,
            MAX_BYTES,
        )
        .await
        .unwrap();
        assert_eq!(text, "  中文\r\nsecond  \n");
    }

    #[tokio::test]
    async fn helper_errors_do_not_expose_output_and_do_not_accept_oversized_text() {
        let error = read_process(
            &mut python(
                "import sys; print('PRIVATE_CLIPBOARD_SENTINEL', file=sys.stderr); sys.exit(1)",
            ),
            READ_TIMEOUT,
            MAX_BYTES,
        )
        .await
        .unwrap_err();
        assert!(!format!("{error:#}").contains("PRIVATE_CLIPBOARD_SENTINEL"));
        let error = read_process(
            &mut python("import sys; sys.stdout.buffer.write(b'x' * 33)"),
            READ_TIMEOUT,
            32,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("limit"));
    }

    #[tokio::test]
    async fn stalled_helper_times_out_without_blocking_the_executor() {
        let started = tokio::time::Instant::now();
        let result = read_process(
            &mut python("import time; time.sleep(60)"),
            Duration::from_millis(100),
            MAX_BYTES,
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}

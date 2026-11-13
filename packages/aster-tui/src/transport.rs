// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Own the Python ACP process independently of cancellable frontend requests.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::process_tree::ProcessTree;

const MAX_FRAME_BYTES: usize = 50 * 1024 * 1024;
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_DIAGNOSTIC_BYTES: usize = 16 * 1024;

#[derive(Default)]
struct DiagnosticTail {
    text: String,
    truncated: bool,
}

impl DiagnosticTail {
    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        if self.text.len() > MAX_DIAGNOSTIC_BYTES {
            let mut boundary = self.text.len() - MAX_DIAGNOSTIC_BYTES;
            while !self.text.is_char_boundary(boundary) {
                boundary += 1;
            }
            self.text.drain(..boundary);
            self.truncated = true;
        }
    }

    fn snapshot(&self) -> String {
        if self.truncated {
            format!("[Earlier backend diagnostics omitted]\n{}", self.text)
        } else {
            self.text.clone()
        }
    }
}

#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub python: PathBuf,
    pub workspace: PathBuf,
    pub model: String,
    pub client_type: Option<String>,
    pub sandbox: bool,
}

#[derive(Debug)]
pub enum Event {
    Message(Value),
    Diagnostic(String),
    Closed(String),
}

struct Outgoing {
    bytes: Vec<u8>,
}

pub struct Transport {
    outgoing: Option<mpsc::Sender<Outgoing>>,
    messages: mpsc::Receiver<Value>,
    diagnostics: mpsc::Receiver<String>,
    diagnostic_tail: Arc<Mutex<DiagnosticTail>>,
    dropped_diagnostics: Arc<AtomicUsize>,
    closed: watch::Receiver<Option<String>>,
    closed_delivered: bool,
    stop: Option<oneshot::Sender<()>>,
    supervisor: Option<JoinHandle<Result<()>>>,
    next_id: u64,
}

impl Transport {
    pub async fn spawn(config: &BackendConfig) -> Result<Self> {
        let mut command = Command::new(&config.python);
        command
            .args(["-u", "-m", "aster_acp", "--model", &config.model])
            .arg(if config.sandbox {
                "--sandbox"
            } else {
                "--no-sandbox"
            })
            .current_dir(&config.workspace)
            .env("PYTHONIOENCODING", "utf-8");
        if let Some(client_type) = &config.client_type {
            command.args(["--client-type", client_type]);
        }
        // There is no await between spawning the child and transferring its
        // ownership to the supervisor, so cancelled startup cannot leak it.
        Self::spawn_command(command, SHUTDOWN_GRACE)
    }

    fn spawn_command(mut command: Command, grace: Duration) -> Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let (mut child, process_tree) = ProcessTree::spawn(&mut command)?;
        let stdin = child.stdin.take().context("Backend stdin is unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("Backend stdout is unavailable")?;
        let stderr = child
            .stderr
            .take()
            .context("Backend stderr is unavailable")?;
        let (outgoing, outgoing_rx) = mpsc::channel(8);
        let (messages_tx, messages) = mpsc::channel(8);
        let (diagnostics_tx, diagnostics) = mpsc::channel(128);
        let dropped_diagnostics = Arc::new(AtomicUsize::new(0));
        let diagnostic_tail = Arc::new(Mutex::new(DiagnosticTail::default()));
        let (closed_tx, closed) = watch::channel(None);
        let (stop, stop_rx) = oneshot::channel();
        let (failure_tx, failure_rx) = mpsc::channel(2);
        let stdout_failure = failure_tx.clone();
        let stdout_task = tokio::spawn(async move {
            let reason = match read_messages(stdout, messages_tx).await {
                Ok(()) => "Aster backend closed its protocol output".to_owned(),
                Err(error) => error.to_string(),
            };
            let _ = stdout_failure.send(reason).await;
        });
        let stderr_task = tokio::spawn(read_diagnostics(
            stderr,
            diagnostics_tx,
            Arc::clone(&dropped_diagnostics),
            Arc::clone(&diagnostic_tail),
        ));
        let writer_task = tokio::spawn(write_messages(stdin, outgoing_rx, failure_tx));
        let supervisor = tokio::spawn(supervise(
            child,
            process_tree,
            stop_rx,
            failure_rx,
            closed_tx,
            writer_task,
            stdout_task,
            stderr_task,
            grace,
        ));
        Ok(Self {
            outgoing: Some(outgoing),
            messages,
            diagnostics,
            diagnostic_tail,
            dropped_diagnostics,
            closed,
            closed_delivered: false,
            stop: Some(stop),
            supervisor: Some(supervisor),
            next_id: 1,
        })
    }

    pub async fn request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .context("ACP request IDs exhausted")?;
        self.send(json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}))
            .await?;
        Ok(id)
    }

    pub async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(json!({"jsonrpc":"2.0", "method":method, "params":params}))
            .await
    }

    pub async fn reply(&mut self, id: Value, result: Value) -> Result<()> {
        self.send(json!({"jsonrpc":"2.0", "id":id, "result":result}))
            .await
    }

    pub async fn reject(&mut self, id: Value, code: i64, message: &str) -> Result<()> {
        self.send(json!({"jsonrpc":"2.0", "id":id, "error":{"code":code,"message":message}}))
            .await
    }

    async fn send(&mut self, value: Value) -> Result<()> {
        let sender = self.outgoing.as_ref().context("ACP backend is closed")?;
        let mut bytes = serde_json::to_vec(&value)?;
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(anyhow!("ACP request exceeds the 50 MiB frame limit"));
        }
        bytes.push(b'\n');
        // The owned writer completes entire frames in order. Enqueueing never
        // waits for the child to read a pipe, keeping the terminal responsive.
        sender
            .try_send(Outgoing { bytes })
            .map_err(|_| anyhow!("ACP request queue is full or the backend is closed"))
    }

    pub fn try_recv(&mut self) -> Option<Event> {
        if let Ok(message) = self.messages.try_recv() {
            return Some(Event::Message(message));
        }
        if let Ok(text) = self.diagnostics.try_recv() {
            return Some(Event::Diagnostic(text));
        }
        let dropped = self.dropped_diagnostics.swap(0, Ordering::Relaxed);
        if dropped != 0 {
            return Some(Event::Diagnostic(format!(
                "[{dropped} backend diagnostic chunks omitted while the display was busy]\n"
            )));
        }
        if !self.closed_delivered {
            if let Some(reason) = self.closed.borrow().clone() {
                self.closed_delivered = true;
                return Some(Event::Closed(reason));
            }
        }
        None
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        self.outgoing.take();
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        // Retain the handle while awaiting it: cancelling shutdown never
        // cancels the independent cleanup task, and a later call can await it.
        if let Some(supervisor) = self.supervisor.as_mut() {
            let result = supervisor
                .await
                .context("ACP backend cleanup task failed")?;
            self.supervisor.take();
            result?;
        }
        Ok(())
    }

    /// Sanitized stderr retained even when the UI queue was full or already drained.
    pub fn diagnostic_tail(&self) -> String {
        self.diagnostic_tail
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .snapshot()
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        self.outgoing.take();
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        // Dropping a JoinHandle detaches rather than aborts cleanup.
    }
}

async fn write_messages(
    mut stdin: ChildStdin,
    mut outgoing: mpsc::Receiver<Outgoing>,
    failures: mpsc::Sender<String>,
) {
    while let Some(frame) = outgoing.recv().await {
        let result = match timeout(WRITE_TIMEOUT, stdin.write_all(&frame.bytes)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(format!("Could not write to the Aster backend: {error}")),
            Err(_) => Err("Aster backend stopped reading protocol requests".to_owned()),
        };
        let failure = result.as_ref().err().cloned();
        if let Some(reason) = failure {
            let _ = failures.send(reason).await;
            return;
        }
    }
}

async fn read_messages(
    reader: impl AsyncRead + Unpin,
    messages: mpsc::Sender<Value>,
) -> Result<()> {
    let mut reader = BufReader::new(reader);
    let mut frame: Vec<u8> = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .await
            .context("Could not read ACP protocol output")?;
        if available.is_empty() {
            if frame.iter().any(|byte| !byte.is_ascii_whitespace()) {
                return Err(anyhow!("Aster backend closed an incomplete JSON-RPC frame"));
            }
            return Ok(());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let length = newline.map_or(available.len(), |index| index + 1);
        if frame.len() + length > MAX_FRAME_BYTES + 1 {
            return Err(anyhow!(
                "Aster backend exceeded the 50 MiB JSON-RPC frame limit"
            ));
        }
        frame.extend_from_slice(&available[..length]);
        reader.consume(length);
        if newline.is_none() {
            continue;
        }
        if frame.iter().all(u8::is_ascii_whitespace) {
            frame.clear();
            continue;
        }
        let message: Value = serde_json::from_slice(&frame).map_err(|_| {
            anyhow!("Aster backend sent invalid JSON-RPC; inspect backend diagnostics")
        })?;
        if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
            || !(message.get("method").is_some() || message.get("id").is_some())
        {
            return Err(anyhow!("Aster backend sent an invalid JSON-RPC envelope"));
        }
        frame.clear();
        if messages.send(message).await.is_err() {
            return Ok(());
        }
    }
}

async fn read_diagnostics(
    mut reader: impl AsyncRead + Unpin,
    diagnostics: mpsc::Sender<String>,
    dropped: Arc<AtomicUsize>,
    tail: Arc<Mutex<DiagnosticTail>>,
) {
    let mut filter = crate::state::TerminalTextFilter::default();
    let mut buffer = [0_u8; 8192];
    let mut pending = Vec::with_capacity(8196);
    loop {
        let count = reader.read(&mut buffer).await.unwrap_or_default();
        pending.extend_from_slice(&buffer[..count]);
        let length = if count == 0 {
            pending.len()
        } else {
            match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(error) if error.error_len().is_none() => error.valid_up_to(),
                Err(_) => pending.len(),
            }
        };
        let text = filter.feed(&String::from_utf8_lossy(&pending[..length]));
        pending.drain(..length);
        if !text.is_empty() {
            tail.lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(&text);
            match diagnostics.try_send(text) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return,
            }
        }
        if count == 0 {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn supervise(
    mut child: Child,
    mut process_tree: ProcessTree,
    mut stop: oneshot::Receiver<()>,
    mut failures: mpsc::Receiver<String>,
    closed: watch::Sender<Option<String>>,
    writer: JoinHandle<()>,
    mut stdout: JoinHandle<()>,
    mut stderr: JoinHandle<()>,
    grace: Duration,
) -> Result<()> {
    let (expected, mut reason, mut status) = tokio::select! {
        _ = &mut stop => (true, String::new(), None),
        status = child.wait() => (false, "Aster backend exited".to_owned(), Some(status)),
        failure = failures.recv() => (false, failure.unwrap_or_else(|| "Aster backend disconnected".to_owned()), None),
    };
    writer.abort();
    let _ = writer.await; // closes stdin before waiting for graceful shutdown
    if status.is_none() {
        status = Some(match timeout(grace, child.wait()).await {
            Ok(status) => status,
            Err(_) => {
                let _ = process_tree.terminate();
                // Keep a direct-child fallback if a native tree operation fails.
                let _ = child.start_kill();
                match timeout(Duration::from_secs(5), child.wait()).await {
                    Ok(status) => status,
                    Err(_) => {
                        stdout.abort();
                        stderr.abort();
                        let error = "Aster backend did not exit after being terminated";
                        let _ = closed.send(Some(error.to_owned()));
                        return Err(anyhow!(error));
                    }
                }
            }
        });
    }
    // A backend can exit while its descendants still own inherited pipes. Stop
    // the tree on every exit path, including normal shutdown and backend failure.
    let tree_cleanup = process_tree.terminate_and_wait().await;
    if expected {
        stdout.abort();
    }
    if timeout(Duration::from_secs(2), &mut stdout).await.is_err() {
        stdout.abort();
        let _ = stdout.await;
    }
    if timeout(Duration::from_secs(1), &mut stderr).await.is_err() {
        stderr.abort();
        let _ = stderr.await;
    }
    let status = status.expect("child wait always produces a result");
    if !expected {
        match &status {
            Ok(exit) => reason.push_str(&format!(" ({exit})")),
            Err(error) => reason.push_str(&format!(": {error}")),
        }
        let _ = closed.send(Some(reason));
    }
    status.context("Could not reap the Aster backend")?;
    tree_cleanup?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{Action, Controller};
    use crate::state::{Entry, State};
    use std::path::{Path, PathBuf};
    use tokio::time::{sleep, Instant};

    fn python() -> PathBuf {
        if let Some(path) = std::env::var_os("ASTER_TEST_PYTHON") {
            return path.into();
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let path = root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        assert!(
            path.is_file(),
            "Set ASTER_TEST_PYTHON or run uv sync --extra tui in the repository"
        );
        path
    }

    fn fixture(script: &str) -> (tempfile::TempDir, Transport) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("backend with spaces.py");
        std::fs::write(&path, script).unwrap();
        let mut command = Command::new(python());
        command
            .arg("-u")
            .arg(&path)
            .current_dir(dir.path())
            .env("PYTHONIOENCODING", "utf-8");
        let transport = Transport::spawn_command(command, Duration::from_millis(100)).unwrap();
        (dir, transport)
    }

    async fn next(transport: &mut Transport) -> Event {
        timeout(Duration::from_secs(30), async {
            loop {
                if let Some(event) = transport.try_recv() {
                    return event;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("backend did not produce an event")
    }

    #[tokio::test]
    async fn subprocess_roundtrip_keeps_ids_unicode_and_noisy_diagnostics_separate() {
        let (_dir, mut transport) = fixture(
            r#"
import sys, json
for line in sys.stdin:
    message=json.loads(line)
    if 'id' in message:
        for i in range(1000):
            print('\x1b[2Jdiagnostic 中文', file=sys.stderr)
        print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':message['params']},ensure_ascii=False),flush=True)
"#,
        );
        assert_eq!(
            transport
                .request("initialize", json!({"text":"中文\nwith newline"}))
                .await
                .unwrap(),
            1
        );
        loop {
            if let Event::Message(value) = next(&mut transport).await {
                assert_eq!(value["result"]["text"], "中文\nwith newline");
                assert_eq!(value["id"], 1);
                break;
            }
        }
        assert_eq!(transport.request("second", json!({})).await.unwrap(), 2);
        loop {
            match next(&mut transport).await {
                Event::Message(value) => {
                    assert_eq!(value["id"], 2);
                    break;
                }
                Event::Diagnostic(text) => assert!(!text.contains('\u{1b}')),
                Event::Closed(reason) => panic!("{reason}"),
            }
        }
        transport.shutdown().await.unwrap();
        transport.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn rejects_protocol_noise_instead_of_repeating_parser_errors() {
        let (_dir, mut transport) = fixture("print('provider debug, not JSON',flush=True)\n");
        let reason = loop {
            if let Event::Closed(reason) = next(&mut transport).await {
                break reason;
            }
        };
        assert!(
            reason.contains("JSON-RPC") || reason.contains("exited"),
            "{reason}"
        );
        transport.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn incomplete_frame_and_wrong_envelope_fail_at_protocol_boundary() {
        for bytes in [
            b"{\"jsonrpc\":\"2.0\"".as_slice(),
            b"{\"id\":1}\n".as_slice(),
        ] {
            let (sender, _) = mpsc::channel(1);
            assert!(read_messages(bytes, sender).await.is_err());
        }
    }

    #[tokio::test]
    async fn split_utf8_and_osc_diagnostics_are_filtered_without_losing_text() {
        let (mut write, read) = tokio::io::duplex(32);
        let (sender, mut receive) = mpsc::channel(16);
        let task = tokio::spawn(read_diagnostics(
            read,
            sender,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(Mutex::new(DiagnosticTail::default())),
        ));
        for byte in "中文\x1b]52;c;secret\x07 visible\x1b[2J".as_bytes() {
            write.write_all(&[*byte]).await.unwrap();
            tokio::task::yield_now().await;
        }
        drop(write);
        task.await.unwrap();
        let mut output = String::new();
        while let Ok(text) = receive.try_recv() {
            output.push_str(&text);
        }
        assert_eq!(output, "中文 visible");
    }

    #[tokio::test]
    async fn diagnostic_tail_retains_bounded_sanitized_failure_when_ui_queue_fills() {
        let input = format!(
            "{}\x1b]52;c;hidden\x07\x1b[2J\n启动失败: final error\n",
            "中文".repeat(MAX_DIAGNOSTIC_BYTES)
        );
        let (sender, _receiver) = mpsc::channel(1);
        let dropped = Arc::new(AtomicUsize::new(0));
        let tail = Arc::new(Mutex::new(DiagnosticTail::default()));
        read_diagnostics(
            input.as_bytes(),
            sender,
            Arc::clone(&dropped),
            Arc::clone(&tail),
        )
        .await;
        assert!(dropped.load(Ordering::Relaxed) > 0);
        let tail = tail.lock().unwrap();
        assert!(tail.truncated);
        assert!(tail.text.len() <= MAX_DIAGNOSTIC_BYTES);
        let text = tail.snapshot();
        assert!(text.ends_with("启动失败: final error\n"));
        assert!(!text.contains('\u{1b}'));
        assert!(!text.contains("hidden"));
    }

    #[tokio::test]
    async fn cancelled_shutdown_still_reaps_child_and_can_be_awaited_again() {
        let (_dir,mut transport)=fixture("import time\nprint('{\"jsonrpc\":\"2.0\",\"method\":\"ready\"}',flush=True)\ntime.sleep(60)\n");
        assert!(matches!(next(&mut transport).await, Event::Message(_)));
        assert!(timeout(Duration::from_millis(1), transport.shutdown())
            .await
            .is_err());
        timeout(Duration::from_secs(5), transport.shutdown())
            .await
            .unwrap()
            .unwrap();
    }

    // The probe also cleans up a leaked child before returning a failing verdict,
    // so running this regression against an older build leaves no sleepers.
    fn assert_descendant_stopped(pid: u32) {
        let result = std::process::Command::new(python())
            .args([
                "-c",
                r#"
import os, sys, time
pid = int(sys.argv[1])
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100001, False, pid)
    if not handle:
        assert ctypes.get_last_error() == 87
        sys.exit(0)
    alive = kernel.WaitForSingleObject(handle, 1000) == 258
    if alive:
        kernel.TerminateProcess(handle, 1)
        kernel.WaitForSingleObject(handle, 5000)
    kernel.CloseHandle(handle)
else:
    import signal
    from pathlib import Path
    def running():
        try:
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            return state != 'Z'
        except FileNotFoundError:
            return False
    until = time.monotonic() + 1
    while running() and time.monotonic() < until:
        time.sleep(.01)
    alive = running()
    if alive:
        os.kill(pid, signal.SIGKILL)
sys.exit(1 if alive else 0)
"#,
                &pid.to_string(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "descendant {pid} survived backend cleanup: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    async fn descendant_cleanup(backend_exit: &str, nested_job: bool) {
        let nested = if nested_job {
            r#"
import ctypes
from ctypes import wintypes
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel.CreateJobObjectW.restype = wintypes.HANDLE
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
inner_job = kernel.CreateJobObjectW(None, None)
assert inner_job and kernel.AssignProcessToJobObject(inner_job, kernel.GetCurrentProcess())
"#
        } else {
            ""
        };
        let script = format!(
            r#"
import json, subprocess, sys, time
{nested}
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(json.dumps({{'jsonrpc':'2.0', 'method':'ready', 'params':{{'pid':child.pid}}}}), flush=True)
{backend_exit}
open('graceful-exit', 'w').close()
"#
        );
        let (dir, mut transport) = fixture(&script);
        let pid = loop {
            if let Event::Message(message) = next(&mut transport).await {
                break message["params"]["pid"].as_u64().unwrap() as u32;
            }
        };
        if backend_exit == "sys.exit(1)" {
            loop {
                if matches!(next(&mut transport).await, Event::Closed(_)) {
                    break;
                }
            }
        }
        transport.shutdown().await.unwrap();
        assert_descendant_stopped(pid);
        if backend_exit == "sys.stdin.read()" {
            assert!(
                dir.path().join("graceful-exit").is_file(),
                "backend was killed before it could handle EOF"
            );
        }
    }

    #[tokio::test]
    async fn forced_shutdown_stops_unmanaged_descendants() {
        descendant_cleanup("time.sleep(60)", false).await;
    }

    #[tokio::test]
    async fn unexpected_backend_exit_stops_unmanaged_descendants() {
        descendant_cleanup("sys.exit(1)", false).await;
    }

    #[tokio::test]
    async fn graceful_shutdown_stops_unmanaged_descendants_after_backend_handles_eof() {
        descendant_cleanup("sys.stdin.read()", false).await;
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn shutdown_stops_descendants_in_nested_windows_jobs() {
        descendant_cleanup("time.sleep(60)", true).await;
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn real_powershell_session_works_inside_backend_job_and_is_cleaned_up() {
        let (_dir, mut transport) = fixture(
            r#"
import asyncio, contextlib, json, sys
with contextlib.redirect_stdout(sys.stderr):
    from aster.tools._powershell_session import PowerShellSession
async def main():
    shell = PowerShellSession()
    output, error, code = await shell.run("Write-Output 'nested-shell-ok'")
    print(json.dumps({'jsonrpc':'2.0','method':'ready','params':{
        'pid':shell._process.pid,'output':output,'error':error,'code':code}}), flush=True)
    await asyncio.sleep(60)
asyncio.run(main())
"#,
        );
        let result = loop {
            match next(&mut transport).await {
                Event::Message(message) => break message["params"].clone(),
                Event::Closed(reason) => panic!("{reason}: {}", transport.diagnostic_tail()),
                Event::Diagnostic(_) => {}
            }
        };
        transport.shutdown().await.unwrap();
        assert_descendant_stopped(result["pid"].as_u64().unwrap() as u32);
        assert_eq!(result["code"], 0, "{result}");
        assert!(
            result["output"]
                .as_str()
                .unwrap()
                .contains("nested-shell-ok"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn nonreading_backend_cannot_block_input_when_queueing_large_prompt() {
        let (_dir,mut transport)=fixture("import time\nprint('{\"jsonrpc\":\"2.0\",\"method\":\"ready\"}',flush=True)\ntime.sleep(60)\n");
        assert!(matches!(next(&mut transport).await, Event::Message(_)));
        timeout(
            Duration::from_secs(1),
            transport.request("session/prompt", json!({"text":"x".repeat(256*1024)})),
        )
        .await
        .unwrap()
        .unwrap();
        timeout(
            Duration::from_secs(1),
            transport.notify("session/cancel", json!({"sessionId":"s"})),
        )
        .await
        .unwrap()
        .unwrap();
        transport.shutdown().await.unwrap();
    }

    async fn send_actions(
        controller: &mut Controller,
        transport: &mut Transport,
        actions: Vec<Action>,
    ) -> bool {
        for action in actions {
            match action {
                Action::Request {
                    method,
                    params,
                    pending,
                } => {
                    let id = transport.request(method, params).await.unwrap();
                    controller.registered(id, pending);
                }
                Action::Notify { method, params } => {
                    transport.notify(method, params).await.unwrap();
                }
                Action::Reply { id, result } => {
                    transport.reply(id, result).await.unwrap();
                }
                Action::Exit => return true,
                Action::PasteClipboard
                | Action::CopySelection { .. }
                | Action::LoadWorkspaceChanges
                | Action::SearchFiles { .. }
                | Action::AttachFile { .. }
                | Action::OpenOriginalImage { .. }
                | Action::Reconnect
                | Action::Rollback { .. } => {
                    panic!("Transport tests do not request local input")
                }
            }
        }
        false
    }

    async fn settle(controller: &mut Controller, transport: &mut Transport) {
        timeout(Duration::from_secs(60), async {
            while controller.state.busy() {
                let actions = match next(transport).await {
                    Event::Message(value) if value["method"] == "session/request_permission" => {
                        let mut actions = controller.permission_request(&value);
                        // This fixture executes only the explicitly scripted
                        // local smoke-test cell. Exercise the real approval UI.
                        assert_eq!(
                            controller.state.overlay,
                            Some(crate::state::Overlay::Approval)
                        );
                        actions.extend(controller.key(crossterm::event::KeyEvent::new(
                            crossterm::event::KeyCode::Char('1'),
                            crossterm::event::KeyModifiers::NONE,
                        )));
                        actions
                    }
                    Event::Message(value) => controller.message(value),
                    Event::Diagnostic(text) => {
                        controller.diagnostic(text);
                        vec![]
                    }
                    Event::Closed(reason) => panic!("Backend closed: {reason}"),
                };
                assert!(
                    !send_actions(controller, transport, actions).await,
                    "{:?}",
                    controller.fatal_error
                );
            }
        })
        .await
        .unwrap();
        assert!(
            controller.fatal_error.is_none(),
            "{:?}",
            controller.fatal_error
        );
    }

    fn real_fixture(blocking: bool) -> (tempfile::TempDir, Transport, Controller) {
        real_fixture_with_args(if blocking { &["--blocking"] } else { &[] })
    }

    fn real_fixture_with_args(arguments: &[&str]) -> (tempfile::TempDir, Transport, Controller) {
        let directory = tempfile::tempdir().unwrap();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixture = root.join("aster-acp/tests/fixtures/fake_agent.py");
        let mut command = Command::new(python());
        command
            .arg("-u")
            .arg(fixture)
            .arg("--noisy")
            .args(arguments)
            .current_dir(directory.path())
            .env("PYTHONIOENCODING", "utf-8")
            .env("ASTER_USER_DIR", directory.path().join("user"));
        let transport = Transport::spawn_command(command, Duration::from_secs(10)).unwrap();
        let controller = Controller::new(
            State::new(
                directory.path().to_string_lossy().into_owned(),
                "fake-model".into(),
                false,
            ),
            None,
            false,
        );
        (directory, transport, controller)
    }

    #[tokio::test]
    async fn real_acp_rewind_forks_the_prefix_and_keeps_the_original_conversation() {
        use crate::state::Overlay;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let (_directory, mut transport, mut controller) = real_fixture_with_args(&["--rewind"]);
        let initialize = controller.initialize();
        send_actions(&mut controller, &mut transport, vec![initialize]).await;
        settle(&mut controller, &mut transport).await;
        let original = controller.state.session_id.clone().unwrap();
        let first = "First durable context";
        let second = "  Second request to edit\nwith exact whitespace  ";
        for prompt in [first, second] {
            let actions = controller.input(prompt.into());
            send_actions(&mut controller, &mut transport, actions).await;
            settle(&mut controller, &mut transport).await;
        }
        let user_messages = |controller: &Controller| -> Vec<String> {
            controller
                .state
                .entries
                .iter()
                .filter_map(|entry| match entry {
                    Entry::User(text) => Some(text.trim_end().to_owned()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(user_messages(&controller), [first, second.trim_end()]);

        let actions = controller.input("/rewind".into());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_eq!(controller.state.overlay, Some(Overlay::Rewind));
        assert_eq!(
            controller.state.choices().len(),
            2,
            "entries: {:?}\ndiagnostics: {:?}",
            controller.state.entries,
            controller.state.diagnostics
        );
        let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;

        let fork = controller.state.session_id.clone().unwrap();
        assert_ne!(fork, original);
        assert_eq!(user_messages(&controller), [first]);
        assert_eq!(controller.state.editor.lines().join("\n"), second);
        assert!(controller.state.queued.is_empty());

        controller.state.editor = tui_textarea::TextArea::default();
        controller.paste("Replacement request".into());
        let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_eq!(user_messages(&controller), [first, "Replacement request"]);
        assert!(controller.state.entries.iter().any(
            |entry| matches!(entry, Entry::Assistant(text) if text.contains("smoke test passed"))
        ));

        let actions = controller.resume(original.clone());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_eq!(
            controller.state.session_id.as_deref(),
            Some(original.as_str())
        );
        assert_eq!(user_messages(&controller), [first, second.trim_end()]);

        let actions = controller.resume(fork.clone());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_eq!(controller.state.session_id.as_deref(), Some(fork.as_str()));
        assert_eq!(user_messages(&controller), [first, "Replacement request"]);
        let actions = controller.input("/exit".into());
        send_actions(&mut controller, &mut transport, actions).await;
        // Exit is acknowledged by session/close; shutdown still bounds process cleanup.
        transport.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn real_acp_backend_prompt_replay_and_close_use_the_existing_python_runtime() {
        let (_directory, mut transport, mut controller) = real_fixture(false);
        let initialize = controller.initialize();
        send_actions(&mut controller, &mut transport, vec![initialize]).await;
        settle(&mut controller, &mut transport).await;
        let prepared_session = controller
            .state
            .session_id
            .clone()
            .expect("prepared at startup");
        assert!(controller.state.sessions.is_empty());
        assert!(controller.state.entries.is_empty());
        let actions = controller.input("检查连接".into());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        let session = controller.state.session_id.clone().unwrap();
        assert_eq!(session, prepared_session);
        assert!(controller.state.entries.iter().any(
            |e| matches!(e,Entry::Assistant(text) if text.contains("ASTER ACP smoke test passed"))
        ));
        assert!(controller
            .state
            .entries
            .iter()
            .any(|e| matches!(e,Entry::Tool(tool) if tool.status=="completed")));
        let actions = controller.input("/new".into());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_ne!(
            controller.state.session_id.as_deref(),
            Some(session.as_str())
        );
        assert!(controller.state.entries.is_empty());
        let actions = controller.input(format!("/resume {session}"));
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        assert_eq!(
            controller.state.session_id.as_deref(),
            Some(session.as_str())
        );
        // ACP appends a newline to delimit replayed turns from the same speaker.
        assert!(
            controller
                .state
                .entries
                .iter()
                .any(|e| matches!(e,Entry::User(text) if text.trim_end()=="检查连接")),
            "replayed: {:?}",
            controller.state.entries
        );
        assert!(controller
            .state
            .entries
            .iter()
            .any(|e| matches!(e,Entry::Assistant(text) if text.contains("smoke test passed"))));
        let actions = controller.input("/new".into());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        transport.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn real_acp_backend_cancel_waits_for_terminal_result_before_next_request() {
        let (_directory, mut transport, mut controller) = real_fixture(true);
        let initialize = controller.initialize();
        send_actions(&mut controller, &mut transport, vec![initialize]).await;
        settle(&mut controller, &mut transport).await;
        let actions = controller.input("block until cancelled".into());
        send_actions(&mut controller, &mut transport, actions).await;
        let started = Instant::now();
        while !controller
            .state
            .entries
            .iter()
            .any(|e| matches!(e, Entry::Tool(_)))
        {
            assert!(started.elapsed() < Duration::from_secs(30));
            let actions = match next(&mut transport).await {
                Event::Message(value) => controller.message(value),
                Event::Diagnostic(text) => {
                    controller.diagnostic(text);
                    vec![]
                }
                Event::Closed(reason) => panic!("{reason}"),
            };
            send_actions(&mut controller, &mut transport, actions).await;
        }
        let actions = controller.cancel();
        send_actions(&mut controller, &mut transport, actions).await;
        assert!(controller.state.busy());
        settle(&mut controller, &mut transport).await;
        assert_eq!(controller.state.status, "Cancelled");
        let actions = controller.input("/new".into());
        send_actions(&mut controller, &mut transport, actions).await;
        settle(&mut controller, &mut transport).await;
        transport.shutdown().await.unwrap();
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Workspace-local submitted input history, independent of ACP conversation state.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

pub const MAX_ENTRIES: usize = 1_000;
const MAX_BYTES: usize = 8 * 1024 * 1024;
const LOCK_TIMEOUT: Duration = Duration::from_millis(250);
const HISTORY_NAME: &str = "tui-input-history.jsonl";
const LOCK_NAME: &str = "tui-input-history.lock";
const TEMP_NAME: &str = "tui-input-history.jsonl.tmp";

#[derive(Clone, Debug)]
pub struct InputHistory {
    directory: PathBuf,
}

impl InputHistory {
    pub fn for_workspace(workspace: &Path) -> Self {
        Self {
            directory: workspace.join(".aster"),
        }
    }

    /// Return oldest-to-newest input without creating files or opening a session.
    /// Incomplete or malformed records are skipped; healthy later records survive.
    pub fn load(&self) -> Result<Vec<String>> {
        self.load_with_limits(MAX_ENTRIES, MAX_BYTES)
    }

    /// Save a submitted prompt immediately. Callers can report failures without
    /// preventing the prompt from running or discarding their in-memory history.
    pub fn append(&self, text: &str) -> Result<()> {
        self.append_with_limits(text, MAX_ENTRIES, MAX_BYTES)
    }

    /// Bootstrap older sessions once, without overwriting another terminal's input.
    pub fn seed_if_empty(&self, entries: &[String]) -> Result<Vec<String>> {
        if entries.is_empty() {
            return self.load();
        }
        fs::create_dir_all(&self.directory).context("Could not create input history directory")?;
        let _lock = HistoryLock::acquire(&self.directory.join(LOCK_NAME))?;
        let existing = self.load()?;
        if !existing.is_empty() {
            return Ok(existing);
        }
        let mut selected = VecDeque::new();
        let mut records = VecDeque::new();
        let mut bytes = 0;
        for text in entries {
            if text.trim().is_empty() || text.len() > MAX_BYTES {
                continue;
            }
            let mut encoded = serde_json::to_vec(text).context("Could not encode input history")?;
            encoded.push(b'\n');
            if encoded.len() > MAX_BYTES {
                continue;
            }
            bytes += encoded.len();
            records.push_back(encoded);
            selected.push_back(text.clone());
            while records.len() > MAX_ENTRIES || bytes > MAX_BYTES {
                bytes -= records
                    .pop_front()
                    .expect("new record fits the bounds")
                    .len();
                selected.pop_front();
            }
        }
        self.replace(records)?;
        Ok(selected.into())
    }

    fn load_with_limits(&self, max_entries: usize, max_bytes: usize) -> Result<Vec<String>> {
        let path = self.directory.join(HISTORY_NAME);
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error).context("Could not open saved input history"),
        };
        let length = file
            .metadata()
            .context("Could not inspect saved input history")?
            .len();
        let offset = length.saturating_sub(max_bytes as u64);
        file.seek(SeekFrom::Start(offset))
            .context("Could not read saved input history")?;
        let mut bytes = Vec::new();
        file.take(max_bytes as u64)
            .read_to_end(&mut bytes)
            .context("Could not read saved input history")?;
        let mut lines = bytes.split(|byte| *byte == b'\n');
        if offset != 0 {
            // The bounded tail may start inside a JSON string, so discard its fragment.
            lines.next();
        }
        let mut history = VecDeque::new();
        for line in lines {
            if let Ok(text) = serde_json::from_slice::<String>(line) {
                if !text.trim().is_empty() {
                    history.push_back(text);
                    if history.len() > max_entries {
                        history.pop_front();
                    }
                }
            }
        }
        Ok(history.into())
    }

    fn append_with_limits(&self, text: &str, max_entries: usize, max_bytes: usize) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }
        if text.len() > max_bytes {
            bail!("Input is too large to save in the 8 MiB history");
        }
        let mut newest = serde_json::to_vec(text).context("Could not encode input history")?;
        newest.push(b'\n');
        if newest.len() > max_bytes {
            bail!("Input is too large to save in the 8 MiB history");
        }
        fs::create_dir_all(&self.directory).context("Could not create input history directory")?;
        let _lock = HistoryLock::acquire(&self.directory.join(LOCK_NAME))?;
        // Reload while holding the stable lock. Saving an instance's startup
        // snapshot would overwrite prompts appended by another TUI process.
        let mut records: VecDeque<Vec<u8>> = self
            .load_with_limits(max_entries, max_bytes)?
            .into_iter()
            .map(|entry| {
                let mut record = serde_json::to_vec(&entry).expect("strings serialize to JSON");
                record.push(b'\n');
                record
            })
            .collect();
        let mut bytes = records.iter().map(Vec::len).sum::<usize>() + newest.len();
        records.push_back(newest);
        while records.len() > max_entries || bytes > max_bytes {
            bytes -= records
                .pop_front()
                .expect("newest record fits the bounds")
                .len();
        }
        self.replace(records)
    }

    fn replace(&self, records: VecDeque<Vec<u8>>) -> Result<()> {
        let temporary = self.directory.join(TEMP_NAME);
        // A previous process may have stopped before its atomic rename. The
        // fixed temporary path is ours and is only touched under the same lock.
        match fs::remove_file(&temporary) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).context("Could not recover input history temporary file")
            }
        }
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temporary)
            .context("Could not write input history")?;
        let cleanup = TemporaryFile(temporary.clone());
        for record in records {
            file.write_all(&record)
                .context("Could not write input history")?;
        }
        file.sync_all().context("Could not flush input history")?;
        drop(file);
        fs::rename(&temporary, self.directory.join(HISTORY_NAME))
            .context("Could not replace input history")?;
        drop(cleanup);
        #[cfg(unix)]
        if let Ok(directory) = File::open(&self.directory) {
            let _ = directory.sync_all();
        }
        Ok(())
    }
}

struct TemporaryFile(PathBuf);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct HistoryLock {
    // Closing this handle releases the OS lock, including on process death.
    _file: File,
}

impl HistoryLock {
    fn acquire(path: &Path) -> Result<Self> {
        let started = Instant::now();
        loop {
            match Self::try_acquire(path) {
                Ok(file) => return Ok(Self { _file: file }),
                Err(error) if lock_contended(&error) => {
                    if started.elapsed() >= LOCK_TIMEOUT {
                        bail!("Input history is busy in another terminal; this prompt was not saved to disk");
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(error).context("Could not lock input history"),
            }
        }
    }

    fn try_acquire(path: &Path) -> io::Result<File> {
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Sharing violations are immediate, so no OS lock call can wait indefinitely.
            options.share_mode(0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: file owns a valid descriptor until this function returns.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(not(any(windows, unix)))]
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Input history locking is unavailable",
        ));
        #[cfg(any(windows, unix))]
        Ok(file)
    }
}

fn lock_contended(error: &io::Error) -> bool {
    #[cfg(windows)]
    if matches!(error.raw_os_error(), Some(32 | 33)) {
        return true;
    }
    error.kind() == io::ErrorKind::WouldBlock
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn empty_load_does_not_create_workspace_files() {
        let workspace = tempfile::tempdir().unwrap();
        assert!(InputHistory::for_workspace(workspace.path())
            .load()
            .unwrap()
            .is_empty());
        assert!(!workspace.path().join(".aster").exists());
    }

    #[test]
    fn bootstrap_keeps_exact_inputs_and_preserves_concurrent_newer_writes() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        let old = vec!["old".into(), "  中文\ninput  \n".into()];
        assert_eq!(history.seed_if_empty(&old).unwrap(), old);
        history.append("newer").unwrap();
        let other = InputHistory::for_workspace(workspace.path());
        assert_eq!(
            other.seed_if_empty(&["stale".into()]).unwrap(),
            ["old", "  中文\ninput  \n", "newer"]
        );
        assert_eq!(other.load().unwrap(), history.load().unwrap());
    }

    #[test]
    fn restart_preserves_exact_multiline_text_and_isolates_workspaces() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        history.append("  中文\n  return True\n\n").unwrap();
        history.append("same").unwrap();
        history.append("same").unwrap();
        let restarted = InputHistory::for_workspace(workspace.path());
        assert_eq!(
            restarted.load().unwrap(),
            ["  中文\n  return True\n\n", "same", "same"]
        );
        assert!(
            InputHistory::for_workspace(workspace.path().join("another").as_path())
                .load()
                .unwrap()
                .is_empty()
        );
        assert!(!workspace.path().join(".aster/sessions").exists());
    }

    #[test]
    fn independently_opened_writers_merge_instead_of_overwriting() {
        let workspace = tempfile::tempdir().unwrap();
        let barrier = Arc::new(Barrier::new(4));
        let threads: Vec<_> = (0..4)
            .map(|writer| {
                let history = InputHistory::for_workspace(workspace.path());
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    assert!(history.load().unwrap().is_empty());
                    barrier.wait();
                    for sequence in 0..8 {
                        let text = format!("writer {writer}: {sequence}\n中文");
                        let deadline = Instant::now() + Duration::from_secs(5);
                        loop {
                            match history.append(&text) {
                                Ok(()) => break,
                                Err(error)
                                    if error.to_string()
                                        == "Input history is busy in another terminal; this prompt was not saved to disk" =>
                                {
                                    assert!(
                                        Instant::now() < deadline,
                                        "writer {writer}, sequence {sequence} could not acquire the history lock: {error:#}"
                                    );
                                    // Busy is a documented bounded failure. Let another
                                    // writer finish, then verify all accepted writes merge.
                                    std::thread::sleep(Duration::from_millis(5));
                                }
                                Err(error) => panic!(
                                    "writer {writer}, sequence {sequence} could not save input history: {error:#}"
                                ),
                            }
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let history = InputHistory::for_workspace(workspace.path())
            .load()
            .unwrap();
        assert_eq!(history.len(), 32);
        for writer in 0..4 {
            let found: Vec<_> = history
                .iter()
                .filter(|text| text.starts_with(&format!("writer {writer}:")))
                .cloned()
                .collect();
            let expected: Vec<_> = (0..8)
                .map(|sequence| format!("writer {writer}: {sequence}\n中文"))
                .collect();
            assert_eq!(found, expected);
        }
    }

    #[test]
    fn prunes_old_records_by_count_and_encoded_bytes() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        for text in ["one", "two", "three", "four"] {
            history.append_with_limits(text, 3, 40).unwrap();
        }
        assert_eq!(history.load().unwrap(), ["two", "three", "four"]);
        history
            .append_with_limits("a larger latest input", 3, 40)
            .unwrap();
        assert_eq!(
            history.load().unwrap(),
            ["three", "four", "a larger latest input"]
        );
        assert!(
            fs::metadata(history.directory.join(HISTORY_NAME))
                .unwrap()
                .len()
                <= 40
        );
        let before = history.load().unwrap();
        assert!(history.append_with_limits(&"\n".repeat(40), 3, 40).is_ok()); // empty inputs are ignored
        assert!(history
            .append_with_limits(&"x\n".repeat(20), 3, 40)
            .is_err());
        assert_eq!(history.load().unwrap(), before);
    }

    #[test]
    fn malformed_records_and_interrupted_temporary_writes_do_not_lose_good_history() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        fs::create_dir_all(&history.directory).unwrap();
        fs::write(
            history.directory.join(HISTORY_NAME),
            b"\"kept\"\nnot-json\n\"last\"\n\"unfinished",
        )
        .unwrap();
        fs::write(history.directory.join(TEMP_NAME), b"interrupted output").unwrap();
        assert_eq!(history.load().unwrap(), ["kept", "last"]);
        history.append("new").unwrap();
        assert_eq!(history.load().unwrap(), ["kept", "last", "new"]);
        assert!(!history.directory.join(TEMP_NAME).exists());
    }

    #[test]
    fn blocked_writer_fails_promptly_and_released_lock_can_be_reused() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        history.append("kept").unwrap();
        let lock = HistoryLock::acquire(&history.directory.join(LOCK_NAME)).unwrap();
        let started = Instant::now();
        assert!(history
            .append("blocked")
            .unwrap_err()
            .to_string()
            .contains("busy"));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(history.load().unwrap(), ["kept"]);
        drop(lock);
        history.append("after release").unwrap();
        assert_eq!(history.load().unwrap(), ["kept", "after release"]);
    }

    #[test]
    fn failed_replace_preserves_previous_history() {
        let workspace = tempfile::tempdir().unwrap();
        let history = InputHistory::for_workspace(workspace.path());
        history.append("kept").unwrap();
        fs::create_dir(history.directory.join(TEMP_NAME)).unwrap();
        assert!(history.append("not saved").is_err());
        assert_eq!(history.load().unwrap(), ["kept"]);
    }
}

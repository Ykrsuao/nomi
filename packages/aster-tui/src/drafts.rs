// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Durable unsent input. Loading always pauses recovery; this module never sends.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::attachments::{self, Attachment};
use crate::file_io::{self, is_link as is_reparse_point, ReadError};

const SCHEMA_VERSION: u32 = 1;
pub const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_QUEUED_DRAFTS: usize = 128;
const MAX_RECORD_BYTES: usize = 32 * 1024 * 1024;
const MAX_DISK_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SCOPES: usize = 128;
const MAX_SCOPE_BYTES: usize = 96;
const LOCK_TIMEOUT: Duration = Duration::from_millis(250);
const PROVISIONAL_SCOPE_PREFIX: &str = "local-draft-";

/// Give a new conversation its own input scope before the backend owns it.
pub fn provisional_scope(identifier: &str) -> String {
    format!("{PROVISIONAL_SCOPE_PREFIX}{identifier}")
}

/// None is the legacy shared new-conversation scope.
pub fn is_provisional_scope(scope: Option<&str>) -> bool {
    scope.is_none_or(|scope| scope.starts_with(PROVISIONAL_SCOPE_PREFIX))
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDraft {
    pub text: String,
    pub attachments: Vec<Attachment>,
}

impl MessageDraft {
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.attachments.is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        if self.text.len() > MAX_MESSAGE_BYTES {
            bail!("Draft text exceeds the 8 MiB limit");
        }
        attachments::validate(&self.attachments)
    }
}

impl From<String> for MessageDraft {
    fn from(text: String) -> Self {
        Self {
            text,
            attachments: Vec::new(),
        }
    }
}

impl From<&str> for MessageDraft {
    fn from(text: &str) -> Self {
        text.to_owned().into()
    }
}

impl Deref for MessageDraft {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for MessageDraft {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.text.fmt(formatter)
    }
}

impl PartialEq<&str> for MessageDraft {
    fn eq(&self, other: &&str) -> bool {
        self.attachments.is_empty() && self.text == *other
    }
}

impl PartialEq<str> for MessageDraft {
    fn eq(&self, other: &str) -> bool {
        self.attachments.is_empty() && self.text == other
    }
}

impl PartialEq<String> for MessageDraft {
    fn eq(&self, other: &String) -> bool {
        self.attachments.is_empty() && self.text == *other
    }
}

impl PartialEq<MessageDraft> for String {
    fn eq(&self, other: &MessageDraft) -> bool {
        other == self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftSnapshot {
    pub composer: MessageDraft,
    pub cursor: (usize, usize),
    pub queued: Vec<MessageDraft>,
    /// Keep the original until a known terminal response, including after a
    /// connection loss. Recovery must ask the user to review ambiguous delivery.
    pub in_flight: Option<MessageDraft>,
    pub queue_paused: bool,
    /// A consumed provisional draft points to its already-published session.
    /// This keeps startup recovery atomic across the two record writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated_to: Option<String>,
}

impl DraftSnapshot {
    pub fn is_empty(&self) -> bool {
        self.composer.text.is_empty()
            && self.composer.attachments.is_empty()
            && self.queued.is_empty()
            && self.in_flight.is_none()
    }

    fn validate(&self) -> Result<()> {
        if let Some(target) = &self.migrated_to {
            scope_key(Some(target))?;
            if is_provisional_scope(Some(target)) {
                bail!("A transferred draft must target a saved session");
            }
            if !self.is_empty() {
                bail!("A transferred draft cannot also contain unsent input");
            }
        }
        if self.queued.len() > MAX_QUEUED_DRAFTS {
            bail!("Saved queue exceeds the 128-message limit");
        }
        let mut raw_bytes = 0_usize;
        for message in std::iter::once(&self.composer)
            .chain(self.queued.iter())
            .chain(self.in_flight.iter())
        {
            raw_bytes = raw_bytes.saturating_add(message.text.len());
            for attachment in &message.attachments {
                raw_bytes = raw_bytes.saturating_add(match attachment {
                    Attachment::Text { text, .. } => text.len(),
                    Attachment::Image { data, .. } => data.len(),
                });
            }
            if raw_bytes > MAX_RECORD_BYTES {
                bail!("Saved draft exceeds the 32 MiB limit");
            }
            message.validate()?;
        }
        if self.cursor.0 > MAX_MESSAGE_BYTES || self.cursor.1 > MAX_MESSAGE_BYTES {
            bail!("Saved composer cursor is invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LoadedDraft {
    /// A saved session or local provisional scope; None is the legacy new draft.
    pub session_id: Option<String>,
    pub revision: u64,
    pub snapshot: DraftSnapshot,
    pub warning: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    schema_version: u32,
    revision: u64,
    session_id: Option<String>,
    snapshot: DraftSnapshot,
}

impl Record {
    fn loaded(mut self, warning: Option<String>) -> LoadedDraft {
        // Even a previously running queue requires an explicit user action after
        // recovery. In-flight input stays intact for that review.
        self.snapshot.queue_paused = true;
        LoadedDraft {
            session_id: self.session_id,
            revision: self.revision,
            snapshot: self.snapshot,
            warning,
        }
    }
}

#[derive(Debug)]
struct UnsupportedSchema(u32);

impl fmt::Display for UnsupportedSchema {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Saved drafts use unsupported schema {}; their files were preserved",
            self.0
        )
    }
}

impl std::error::Error for UnsupportedSchema {}

#[derive(Clone, Debug)]
pub struct DraftStore {
    workspace: PathBuf,
    directory: PathBuf,
}

impl DraftStore {
    pub fn for_workspace(workspace: &Path) -> Result<Self> {
        let workspace = workspace
            .canonicalize()
            .context("Could not locate draft workspace")?;
        let directory = workspace.join(".aster").join("tui-drafts");
        Ok(Self {
            workspace,
            directory,
        })
    }

    /// A missing store is a read-only empty result. Corrupt records are never
    /// treated as empty input; a valid backup is returned with a visible warning.
    pub fn load(&self, session_id: Option<&str>) -> Result<Option<LoadedDraft>> {
        let key = scope_key(session_id)?;
        if !self.ensure_directory(false)? {
            return Ok(None);
        }
        Ok(self
            .read_existing(&key, session_id)?
            .map(|(record, warning)| record.loaded(warning)))
    }

    /// The highest committed global revision identifies the latest scope,
    /// including unsent new-conversation input. No separate "last session"
    /// pointer can become stale between publishing a draft and a process crash.
    pub fn load_latest(&self) -> Result<Option<LoadedDraft>> {
        if !self.ensure_directory(false)? {
            return Ok(None);
        }
        let (scopes, _) = self.inventory()?;
        let mut latest: Option<(Record, Option<String>)> = None;
        let mut unreadable = 0;
        let mut recovered = 0;
        for (key, session_id) in scopes {
            match self.read_existing(&key, session_id.as_deref()) {
                Ok(Some((record, warning))) => {
                    recovered += usize::from(warning.is_some());
                    if latest
                        .as_ref()
                        .is_none_or(|(old, _)| old.revision < record.revision)
                    {
                        latest = Some((record, warning));
                    }
                }
                Ok(None) => {}
                Err(error) if error.is::<UnsupportedSchema>() => return Err(error),
                Err(_) => unreadable += 1,
            }
        }
        if recovered != 0 {
            if let Some((_, warning)) = latest.as_mut() {
                if warning.is_none() {
                    *warning = Some("Another saved draft was recovered from an older backup; its original session's input was preserved.".into());
                }
            }
        }
        if unreadable != 0 {
            let notice = "Some saved drafts could not be read; their files were preserved.";
            let Some((_, warning)) = latest.as_mut() else {
                bail!("{notice}");
            };
            *warning = Some(match warning.take() {
                Some(existing) => format!("{existing} {notice}"),
                None => notice.into(),
            });
        }
        if let Some((record, warning)) = &latest {
            if let Some((destination, destination_warning)) = self.load_transfer_target(record)? {
                return Ok(Some(
                    destination.loaded(destination_warning.or_else(|| warning.clone())),
                ));
            }
        }
        Ok(latest.map(|(record, warning)| record.loaded(warning)))
    }

    /// List recoverable input without activating any conversation or consuming it.
    pub fn list_pending(&self) -> Result<Vec<LoadedDraft>> {
        if !self.ensure_directory(false)? {
            return Ok(Vec::new());
        }
        let (scopes, _) = self.inventory()?;
        let mut pending = Vec::new();
        let mut unreadable = 0;
        let mut skipped_recovery = false;
        for (key, scope) in scopes {
            match self.read_existing(&key, scope.as_deref()) {
                Ok(Some((record, warning))) => {
                    if record.snapshot.is_empty() {
                        skipped_recovery |= warning.is_some();
                        match self.load_transfer_target(&record) {
                            Ok(Some((_, warning))) => skipped_recovery |= warning.is_some(),
                            Ok(None) => {}
                            Err(error) if error.is::<UnsupportedSchema>() => return Err(error),
                            Err(_) => unreadable += 1,
                        }
                    } else {
                        pending.push(record.loaded(warning));
                    }
                }
                Ok(None) => {}
                Err(error) if error.is::<UnsupportedSchema>() => return Err(error),
                Err(_) => unreadable += 1,
            }
        }
        pending.sort_by_key(|draft| std::cmp::Reverse(draft.revision));
        let mut notices = Vec::new();
        if skipped_recovery {
            notices.push(
                "Another saved draft was recovered from an older backup; its files were preserved.",
            );
        }
        if unreadable != 0 {
            notices.push("Some saved drafts could not be read; their files were preserved.");
        }
        if !notices.is_empty() {
            let notice = notices.join(" ");
            let Some(first) = pending.first_mut() else {
                bail!("{notice}");
            };
            first.warning = Some(match first.warning.take() {
                Some(existing) => format!("{existing} {notice}"),
                None => notice,
            });
        }
        Ok(pending)
    }

    /// Compare-and-swap protects unsent input in another terminal. Pass None
    /// only for an absent scope, and retain the returned revision for later saves.
    /// Callers may debounce composer changes but must save before dispatching.
    /// An empty named provisional scope may be reclaimed; start a new scope
    /// for later input after successfully clearing one.
    pub fn save(
        &self,
        session_id: Option<&str>,
        expected_revision: Option<u64>,
        snapshot: &DraftSnapshot,
    ) -> Result<u64> {
        let key = scope_key(session_id)?;
        snapshot.validate()?;
        if snapshot.migrated_to.is_some() && !is_provisional_scope(session_id) {
            bail!("Only provisional input can be transferred to a session");
        }
        self.ensure_directory(true)?;
        let lock = DraftLock::acquire(&self.directory.join(".lock"))?;
        let existing = self.read_existing(&key, session_id)?;
        let current_revision = existing.as_ref().map(|(record, _)| record.revision);
        if current_revision != expected_revision {
            bail!("Draft changed in another terminal; newer saved input was preserved");
        }
        let (scopes, mut disk_bytes) = self.inventory()?;
        let sequence = self.sequence(&scopes)?.max(current_revision.unwrap_or(0));
        let revision = sequence
            .checked_add(1)
            .context("Draft revision counter is exhausted")?;
        let record = Record {
            schema_version: SCHEMA_VERSION,
            revision,
            session_id: session_id.map(str::to_owned),
            snapshot: snapshot.clone(),
        };
        if self
            .load_transfer_target(&record)?
            .is_some_and(|(_, warning)| warning.is_some())
        {
            bail!("The destination's newest saved draft is unreadable; source input was preserved");
        }
        let bytes = encode_bounded(&record, MAX_RECORD_BYTES)?;
        let backup = existing
            .as_ref()
            .filter(|(_, warning)| warning.is_none())
            .map(|(record, _)| encode_bounded(record, MAX_RECORD_BYTES))
            .transpose()?;
        // Include temporary copies in the bound. Only empty named
        // provisional records may be reclaimed to make room for another save.
        let incoming_bytes = (bytes.len() as u64)
            .saturating_add(backup.as_ref().map_or(0, |bytes| bytes.len() as u64));
        let adding_scope = !scopes.contains_key(&key);
        let needs_space = |scope_count: usize, stored_bytes: u64| {
            (adding_scope && scope_count >= MAX_SCOPES)
                || stored_bytes.saturating_add(incoming_bytes) > MAX_DISK_BYTES
        };
        let mut scope_count = scopes.len();
        if needs_space(scope_count, disk_bytes) {
            for (candidate_key, scope) in &scopes {
                if candidate_key == &key
                    || scope.is_none()
                    || !is_provisional_scope(scope.as_deref())
                {
                    continue;
                }
                let Ok(Some(candidate)) = self.read_record(
                    &self.directory.join(format!("{candidate_key}.json")),
                    scope.as_deref(),
                ) else {
                    continue;
                };
                if self.reclaim_empty_provisional(&candidate, sequence, &lock)? {
                    let (remaining, remaining_bytes) = self.inventory()?;
                    scope_count = remaining.len();
                    disk_bytes = remaining_bytes;
                    if !needs_space(scope_count, disk_bytes) {
                        break;
                    }
                }
            }
        }
        if adding_scope && scope_count >= MAX_SCOPES {
            bail!("The workspace already has 128 saved draft scopes; no unsent input was removed");
        }
        if disk_bytes.saturating_add(incoming_bytes) > MAX_DISK_BYTES {
            bail!("Draft storage reached its 256 MiB limit; no unsent input was removed");
        }
        // Allocate first: an interrupted publication can leave a harmless gap,
        // but two scopes cannot be assigned the same committed revision.
        self.publish(
            &self.directory.join(".sequence"),
            revision.to_string().as_bytes(),
        )?;
        if let Some(backup) = backup {
            self.publish(&self.directory.join(format!("{key}.bak")), &backup)?;
        }
        self.publish(&self.directory.join(format!("{key}.json")), &bytes)?;
        Ok(revision)
    }

    fn ensure_directory(&self, create: bool) -> Result<bool> {
        for path in [self.workspace.join(".aster"), self.directory.clone()] {
            match fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.file_type().is_dir() && !is_reparse_point(&metadata) => {}
                Ok(_) => bail!("Draft storage directory must not be a file or symbolic link"),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    if !create {
                        return Ok(false);
                    }
                    let builder = fs::DirBuilder::new();
                    #[cfg(unix)]
                    let builder = {
                        use std::os::unix::fs::DirBuilderExt;
                        let mut builder = builder;
                        builder.mode(0o700);
                        builder
                    };
                    match builder.create(&path) {
                        Ok(()) => {}
                        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                            let metadata = fs::symlink_metadata(&path)?;
                            if !metadata.file_type().is_dir() || is_reparse_point(&metadata) {
                                bail!("Draft storage directory changed while opening it");
                            }
                        }
                        Err(error) => return Err(error).context("Could not create draft storage"),
                    }
                }
                Err(error) => return Err(error).context("Could not inspect draft storage"),
            }
        }
        Ok(true)
    }

    fn read_existing(
        &self,
        key: &str,
        session_id: Option<&str>,
    ) -> Result<Option<(Record, Option<String>)>> {
        let primary = self.read_record(&self.directory.join(format!("{key}.json")), session_id);
        match primary {
            Ok(Some(record)) => return Ok(Some((record, None))),
            Err(ref error) if error.is::<UnsupportedSchema>() => {
                return Err(primary.unwrap_err());
            }
            _ => {}
        }
        let backup = self.read_record(&self.directory.join(format!("{key}.bak")), session_id);
        match backup {
            Ok(Some(record)) => Ok(Some((record, Some(
                "The newest saved draft was unreadable; recovered its last valid copy. Review it before sending.".into(),
            )))),
            Ok(None) if primary.as_ref().is_ok_and(Option::is_none) => Ok(None),
            Err(error) if error.is::<UnsupportedSchema>() => Err(error),
            _ => bail!("Saved draft is unreadable and has no valid backup; its files were preserved"),
        }
    }

    fn read_record(&self, path: &Path, session_id: Option<&str>) -> Result<Option<Record>> {
        let Some(bytes) = read_bounded(path, MAX_RECORD_BYTES)? else {
            return Ok(None);
        };
        let record: Record =
            serde_json::from_slice(&bytes).context("Saved draft is not valid JSON")?;
        if record.schema_version != SCHEMA_VERSION {
            return Err(UnsupportedSchema(record.schema_version).into());
        }
        if record.revision == 0 || record.session_id.as_deref() != session_id {
            bail!("Saved draft revision or session scope is invalid");
        }
        record.snapshot.validate()?;
        if record.snapshot.migrated_to.is_some() && !is_provisional_scope(session_id) {
            bail!("Only provisional input can be transferred to a session");
        }
        Ok(Some(record))
    }

    fn load_transfer_target(&self, record: &Record) -> Result<Option<(Record, Option<String>)>> {
        let Some(target) = &record.snapshot.migrated_to else {
            return Ok(None);
        };
        let key = scope_key(Some(target))?;
        let Some((destination, warning)) = self.read_existing(&key, Some(target))? else {
            bail!("The transferred draft is unavailable; saved files were preserved");
        };
        if !is_provisional_scope(record.session_id.as_deref())
            || is_provisional_scope(Some(target))
            || destination.snapshot.migrated_to.is_some()
        {
            bail!("Invalid draft transfer; saved files were preserved");
        }
        Ok(Some((destination, warning)))
    }

    fn reclaim_empty_provisional(
        &self,
        expected: &Record,
        sequence: u64,
        _lock: &DraftLock,
    ) -> Result<bool> {
        let Some(scope) = expected.session_id.as_deref() else {
            // The legacy shared scope can still belong to older clients.
            return Ok(false);
        };
        if !is_provisional_scope(Some(scope))
            || !expected.snapshot.is_empty()
            || expected.revision > sequence
        {
            return Ok(false);
        }
        if let Some(target) = expected.snapshot.migrated_to.as_deref() {
            if is_provisional_scope(Some(target)) {
                return Ok(false);
            }
            let target_key = scope_key(Some(target))?;
            let Ok(Some(destination)) = self.read_record(
                &self.directory.join(format!("{target_key}.json")),
                Some(target),
            ) else {
                return Ok(false);
            };
            if destination.snapshot.migrated_to.is_some() {
                return Ok(false);
            }
        }
        let key = scope_key(Some(scope))?;
        let paths = ["bak", "tmp", "json"]
            .map(|extension| self.directory.join(format!("{key}.{extension}")));
        for path in &paths {
            match fs::symlink_metadata(path) {
                Ok(metadata) if metadata.file_type().is_file() && !is_reparse_point(&metadata) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                _ => return Ok(false),
            }
        }
        let Ok(Some(current)) = self.read_record(&paths[2], Some(scope)) else {
            return Ok(false);
        };
        if current.revision != expected.revision
            || current.snapshot.migrated_to != expected.snapshot.migrated_to
            || !current.snapshot.is_empty()
        {
            return Ok(false);
        }
        // Persist the high-water mark before deleting any allocator evidence,
        // even if the incoming save later fails or the process stops here.
        self.publish(
            &self.directory.join(".sequence"),
            sequence.to_string().as_bytes(),
        )?;
        // A leftover backup must never resurrect cleared or transferred input.
        // Preserve the empty primary until backup and temporary removals are durable.
        for path in &paths[..2] {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).context("Could not reclaim empty provisional draft")
                }
            }
        }
        #[cfg(unix)]
        File::open(&self.directory)?
            .sync_all()
            .context("Could not flush draft directory")?;
        fs::remove_file(&paths[2]).context("Could not reclaim empty provisional draft")?;
        #[cfg(unix)]
        File::open(&self.directory)?
            .sync_all()
            .context("Could not flush draft directory")?;
        Ok(true)
    }

    fn inventory(&self) -> Result<(BTreeMap<String, Option<String>>, u64)> {
        let mut scopes = BTreeMap::new();
        let mut bytes = 0_u64;
        for (index, entry) in fs::read_dir(&self.directory)
            .context("Could not list saved drafts")?
            .enumerate()
        {
            if index >= MAX_SCOPES * 4 + 16 {
                bail!("Draft directory contains too many entries; its files were preserved");
            }
            let entry = entry.context("Could not inspect saved draft entry")?;
            let metadata =
                fs::symlink_metadata(entry.path()).context("Could not inspect saved draft")?;
            if metadata.file_type().is_file() {
                bytes = bytes.saturating_add(metadata.len());
            }
            let path = entry.path();
            if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("json" | "bak")
            ) {
                if let Some(key) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(scope) = parse_scope_key(key) {
                        scopes.insert(key.to_owned(), scope);
                    }
                }
            }
        }
        if scopes.len() > MAX_SCOPES || bytes > MAX_DISK_BYTES {
            bail!("Draft storage exceeds its bounded size; its files were preserved");
        }
        Ok((scopes, bytes))
    }

    fn sequence(&self, scopes: &BTreeMap<String, Option<String>>) -> Result<u64> {
        if let Ok(Some(bytes)) = read_bounded(&self.directory.join(".sequence"), 32) {
            if let Ok(sequence) = std::str::from_utf8(&bytes).unwrap_or("").parse::<u64>() {
                return Ok(sequence);
            }
        }
        // Recover an absent/corrupt allocator without trusting wall-clock time.
        let mut sequence = 0;
        for (key, scope) in scopes {
            if let Some((record, _)) = self.read_existing(key, scope.as_deref())? {
                sequence = sequence.max(record.revision);
            }
        }
        Ok(sequence)
    }

    fn publish(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let temporary = path.with_extension("tmp");
        // A stable temp name is touched only while holding the process-safe lock.
        match fs::symlink_metadata(&temporary) {
            Ok(metadata) if metadata.file_type().is_file() && !is_reparse_point(&metadata) => {
                fs::remove_file(&temporary)
                    .context("Could not remove interrupted draft temporary file")?;
            }
            Ok(_) => bail!("Draft temporary path is not a regular file"),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("Could not inspect draft temporary path"),
        }
        if let Ok(metadata) = fs::symlink_metadata(path) {
            if !metadata.file_type().is_file() || is_reparse_point(&metadata) {
                bail!("Draft destination is not a regular file");
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
            .context("Could not create draft temporary file")?;
        let cleanup = TemporaryFile(temporary.clone());
        file.write_all(bytes)
            .context("Could not write saved draft")?;
        file.sync_all().context("Could not flush saved draft")?;
        drop(file);
        fs::rename(&temporary, path).context("Could not publish saved draft")?;
        drop(cleanup);
        #[cfg(unix)]
        File::open(&self.directory)?
            .sync_all()
            .context("Could not flush draft directory")?;
        Ok(())
    }
}

fn scope_key(session_id: Option<&str>) -> Result<String> {
    let Some(session_id) = session_id else {
        return Ok("new".into());
    };
    if session_id.is_empty() || session_id.len() > MAX_SCOPE_BYTES {
        bail!("Draft session ID must contain between 1 and 96 UTF-8 bytes");
    }
    // Lowercase hex is collision-free on Windows's case-insensitive filesystem
    // and keeps arbitrary session IDs out of path components.
    use fmt::Write;
    let mut key = String::from("s-");
    for byte in session_id.bytes() {
        write!(key, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(key)
}

fn parse_scope_key(key: &str) -> Option<Option<String>> {
    if key == "new" {
        return Some(None);
    }
    let hex = key.strip_prefix("s-")?;
    if hex.is_empty() || hex.len() % 2 != 0 || hex.len() > MAX_SCOPE_BYTES * 2 {
        return None;
    }
    let bytes: Option<Vec<u8>> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| match b {
                b'0'..=b'9' => Some(b - b'0'),
                b'a'..=b'f' => Some(b - b'a' + 10),
                _ => None,
            };
            Some(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect();
    Some(Some(String::from_utf8(bytes?).ok()?))
}

fn read_bounded(path: &Path, limit: usize) -> Result<Option<Vec<u8>>> {
    match file_io::read_regular(path, limit) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(ReadError::Inspect(error)) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(ReadError::Inspect(error)) => Err(error).context("Could not inspect saved draft file"),
        Err(ReadError::NotRegular) => bail!("Saved draft path must be a regular file"),
        Err(ReadError::TooLarge | ReadError::GrewTooLarge) => {
            bail!("Saved draft file exceeds its size limit");
        }
        Err(ReadError::Open(error)) => Err(error).context("Could not open saved draft file"),
        Err(ReadError::OpenedMetadata(error)) => Err(error.into()),
        Err(ReadError::ChangedType) => bail!("Saved draft file changed while opening it"),
        Err(ReadError::Read(error)) => Err(error).context("Could not read saved draft file"),
    }
}

fn encode_bounded(record: &Record, limit: usize) -> Result<Vec<u8>> {
    struct BoundedBuffer {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl Write for BoundedBuffer {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if buffer.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(io::Error::other(
                    "Saved draft exceeds its encoded size limit",
                ));
            }
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut buffer, record)
        .context("Could not encode saved draft within 32 MiB")?;
    Ok(buffer.bytes)
}

struct TemporaryFile(PathBuf);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct DraftLock {
    _file: File,
}

impl DraftLock {
    fn acquire(path: &Path) -> Result<Self> {
        let started = Instant::now();
        loop {
            match Self::try_acquire(path) {
                Ok(file) => return Ok(Self { _file: file }),
                Err(error) if lock_contended(&error) => {
                    if started.elapsed() >= LOCK_TIMEOUT {
                        bail!("Draft storage is busy in another terminal; input remains in memory");
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(error).context("Could not lock saved drafts"),
            }
        }
    }

    fn try_acquire(path: &Path) -> io::Result<File> {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            if !metadata.file_type().is_file() || is_reparse_point(&metadata) {
                return Err(io::Error::other("Draft lock must be a regular file"));
            }
        }
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: the file owns a live descriptor until the guard is dropped.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(not(any(windows, unix)))]
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Draft locking is unavailable",
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
    fn bounded_reader_keeps_optional_missing_files_and_draft_diagnostics() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot");
        assert!(read_bounded(&path, 1).unwrap().is_none());
        assert_eq!(
            read_bounded(directory.path(), 1).unwrap_err().to_string(),
            "Saved draft path must be a regular file"
        );
        fs::write(&path, b"too large").unwrap();
        assert_eq!(
            read_bounded(&path, 1).unwrap_err().to_string(),
            "Saved draft file exceeds its size limit"
        );
    }

    fn snapshot(text: &str) -> DraftSnapshot {
        DraftSnapshot {
            composer: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn empty_reads_do_not_create_files() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        assert!(store.load(None).unwrap().is_none());
        assert!(store.load_latest().unwrap().is_none());
        assert!(store.list_pending().unwrap().is_empty());
        assert!(!workspace.path().join(".aster").exists());
    }

    #[test]
    fn whitespace_only_composer_is_saved_and_listed_for_recovery() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let scope = provisional_scope("whitespace");
        let saved = snapshot(" \t\n  ");
        assert!(saved.composer.is_empty());
        assert!(!saved.is_empty());
        let revision = store.save(Some(&scope), None, &saved).unwrap();
        let loaded = store.load(Some(&scope)).unwrap().unwrap();
        assert_eq!(loaded.revision, revision);
        assert_eq!(loaded.snapshot.composer, saved.composer);
        let pending = store.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].session_id.as_deref(), Some(scope.as_str()));
        assert_eq!(pending[0].snapshot.composer, saved.composer);
    }

    #[test]
    fn independent_provisional_scopes_keep_inputs_separate_and_preserve_legacy_drafts() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let first = provisional_scope("first");
        let second = provisional_scope("second");
        assert!(is_provisional_scope(None));
        assert!(is_provisional_scope(Some(&first)));
        assert!(!is_provisional_scope(Some("session")));
        store.save(None, None, &snapshot("legacy input")).unwrap();
        let revision = store
            .save(Some(&first), None, &snapshot("first input"))
            .unwrap();
        let second_input = DraftSnapshot {
            queued: vec!["second queued input".into()],
            ..Default::default()
        };
        store.save(Some(&second), None, &second_input).unwrap();
        store
            .save(
                Some("session"),
                None,
                &DraftSnapshot {
                    in_flight: Some("interrupted input".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        let attached = DraftSnapshot {
            composer: MessageDraft {
                text: String::new(),
                attachments: vec![Attachment::Text {
                    name: "context.txt".into(),
                    uri: "file:///saved/context.txt".into(),
                    text: "saved context".into(),
                }],
            },
            ..Default::default()
        };
        store.save(Some("attached"), None, &attached).unwrap();
        store
            .save(Some("empty"), None, &DraftSnapshot::default())
            .unwrap();

        let sequence = fs::read(store.directory.join(".sequence")).unwrap();
        let pending = store.list_pending().unwrap();
        assert_eq!(
            pending
                .iter()
                .map(|draft| draft.session_id.as_deref())
                .collect::<Vec<_>>(),
            [
                Some("attached"),
                Some("session"),
                Some(second.as_str()),
                Some(first.as_str()),
                None
            ]
        );
        assert!(pending
            .windows(2)
            .all(|pair| pair[0].revision > pair[1].revision));
        assert!(pending.iter().all(|draft| draft.snapshot.queue_paused));
        assert_eq!(pending[0].snapshot.composer, attached.composer);
        assert_eq!(
            pending[1].snapshot.in_flight.as_ref().unwrap().text,
            "interrupted input"
        );
        assert_eq!(pending[2].snapshot.queued, second_input.queued);
        assert_eq!(pending[4].snapshot.composer, "legacy input");
        assert_eq!(
            fs::read(store.directory.join(".sequence")).unwrap(),
            sequence
        );

        store
            .save(
                Some(&first),
                Some(revision),
                &snapshot("updated first input"),
            )
            .unwrap();
        assert!(store
            .save(Some(&first), Some(revision), &snapshot("stale input"))
            .is_err());
        assert_eq!(
            store.load(Some(&second)).unwrap().unwrap().snapshot.queued,
            second_input.queued
        );
        assert_eq!(
            store.load(None).unwrap().unwrap().snapshot.composer,
            "legacy input"
        );
    }

    #[test]
    fn provisional_transfers_list_the_published_session_once() {
        let local = provisional_scope("unsent");
        for scope in [None, Some(local.as_str())] {
            let workspace = tempfile::tempdir().unwrap();
            let store = DraftStore::for_workspace(workspace.path()).unwrap();
            let revision = store
                .save(scope, None, &snapshot("original input"))
                .unwrap();
            let saved = DraftSnapshot {
                in_flight: Some("original input".into()),
                queued: vec!["following input".into()],
                ..Default::default()
            };
            store.save(Some("session"), None, &saved).unwrap();
            store
                .save(
                    scope,
                    Some(revision),
                    &DraftSnapshot {
                        migrated_to: Some("session".into()),
                        ..Default::default()
                    },
                )
                .unwrap();

            let latest = store.load_latest().unwrap().unwrap();
            assert_eq!(latest.session_id.as_deref(), Some("session"));
            let pending = store.list_pending().unwrap();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].snapshot.in_flight, saved.in_flight);
            assert_eq!(pending[0].snapshot.queued, saved.queued);
            assert_eq!(pending[0].session_id.as_deref(), Some("session"));

            fs::remove_file(
                store
                    .directory
                    .join(format!("{}.json", scope_key(Some("session")).unwrap())),
            )
            .unwrap();
            assert!(store.load_latest().is_err());
            assert!(store.list_pending().is_err());
            assert_eq!(
                store
                    .load(scope)
                    .unwrap()
                    .unwrap()
                    .snapshot
                    .migrated_to
                    .as_deref(),
                Some("session")
            );
        }
    }

    #[test]
    fn transfers_reject_local_targets_missing_sessions_and_chains() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let source = provisional_scope("source");
        let local_target = provisional_scope("target");
        let revision = store
            .save(Some(&source), None, &snapshot("keep source"))
            .unwrap();
        store
            .save(Some(&local_target), None, &snapshot("keep target"))
            .unwrap();
        for target in [&local_target, "missing"] {
            assert!(store
                .save(
                    Some(&source),
                    Some(revision),
                    &DraftSnapshot {
                        migrated_to: Some(target.into()),
                        ..Default::default()
                    }
                )
                .is_err());
        }
        assert_eq!(
            store
                .load(Some(&source))
                .unwrap()
                .unwrap()
                .snapshot
                .composer,
            "keep source"
        );
        store
            .save(Some("target"), None, &snapshot("published input"))
            .unwrap();
        assert!(store
            .save(
                Some("another-session"),
                None,
                &DraftSnapshot {
                    migrated_to: Some("target".into()),
                    ..Default::default()
                }
            )
            .is_err());
        store
            .save(
                Some(&source),
                Some(revision),
                &DraftSnapshot {
                    migrated_to: Some("target".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        let target_path = store
            .directory
            .join(format!("{}.json", scope_key(Some("target")).unwrap()));
        let mut record: Record = serde_json::from_slice(&fs::read(&target_path).unwrap()).unwrap();
        record.snapshot = DraftSnapshot {
            migrated_to: Some("chain-target".into()),
            ..Default::default()
        };
        let invalid = serde_json::to_vec(&record).unwrap();
        fs::write(&target_path, &invalid).unwrap();
        assert!(store.load_latest().is_err());
        assert_eq!(fs::read(&target_path).unwrap(), invalid);
    }

    #[test]
    fn transfers_preserve_source_when_destination_only_has_an_older_backup() {
        for missing_primary in [false, true] {
            let workspace = tempfile::tempdir().unwrap();
            let store = DraftStore::for_workspace(workspace.path()).unwrap();
            let scope = provisional_scope("source");
            let revision = store
                .save(Some(&scope), None, &snapshot("original input"))
                .unwrap();
            let first = store
                .save(Some("session"), None, &snapshot("older destination"))
                .unwrap();
            store
                .save(Some("session"), Some(first), &snapshot("original input"))
                .unwrap();
            let source_path = store
                .directory
                .join(format!("{}.json", scope_key(Some(&scope)).unwrap()));
            let source_bytes = fs::read(&source_path).unwrap();
            let target_path = store
                .directory
                .join(format!("{}.json", scope_key(Some("session")).unwrap()));
            if missing_primary {
                fs::remove_file(&target_path).unwrap();
            } else {
                fs::write(&target_path, b"broken newest destination").unwrap();
            }
            assert!(store
                .load(Some("session"))
                .unwrap()
                .unwrap()
                .warning
                .is_some());
            let error = store
                .save(
                    Some(&scope),
                    Some(revision),
                    &DraftSnapshot {
                        migrated_to: Some("session".into()),
                        ..Default::default()
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains("source input was preserved"));
            assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
            assert_eq!(
                store.load(Some(&scope)).unwrap().unwrap().snapshot.composer,
                "original input"
            );
        }
    }

    #[test]
    fn scope_quota_reclaims_only_consumed_named_provisional_records() {
        assert_scope_quota_reclaims_empty_provisional_record(Some("session"));
    }

    #[test]
    fn scope_quota_reclaims_cleared_named_provisional_records() {
        assert_scope_quota_reclaims_empty_provisional_record(None);
    }

    fn assert_scope_quota_reclaims_empty_provisional_record(target: Option<&str>) {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let pending = provisional_scope("pending");
        let consumed = provisional_scope("zz-consumed");
        let legacy = store.save(None, None, &snapshot("legacy input")).unwrap();
        store
            .save(None, Some(legacy), &DraftSnapshot::default())
            .unwrap();
        store
            .save(Some(&pending), None, &snapshot("pending input"))
            .unwrap();
        let source = store
            .save(Some(&consumed), None, &snapshot("transferred input"))
            .unwrap();
        store
            .save(Some("session"), None, &snapshot("transferred input"))
            .unwrap();
        store
            .save(Some("empty-session"), None, &DraftSnapshot::default())
            .unwrap();
        for (name, saved) in [
            ("whitespace", snapshot(" \t\n ")),
            (
                "attached",
                DraftSnapshot {
                    composer: MessageDraft {
                        attachments: vec![Attachment::Text {
                            name: "context.txt".into(),
                            uri: "file:///saved/context.txt".into(),
                            text: "saved context".into(),
                        }],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
            (
                "queued",
                DraftSnapshot {
                    queued: vec!["queued input".into()],
                    ..Default::default()
                },
            ),
            (
                "interrupted",
                DraftSnapshot {
                    in_flight: Some("interrupted input".into()),
                    ..Default::default()
                },
            ),
        ] {
            store
                .save(Some(&provisional_scope(name)), None, &saved)
                .unwrap();
        }
        let damaged = provisional_scope("damaged");
        let revision = store
            .save(Some(&damaged), None, &snapshot("backup input"))
            .unwrap();
        store
            .save(Some(&damaged), Some(revision), &DraftSnapshot::default())
            .unwrap();
        fs::write(
            store
                .directory
                .join(format!("{}.json", scope_key(Some(&damaged)).unwrap())),
            b"unreadable primary",
        )
        .unwrap();
        for index in store.inventory().unwrap().0.len()..MAX_SCOPES {
            store
                .save(Some(&format!("session-{index}")), None, &snapshot("kept"))
                .unwrap();
        }
        let transferred = store
            .save(
                Some(&consumed),
                Some(source),
                &DraftSnapshot {
                    migrated_to: target.map(str::to_owned),
                    ..Default::default()
                },
            )
            .unwrap();
        let source_key = scope_key(Some(&consumed)).unwrap();
        fs::write(
            store.directory.join(format!("{source_key}.tmp")),
            b"partial",
        )
        .unwrap();
        let preserved: Vec<_> = fs::read_dir(&store.directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                matches!(
                    path.extension().and_then(|ext| ext.to_str()),
                    Some("json" | "bak")
                ) && path.file_stem().and_then(|stem| stem.to_str()) != Some(&source_key)
            })
            .map(|path| {
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        fs::remove_file(store.directory.join(".sequence")).unwrap();
        let incoming = provisional_scope("incoming");
        let revision = store
            .save(Some(&incoming), None, &snapshot("new input"))
            .unwrap();
        assert!(revision > transferred);
        for extension in ["json", "bak", "tmp"] {
            assert!(!store
                .directory
                .join(format!("{source_key}.{extension}"))
                .exists());
        }
        for (path, bytes) in preserved {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
        assert_eq!(store.inventory().unwrap().0.len(), MAX_SCOPES);
        assert!(store
            .save(
                Some(&consumed),
                Some(transferred),
                &snapshot("stale writer")
            )
            .is_err());
    }

    #[test]
    fn disk_quota_reclaims_multiple_consumed_sources_before_scope_limit() {
        assert_disk_quota_reclaims_empty_provisional_records(Some("session"));
    }

    #[test]
    fn disk_quota_reclaims_multiple_cleared_sources_before_scope_limit() {
        assert_disk_quota_reclaims_empty_provisional_records(None);
    }

    fn assert_disk_quota_reclaims_empty_provisional_records(target: Option<&str>) {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        store
            .save(Some("session"), None, &snapshot("kept"))
            .unwrap();
        let sources = [provisional_scope("first"), provisional_scope("second")];
        for source in &sources {
            let revision = store
                .save(Some(source), None, &snapshot(&"x".repeat(1024)))
                .unwrap();
            store
                .save(
                    Some(source),
                    Some(revision),
                    &DraftSnapshot {
                        migrated_to: target.map(str::to_owned),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let (_, stored_bytes) = store.inventory().unwrap();
        let padding = store.directory.join("quota-padding");
        File::create(&padding)
            .unwrap()
            .set_len(MAX_DISK_BYTES - stored_bytes - 512)
            .unwrap();
        let saved = snapshot(&"y".repeat(2300));
        store.save(Some("incoming"), None, &saved).unwrap();
        for source in sources {
            assert!(store.load(Some(&source)).unwrap().is_none());
        }
        assert_eq!(store.inventory().unwrap().0.len(), 2);
        assert_eq!(
            fs::metadata(padding).unwrap().len(),
            MAX_DISK_BYTES - stored_bytes - 512
        );
        assert_eq!(
            store
                .load(Some("incoming"))
                .unwrap()
                .unwrap()
                .snapshot
                .composer,
            saved.composer
        );
    }

    #[test]
    fn reclamation_rejects_stale_sources_legacy_scopes_and_damaged_destinations() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let scope = provisional_scope("source");
        let first = store.save(Some("session"), None, &snapshot("old")).unwrap();
        store
            .save(Some("session"), Some(first), &snapshot("new"))
            .unwrap();
        store
            .save(Some("other"), None, &snapshot("other input"))
            .unwrap();
        let marker = DraftSnapshot {
            migrated_to: Some("session".into()),
            ..Default::default()
        };
        store.save(None, None, &marker).unwrap();
        let revision = store
            .save(Some(&scope), None, &snapshot("original"))
            .unwrap();
        let revision = store.save(Some(&scope), Some(revision), &marker).unwrap();
        let source_path = store
            .directory
            .join(format!("{}.json", scope_key(Some(&scope)).unwrap()));
        let backup_path = source_path.with_extension("bak");
        let original = fs::read(&source_path).unwrap();
        let backup = fs::read(&backup_path).unwrap();
        let expected: Record = serde_json::from_slice(&original).unwrap();
        let lock = DraftLock::acquire(&store.directory.join(".lock")).unwrap();
        let legacy = store
            .read_record(&store.directory.join("new.json"), None)
            .unwrap()
            .unwrap();
        assert!(!store
            .reclaim_empty_provisional(&legacy, revision + 1, &lock)
            .unwrap());
        for changed_revision in [false, true] {
            let mut changed = expected.clone();
            if changed_revision {
                changed.revision += 1;
            } else {
                changed.snapshot.migrated_to = Some("other".into());
            }
            let bytes = serde_json::to_vec(&changed).unwrap();
            fs::write(&source_path, &bytes).unwrap();
            assert!(!store
                .reclaim_empty_provisional(&expected, revision + 1, &lock)
                .unwrap());
            assert_eq!(fs::read(&source_path).unwrap(), bytes);
            assert_eq!(fs::read(&backup_path).unwrap(), backup);
        }
        fs::write(&source_path, &original).unwrap();
        let target_path = store
            .directory
            .join(format!("{}.json", scope_key(Some("session")).unwrap()));
        let target_bytes = fs::read(&target_path).unwrap();
        fs::write(&target_path, b"broken newest destination").unwrap();
        assert!(store
            .load(Some("session"))
            .unwrap()
            .unwrap()
            .warning
            .is_some());
        assert!(!store
            .reclaim_empty_provisional(&expected, revision, &lock)
            .unwrap());
        assert_eq!(fs::read(&source_path).unwrap(), original);
        assert_eq!(fs::read(&backup_path).unwrap(), backup);
        fs::write(&target_path, &target_bytes).unwrap();
        fs::remove_file(store.directory.join(".sequence")).unwrap();
        assert!(store
            .reclaim_empty_provisional(&expected, revision, &lock)
            .unwrap());
        assert_eq!(
            fs::read_to_string(store.directory.join(".sequence")).unwrap(),
            revision.to_string()
        );
        assert!(store.load(Some(&scope)).unwrap().is_none());
        assert!(store.directory.join("new.json").exists());
    }

    #[test]
    fn pending_list_reports_backup_recovery_and_unreadable_records() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let first = store
            .save(None, None, &snapshot("last good input"))
            .unwrap();
        store
            .save(None, Some(first), &snapshot("newer input"))
            .unwrap();
        let healthy = store
            .save(Some("healthy"), None, &snapshot("other input"))
            .unwrap();
        fs::write(store.directory.join("new.json"), b"broken primary").unwrap();
        let pending = store.list_pending().unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[1].snapshot.composer, "last good input");
        assert!(pending[1]
            .warning
            .as_deref()
            .unwrap()
            .contains("last valid copy"));

        fs::write(store.directory.join("new.bak"), b"broken backup").unwrap();
        let pending = store.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].session_id.as_deref(), Some("healthy"));
        assert!(pending[0]
            .warning
            .as_deref()
            .unwrap()
            .contains("could not be read"));
        store
            .save(Some("healthy"), Some(healthy), &DraftSnapshot::default())
            .unwrap();
        assert!(store.list_pending().is_err());
        assert_eq!(
            fs::read(store.directory.join("new.json")).unwrap(),
            b"broken primary"
        );
        assert_eq!(
            fs::read(store.directory.join("new.bak")).unwrap(),
            b"broken backup"
        );
    }

    #[test]
    fn pending_list_does_not_hide_a_damaged_draft_with_an_empty_backup() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let revision = store.save(None, None, &DraftSnapshot::default()).unwrap();
        store
            .save(None, Some(revision), &snapshot("new input"))
            .unwrap();
        fs::write(store.directory.join("new.json"), b"broken").unwrap();
        assert!(store.list_pending().is_err());
        assert_eq!(
            fs::read(store.directory.join("new.json")).unwrap(),
            b"broken"
        );
    }

    #[test]
    fn restart_recovers_unicode_queue_and_original_inflight_without_resending() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let attachment = Attachment::Text {
            name: "selected.txt".into(),
            uri: "file:///removed/selected.txt".into(),
            text: "saved bytes".into(),
        };
        let mut saved = snapshot("  中文\nsecond  ");
        saved.cursor = (1, 4);
        saved.composer.attachments.push(attachment.clone());
        saved.queued = vec![
            "queued first".into(),
            MessageDraft {
                text: "queued with file".into(),
                attachments: vec![attachment],
            },
        ];
        saved.in_flight = Some("original still running".into());
        let revision = store.save(Some("session-1"), None, &saved).unwrap();
        let restarted = DraftStore::for_workspace(workspace.path()).unwrap();
        let loaded = restarted.load_latest().unwrap().unwrap();
        assert_eq!(loaded.session_id.as_deref(), Some("session-1"));
        assert_eq!(loaded.revision, revision);
        assert!(loaded.snapshot.queue_paused);
        saved.queue_paused = true;
        assert_eq!(loaded.snapshot, saved);
        assert!(loaded.warning.is_none());
        assert!(!workspace.path().join(".aster/sessions").exists());
    }

    #[test]
    fn latest_scope_includes_unsent_new_input_and_isolates_case_and_path_like_ids() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        for name in ["new", "A", "a", "../nested/中文"] {
            store.save(Some(name), None, &snapshot(name)).unwrap();
        }
        let revision = store.save(None, None, &snapshot("brand new")).unwrap();
        let latest = store.load_latest().unwrap().unwrap();
        assert!(latest.session_id.is_none());
        assert_eq!(latest.revision, revision);
        for name in ["new", "A", "a", "../nested/中文"] {
            assert_eq!(
                store
                    .load(Some(name))
                    .unwrap()
                    .unwrap()
                    .snapshot
                    .composer
                    .text,
                name
            );
        }
        let old = store.load(Some("A")).unwrap().unwrap();
        store
            .save(Some("A"), Some(old.revision), &snapshot("now newest"))
            .unwrap();
        assert_eq!(
            store.load_latest().unwrap().unwrap().session_id.as_deref(),
            Some("A")
        );
        let other = tempfile::tempdir().unwrap();
        assert!(DraftStore::for_workspace(other.path())
            .unwrap()
            .load_latest()
            .unwrap()
            .is_none());
    }

    #[test]
    fn concurrent_stale_writers_cannot_overwrite_each_others_unsent_input() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let revision = store.save(None, None, &snapshot("original")).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let threads: Vec<_> = (0..2)
            .map(|index| {
                let store = store.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.save(None, Some(revision), &snapshot(&format!("writer {index}")))
                })
            })
            .collect();
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results
            .iter()
            .filter_map(|r| r.as_ref().err())
            .all(|error| error.to_string().contains("another terminal")));
        assert!(store
            .load(None)
            .unwrap()
            .unwrap()
            .snapshot
            .composer
            .text
            .starts_with("writer "));
        assert!(store
            .save(None, None, &snapshot("stale new terminal"))
            .is_err());
    }

    #[test]
    fn corrupt_primary_recovers_backup_and_bad_records_are_never_treated_as_empty() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let first = store.save(None, None, &snapshot("last good")).unwrap();
        store.save(None, Some(first), &snapshot("later")).unwrap();
        fs::write(store.directory.join("new.json"), b"{partial").unwrap();
        let recovered = store.load(None).unwrap().unwrap();
        assert_eq!(recovered.revision, first);
        assert_eq!(recovered.snapshot.composer, "last good");
        assert!(recovered.warning.is_some());
        let next = store
            .save(None, Some(first), &snapshot("recovered and edited"))
            .unwrap();
        assert!(next > first + 1);
        assert!(store.load(None).unwrap().unwrap().warning.is_none());
        fs::write(store.directory.join("new.json"), b"broken").unwrap();
        fs::write(store.directory.join("new.bak"), b"also broken").unwrap();
        assert!(store.load(None).is_err());
        assert!(store
            .save(None, None, &snapshot("must not replace"))
            .is_err());
        assert_eq!(
            fs::read(store.directory.join("new.json")).unwrap(),
            b"broken"
        );
    }

    #[test]
    fn unsupported_schema_never_falls_back_to_an_older_backup_and_overwrites_it() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let first = store.save(None, None, &snapshot("old")).unwrap();
        store.save(None, Some(first), &snapshot("new")).unwrap();
        let path = store.directory.join("new.json");
        let mut record: Record = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record.schema_version += 1;
        fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(store.load(None).unwrap_err().is::<UnsupportedSchema>());
        assert!(store.load_latest().unwrap_err().is::<UnsupportedSchema>());
        assert!(store.list_pending().unwrap_err().is::<UnsupportedSchema>());
        assert!(store
            .save(None, Some(first), &snapshot("rejected"))
            .is_err());
    }

    #[test]
    fn interrupted_and_failed_publication_preserve_the_last_committed_draft() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let revision = store.save(None, None, &snapshot("kept")).unwrap();
        fs::write(store.directory.join("new.tmp"), "incomplete").unwrap();
        assert_eq!(store.load(None).unwrap().unwrap().snapshot.composer, "kept");
        let revision = store
            .save(None, Some(revision), &snapshot("saved"))
            .unwrap();
        assert!(!store.directory.join("new.tmp").exists());
        fs::create_dir(store.directory.join("new.tmp")).unwrap();
        assert!(store
            .save(None, Some(revision), &snapshot("failed"))
            .is_err());
        assert_eq!(
            store.load(None).unwrap().unwrap().snapshot.composer,
            "saved"
        );
    }

    #[test]
    fn recovery_allocator_is_monotonic_after_missing_sequence() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let first = store.save(Some("one"), None, &snapshot("one")).unwrap();
        let second = store.save(Some("two"), None, &snapshot("two")).unwrap();
        fs::remove_file(store.directory.join(".sequence")).unwrap();
        let third = store
            .save(Some("one"), Some(first), &snapshot("updated"))
            .unwrap();
        assert!(third > second);
        assert_eq!(
            store.load_latest().unwrap().unwrap().snapshot.composer,
            "updated"
        );
    }

    #[test]
    fn lock_contention_is_bounded_and_release_allows_saving() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let revision = store.save(None, None, &snapshot("kept")).unwrap();
        let lock = DraftLock::acquire(&store.directory.join(".lock")).unwrap();
        let started = Instant::now();
        let error = store
            .save(None, Some(revision), &snapshot("blocked"))
            .unwrap_err();
        assert!(error.to_string().contains("busy"));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(store.load(None).unwrap().unwrap().snapshot.composer, "kept");
        drop(lock);
        store
            .save(None, Some(revision), &snapshot("after release"))
            .unwrap();
    }

    #[test]
    fn rejects_invalid_snapshots_scopes_and_encoded_overflow() {
        let workspace = tempfile::tempdir().unwrap();
        let store = DraftStore::for_workspace(workspace.path()).unwrap();
        let mut saved = snapshot("valid");
        saved.queued = vec!["message".into(); MAX_QUEUED_DRAFTS + 1];
        assert!(store.save(None, None, &saved).is_err());
        assert!(store.save(Some(""), None, &snapshot("valid")).is_err());
        assert!(store
            .save(
                Some(&"x".repeat(MAX_SCOPE_BYTES + 1)),
                None,
                &snapshot("valid")
            )
            .is_err());
        let mut record = Record {
            schema_version: SCHEMA_VERSION,
            revision: 1,
            session_id: None,
            snapshot: snapshot(&"\u{0001}".repeat(128)),
        };
        assert!(encode_bounded(&record, 512).is_err());
        record.snapshot.composer = "small".into();
        assert!(encode_bounded(&record, 512).is_ok());
    }
}

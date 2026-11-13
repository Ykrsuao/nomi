// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bounded reads of regular files without following a replaced final path component.

use std::fs::{self, OpenOptions};
use std::io::{self, Read};
use std::path::Path;

/// Keep operation failures distinct so each store retains its own recovery policy and diagnostics.
#[derive(Debug)]
pub(crate) enum ReadError {
    Inspect(io::Error),
    NotRegular,
    TooLarge,
    Open(io::Error),
    OpenedMetadata(io::Error),
    ChangedType,
    Read(io::Error),
    GrewTooLarge,
}

pub(crate) fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

pub(crate) fn read_regular(path: &Path, limit: usize) -> Result<Vec<u8>, ReadError> {
    let metadata = fs::symlink_metadata(path).map_err(ReadError::Inspect)?;
    if !metadata.is_file() || is_link(&metadata) {
        return Err(ReadError::NotRegular);
    }
    if metadata.len() > limit as u64 {
        return Err(ReadError::TooLarge);
    }
    read_after_inspection(path, limit)
}

fn read_after_inspection(path: &Path, limit: usize) -> Result<Vec<u8>, ReadError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;

        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path).map_err(ReadError::Open)?;
    // The path may have changed since inspection. Check the opened object, not
    // the path again, before reading; Windows opens a reparse point itself.
    let metadata = file.metadata().map_err(ReadError::OpenedMetadata)?;
    if !metadata.is_file() || is_link(&metadata) {
        return Err(ReadError::ChangedType);
    }
    if metadata.len() > limit as u64 {
        return Err(ReadError::GrewTooLarge);
    }
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(ReadError::Read)?;
    if bytes.len() > limit {
        return Err(ReadError::GrewTooLarge);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_bytes_and_accepts_only_files_within_the_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot");
        let bytes = "  中文\r\n\0snapshot  \n".as_bytes();
        fs::write(&path, bytes).unwrap();
        assert_eq!(read_regular(&path, bytes.len()).unwrap(), bytes);
        assert!(matches!(
            read_regular(&path, bytes.len() - 1),
            Err(ReadError::TooLarge)
        ));
        fs::write(&path, []).unwrap();
        assert!(read_regular(&path, 0).unwrap().is_empty());
    }

    #[test]
    fn distinguishes_missing_paths_from_nonregular_files() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            read_regular(&directory.path().join("missing"), 100),
            Err(ReadError::Inspect(error)) if error.kind() == io::ErrorKind::NotFound
        ));
        assert!(matches!(
            read_regular(directory.path(), 100),
            Err(ReadError::NotRegular)
        ));
    }

    #[test]
    fn rejects_growth_and_directory_substitution_after_inspection() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot");
        fs::write(&path, b"a").unwrap();
        assert_eq!(fs::symlink_metadata(&path).unwrap().len(), 1);
        fs::write(&path, b"larger").unwrap();
        assert!(matches!(
            read_after_inspection(&path, 1),
            Err(ReadError::GrewTooLarge)
        ));

        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(matches!(
            read_after_inspection(&path, 100),
            Err(ReadError::Open(_) | ReadError::ChangedType)
        ));
    }

    #[cfg(any(windows, unix))]
    fn create_file_symlink(target: &Path, link: &Path) -> bool {
        #[cfg(windows)]
        let result = std::os::windows::fs::symlink_file(target, link);
        #[cfg(unix)]
        let result = std::os::unix::fs::symlink(target, link);
        match result {
            Ok(()) => true,
            #[cfg(windows)]
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::Unsupported
                ) || matches!(error.raw_os_error(), Some(1 | 50 | 1314)) =>
            {
                eprintln!("Skipping symlink fixture: this Windows host cannot create it ({error})");
                false
            }
            Err(error) => panic!("Could not create symlink fixture: {error}"),
        }
    }

    #[cfg(any(windows, unix))]
    #[test]
    fn rejects_symlinks_before_open_and_after_path_substitution() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("snapshot");
        let target = directory.path().join("other-file");
        fs::write(&path, b"reviewed").unwrap();
        fs::write(&target, b"must not be read").unwrap();
        assert!(fs::symlink_metadata(&path).unwrap().is_file());
        fs::remove_file(&path).unwrap();
        if !create_file_symlink(&target, &path) {
            return;
        }
        assert!(matches!(
            read_regular(&path, 100),
            Err(ReadError::NotRegular)
        ));
        // Exercise the open stage independently of the earlier path check,
        // representing a symlink swapped in immediately after that check.
        assert!(matches!(
            read_after_inspection(&path, 100),
            Err(ReadError::Open(_) | ReadError::ChangedType)
        ));
        fs::remove_file(&target).unwrap();
        assert!(matches!(
            read_regular(&path, 100),
            Err(ReadError::NotRegular)
        ));
    }
}

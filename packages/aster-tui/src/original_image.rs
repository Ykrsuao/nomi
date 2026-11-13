// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Open an attachment's original bytes without exposing source paths or changing its snapshot.

use std::fs::OpenOptions;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::attachments::Attachment;

/// Keep this value while the viewer may need the file. Dropping it removes the
/// private temporary directory; the external viewer is never owned or stopped.
#[derive(Debug)]
pub struct OriginalImage {
    _directory: tempfile::TempDir,
    path: PathBuf,
}

impl OriginalImage {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Save the exact attachment bytes, using only its validated type for the name.
pub fn prepare(attachment: &Attachment) -> Result<OriginalImage> {
    prepare_in(attachment, &std::env::temp_dir())
}

fn prepare_in(attachment: &Attachment, parent: &Path) -> Result<OriginalImage> {
    attachment.validate()?;
    let Attachment::Image {
        mime_type, data, ..
    } = attachment
    else {
        bail!("Choose an image attachment to open its original.");
    };
    let filename = match mime_type.as_str() {
        "image/png" => "original.png",
        "image/jpeg" => "original.jpg",
        "image/gif" => "original.gif",
        "image/webp" => "original.webp",
        _ => bail!("This image format cannot be opened."),
    };
    let bytes = STANDARD.decode(data).context("Invalid image encoding")?;
    let mut builder = tempfile::Builder::new();
    builder.prefix("aster-image-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    let directory = builder
        .tempdir_in(parent)
        .context("Could not create a temporary image directory")?;
    write_snapshot(directory, filename, &bytes)
}

fn write_snapshot(
    directory: tempfile::TempDir,
    filename: &str,
    bytes: &[u8],
) -> Result<OriginalImage> {
    let path = std::path::absolute(directory.path().join(filename))
        .context("Could not resolve the temporary image path")?;
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .context("Could not create the original image copy")?;
    file.write_all(bytes)
        .context("Could not save the original image copy")?;
    file.flush()
        .context("Could not finish saving the original image copy")?;
    drop(file);
    Ok(OriginalImage {
        _directory: directory,
        path,
    })
}

/// Prepare and launch outside the event loop. A successful result means the OS
/// accepted the launch, not that the user has closed the image viewer.
pub async fn open(attachment: Attachment) -> Result<OriginalImage> {
    open_with(attachment, platform::open).await
}

async fn open_with<F, Fut>(attachment: Attachment, launch: F) -> Result<OriginalImage>
where
    F: FnOnce(OriginalImage) -> Fut,
    Fut: Future<Output = Result<OriginalImage>>,
{
    let original = tokio::task::spawn_blocking(move || prepare(&attachment))
        .await
        .context("Image preparation was interrupted")??;
    launch(original).await
}

#[cfg(windows)]
mod platform {
    use std::io;
    use std::mem::size_of;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::System::Com::{
        CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
    };
    use windows_sys::Win32::UI::Shell::{
        ShellExecuteExW, SEE_MASK_FLAG_NO_UI, SEE_MASK_NOASYNC, SHELLEXECUTEINFOW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    use super::*;

    struct ComApartment;

    impl Drop for ComApartment {
        fn drop(&mut self) {
            // SAFETY: constructed only after successful initialization on this thread.
            unsafe { CoUninitialize() };
        }
    }

    pub(super) async fn open(original: OriginalImage) -> Result<OriginalImage> {
        tokio::task::spawn_blocking(move || {
            let filename = wide_path(original.path())?;
            // SAFETY: no reserved pointer is supplied; this blocking thread owns
            // its successful COM initialization and balances it before returning.
            let initialized = unsafe {
                CoInitializeEx(
                    std::ptr::null(),
                    (COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) as u32,
                )
            };
            if initialized < 0 {
                bail!("Could not initialize the image viewer (HRESULT {initialized:#010x}).");
            }
            let _apartment = ComApartment;
            let verb = [b'o' as u16, b'p' as u16, b'e' as u16, b'n' as u16, 0];
            let mut request = SHELLEXECUTEINFOW {
                cbSize: size_of::<SHELLEXECUTEINFOW>() as u32,
                fMask: SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI,
                lpVerb: verb.as_ptr(),
                lpFile: filename.as_ptr(),
                nShow: SW_SHOWNORMAL,
                ..Default::default()
            };
            // SAFETY: all UTF-16 strings are terminated and live until the
            // synchronous launch request returns. No viewer process is retained.
            if unsafe { ShellExecuteExW(&mut request) } == 0 {
                return Err(io::Error::last_os_error())
                    .context("Could not open the original image in its default app");
            }
            Ok(original)
        })
        .await
        .context("Image viewer launch was interrupted")?
    }

    pub(super) fn wide_path(path: &Path) -> Result<Vec<u16>> {
        let mut units: Vec<u16> = path.as_os_str().encode_wide().collect();
        if units.contains(&0) {
            bail!("The image path contains an invalid character.");
        }
        units.push(0);
        Ok(units)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod platform {
    use std::process::Stdio;
    use std::time::Duration;

    use tokio::process::Command;

    use super::*;

    pub(super) async fn open(original: OriginalImage) -> Result<OriginalImage> {
        let program = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        let mut child = Command::new(program)
            .arg(original.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("Could not start {program} to open the original image"))?;
        // Some desktop helpers remain alive until the viewer closes. Surface
        // immediate launch failures, then reap a slow helper without killing it.
        match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
            Ok(result) => {
                let status = result.context("Could not check the image viewer launch")?;
                if !status.success() {
                    bail!("{program} could not open the original image ({status}).");
                }
            }
            Err(_) => {
                tokio::spawn(async move {
                    let _ = child.wait().await;
                });
            }
        }
        Ok(original)
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod platform {
    use super::*;

    pub(super) async fn open(_original: OriginalImage) -> Result<OriginalImage> {
        bail!("Opening original images is not supported on this platform.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> (Attachment, Vec<u8>) {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 96, 80);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            let pixels: Vec<u8> = (0..96 * 80)
                .flat_map(|pixel| [(pixel % 251) as u8, 80, 160, 255])
                .collect();
            writer.write_image_data(&pixels).unwrap();
            writer.finish().unwrap();
        }
        let attachment = Attachment::from_image("../../中文 image.exe", &bytes).unwrap();
        (attachment, bytes)
    }

    #[test]
    fn saves_original_bytes_with_a_safe_name_and_removes_them_on_drop() {
        let (attachment, bytes) = image();
        let root = tempfile::tempdir().unwrap();
        let original = prepare_in(&attachment, root.path()).unwrap();
        let path = original.path().to_owned();
        assert_eq!(path.file_name().unwrap(), "original.png");
        assert!(path.is_absolute());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(attachment.content_block()["data"], STANDARD.encode(&bytes));
        drop(original);
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn derives_extensions_from_validated_image_types_not_attachment_names() {
        for (bytes, filename) in [
            (&b"\x89PNG\r\n\x1a\nimage"[..], "original.png"),
            (&b"\xff\xd8\xffimage"[..], "original.jpg"),
            (&b"GIF89aimage"[..], "original.gif"),
            (&b"RIFF\0\0\0\0WEBPimage"[..], "original.webp"),
        ] {
            let attachment = Attachment::from_image("../wrong.exe;other.jpg", bytes).unwrap();
            let original = prepare(&attachment).unwrap();
            assert_eq!(original.path().file_name().unwrap(), filename);
            assert_eq!(std::fs::read(original.path()).unwrap(), bytes);
        }
    }

    #[test]
    fn rejects_invalid_attachments_before_creating_any_files() {
        let root = tempfile::tempdir().unwrap();
        let invalid = [
            Attachment::Text {
                name: "file.txt".into(),
                uri: "file:///file.txt".into(),
                text: "text".into(),
            },
            Attachment::Image {
                name: "image.png".into(),
                mime_type: "image/png".into(),
                data: "invalid!".into(),
            },
            Attachment::Image {
                name: "image.png".into(),
                mime_type: "image/png".into(),
                data: STANDARD.encode(b"GIF89aimage"),
            },
        ];
        for attachment in invalid {
            assert!(prepare_in(&attachment, root.path()).is_err());
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        }
    }

    #[test]
    fn a_failed_write_removes_its_temporary_directory() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().to_owned();
        std::fs::write(path.join("original.png"), b"collision").unwrap();
        assert!(write_snapshot(directory, "original.png", b"image").is_err());
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn failed_launch_removes_the_copy_and_keeps_the_attachment() {
        let (attachment, bytes) = image();
        let mut observed = None;
        let result = open_with(attachment.clone(), |original| {
            observed = Some(original.path().to_owned());
            async move {
                assert!(original.path().is_file());
                bail!("simulated viewer launch failure");
            }
        })
        .await;
        assert!(result.unwrap_err().to_string().contains("simulated"));
        let path = observed.unwrap();
        assert!(!path.parent().unwrap().exists());
        assert_eq!(attachment.content_block()["data"], STANDARD.encode(bytes));
    }

    #[tokio::test]
    async fn successful_launch_retains_the_copy_until_the_caller_releases_it() {
        let (attachment, bytes) = image();
        let original = open_with(attachment, |original| async move {
            assert!(original.path().is_file());
            Ok(original)
        })
        .await
        .unwrap();
        let path = original.path().to_owned();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        drop(original);
        assert!(!path.parent().unwrap().exists());
    }

    #[cfg(unix)]
    #[test]
    fn image_and_directory_are_private_to_the_current_user() {
        use std::os::unix::fs::PermissionsExt;
        let original = prepare(&image().0).unwrap();
        for path in [original.path(), original.path().parent().unwrap()] {
            let permissions = std::fs::metadata(path).unwrap().permissions();
            assert_eq!(permissions.mode() & 0o077, 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_keep_unicode_and_spaces_and_reject_embedded_nuls() {
        let path = Path::new("C:\\中文 空格\\original.png");
        let expected: Vec<u16> = path.to_str().unwrap().encode_utf16().chain([0]).collect();
        assert_eq!(platform::wide_path(path).unwrap(), expected);
        assert!(platform::wide_path(Path::new("C:\\image\0hidden.png")).is_err());
    }
}

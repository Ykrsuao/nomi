// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native clipboard access for the isolated clipboard worker.

use std::io::{self, Write};
use std::mem::size_of;
use std::ptr;

use anyhow::{bail, Context, Result};
use windows_sys::Win32::Foundation::{GlobalFree, HGLOBAL, HWND};
use windows_sys::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
    RegisterClipboardFormatW, SetClipboardData,
};
use windows_sys::Win32::System::Memory::{
    GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
};
use windows_sys::Win32::System::Ole::{CF_DIB, CF_DIBV5, CF_UNICODETEXT};
use windows_sys::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, HWND_MESSAGE};

use super::ClipboardContent;
use crate::attachments::{Attachment, MAX_IMAGE_BYTES};

const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;
const MAX_PIXELS: usize = 16 * 1024 * 1024;
const MAX_DIB_BYTES: usize = MAX_PIXELS * 4 + 4096;

struct OpenClipboardGuard;

impl Drop for OpenClipboardGuard {
    fn drop(&mut self) {
        // SAFETY: This guard is only constructed after OpenClipboard succeeds.
        unsafe {
            CloseClipboard();
        }
    }
}

struct LockedGlobalGuard(HGLOBAL);

impl Drop for LockedGlobalGuard {
    fn drop(&mut self) {
        // SAFETY: The handle was locked successfully and its owner outlives this guard.
        unsafe {
            GlobalUnlock(self.0);
        }
    }
}

struct ClipboardOwner(HWND);

impl Drop for ClipboardOwner {
    fn drop(&mut self) {
        // SAFETY: This worker created the window and destroys it on the same thread.
        unsafe {
            DestroyWindow(self.0);
        }
    }
}

struct OwnedGlobal(HGLOBAL);

impl Drop for OwnedGlobal {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: Ownership has not been transferred; the block is unlocked.
            unsafe {
                GlobalFree(self.0);
            }
        }
    }
}

fn encode_write_text(text: &str) -> Result<Vec<u16>> {
    super::validate_write_text(text)?;
    Ok(text.encode_utf16().chain([0]).collect())
}

/// A private worker owns a message-only window for the clipboard transaction.
/// Allocate and fill the text before replacing any clipboard data.
pub(super) fn write_text(text: &str) -> Result<()> {
    let units = encode_write_text(text)?;
    // SAFETY: The bounded length includes the UTF-16 terminator. Clipboard data
    // must use movable global storage, which remains ours until SetClipboardData.
    let handle = unsafe { GlobalAlloc(GMEM_MOVEABLE, units.len() * size_of::<u16>()) };
    if handle.is_null() {
        return Err(io::Error::last_os_error()).context("Could not allocate clipboard text");
    }
    let mut memory = OwnedGlobal(handle);
    // SAFETY: The worker owns this live allocation.
    let pointer = unsafe { GlobalLock(memory.0) };
    if pointer.is_null() {
        return Err(io::Error::last_os_error()).context("Could not lock clipboard text");
    }
    let locked = LockedGlobalGuard(memory.0);
    // SAFETY: GlobalAlloc reserved exactly enough aligned storage; source and
    // destination allocations are distinct and live for the entire copy.
    unsafe {
        ptr::copy_nonoverlapping(units.as_ptr(), pointer.cast::<u16>(), units.len());
    }
    drop(locked);

    const STATIC_CLASS: [u16; 7] = [83, 84, 65, 84, 73, 67, 0];
    // SAFETY: STATIC is a predefined class. The parent creates a message-only
    // window with no visible UI; no custom window procedure or pointers are used.
    let window = unsafe {
        CreateWindowExW(
            0,
            STATIC_CLASS.as_ptr(),
            ptr::null(),
            0,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
        )
    };
    if window.is_null() {
        return Err(io::Error::last_os_error()).context("Could not create clipboard owner");
    }
    let owner = ClipboardOwner(window);
    // SAFETY: The owner remains alive until after the clipboard guard closes it.
    if unsafe { OpenClipboard(owner.0) } == 0 {
        return Err(io::Error::last_os_error()).context("Could not open the Windows clipboard");
    }
    let _clipboard = OpenClipboardGuard;
    // SAFETY: The clipboard is open with a valid owner and replacement data is ready.
    if unsafe { EmptyClipboard() } == 0 {
        return Err(io::Error::last_os_error()).context("Could not prepare clipboard text");
    }
    // SAFETY: The unlocked, terminated UTF-16 block becomes system-owned on success.
    if unsafe { SetClipboardData(u32::from(CF_UNICODETEXT), memory.0) }.is_null() {
        return Err(io::Error::last_os_error()).context("Could not set clipboard text");
    }
    memory.0 = ptr::null_mut();
    Ok(())
}

/// Read text without changing clipboard contents. The caller isolates potentially
/// blocking delayed rendering in a worker process with a deadline.
pub(super) fn read_text() -> Result<String> {
    let _clipboard = open()?;
    read_text_open()
}

pub(super) fn read() -> Result<ClipboardContent> {
    let _clipboard = open()?;
    // SAFETY: a terminated UTF-16 constant registers the standard PNG name;
    // registration does not change or take ownership of the clipboard contents.
    let png_format = unsafe { RegisterClipboardFormatW([80_u16, 78, 71, 0].as_ptr()) };
    let mut image_error = None;
    for format in [png_format, u32::from(CF_DIBV5), u32::from(CF_DIB)] {
        // SAFETY: the clipboard remains open while testing and reading formats.
        if format == 0 || unsafe { IsClipboardFormatAvailable(format) } == 0 {
            continue;
        }
        let image = if format == png_format {
            copy_bytes(format, MAX_IMAGE_BYTES)
        } else {
            copy_bytes(format, MAX_DIB_BYTES).and_then(|bytes| dib_to_png(&bytes))
        };
        match image.and_then(|bytes| Attachment::from_image("clipboard.png", &bytes)) {
            Ok(attachment) => return Ok(ClipboardContent::Image(attachment)),
            Err(error) => image_error = Some(error),
        }
    }
    // SAFETY: the clipboard is still held open by the guard.
    if unsafe { IsClipboardFormatAvailable(u32::from(CF_UNICODETEXT)) } != 0 {
        return read_text_open().map(ClipboardContent::Text);
    }
    Err(image_error.unwrap_or_else(|| {
        anyhow::anyhow!("The clipboard does not contain readable text or a supported image")
    }))
}

fn open() -> Result<OpenClipboardGuard> {
    // SAFETY: A null owner is valid for read-only clipboard access.
    if unsafe { OpenClipboard(ptr::null_mut()) } == 0 {
        return Err(io::Error::last_os_error()).context("Could not open the Windows clipboard");
    }
    Ok(OpenClipboardGuard)
}

fn read_text_open() -> Result<String> {
    // SAFETY: The clipboard remains open for the duration of these calls.
    if unsafe { IsClipboardFormatAvailable(u32::from(CF_UNICODETEXT)) } == 0 {
        bail!("The clipboard does not contain text");
    }
    // SAFETY: The format is available and the clipboard is open. The returned
    // memory belongs to the clipboard and must never be freed by this worker.
    let handle = unsafe { GetClipboardData(u32::from(CF_UNICODETEXT)) };
    if handle.is_null() {
        return Err(io::Error::last_os_error()).context("Could not read Windows clipboard text");
    }
    // SAFETY: CF_UNICODETEXT supplies a global-memory handle.
    let size = unsafe { GlobalSize(handle) };
    let units = block_units(size)?;
    // SAFETY: The handle stays valid while the clipboard is open.
    let pointer = unsafe { GlobalLock(handle) };
    if pointer.is_null() {
        return Err(io::Error::last_os_error()).context("Could not lock Windows clipboard text");
    }
    // Declared after _clipboard so the memory is unlocked before the clipboard closes.
    let _locked = LockedGlobalGuard(handle);
    // SAFETY: GlobalLock returns aligned storage of GlobalSize bytes, kept locked
    // throughout decoding. Ignore any odd trailing allocation-padding byte.
    let text = unsafe { std::slice::from_raw_parts(pointer.cast::<u16>(), units) };
    decode_text(text, MAX_TEXT_BYTES)
}

fn copy_bytes(format: u32, limit: usize) -> Result<Vec<u8>> {
    // SAFETY: the caller keeps the clipboard open; these image formats use an
    // HGLOBAL owned by the clipboard, which must never be freed by this worker.
    let handle = unsafe { GetClipboardData(format) };
    if handle.is_null() {
        return Err(io::Error::last_os_error()).context("Could not read Windows clipboard image");
    }
    // SAFETY: image data is provided as a global-memory block.
    let size = unsafe { GlobalSize(handle) };
    if size == 0 || size > limit {
        bail!("Windows clipboard image exceeds its bounded memory limit or is empty");
    }
    // SAFETY: the global-memory block stays owned by the open clipboard.
    let pointer = unsafe { GlobalLock(handle) };
    if pointer.is_null() {
        return Err(io::Error::last_os_error()).context("Could not lock Windows clipboard image");
    }
    let _locked = LockedGlobalGuard(handle);
    // SAFETY: the locked block has GlobalSize readable bytes; its size was
    // validated before constructing the slice or allocating the snapshot.
    Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) }.to_vec())
}

/// Decode only common uncompressed true-color DIBs. Dimensions, row padding,
/// masks and memory extents are checked before any allocation or pixel access.
fn dib_to_png(bytes: &[u8]) -> Result<Vec<u8>> {
    fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
        let bytes = bytes
            .get(offset..offset + 2)
            .context("Truncated clipboard bitmap header")?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
        let bytes = bytes
            .get(offset..offset + 4)
            .context("Truncated clipboard bitmap header")?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    if bytes.len() > MAX_DIB_BYTES {
        bail!("Clipboard bitmap exceeds its memory limit");
    }
    let header = u32_at(bytes, 0)? as usize;
    if !matches!(header, 40 | 52 | 56 | 108 | 124) || bytes.len() < header {
        bail!("Unsupported or truncated clipboard bitmap header");
    }
    let width = u32_at(bytes, 4)? as i32;
    let signed_height = u32_at(bytes, 8)? as i32;
    if width <= 0 || signed_height == 0 {
        bail!("Clipboard bitmap has invalid dimensions");
    }
    let width = width as usize;
    let height = signed_height.unsigned_abs() as usize;
    if width > 16_384
        || height > 16_384
        || width
            .checked_mul(height)
            .is_none_or(|pixels| pixels > MAX_PIXELS)
    {
        bail!("Clipboard bitmap exceeds the 16-megapixel limit");
    }
    let bits = u16_at(bytes, 14)?;
    let compression = u32_at(bytes, 16)?;
    if u16_at(bytes, 12)? != 1
        || !matches!(bits, 24 | 32)
        || !matches!(compression, 0 | 3 | 6)
        || (bits == 24 && compression != 0)
    {
        bail!("Clipboard bitmap must be uncompressed 24-bit or 32-bit color");
    }
    let mut offset = header;
    let masks = if compression == 0 {
        [0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0]
    } else {
        if header == 40 {
            offset += if compression == 6 { 16 } else { 12 };
        } else if compression == 6 && header < 56 {
            bail!("Clipboard bitmap alpha mask is missing");
        }
        [
            u32_at(bytes, 40)?,
            u32_at(bytes, 44)?,
            u32_at(bytes, 48)?,
            if header >= 56 || compression == 6 {
                u32_at(bytes, 52)?
            } else {
                0
            },
        ]
    };
    let mut used = 0;
    for (index, mask) in masks.iter().copied().enumerate() {
        if mask == 0 {
            if index == 3 {
                continue;
            }
            bail!("Clipboard bitmap color mask is empty");
        }
        let shifted = mask >> mask.trailing_zeros();
        if mask & used != 0 || shifted & shifted.wrapping_add(1) != 0 {
            bail!("Clipboard bitmap color masks overlap or are not contiguous");
        }
        used |= mask;
    }
    let palette = u32_at(bytes, 32)? as usize;
    if palette > 256 {
        bail!("Clipboard bitmap has an unsupported color table");
    }
    offset += palette * 4;
    // DIBV5 profiles need color-management handling beyond raw RGB conversion.
    if header == 124 && (u32_at(bytes, 112)? != 0 || u32_at(bytes, 116)? != 0) {
        bail!("Clipboard bitmap with an embedded color profile is unsupported; copy it as PNG");
    }
    let stride = (width * usize::from(bits)).div_ceil(32) * 4;
    let pixel_bytes = stride
        .checked_mul(height)
        .context("Clipboard bitmap rows overflow")?;
    let end = offset
        .checked_add(pixel_bytes)
        .context("Clipboard bitmap extent overflows")?;
    if end > bytes.len() {
        bail!("Clipboard bitmap pixel data is truncated");
    }
    let declared_bytes = u32_at(bytes, 20)? as usize;
    if declared_bytes != 0
        && (declared_bytes < pixel_bytes || declared_bytes > bytes.len() - offset)
    {
        bail!("Clipboard bitmap declares an invalid pixel data size");
    }
    let mut rgba = Vec::with_capacity(width * height * 4);
    let pixel_size = usize::from(bits / 8);
    for y in 0..height {
        let source_y = if signed_height > 0 { height - 1 - y } else { y };
        let row = &bytes[offset + source_y * stride..offset + (source_y + 1) * stride];
        for x in 0..width {
            let pixel = &row[x * pixel_size..(x + 1) * pixel_size];
            let value = u32::from(pixel[0])
                | (u32::from(pixel[1]) << 8)
                | (u32::from(pixel[2]) << 16)
                | if pixel_size == 4 {
                    u32::from(pixel[3]) << 24
                } else {
                    0
                };
            rgba.extend(masks.map(|mask| component(value, mask)));
        }
    }
    struct PngBuffer(Vec<u8>);
    impl Write for PngBuffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > MAX_IMAGE_BYTES.saturating_sub(self.0.len()) {
                return Err(io::Error::other(
                    "Encoded clipboard PNG exceeds the 8 MiB limit",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = PngBuffer(Vec::new());
    let mut encoder = png::Encoder::new(&mut output, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder
        .write_header()
        .context("Could not encode clipboard PNG")?;
    writer
        .write_image_data(&rgba)
        .context("Could not encode clipboard image within 8 MiB")?;
    writer.finish().context("Could not finish clipboard PNG")?;
    Ok(output.0)
}

fn component(value: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 255;
    }
    let shift = mask.trailing_zeros();
    let maximum = u64::from(mask >> shift);
    let value = u64::from((value & mask) >> shift);
    ((value * 255 + maximum / 2) / maximum) as u8
}

fn block_units(bytes: usize) -> Result<usize> {
    if bytes < size_of::<u16>() {
        bail!("Windows clipboard text is empty or invalid");
    }
    if bytes > MAX_TEXT_BYTES {
        bail!("Clipboard text exceeds the 8 MiB limit");
    }
    Ok(bytes / size_of::<u16>())
}

fn decode_text(units: &[u16], max_bytes: usize) -> Result<String> {
    let Some(end) = units.iter().position(|unit| *unit == 0) else {
        bail!("Windows clipboard text has no terminating NUL");
    };
    let mut text = String::with_capacity(end.min(max_bytes));
    for unit in char::decode_utf16(units[..end].iter().copied()) {
        let character = unit.context("Windows clipboard text contains invalid UTF-16")?;
        if character.len_utf8() > max_bytes.saturating_sub(text.len()) {
            bail!("Clipboard text exceeds the 8 MiB limit");
        }
        text.push(character);
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dib(width: i32, height: i32, bits: u16, header: usize, compression: u32) -> Vec<u8> {
        let mut bytes = vec![0; header];
        bytes[0..4].copy_from_slice(&(header as u32).to_le_bytes());
        bytes[4..8].copy_from_slice(&width.to_le_bytes());
        bytes[8..12].copy_from_slice(&height.to_le_bytes());
        bytes[12..14].copy_from_slice(&1_u16.to_le_bytes());
        bytes[14..16].copy_from_slice(&bits.to_le_bytes());
        bytes[16..20].copy_from_slice(&compression.to_le_bytes());
        bytes
    }

    fn png_pixels(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
        let decoder = png::Decoder::new(bytes);
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut pixels).unwrap();
        pixels.truncate(info.buffer_size());
        (info.width, info.height, pixels)
    }

    #[test]
    fn converts_padded_bottom_up_24_bit_bitmap_to_png() {
        let mut bytes = dib(2, 2, 24, 40, 0);
        bytes.extend([255, 0, 0, 0, 255, 0, 0, 0]); // blue, green, padding
        bytes.extend([0, 0, 255, 255, 255, 255, 0, 0]); // red, white, padding
        let (width, height, pixels) = png_pixels(&dib_to_png(&bytes).unwrap());
        assert_eq!((width, height), (2, 2));
        assert_eq!(
            pixels,
            [255, 0, 0, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 255,]
        );
    }

    #[test]
    fn supports_top_down_masked_alpha_and_opaque_legacy_32_bit_bitmaps() {
        let mut legacy = dib(1, -1, 32, 40, 0);
        legacy.extend([3, 2, 1, 0]);
        assert_eq!(png_pixels(&dib_to_png(&legacy).unwrap()).2, [1, 2, 3, 255]);
        let mut masked = dib(1, -1, 32, 56, 3);
        for (index, mask) in [0x0000_00ff_u32, 0x0000_ff00, 0x00ff_0000, 0xff00_0000]
            .iter()
            .enumerate()
        {
            masked[40 + index * 4..44 + index * 4].copy_from_slice(&mask.to_le_bytes());
        }
        masked.extend([1, 2, 3, 128]);
        assert_eq!(png_pixels(&dib_to_png(&masked).unwrap()).2, [1, 2, 3, 128]);
    }

    #[test]
    fn rejects_truncated_or_excessive_bitmap_data_before_allocating_pixels() {
        for bytes in [
            vec![],
            dib(i32::MAX, 1, 32, 40, 0),
            dib(1, i32::MIN, 32, 40, 0),
            dib(16_384, 16_384, 32, 40, 0),
            dib(1, 1, 8, 40, 0),
            dib(1, 1, 32, 40, 0),
            dib(1, 1, 24, 40, 3),
        ] {
            assert!(dib_to_png(&bytes).is_err());
        }
        let mut overlap = dib(1, 1, 32, 56, 3);
        for index in 0..3 {
            overlap[40 + index * 4..44 + index * 4].copy_from_slice(&255_u32.to_le_bytes());
        }
        overlap.extend([0; 4]);
        assert!(dib_to_png(&overlap).is_err());
    }

    #[test]
    fn preserves_unicode_newlines_and_trailing_whitespace() {
        let expected = "  中文 🦀\r\nsecond line\t \r\n";
        let mut units: Vec<u16> = expected.encode_utf16().collect();
        units.extend([0, 0xD800]);
        assert_eq!(decode_text(&units, MAX_TEXT_BYTES).unwrap(), expected);
    }

    #[test]
    fn write_encoding_preserves_text_and_checks_windows_allocation_size() {
        let text = "  中文👩‍💻\nline\t \r\n";
        let units = encode_write_text(text).unwrap();
        assert_eq!(units.last(), Some(&0));
        assert_eq!(decode_text(&units, MAX_TEXT_BYTES).unwrap(), text);
        assert_eq!(encode_write_text("").unwrap(), vec![0]);
        assert!(encode_write_text("embedded\0nul").is_err());
        assert!(encode_write_text(&"x".repeat(MAX_TEXT_BYTES / 2)).is_err());
    }

    #[test]
    fn empty_text_is_valid_and_nul_ends_the_text() {
        assert_eq!(decode_text(&[0], 0).unwrap(), "");
        assert_eq!(decode_text(&[b'a' as u16, 0, b'b' as u16], 1).unwrap(), "a");
    }

    #[test]
    fn rejects_missing_terminator_and_unpaired_surrogates() {
        for units in [&[b'a' as u16][..], &[0xD800, 0], &[0xDC00, 0]] {
            assert!(decode_text(units, MAX_TEXT_BYTES).is_err());
        }
    }

    #[test]
    fn bounds_decoded_utf8_bytes_instead_of_code_units() {
        let units = [0x4E2D, 0x6587, 0];
        assert_eq!(decode_text(&units, 6).unwrap(), "中文");
        assert!(decode_text(&units, 5).is_err());
        assert!(decode_text(&[b'a' as u16, 0], 0).is_err());
    }

    #[test]
    fn bounds_memory_before_creating_a_slice() {
        assert!(block_units(0).is_err());
        assert!(block_units(1).is_err());
        assert_eq!(block_units(3).unwrap(), 1);
        assert_eq!(block_units(MAX_TEXT_BYTES).unwrap(), MAX_TEXT_BYTES / 2);
        assert!(block_units(MAX_TEXT_BYTES + 1).is_err());
        assert!(block_units(usize::MAX).is_err());
    }
}

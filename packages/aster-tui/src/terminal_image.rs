// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bounded, worker-ready Sixel previews for the attachment inspector.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};

use icy_sixel::{
    dither::sixel_dither as SixelDither, output::sixel_output as SixelOutput, DiffusionMethod,
    MethodForLargest, MethodForRep, PixelFormat, Quality,
};
use ratatui::{buffer::Buffer, layout::Rect, style::Color};

use crate::attachments::{Attachment, MAX_IMAGE_BYTES};
use crate::image_preview::decode_png_pixels;

const MAX_PIXEL_DIMENSION: usize = 1024;
const MAX_SIXEL_BYTES: usize = 8 * 1024 * 1024;
const MIN_SIXEL_HEIGHT: usize = 6;

/// Pixel geometry reported by the terminal, including virtual Sixel pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellPixels {
    pub width: u16,
    pub height: u16,
}

impl CellPixels {
    pub fn new(width: u16, height: u16) -> Option<Self> {
        ((1..=512).contains(&width) && (1..=512).contains(&height))
            .then_some(Self { width, height })
    }
}

/// An opaque job identity. A failed worker must retain its own key so it cannot
/// replace the result of a more recent attachment or resize request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderKey {
    content: u64,
    encoded_len: usize,
    max_width: usize,
    max_height: usize,
    cell_pixels: CellPixels,
    background: [u8; 3],
}

impl RenderKey {
    pub fn failed(self, message: String) -> RenderResult {
        RenderResult {
            key: self,
            result: Err(message),
        }
    }
}

#[derive(Debug)]
pub struct RenderRequest {
    key: RenderKey,
    attachment: Attachment,
}

impl RenderRequest {
    pub fn key(&self) -> RenderKey {
        self.key.clone()
    }

    /// Run outside the UI loop: PNG decoding and palette encoding can both be
    /// substantial work. This method performs no terminal or filesystem I/O.
    pub fn render(self) -> RenderResult {
        let result = encode(&self.attachment, &self.key);
        RenderResult {
            key: self.key,
            result,
        }
    }
}

#[derive(Debug)]
pub struct RenderResult {
    key: RenderKey,
    result: Result<SixelImage, String>,
}

#[derive(Debug)]
struct CachedImage {
    key: RenderKey,
    result: Result<SixelImage, String>,
}

/// The caller can choose a fallback without duplicating Sixel viewport limits.
#[derive(Debug, PartialEq, Eq)]
pub enum PreviewState<'a> {
    /// This attachment or viewport cannot use a native image preview.
    Unsupported,
    /// A render for the requested image and viewport is queued or running.
    Pending,
    Ready(&'a SixelImage),
}

/// Keeps one completed image/error and at most one latest queued request. The
/// application owns the worker, so rapid resizes never spawn unbounded jobs.
#[derive(Debug, Default)]
pub struct PreviewCache {
    desired: Option<RenderKey>,
    pending: Option<RenderKey>,
    queued: Option<RenderRequest>,
    entry: Option<CachedImage>,
}

impl PreviewCache {
    pub fn get_or_request(
        &mut self,
        attachment: &Attachment,
        columns: u16,
        rows: u16,
        cell_pixels: CellPixels,
        background: [u8; 3],
    ) -> Result<PreviewState<'_>, String> {
        let Attachment::Image {
            mime_type, data, ..
        } = attachment
        else {
            self.hide();
            return Ok(PreviewState::Unsupported);
        };
        // A second cell guards Ratatui 0.29's treatment of an escape payload as
        // a wide symbol. That cell is cleared together with the image below.
        if mime_type != "image/png"
            || columns < 2
            || rows == 0
            || CellPixels::new(cell_pixels.width, cell_pixels.height).is_none()
        {
            self.hide();
            return Ok(PreviewState::Unsupported);
        }
        if data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
            self.hide();
            return Err("PNG is too large to preview (8 MiB encoded image limit).".into());
        }
        let max_width =
            (usize::from(columns) * usize::from(cell_pixels.width)).min(MAX_PIXEL_DIMENSION);
        let max_height =
            (usize::from(rows) * usize::from(cell_pixels.height)).min(MAX_PIXEL_DIMENSION);
        if max_height < MIN_SIXEL_HEIGHT {
            self.hide();
            return Ok(PreviewState::Unsupported);
        }
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let key = RenderKey {
            content: hasher.finish(),
            encoded_len: data.len(),
            max_width,
            max_height,
            cell_pixels,
            background,
        };
        self.desired = Some(key.clone());
        if self.entry.as_ref().is_some_and(|entry| entry.key == key) {
            self.pending = None;
            self.queued = None;
            return match &self.entry.as_ref().expect("cache key was checked").result {
                Ok(image) => Ok(PreviewState::Ready(image)),
                Err(error) => Err(error.clone()),
            };
        }
        if self.pending.as_ref() != Some(&key) {
            self.pending = Some(key.clone());
            self.queued = Some(RenderRequest {
                key,
                attachment: attachment.clone(),
            });
        }
        Ok(PreviewState::Pending)
    }

    pub fn take_request(&mut self) -> Option<RenderRequest> {
        self.queued.take()
    }

    pub fn complete(&mut self, completed: RenderResult) -> bool {
        if self.desired.as_ref() != Some(&completed.key)
            || self.pending.as_ref() != Some(&completed.key)
        {
            return false;
        }
        self.pending = None;
        self.queued = None;
        self.entry = Some(CachedImage {
            key: completed.key,
            result: completed.result,
        });
        true
    }

    /// Borrow only the currently requested image, never a stale cached result.
    pub fn ready(&self) -> Option<&SixelImage> {
        let entry = self.entry.as_ref()?;
        if self.desired.as_ref() != Some(&entry.key) {
            return None;
        }
        entry.result.as_ref().ok()
    }

    pub fn hide(&mut self) {
        self.desired = None;
        self.pending = None;
        self.queued = None;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct SixelImage {
    pub pixel_width: u16,
    /// Includes background padding when the source is shorter than six pixels.
    pub pixel_height: u16,
    /// Reserved cell footprint, including a guard for a one-column image.
    pub columns: u16,
    pub rows: u16,
    background: [u8; 3],
    sequence: String,
}

impl SixelImage {
    /// Render after the inspector's text, and only while it is the top overlay.
    /// The caller must first select Sixel scrolling mode (DECSDM reset), which
    /// places the image at the current cursor rather than the page origin.
    pub fn render(&self, area: Rect, buffer: &mut Buffer) {
        let footprint = Rect::new(area.x, area.y, self.columns, self.rows);
        if self.columns > area.width
            || self.rows > area.height
            || !buffer.area.contains(footprint.as_position())
            || footprint.right() > buffer.area.right()
            || footprint.bottom() > buffer.area.bottom()
        {
            return;
        }
        for y in footprint.top()..footprint.bottom() {
            for x in footprint.left()..footprint.right() {
                if let Some(cell) = buffer.cell_mut((x, y)) {
                    cell.set_skip(true);
                }
            }
        }
        if let Some(cell) = buffer.cell_mut((area.x, area.y)) {
            let [r, g, b] = self.background;
            cell.set_symbol(&self.sequence)
                .set_bg(Color::Rgb(r, g, b))
                .set_skip(false);
        }
    }
}

fn encode(attachment: &Attachment, key: &RenderKey) -> Result<SixelImage, String> {
    attachment
        .validate()
        .map_err(|error| format!("PNG preview is unavailable: {error}."))?;
    let Attachment::Image { data, .. } = attachment else {
        return Err("This attachment has no PNG preview.".into());
    };
    let mut pixels = decode_png_pixels(data, key.max_width, key.max_height, key.background)?;
    // icy_sixel 0.1.3 emits an initial graphics newline for heights below six.
    // Pad that case within the checked viewport instead of shifting the image.
    if pixels.height < MIN_SIXEL_HEIGHT {
        let extra = pixels.width * (MIN_SIXEL_HEIGHT - pixels.height);
        pixels
            .pixels
            .try_reserve_exact(extra)
            .map_err(|_| "Not enough memory to create the PNG preview.".to_owned())?;
        pixels
            .pixels
            .resize(pixels.width * MIN_SIXEL_HEIGHT, key.background);
        pixels.height = MIN_SIXEL_HEIGHT;
    }
    let data = sixel_data(&pixels.pixels, pixels.width, pixels.height)?;
    let columns = pixels
        .width
        .div_ceil(usize::from(key.cell_pixels.width))
        .max(2) as u16;
    let rows = pixels.height.div_ceil(usize::from(key.cell_pixels.height)) as u16;
    // Clear all occupied cells before painting. Skipped cells are not written
    // by Ratatui, so without ECH old text can show beside partial image cells.
    let erase_row = format!("\x1b[{columns}X");
    let mut sequence = String::with_capacity(data.len() + usize::from(rows) * 12 + 32);
    sequence.push_str("\x1b7");
    for row in 0..rows {
        sequence.push_str(&erase_row);
        if row + 1 < rows {
            sequence.push_str("\x1b[B");
        }
    }
    sequence.push_str("\x1b8");
    sequence.push_str(&data);
    // Crossterm assumes a printed buffer cell advances one column. Sixel moves
    // the cursor vertically, so restore the saved origin and that one column.
    sequence.push_str("\x1b8\x1b[1C");
    if sequence.len() > MAX_SIXEL_BYTES {
        return Err("PNG preview exceeds the 8 MiB terminal image limit.".into());
    }
    Ok(SixelImage {
        pixel_width: pixels.width as u16,
        pixel_height: pixels.height as u16,
        columns,
        rows,
        background: key.background,
        sequence,
    })
}

struct BoundedOutput(Vec<u8>);

impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_SIXEL_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other(
                "PNG preview exceeds the 8 MiB terminal image limit.",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn sixel_data(pixels: &[[u8; 3]], width: usize, height: usize) -> Result<String, String> {
    let encode_error = |error| format!("PNG preview could not be encoded: {error}.");
    let mut dither = SixelDither::new(256).map_err(encode_error)?;
    dither.set_optimize_palette(true);
    // This pinned encoder samples every sixth pixel even in FULL mode for
    // small inputs. Repeating each pixel in the palette input preserves thin
    // strokes that its default sampling can miss completely. At our 1 Mpixel
    // bound this takes at most 18 MiB; larger samples use a denser library scan.
    let mut palette_input = Vec::new();
    palette_input
        .try_reserve_exact(pixels.len() * 18)
        .map_err(|_| "Not enough memory to prepare the PNG palette.".to_owned())?;
    for pixel in pixels {
        for _ in 0..6 {
            palette_input.extend_from_slice(pixel);
        }
    }
    dither
        .initialize(
            &palette_input,
            (width * 6) as i32,
            height as i32,
            PixelFormat::RGB888,
            MethodForLargest::Auto,
            MethodForRep::Auto,
            Quality::FULL,
        )
        .map_err(encode_error)?;
    drop(palette_input);
    // FULL is only needed while making the palette. The encoder's normal
    // palette cache is required by its encoding path and avoids costly scans.
    dither.set_quality_mode(Quality::HIGH);
    if dither.get_num_of_histogram_colors() > 256 {
        dither.set_diffusion_type(DiffusionMethod::Stucki);
    }
    let mut rgb = pixels.as_flattened().to_vec();
    let mut encoded = BoundedOutput(Vec::new());
    SixelOutput::new(&mut encoded)
        .encode(&mut rgb, width as i32, height as i32, 0, &mut dither)
        .map_err(encode_error)?;
    String::from_utf8(encoded.0).map_err(|_| "PNG preview encoding was not valid UTF-8.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, pixels: &[u8]) -> Attachment {
        let mut data = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut data, width, height);
            encoder.set_color(png::ColorType::Rgba);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            writer.finish().unwrap();
        }
        Attachment::from_image("preview.png", &data).unwrap()
    }

    fn geometry() -> CellPixels {
        CellPixels::new(10, 20).unwrap()
    }

    fn assert_palette_color(actual: [u8; 3], expected: [u8; 3]) {
        // The encoder quantizes its palette before converting channels to
        // Sixel percentages. Pixel positions must remain exact; colors may not.
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 10,
                "{actual} differs from {expected}"
            );
        }
    }

    fn request(cache: &mut PreviewCache, image: &Attachment, columns: u16) -> RenderRequest {
        assert_eq!(
            cache
                .get_or_request(image, columns, 10, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Pending
        );
        cache.take_request().expect("new image was queued")
    }

    /// Minimal decoder for the emitted standard Sixel subset. It reads raster
    /// attributes, color registers, repeat counts, carriage returns and bands;
    /// it does not share the encoder's fitting or palette implementation.
    fn raster(image: &SixelImage) -> (usize, usize, Vec<[u8; 3]>) {
        fn number(bytes: &[u8], cursor: &mut usize) -> usize {
            let mut value = 0;
            let start = *cursor;
            while *cursor < bytes.len() && bytes[*cursor].is_ascii_digit() {
                value = value * 10 + usize::from(bytes[*cursor] - b'0');
                *cursor += 1;
            }
            assert!(*cursor > start, "expected Sixel number");
            value
        }

        let start = image.sequence.find("\x1bPq").unwrap() + 3;
        let end = image.sequence[start..].find("\x1b\\").unwrap() + start;
        let bytes = &image.sequence.as_bytes()[start..end];
        assert_eq!(bytes[0], b'"');
        let mut cursor = 1;
        let mut attributes = Vec::new();
        loop {
            attributes.push(number(bytes, &mut cursor));
            if bytes.get(cursor) != Some(&b';') {
                break;
            }
            cursor += 1;
        }
        assert_eq!(&attributes[..2], &[1, 1]);
        let (width, height) = (attributes[2], attributes[3]);
        let mut pixels = vec![[0; 3]; width * height];
        let mut palette = [[0; 3]; 256];
        let (mut x, mut y, mut color) = (0, 0, 0);
        while cursor < bytes.len() {
            let command = bytes[cursor];
            cursor += 1;
            match command {
                b'#' => {
                    color = number(bytes, &mut cursor);
                    if bytes.get(cursor) == Some(&b';') {
                        let mut values = Vec::new();
                        while bytes.get(cursor) == Some(&b';') {
                            cursor += 1;
                            values.push(number(bytes, &mut cursor));
                        }
                        assert_eq!(values[0], 2, "RGB palette expected");
                        for (channel, value) in palette[color].iter_mut().zip(&values[1..]) {
                            *channel = ((value * 255 + 50) / 100) as u8;
                        }
                    }
                }
                b'$' => x = 0,
                b'-' => {
                    x = 0;
                    y += 6;
                }
                b'!' | b'?'..=b'~' => {
                    let (repeat, sixel) = if command == b'!' {
                        let repeat = number(bytes, &mut cursor);
                        let sixel = bytes[cursor];
                        cursor += 1;
                        (repeat, sixel)
                    } else {
                        (1, command)
                    };
                    assert!((b'?'..=b'~').contains(&sixel));
                    for offset in 0..repeat {
                        for bit in 0..6 {
                            if (sixel - b'?') & (1 << bit) != 0 {
                                assert!(x + offset < width && y + bit < height);
                                pixels[(y + bit) * width + x + offset] = palette[color];
                            }
                        }
                    }
                    x += repeat;
                }
                _ => panic!("unexpected Sixel command {command}"),
            }
        }
        (width, height, pixels)
    }

    #[test]
    fn renders_native_pixels_and_preserves_one_pixel_strokes() {
        let mut rgba = vec![255; 80 * 40 * 4];
        for y in 0..40 {
            rgba[(y * 80 + 21) * 4..(y * 80 + 21) * 4 + 3].fill(0);
        }
        let attachment = png(80, 40, &rgba);
        let original = attachment.clone();
        let mut cache = PreviewCache::default();
        assert_eq!(
            cache
                .get_or_request(&attachment, 8, 2, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Pending
        );
        let job = cache.take_request().unwrap();
        let result = std::thread::spawn(move || job.render()).join().unwrap();
        assert!(cache.complete(result));
        let image = cache.ready().unwrap();
        assert_eq!((image.pixel_width, image.pixel_height), (80, 40));
        assert_eq!((image.columns, image.rows), (8, 2));
        let (width, height, pixels) = raster(image);
        assert_eq!((width, height), (80, 40));
        for y in 0..height {
            assert_palette_color(pixels[y * width + 20], [255; 3]);
            assert_palette_color(pixels[y * width + 21], [0; 3]);
            assert_palette_color(pixels[y * width + 22], [255; 3]);
        }
        assert_eq!(attachment, original);
        let mut halfblocks = crate::image_preview::PreviewCache::default();
        let small = halfblocks
            .preview(&attachment, 8, 2, [0; 3])
            .unwrap()
            .unwrap();
        assert_eq!((small.width, small.height), (8, 4));
    }

    #[test]
    fn does_not_enlarge_and_pads_tiny_images_without_moving_pixels() {
        let attachment = png(2, 1, &[255, 0, 0, 255, 0, 255, 0, 0]);
        let mut cache = PreviewCache::default();
        cache
            .get_or_request(&attachment, 20, 10, geometry(), [0, 0, 255])
            .unwrap();
        let job = cache.take_request().unwrap();
        assert!(cache.complete(job.render()));
        let image = cache.ready().unwrap();
        assert_eq!((image.pixel_width, image.pixel_height), (2, 6));
        assert_eq!((image.columns, image.rows), (2, 1));
        let (_, _, pixels) = raster(image);
        assert_palette_color(pixels[0], [255, 0, 0]);
        for pixel in &pixels[1..] {
            assert_palette_color(*pixel, [0, 0, 255]);
        }
        assert!(image.sequence.starts_with("\x1b7\x1b[2X\x1b8\x1bP"));
        assert!(image.sequence.ends_with("\x1b\\\x1b8\x1b[1C"));
    }

    #[test]
    fn pending_jobs_are_deduplicated_and_only_the_latest_request_is_accepted() {
        let attachment = png(80, 40, &[255; 80 * 40 * 4]);
        let mut cache = PreviewCache::default();
        let first = request(&mut cache, &attachment, 8);
        assert_eq!(
            cache
                .get_or_request(&attachment, 8, 10, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Pending
        );
        assert!(cache.take_request().is_none());
        assert!(cache.ready().is_none());
        cache
            .get_or_request(&attachment, 5, 10, geometry(), [0; 3])
            .unwrap();
        cache
            .get_or_request(&attachment, 4, 10, geometry(), [0; 3])
            .unwrap();
        let newest = cache.take_request().unwrap();
        assert!(!cache.complete(first.render()));
        assert!(cache.ready().is_none());
        assert!(cache.complete(newest.render()));
        assert_eq!(cache.ready().unwrap().pixel_width, 40);
        assert!(matches!(
            cache
                .get_or_request(&attachment, 4, 10, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Ready(_)
        ));
        assert!(cache.take_request().is_none());
    }

    #[test]
    fn hide_discards_in_flight_results_and_keeps_a_completed_image_reusable() {
        let attachment = png(4, 8, &[255; 4 * 8 * 4]);
        let mut cache = PreviewCache::default();
        let abandoned = request(&mut cache, &attachment, 8);
        cache.hide();
        assert!(!cache.complete(abandoned.render()));
        assert!(cache.ready().is_none());
        let current = request(&mut cache, &attachment, 8);
        assert!(cache.complete(current.render()));
        cache.hide();
        assert!(cache.ready().is_none());
        assert!(matches!(
            cache
                .get_or_request(&attachment, 8, 10, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Ready(_)
        ));
        assert!(cache.take_request().is_none());
    }

    #[test]
    fn image_bytes_geometry_and_background_each_invalidate_the_cache() {
        let first = png(4, 8, &[255; 4 * 8 * 4]);
        let second = png(4, 8, &[0, 0, 0, 255].repeat(4 * 8));
        let mut cache = PreviewCache::default();
        let job = request(&mut cache, &first, 8);
        assert!(cache.complete(job.render()));
        let mut renamed = first.clone();
        if let Attachment::Image { name, .. } = &mut renamed {
            *name = "same pixels.png".into();
        }
        assert!(matches!(
            cache
                .get_or_request(&renamed, 8, 10, geometry(), [0; 3])
                .unwrap(),
            PreviewState::Ready(_)
        ));
        for (attachment, cells, background) in [
            (&second, geometry(), [0; 3]),
            (&second, CellPixels::new(8, 16).unwrap(), [0; 3]),
            (&second, CellPixels::new(8, 16).unwrap(), [255; 3]),
        ] {
            assert_eq!(
                cache
                    .get_or_request(attachment, 8, 10, cells, background)
                    .unwrap(),
                PreviewState::Pending
            );
            assert!(cache.ready().is_none());
            let job = cache.take_request().unwrap();
            assert!(cache.complete(job.render()));
        }
    }

    #[test]
    fn errors_are_cached_and_a_stale_worker_failure_does_not_poison_new_work() {
        let attachment = png(4, 8, &[255; 4 * 8 * 4]);
        let mut cache = PreviewCache::default();
        let old = request(&mut cache, &attachment, 8);
        let new = request(&mut cache, &attachment, 7);
        assert!(!cache.complete(old.key().failed("old worker stopped".into())));
        assert!(cache.complete(new.key().failed("worker stopped".into())));
        assert_eq!(
            cache.get_or_request(&attachment, 7, 10, geometry(), [0; 3]),
            Err("worker stopped".into())
        );
        assert!(cache.take_request().is_none());
        let invalid = Attachment::Image {
            name: "invalid.png".into(),
            mime_type: "image/png".into(),
            data: "not base64!".into(),
        };
        let invalid_job = request(&mut cache, &invalid, 8);
        assert!(cache.complete(invalid_job.render()));
        assert!(cache
            .get_or_request(&invalid, 8, 10, geometry(), [0; 3])
            .unwrap_err()
            .contains("PNG preview is unavailable"));
        assert!(cache.ready().is_none());
        assert!(cache.take_request().is_none());
    }

    #[test]
    fn unsupported_inputs_and_impossible_viewports_do_not_queue_or_leave_stale_images() {
        let attachment = png(4, 8, &[255; 4 * 8 * 4]);
        let unsupported = Attachment::Image {
            name: "photo.jpg".into(),
            mime_type: "image/jpeg".into(),
            data: "not decoded".into(),
        };
        let mut cache = PreviewCache::default();
        let job = request(&mut cache, &attachment, 8);
        assert!(cache.complete(job.render()));
        for (image, columns, rows, cells) in [
            (&unsupported, 8, 10, geometry()),
            (&attachment, 1, 10, geometry()),
            (&attachment, 8, 0, geometry()),
            (&attachment, 8, 1, CellPixels::new(2, 5).unwrap()),
            (
                &attachment,
                8,
                10,
                CellPixels {
                    width: 0,
                    height: 20,
                },
            ),
        ] {
            assert_eq!(
                cache
                    .get_or_request(image, columns, rows, cells, [0; 3])
                    .unwrap(),
                PreviewState::Unsupported
            );
            assert!(cache.take_request().is_none());
            assert!(cache.ready().is_none());
        }
        assert!(CellPixels::new(513, 20).is_none());
        assert!(CellPixels::new(10, 0).is_none());
    }

    #[test]
    fn output_and_input_budgets_apply_before_terminal_encoding() {
        let attachment = png(1200, 600, &vec![255; 1200 * 600 * 4]);
        let mut cache = PreviewCache::default();
        cache
            .get_or_request(&attachment, u16::MAX, u16::MAX, geometry(), [0; 3])
            .unwrap();
        let job = cache.take_request().unwrap();
        assert_eq!((job.key.max_width, job.key.max_height), (1024, 1024));
        assert!(cache.complete(job.render()));
        let image = cache.ready().unwrap();
        assert_eq!((image.pixel_width, image.pixel_height), (1024, 512));
        assert!(image.sequence.len() <= MAX_SIXEL_BYTES);
        let oversized = Attachment::Image {
            name: "large.png".into(),
            mime_type: "image/png".into(),
            data: "A".repeat(MAX_IMAGE_BYTES.div_ceil(3) * 4 + 4),
        };
        assert!(cache
            .get_or_request(&oversized, 8, 10, geometry(), [0; 3])
            .unwrap_err()
            .contains("8 MiB"));
        assert!(cache.take_request().is_none());
        assert!(cache.ready().is_none());
    }

    #[test]
    fn closing_moving_and_shrinking_repaint_every_uncovered_image_cell() {
        let attachment = png(40, 40, &[255; 40 * 40 * 4]);
        let mut cache = PreviewCache::default();
        let job = request(&mut cache, &attachment, 8);
        assert!(cache.complete(job.render()));
        let image = cache.ready().unwrap();
        let screen = Rect::new(0, 0, 20, 10);
        let base = Buffer::filled(screen, ratatui::buffer::Cell::new("x"));
        let area = Rect::new(3, 2, image.columns, image.rows);
        let mut shown = base.clone();
        image.render(area, &mut shown);
        let writes = base.diff(&shown);
        assert_eq!(
            writes
                .iter()
                .filter(|(_, _, cell)| cell.symbol().contains("\x1bP"))
                .count(),
            1
        );
        assert!(shown
            .diff(&shown)
            .iter()
            .all(|(_, _, cell)| !cell.symbol().contains("\x1bP")));
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                assert!(shown
                    .diff(&base)
                    .iter()
                    .any(|(cx, cy, _)| (*cx, *cy) == (x, y)));
            }
        }
        let mut moved = base.clone();
        image.render(Rect::new(4, 2, image.columns, image.rows), &mut moved);
        let moved_writes = shown.diff(&moved);
        assert!(moved_writes
            .iter()
            .any(|(x, y, cell)| (*x, *y) == (4, 2) && cell.symbol().contains("\x1bP")));
        assert!(moved_writes.iter().any(|(x, y, _)| (*x, *y) == (3, 2)));
        assert!(moved_writes.iter().any(|(x, y, _)| (*x, *y) == (3, 3)));
        let mut too_small = base.clone();
        image.render(Rect::new(3, 2, 1, 1), &mut too_small);
        assert_eq!(too_small, base);
        let mut outside = base.clone();
        image.render(Rect::new(19, 9, image.columns, image.rows), &mut outside);
        assert_eq!(outside, base);
    }
}

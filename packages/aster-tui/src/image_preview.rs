// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bounded PNG sampling shared by half-block and pixel-resolution previews.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Cursor;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::attachments::{Attachment, MAX_IMAGE_BYTES};

const MAX_DIMENSION: usize = 16_384;
const MAX_PIXELS: usize = 16_000_000;
const MAX_DECODE_BYTES: usize = 64 * 1024 * 1024;
const MAX_COLUMNS: usize = 512;
const MAX_ROWS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thumbnail {
    pub width: usize,
    /// Pixel rows, not terminal rows. An odd final row uses the UI's background below it.
    pub height: usize,
    /// Row-major, opaque RGB pixels already blended against the requested background.
    pub pixels: Vec<[u8; 3]>,
}

#[derive(Debug, PartialEq, Eq)]
struct CacheKey {
    content: u64,
    encoded_len: usize,
    columns: usize,
    pixel_rows: usize,
    background: [u8; 3],
}

#[derive(Debug)]
struct CachedPreview {
    key: CacheKey,
    result: Result<Thumbnail, String>,
}

/// Retains one bounded thumbnail or decode error; no source image is retained twice.
#[derive(Debug, Default)]
pub struct PreviewCache {
    entry: Option<CachedPreview>,
    #[cfg(test)]
    decode_attempts: usize,
}

impl PreviewCache {
    /// Non-PNG attachments and zero-size viewports use the metadata-only fallback.
    /// Half-block pixels are approximately square in a terminal with 2:1 cells.
    /// Images are fitted without enlargement to at most 512 columns and 256 rows.
    pub fn preview(
        &mut self,
        attachment: &Attachment,
        max_columns: usize,
        max_rows: usize,
        background: [u8; 3],
    ) -> Result<Option<&Thumbnail>, String> {
        let Attachment::Image {
            mime_type, data, ..
        } = attachment
        else {
            return Ok(None);
        };
        if mime_type != "image/png" || max_columns == 0 || max_rows == 0 {
            return Ok(None);
        }
        if data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
            return Err("PNG is too large to preview (8 MiB encoded image limit).".into());
        }
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let key = CacheKey {
            content: hasher.finish(),
            encoded_len: data.len(),
            columns: max_columns.min(MAX_COLUMNS),
            pixel_rows: max_rows.min(MAX_ROWS) * 2,
            background,
        };
        if self.entry.as_ref().is_none_or(|entry| entry.key != key) {
            #[cfg(test)]
            {
                self.decode_attempts += 1;
            }
            let result = decode_png_pixels(data, key.columns, key.pixel_rows, background);
            self.entry = Some(CachedPreview { key, result });
        }
        match &self
            .entry
            .as_ref()
            .expect("preview entry was initialized")
            .result
        {
            Ok(thumbnail) => Ok(Some(thumbnail)),
            Err(error) => Err(error.clone()),
        }
    }
}

fn dimensions(width: u32, height: u32) -> Result<(usize, usize), String> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(
            "PNG dimensions are too large to preview (16,384 pixels per side maximum).".into(),
        );
    }
    if width
        .checked_mul(height)
        .is_none_or(|pixels| pixels > MAX_PIXELS)
    {
        return Err("PNG is too large to preview (16 megapixels maximum).".into());
    }
    Ok((width, height))
}

/// Decode and fit PNG pixels without enlarging the source. Callers choose their
/// own output budget; the image/header limits apply to every preview protocol.
pub(crate) fn decode_png_pixels(
    data: &str,
    max_width: usize,
    max_height: usize,
    background: [u8; 3],
) -> Result<Thumbnail, String> {
    if max_width == 0 || max_height == 0 {
        return Err("PNG preview area is empty.".into());
    }
    if data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
        return Err("PNG is too large to preview (8 MiB encoded image limit).".into());
    }
    let encoded = STANDARD
        .decode(data)
        .map_err(|_| "PNG preview is unavailable: invalid base64 image data.".to_owned())?;
    if encoded.len() > MAX_IMAGE_BYTES {
        return Err("PNG is too large to preview (8 MiB image limit).".into());
    }
    let mut decoder = png::Decoder::new(Cursor::new(encoded));
    decoder.set_limits(png::Limits {
        bytes: MAX_DECODE_BYTES,
    });
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let header = decoder.read_header_info().map_err(decode_error)?;
    dimensions(header.width, header.height)?;
    let mut reader = decoder.read_info().map_err(decode_error)?;
    let size = reader.output_buffer_size();
    if size > MAX_DECODE_BYTES {
        return Err("PNG requires too much memory to preview (64 MiB output limit).".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| "Not enough memory to preview this PNG.".to_owned())?;
    bytes.resize(size, 0);
    let info = reader.next_frame(&mut bytes).map_err(decode_error)?;
    // Validate the remaining PNG structure without rendering further animation frames.
    reader.finish().map_err(decode_error)?;
    let (width, height) = dimensions(info.width, info.height)?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => {
            return Err("PNG palette could not be expanded for preview.".into())
        }
    };
    if info.bit_depth != png::BitDepth::Eight
        || info.line_size != width * channels
        || info.buffer_size() != width * height * channels
        || info.buffer_size() > bytes.len()
    {
        return Err("PNG has an unsupported or incomplete pixel layout.".into());
    }
    let (out_width, out_height) = fitted_dimensions(
        width,
        height,
        max_width.min(MAX_DIMENSION),
        max_height.min(MAX_DIMENSION),
    );
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(out_width * out_height)
        .map_err(|_| "Not enough memory to create the PNG thumbnail.".to_owned())?;
    for y in 0..out_height {
        let source_y = ((y as f64 + 0.5) * height as f64 / out_height as f64 - 0.5)
            .clamp(0.0, (height - 1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let fy = source_y - y0 as f64;
        for x in 0..out_width {
            let source_x = ((x as f64 + 0.5) * width as f64 / out_width as f64 - 0.5)
                .clamp(0.0, (width - 1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let fx = source_x - x0 as f64;
            let samples = [
                (x0, y0, (1.0 - fx) * (1.0 - fy)),
                (x1, y0, fx * (1.0 - fy)),
                (x0, y1, (1.0 - fx) * fy),
                (x1, y1, fx * fy),
            ];
            let mut pixel = [0.0; 3];
            for (x, y, weight) in samples {
                let start = (y * width + x) * channels;
                let source = &bytes[start..start + channels];
                let (rgb, alpha) = match source {
                    [gray] => ([*gray; 3], 255),
                    [gray, alpha] => ([*gray; 3], *alpha),
                    [r, g, b] => ([*r, *g, *b], 255),
                    [r, g, b, alpha] => ([*r, *g, *b], *alpha),
                    _ => unreachable!("channel count was checked"),
                };
                for channel in 0..3 {
                    let blended = (u32::from(rgb[channel]) * u32::from(alpha)
                        + u32::from(background[channel]) * (255 - u32::from(alpha)))
                        as f64
                        / 255.0;
                    pixel[channel] += blended * weight;
                }
            }
            pixels.push(pixel.map(|channel| channel.round().clamp(0.0, 255.0) as u8));
        }
    }
    Ok(Thumbnail {
        width: out_width,
        height: out_height,
        pixels,
    })
}

fn decode_error(error: png::DecodingError) -> String {
    match error {
        png::DecodingError::LimitsExceeded => {
            "PNG requires too much memory to preview (64 MiB decoder limit).".into()
        }
        _ => format!("PNG preview is unavailable: invalid or incomplete image ({error})."),
    }
}

fn fitted_dimensions(
    width: usize,
    height: usize,
    columns: usize,
    pixel_rows: usize,
) -> (usize, usize) {
    if width <= columns && height <= pixel_rows {
        (width, height)
    } else if width * pixel_rows > height * columns {
        (columns, (height * columns / width).max(1))
    } else {
        ((width * pixel_rows / height).max(1), pixel_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_image(
        width: u32,
        height: u32,
        color: png::ColorType,
        depth: png::BitDepth,
        pixels: &[u8],
    ) -> Attachment {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(color);
            encoder.set_depth(depth);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            writer.finish().unwrap();
        }
        Attachment::from_image("preview.png", &bytes).unwrap()
    }

    fn rgba(width: u32, height: u32, pixels: &[u8]) -> Attachment {
        png_image(
            width,
            height,
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            pixels,
        )
    }

    #[test]
    fn keeps_rgb_colors_and_odd_pixel_rows_without_enlarging() {
        let image = png_image(
            1,
            3,
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            &[255, 0, 0, 0, 255, 0, 0, 0, 255],
        );
        let mut cache = PreviewCache::default();
        let thumbnail = cache
            .preview(&image, 20, 20, [10, 20, 30])
            .unwrap()
            .unwrap();
        assert_eq!((thumbnail.width, thumbnail.height), (1, 3));
        assert_eq!(thumbnail.pixels, [[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
    }

    #[test]
    fn composites_transparent_and_partial_alpha_against_background() {
        let image = rgba(3, 1, &[255, 0, 0, 255, 255, 0, 0, 0, 255, 0, 0, 128]);
        let mut cache = PreviewCache::default();
        let thumbnail = cache
            .preview(&image, 20, 20, [10, 20, 30])
            .unwrap()
            .unwrap();
        assert_eq!(thumbnail.pixels, [[255, 0, 0], [10, 20, 30], [133, 10, 15]]);
    }

    #[test]
    fn expands_grayscale_low_bit_depth_and_strips_sixteen_bit_channels() {
        let cases = [
            (
                png_image(
                    2,
                    1,
                    png::ColorType::Grayscale,
                    png::BitDepth::One,
                    &[0b1000_0000],
                ),
                vec![[255; 3], [0; 3]],
            ),
            (
                png_image(
                    1,
                    1,
                    png::ColorType::GrayscaleAlpha,
                    png::BitDepth::Eight,
                    &[128, 128],
                ),
                vec![[69, 74, 79]],
            ),
            (
                png_image(
                    1,
                    1,
                    png::ColorType::Rgb,
                    png::BitDepth::Sixteen,
                    &[0x12, 0x34, 0xab, 0xcd, 0xff, 0x00],
                ),
                vec![[0x12, 0xab, 0xff]],
            ),
        ];
        let mut cache = PreviewCache::default();
        for (image, expected) in cases {
            assert_eq!(
                cache
                    .preview(&image, 20, 20, [10, 20, 30])
                    .unwrap()
                    .unwrap()
                    .pixels,
                expected
            );
        }
    }

    #[test]
    fn expands_indexed_palette_transparency() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Indexed);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_palette(vec![255, 0, 0, 0, 255, 0]);
            encoder.set_trns(vec![0, 128]);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[0, 1]).unwrap();
            writer.finish().unwrap();
        }
        let image = Attachment::from_image("palette.png", &bytes).unwrap();
        let mut cache = PreviewCache::default();
        assert_eq!(
            cache
                .preview(&image, 2, 1, [10, 20, 30])
                .unwrap()
                .unwrap()
                .pixels,
            [[10, 20, 30], [5, 138, 15]]
        );
    }

    #[test]
    fn fits_square_pixels_into_two_pixel_terminal_rows() {
        let mut cache = PreviewCache::default();
        for (width, height, columns, rows, expected) in [
            (8, 8, 8, 2, (4, 4)),
            (8, 4, 4, 2, (4, 2)),
            (4, 8, 10, 2, (2, 4)),
            (1, 100, 1, 1, (1, 2)),
        ] {
            let image = rgba(
                width,
                height,
                &[0, 255, 255, 255].repeat((width * height) as usize),
            );
            let thumbnail = cache
                .preview(&image, columns, rows, [0; 3])
                .unwrap()
                .unwrap();
            assert_eq!((thumbnail.width, thumbnail.height), expected);
            assert_eq!(thumbnail.pixels.len(), expected.0 * expected.1);
            assert!(thumbnail.pixels.iter().all(|pixel| *pixel == [0, 255, 255]));
        }
    }

    #[test]
    fn resizes_with_bilinear_color_and_alpha_sampling() {
        let image = rgba(
            2,
            2,
            &[
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
        );
        let mut cache = PreviewCache::default();
        assert_eq!(
            cache.preview(&image, 1, 1, [0; 3]).unwrap().unwrap().pixels,
            [[128; 3]]
        );
        let transparent = rgba(2, 1, &[255, 0, 0, 0, 0, 0, 255, 255]);
        assert_eq!(
            cache
                .preview(&transparent, 1, 1, [0; 3])
                .unwrap()
                .unwrap()
                .pixels,
            [[0, 0, 128]]
        );
    }

    #[test]
    fn metadata_fallback_and_empty_viewports_do_not_decode() {
        let mut cache = PreviewCache::default();
        let image = Attachment::Image {
            name: "other.jpg".into(),
            mime_type: "image/jpeg".into(),
            data: "not decoded".into(),
        };
        assert!(cache.preview(&image, 10, 10, [0; 3]).unwrap().is_none());
        let text = Attachment::Text {
            name: "text".into(),
            uri: "file:///text".into(),
            text: "text".into(),
        };
        assert!(cache.preview(&text, 10, 10, [0; 3]).unwrap().is_none());
        let image = rgba(1, 1, &[0; 4]);
        assert!(cache.preview(&image, 0, 10, [0; 3]).unwrap().is_none());
        assert!(cache.preview(&image, 10, 0, [0; 3]).unwrap().is_none());
        assert_eq!(cache.decode_attempts, 0);
    }

    #[test]
    fn rejects_invalid_encoding_corruption_and_truncation_and_caches_errors() {
        let mut cache = PreviewCache::default();
        for data in [
            "invalid!".to_owned(),
            STANDARD.encode(b"not a PNG"),
            STANDARD.encode(b"\x89PNG\r\n\x1a\n"),
        ] {
            let image = Attachment::Image {
                name: "bad.png".into(),
                mime_type: "image/png".into(),
                data,
            };
            let error = cache.preview(&image, 10, 10, [0; 3]).unwrap_err();
            let attempts = cache.decode_attempts;
            assert_eq!(cache.preview(&image, 10, 10, [0; 3]).unwrap_err(), error);
            assert_eq!(cache.decode_attempts, attempts);
            assert!(error.contains("PNG preview is unavailable"));
        }
        let Attachment::Image { mut data, .. } = rgba(1, 1, &[255; 4]) else {
            unreachable!()
        };
        let mut bytes = STANDARD.decode(&data).unwrap();
        bytes.truncate(bytes.len() - 12); // Missing IEND must not become a valid thumbnail.
        data = STANDARD.encode(bytes);
        let truncated = Attachment::Image {
            name: "truncated.png".into(),
            mime_type: "image/png".into(),
            data,
        };
        assert!(cache.preview(&truncated, 10, 10, [0; 3]).is_err());
    }

    #[test]
    fn rejects_oversized_encoded_images_and_headers_before_pixel_allocation() {
        let mut cache = PreviewCache::default();
        let oversized = Attachment::Image {
            name: "large.png".into(),
            mime_type: "image/png".into(),
            data: "A".repeat(MAX_IMAGE_BYTES.div_ceil(3) * 4 + 4),
        };
        assert!(cache
            .preview(&oversized, 10, 10, [0; 3])
            .unwrap_err()
            .contains("8 MiB"));
        assert_eq!(cache.decode_attempts, 0);
        for (width, height, reason) in [
            (MAX_DIMENSION as u32 + 1, 1, "dimensions"),
            (5000, 4000, "megapixels"),
        ] {
            let mut bytes = Vec::new();
            {
                let _writer = png::Encoder::new(&mut bytes, width, height)
                    .write_header()
                    .unwrap();
            }
            let image = Attachment::from_image("large.png", &bytes).unwrap();
            assert!(cache
                .preview(&image, 10, 10, [0; 3])
                .unwrap_err()
                .contains(reason));
        }
    }

    #[test]
    fn reuses_identical_content_and_keys_on_bounds_background_and_image_bytes() {
        let image = rgba(4, 4, &[100, 0, 0, 128].repeat(16));
        let mut cache = PreviewCache::default();
        cache.preview(&image, 10, 10, [0; 3]).unwrap();
        let mut renamed = image.clone();
        if let Attachment::Image { name, .. } = &mut renamed {
            *name = "same bytes under another name.png".into();
        }
        cache.preview(&renamed, 10, 10, [0; 3]).unwrap();
        assert_eq!(cache.decode_attempts, 1);
        cache.preview(&image, 2, 1, [0; 3]).unwrap();
        assert_eq!(cache.decode_attempts, 2);
        cache.preview(&image, 2, 1, [255; 3]).unwrap();
        assert_eq!(cache.decode_attempts, 3);
        cache
            .preview(&rgba(4, 4, &[0, 0, 100, 128].repeat(16)), 2, 1, [255; 3])
            .unwrap();
        assert_eq!(cache.decode_attempts, 4);
    }

    #[test]
    fn thumbnail_caps_prevent_large_viewport_allocation_and_overflow() {
        let image = rgba(600, 600, &[20, 40, 60, 255].repeat(600 * 600));
        let mut cache = PreviewCache::default();
        let thumbnail = cache
            .preview(&image, usize::MAX, usize::MAX, [0; 3])
            .unwrap()
            .unwrap();
        assert_eq!(
            (thumbnail.width, thumbnail.height),
            (MAX_COLUMNS, MAX_ROWS * 2)
        );
        assert_eq!(thumbnail.pixels.len(), MAX_COLUMNS * MAX_ROWS * 2);
        cache.preview(&image, 10_000, 10_000, [0; 3]).unwrap();
        assert_eq!(cache.decode_attempts, 1);
    }

    #[test]
    fn shared_decoder_accepts_pixel_bounds_without_enlargement() {
        let image = rgba(80, 40, &[10, 20, 30, 255].repeat(80 * 40));
        let Attachment::Image { data, .. } = image else {
            unreachable!()
        };
        for (bounds, expected) in [
            ((80, 40), (80, 40)),
            ((800, 400), (80, 40)),
            ((40, 40), (40, 20)),
        ] {
            let preview = decode_png_pixels(&data, bounds.0, bounds.1, [0; 3]).unwrap();
            assert_eq!((preview.width, preview.height), expected);
            assert!(preview.pixels.iter().all(|pixel| *pixel == [10, 20, 30]));
        }
        assert!(decode_png_pixels(&data, 0, 40, [0; 3]).is_err());
    }
}

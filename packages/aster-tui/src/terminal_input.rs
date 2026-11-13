// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Terminal input with native Windows bracketed-paste support.

use crate::terminal_image::CellPixels;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphicsCapabilities {
    pub sixel: bool,
    pub cell_pixels: Option<CellPixels>,
    /// DECSDM must be known so cursor-relative drawing can restore the user's mode.
    pub sixel_display_mode: Option<bool>,
}

impl GraphicsCapabilities {
    pub fn sixel_cell_pixels(self) -> Option<CellPixels> {
        self.cell_pixels
            .filter(|_| self.sixel && self.sixel_display_mode.is_some())
    }
}

#[derive(Debug)]
pub enum TerminalEvent {
    Input(crossterm::event::Event),
    Graphics(GraphicsCapabilities),
}

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::TerminalInput;

#[cfg(not(windows))]
pub struct TerminalInput(crossterm::event::EventStream);

#[cfg(not(windows))]
impl TerminalInput {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self(crossterm::event::EventStream::new()))
    }

    pub async fn next(&mut self) -> Option<std::io::Result<TerminalEvent>> {
        use futures_util::StreamExt;
        self.0
            .next()
            .await
            .map(|event| event.map(TerminalEvent::Input))
    }

    pub fn refresh_graphics(&self) -> std::io::Result<()> {
        // EventStream owns the Unix input parser. Keep the character preview
        // until that parser can safely consume graphics capability replies.
        Ok(())
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Scope the application title to the interactive terminal session.

pub(crate) struct Title {
    #[cfg(windows)]
    previous: Option<Vec<u16>>,
    #[cfg(not(windows))]
    saved: bool,
}

impl Title {
    pub(crate) fn enter() -> Self {
        #[cfg(windows)]
        {
            use windows_sys::Win32::{
                Foundation::{GetLastError, SetLastError},
                System::Console::{GetConsoleTitleW, SetConsoleTitleW},
            };

            let mut previous = vec![0u16; 32_768];
            // SAFETY: the buffer covers the supplied length and both titles are NUL-terminated.
            unsafe {
                SetLastError(0);
                let len = GetConsoleTitleW(previous.as_mut_ptr(), previous.len() as u32);
                if len == 0 && GetLastError() != 0 {
                    return Self { previous: None };
                }
                previous.truncate(len as usize + 1);
                let title: Vec<u16> = "Aster\0".encode_utf16().collect();
                SetConsoleTitleW(title.as_ptr());
            }
            Self {
                previous: Some(previous),
            }
        }
        #[cfg(not(windows))]
        {
            use std::io::Write;

            // Xterm-compatible terminals save and restore titles without a query
            // that could consume keystrokes from the application's input reader.
            let saved = std::io::stdout()
                .write_all(b"\x1b[22;0t\x1b]0;Aster\x07")
                .is_ok();
            Self { saved }
        }
    }
}

impl Drop for Title {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(previous) = &self.previous {
            // SAFETY: the captured title remains allocated and NUL-terminated.
            unsafe {
                windows_sys::Win32::System::Console::SetConsoleTitleW(previous.as_ptr());
            }
        }
        #[cfg(not(windows))]
        if self.saved {
            use std::io::Write;
            let mut output = std::io::stdout();
            let _ = output.write_all(b"\x1b[23;0t");
            let _ = output.flush();
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Read console records while preserving the VT paste boundaries that the
//! legacy Windows crossterm input backend discards.

use std::collections::HashSet;
use std::io::{self, Write};
use std::ptr;
use std::thread::{self, JoinHandle};

use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use termwiz::input::{InputEvent, InputParser, KeyCode as VtKey, Modifiers, MouseButtons};
use tokio::sync::mpsc;
use windows_sys::Win32::Foundation::{
    CloseHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, ReadConsoleInputW, SetConsoleMode,
    CONSOLE_SCREEN_BUFFER_INFO, ENABLE_VIRTUAL_TERMINAL_INPUT, FOCUS_EVENT, INPUT_RECORD,
    KEY_EVENT, KEY_EVENT_RECORD, MOUSE_EVENT, MOUSE_EVENT_RECORD, STD_INPUT_HANDLE,
    STD_OUTPUT_HANDLE, WINDOW_BUFFER_SIZE_EVENT,
};
use windows_sys::Win32::System::Threading::{
    CreateEventW, SetEvent, WaitForMultipleObjects, INFINITE,
};

use super::{GraphicsCapabilities, TerminalEvent};
use crate::terminal_image::CellPixels;

const ESCAPE_WAIT_MS: u32 = 25;
const MOUSE_ON: &[u8] = b"\x1b[?1000h\x1b[?1002h\x1b[?1006h";
const MOUSE_OFF: &[u8] = b"\x1b[?1006l\x1b[?1002l\x1b[?1000l";
const GRAPHICS_QUERY: &[u8] = b"\x1b[c\x1b[16t\x1b[?80$p";
const CELL_SIZE_QUERY: &[u8] = b"\x1b[16t";

pub struct TerminalInput {
    receiver: mpsc::Receiver<io::Result<TerminalEvent>>,
    reader: Option<JoinHandle<()>>,
    input: usize,
    stop: usize,
    original_vt: u32,
    original_sixel_display: Option<bool>,
}

impl TerminalInput {
    /// Call after enabling raw mode and mouse capture. This reader is the sole
    /// owner of console input until it is dropped.
    pub fn new() -> io::Result<Self> {
        // SAFETY: GetStdHandle returns a borrowed process handle, not memory.
        let input = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
        if input.is_null() || input == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let mut mode = 0;
        // SAFETY: input is a valid handle and mode points to writable storage.
        if unsafe { GetConsoleMode(input, &mut mode) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: A null name/security descriptor creates a private manual-reset event.
        let stop = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
        if stop.is_null() {
            return Err(io::Error::last_os_error());
        }
        let (sender, receiver) = mpsc::channel(256);
        let mut result = Self {
            receiver,
            reader: None,
            input: input as usize,
            stop: stop as usize,
            original_vt: mode & ENABLE_VIRTUAL_TERMINAL_INPUT,
            original_sixel_display: None,
        };
        // SAFETY: Only the VT-input bit is added; the caller owns other mode bits.
        if unsafe { SetConsoleMode(input, mode | ENABLE_VIRTUAL_TERMINAL_INPUT) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut stdout = io::stdout().lock();
        stdout.write_all(MOUSE_ON)?;
        stdout.flush()?;
        drop(stdout);
        // Windows handles are integer capabilities usable from another thread.
        // The owner keeps both alive until the reader has joined.
        let input = result.input;
        let stop = result.stop;
        result.reader = Some(thread::Builder::new().name("terminal-input".into()).spawn(
            move || {
                if let Err(error) = read_loop(input as HANDLE, stop as HANDLE, &sender) {
                    let _ = sender.blocking_send(Err(error));
                }
            },
        )?);
        // The existing input reader consumes replies; startup never waits for
        // a terminal to answer or starts a competing reader.
        let mut stdout = io::stdout().lock();
        stdout.write_all(GRAPHICS_QUERY)?;
        stdout.flush()?;
        Ok(result)
    }

    /// `recv` is cancellation-safe: dropping a pending select branch does not
    /// consume an event or discard a partially assembled paste.
    pub async fn next(&mut self) -> Option<io::Result<TerminalEvent>> {
        let event = self.receiver.recv().await?;
        if let Ok(TerminalEvent::Graphics(capabilities)) = &event {
            if self.original_sixel_display.is_none() && capabilities.sixel_cell_pixels().is_some() {
                // Reset DECSDM to anchor Sixel at the cursor, not the page origin.
                // Preserve the reported mode because Windows Terminal shares it
                // between the primary and alternate screen.
                self.original_sixel_display = capabilities.sixel_display_mode;
                let mut stdout = io::stdout().lock();
                if let Err(error) = stdout.write_all(b"\x1b[?80l").and_then(|()| stdout.flush()) {
                    return Some(Err(error));
                }
            }
        }
        Some(event)
    }

    pub fn refresh_graphics(&self) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        stdout.write_all(CELL_SIZE_QUERY)?;
        stdout.flush()
    }
}

impl Drop for TerminalInput {
    fn drop(&mut self) {
        // Unblock both possible waits: the output channel and the console event.
        self.receiver.close();
        // SAFETY: stop remains owned and valid through the following join.
        unsafe { SetEvent(self.stop as HANDLE) };
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        let mut stdout = io::stdout().lock();
        if let Some(display) = self.original_sixel_display {
            let _ = stdout.write_all(if display { b"\x1b[?80h" } else { b"\x1b[?80l" });
        }
        let _ = stdout.write_all(MOUSE_OFF);
        let _ = stdout.flush();
        let mut mode = 0;
        // SAFETY: The borrowed console input handle remains valid for this lifetime.
        unsafe {
            if GetConsoleMode(self.input as HANDLE, &mut mode) != 0 {
                let _ = SetConsoleMode(
                    self.input as HANDLE,
                    (mode & !ENABLE_VIRTUAL_TERMINAL_INPUT) | self.original_vt,
                );
            }
            CloseHandle(self.stop as HANDLE);
        }
    }
}

fn read_loop(
    input: HANDLE,
    stop: HANDLE,
    sender: &mpsc::Sender<io::Result<TerminalEvent>>,
) -> io::Result<()> {
    let handles = [stop, input];
    let mut decoder = Decoder::new();
    loop {
        let timeout = if decoder.has_lone_escape() {
            ESCAPE_WAIT_MS
        } else {
            INFINITE
        };
        // SAFETY: Both handles outlive this thread; handles points to two valid entries.
        let ready = unsafe { WaitForMultipleObjects(2, handles.as_ptr(), 0, timeout) };
        let mut events = Vec::new();
        let previous_graphics = decoder.graphics;
        match ready {
            WAIT_OBJECT_0 => return Ok(()),
            WAIT_TIMEOUT => decoder.flush_escape(&mut events),
            value if value == WAIT_OBJECT_0 + 1 => {
                // SAFETY: INPUT_RECORD is a C union of integers, all-zero is valid storage.
                let mut records: [INPUT_RECORD; 128] = unsafe { std::mem::zeroed() };
                let mut read = 0;
                // SAFETY: The buffer is writable for its stated size, input is valid,
                // and this thread is the only console-input consumer.
                if unsafe {
                    ReadConsoleInputW(input, records.as_mut_ptr(), records.len() as u32, &mut read)
                } == 0
                {
                    return Err(io::Error::last_os_error());
                }
                for record in &records[..read as usize] {
                    decoder.record(record, &mut events);
                }
            }
            _ => return Err(io::Error::last_os_error()),
        }
        if decoder.graphics != previous_graphics
            && sender
                .blocking_send(Ok(TerminalEvent::Graphics(decoder.graphics)))
                .is_err()
        {
            return Ok(());
        }
        for event in events {
            if sender
                .blocking_send(Ok(TerminalEvent::Input(event)))
                .is_err()
            {
                return Ok(());
            }
        }
    }
}

struct Decoder {
    parser: InputParser,
    sequence: String,
    sequence_modifiers: KeyModifiers,
    sequence_kind: KeyEventKind,
    paste: Option<String>,
    surrogate: Option<u16>,
    held_keys: HashSet<u16>,
    alt_numpad_duplicate: Option<u16>,
    mouse_buttons: MouseButtons,
    graphics: GraphicsCapabilities,
}

impl Decoder {
    fn new() -> Self {
        Self {
            parser: InputParser::new(),
            sequence: String::new(),
            sequence_modifiers: KeyModifiers::NONE,
            sequence_kind: KeyEventKind::Press,
            paste: None,
            surrogate: None,
            held_keys: HashSet::new(),
            alt_numpad_duplicate: None,
            mouse_buttons: MouseButtons::NONE,
            graphics: GraphicsCapabilities::default(),
        }
    }

    fn has_lone_escape(&self) -> bool {
        self.sequence == "\x1b" && self.paste.is_none()
    }

    fn flush_escape(&mut self, events: &mut Vec<Event>) {
        if self.has_lone_escape() {
            events.push(Event::Key(KeyEvent::new_with_kind(
                KeyCode::Esc,
                self.sequence_modifiers,
                self.sequence_kind,
            )));
            self.sequence.clear();
        }
    }

    fn record(&mut self, record: &INPUT_RECORD, events: &mut Vec<Event>) {
        if u32::from(record.EventType) != KEY_EVENT {
            self.alt_numpad_duplicate = None;
        }
        // SAFETY: EventType selects the initialized member of the Win32 union.
        unsafe {
            match u32::from(record.EventType) {
                KEY_EVENT => self.key(&record.Event.KeyEvent, events),
                MOUSE_EVENT => self.native_mouse(&record.Event.MouseEvent, events),
                WINDOW_BUFFER_SIZE_EVENT => {
                    let size = record.Event.WindowBufferSizeEvent.dwSize;
                    if size.X > 0 && size.Y > 0 {
                        events.push(Event::Resize(size.X as u16, size.Y as u16));
                    }
                }
                FOCUS_EVENT => {
                    let focused = record.Event.FocusEvent.bSetFocus != 0;
                    if !focused {
                        self.held_keys.clear();
                    }
                    events.push(if focused {
                        Event::FocusGained
                    } else {
                        Event::FocusLost
                    });
                }
                _ => {}
            }
        }
    }

    fn key(&mut self, record: &KEY_EVENT_RECORD, events: &mut Vec<Event>) {
        let vk = record.wVirtualKeyCode;
        let modifiers = native_modifiers(record.dwControlKeyState);
        // SAFETY: Unicode input is selected by ReadConsoleInputW.
        let unit = unsafe { record.uChar.UnicodeChar };
        let duplicate = self.alt_numpad_duplicate.take();
        if record.bKeyDown == 0 {
            self.held_keys.remove(&vk);
            // Conhost can send a VT character, a VK_PACKET release, and then
            // an adjacent duplicate VT character. Suppress only that exact
            // sequence, never characters based on their arrival speed.
            if vk == 0xe7 && unit != 0 && modifiers.contains(KeyModifiers::ALT) {
                self.alt_numpad_duplicate = Some(unit);
            } else if vk == 0x12 && unit != 0 {
                // Legacy Alt+numpad emits its resulting character on Alt-up.
                self.utf16(unit, KeyModifiers::NONE, KeyEventKind::Press, events);
            }
            return;
        }
        if vk == 0 && unit != 0 && duplicate == Some(unit) {
            return;
        }
        if (0x60..=0x69).contains(&vk) && modifiers == KeyModifiers::ALT {
            return;
        }
        let kind = if vk != 0 && !self.held_keys.insert(vk) {
            KeyEventKind::Repeat
        } else {
            KeyEventKind::Press
        };
        if unit != 0 {
            for repeat in 0..record.wRepeatCount.max(1) {
                let kind = if repeat == 0 {
                    kind
                } else {
                    KeyEventKind::Repeat
                };
                self.utf16(unit, modifiers, kind, events);
            }
        } else if let Some(code) = native_key(vk, modifiers) {
            self.flush_escape(events);
            events.push(Event::Key(KeyEvent::new_with_kind(code, modifiers, kind)));
        }
    }

    fn utf16(
        &mut self,
        unit: u16,
        modifiers: KeyModifiers,
        kind: KeyEventKind,
        events: &mut Vec<Event>,
    ) {
        if (0xD800..=0xDBFF).contains(&unit) {
            if self.surrogate.replace(unit).is_some() {
                self.character(char::REPLACEMENT_CHARACTER, modifiers, kind, events);
            }
            return;
        }
        if let Some(high) = self.surrogate.take() {
            if (0xDC00..=0xDFFF).contains(&unit) {
                let value = 0x10000 + ((u32::from(high) - 0xD800) << 10) + u32::from(unit) - 0xDC00;
                self.character(
                    char::from_u32(value).unwrap_or(char::REPLACEMENT_CHARACTER),
                    modifiers,
                    kind,
                    events,
                );
                return;
            }
            self.character(char::REPLACEMENT_CHARACTER, modifiers, kind, events);
        }
        self.character(
            char::from_u32(u32::from(unit)).unwrap_or(char::REPLACEMENT_CHARACTER),
            modifiers,
            kind,
            events,
        );
    }

    fn character(
        &mut self,
        ch: char,
        modifiers: KeyModifiers,
        kind: KeyEventKind,
        events: &mut Vec<Event>,
    ) {
        if let Some(paste) = self.paste.as_mut() {
            paste.push(ch);
            if paste.ends_with("\x1b[201~") {
                paste.truncate(paste.len() - "\x1b[201~".len());
                events.push(Event::Paste(self.paste.take().unwrap()));
            }
            return;
        }
        if self.sequence.is_empty() {
            if ch == '\x1b' {
                self.sequence.push(ch);
                self.sequence_modifiers = modifiers;
                self.sequence_kind = kind;
            } else {
                let mut utf8 = [0; 4];
                self.parse(
                    ch.encode_utf8(&mut utf8).as_bytes(),
                    modifiers,
                    kind,
                    events,
                );
            }
            return;
        }
        if ch == '\x1b' {
            self.flush_escape(events);
            self.sequence.clear();
            self.sequence.push(ch);
            self.sequence_modifiers = modifiers;
            self.sequence_kind = kind;
            return;
        }
        self.sequence.push(ch);
        // Only a lone ESC has an ambiguity timeout. A recognized CSI prefix and
        // paste marker may cross arbitrarily delayed console-read boundaries.
        if self.sequence == "\x1b[" || self.sequence == "\x1bO" {
            return;
        }
        if self.sequence.starts_with("\x1b[") && !('@'..='~').contains(&ch) {
            if self.sequence.len() > 128 {
                self.sequence.clear();
            }
            return;
        }
        let sequence = std::mem::take(&mut self.sequence);
        if self.terminal_report(&sequence) {
            return;
        }
        match sequence.as_str() {
            "\x1b[200~" => self.paste = Some(String::new()),
            "\x1b[201~" => {}
            "\x1b[I" => events.push(Event::FocusGained),
            "\x1b[O" => events.push(Event::FocusLost),
            _ => self.parse(
                sequence.as_bytes(),
                modifiers | self.sequence_modifiers,
                self.sequence_kind,
                events,
            ),
        }
    }

    /// Consume responses before termwiz can turn unknown CSI reports into
    /// ordinary characters. Bracketed paste is handled earlier and stays literal.
    fn terminal_report(&mut self, sequence: &str) -> bool {
        if let Some(mode) = sequence
            .strip_prefix("\x1b[?80;")
            .and_then(|text| text.strip_suffix("$y"))
        {
            self.graphics.sixel_display_mode = match mode {
                "1" => Some(true),
                "2" | "4" => Some(false),
                _ => None,
            };
            return true;
        }
        if let Some(parameters) = sequence
            .strip_prefix("\x1b[?")
            .and_then(|text| text.strip_suffix('c'))
        {
            let values = parameters
                .split(';')
                .map(str::parse::<u16>)
                .collect::<Result<Vec<_>, _>>();
            if let Ok(values) = values {
                // The first parameter identifies the terminal family, not a feature.
                self.graphics.sixel = values.iter().skip(1).any(|value| *value == 4);
            }
            return true;
        }
        if let Some(parameters) = sequence
            .strip_prefix("\x1b[6;")
            .and_then(|text| text.strip_suffix('t'))
        {
            self.graphics.cell_pixels = parameters.split_once(';').and_then(|(height, width)| {
                CellPixels::new(width.parse().ok()?, height.parse().ok()?)
            });
            return true;
        }
        false
    }

    fn parse(
        &mut self,
        bytes: &[u8],
        modifiers: KeyModifiers,
        kind: KeyEventKind,
        events: &mut Vec<Event>,
    ) {
        for event in self.parser.parse_as_vec(bytes, false) {
            match event {
                InputEvent::Key(key) => {
                    let modifiers = modifiers | vt_modifiers(&key.modifiers);
                    if let Some(code) = vt_key(key.key, modifiers) {
                        events.push(Event::Key(KeyEvent::new_with_kind(code, modifiers, kind)));
                    }
                }
                InputEvent::Mouse(mouse) => self.mouse(
                    mouse.x.saturating_sub(1),
                    mouse.y.saturating_sub(1),
                    mouse.mouse_buttons,
                    vt_modifiers(&mouse.modifiers),
                    events,
                ),
                InputEvent::Paste(text) => events.push(Event::Paste(text)),
                _ => {}
            }
        }
    }

    fn mouse(
        &mut self,
        column: u16,
        row: u16,
        buttons: MouseButtons,
        modifiers: KeyModifiers,
        events: &mut Vec<Event>,
    ) {
        let positive = buttons.contains(MouseButtons::WHEEL_POSITIVE);
        let kind = if buttons.contains(MouseButtons::VERT_WHEEL) {
            if positive {
                MouseEventKind::ScrollUp
            } else {
                MouseEventKind::ScrollDown
            }
        } else if buttons.contains(MouseButtons::HORZ_WHEEL) {
            if positive {
                MouseEventKind::ScrollRight
            } else {
                MouseEventKind::ScrollLeft
            }
        } else {
            let old = mouse_button(&self.mouse_buttons);
            let new = mouse_button(&buttons);
            self.mouse_buttons = buttons;
            match (old, new) {
                (Some(old), None) => MouseEventKind::Up(old),
                (Some(old), Some(new)) if old == new => MouseEventKind::Drag(new),
                (_, Some(new)) => MouseEventKind::Down(new),
                _ => MouseEventKind::Moved,
            }
        };
        events.push(Event::Mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers,
        }));
    }

    fn native_mouse(&mut self, mouse: &MOUSE_EVENT_RECORD, events: &mut Vec<Event>) {
        let mut buttons = MouseButtons::NONE;
        if mouse.dwButtonState & 1 != 0 {
            buttons |= MouseButtons::LEFT;
        }
        if mouse.dwButtonState & 2 != 0 {
            buttons |= MouseButtons::RIGHT;
        }
        if mouse.dwButtonState & 4 != 0 {
            buttons |= MouseButtons::MIDDLE;
        }
        if mouse.dwEventFlags & (4 | 8) != 0 {
            buttons |= if mouse.dwEventFlags & 4 != 0 {
                MouseButtons::VERT_WHEEL
            } else {
                MouseButtons::HORZ_WHEEL
            };
            if (mouse.dwButtonState >> 16) as i16 > 0 {
                buttons |= MouseButtons::WHEEL_POSITIVE;
            }
        }
        // Win32 mouse coordinates are relative to the screen buffer, not its viewport.
        let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: stdout is borrowed and info is valid writable storage.
        let offset = unsafe {
            if GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &mut info) != 0 {
                info.srWindow.Top
            } else {
                0
            }
        };
        self.mouse(
            mouse.dwMousePosition.X.max(0) as u16,
            mouse.dwMousePosition.Y.saturating_sub(offset).max(0) as u16,
            buttons,
            native_modifiers(mouse.dwControlKeyState),
            events,
        );
    }
}

fn native_modifiers(state: u32) -> KeyModifiers {
    let mut result = KeyModifiers::NONE;
    if state & 0x0003 != 0 {
        result |= KeyModifiers::ALT;
    }
    if state & 0x000c != 0 {
        result |= KeyModifiers::CONTROL;
    }
    if state & 0x0010 != 0 {
        result |= KeyModifiers::SHIFT;
    }
    result
}

fn vt_modifiers(modifiers: &Modifiers) -> KeyModifiers {
    let mut result = KeyModifiers::NONE;
    if modifiers.contains(Modifiers::ALT) {
        result |= KeyModifiers::ALT;
    }
    if modifiers.contains(Modifiers::CTRL) {
        result |= KeyModifiers::CONTROL;
    }
    if modifiers.contains(Modifiers::SHIFT) {
        result |= KeyModifiers::SHIFT;
    }
    if modifiers.contains(Modifiers::SUPER) {
        result |= KeyModifiers::SUPER;
    }
    result
}

fn mouse_button(buttons: &MouseButtons) -> Option<MouseButton> {
    if buttons.contains(MouseButtons::LEFT) {
        Some(MouseButton::Left)
    } else if buttons.contains(MouseButtons::RIGHT) {
        Some(MouseButton::Right)
    } else if buttons.contains(MouseButtons::MIDDLE) {
        Some(MouseButton::Middle)
    } else {
        None
    }
}

fn native_key(vk: u16, modifiers: KeyModifiers) -> Option<KeyCode> {
    Some(match vk {
        0x08 => KeyCode::Backspace,
        0x09 if modifiers.contains(KeyModifiers::SHIFT) => KeyCode::BackTab,
        0x09 => KeyCode::Tab,
        0x0d => KeyCode::Enter,
        0x1b => KeyCode::Esc,
        0x20 => KeyCode::Char(' '),
        0x21 => KeyCode::PageUp,
        0x22 => KeyCode::PageDown,
        0x23 => KeyCode::End,
        0x24 => KeyCode::Home,
        0x25 => KeyCode::Left,
        0x26 => KeyCode::Up,
        0x27 => KeyCode::Right,
        0x28 => KeyCode::Down,
        0x2d => KeyCode::Insert,
        0x2e => KeyCode::Delete,
        0x30..=0x39 => KeyCode::Char(char::from_u32(vk.into())?),
        0x41..=0x5a => KeyCode::Char(char::from_u32(vk.into())?.to_ascii_lowercase()),
        0x70..=0x87 => KeyCode::F((vk - 0x70 + 1) as u8),
        _ => return None,
    })
}

fn vt_key(key: VtKey, modifiers: KeyModifiers) -> Option<KeyCode> {
    Some(match key {
        VtKey::Char(ch) => KeyCode::Char(ch),
        VtKey::Escape => KeyCode::Esc,
        VtKey::Enter => KeyCode::Enter,
        VtKey::Backspace => KeyCode::Backspace,
        VtKey::Tab if modifiers.contains(KeyModifiers::SHIFT) => KeyCode::BackTab,
        VtKey::Tab => KeyCode::Tab,
        VtKey::LeftArrow | VtKey::ApplicationLeftArrow => KeyCode::Left,
        VtKey::RightArrow | VtKey::ApplicationRightArrow => KeyCode::Right,
        VtKey::UpArrow | VtKey::ApplicationUpArrow => KeyCode::Up,
        VtKey::DownArrow | VtKey::ApplicationDownArrow => KeyCode::Down,
        VtKey::Home | VtKey::KeyPadHome => KeyCode::Home,
        VtKey::End | VtKey::KeyPadEnd => KeyCode::End,
        VtKey::PageUp | VtKey::KeyPadPageUp => KeyCode::PageUp,
        VtKey::PageDown | VtKey::KeyPadPageDown => KeyCode::PageDown,
        VtKey::Insert => KeyCode::Insert,
        VtKey::Delete => KeyCode::Delete,
        VtKey::Function(n) => KeyCode::F(n),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(decoder: &mut Decoder, text: &str, events: &mut Vec<Event>) {
        for unit in text.encode_utf16() {
            decoder.utf16(unit, KeyModifiers::NONE, KeyEventKind::Press, events);
        }
    }

    #[test]
    fn fragmented_graphics_reports_never_become_composer_keys() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        feed(&mut decoder, "a\x1b[?61;4;6;7;14;21;22;", &mut events);
        decoder.flush_escape(&mut events);
        assert!(!decoder.graphics.sixel);
        feed(&mut decoder, "23;24;28;32;42;52c\x1b[6;20;", &mut events);
        decoder.flush_escape(&mut events);
        assert!(decoder.graphics.sixel);
        assert!(decoder.graphics.sixel_cell_pixels().is_none());
        feed(&mut decoder, "10t\x1b[?80;", &mut events);
        decoder.flush_escape(&mut events);
        assert!(decoder.graphics.sixel_cell_pixels().is_none());
        feed(&mut decoder, "2$yb", &mut events);
        assert_eq!(
            decoder.graphics.sixel_cell_pixels(),
            CellPixels::new(10, 20)
        );
        assert_eq!(
            events,
            [
                Event::Key(KeyCode::Char('a').into()),
                Event::Key(KeyCode::Char('b').into())
            ]
        );
    }

    #[test]
    fn graphics_reports_inside_a_paste_remain_literal() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        let replies = "\x1b[?61;4c\x1b[6;20;10t\x1b[?80;1$y";
        feed(&mut decoder, "\x1b[200~", &mut events);
        feed(&mut decoder, replies, &mut events);
        feed(&mut decoder, "\x1b[201~", &mut events);
        assert_eq!(events, [Event::Paste(replies.into())]);
        assert_eq!(decoder.graphics, GraphicsCapabilities::default());
    }

    #[test]
    fn invalid_graphics_geometry_and_modes_disable_native_rendering() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        feed(
            &mut decoder,
            "\x1b[?4c\x1b[6;20;10t\x1b[?80;2$y",
            &mut events,
        );
        assert!(
            !decoder.graphics.sixel,
            "DA1's model code is not a Sixel feature"
        );
        feed(&mut decoder, "\x1b[?61;4c", &mut events);
        assert!(decoder.graphics.sixel_cell_pixels().is_some());
        for report in [
            "\x1b[6;0;10t",
            "\x1b[6;20;513t",
            "\x1b[6;20;999999t",
            "\x1b[6;20;10;7t",
            "\x1b[6;;10t",
        ] {
            feed(&mut decoder, report, &mut events);
            assert!(decoder.graphics.cell_pixels.is_none(), "{report:?}");
        }
        feed(&mut decoder, "\x1b[6;24;12t", &mut events);
        assert_eq!(
            decoder.graphics.sixel_cell_pixels(),
            CellPixels::new(12, 24)
        );
        for report in ["\x1b[?80;0$y", "\x1b[?80;3$y", "\x1b[?80;9$y"] {
            feed(&mut decoder, report, &mut events);
            assert!(decoder.graphics.sixel_cell_pixels().is_none());
        }
        feed(&mut decoder, "\x1b[?80;1$y", &mut events);
        assert_eq!(decoder.graphics.sixel_display_mode, Some(true));
        assert!(decoder.graphics.sixel_cell_pixels().is_some());
        assert!(
            events.is_empty(),
            "terminal reports leaked into input: {events:?}"
        );
    }

    #[test]
    fn bracketed_multiline_paste_is_atomic_and_preserves_controls_and_unicode() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        let text = "  中文 🦀\r\n\nnext\t \x03\x1b[D\r\n";
        feed(&mut decoder, "\x1b[200~", &mut events);
        feed(&mut decoder, text, &mut events);
        decoder.flush_escape(&mut events);
        assert!(events.is_empty());
        feed(&mut decoder, "\x1b[201~", &mut events);
        assert_eq!(events, [Event::Paste(text.into())]);
        feed(&mut decoder, "\r", &mut events);
        assert_eq!(events.last(), Some(&Event::Key(KeyCode::Enter.into())));
    }

    #[test]
    fn marker_and_surrogate_boundaries_survive_delayed_reads() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        for part in ["\x1b[", "2", "00", "~", "first\n", "\x1b[20", "1", "~"] {
            feed(&mut decoder, part, &mut events);
            decoder.flush_escape(&mut events);
        }
        assert_eq!(events, [Event::Paste("first\n".into())]);
        events.clear();
        decoder.utf16(0xd83e, KeyModifiers::NONE, KeyEventKind::Press, &mut events);
        decoder.flush_escape(&mut events);
        assert!(events.is_empty());
        decoder.utf16(0xdd80, KeyModifiers::NONE, KeyEventKind::Press, &mut events);
        assert_eq!(events, [Event::Key(KeyCode::Char('🦀').into())]);
    }

    #[test]
    fn escape_timeout_and_two_escapes_remain_distinct_from_alt_keys() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        feed(&mut decoder, "\x1b", &mut events);
        assert!(events.is_empty());
        assert!(decoder.has_lone_escape());
        decoder.flush_escape(&mut events);
        feed(&mut decoder, "\x1b\x1b", &mut events);
        decoder.flush_escape(&mut events);
        assert_eq!(events, vec![Event::Key(KeyCode::Esc.into()); 3]);
        events.clear();
        feed(&mut decoder, "\x1b\r", &mut events);
        assert_eq!(
            events,
            [Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT))]
        );
    }

    #[test]
    fn shortcuts_navigation_mouse_and_focus_are_decoded() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        feed(&mut decoder, "\x03\x16\x1b[A\x1b[1;5D\x1b[Z", &mut events);
        assert_eq!(
            events,
            [
                Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
                Event::Key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)),
                Event::Key(KeyCode::Up.into()),
                Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL)),
                Event::Key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
            ]
        );
        events.clear();
        feed(
            &mut decoder,
            "\x1b[<0;10;5M\x1b[<0;10;5m\x1b[I\x1b[O",
            &mut events,
        );
        assert_eq!(
            events,
            [
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Down(MouseButton::Left),
                    column: 9,
                    row: 4,
                    modifiers: KeyModifiers::NONE
                }),
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Up(MouseButton::Left),
                    column: 9,
                    row: 4,
                    modifiers: KeyModifiers::NONE
                }),
                Event::FocusGained,
                Event::FocusLost,
            ]
        );
    }

    #[test]
    fn native_modifiers_and_held_key_repeats_are_preserved() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        // SAFETY: An all-zero KEY_EVENT_RECORD is valid, then populate its Unicode field.
        let mut key: KEY_EVENT_RECORD = unsafe { std::mem::zeroed() };
        key.bKeyDown = 1;
        key.wRepeatCount = 1;
        key.wVirtualKeyCode = 0x43;
        key.uChar.UnicodeChar = 3;
        key.dwControlKeyState = 0x0002 | 0x0008;
        decoder.key(&key, &mut events);
        decoder.key(&key, &mut events);
        let modifiers = KeyModifiers::CONTROL | KeyModifiers::ALT;
        assert_eq!(
            events,
            [
                Event::Key(KeyEvent::new_with_kind(
                    KeyCode::Char('c'),
                    modifiers,
                    KeyEventKind::Press
                )),
                Event::Key(KeyEvent::new_with_kind(
                    KeyCode::Char('c'),
                    modifiers,
                    KeyEventKind::Repeat
                )),
            ]
        );
        key.bKeyDown = 0;
        decoder.key(&key, &mut events);
        key.bKeyDown = 1;
        decoder.key(&key, &mut events);
        assert_eq!(events.last(), events.first());
    }

    #[test]
    fn native_shift_tab_and_alt_numpad_keep_their_meaning() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        // SAFETY: KEY_EVENT_RECORD is a C record consisting entirely of integers.
        let mut key: KEY_EVENT_RECORD = unsafe { std::mem::zeroed() };
        key.bKeyDown = 1;
        key.wRepeatCount = 1;
        key.wVirtualKeyCode = 0x09;
        key.uChar.UnicodeChar = 9;
        key.dwControlKeyState = 0x10;
        decoder.key(&key, &mut events);
        assert_eq!(
            events,
            [Event::Key(KeyEvent::new(
                KeyCode::BackTab,
                KeyModifiers::SHIFT
            ))]
        );
        events.clear();
        key.wVirtualKeyCode = 0x66;
        key.uChar.UnicodeChar = b'6'.into();
        key.dwControlKeyState = 0x02;
        decoder.key(&key, &mut events);
        assert!(events.is_empty());
        key.bKeyDown = 0;
        key.wVirtualKeyCode = 0x12;
        key.uChar.UnicodeChar = 0xe9;
        decoder.key(&key, &mut events);
        assert_eq!(events, [Event::Key(KeyCode::Char('é').into())]);
    }

    #[test]
    fn alt_numpad_packet_suppresses_only_its_adjacent_duplicate() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        // SAFETY: KEY_EVENT_RECORD is a C record consisting entirely of integers.
        let mut key: KEY_EVENT_RECORD = unsafe { std::mem::zeroed() };
        key.bKeyDown = 1;
        key.wRepeatCount = 1;
        key.uChar.UnicodeChar = 0xe9;
        decoder.key(&key, &mut events);
        key.bKeyDown = 0;
        key.wVirtualKeyCode = 0xe7;
        key.dwControlKeyState = 0x02;
        decoder.key(&key, &mut events);
        key.bKeyDown = 1;
        key.wVirtualKeyCode = 0;
        key.dwControlKeyState = 0;
        decoder.key(&key, &mut events);
        assert_eq!(events, [Event::Key(KeyCode::Char('é').into())]);
        decoder.key(&key, &mut events);
        assert_eq!(events, vec![Event::Key(KeyCode::Char('é').into()); 2]);
    }

    #[test]
    fn native_resize_and_focus_do_not_split_a_paste_or_surrogate_pair() {
        let mut decoder = Decoder::new();
        let mut events = Vec::new();
        feed(&mut decoder, "\x1b[200~", &mut events);
        decoder.utf16(0xd83e, KeyModifiers::NONE, KeyEventKind::Press, &mut events);
        // SAFETY: INPUT_RECORD is a C union of integer records, initialized by EventType.
        let mut record: INPUT_RECORD = unsafe { std::mem::zeroed() };
        record.EventType = WINDOW_BUFFER_SIZE_EVENT as u16;
        record.Event.WindowBufferSizeEvent.dwSize.X = 120;
        record.Event.WindowBufferSizeEvent.dwSize.Y = 36;
        decoder.record(&record, &mut events);
        record.EventType = FOCUS_EVENT as u16;
        record.Event.FocusEvent.bSetFocus = 0;
        decoder.record(&record, &mut events);
        record.Event.FocusEvent.bSetFocus = 1;
        decoder.record(&record, &mut events);
        decoder.utf16(0xdd80, KeyModifiers::NONE, KeyEventKind::Press, &mut events);
        feed(&mut decoder, "\x1b[201~", &mut events);
        assert_eq!(
            events,
            [
                Event::Resize(120, 36),
                Event::FocusLost,
                Event::FocusGained,
                Event::Paste("🦀".into()),
            ]
        );
    }
}

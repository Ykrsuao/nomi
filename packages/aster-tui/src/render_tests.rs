// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;

use ratatui::{
    backend::{Backend, ClearType, TestBackend, WindowSize},
    buffer::Cell,
    layout::{Position, Rect, Size},
    Terminal,
};

use super::{render, HitAreas};
use crate::state::{Overlay, Phase, State};

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Hide,
    Draw { visible: bool },
    Clear { visible: bool },
    Move(Position),
    Show(Position),
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    Draw,
    Move,
    Flush,
}

/// TestBackend does not move its cursor while drawing cells. This adapter models
/// the physical terminal's write position so showing before moving is observable.
struct AuditBackend {
    inner: TestBackend,
    events: Vec<Event>,
    visible: bool,
    fail: Option<Failure>,
}

impl AuditBackend {
    fn new() -> Self {
        Self {
            inner: TestBackend::new(100, 30),
            events: Vec::new(),
            visible: true,
            fail: None,
        }
    }

    fn check(&self, point: Failure) -> io::Result<()> {
        if self.fail == Some(point) {
            Err(io::Error::other("injected render failure"))
        } else {
            Ok(())
        }
    }
}

impl Backend for AuditBackend {
    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        self.events.push(Event::Draw {
            visible: self.visible,
        });
        self.check(Failure::Draw)?;
        let cells: Vec<_> = content.collect();
        self.inner.draw(cells.iter().copied())?;
        if let Some((x, y, _)) = cells.last() {
            self.inner.set_cursor_position((*x + 1, *y))?;
        }
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        self.events.push(Event::Hide);
        self.visible = false;
        self.inner.hide_cursor()
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        self.events
            .push(Event::Show(self.inner.get_cursor_position()?));
        self.visible = true;
        self.inner.show_cursor()
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        self.inner.get_cursor_position()
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let position = position.into();
        self.events.push(Event::Move(position));
        self.check(Failure::Move)?;
        self.inner.set_cursor_position(position)
    }

    fn clear(&mut self) -> io::Result<()> {
        self.clear_region(ClearType::All)
    }

    fn clear_region(&mut self, region: ClearType) -> io::Result<()> {
        self.events.push(Event::Clear {
            visible: self.visible,
        });
        self.inner.clear_region(region)
    }

    fn size(&self) -> io::Result<Size> {
        self.inner.size()
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        self.inner.window_size()
    }

    fn flush(&mut self) -> io::Result<()> {
        self.events.push(Event::Flush);
        self.check(Failure::Flush)?;
        self.inner.flush()
    }
}

fn state() -> State {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.editor.insert_str("draft");
    state
}

fn frame(terminal: &mut Terminal<AuditBackend>, state: &mut State) -> HitAreas {
    terminal.backend_mut().events.clear();
    render(terminal, state).unwrap()
}

fn assert_cursor(terminal: &mut Terminal<AuditBackend>, expected: Option<Position>) {
    let backend = terminal.backend_mut();
    assert_eq!(backend.events.first(), Some(&Event::Hide));
    assert!(backend.events.contains(&Event::Draw { visible: false }));
    assert!(!backend.events.contains(&Event::Draw { visible: true }));
    assert!(!backend.events.contains(&Event::Clear { visible: true }));
    let shows: Vec<_> = backend
        .events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            Event::Show(position) => Some((index, *position)),
            _ => None,
        })
        .collect();
    if let Some(expected) = expected {
        assert_eq!(shows.len(), 1, "{:?}", backend.events);
        let (index, position) = shows[0];
        assert_eq!(position, expected);
        assert_eq!(backend.events.get(index - 1), Some(&Event::Move(expected)));
        assert_eq!(backend.get_cursor_position().unwrap(), expected);
        assert!(backend.visible);
    } else {
        assert!(shows.is_empty(), "{:?}", backend.events);
        assert!(!backend.visible);
    }
}

#[test]
fn timer_redraw_keeps_cursor_hidden_until_back_in_composer() {
    let mut terminal = Terminal::new(AuditBackend::new()).unwrap();
    let mut state = state();
    for seconds in [None, Some(1.25), Some(1.45)] {
        state.phase = if seconds.is_some() {
            Phase::Busy
        } else {
            Phase::Ready
        };
        state.last_elapsed = seconds;
        let hits = frame(&mut terminal, &mut state);
        // The five-character draft follows the prompt gutter.
        let expected = Position::new(hits.composer.x + 9, hits.composer.y + 1);
        assert_cursor(&mut terminal, Some(expected));
    }
    assert_eq!(state.editor.lines(), &["draft"]);
}

#[test]
fn overlays_move_to_search_or_hide_and_restore_the_draft_cursor() {
    let mut terminal = Terminal::new(AuditBackend::new()).unwrap();
    let mut state = state();
    frame(&mut terminal, &mut state);
    state.open(Overlay::Search);
    state.query = "中文".into();
    let hits = frame(&mut terminal, &mut state);
    let query = hits.query.unwrap();
    assert_cursor(&mut terminal, Some(Position::new(query.x + 6, query.y)));
    for overlay in [Overlay::Help, Overlay::Diagnostics, Overlay::Changes] {
        state.open(overlay);
        state.conversation_changes = true;
        frame(&mut terminal, &mut state);
        assert_cursor(&mut terminal, None);
    }
    state.overlay = None;
    let hits = frame(&mut terminal, &mut state);
    assert_cursor(
        &mut terminal,
        Some(Position::new(hits.composer.x + 9, hits.composer.y + 1)),
    );
    assert_eq!(state.editor.lines(), &["draft"]);
}

#[test]
fn resizing_recomputes_cursor_and_hides_it_when_input_does_not_fit() {
    let mut terminal = Terminal::new(AuditBackend::new()).unwrap();
    let mut state = state();
    frame(&mut terminal, &mut state);
    for (width, height) in [(48, 12), (120, 36), (3, 2), (80, 24)] {
        terminal.backend_mut().inner.resize(width, height);
        let hits = frame(&mut terminal, &mut state);
        if width < 4 {
            assert_cursor(&mut terminal, None);
        } else {
            let expected = Position::new(hits.composer.x + 9, hits.composer.y + 1);
            assert!(Rect::new(0, 0, width, height).contains(expected));
            assert_cursor(&mut terminal, Some(expected));
        }
    }
}

#[test]
fn failed_draw_flush_or_position_never_shows_a_stray_cursor() {
    for failure in [Failure::Draw, Failure::Flush, Failure::Move] {
        let mut terminal = Terminal::new(AuditBackend::new()).unwrap();
        let mut state = state();
        frame(&mut terminal, &mut state);
        terminal.backend_mut().events.clear();
        terminal.backend_mut().fail = Some(failure);
        state.last_elapsed = Some(2.5);
        let result = render(&mut terminal, &mut state);
        assert!(result.is_err(), "{failure:?}");
        let backend = terminal.backend();
        assert_eq!(backend.events.first(), Some(&Event::Hide));
        assert!(!backend.visible, "{failure:?}: {:?}", backend.events);
        assert!(!backend
            .events
            .iter()
            .any(|event| matches!(event, Event::Show(_))));
    }
}

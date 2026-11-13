// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Visual wrapping around the original text area's editing and undo state.

use std::borrow::Cow;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::{
    layout::{Position, Rect},
    text::Span,
    widgets::Paragraph,
    Frame,
};
use tui_textarea::{CursorMove, TextArea};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Default)]
pub(crate) struct ComposerState {
    width: usize,
    top: usize,
    preferred_column: Option<usize>,
    last_cursor: Option<(usize, usize)>,
    pointer_anchor: Option<(usize, usize)>,
}

impl ComposerState {
    pub(crate) fn reset_navigation(&mut self) {
        self.preferred_column = None;
        self.last_cursor = None;
        self.cancel_pointer();
    }

    pub(crate) fn cancel_pointer(&mut self) {
        self.pointer_anchor = None;
    }
}

struct Glyph<'a> {
    text: Cow<'a, str>,
    start: usize,
    end: usize,
    width: usize,
}

struct VisualRow<'a> {
    line: usize,
    start: usize,
    end: usize,
    width: usize,
    glyphs: Vec<Glyph<'a>>,
}

impl VisualRow<'_> {
    fn column_at(&self, cell: usize, final_row: bool) -> usize {
        let mut x = 0;
        for glyph in &self.glyphs {
            if x + glyph.width > cell {
                return glyph.start;
            }
            x += glyph.width;
        }
        if final_row {
            self.end
        } else {
            self.glyphs.last().map_or(self.start, |glyph| glyph.start)
        }
    }
}

struct Layout<'a> {
    rows: Vec<VisualRow<'a>>,
    cursor: (usize, usize),
}

fn layout<'a>(editor: &'a TextArea<'_>, columns: usize) -> Layout<'a> {
    let columns = columns.max(1);
    let cursor = editor.cursor();
    let mut rows = Vec::new();
    for (line, text) in editor.lines().iter().enumerate() {
        let mut row = VisualRow {
            line,
            start: 0,
            end: 0,
            width: 0,
            glyphs: Vec::new(),
        };
        let mut column = 0;
        for grapheme in text.graphemes(true) {
            let count = grapheme.chars().count();
            let mut cells = if grapheme == "\t" {
                let stop = usize::from(editor.tab_length()).max(1);
                stop - row.width % stop
            } else {
                grapheme.width()
            };
            if row.width > 0 && row.width + cells > columns {
                rows.push(row);
                row = VisualRow {
                    line,
                    start: column,
                    end: column,
                    width: 0,
                    glyphs: Vec::new(),
                };
                if grapheme == "\t" {
                    cells = usize::from(editor.tab_length()).max(1);
                }
            }
            let display = if grapheme == "\t" {
                cells = cells.min(columns);
                Cow::Owned(" ".repeat(cells))
            } else if cells > columns {
                cells = 1;
                Cow::Borrowed("�")
            } else {
                Cow::Borrowed(grapheme)
            };
            row.glyphs.push(Glyph {
                text: display,
                start: column,
                end: column + count,
                width: cells,
            });
            row.width += cells;
            column += count;
            row.end = column;
        }
        let cursor_after_full_line = cursor == (line, column) && row.width >= columns;
        rows.push(row);
        if cursor_after_full_line {
            rows.push(VisualRow {
                line,
                start: column,
                end: column,
                width: 0,
                glyphs: Vec::new(),
            });
        }
    }
    let visual_row = rows
        .iter()
        .rposition(|row| row.line == cursor.0 && row.start <= cursor.1)
        .unwrap_or(0);
    let x = rows.get(visual_row).map_or(0, |row| {
        row.glyphs
            .iter()
            .take_while(|glyph| glyph.end <= cursor.1)
            .map(|glyph| glyph.width)
            .sum()
    });
    Layout {
        rows,
        cursor: (visual_row, x),
    }
}

pub(crate) fn row_count(editor: &TextArea<'_>, columns: usize) -> usize {
    layout(editor, columns).rows.len().max(1)
}

/// History takes over only when an unmodified arrow would leave the visual draft.
pub(crate) fn at_vertical_edge(
    editor: &TextArea<'_>,
    state: &ComposerState,
    previous: bool,
) -> bool {
    if editor.is_selecting() {
        return false;
    }
    if state.width == 0 {
        return if previous {
            editor.cursor().0 == 0
        } else {
            editor.cursor().0 + 1 == editor.lines().len()
        };
    }
    let layout = layout(editor, state.width);
    if previous {
        layout.cursor.0 == 0
    } else {
        layout.cursor.0 + 1 == layout.rows.len()
    }
}

fn move_to(editor: &mut TextArea<'_>, target: (usize, usize)) {
    if let (Ok(row), Ok(column)) = (u16::try_from(target.0), u16::try_from(target.1)) {
        editor.move_cursor(CursorMove::Jump(row, column));
        return;
    }
    // TextArea's absolute move uses u16, but pasted lines and cursor positions use usize.
    while editor.cursor().0 < target.0 {
        editor.move_cursor(CursorMove::Down);
    }
    while editor.cursor().0 > target.0 {
        editor.move_cursor(CursorMove::Up);
    }
    while editor.cursor().1 < target.1 {
        editor.move_cursor(CursorMove::Forward);
    }
    while editor.cursor().1 > target.1 {
        editor.move_cursor(CursorMove::Back);
    }
}

/// Consume visual navigation, leaving editing, history and application shortcuts intact.
pub(crate) fn handle_key(
    editor: &mut TextArea<'_>,
    state: &mut ComposerState,
    key: KeyEvent,
) -> bool {
    state.cancel_pointer();
    if state.width == 0
        || key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        || !matches!(
            key.code,
            KeyCode::Up | KeyCode::Down | KeyCode::Home | KeyCode::End
        )
    {
        state.preferred_column = None;
        return false;
    }
    let cursor = editor.cursor();
    if state.last_cursor != Some(cursor) {
        state.preferred_column = None;
    }
    let layout = layout(editor, state.width);
    let (current, x) = layout.cursor;
    let target_row = match key.code {
        KeyCode::Up => current.saturating_sub(1),
        KeyCode::Down => (current + 1).min(layout.rows.len().saturating_sub(1)),
        _ => current,
    };
    let row = &layout.rows[target_row];
    let final_row = layout
        .rows
        .get(target_row + 1)
        .is_none_or(|next| next.line != row.line);
    let column = match key.code {
        KeyCode::Home => {
            state.preferred_column = None;
            row.start
        }
        KeyCode::End => {
            state.preferred_column = None;
            row.end
        }
        _ => row.column_at(*state.preferred_column.get_or_insert(x), final_row),
    };
    let target = (row.line, column);
    if key.modifiers.contains(KeyModifiers::SHIFT) {
        if !editor.is_selecting() {
            editor.start_selection();
        }
    } else {
        editor.cancel_selection();
    }
    move_to(editor, target);
    state.last_cursor = Some(editor.cursor());
    true
}

/// Map pointer gestures through the same visual rows used to draw the editor.
pub(crate) fn handle_mouse(
    editor: &mut TextArea<'_>,
    state: &mut ComposerState,
    area: Rect,
    mouse: MouseEvent,
) -> bool {
    if area.is_empty() {
        state.cancel_pointer();
        return false;
    }
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            state.cancel_pointer();
            if !area.contains(Position::new(mouse.column, mouse.row)) {
                return false;
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if state.pointer_anchor.is_some() => {}
        MouseEventKind::Up(MouseButton::Left) => {
            // The last drag already placed the caret. Remapping after a redraw can
            // move it again when a full-line cursor changed the visual row count.
            return state.pointer_anchor.take().is_some();
        }
        MouseEventKind::Down(_) => {
            state.cancel_pointer();
            return false;
        }
        _ => return false,
    }

    state.width = usize::from(area.width);
    let layout = layout(editor, state.width);
    let height = usize::from(area.height);
    let max_top = layout.rows.len().saturating_sub(height);
    state.top = state.top.min(max_top);
    if mouse.row < area.y {
        state.top = state.top.saturating_sub(usize::from(area.y - mouse.row));
    } else if mouse.row >= area.bottom() {
        state.top = state
            .top
            .saturating_add(usize::from(mouse.row - area.bottom()) + 1)
            .min(max_top);
    }
    let row_index =
        state.top + usize::from(mouse.row.saturating_sub(area.y)).min(height.saturating_sub(1));
    let column = usize::from(mouse.column.saturating_sub(area.x)).min(state.width);
    let target = if let Some(row) = layout.rows.get(row_index) {
        // Blank cells after a wrapped segment belong to its end; keyboard vertical
        // navigation has a different policy to preserve its preferred column.
        (row.line, row.column_at(column, true))
    } else {
        let row = layout.rows.last().expect("text area always has a line");
        (row.line, row.end)
    };

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            let anchor = if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                let cursor = editor.cursor();
                let anchor = editor.selection_range().map_or(cursor, |(start, end)| {
                    if cursor == start {
                        end
                    } else {
                        start
                    }
                });
                if !editor.is_selecting() {
                    editor.start_selection();
                }
                anchor
            } else {
                editor.cancel_selection();
                target
            };
            move_to(editor, target);
            state.pointer_anchor = Some(anchor);
            if editor.cursor() == anchor {
                editor.cancel_selection();
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            let anchor = state.pointer_anchor.expect("drag started in the editor");
            if !editor.is_selecting() {
                move_to(editor, anchor);
                editor.start_selection();
            }
            move_to(editor, target);
            if editor.cursor() == anchor {
                editor.cancel_selection();
            }
        }
        _ => unreachable!(),
    }
    state.preferred_column = None;
    state.last_cursor = Some(editor.cursor());
    true
}

pub(crate) fn draw(
    frame: &mut Frame,
    area: Rect,
    editor: &mut TextArea<'_>,
    state: &mut ComposerState,
    focused: bool,
) -> Option<Position> {
    if !focused || area.is_empty() {
        state.cancel_pointer();
    }
    if state.width != usize::from(area.width) {
        state.width = usize::from(area.width);
        state.preferred_column = None;
    }
    if area.is_empty() {
        return None;
    }
    let style = editor.style();
    let cursor_style = editor.cursor_style();
    let selection_style = editor.selection_style();
    let selection = editor.selection_range();
    let cursor = editor.cursor();
    let layout = layout(editor, state.width);
    let height = usize::from(area.height);
    state.top = state.top.min(layout.rows.len().saturating_sub(height));
    if layout.cursor.0 < state.top {
        state.top = layout.cursor.0;
    } else if layout.cursor.0 >= state.top + height {
        state.top = layout.cursor.0 + 1 - height;
    }
    if editor.lines() == [""] {
        frame.render_widget(
            Paragraph::new(editor.placeholder_text())
                .style(editor.placeholder_style().unwrap_or(style)),
            Rect::new(
                area.x + 1,
                area.y,
                area.width.saturating_sub(1),
                area.height,
            ),
        );
    }
    for (offset, row) in layout.rows.iter().skip(state.top).take(height).enumerate() {
        let mut x = 0;
        for glyph in &row.glyphs {
            let selected = selection.is_some_and(|(start, end)| {
                (row.line, glyph.end) > start && (row.line, glyph.start) < end
            });
            let is_cursor =
                focused && cursor.0 == row.line && cursor.1 >= glyph.start && cursor.1 < glyph.end;
            let glyph_style = if is_cursor {
                style.patch(cursor_style)
            } else if selected {
                style.patch(selection_style)
            } else {
                style
            };
            frame.render_widget(
                Span::styled(glyph.text.as_ref(), glyph_style),
                Rect::new(
                    area.x + x as u16,
                    area.y + offset as u16,
                    glyph.width as u16,
                    1,
                ),
            );
            x += glyph.width;
        }
        let end_cursor = focused && layout.cursor.0 == state.top + offset && layout.cursor.1 == x;
        let selected_newline = selection
            .is_some_and(|(start, end)| (row.line, row.end) >= start && (row.line, row.end) < end)
            && layout
                .rows
                .get(state.top + offset + 1)
                .is_none_or(|next| next.line != row.line);
        if x < state.width && (end_cursor || selected_newline) {
            frame.render_widget(
                Span::styled(
                    " ",
                    style.patch(if end_cursor {
                        cursor_style
                    } else {
                        selection_style
                    }),
                ),
                Rect::new(area.x + x as u16, area.y + offset as u16, 1, 1),
            );
        }
    }
    focused.then(|| {
        Position::new(
            area.x + layout.cursor.1.min(state.width.saturating_sub(1)) as u16,
            area.y + layout.cursor.0.saturating_sub(state.top) as u16,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{
        backend::TestBackend,
        style::{Color, Style},
        Terminal,
    };

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    fn render_editor(
        editor: &mut TextArea<'_>,
        state: &mut ComposerState,
        area: Rect,
        focused: bool,
    ) {
        let mut terminal =
            Terminal::new(TestBackend::new(area.right().max(1), area.bottom().max(1))).unwrap();
        terminal
            .draw(|frame| {
                let _ = draw(frame, area, editor, state, focused);
            })
            .unwrap();
    }

    #[test]
    fn wrapping_preserves_cjk_graphemes_tabs_text_and_undo() {
        let mut editor = TextArea::default();
        let text = "ab中文e\u{301}👩‍💻\tend\nsecond";
        editor.insert_str(text);
        let cursor = editor.cursor();
        let layout = layout(&editor, 5);
        assert!(layout.rows.len() > 2);
        assert!(layout.rows.iter().all(|row| row.width <= 5));
        assert!(layout
            .rows
            .iter()
            .flat_map(|row| &row.glyphs)
            .any(|g| g.text == "👩‍💻"));
        assert!(layout
            .rows
            .iter()
            .flat_map(|row| &row.glyphs)
            .any(|g| g.text == "e\u{301}"));
        assert_eq!(editor.lines().join("\n"), text);
        assert_eq!(editor.cursor(), cursor);
        assert!(editor.undo());
        assert_eq!(editor.lines(), &[""]);
    }

    #[test]
    fn exact_width_cursor_gets_a_visible_row_and_narrow_layout_is_safe() {
        let mut editor = TextArea::from(["abcd"]);
        editor.move_cursor(CursorMove::End);
        assert_eq!(row_count(&editor, 4), 2);
        assert_eq!(layout(&editor, 4).cursor, (1, 0));
        for columns in [0, 1] {
            let editor = TextArea::from(["中文👩‍💻"]);
            assert!(layout(&editor, columns)
                .rows
                .iter()
                .all(|row| row.width <= 1));
        }
    }

    #[test]
    fn visual_navigation_keeps_column_and_shift_selection() {
        let mut editor = TextArea::from(["abcd中文efgh", "x", "12345678"]);
        let mut state = ComposerState {
            width: 4,
            ..ComposerState::default()
        };
        editor.move_cursor(CursorMove::Jump(0, 2));
        assert!(handle_key(&mut editor, &mut state, key(KeyCode::Down)));
        assert_eq!(editor.cursor(), (0, 5));
        assert!(handle_key(&mut editor, &mut state, key(KeyCode::Down)));
        assert_eq!(editor.cursor(), (0, 8));
        assert!(handle_key(&mut editor, &mut state, key(KeyCode::Down)));
        assert_eq!(editor.cursor(), (1, 1));
        assert!(handle_key(&mut editor, &mut state, key(KeyCode::Down)));
        assert_eq!(editor.cursor(), (2, 2));
        assert!(handle_key(
            &mut editor,
            &mut state,
            KeyEvent::new(KeyCode::Home, KeyModifiers::SHIFT)
        ));
        assert_eq!(editor.selection_range(), Some(((2, 0), (2, 2))));
        handle_key(&mut editor, &mut state, key(KeyCode::End));
        assert!(!editor.is_selecting());
        assert_eq!(editor.cursor(), (2, 4));
        editor.insert_str("!");
        assert_eq!(editor.lines()[2], "1234!5678");
    }

    #[test]
    fn drawing_keeps_cursor_visible_and_selection_styled_without_mutating_editor() {
        let mut editor = TextArea::default();
        editor.insert_str("abcdefghijklmnopqrstuvwxyz中文");
        editor.set_style(Style::default().fg(Color::White));
        editor.set_cursor_style(Style::default().bg(Color::Yellow));
        editor.set_selection_style(Style::default().bg(Color::Blue));
        editor.start_selection();
        editor.move_cursor(CursorMove::Back);
        editor.move_cursor(CursorMove::Back);
        let cursor = editor.cursor();
        let selection = editor.selection_range();
        let mut state = ComposerState::default();
        let mut terminal = Terminal::new(TestBackend::new(8, 3)).unwrap();
        let mut drawn_cursor = None;
        terminal
            .draw(|frame| drawn_cursor = draw(frame, frame.area(), &mut editor, &mut state, true))
            .unwrap();
        assert!(drawn_cursor.is_some_and(|cursor| cursor.x < 8 && cursor.y < 3));
        assert!(state.top > 0);
        assert!(terminal
            .backend()
            .buffer()
            .content
            .iter()
            .any(|cell| cell.bg == Color::Blue));
        assert_eq!(editor.cursor(), cursor);
        assert_eq!(editor.selection_range(), selection);
        assert_eq!(editor.lines(), &["abcdefghijklmnopqrstuvwxyz中文"]);
        assert!(editor.undo());
        assert_eq!(editor.lines(), &[""]);
    }

    #[test]
    fn empty_editor_keeps_placeholder_and_cursor_in_separate_cells() {
        let mut editor = TextArea::default();
        editor.set_placeholder_text("Type a message...");
        let mut state = ComposerState::default();
        let mut terminal = Terminal::new(TestBackend::new(24, 1)).unwrap();
        let mut drawn_cursor = None;
        terminal
            .draw(|frame| drawn_cursor = draw(frame, frame.area(), &mut editor, &mut state, true))
            .unwrap();
        assert_eq!(drawn_cursor, Some(Position::ORIGIN));
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.starts_with(" Type a message..."));
        assert_eq!(editor.lines(), &[""]);
    }

    #[test]
    fn pointer_places_caret_on_wrapped_graphemes_and_tabs_without_editing() {
        let mut editor = TextArea::default();
        let text = "ab中文e\u{301}👩‍💻\tend";
        editor.insert_str(text);
        let mut state = ComposerState::default();
        let area = Rect::new(4, 3, 6, 4);
        render_editor(&mut editor, &mut state, area, true);
        for (x, y, column) in [
            (7, 3, 2),
            (9, 3, 3),
            (4, 4, 4),
            (6, 4, 6),
            (7, 4, 9),
            (8, 4, 10),
            (4, 6, 13),
        ] {
            assert!(handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Down(MouseButton::Left), x, y),
            ));
            assert_eq!(editor.cursor(), (0, column));
            assert!(!editor.is_selecting());
            assert_eq!(editor.lines(), &[text]);
            render_editor(&mut editor, &mut state, area, true);
            assert!(handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Up(MouseButton::Left), x, y),
            ));
            assert_eq!(editor.cursor(), (0, column));
        }
        assert!(editor.undo());
        assert_eq!(editor.lines(), &[""]);
    }

    #[test]
    fn pointer_drag_and_shift_click_preserve_selection_anchor() {
        let mut editor = TextArea::from(["abcd", "efgh"]);
        let mut state = ComposerState::default();
        let area = Rect::new(2, 2, 5, 2);
        render_editor(&mut editor, &mut state, area, true);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 3, 2),
        );
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 5, 3),
        );
        assert_eq!(editor.selection_range(), Some(((0, 1), (1, 3))));
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Up(MouseButton::Left), 5, 3),
        );
        assert!(!handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 2, 2),
        ));
        assert_eq!(editor.selection_range(), Some(((0, 1), (1, 3))));
        for (x, y, selection) in [(2, 2, ((0, 0), (0, 1))), (4, 3, ((0, 1), (1, 2)))] {
            let mut click = mouse(MouseEventKind::Down(MouseButton::Left), x, y);
            click.modifiers = KeyModifiers::SHIFT;
            handle_mouse(&mut editor, &mut state, area, click);
            assert_eq!(editor.selection_range(), Some(selection));
            handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Up(MouseButton::Left), x, y),
            );
        }
        editor.insert_str("X");
        assert_eq!(editor.lines(), &["aXgh"]);
    }

    #[test]
    fn pointer_drag_selects_whole_emoji_across_visual_rows() {
        let mut editor = TextArea::from(["a中文👩‍💻b"]);
        let mut state = ComposerState::default();
        let area = Rect::new(2, 2, 6, 2);
        render_editor(&mut editor, &mut state, area, true);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 2, 2),
        );
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 3, 3),
        );
        assert_eq!(editor.selection_range(), Some(((0, 0), (0, 3))));
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 4, 3),
        );
        assert_eq!(editor.selection_range(), Some(((0, 0), (0, 6))));
        editor.insert_str("X");
        assert_eq!(editor.lines(), &["Xb"]);
    }

    #[test]
    fn pointer_drag_scrolls_beyond_viewport_and_clamps_to_line_ends() {
        let mut editor = TextArea::from(["zero", "one", "two", "three", "four"]);
        let mut state = ComposerState::default();
        let area = Rect::new(2, 2, 6, 2);
        render_editor(&mut editor, &mut state, area, true);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 3, 3),
        );
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 10, 4),
        );
        assert_eq!(state.top, 1);
        assert_eq!(editor.selection_range(), Some(((1, 1), (2, 3))));
        render_editor(&mut editor, &mut state, area, true);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 10, 5),
        );
        assert_eq!(state.top, 3);
        assert_eq!(editor.selection_range(), Some(((1, 1), (4, 4))));
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Up(MouseButton::Left), 10, 5),
        );
        assert_eq!(state.top, 3);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 3, 2),
        );
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 0, 1),
        );
        assert_eq!(state.top, 2);
        assert_eq!(editor.selection_range(), Some(((2, 0), (3, 1))));
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Drag(MouseButton::Left), 0, 0),
        );
        assert_eq!(state.top, 0);
        assert_eq!(editor.selection_range(), Some(((0, 0), (3, 1))));
    }

    #[test]
    fn pointer_release_does_not_remap_after_full_width_cursor_row_disappears() {
        let mut editor = TextArea::from(["abcdefghijkl"]);
        editor.move_cursor(CursorMove::End);
        let mut state = ComposerState::default();
        let area = Rect::new(0, 0, 4, 2);
        render_editor(&mut editor, &mut state, area, true);
        assert_eq!(state.top, 2);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 1, 0),
        );
        assert_eq!(editor.cursor(), (0, 9));
        render_editor(&mut editor, &mut state, area, true);
        assert_eq!(state.top, 1);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Up(MouseButton::Left), 1, 0),
        );
        assert_eq!(editor.cursor(), (0, 9));
        assert!(!editor.is_selecting());
    }

    #[test]
    fn pointer_ignores_foreign_drags_and_cancels_when_focus_is_lost() {
        let mut editor = TextArea::from(["abcd"]);
        let mut state = ComposerState::default();
        let area = Rect::new(2, 2, 5, 2);
        for event in [
            mouse(MouseEventKind::Down(MouseButton::Left), 1, 2),
            mouse(MouseEventKind::Drag(MouseButton::Left), 4, 2),
            mouse(MouseEventKind::Up(MouseButton::Left), 4, 2),
            mouse(MouseEventKind::Down(MouseButton::Right), 4, 2),
        ] {
            assert!(!handle_mouse(&mut editor, &mut state, area, event));
            assert_eq!(editor.cursor(), (0, 0));
        }
        for cancel in 0..4 {
            handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Down(MouseButton::Left), 3, 2),
            );
            match cancel {
                0 => render_editor(&mut editor, &mut state, area, false),
                1 => {
                    handle_mouse(
                        &mut editor,
                        &mut state,
                        Rect::default(),
                        mouse(MouseEventKind::Drag(MouseButton::Left), 4, 2),
                    );
                }
                2 => state.reset_navigation(),
                _ => {
                    handle_key(&mut editor, &mut state, key(KeyCode::Right));
                }
            }
            assert!(!handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Drag(MouseButton::Left), 4, 2),
            ));
            assert_eq!(editor.cursor(), (0, 1));
            assert!(!editor.is_selecting());
        }
    }

    #[test]
    fn pointer_handles_narrow_cells_and_columns_beyond_u16() {
        let mut editor = TextArea::from(["中👩‍💻e\u{301}"]);
        let mut state = ComposerState::default();
        let area = Rect::new(0, 0, 1, 4);
        render_editor(&mut editor, &mut state, area, true);
        for (row, column) in [(0, 0), (1, 1), (2, 4)] {
            handle_mouse(
                &mut editor,
                &mut state,
                area,
                mouse(MouseEventKind::Down(MouseButton::Left), 0, row),
            );
            assert_eq!(editor.cursor(), (0, column));
        }

        let mut editor = TextArea::from(["x".repeat(65_550)]);
        editor.move_cursor(CursorMove::End);
        let mut state = ComposerState::default();
        let area = Rect::new(0, 0, 10, 2);
        render_editor(&mut editor, &mut state, area, true);
        handle_mouse(
            &mut editor,
            &mut state,
            area,
            mouse(MouseEventKind::Down(MouseButton::Left), 2, 0),
        );
        assert_eq!(editor.cursor(), (0, 65_542));
    }
}

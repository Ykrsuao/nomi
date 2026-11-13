// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Terminal-native workspace rendering. Scroll positions are measured in wrapped rows.

use ratatui::{
    backend::Backend,
    layout::{Alignment, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame, Terminal,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::state::{clean, Entry, Overlay, Phase, State, Tool};
use crate::terminal_image::PreviewState;

#[derive(Default)]
pub struct HitAreas {
    /// Desired hardware cursor, applied by `render` after the frame is written.
    pub cursor: Option<Position>,
    pub transcript: Rect,
    pub composer: Rect,
    /// Exact text viewport, excluding the composer border and prompt gutter.
    pub composer_text: Rect,
    pub query: Option<Rect>,
    pub sidebar: Rect,
    pub tool_rows: Vec<(Rect, String)>,
    pub detail_rows: Vec<(Rect, usize)>,
    pub session_rows: Vec<(Rect, String)>,
    pub choice_rows: Vec<(Rect, usize)>,
    pub attachment_preview: Vec<(Rect, usize)>,
    /// Visible original-image actions and thumbnail cells in the attachment panel.
    pub attachment_original: Vec<Rect>,
    pub attachment_remove: Vec<(Rect, usize)>,
    pub attachment_overflow: Option<Rect>,
    pub attachment_undo: Option<Rect>,
    pub slash_rows: Vec<(Rect, usize)>,
    pub welcome_rows: Vec<(Rect, &'static str)>,
}

#[derive(Clone, Copy)]
struct Palette {
    bg: Color,
    panel: Color,
    fg: Color,
    muted: Color,
    faint: Color,
    accent: Color,
    green: Color,
    amber: Color,
    red: Color,
}

impl Palette {
    fn new(monochrome: bool) -> Self {
        if monochrome {
            return Self {
                bg: Color::Reset,
                panel: Color::Reset,
                fg: Color::Reset,
                muted: Color::Reset,
                faint: Color::Reset,
                accent: Color::Reset,
                green: Color::Reset,
                amber: Color::Reset,
                red: Color::Reset,
            };
        }
        Self {
            bg: Color::Rgb(20, 20, 20),
            panel: Color::Rgb(32, 32, 32),
            fg: Color::Rgb(225, 225, 225),
            muted: Color::Rgb(128, 128, 128),
            faint: Color::Rgb(64, 64, 64),
            accent: Color::Rgb(200, 200, 200),
            green: Color::Rgb(147, 186, 133),
            amber: Color::Rgb(220, 186, 115),
            red: Color::Rgb(224, 130, 130),
        }
    }
    fn text(self) -> Style {
        Style::default().fg(self.fg).bg(self.bg)
    }
    fn muted(self) -> Style {
        self.text().fg(self.muted)
    }
    fn accent(self) -> Style {
        self.text().fg(self.accent)
    }
    fn bold(self) -> Style {
        self.text().add_modifier(Modifier::BOLD)
    }
}

#[derive(Clone)]
struct Row {
    line: Line<'static>,
    tool: Option<String>,
    entry: Option<usize>,
    gutter: Option<Span<'static>>,
    /// An expansion control is clickable but is not part of searchable content.
    toggle: Option<usize>,
}

struct CachedAssistant {
    source: String,
    rows: Vec<Row>,
}

/// Completed replies need parsing and wrapping only when their text or layout changes.
#[derive(Default)]
pub(crate) struct TranscriptCache {
    layout: Option<(usize, bool)>,
    assistants: Vec<Option<CachedAssistant>>,
}

fn width(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}

fn fit(text: &str, cells: usize) -> String {
    if width(text) <= cells {
        return text.to_owned();
    }
    if cells == 0 {
        return String::new();
    }
    let mut result = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w > cells - 1 {
            break;
        }
        result.push(c);
        used += w;
    }
    result.push('…');
    result
}

fn elapsed(seconds: f64) -> String {
    let centiseconds = (seconds * 100.0).round() as u64;
    if centiseconds < 6_000 {
        format!("{}.{:02}s", centiseconds / 100, centiseconds % 100)
    } else {
        format!(
            "{}m {:02}.{:02}s",
            centiseconds / 6_000,
            centiseconds / 100 % 60,
            centiseconds % 100
        )
    }
}

fn inset(area: Rect, x: u16, y: u16) -> Rect {
    let x = x.min(area.width / 2);
    let y = y.min(area.height / 2);
    Rect::new(
        area.x + x,
        area.y + y,
        area.width.saturating_sub(2 * x),
        area.height.saturating_sub(2 * y),
    )
}

fn line_at(frame: &mut Frame, area: Rect, line: Line<'_>) {
    if area.width > 0 && area.height > 0 {
        frame.render_widget(
            Paragraph::new(line),
            Rect::new(area.x, area.y, area.width, 1),
        );
    }
}

fn pair(frame: &mut Frame, area: Rect, left: &str, right: &str, style: Style, muted: Style) {
    let right_width = width(right).min((area.width / 2) as usize);
    let available = area.width as usize;
    if available < 42 || right.is_empty() {
        line_at(frame, area, Line::styled(fit(left, available), style));
        return;
    }
    let left = fit(left, available.saturating_sub(right_width + 2));
    let right = fit(right, right_width);
    let gap = available.saturating_sub(width(&left) + width(&right));
    line_at(
        frame,
        area,
        Line::from(vec![
            Span::styled(left, style),
            Span::raw(" ".repeat(gap)),
            Span::styled(right, muted),
        ]),
    );
}

fn usage_header(frame: &mut Frame, area: Rect, state: &State, style: Style) {
    let available = area.width as usize;
    let compact = available < 60;
    let context = if let Some(ratio) = state.context_ratio {
        if compact {
            format!("ctx {:.0}%", ratio * 100.0)
        } else {
            format!("{:.0}% context", ratio * 100.0)
        }
    } else if state.usage.is_empty() {
        if compact { "ctx —" } else { "context —" }.into()
    } else if compact {
        if let Some(tokens) = state.usage.strip_suffix(" tokens") {
            format!("{tokens} tok")
        } else {
            state.usage.replacen("context ", "ctx ", 1)
        }
    } else {
        state.usage.clone()
    };
    let cost = if state.cost.is_empty() {
        "cost —"
    } else {
        &state.cost
    };
    // The header keeps both indicators visible on narrow terminals instead of
    // using pair(), which deliberately drops its right side below 42 columns.
    let separator = if available >= 12 { " · " } else { " " };
    let cost_width = width(cost).min(available.saturating_sub(width(separator)) / 2);
    let context_width = available.saturating_sub(width(separator) + cost_width);
    let context = fit(&context, context_width);
    let cost = fit(
        cost,
        available.saturating_sub(width(&context) + width(separator)),
    );
    let metrics = format!("{context}{separator}{cost}");
    let workspace_width = available.saturating_sub(width(&metrics) + 2);
    let workspace = if workspace_width >= 12 {
        fit(&state.workspace, workspace_width)
    } else {
        String::new()
    };
    let gap = available.saturating_sub(width(&workspace) + width(&metrics));
    line_at(
        frame,
        area,
        Line::styled(format!("{workspace}{}{metrics}", " ".repeat(gap)), style),
    );
}

fn markdown(text: &str, p: Palette) -> Vec<Line<'static>> {
    crate::markdown::render_transcript(
        text,
        crate::markdown::Theme {
            text: p.text(),
            muted: p.muted(),
            accent: p.accent(),
            code: p.text().bg(p.panel),
            added: p.text().fg(p.green).bg(p.panel),
            removed: p.text().fg(p.red).bg(p.panel),
            syntax: !matches!(p.bg, Color::Reset),
        },
    )
}

/// Wrap styled cells before scrolling so CJK and expanded tools share one row model.
fn wrap(line: Line<'static>, max_width: usize) -> Vec<Line<'static>> {
    if max_width == 0 {
        return Vec::new();
    }
    let base = line.style;
    let chars: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| {
            span.content
                .chars()
                .map(move |c| (c, base.patch(span.style)))
        })
        .collect();
    if chars.is_empty() {
        return vec![Line::default().style(base)];
    }
    let mut rows = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = start;
        let mut used = 0;
        let mut break_at = None;
        while end < chars.len() {
            let cells = chars[end].0.width().unwrap_or(0);
            if used + cells > max_width {
                break;
            }
            used += cells;
            if chars[end].0.is_whitespace() {
                break_at = Some(end + 1);
            }
            end += 1;
        }
        if end == start {
            end += 1;
        } else if end < chars.len() {
            if let Some(boundary) = break_at.filter(|b| *b > start + 1) {
                end = boundary;
            }
        }
        let mut spans: Vec<Span<'static>> = Vec::new();
        for (c, style) in &chars[start..end] {
            if let Some(last) = spans.last_mut().filter(|s| s.style == *style) {
                last.content.to_mut().push(*c);
            } else {
                spans.push(Span::styled(c.to_string(), *style));
            }
        }
        rows.push(Line::from(spans).style(base));
        start = end;
    }
    rows
}

fn append(rows: &mut Vec<Row>, line: Line<'static>, columns: usize, tool: Option<&str>) {
    rows.extend(wrap(line, columns).into_iter().map(|line| Row {
        line,
        tool: tool.map(str::to_owned),
        entry: None,
        gutter: None,
        toggle: None,
    }));
}

fn append_with_gutter(
    rows: &mut Vec<Row>,
    line: Line<'static>,
    columns: usize,
    gutter: Span<'static>,
    continuation: Span<'static>,
    tool: Option<&str>,
) {
    let available = columns.saturating_sub(width(&gutter.content) + 1).max(1);
    for (index, line) in wrap(line, available).into_iter().enumerate() {
        rows.push(Row {
            line,
            tool: tool.map(str::to_owned),
            entry: None,
            gutter: Some(if index == 0 {
                gutter.clone()
            } else {
                continuation.clone()
            }),
            toggle: None,
        });
    }
}

fn tool_color(tool: &Tool, p: Palette) -> Color {
    match tool.status.as_str() {
        "completed" => p.green,
        "failed" => p.red,
        "cancelled" | "canceled" => p.muted,
        _ => p.amber,
    }
}

fn tool_icon(tool: &Tool) -> &'static str {
    match tool.status.as_str() {
        "completed" => "✓",
        "failed" => "×",
        "cancelled" | "canceled" => "−",
        _ => "◌",
    }
}

fn tool_status_suffix(tool: &Tool) -> &'static str {
    // Match only the backend's outcome titles. Commands and paths containing
    // words such as "failed" still need their independent status indicator.
    let title = tool.title.trim();
    match tool.status.as_str() {
        "failed" => {
            if matches!(
                title,
                "Python failed" | "Python interrupted" | "Cancelled" | "Canceled" | "Unfinished"
            ) {
                ""
            } else {
                " · failed"
            }
        }
        "cancelled" | "canceled" => {
            if matches!(title, "Cancelled" | "Canceled") {
                ""
            } else {
                " · cancelled"
            }
        }
        "pending" => " · pending",
        "in_progress" => " · running",
        _ => "",
    }
}

fn diff_lines(text: &str, p: Palette) -> Vec<Line<'static>> {
    fn raw(text: &str, p: Palette) -> Vec<Line<'static>> {
        text.lines()
            .map(|line| {
                let color = if line.starts_with("+++") || line.starts_with("---") {
                    p.muted
                } else if line.starts_with('+') {
                    p.green
                } else if line.starts_with('-') {
                    p.red
                } else if line.starts_with("@@") {
                    p.accent
                } else {
                    p.fg
                };
                Line::styled(line.replace('\t', "    "), p.text().fg(color))
            })
            .collect()
    }
    fn range(text: &str, prefix: char) -> Option<(usize, usize)> {
        let text = text.strip_prefix(prefix)?;
        let (start, count) = text.split_once(',').unwrap_or((text, "1"));
        Some((start.parse().ok()?, count.parse().ok()?))
    }
    fn hunk(line: &str) -> Option<(usize, usize, usize, usize, &str)> {
        let (ranges, label) = line.strip_prefix("@@ ")?.split_once(" @@")?;
        let mut parts = ranges.split_whitespace();
        let (old, old_count) = range(parts.next()?, '-')?;
        let (new, new_count) = range(parts.next()?, '+')?;
        if parts.next().is_some() {
            return None;
        }
        Some((old, new, old_count, new_count, label.trim()))
    }
    let lines: Vec<_> = text.lines().collect();
    if !lines.iter().any(|line| hunk(line).is_some()) {
        return raw(text, p);
    }
    let mut rows = Vec::new();
    let (mut old, mut new, mut old_left, mut new_left) = (0usize, 0usize, 0usize, 0usize);
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if line.starts_with("\\ No newline") {
            rows.push(Line::styled(format!("       {line}"), p.muted()));
        } else if old_left > 0 || new_left > 0 {
            let (number, color, background) = match line.as_bytes().first() {
                Some(b'-') if old_left > 0 => {
                    let number = old;
                    old = old.saturating_add(1);
                    old_left -= 1;
                    (number, p.red, Color::Rgb(48, 28, 28))
                }
                Some(b'+') if new_left > 0 => {
                    let number = new;
                    new = new.saturating_add(1);
                    new_left -= 1;
                    (number, p.green, Color::Rgb(24, 44, 29))
                }
                Some(b' ') if old_left > 0 && new_left > 0 => {
                    let number = new;
                    old = old.saturating_add(1);
                    old_left -= 1;
                    new = new.saturating_add(1);
                    new_left -= 1;
                    (number, p.fg, p.bg)
                }
                _ => return raw(text, p),
            };
            let background = if p.bg == Color::Reset {
                Color::Reset
            } else {
                background
            };
            let style = p.text().fg(color).bg(background);
            rows.push(
                Line::from(vec![
                    Span::styled(format!("{number:>4} │ "), p.muted().bg(background)),
                    Span::styled(line.replace('\t', "    "), style),
                ])
                .style(style),
            );
        } else if line.starts_with("--- ")
            && lines
                .get(index + 1)
                .is_some_and(|line| line.starts_with("+++ "))
        {
            let old_path = line[4..].split('\t').next().unwrap_or("");
            index += 1;
            let new_path = lines[index][4..].split('\t').next().unwrap_or("");
            let path = if new_path == "/dev/null" {
                old_path
            } else {
                new_path
            };
            let path = path
                .strip_prefix("a/")
                .or_else(|| path.strip_prefix("b/"))
                .unwrap_or(path);
            rows.push(Line::styled(path.to_owned(), p.bold()));
        } else if let Some((old_start, new_start, old_count, new_count, label)) = hunk(line) {
            old = old_start;
            new = new_start;
            old_left = old_count;
            new_left = new_count;
            rows.push(Line::styled(
                format!(
                    "   ⋯  lines {old} → {new}{}",
                    if label.is_empty() {
                        String::new()
                    } else {
                        format!(" · {label}")
                    }
                ),
                p.muted(),
            ));
        } else if line.starts_with("@@") {
            return raw(text, p);
        } else {
            rows.extend(raw(line, p));
        }
        index += 1;
    }
    if old_left > 0 || new_left > 0 {
        return raw(text, p);
    }
    rows
}

#[cfg(test)]
mod diff_tests {
    use super::*;

    fn plain(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn mixed_hunks_use_old_numbers_for_deletions_and_new_numbers_after_insertions() {
        let p = Palette::new(false);
        let rows = diff_lines("--- a/src/main.rs\n+++ b/src/main.rs\n@@ -10,3 +10,4 @@ function\n keep\n-old_title\n+new_title\n+extra\n tail\n@@ -30 +31 @@\n-last\n+final", p);
        let text: Vec<_> = rows.iter().map(plain).collect();
        assert_eq!(
            text,
            [
                "src/main.rs",
                "   ⋯  lines 10 → 10 · function",
                "  10 │  keep",
                "  11 │ -old_title",
                "  11 │ +new_title",
                "  12 │ +extra",
                "  13 │  tail",
                "   ⋯  lines 30 → 31",
                "  30 │ -last",
                "  31 │ +final"
            ]
        );
        assert_eq!(rows[3].style.bg, Some(Color::Rgb(48, 28, 28)));
        assert_eq!(rows[4].style.bg, Some(Color::Rgb(24, 44, 29)));
        assert!(rows[4]
            .spans
            .iter()
            .all(|span| span.style.bg == rows[4].style.bg));
    }

    #[test]
    fn malformed_or_incomplete_diffs_keep_the_raw_text() {
        for text in [
            "+raw output\n-context",
            "@@ invalid @@\n+text",
            "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n only one",
            "@@ -1 +1 @@\n-not enough",
        ] {
            assert_eq!(
                diff_lines(text, Palette::new(false))
                    .iter()
                    .map(plain)
                    .collect::<Vec<_>>()
                    .join("\n"),
                text
            );
        }
    }

    #[test]
    fn new_files_and_monochrome_preserve_signs_and_reset_colors() {
        let rows = diff_lines(
            "--- /dev/null\n+++ b/中文.txt\n@@ -0,0 +1,2 @@\n+first\n+第二行",
            Palette::new(true),
        );
        assert_eq!(plain(&rows[0]), "中文.txt");
        assert_eq!(plain(&rows[2]), "   1 │ +first");
        assert_eq!(plain(&rows[3]), "   2 │ +第二行");
        assert!(rows.iter().all(|line| line.style.bg == Some(Color::Reset)));
    }
}

fn transcript_rows(state: &mut State, columns: usize, p: Palette) -> Vec<Row> {
    let cache = &mut state.transcript_cache;
    if cache.layout != Some((columns, state.monochrome)) {
        cache.assistants.clear();
        cache.layout = Some((columns, state.monochrome));
    }
    cache.assistants.resize_with(state.entries.len(), || None);
    let mut rows = Vec::new();
    for (index, entry) in state.entries.iter().enumerate() {
        if !matches!(entry, Entry::Assistant(_)) {
            cache.assistants[index] = None;
        }
        let consecutive_tools = index > 0
            && matches!((&state.entries[index - 1], entry),
                (Entry::Tool(previous), Entry::Tool(_)) if !previous.expanded);
        if !rows.is_empty() && !consecutive_tools {
            rows.push(Row {
                line: Line::default(),
                tool: None,
                entry: None,
                gutter: None,
                toggle: None,
            });
        }
        let start = rows.len();
        match entry {
            Entry::User(text) => {
                let background = if state.monochrome {
                    Color::Reset
                } else {
                    Color::Rgb(36, 36, 36)
                };
                let style = if state.monochrome { p.bold() } else { p.text() }.bg(background);
                let mut first = true;
                for line in text.lines() {
                    append_with_gutter(
                        &mut rows,
                        Line::styled(line.to_owned(), style),
                        columns,
                        Span::styled(if first { "❯ " } else { "  " }, p.accent().bg(background)),
                        Span::styled("  ", style),
                        None,
                    );
                    first = false;
                }
                if first {
                    append_with_gutter(
                        &mut rows,
                        Line::styled("", style),
                        columns,
                        Span::styled("❯ ", p.accent().bg(background)),
                        Span::styled("  ", style),
                        None,
                    );
                }
                let total = rows.len() - start;
                if total > 3 {
                    let expanded = state.expanded_entries.contains(&index);
                    if !expanded {
                        rows.truncate(start + 2);
                    }
                    let label = if expanded {
                        "Show less · click or Ctrl+O".to_owned()
                    } else {
                        format!("… {} more lines · click to expand", total - 2)
                    };
                    rows.push(Row {
                        line: Line::styled(
                            fit(&label, columns.saturating_sub(3)),
                            p.muted().bg(background),
                        ),
                        tool: None,
                        entry: None,
                        gutter: Some(Span::styled("  ", style)),
                        toggle: Some(index),
                    });
                }
            }
            Entry::Assistant(text) => {
                let cached = &mut cache.assistants[index];
                if cached.as_ref().is_none_or(|cached| cached.source != *text) {
                    let mut rendered = Vec::new();
                    for line in markdown(text, p) {
                        append(&mut rendered, line, columns, None);
                    }
                    *cached = Some(CachedAssistant {
                        source: text.clone(),
                        rows: rendered,
                    });
                }
                rows.extend(cached.as_ref().unwrap().rows.iter().cloned());
            }
            Entry::Notice { text, error } => {
                for (index, line) in text.lines().enumerate() {
                    let prefix = if index > 0 {
                        "  "
                    } else if *error {
                        "! "
                    } else {
                        "· "
                    };
                    let style = p.text().fg(if *error { p.red } else { p.muted });
                    append_with_gutter(
                        &mut rows,
                        Line::styled(line.to_owned(), style),
                        columns,
                        Span::styled(prefix, style),
                        Span::styled("  ", style),
                        None,
                    );
                }
            }
            Entry::Tool(tool) => {
                let active = matches!(tool.status.as_str(), "pending" | "in_progress");
                let terminal = tool_status_suffix(tool);
                let duration = tool.elapsed_seconds();
                let verb_end = tool
                    .title
                    .find(char::is_whitespace)
                    .unwrap_or(tool.title.len());
                let mut spans = vec![
                    Span::styled(tool.title[..verb_end].to_owned(), p.bold()),
                    Span::styled(tool.title[verb_end..].to_owned(), p.muted()),
                ];
                if !tool.diff.is_empty() {
                    let added = tool
                        .diff
                        .lines()
                        .filter(|line| line.starts_with('+') && !line.starts_with("+++"))
                        .count();
                    let removed = tool
                        .diff
                        .lines()
                        .filter(|line| line.starts_with('-') && !line.starts_with("---"))
                        .count();
                    spans.push(Span::styled(format!("  +{added}"), p.text().fg(p.green)));
                    spans.push(Span::styled(format!(" −{removed}"), p.text().fg(p.red)));
                }
                spans.push(Span::styled(terminal, p.text().fg(tool_color(tool, p))));
                if let Some(seconds) = duration {
                    spans.push(Span::styled(format!("  {}", elapsed(seconds)), p.muted()));
                }
                append_with_gutter(
                    &mut rows,
                    Line::from(spans),
                    columns,
                    Span::styled(
                        "◆ ",
                        p.text().fg(if tool.status == "completed" {
                            p.muted
                        } else {
                            tool_color(tool, p)
                        }),
                    ),
                    Span::styled("  ", p.text()),
                    Some(&tool.id),
                );
                if tool.expanded {
                    let output_start = rows.len();
                    for line in tool.content.lines() {
                        append_with_gutter(
                            &mut rows,
                            Line::styled(line.replace('\t', "    "), p.muted()),
                            columns,
                            Span::styled("  │ ", p.muted()),
                            Span::styled("  │ ", p.muted()),
                            Some(&tool.id),
                        );
                    }
                    let output_rows = rows.len() - output_start;
                    if output_rows > 8 && !state.expanded_entries.contains(&index) {
                        let tail = rows.split_off(rows.len() - 3);
                        rows.truncate(output_start + 2);
                        rows.push(Row {
                            line: Line::styled(
                                fit(
                                    &format!("… {} more lines · click to expand", output_rows - 5),
                                    columns.saturating_sub(5),
                                ),
                                p.muted(),
                            ),
                            tool: None,
                            entry: None,
                            gutter: Some(Span::styled("  │ ", p.muted())),
                            toggle: Some(index),
                        });
                        rows.extend(tail);
                    }
                    for line in diff_lines(&tool.diff, p) {
                        append_with_gutter(
                            &mut rows,
                            line,
                            columns,
                            Span::styled("  │ ", p.muted()),
                            Span::styled("  │ ", p.muted()),
                            Some(&tool.id),
                        );
                    }
                    if output_rows > 8 && state.expanded_entries.contains(&index) {
                        rows.push(Row {
                            line: Line::styled("Show less · click or Ctrl+O", p.muted()),
                            tool: None,
                            entry: None,
                            gutter: Some(Span::styled("    ", p.muted())),
                            toggle: Some(index),
                        });
                    }
                } else if active {
                    if let Some(last) = tool
                        .content
                        .lines()
                        .rev()
                        .find(|line| !line.trim().is_empty())
                    {
                        append_with_gutter(
                            &mut rows,
                            Line::styled(fit(last, columns.saturating_sub(5)), p.muted()),
                            columns,
                            Span::styled("    ", p.text()),
                            Span::styled("    ", p.text()),
                            Some(&tool.id),
                        );
                    }
                }
            }
        }
        for row in &mut rows[start..] {
            if row.toggle.is_none() {
                row.entry = Some(index);
            }
        }
    }
    rows
}

fn render_rows(
    frame: &mut Frame,
    area: Rect,
    rows: &[Row],
    start: usize,
    hits: &mut Vec<(Rect, String)>,
    details: &mut Vec<(Rect, usize)>,
) {
    for (index, row) in rows
        .iter()
        .skip(start)
        .take(area.height as usize)
        .enumerate()
    {
        let rect = Rect::new(area.x, area.y + index as u16, area.width, 1);
        let gutter_width = row
            .gutter
            .as_ref()
            .map_or(0, |span| width(&span.content) as u16)
            .min(rect.width);
        if let Some(gutter) = &row.gutter {
            line_at(
                frame,
                Rect::new(rect.x, rect.y, gutter_width, 1),
                Line::from(gutter.clone()),
            );
        }
        let content = Rect::new(
            rect.x + gutter_width,
            rect.y,
            rect.width.saturating_sub(gutter_width),
            1,
        );
        frame.render_widget(Block::default().style(row.line.style), content);
        line_at(frame, content, row.line.clone());
        if let Some(id) = &row.tool {
            hits.push((rect, id.clone()));
        }
        if let Some(index) = row.toggle {
            details.push((rect, index));
        }
    }
}

#[cfg(test)]
mod transcript_tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    use std::time::{Duration, Instant};

    #[test]
    fn cached_replies_follow_text_width_theme_and_highlight_changes() {
        let p = Palette::new(false);
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.monochrome = false;
        state
            .entries
            .push(Entry::Assistant("Read **original** response.".into()));
        let mut rows = transcript_rows(&mut state, 80, p);
        highlight_rows(&mut rows, "original", p);
        assert!(rows
            .iter()
            .flat_map(|row| &row.line.spans)
            .any(|s| s.style.bg == Some(p.amber)));
        let rows = transcript_rows(&mut state, 80, p);
        assert!(rows
            .iter()
            .flat_map(|row| &row.line.spans)
            .all(|s| s.style.bg != Some(p.amber)));

        state.entries[0] =
            Entry::Assistant("Updated **中文 response** with a different body and length.".into());
        let wide = transcript_rows(&mut state, 80, p);
        let narrow = transcript_rows(&mut state, 16, p);
        let text = |rows: &[Row]| {
            rows.iter()
                .flat_map(|row| &row.line.spans)
                .map(|span| span.content.as_ref())
                .collect::<String>()
        };
        assert_eq!(
            text(&wide),
            "Updated 中文 response with a different body and length."
        );
        assert_eq!(text(&narrow), text(&wide));
        assert!(narrow.len() > wide.len());

        state.monochrome = true;
        let mono = transcript_rows(&mut state, 16, Palette::new(true));
        assert!(
            mono.iter()
                .flat_map(|row| &row.line.spans)
                .all(|span| span.style.fg == Some(Color::Reset)
                    && span.style.bg == Some(Color::Reset))
        );
        state.entries.clear();
        assert!(transcript_rows(&mut state, 16, Palette::new(true)).is_empty());
        state
            .entries
            .push(Entry::Assistant("Different conversation".into()));
        assert_eq!(
            text(&transcript_rows(&mut state, 16, Palette::new(true))),
            "Different conversation"
        );
    }

    #[test]
    fn user_and_code_backgrounds_fill_rows_without_changing_search_text() {
        let p = Palette::new(false);
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.monochrome = false;
        let user = "保留中文和缩进，跨行查找仍能命中";
        state.entries = vec![
            Entry::User(user.into()),
            Entry::Assistant("```rust\nlet value = 1;\n```".into()),
        ];
        let rows = transcript_rows(&mut state, 20, p);
        let pure: String = rows
            .iter()
            .filter(|row| row.entry == Some(0))
            .flat_map(|row| &row.line.spans)
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(pure, user);
        assert!(search_row(&rows, 0, "跨行查找").is_some());
        let code_row = rows
            .iter()
            .position(|row| {
                row.line
                    .spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
                    .contains("let value")
            })
            .unwrap();
        let mut terminal = Terminal::new(TestBackend::new(20, 12)).unwrap();
        terminal
            .draw(|frame| {
                render_rows(
                    frame,
                    Rect::new(0, 0, 20, 12),
                    &rows,
                    0,
                    &mut Vec::new(),
                    &mut Vec::new(),
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(19, 0)].bg, Color::Rgb(36, 36, 36));
        assert_eq!(buffer[(19, code_row as u16)].bg, p.panel);
        assert_eq!(buffer[(0, 0)].symbol(), "❯");
    }

    #[test]
    fn completed_tools_form_a_compact_group_and_cancelled_tools_do_not_look_active() {
        let p = Palette::new(false);
        let mut state = State::new("workspace".into(), "model".into(), false);
        for (id, status) in [("read", "completed"), ("test", "cancelled")] {
            state.entries.push(Entry::Tool(Tool {
                id: id.into(),
                title: id.into(),
                status: status.into(),
                content: "old output".into(),
                diff: String::new(),
                expanded: false,
                started: Some(Instant::now() - Duration::from_secs(65)),
                elapsed: None,
            }));
        }
        let rows = transcript_rows(&mut state, 80, p);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].gutter.as_ref().unwrap().content, "◆ ");
        assert_eq!(rows[0].gutter.as_ref().unwrap().style.fg, Some(p.muted));
        let cancelled: String = rows[1]
            .line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(cancelled.contains("cancelled"));
        assert!(!cancelled.contains("1m"));
        assert!(!cancelled.contains("◌"));
        let mut terminal = Terminal::new(TestBackend::new(80, 5)).unwrap();
        let mut hits = Vec::new();
        terminal
            .draw(|frame| {
                render_rows(
                    frame,
                    Rect::new(0, 0, 80, 5),
                    &rows,
                    0,
                    &mut hits,
                    &mut Vec::new(),
                )
            })
            .unwrap();
        assert_eq!(
            hits.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>(),
            ["read", "test"]
        );
    }

    #[test]
    fn literal_box_drawing_text_does_not_become_a_code_panel() {
        for monochrome in [false, true] {
            let mut state = State::new("workspace".into(), "model".into(), false);
            state.monochrome = monochrome;
            state.entries.push(Entry::Assistant(
                "┌─ literal text\nordinary words\n└─".into(),
            ));
            let rows = transcript_rows(&mut state, 80, Palette::new(monochrome));
            let plain: Vec<String> = rows
                .iter()
                .map(|row| {
                    row.line
                        .spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect()
                })
                .collect();
            assert_eq!(plain, ["┌─ literal text", "ordinary words", "└─"]);
            assert!(rows.iter().all(|row| row.gutter.is_none()));
        }
    }

    #[test]
    fn tool_heading_emphasizes_verb_and_expanded_output_stays_on_canvas() {
        let p = Palette::new(false);
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.entries.push(Entry::Tool(Tool {
            id: "read".into(),
            title: "Read src/main.rs".into(),
            status: "completed".into(),
            content: "loaded source".into(),
            diff: "+changed".into(),
            expanded: true,
            started: Some(Instant::now()),
            elapsed: Some(0.3),
        }));
        let rows = transcript_rows(&mut state, 80, p);
        assert_eq!(rows[0].line.spans[0].content, "Read");
        assert!(rows[0].line.spans[0]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
        assert_eq!(rows[0].line.spans[1].content, " src/main.rs");
        assert_eq!(rows[0].line.spans[1].style.fg, Some(p.muted));
        assert_eq!(rows[1].line.style.bg, Some(p.bg));
        assert_eq!(rows[2].line.style.bg, Some(p.bg));
        assert!(rows.iter().all(|row| row.tool.as_deref() == Some("read")));
    }

    #[test]
    fn terminal_status_does_not_repeat_outcome_already_in_tool_title() {
        for monochrome in [false, true] {
            for expanded in [false, true] {
                for (title, status, expected) in [
                    ("Python failed", "failed", "Python failed"),
                    ("Python interrupted", "failed", "Python interrupted"),
                    ("Cancelled", "failed", "Cancelled"),
                    ("Cancelled", "cancelled", "Cancelled"),
                    ("Canceled", "canceled", "Canceled"),
                    ("Unfinished", "failed", "Unfinished"),
                    ("Running Python", "in_progress", "Running Python · running"),
                    ("$ echo failed", "failed", "$ echo failed · failed"),
                    ("Read failed.log", "failed", "Read failed.log · failed"),
                    (
                        "$ echo Cancelled",
                        "cancelled",
                        "$ echo Cancelled · cancelled",
                    ),
                    ("Python failed", "pending", "Python failed · pending"),
                ] {
                    let p = Palette::new(monochrome);
                    let mut state = State::new("workspace".into(), "model".into(), false);
                    state.monochrome = monochrome;
                    state.entries.push(Entry::Tool(Tool {
                        id: "tool".into(),
                        title: title.into(),
                        status: status.into(),
                        content: "output".into(),
                        diff: String::new(),
                        expanded,
                        started: None,
                        elapsed: Some(0.01),
                    }));
                    let rows = transcript_rows(&mut state, 80, p);
                    let mut terminal = Terminal::new(TestBackend::new(80, 5)).unwrap();
                    terminal
                        .draw(|frame| {
                            render_rows(
                                frame,
                                Rect::new(0, 0, 80, 5),
                                &rows,
                                0,
                                &mut Vec::new(),
                                &mut Vec::new(),
                            )
                        })
                        .unwrap();
                    let buffer = terminal.backend().buffer();
                    let heading: String = (0..80).map(|x| buffer[(x, 0)].symbol()).collect();
                    assert_eq!(heading.trim_end(), format!("◆ {expected}  0.01s"));
                    if status == "failed" {
                        assert_eq!(buffer[(0, 0)].fg, p.red);
                    }
                }
            }
        }
    }
}

fn search_row(rows: &[Row], entry: usize, query: &str) -> Option<usize> {
    let start = rows.iter().position(|row| row.entry == Some(entry))?;
    let mut text = String::new();
    let mut boundaries = Vec::new();
    for (index, row) in rows.iter().enumerate().skip(start) {
        if row.entry != Some(entry) {
            break;
        }
        boundaries.push((text.len(), index));
        for span in &row.line.spans {
            text.push_str(&span.content);
        }
    }
    crate::search::match_range(&text, query)
        .and_then(|found| {
            boundaries
                .into_iter()
                .rev()
                .find(|(offset, _)| *offset <= found.start)
        })
        .map(|(_, index)| index)
        .or(Some(start))
}

fn highlight(line: &mut Line<'static>, query: &str, p: Palette) {
    let text: String = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    let Some(found) = crate::search::match_range(&text, query) else {
        return;
    };
    highlight_range(line, found, p);
}

fn highlight_range(line: &mut Line<'static>, found: std::ops::Range<usize>, p: Palette) {
    let mut offset = 0;
    let mut spans = Vec::new();
    for span in std::mem::take(&mut line.spans) {
        let end = offset + span.content.len();
        if offset < found.end && end > found.start {
            let left = found.start.saturating_sub(offset);
            let right = (found.end - offset).min(span.content.len());
            if left > 0 {
                spans.push(Span::styled(span.content[..left].to_owned(), span.style));
            }
            let style = if p.bg == Color::Reset {
                span.style.add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                span.style.fg(p.bg).bg(p.amber).add_modifier(Modifier::BOLD)
            };
            spans.push(Span::styled(span.content[left..right].to_owned(), style));
            if right < span.content.len() {
                spans.push(Span::styled(span.content[right..].to_owned(), span.style));
            }
        } else {
            spans.push(span);
        }
        offset = end;
    }
    line.spans = spans;
}

fn highlight_rows(rows: &mut [Row], query: &str, p: Palette) {
    let mut start = 0;
    while start < rows.len() {
        if rows[start].entry.is_none() {
            start += 1;
            continue;
        }
        let entry = rows[start].entry;
        let end = rows[start..]
            .iter()
            .position(|row| row.entry != entry)
            .map_or(rows.len(), |count| start + count);
        let mut text = String::new();
        let mut offsets = Vec::new();
        for row in &rows[start..end] {
            let from = text.len();
            for span in &row.line.spans {
                text.push_str(&span.content);
            }
            offsets.push(from..text.len());
        }
        let mut offset = 0;
        while let Some(found) = crate::search::match_range(&text[offset..], query) {
            let range = offset + found.start..offset + found.end;
            for (row, boundary) in rows[start..end].iter_mut().zip(&offsets) {
                if range.start < boundary.end && range.end > boundary.start {
                    highlight_range(
                        &mut row.line,
                        range.start.saturating_sub(boundary.start)
                            ..range.end.min(boundary.end) - boundary.start,
                        p,
                    );
                }
            }
            offset = range.end;
        }
        start = end;
    }
}

/// Include enough neighboring text to recognize matches clipped by the viewport.
fn highlight_visible_rows(
    rows: &mut [Row],
    visible_start: usize,
    visible_end: usize,
    query: &str,
    p: Palette,
) {
    if visible_start >= visible_end || visible_start >= rows.len() || query.is_empty() {
        return;
    }
    let visible_end = visible_end.min(rows.len());
    let context_bytes = query.chars().count().saturating_mul(4);
    let row_bytes = |row: &Row| {
        row.line
            .spans
            .iter()
            .map(|span| span.content.len())
            .sum::<usize>()
            .max(1)
    };
    let mut start = visible_start;
    let mut bytes = 0;
    while start > 0 && bytes < context_bytes && rows[start - 1].entry == rows[visible_start].entry {
        start -= 1;
        bytes = bytes.saturating_add(row_bytes(&rows[start]));
    }
    let mut end = visible_end;
    bytes = 0;
    while end < rows.len()
        && bytes < context_bytes
        && rows[end].entry == rows[visible_end - 1].entry
    {
        bytes = bytes.saturating_add(row_bytes(&rows[end]));
        end += 1;
    }
    highlight_rows(&mut rows[start..end], query, p);
}

fn search_preview(text: &str, query: &str, columns: usize) -> String {
    let Some(found) = crate::search::match_range(text, query) else {
        return fit(text, columns);
    };
    let mut start = found.start;
    let mut used = 0;
    for (index, c) in text[..found.start].char_indices().rev() {
        used += c.width().unwrap_or(0);
        if used > columns / 3 {
            break;
        }
        start = index;
    }
    fit(
        &format!("{}{}", if start > 0 { "…" } else { "" }, &text[start..]),
        columns,
    )
}

fn welcome(frame: &mut Frame, area: Rect, p: Palette, hits: &mut HitAreas) {
    let menu_width = area.width.saturating_sub(4).min(51);
    if menu_width < 12 {
        return;
    }
    let x = area.x + (area.width - menu_width) / 2;
    let mut y = area.y + area.height.saturating_sub(12) / 3;
    // ASTER wordmark; measure the artwork so narrow terminals never crop letters.
    let logo = [
        " █████╗  ███████╗ ████████╗ ███████╗ ██████╗ ",
        "██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗",
        "███████║ ███████╗    ██║    █████╗   ██████╔╝",
        "██╔══██║ ╚════██║    ██║    ██╔══╝   ██╔══██╗",
        "██║  ██║ ███████║    ██║    ███████╗ ██║  ██║",
        "╚═╝  ╚═╝ ╚══════╝    ╚═╝    ╚══════╝ ╚═╝  ╚═╝",
    ];
    let logo_width = logo.iter().map(|line| width(line)).max().unwrap_or(0);
    if area.height >= 18 && menu_width as usize >= logo_width {
        for line in logo {
            line_at(
                frame,
                Rect::new(x, y, menu_width, 1),
                Line::styled(line, p.muted()).alignment(Alignment::Center),
            );
            y += 1;
        }
        y += 2;
    }
    for (label, key, command) in [
        ("Resume session", "ctrl+r", "/sessions"),
        ("Commands", "ctrl+k", "/"),
        ("Keyboard shortcuts", "ctrl+x", "/help"),
        ("Quit", "ctrl+q", "/exit"),
    ] {
        if y >= area.bottom() {
            break;
        }
        let rect = Rect::new(x, y, menu_width, 1);
        let label = fit(
            label,
            menu_width.saturating_sub(width(key) as u16 + 2) as usize,
        );
        let gap = menu_width as usize - width(&label) - width(key).min(menu_width as usize);
        line_at(
            frame,
            rect,
            Line::from(vec![
                Span::styled(label, p.bold()),
                Span::raw(" ".repeat(gap)),
                Span::styled(key, p.muted()),
            ]),
        );
        hits.welcome_rows.push((rect, command));
        y += 1;
    }
    if y + 3 < area.bottom() {
        line_at(
            frame,
            Rect::new(x, y + 2, menu_width, 1),
            Line::styled("Type a message to start building.", p.muted()),
        );
    }
}

fn sidebar(frame: &mut Frame, area: Rect, state: &State, p: Palette, hits: &mut HitAreas) {
    if area.width == 0 {
        return;
    }
    frame.render_widget(
        Block::default()
            .borders(Borders::LEFT)
            .border_style(p.text().fg(p.faint)),
        area,
    );
    let inner = Rect::new(
        area.x + 2,
        area.y,
        area.width.saturating_sub(3),
        area.height,
    );
    pair(frame, inner, "Conversations", "Ctrl+R", p.bold(), p.muted());
    let session_slots = (inner.height / 3).clamp(2, 8) as usize;
    let mut y = inner.y + 2;
    if state.sessions.is_empty() {
        line_at(
            frame,
            Rect::new(inner.x, y, inner.width, 1),
            Line::styled("No saved conversations", p.muted()),
        );
        y += 1;
    }
    for session in state.sessions.iter().take(session_slots) {
        if y + 1 >= area.bottom() {
            break;
        }
        let current = state.session_id.as_deref() == Some(&session.id);
        let marker = if current { "● " } else { "  " };
        let title = if session.title.is_empty() {
            "Untitled conversation"
        } else {
            &session.title
        };
        let rect = Rect::new(inner.x, y, inner.width, 1);
        line_at(
            frame,
            rect,
            Line::styled(
                fit(&format!("{marker}{title}"), inner.width as usize),
                if current { p.accent() } else { p.text() },
            ),
        );
        hits.session_rows.push((rect, session.id.clone()));
        y += 2;
    }
    y += 1;
    if y + 3 < area.bottom() {
        line_at(
            frame,
            Rect::new(inner.x, y, inner.width, 1),
            Line::styled("Activity", p.bold()),
        );
        y += 2;
        let tools: Vec<&Tool> = state
            .entries
            .iter()
            .rev()
            .filter_map(|e| {
                if let Entry::Tool(t) = e {
                    Some(t)
                } else {
                    None
                }
            })
            .take(5)
            .collect();
        if tools.is_empty() {
            line_at(
                frame,
                Rect::new(inner.x, y, inner.width, 1),
                Line::styled(
                    fit("Tools appear here as you work.", inner.width as usize),
                    p.muted(),
                ),
            );
        }
        for tool in tools {
            if y >= area.bottom().saturating_sub(3) {
                break;
            }
            let rect = Rect::new(inner.x, y, inner.width, 1);
            line_at(
                frame,
                rect,
                Line::styled(
                    fit(
                        &format!("{} {}", tool_icon(tool), tool.title),
                        inner.width as usize,
                    ),
                    p.text().fg(tool_color(tool, p)),
                ),
            );
            hits.tool_rows.push((rect, tool.id.clone()));
            y += 2;
        }
    }
    if inner.height > 5 {
        line_at(
            frame,
            Rect::new(inner.x, area.bottom() - 2, inner.width, 1),
            Line::styled("Alt+B hide sidebar", p.muted()),
        );
    }
}

fn approval_overlay(frame: &mut Frame, state: &mut State, p: Palette, hits: &mut HitAreas) {
    state.approval_page_rows = 1;
    let Some(approval) = state.approvals.front() else {
        return;
    };
    let screen = frame.area();
    let margin = if screen.width >= 80 { 4 } else { 1 };
    let panel_width = screen.width.saturating_sub(margin * 2).min(84);
    let columns = panel_width.saturating_sub(4) as usize;
    if columns == 0 {
        return;
    }
    let panel = p.text().bg(p.panel);
    let muted = p.muted().bg(p.panel);
    let mut header = Vec::new();
    let title = approval
        .preview
        .as_ref()
        .map(|preview| format!("{}?", preview.summary))
        .unwrap_or_else(|| fit(&approval.title, columns));
    append(
        &mut header,
        Line::styled(title, panel.add_modifier(Modifier::BOLD)),
        columns,
        None,
    );
    append(
        &mut header,
        Line::styled(
            if state.sandbox {
                "Sandbox execution"
            } else {
                "Native execution · this computer"
            },
            panel.fg(p.amber),
        ),
        columns,
        None,
    );
    append(
        &mut header,
        Line::styled("Allow once approves the entire Python cell.", muted),
        columns,
        None,
    );
    let can_summarize = approval
        .preview
        .as_ref()
        .is_some_and(|preview| preview.complete);
    let show_code = approval.show_code || !can_summarize;
    let mut details = Vec::new();
    if show_code {
        append(
            &mut details,
            Line::styled(
                if can_summarize {
                    "Full Python code"
                } else {
                    "Full code · preview unavailable or incomplete"
                },
                muted,
            ),
            columns,
            None,
        );
        for line in approval.code.lines() {
            append(
                &mut details,
                Line::styled(line.replace('\t', "    "), panel),
                columns,
                None,
            );
        }
    } else if let Some(preview) = &approval.preview {
        for detail in &preview.details {
            for line in detail.lines() {
                append(
                    &mut details,
                    Line::styled(line.replace('\t', "    "), panel),
                    columns,
                    None,
                );
            }
        }
    }
    state.selected = state.selected.min(approval.options.len().saturating_sub(1));
    let mut options = Vec::new();
    let mut option_ranges = Vec::new();
    for (index, option) in approval.options.iter().enumerate() {
        let selected = index == state.selected;
        let style = if selected {
            panel.fg(p.accent).bg(p.faint).add_modifier(Modifier::BOLD)
        } else {
            panel
        };
        let start = options.len();
        append(
            &mut options,
            Line::styled(
                format!(
                    "{} {}. {}",
                    if selected { "›" } else { " " },
                    index + 1,
                    option.name
                ),
                style,
            ),
            columns,
            None,
        );
        option_ranges.push((start, options.len(), index));
    }
    let mut footer = Vec::new();
    let toggle = if !can_summarize {
        ""
    } else if show_code {
        " · Tab summary"
    } else {
        " · Tab full code"
    };
    append(
        &mut footer,
        Line::styled(
            format!("↑↓ choose · Enter confirm{toggle} · PgUp/PgDn scroll"),
            muted,
        ),
        columns,
        None,
    );
    let max_height = screen.height.saturating_sub(if screen.height >= 24 {
        4
    } else if screen.height >= 16 {
        2
    } else {
        0
    });
    let desired = header.len()
        + details.len().min(12)
        + options.len()
        + footer.len()
        + 5
        + usize::from(details.len() > 12);
    let panel_height = desired.min(u16::MAX as usize) as u16;
    let panel_height = panel_height.max(8).min(max_height);
    let detail_rows = details.len().clamp(1, 3);
    let available_rows = panel_height.saturating_sub(2) as usize;
    let compact = header.len() + options.len() + footer.len() + detail_rows > available_rows;
    if compact {
        header.clear();
        append(
            &mut header,
            Line::styled("Entire Python cell", muted),
            columns,
            None,
        );
        if header.len() + options.len() + footer.len() + detail_rows > available_rows {
            footer.clear();
            append(
                &mut footer,
                Line::styled(
                    format!("↑↓/↵{} PgUp/PgDn", if can_summarize { " Tab" } else { "" }),
                    muted,
                ),
                columns,
                None,
            );
        }
    }
    let area = Rect::new(
        screen.x + (screen.width - panel_width) / 2,
        screen.y + (screen.height - panel_height) / 2,
        panel_width,
        panel_height,
    );
    for cell in &mut frame.buffer_mut().content {
        let dimmed = match cell.fg {
            Color::Rgb(r, g, b) => Color::Rgb(r / 2, g / 2, b / 2),
            Color::Reset => Color::Reset,
            _ => p.muted,
        };
        cell.set_style(
            Style::default()
                .fg(dimmed)
                .bg(p.bg)
                .add_modifier(Modifier::DIM)
                .remove_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
    }
    frame.render_widget(Clear, area);
    let block = Block::default()
        .style(panel)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(muted.fg(p.faint))
        .title(Span::styled(
            if compact {
                if state.sandbox {
                    " Sandbox execution "
                } else {
                    " Native execution "
                }
            } else {
                " Review execution "
            },
            panel.add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Line::styled(" Esc reject ", muted).alignment(Alignment::Right));
    let content = inset(block.inner(area), 1, 0);
    frame.render_widget(block, area);
    if content.height == 0 || content.width == 0 {
        return;
    }
    // Spend scarce rows on the operation being approved, before decorative gaps.
    let fixed_rows = header.len() + options.len() + footer.len();
    let gaps = (content.height as usize)
        .saturating_sub(fixed_rows + detail_rows)
        .min(3);
    let header_gap = u16::from(gaps >= 1);
    let options_gap = u16::from(gaps >= 2);
    let footer_gap = u16::from(gaps >= 3);
    let header_height = header.len().min(content.height.saturating_sub(2) as usize) as u16;
    let footer_height = footer
        .len()
        .min(content.height.saturating_sub(header_height + 2) as usize)
        as u16;
    let footer_area = Rect::new(
        content.x,
        content.bottom() - footer_height,
        content.width,
        footer_height,
    );
    render_rows(
        frame,
        footer_area,
        &footer,
        0,
        &mut Vec::new(),
        &mut Vec::new(),
    );
    let options_bottom = footer_area.y.saturating_sub(footer_gap).max(content.y);
    let options_height = options.len().min(
        options_bottom.saturating_sub(content.y + header_height + header_gap + options_gap + 1)
            as usize,
    ) as u16;
    let options_area = Rect::new(
        content.x,
        options_bottom - options_height,
        content.width,
        options_height,
    );
    let selected_end = option_ranges
        .get(state.selected)
        .map(|range| range.1)
        .unwrap_or(0);
    let options_start = selected_end.saturating_sub(options_height as usize);
    for &(start, end, index) in &option_ranges {
        let visible_start = start.max(options_start);
        let visible_end = end.min(options_start + options_height as usize);
        if visible_end > visible_start {
            let rect = Rect::new(
                content.x,
                options_area.y + (visible_start - options_start) as u16,
                content.width,
                (visible_end - visible_start) as u16,
            );
            if index == state.selected {
                frame.render_widget(Block::default().style(panel.bg(p.faint)), rect);
            }
            hits.choice_rows.push((rect, index));
        }
    }
    render_rows(
        frame,
        options_area,
        &options,
        options_start,
        &mut Vec::new(),
        &mut Vec::new(),
    );
    render_rows(
        frame,
        Rect::new(content.x, content.y, content.width, header_height),
        &header,
        0,
        &mut Vec::new(),
        &mut Vec::new(),
    );
    let details_y = (content.y + header_height + header_gap).min(options_area.y);
    let details_height = options_area.y.saturating_sub(details_y + options_gap);
    let viewport = render_detail(
        frame,
        state,
        Rect::new(content.x, details_y, content.width, details_height),
        details,
        p,
    );
    state.approval_page_rows = viewport.height.max(1);
}

fn overlay(frame: &mut Frame, state: &mut State, p: Palette, hits: &mut HitAreas) {
    let Some(kind) = state.overlay else {
        return;
    };
    if kind == Overlay::Approval {
        approval_overlay(frame, state, p, hits);
        return;
    }
    let screen = frame.area();
    let searchable = matches!(
        kind,
        Overlay::Commands
            | Overlay::Sessions
            | Overlay::Drafts
            | Overlay::Tools
            | Overlay::Search
            | Overlay::Queue
            | Overlay::Rewind
            | Overlay::Tree
            | Overlay::Files
            | Overlay::Attachments
            | Overlay::Models
            | Overlay::Modes
            | Overlay::Permissions
            | Overlay::Tasks
    ) || (kind == Overlay::Changes && !state.conversation_changes);
    let compact = matches!(
        kind,
        Overlay::Commands
            | Overlay::Sessions
            | Overlay::Drafts
            | Overlay::Search
            | Overlay::Queue
            | Overlay::Rewind
            | Overlay::Tree
            | Overlay::Files
            | Overlay::Models
            | Overlay::Modes
            | Overlay::Permissions
    );
    let choices = state.choices();
    state.selected = state.selected.min(choices.len().saturating_sub(1));
    let selected_image = state
        .selected_attachment()
        .is_some_and(crate::attachments::Attachment::is_image);
    let draft_warning = matches!(kind, Overlay::Queue | Overlay::Commands | Overlay::Rewind)
        && !state.editor.lines().join("\n").is_empty();
    let horizontal_margin = if screen.width >= 80 { 4 } else { 1 };
    let max_height = screen.height.saturating_sub(if screen.height >= 24 {
        4
    } else if screen.height >= 16 {
        2
    } else {
        0
    });
    let panel_width = screen
        .width
        .saturating_sub(horizontal_margin * 2)
        .min(if compact { 84 } else { 112 });
    let metadata_rows = if kind == Overlay::Rewind {
        2
    } else {
        u16::from(matches!(
            kind,
            Overlay::Search | Overlay::Queue | Overlay::Tree
        ))
    };
    let desired_height = if compact {
        5 + metadata_rows
            + u16::from(draft_warning)
            + if kind == Overlay::Tree {
                choices.len().clamp(1, 12) as u16
            } else {
                choices.len().clamp(1, 7) as u16 * 2
            }
    } else {
        max_height
    };
    let panel_height = desired_height.max(8).min(max_height);
    let area = Rect::new(
        screen.x + (screen.width - panel_width) / 2,
        screen.y + (screen.height - panel_height) / 2,
        panel_width,
        panel_height,
    );
    // Keep the conversation visible behind a focused, consistently colored panel.
    for cell in &mut frame.buffer_mut().content {
        let dimmed = match cell.fg {
            Color::Rgb(r, g, b) => Color::Rgb(r / 2, g / 2, b / 2),
            Color::Reset => Color::Reset,
            _ => p.muted,
        };
        cell.set_style(
            Style::default()
                .fg(dimmed)
                .bg(p.bg)
                .add_modifier(Modifier::DIM)
                .remove_modifier(Modifier::BOLD | Modifier::REVERSED),
        );
    }
    frame.render_widget(Clear, area);
    let panel = p.text().bg(p.panel);
    let muted = p.muted().bg(p.panel);
    let title = match kind {
        Overlay::Commands => " Commands ",
        Overlay::Sessions => " Conversations ",
        Overlay::Drafts => " Saved drafts · newest first ",
        Overlay::Tools => " Tool inspector ",
        Overlay::Changes => " Changes ",
        Overlay::Help => " Keyboard shortcuts ",
        Overlay::Diagnostics => " Diagnostics ",
        Overlay::Search => " Search conversation ",
        Overlay::Queue => " Message queue ",
        Overlay::Rewind => " Rewind conversation ",
        Overlay::Tree => " Conversation tree ",
        Overlay::Files => " Attach workspace files ",
        Overlay::Attachments => " Attached context ",
        Overlay::Models => " Session model ",
        Overlay::Modes => " Session mode ",
        Overlay::Permissions => " Execution permissions ",
        Overlay::Approval => " Review execution ",
        Overlay::Tasks => " Task progress ",
        Overlay::Rollback => " Choose rollback scope ",
        Overlay::RollbackConfirm => " Review file rollback ",
        Overlay::UndoConfirm => " Undo file rollback ",
    };
    let block = Block::default()
        .style(panel)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(muted.fg(p.faint))
        .title(Span::styled(title, panel.add_modifier(Modifier::BOLD)))
        .title_bottom(Line::styled(" Esc close ", muted).alignment(Alignment::Right));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let mut content = inset(inner, 1, 0);
    if content.width == 0 || content.height == 0 {
        return;
    }
    if kind == Overlay::Changes {
        let tab = if state.conversation_changes {
            "Git workspace  ·  [Conversation]  ·  Tab switch"
        } else {
            "[Git workspace]  ·  Conversation  ·  Tab switch"
        };
        line_at(
            frame,
            content,
            Line::styled(fit(tab, content.width as usize), panel),
        );
        content.y += 1;
        content.height = content.height.saturating_sub(1);
        if content.height == 0 {
            return;
        }
    }
    if searchable {
        hits.query = Some(Rect::new(content.x, content.y, content.width, 1));
        let query_width = content.width.saturating_sub(3) as usize;
        let mut query = state.query.clone();
        if width(&query) > query_width {
            let mut used = 1;
            query = state
                .query
                .chars()
                .rev()
                .take_while(|c| {
                    used += c.width().unwrap_or(0);
                    used <= query_width
                })
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            query.insert(0, '…');
        }
        let placeholder = match kind {
            Overlay::Search => "Find in messages and tool output…",
            Overlay::Sessions => "Find a conversation…",
            Overlay::Drafts => "Find saved input…",
            Overlay::Queue => "Filter queued messages…",
            Overlay::Rewind => "Find a message to edit…",
            Overlay::Tree if state.tree_search_supported => "Search complete messages…",
            Overlay::Tree => "Search previews only…",
            Overlay::Tools => "Find a tool call…",
            Overlay::Changes => "Filter files or staged / unstaged / untracked…",
            Overlay::Files => "Find a file by path…",
            Overlay::Attachments => "Filter attached files and images…",
            Overlay::Models => "Choose a configured model…",
            Overlay::Tasks => "Find a plan step or background job…",
            _ => "What would you like to do?",
        };
        line_at(
            frame,
            content,
            Line::from(vec![
                Span::styled("› ", p.accent().bg(p.panel)),
                Span::styled(
                    if query.is_empty() {
                        fit(placeholder, query_width)
                    } else {
                        query.clone()
                    },
                    if query.is_empty() { muted } else { panel },
                ),
            ])
            .style(panel),
        );
        if content.width > 2 {
            hits.cursor = Some(Position::new(
                content.x + 2 + width(&query).min(query_width) as u16,
                content.y,
            ));
        }
        if content.height > 1 {
            line_at(
                frame,
                Rect::new(content.x, content.y + 1, content.width, 1),
                Line::styled("─".repeat(content.width as usize), muted.fg(p.faint)),
            );
        }
        content.y += 2.min(content.height);
        content.height = content.height.saturating_sub(2);
    }
    if content.height == 0 {
        return;
    }
    let footer = Rect::new(content.x, content.bottom() - 1, content.width, 1);
    let footer_text = match kind {
        Overlay::Queue => "Enter edit · Del remove · Ctrl+S continue",
        Overlay::Search => "↑↓ choose · Enter jump",
        Overlay::Commands => "↑↓ choose · Enter run",
        Overlay::Sessions => "↑↓ choose · Enter resume",
        Overlay::Drafts => "↑↓ choose · Enter restore for review",
        Overlay::Rewind => "↑↓ choose · Enter: edit from here",
        Overlay::Tree => {
            let selected = choices
                .get(state.selected)
                .and_then(|choice| state.session_tree.node(&choice.value));
            if state.tree_search_pending() {
                "Searching messages…"
            } else if state.tree_search_error.is_some() {
                "Edit query to search again"
            } else if selected.is_none() {
                "No message selected"
            } else if selected.is_some_and(|node| !node.navigable) {
                "History only · no restore point"
            } else if selected.is_some_and(|node| node.role == crate::session_tree::TreeRole::User)
            {
                if content.width < 30 {
                    "↑↓ · Enter edit"
                } else {
                    "↑↓ choose · Enter edit from here"
                }
            } else if content.width < 30 {
                "↑↓ · Enter resume"
            } else {
                "↑↓ choose · Enter continue here"
            }
        }
        Overlay::Files => "↑↓ choose · Enter attach snapshot",
        Overlay::Attachments if selected_image && content.width < 30 => {
            "Enter original · Del remove"
        }
        Overlay::Attachments if selected_image => "Enter open original · Del remove · ↑↓ choose",
        Overlay::Attachments => "↑↓ choose · Del remove attachment",
        Overlay::Models | Overlay::Modes | Overlay::Permissions => {
            "↑↓ choose · Enter apply to this session"
        }
        Overlay::Approval => "↑↓ choose · Enter confirm · Esc reject · PgDn code",
        Overlay::Tasks => "↑↓ select · Ctrl+X stop job · Ctrl+R refresh",
        Overlay::Rollback => "↑↓ scope · Enter preview · Esc cancel",
        Overlay::RollbackConfirm | Overlay::UndoConfirm => {
            "↑↓ choose · Enter confirm · PgDn preview · Esc cancel"
        }
        Overlay::Tools => "↑↓ choose · PgUp/PgDn scroll",
        Overlay::Changes if !state.conversation_changes => {
            "↑↓ file · Ctrl+U rollback · Ctrl+R refresh · PgDn scroll"
        }
        _ => "PgUp/PgDn scroll",
    };
    let count = if searchable
        && !choices.is_empty()
        && (!matches!(kind, Overlay::Queue | Overlay::Rewind | Overlay::Tree)
            || content.width >= 64)
    {
        format!("{} / {}", state.selected + 1, choices.len())
    } else {
        String::new()
    };
    pair(frame, footer, footer_text, &count, muted, muted);
    if selected_image {
        let action_width = if content.width < 30 {
            width("Enter original")
        } else {
            width("Enter open original")
        };
        hits.attachment_original.push(Rect::new(
            footer.x,
            footer.y,
            (action_width as u16).min(footer.width),
            1,
        ));
    }
    content.height = content.height.saturating_sub(1);
    if content.height == 0 {
        return;
    }
    if kind == Overlay::Attachments {
        if let Some((message, error)) = &state.original_image_notice {
            let mut rows = Vec::new();
            append(
                &mut rows,
                Line::styled(clean(message), if *error { panel.fg(p.red) } else { muted }),
                content.width as usize,
                None,
            );
            let height = rows.len().min(content.height.saturating_sub(1) as usize) as u16;
            render_rows(
                frame,
                Rect::new(content.x, content.y, content.width, height),
                &rows,
                0,
                &mut Vec::new(),
                &mut Vec::new(),
            );
            content.y += height;
            content.height -= height;
        }
    }
    match kind {
        Overlay::Rollback | Overlay::RollbackConfirm | Overlay::UndoConfirm => {
            let (path, summary, preview) = match kind {
                Overlay::Rollback => (
                    state
                        .rollback
                        .file
                        .as_ref()
                        .map(|file| clean(&file.path))
                        .unwrap_or_default(),
                    "Choose all hunks or one hunk, then review the proposed changes.".to_owned(),
                    state
                        .rollback
                        .file
                        .as_ref()
                        .map(|file| clean(&file.patch))
                        .unwrap_or_default(),
                ),
                Overlay::RollbackConfirm => (
                    state
                        .rollback
                        .plan
                        .as_ref()
                        .map(|plan| clean(&plan.path))
                        .unwrap_or_default(),
                    state
                        .rollback
                        .plan
                        .as_ref()
                        .map(|plan| {
                            format!(
                                "{} · backup saved before writing · /undo restores it",
                                plan.scope
                            )
                        })
                        .unwrap_or_else(|| "Checking the selected file and Git index…".into()),
                    state
                        .rollback
                        .plan
                        .as_ref()
                        .map(|plan| clean(&plan.preview))
                        .unwrap_or_default(),
                ),
                _ => (
                    state
                        .rollback
                        .receipt
                        .as_ref()
                        .map(|receipt| clean(&receipt.path))
                        .unwrap_or_default(),
                    state
                        .rollback
                        .receipt
                        .as_ref()
                        .map(|receipt| {
                            format!(
                                "{} · restore saved content · later edits block undo",
                                clean(&receipt.scope)
                            )
                        })
                        .unwrap_or_else(|| "Reading the last rollback backup…".into()),
                    state
                        .rollback
                        .receipt
                        .as_ref()
                        .map(|receipt| {
                            format!(
                                "{}Saved backup:\n{}\n\n{}",
                                receipt
                                    .warning
                                    .as_ref()
                                    .map(|warning| format!("{}\n\n", clean(warning)))
                                    .unwrap_or_default(),
                                clean(&receipt.backup_path.to_string_lossy()),
                                clean(&receipt.preview)
                            )
                        })
                        .unwrap_or_default(),
                ),
            };
            for line in [path, summary] {
                if content.height == 0 {
                    break;
                }
                line_at(
                    frame,
                    content,
                    Line::styled(fit(&line, content.width as usize), panel),
                );
                content.y += 1;
                content.height = content.height.saturating_sub(1);
            }
            if state.rollback.loading {
                if content.height > 0 {
                    line_at(frame, content, Line::styled("Working…", panel.fg(p.amber)));
                }
                content.y += u16::from(content.height > 0);
                content.height = content.height.saturating_sub(1);
            } else {
                let visible = choices
                    .len()
                    .min(5)
                    .min(content.height.saturating_sub(3).max(1) as usize);
                let start = state.selected.saturating_sub(visible.saturating_sub(1));
                for (index, choice) in choices.iter().enumerate().skip(start).take(visible) {
                    if content.height == 0 {
                        break;
                    }
                    let style = if index == state.selected {
                        panel.fg(p.accent).bg(p.faint).add_modifier(Modifier::BOLD)
                    } else {
                        panel
                    };
                    let rect = Rect::new(content.x, content.y, content.width, 1);
                    frame.render_widget(Block::default().style(style), rect);
                    line_at(
                        frame,
                        rect,
                        Line::styled(
                            fit(
                                &format!(
                                    "{} {}",
                                    if index == state.selected { "›" } else { " " },
                                    choice.label
                                ),
                                content.width as usize,
                            ),
                            style,
                        ),
                    );
                    hits.choice_rows.push((rect, index));
                    content.y += 1;
                    content.height -= 1;
                }
            }
            if content.height > 1 {
                content.y += 1;
                content.height -= 1;
                let mut rows = Vec::new();
                for line in diff_lines(&preview, p) {
                    append(&mut rows, line, content.width as usize, None);
                }
                render_detail(frame, state, content, rows, p);
            }
        }
        Overlay::Commands
        | Overlay::Sessions
        | Overlay::Drafts
        | Overlay::Tools
        | Overlay::Search
        | Overlay::Queue
        | Overlay::Rewind
        | Overlay::Tree
        | Overlay::Files
        | Overlay::Attachments
        | Overlay::Models
        | Overlay::Modes
        | Overlay::Permissions
        | Overlay::Tasks
        | Overlay::Changes
            if searchable =>
        {
            if kind == Overlay::Changes && content.height >= 8 {
                let label = if state.changes_loading {
                    "Reading Git workspace…".to_owned()
                } else {
                    format!(
                        "{} files · {}",
                        choices.len(),
                        state
                            .workspace_changes
                            .as_ref()
                            .map(|s| clean(&s.root.to_string_lossy()))
                            .unwrap_or_default()
                    )
                };
                line_at(
                    frame,
                    content,
                    Line::styled(fit(&label, content.width as usize), muted),
                );
                content.y += 1;
                content.height = content.height.saturating_sub(1);
                if content.height == 0 {
                    return;
                }
            }
            if kind == Overlay::Rewind {
                for label in [
                    "Files unchanged · Conversation only",
                    "Original conversation kept",
                ] {
                    if content.height == 0 {
                        return;
                    }
                    line_at(
                        frame,
                        content,
                        Line::styled(fit(label, content.width as usize), muted),
                    );
                    content.y += 1;
                    content.height = content.height.saturating_sub(1);
                }
            }
            if kind == Overlay::Tree && content.height > 1 {
                line_at(
                    frame,
                    content,
                    Line::styled(
                        fit(
                            if state.tree_search_supported {
                                "● current · full-text search · branches kept"
                            } else {
                                "● current · search previews only · branches kept"
                            },
                            content.width as usize,
                        ),
                        muted,
                    ),
                );
                content.y += 1;
                content.height -= 1;
            }
            if matches!(kind, Overlay::Search | Overlay::Queue) {
                let label = if kind == Overlay::Search {
                    format!("{} matching messages", choices.len())
                } else {
                    format!(
                        "{} queued · {}",
                        choices.len(),
                        if state.queue_paused {
                            "paused"
                        } else {
                            "sending in order"
                        }
                    )
                };
                line_at(
                    frame,
                    content,
                    Line::styled(fit(&label, content.width as usize), muted),
                );
                content.y += 1;
                content.height = content.height.saturating_sub(1);
            }
            if draft_warning && content.height > 0 {
                let warning = if kind == Overlay::Commands {
                    "Clear the draft to insert a command with arguments."
                } else {
                    "Clear the draft first to edit a message."
                };
                line_at(
                    frame,
                    content,
                    Line::styled(fit(warning, content.width as usize), panel.fg(p.amber)),
                );
                content.y += 1;
                content.height = content.height.saturating_sub(1);
            }
            if content.height == 0 {
                return;
            }
            if choices.is_empty() {
                let text = if kind == Overlay::Changes {
                    if state.changes_loading {
                        "Reading Git workspace…"
                    } else if let Some(error) = state.changes_error.as_deref() {
                        error
                    } else if !state.query.is_empty() {
                        "No matching files."
                    } else if let Some(notice) = state
                        .workspace_changes
                        .as_ref()
                        .and_then(|s| s.notice.as_deref())
                    {
                        notice
                    } else if state.workspace_changes.is_none() {
                        "Press Ctrl+R to read Git workspace changes."
                    } else {
                        "Working tree clean. No staged, unstaged or untracked files."
                    }
                } else if kind == Overlay::Queue && state.query.is_empty() {
                    "No queued messages. Send a message while the agent works to add one."
                } else if kind == Overlay::Drafts && state.query.is_empty() {
                    "No saved drafts to recover. New input is saved as you type."
                } else if kind == Overlay::Rewind && state.rewind_loading {
                    "Loading rewind points…"
                } else if kind == Overlay::Tree && state.tree_loading {
                    "Loading conversation tree…"
                } else if kind == Overlay::Tree && state.tree_search_error.is_some() {
                    state.tree_search_error.as_deref().unwrap_or_default()
                } else if kind == Overlay::Tree && state.tree_search_pending() {
                    "Searching complete messages…"
                } else if kind == Overlay::Tree && state.query.is_empty() {
                    "No messages in this conversation tree yet. Send a message to begin."
                } else if kind == Overlay::Rewind && state.query.is_empty() {
                    "No saved rewind points. Older messages without a checkpoint cannot be restored; new messages will appear here."
                } else if matches!(
                    kind,
                    Overlay::Models | Overlay::Modes | Overlay::Permissions
                ) {
                    "Waiting for the session's available settings…"
                } else if kind == Overlay::Tasks {
                    "No plan steps or background jobs have been reported yet."
                } else if kind == Overlay::Files {
                    if state.files_loading {
                        "Searching workspace files…"
                    } else if !state.files_notice.is_empty() {
                        &state.files_notice
                    } else {
                        "No matching files. Use /attach with a full path to attach a file directly."
                    }
                } else if kind == Overlay::Attachments {
                    "No attachments. Type @ or use Alt+A to select a file; Ctrl+V pastes an image."
                } else if kind == Overlay::Search && state.query.is_empty() {
                    "Search this conversation. Your draft will stay intact."
                } else if state.query.is_empty() {
                    "Nothing here yet."
                } else {
                    "No matches. Try another search."
                };
                let mut rows = Vec::new();
                append(
                    &mut rows,
                    Line::styled(text.to_owned(), muted),
                    content.width as usize,
                    None,
                );
                render_rows(frame, content, &rows, 0, &mut Vec::new(), &mut Vec::new());
                return;
            }
            let tools = matches!(
                kind,
                Overlay::Tools | Overlay::Changes | Overlay::Tasks | Overlay::Attachments
            );
            if kind == Overlay::Changes && content.height < 4 {
                line_at(
                    frame,
                    content,
                    Line::styled(
                        fit(
                            "Increase window height to view changes.",
                            content.width as usize,
                        ),
                        muted,
                    ),
                );
                return;
            }
            let side_by_side = tools && content.width >= 82;
            let list_area = if side_by_side {
                Rect::new(content.x, content.y, 30, content.height)
            } else if tools {
                Rect::new(
                    content.x,
                    content.y,
                    content.width,
                    if kind == Overlay::Changes {
                        content
                            .height
                            .saturating_sub(5)
                            .clamp(1, 3)
                            .min(choices.len() as u16)
                    } else {
                        content.height.min(4)
                    },
                )
            } else {
                content
            };
            let stride = if tools || kind == Overlay::Tree { 1 } else { 2 };
            let visible = (list_area.height as usize / stride).max(1);
            let start = state.selected.saturating_sub(visible.saturating_sub(1));
            for (index, choice) in choices.iter().enumerate().skip(start).take(visible) {
                let y = list_area.y + ((index - start) * stride) as u16;
                let selected = index == state.selected;
                let background = if selected { p.faint } else { p.panel };
                let style = if selected {
                    panel
                        .fg(p.accent)
                        .bg(background)
                        .add_modifier(Modifier::BOLD)
                } else {
                    panel
                };
                let item_rect = Rect::new(
                    list_area.x,
                    y,
                    list_area.width,
                    (stride as u16).min(list_area.bottom().saturating_sub(y)),
                );
                hits.choice_rows.push((item_rect, index));
                frame.render_widget(Block::default().style(panel.bg(background)), item_rect);
                let label = if kind == Overlay::Tree && list_area.width < 30 {
                    state
                        .session_tree
                        .compact_label(&choice.value)
                        .unwrap_or_else(|| choice.label.clone())
                } else {
                    choice.label.clone()
                };
                line_at(
                    frame,
                    item_rect,
                    Line::styled(
                        fit(
                            &format!(
                                "{} {}{}",
                                if selected { "›" } else { " " },
                                if kind == Overlay::Changes {
                                    format!(
                                        "[{}] ",
                                        choice.detail.split(" · ").next().unwrap_or("")
                                    )
                                } else {
                                    String::new()
                                },
                                label
                            ),
                            list_area.width as usize,
                        ),
                        style,
                    ),
                );
                if stride == 2 && item_rect.height > 1 {
                    let columns = list_area.width.saturating_sub(2) as usize;
                    let detail = if kind == Overlay::Search {
                        search_preview(&choice.detail, state.query.trim(), columns)
                    } else {
                        fit(&choice.detail, columns)
                    };
                    let mut line = Line::styled(detail, muted.bg(background));
                    if kind == Overlay::Search {
                        highlight(&mut line, state.query.trim(), p);
                    }
                    line_at(
                        frame,
                        Rect::new(list_area.x + 2, y + 1, list_area.width.saturating_sub(2), 1),
                        line,
                    );
                }
            }
            if tools {
                let detail = if side_by_side {
                    frame.render_widget(
                        Block::default()
                            .borders(Borders::LEFT)
                            .border_style(muted.fg(p.faint)),
                        Rect::new(content.x + 31, content.y, 1, content.height),
                    );
                    Rect::new(
                        content.x + 33,
                        content.y,
                        content.width.saturating_sub(33),
                        content.height,
                    )
                } else {
                    Rect::new(
                        content.x,
                        content.y + list_area.height + 1,
                        content.width,
                        content.height.saturating_sub(list_area.height + 1),
                    )
                };
                let mut rows = Vec::new();
                let mut original_rows = Vec::new();
                let mut native_image = None;
                if kind == Overlay::Changes {
                    if let Some(file) = state.selected_change() {
                        if side_by_side {
                            append(
                                &mut rows,
                                Line::styled(clean(&file.path), panel.add_modifier(Modifier::BOLD)),
                                detail.width as usize,
                                None,
                            );
                            append(
                                &mut rows,
                                Line::styled(
                                    format!("{} · {}", file.stage.label(), clean(&file.status)),
                                    muted,
                                ),
                                detail.width as usize,
                                None,
                            );
                            if let Some(original) = &file.original_path {
                                append(
                                    &mut rows,
                                    Line::styled(format!("From {}", clean(original)), muted),
                                    detail.width as usize,
                                    None,
                                );
                            }
                            append(
                                &mut rows,
                                Line::styled("", panel),
                                detail.width as usize,
                                None,
                            );
                        }
                        if let Some(notice) = state
                            .workspace_changes
                            .as_ref()
                            .and_then(|s| s.notice.as_ref())
                        {
                            append(
                                &mut rows,
                                Line::styled(clean(notice), panel.fg(p.amber)),
                                detail.width as usize,
                                None,
                            );
                        }
                        for line in diff_lines(&clean(&file.patch), p) {
                            append(&mut rows, line, detail.width as usize, None);
                        }
                    }
                } else if kind == Overlay::Attachments {
                    if let Some(attachment) = choices
                        .get(state.selected)
                        .and_then(|choice| choice.value.parse::<usize>().ok())
                        .and_then(|index| state.attachments.get(index))
                    {
                        let native_pixels = (!state.monochrome)
                            .then(|| state.terminal_graphics.sixel_cell_pixels())
                            .flatten();
                        let native_png =
                            native_pixels.is_some() && attachment.mime_type() == "image/png";
                        let name = clean(attachment.name());
                        append(
                            &mut rows,
                            Line::styled(
                                if native_png {
                                    fit(&name, detail.width as usize)
                                } else {
                                    name
                                },
                                panel.add_modifier(Modifier::BOLD),
                            ),
                            detail.width as usize,
                            None,
                        );
                        append(
                            &mut rows,
                            Line::styled(
                                format!(
                                    "{} · {} bytes · saved snapshot",
                                    attachment.mime_type(),
                                    attachment.byte_len()
                                ),
                                muted,
                            ),
                            detail.width as usize,
                            None,
                        );
                        match attachment {
                            crate::attachments::Attachment::Text { text, .. } => {
                                for line in clean(text).lines() {
                                    append(
                                        &mut rows,
                                        Line::styled(line.replace('\t', "    "), panel),
                                        detail.width as usize,
                                        None,
                                    );
                                }
                            }
                            crate::attachments::Attachment::Image { .. } => {
                                let action_start = rows.len();
                                append(
                                    &mut rows,
                                    Line::styled(
                                        "Open original · Enter",
                                        panel.fg(p.accent).add_modifier(Modifier::UNDERLINED),
                                    ),
                                    detail.width as usize,
                                    None,
                                );
                                original_rows.extend(
                                    (action_start..rows.len())
                                        .map(|index| (index, rows[index].line.width() as u16)),
                                );
                                if !state.monochrome {
                                    let background = match p.panel {
                                        Color::Rgb(r, g, b) => [r, g, b],
                                        _ => [32, 32, 32],
                                    };
                                    let available =
                                        (detail.height as usize).saturating_sub(rows.len());
                                    let preview = if let Some(cell_pixels) = native_pixels {
                                        state.terminal_image.get_or_request(
                                            attachment,
                                            detail.width,
                                            available as u16,
                                            cell_pixels,
                                            background,
                                        )
                                    } else {
                                        Ok(PreviewState::Unsupported)
                                    };
                                    match preview {
                                        Ok(PreviewState::Ready(image)) => {
                                            native_image =
                                                Some((rows.len(), image.columns, image.rows));
                                            for _ in 0..image.rows {
                                                append(
                                                    &mut rows,
                                                    Line::styled("", panel),
                                                    detail.width as usize,
                                                    None,
                                                );
                                            }
                                        }
                                        Ok(PreviewState::Pending) => {
                                            append(
                                                &mut rows,
                                                Line::styled("Rendering image…", muted),
                                                detail.width as usize,
                                                None,
                                            );
                                            while rows.len() < detail.height as usize {
                                                append(
                                                    &mut rows,
                                                    Line::styled("", panel),
                                                    detail.width as usize,
                                                    None,
                                                );
                                            }
                                        }
                                        Ok(PreviewState::Unsupported) => {
                                            state.terminal_image.hide();
                                            match state.image_preview.preview(
                                                attachment,
                                                detail.width as usize,
                                                available,
                                                background,
                                            ) {
                                                Ok(Some(thumbnail)) => {
                                                    for y in (0..thumbnail.height).step_by(2) {
                                                        let mut pixels =
                                                            Vec::with_capacity(thumbnail.width);
                                                        for x in 0..thumbnail.width {
                                                            let [r, g, b] = thumbnail.pixels
                                                                [y * thumbnail.width + x];
                                                            let [br, bg, bb] = if y + 1
                                                                < thumbnail.height
                                                            {
                                                                thumbnail.pixels
                                                                    [(y + 1) * thumbnail.width + x]
                                                            } else {
                                                                background
                                                            };
                                                            pixels.push(Span::styled(
                                                                "▀",
                                                                Style::default()
                                                                    .fg(Color::Rgb(r, g, b))
                                                                    .bg(Color::Rgb(br, bg, bb)),
                                                            ));
                                                        }
                                                        append(
                                                            &mut rows,
                                                            Line::from(pixels),
                                                            detail.width as usize,
                                                            None,
                                                        );
                                                        original_rows.push((
                                                            rows.len() - 1,
                                                            thumbnail.width as u16,
                                                        ));
                                                    }
                                                }
                                                Ok(None) => {}
                                                Err(error) => append(
                                                    &mut rows,
                                                    Line::styled(error, panel.fg(p.amber)),
                                                    detail.width as usize,
                                                    None,
                                                ),
                                            }
                                        }
                                        Err(error) => append(
                                            &mut rows,
                                            Line::styled(error, panel.fg(p.amber)),
                                            detail.width as usize,
                                            None,
                                        ),
                                    }
                                }
                            }
                        }
                    }
                } else if kind == Overlay::Tasks {
                    if let Some(choice) = choices.get(state.selected) {
                        let (collection, id) = choice.value.split_once(':').unwrap_or(("", ""));
                        let field = if collection == "job" { "jobs" } else { "steps" };
                        if let Some(task) = state.tasks[field]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .find(|task| task["id"] == id)
                        {
                            for text in [
                                choice.label.clone(),
                                format!(
                                    "{} · ID: {id}",
                                    task["status"].as_str().unwrap_or("unknown")
                                ),
                                clean(task["description"].as_str().unwrap_or("")),
                                clean(state.tasks["waitReason"].as_str().unwrap_or("")),
                            ] {
                                append(
                                    &mut rows,
                                    Line::styled(text, panel),
                                    detail.width as usize,
                                    None,
                                );
                            }
                            if let Some(blocked) = task["blockedBy"]
                                .as_array()
                                .filter(|items| !items.is_empty())
                            {
                                append(
                                    &mut rows,
                                    Line::styled(
                                        format!(
                                            "Waiting on: {}",
                                            blocked
                                                .iter()
                                                .filter_map(serde_json::Value::as_str)
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        ),
                                        panel.fg(p.amber),
                                    ),
                                    detail.width as usize,
                                    None,
                                );
                            }
                            for comment in task["comments"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(serde_json::Value::as_str)
                            {
                                append(
                                    &mut rows,
                                    Line::styled(clean(comment), muted),
                                    detail.width as usize,
                                    None,
                                );
                            }
                            let output =
                                task["output"]
                                    .as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| {
                                        if task["output"].is_null() {
                                            String::new()
                                        } else {
                                            task["output"].to_string()
                                        }
                                    });
                            for line in clean(&output).lines() {
                                append(
                                    &mut rows,
                                    Line::styled(line.to_owned(), panel),
                                    detail.width as usize,
                                    None,
                                );
                            }
                        }
                    }
                } else if let Some(tool) = state.selected_tool() {
                    append(
                        &mut rows,
                        Line::styled(tool.title.clone(), panel.add_modifier(Modifier::BOLD)),
                        detail.width as usize,
                        None,
                    );
                    append(
                        &mut rows,
                        Line::styled(
                            format!(
                                "{} {}{}",
                                tool_icon(tool),
                                tool.status,
                                tool.elapsed_seconds()
                                    .map(|seconds| format!(" · {}", elapsed(seconds)))
                                    .unwrap_or_default()
                            ),
                            panel.fg(tool_color(tool, p)),
                        ),
                        detail.width as usize,
                        None,
                    );
                    append(
                        &mut rows,
                        Line::styled("", panel),
                        detail.width as usize,
                        None,
                    );
                    for line in tool.content.lines() {
                        append(
                            &mut rows,
                            Line::styled(line.replace('\t', "    "), panel),
                            detail.width as usize,
                            None,
                        );
                    }
                    for line in diff_lines(&tool.diff, p) {
                        append(&mut rows, line, detail.width as usize, None);
                    }
                }
                let viewport = render_detail(frame, state, detail, rows, p);
                let start = state.detail_scroll as usize;
                if let Some((first_row, columns, image_rows)) = native_image {
                    // Never draw a partial Sixel across the filter, list, footer,
                    // or a detail-scroll indicator. The fitted image normally
                    // occupies a fixed, unscrolled area below its metadata.
                    if first_row >= start
                        && first_row - start + usize::from(image_rows)
                            <= usize::from(viewport.height)
                        && columns <= viewport.width
                    {
                        let area = Rect::new(
                            viewport.x,
                            viewport.y + (first_row - start) as u16,
                            columns,
                            image_rows,
                        );
                        if let Some(image) = state.terminal_image.ready() {
                            image.render(area, frame.buffer_mut());
                            hits.attachment_original.push(area);
                        }
                    }
                }
                for (index, width) in original_rows {
                    if index >= start
                        && index - start < viewport.height as usize
                        && viewport.width > 0
                    {
                        hits.attachment_original.push(Rect::new(
                            viewport.x,
                            viewport.y + (index - start) as u16,
                            width.min(viewport.width),
                            1,
                        ));
                    }
                }
            }
        }
        Overlay::Changes => {
            let mut rows = Vec::new();
            for entry in &state.entries {
                if let Entry::Tool(tool) = entry {
                    if tool.diff.is_empty() {
                        continue;
                    }
                    append(
                        &mut rows,
                        Line::styled(tool.title.clone(), panel.add_modifier(Modifier::BOLD)),
                        content.width as usize,
                        None,
                    );
                    for line in diff_lines(&tool.diff, p) {
                        append(&mut rows, line, content.width as usize, None);
                    }
                    append(
                        &mut rows,
                        Line::styled("", panel),
                        content.width as usize,
                        None,
                    );
                }
            }
            if rows.is_empty() {
                append(
                    &mut rows,
                    Line::styled("No file changes reported in this conversation.", muted),
                    content.width as usize,
                    None,
                );
            }
            render_detail(frame, state, content, rows, p);
        }
        Overlay::Diagnostics => {
            let mut rows = Vec::new();
            for message in &state.diagnostics {
                for line in message.lines() {
                    append(
                        &mut rows,
                        Line::styled(line.to_owned(), muted),
                        content.width as usize,
                        None,
                    );
                }
            }
            if rows.is_empty() {
                append(
                    &mut rows,
                    Line::styled("No backend diagnostics.", muted),
                    content.width as usize,
                    None,
                );
            }
            render_detail(frame, state, content, rows, p);
        }
        Overlay::Help => {
            let help = [
                ("WRITING", ""),
                ("Enter", "Send / queue while working"),
                (
                    "Alt+S / /steer",
                    "Correct the running task at its next step",
                ),
                ("@ / Alt+A", "Find and attach a file snapshot"),
                ("/attachments", "Review or remove attached context"),
                ("Click attachment", "Preview the selected file or image"),
                (
                    "Enter (image panel)",
                    "Open the original image in your viewer",
                ),
                ("Click ×", "Remove that attachment"),
                ("Alt+Backspace", "Remove the last attachment"),
                (
                    "Backspace (empty)",
                    "Remove the last attachment with an empty draft",
                ),
                ("/reconnect", "Reconnect; recovered work stays paused"),
                ("/retry", "Restore interrupted input for review"),
                ("/drafts", "Choose saved input to restore"),
                ("/model /mode", "Choose the session model and work mode"),
                ("/permissions", "Set approval for generated execution"),
                ("Alt+T /tasks", "Inspect plan steps and background jobs"),
                ("Alt+Enter", "Insert a new line"),
                ("Ctrl+V / right click", "Paste clipboard text or an image"),
                ("Click / drag", "Place the cursor / select draft text"),
                ("Ctrl+C / Ctrl+X", "Copy / cut selection with no panel open"),
                ("Ctrl+Z", "Undo text or the last attachment removal"),
                ("↑ / ↓ at edge", "Recall input / return to draft"),
                ("Esc", "Stop a task / close an open panel"),
                ("Esc while preparing", "Recall a waiting message"),
                ("Esc twice (draft)", "Clear draft; Ctrl+Z / Ctrl+U undo"),
                ("Esc twice (empty)", "Choose a message to edit and rewind"),
                ("/rewind", "Restore conversation; keep files unchanged"),
                ("/tree", "Browse branches and continue from a message"),
                (
                    "Ctrl+C twice",
                    "Exit within 2s (panel or no draft selection)",
                ),
                ("Ctrl+D", "Exit with an empty draft"),
                ("", ""),
                ("WORKSPACE", ""),
                ("/", "Suggest commands above the draft"),
                ("Tab", "Complete the selected command"),
                ("Ctrl+K", "Open command palette"),
                ("Ctrl+R", "Find saved conversations"),
                ("Ctrl+F", "Search this conversation"),
                ("Ctrl+E", "Manage queued messages"),
                ("Ctrl+S", "Continue a paused queue"),
                ("Ctrl+O", "Expand / collapse all details"),
                ("Alt+B", "Toggle the sidebar"),
                ("Ctrl+T", "Browse tool output"),
                ("Ctrl+G", "Inspect file changes"),
                ("Ctrl+P / Ctrl+N", "Previous / next input"),
                ("Ctrl+End", "Jump to newest message"),
                ("Ctrl+Q", "Save and exit"),
                ("PgUp / PgDn", "Scroll conversation or inspector"),
                ("Mouse wheel", "Scroll conversation"),
                ("Click a tool", "Expand / collapse its output"),
                ("", ""),
                ("PANELS", ""),
                ("↑ / ↓", "Choose an item"),
                ("Enter", "Run command / resume session"),
                ("Esc", "Close panel and keep draft"),
                ("Ctrl+X / F1", "Help from composer (Ctrl+X: no selection)"),
            ];
            let mut rows = Vec::new();
            for (key, explanation) in help {
                append(
                    &mut rows,
                    Line::from(vec![
                        Span::styled(
                            format!("{key:18}"),
                            if explanation.is_empty() {
                                muted
                            } else {
                                panel.fg(p.accent)
                            },
                        ),
                        Span::styled(explanation, panel),
                    ])
                    .style(panel),
                    content.width as usize,
                    None,
                );
            }
            render_detail(frame, state, content, rows, p);
        }
        _ => {}
    }
}

fn render_detail(
    frame: &mut Frame,
    state: &mut State,
    area: Rect,
    rows: Vec<Row>,
    p: Palette,
) -> Rect {
    if area.width == 0 || area.height == 0 {
        return Rect::new(area.x, area.y, 0, 0);
    }
    let has_scroll = rows.len() > area.height as usize;
    let visible_height = area
        .height
        .saturating_sub(u16::from(has_scroll && area.height > 1));
    let max = rows.len().saturating_sub(visible_height as usize);
    state.detail_scroll = (state.detail_scroll as usize)
        .min(max)
        .min(u16::MAX as usize) as u16;
    let viewport = Rect::new(area.x, area.y, area.width, visible_height);
    render_rows(
        frame,
        viewport,
        &rows,
        state.detail_scroll as usize,
        &mut Vec::new(),
        &mut Vec::new(),
    );
    if has_scroll && area.height > 1 {
        let label = format!(
            "{}–{} of {} lines",
            state.detail_scroll as usize + 1,
            (state.detail_scroll as usize + visible_height as usize).min(rows.len()),
            rows.len()
        );
        line_at(
            frame,
            Rect::new(area.x, area.bottom() - 1, area.width, 1),
            Line::styled(fit(&label, area.width as usize), p.muted().bg(p.panel)),
        );
    }
    viewport
}

fn shortcuts(frame: &mut Frame, area: Rect, items: &[(&str, &str)], p: Palette) {
    let mut spans = Vec::new();
    let mut used = 0;
    for (key, action) in items {
        let separator = if spans.is_empty() { "" } else { "  │  " };
        let needed = width(separator) + width(key) + width(action) + 1;
        if used + needed > area.width as usize {
            break;
        }
        spans.push(Span::styled(separator, p.muted()));
        spans.push(Span::styled(*key, p.bold().fg(p.accent)));
        spans.push(Span::styled(format!(":{action}"), p.muted()));
        used += needed;
    }
    line_at(frame, area, Line::from(spans));
}

fn attachment_name(name: &str, cells: usize) -> String {
    let name = clean(name).replace(['\r', '\n', '\t'], " ");
    if name.width() <= cells {
        return name;
    }
    let mut result = String::new();
    let mut used = 0;
    for grapheme in name.graphemes(true) {
        let next = grapheme.width();
        if used + next >= cells {
            break;
        }
        result.push_str(grapheme);
        used += next;
    }
    result.push('…');
    result
}

fn attachment_row(
    frame: &mut Frame,
    state: &State,
    mut area: Rect,
    p: Palette,
    hits: &mut HitAreas,
) {
    if area.is_empty() {
        return;
    }
    let can_undo = state.can_undo_attachment();
    if can_undo {
        let label = if area.width >= 4 { "Undo" } else { "↶" };
        let undo_width = label.width() as u16;
        let undo = Rect::new(area.right() - undo_width, area.y, undo_width, 1);
        line_at(
            frame,
            undo,
            Line::styled(label, p.bold().add_modifier(Modifier::UNDERLINED)),
        );
        if state.overlay.is_none() {
            hits.attachment_undo = Some(undo);
        }
        area.width = area
            .width
            .saturating_sub(undo_width + u16::from(area.width >= undo_width + 2));
        if area.is_empty() {
            return;
        }
    }
    if state.attachments.is_empty() {
        let label = if state.attachment_loading {
            "Reading attachment…"
        } else if state.recovery.is_some() {
            "Interrupted input saved · /retry to review"
        } else if can_undo {
            "Attachment removed"
        } else {
            ""
        };
        line_at(
            frame,
            area,
            Line::styled(fit(label, area.width as usize), p.accent()),
        );
        return;
    }
    let hints: &[&str] = match (state.attachment_loading, state.recovery.is_some()) {
        (true, true) => &["Reading attachment… · /retry", "Reading… · /retry"],
        (true, false) => &["Reading attachment…", "Reading…"],
        (false, true) => &[
            "Click preview · Alt+Backspace remove last · /retry",
            "/retry",
        ],
        (false, false) => &[
            "Click preview · Alt+Backspace remove last",
            "Click preview · × remove",
        ],
    };
    let chip_minimum = if state.attachment_loading || state.recovery.is_some() {
        8
    } else {
        14
    };
    let hint = hints
        .iter()
        .copied()
        .find(|hint| area.width as usize >= hint.width() + chip_minimum + 2)
        .unwrap_or("");
    let hint_width = hint.width() as u16;
    let budget = area
        .width
        .saturating_sub(hint_width + u16::from(hint_width > 0) * 2);
    if budget == 0 {
        return;
    }
    let last = state.attachments.len() - 1;
    if budget < 5 {
        line_at(frame, area, Line::styled("×", p.bold()));
        if state.overlay.is_none() {
            hits.attachment_remove
                .push((Rect::new(area.x, area.y, 1, 1), last));
        }
        return;
    }
    let prefix_width = format!("+{last} ").width();
    let show_hidden = last > 0 && budget as usize >= prefix_width + 5;
    let name_width = (budget as usize - 4 - if show_hidden { prefix_width } else { 0 }).min(24);
    let names: Vec<_> = state
        .attachments
        .iter()
        .map(|item| attachment_name(item.name(), name_width))
        .collect();
    let mut first = last;
    let mut used = names[last].width() + 4;
    for index in (0..last).rev() {
        let needed = used + names[index].width() + 5;
        let prefix = if index > 0 && show_hidden {
            format!("+{index} ").width()
        } else {
            0
        };
        if needed + prefix > budget as usize {
            break;
        }
        first = index;
        used = needed;
    }
    let mut x = area.x;
    if first > 0 && show_hidden {
        let prefix = format!("+{first} ");
        line_at(
            frame,
            Rect::new(x, area.y, prefix.width() as u16, 1),
            Line::styled(prefix.clone(), p.muted()),
        );
        if state.overlay.is_none() {
            hits.attachment_overflow = Some(Rect::new(x, area.y, prefix.width() as u16 - 1, 1));
        }
        x += prefix.width() as u16;
    }
    for (index, name) in names.iter().enumerate().skip(first) {
        let name_width = name.width() as u16;
        let style = p.accent().bg(p.panel);
        line_at(
            frame,
            Rect::new(x, area.y, name_width + 4, 1),
            Line::from(vec![
                Span::styled(format!("[{name} "), style),
                Span::styled("×", style.add_modifier(Modifier::BOLD)),
                Span::styled("]", style),
            ]),
        );
        if state.overlay.is_none() {
            if name_width > 0 {
                hits.attachment_preview
                    .push((Rect::new(x + 1, area.y, name_width, 1), index));
            }
            hits.attachment_remove
                .push((Rect::new(x + name_width + 2, area.y, 1, 1), index));
        }
        x = x.saturating_add(name_width + 5);
    }
    if hint_width > 0 {
        line_at(
            frame,
            Rect::new(area.right() - hint_width, area.y, hint_width, 1),
            Line::styled(hint, p.muted()),
        );
    }
}

/// Suggestions stay in the conversation surface; the composer keeps keyboard focus.
fn slash_menu(frame: &mut Frame, state: &State, body: Rect, p: Palette, hits: &mut HitAreas) {
    let choices = state.slash_choices();
    if choices.is_empty() || body.height < 3 || body.width < 8 {
        return;
    }
    let selected = state.slash_selected.min(choices.len() - 1);
    let label_width = choices
        .iter()
        .map(|choice| width(&choice.label))
        .max()
        .unwrap_or(0)
        .min(40)
        .min(body.width.saturating_sub(6) as usize / 2);
    let detail_width = (body.width as usize).saturating_sub(label_width + 6).max(1);
    let mut lines = Vec::new();
    let mut starts = Vec::new();
    for (index, choice) in choices.iter().enumerate() {
        starts.push(lines.len());
        let selected_row = index == selected;
        let background = if state.monochrome {
            Color::Reset
        } else if selected_row {
            Color::Rgb(70, 70, 70)
        } else {
            p.faint
        };
        let style = if selected_row && state.monochrome {
            p.text().add_modifier(Modifier::REVERSED)
        } else {
            p.text().bg(background)
        };
        let description = if state.monochrome {
            Color::Reset
        } else {
            Color::Rgb(174, 174, 174)
        };
        let detail = wrap(
            Line::styled(choice.detail.clone(), style.fg(description)),
            detail_width,
        );
        for (row, detail) in detail.into_iter().enumerate() {
            let label = if row == 0 {
                fit(&choice.label, label_width)
            } else {
                String::new()
            };
            let mut spans = vec![
                Span::styled(
                    if row == 0 && selected_row {
                        " > "
                    } else {
                        "   "
                    },
                    style,
                ),
                Span::styled(
                    label.clone(),
                    if selected_row {
                        style.add_modifier(Modifier::BOLD)
                    } else {
                        style
                    },
                ),
                Span::styled(
                    " ".repeat(label_width.saturating_sub(width(&label)) + 2),
                    style,
                ),
            ];
            spans.extend(detail.spans);
            lines.push((Line::from(spans).style(style), index));
        }
    }
    let visible = lines
        .len()
        .min(8)
        .min(body.height.saturating_sub(2) as usize);
    let start = starts[selected]
        .saturating_sub(visible / 2)
        .min(lines.len().saturating_sub(visible));
    let area = Rect::new(
        body.x,
        body.bottom() - visible as u16 - 2,
        body.width,
        visible as u16 + 2,
    );
    frame.render_widget(
        Block::default()
            .style(p.text())
            .borders(Borders::TOP | Borders::BOTTOM)
            .border_style(p.text().fg(p.faint))
            .title_top(
                Line::styled(format!(" {} ", choices.len()), p.muted()).alignment(Alignment::Right),
            ),
        area,
    );
    hits.tool_rows.retain(|(rect, _)| !rect.intersects(area));
    hits.detail_rows.retain(|(rect, _)| !rect.intersects(area));
    hits.welcome_rows.retain(|(rect, _)| !rect.intersects(area));
    for (offset, (line, index)) in lines.iter().skip(start).take(visible).enumerate() {
        let rect = Rect::new(area.x, area.y + offset as u16 + 1, area.width, 1);
        frame.render_widget(Block::default().style(line.style), rect);
        line_at(frame, rect, line.clone());
        hits.slash_rows.push((rect, *index));
    }
    if lines.len() > visible {
        let thumb = (visible * visible / lines.len()).max(1);
        let offset = start * (visible - thumb) / (lines.len() - visible);
        for row in 0..visible {
            line_at(
                frame,
                Rect::new(area.right() - 1, area.y + row as u16 + 1, 1, 1),
                Line::styled(
                    if (offset..offset + thumb).contains(&row) {
                        "█"
                    } else {
                        "│"
                    },
                    p.muted().bg(p.faint),
                ),
            );
        }
    }
}

pub fn render<B: Backend>(
    terminal: &mut Terminal<B>,
    state: &mut State,
) -> std::io::Result<HitAreas> {
    // Ratatui 0.29 shows a Frame cursor before moving it. During timer redraws
    // that briefly exposes the cursor at the last updated digit. Keep hardware
    // cursor ownership here, including positioning for IME candidate windows.
    terminal.hide_cursor()?;
    let mut hits = HitAreas::default();
    terminal.draw(|frame| hits = draw(frame, state))?;
    if let Some(position) = hits.cursor {
        terminal.set_cursor_position(position)?;
        terminal.show_cursor()?;
    }
    Ok(hits)
}

/// Paint cells and return layout positions; use `render` to present a live frame.
pub fn draw(frame: &mut Frame, state: &mut State) -> HitAreas {
    let p = Palette::new(state.monochrome);
    let screen = frame.area();
    if state.monochrome
        || state.terminal_graphics.sixel_cell_pixels().is_none()
        || !state.selected_attachment().is_some_and(|attachment| {
            attachment.is_image() && attachment.mime_type() == "image/png"
        })
        || screen.width < 4
        || screen.height < 3
    {
        state.terminal_image.hide();
    }
    frame.render_widget(Block::default().style(p.text()), screen);
    if screen.width < 4 || screen.height < 3 {
        return HitAreas::default();
    }
    let spacious = screen.height >= 18;
    let margin = if screen.width >= 60 { 2 } else { 1 };
    let available = inset(screen, margin, 0);
    let rail_width = if state.show_sidebar && screen.width >= 104 && screen.height >= 20 {
        28
    } else {
        0
    };
    let rail_space = if rail_width > 0 { rail_width + 3 } else { 0 };
    let inner = Rect::new(
        available.x,
        available.y,
        available.width.saturating_sub(rail_space),
        available.height,
    );
    let top = u16::from(spacious);
    let header_height = if spacious { 3 } else { 1 };
    usage_header(
        frame,
        Rect::new(inner.x, top, inner.width, 1),
        state,
        p.muted(),
    );
    let editor_width = inner.width.saturating_sub(2);
    let editor_width = editor_width.saturating_sub(if editor_width >= 12 { 4 } else { 0 });
    let composer_height =
        (crate::composer::row_count(&state.editor, editor_width as usize).min(8) as u16 + 2)
            .min(screen.height.saturating_sub(header_height + 4))
            .max(1);
    let footer_height = if spacious { 3 } else { 1 };
    let composer_y = screen
        .height
        .saturating_sub(composer_height + footer_height);
    let queue_height = u16::from(!state.queued.is_empty() && screen.height >= 12);
    let attachment_height = u16::from(
        (screen.height >= 8 && (!state.attachments.is_empty() || state.can_undo_attachment()))
            || (screen.height >= 14 && (state.attachment_loading || state.recovery.is_some())),
    );
    let steps = state.tasks["steps"].as_array();
    let jobs = state.tasks["jobs"].as_array();
    let plan_height = u16::from(
        screen.height >= 16
            && (steps.is_some_and(|items| !items.is_empty())
                || jobs.is_some_and(|items| !items.is_empty())),
    );
    let activity_height = u16::from(
        screen.height >= 12
            && ((state.busy() && (!state.preparing() || !state.queued.is_empty()))
                || state.phase == Phase::Failed
                || (!state.preparing() && state.last_elapsed.is_some())
                || state.exit_confirmation.is_some()),
    );
    let gap = u16::from(screen.height >= 12);
    let body = Rect::new(
        inner.x,
        header_height,
        inner.width,
        composer_y.saturating_sub(
            header_height + queue_height + activity_height + attachment_height + plan_height + gap,
        ),
    );
    let transcript = body;
    let rail = Rect::new(
        body.right() + if rail_width > 0 { 3 } else { 0 },
        body.y,
        rail_width,
        body.height,
    );
    let composer = Rect::new(inner.x, composer_y, inner.width, composer_height);
    let mut hits = HitAreas {
        transcript,
        composer,
        sidebar: rail,
        ..HitAreas::default()
    };
    if state.entries.is_empty() {
        state.transcript_cache = TranscriptCache::default();
        if state.editor.lines() == [""] && state.slash_choices().is_empty() {
            welcome(frame, transcript, p, &mut hits);
        }
    } else {
        let mut rows = transcript_rows(state, transcript.width as usize, p);
        // Keep the viewed rows steady while new output arrives below them.
        // Width changes intentionally reflow relative to the newest output.
        if state.scroll > 0 {
            if let Some((previous, width)) = state.rendered_rows {
                if width == transcript.width {
                    state.scroll = state
                        .scroll
                        .saturating_add(rows.len())
                        .saturating_sub(previous);
                }
            }
        }
        state.rendered_rows = Some((rows.len(), transcript.width));
        let max = rows.len().saturating_sub(transcript.height as usize);
        state.scroll = state.scroll.min(max);
        if let Some(index) = state.focus_entry.take() {
            if let Some(target) = search_row(&rows, index, &state.search_highlight) {
                state.scroll = max.saturating_sub(target.saturating_sub(1));
            }
        }
        let start = max.saturating_sub(state.scroll);
        if !state.search_highlight.is_empty() {
            let end = (start + transcript.height as usize).min(rows.len());
            highlight_visible_rows(&mut rows, start, end, &state.search_highlight, p);
        }
        render_rows(
            frame,
            transcript,
            &rows,
            start,
            &mut hits.tool_rows,
            &mut hits.detail_rows,
        );
        for (offset, row) in rows
            .iter()
            .skip(start)
            .take(transcript.height as usize)
            .enumerate()
        {
            if let Some(index) = row
                .entry
                .filter(|index| matches!(state.entries.get(*index), Some(Entry::User(_))))
            {
                hits.detail_rows.push((
                    Rect::new(
                        transcript.x,
                        transcript.y + offset as u16,
                        transcript.width,
                        1,
                    ),
                    index,
                ));
            }
        }
        if state.scroll > 0 && transcript.height > 0 {
            let text = format!(
                " ↑ {} rows above latest · Ctrl+End to follow ",
                state.scroll
            );
            line_at(
                frame,
                Rect::new(transcript.x, transcript.bottom() - 1, transcript.width, 1),
                Line::styled(
                    fit(&text, transcript.width as usize),
                    p.accent().bg(p.panel),
                ),
            );
            hits.tool_rows
                .retain(|(rect, _)| rect.y != transcript.bottom() - 1);
            hits.detail_rows
                .retain(|(rect, _)| rect.y != transcript.bottom() - 1);
        }
    }
    sidebar(frame, rail, state, p, &mut hits);
    if plan_height > 0 {
        let steps = state.tasks["steps"].as_array();
        let completed = steps
            .into_iter()
            .flatten()
            .filter(|step| step["status"] == "completed")
            .count();
        let total = steps.map_or(0, Vec::len);
        let running = state.tasks["jobs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|job| job["status"] == "running")
            .count();
        let current = clean(state.tasks["currentTask"].as_str().unwrap_or(""));
        let label = if total > 0 {
            format!("Plan {completed}/{total} · {current}")
        } else {
            "Background work".into()
        };
        let label = if running > 0 {
            format!("{label} · {running} running")
        } else {
            label
        };
        pair(
            frame,
            Rect::new(
                inner.x,
                composer_y - queue_height - activity_height - attachment_height - plan_height,
                inner.width,
                1,
            ),
            &label,
            "Alt+T tasks",
            p.accent(),
            p.muted(),
        );
    }
    if attachment_height > 0 {
        attachment_row(
            frame,
            state,
            Rect::new(
                inner.x,
                composer_y - queue_height - activity_height - attachment_height,
                inner.width,
                1,
            ),
            p,
            &mut hits,
        );
    }
    if queue_height > 0 {
        let y = composer_y - queue_height - activity_height;
        let next = state
            .queued
            .front()
            .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        let label = format!(
            "{} {} queued · {next}",
            if state.queue_paused { "Ⅱ" } else { "↳" },
            state.queued.len(),
        );
        pair(
            frame,
            Rect::new(inner.x, y, inner.width, 1),
            &label,
            if inner.width < 70 {
                ""
            } else if state.queue_paused {
                "Paused · Ctrl+E manage"
            } else {
                "Ctrl+E manage"
            },
            p.accent(),
            p.muted(),
        );
    }
    let border = if state.monochrome {
        Color::Reset
    } else {
        Color::Rgb(140, 140, 148)
    };
    let model_label = if state.runtime_supported {
        format!(
            "{} · {} · {}",
            state.model,
            state.mode,
            match state.approval_policy.as_str() {
                "ask" => "approval: ask",
                "session" => "approval: session",
                _ => "approval: loading",
            }
        )
    } else {
        state.model.clone()
    };
    let model = format!(
        " {}{} ",
        fit(&model_label, composer.width.saturating_sub(16) as usize),
        if state.sandbox { " · sandbox" } else { "" }
    );
    let mut block = Block::default()
        .style(p.text())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(p.text().fg(border))
        .title_bottom(Line::styled(model, p.muted()).alignment(Alignment::Right));
    if !state.title.is_empty() {
        block = block.title_top(Line::styled(
            format!(
                " {} ",
                fit(&state.title, composer.width.saturating_sub(6) as usize)
            ),
            p.muted(),
        ));
    }
    frame.render_widget(block, composer);
    state.editor.set_block(Block::default());
    state.editor.set_style(p.text());
    state.editor.set_cursor_line_style(p.text());
    state.editor.set_cursor_style(if state.overlay.is_some() {
        p.text()
    } else if state.monochrome {
        p.text().add_modifier(Modifier::REVERSED)
    } else {
        p.text().fg(p.bg).bg(p.fg)
    });
    state
        .editor
        .set_placeholder_text(if state.started.is_some() {
            "Queue a follow-up..."
        } else {
            "Type a message..."
        });
    state.editor.set_placeholder_style(p.muted());
    let content = inset(composer, 1, 1);
    let gutter = if content.width >= 12 { 3 } else { 0 };
    let editor_area = Rect::new(
        content.x + gutter,
        content.y,
        content
            .width
            .saturating_sub(gutter + u16::from(content.width >= 12)),
        content.height,
    );
    hits.composer_text = editor_area;
    if gutter > 0 {
        line_at(
            frame,
            Rect::new(content.x + 1, content.y, 1, 1),
            Line::styled(">", p.accent()),
        );
    }
    hits.cursor = crate::composer::draw(
        frame,
        editor_area,
        &mut state.editor,
        &mut state.composer,
        state.overlay.is_none(),
    );
    let color = match state.phase {
        Phase::Ready => p.muted,
        Phase::Failed => p.red,
        Phase::Cancelling => p.amber,
        _ => p.accent,
    };
    let duration = if state.preparing() {
        None
    } else {
        state
            .started
            .map(|t| t.elapsed().as_secs_f64())
            .or(state.last_elapsed)
    };
    let confirming_exit = state.exit_confirmation.is_some();
    let exit_hint = if inner.width >= 36 {
        "Press Ctrl+C again within 2s to exit"
    } else if inner.width >= 24 {
        "Ctrl+C again (2s) to exit"
    } else {
        "Ctrl+C again to exit"
    };
    let mut status = if confirming_exit {
        exit_hint.into()
    } else if state.preparing() {
        if state.queue_paused {
            "Preparing… queue paused"
        } else {
            "Preparing…"
        }
        .into()
    } else {
        format!("{} {}", if state.busy() { "·" } else { "✓" }, state.status)
    };
    if !confirming_exit {
        if let Some(seconds) = duration {
            status.push_str(&format!(" · {}", elapsed(seconds)));
        }
        if !state.queued.is_empty() && queue_height == 0 {
            status.push_str(&format!(
                " · {} queued{} · Ctrl+E",
                state.queued.len(),
                if state.queue_paused { " (paused)" } else { "" }
            ));
        }
    }
    if activity_height > 0 {
        pair(
            frame,
            Rect::new(inner.x, composer_y - 1, inner.width, 1),
            &status,
            if confirming_exit {
                ""
            } else if state.preparing() {
                if inner.width >= 50 {
                    "Esc recalls your message"
                } else {
                    "Esc recalls"
                }
            } else if state.phase == Phase::Cancelling {
                "Ctrl+C twice to exit"
            } else if state.busy() {
                "Esc to stop"
            } else {
                ""
            },
            p.text().fg(color),
            p.muted(),
        );
    }
    let choices = state.slash_choices();
    if !choices.is_empty() {
        slash_menu(frame, state, body, p, &mut hits);
    }
    let footer = Rect::new(
        inner.x,
        screen.height - if spacious { 2 } else { 1 },
        inner.width,
        1,
    );
    if !choices.is_empty() {
        shortcuts(
            frame,
            footer,
            &[
                ("↑↓", "select"),
                ("Tab", "complete"),
                ("Enter", "run"),
                ("Esc", "close"),
            ],
            p,
        );
    } else if state.overlay.is_some() {
        shortcuts(
            frame,
            footer,
            &[
                (
                    "Esc",
                    if state.overlay == Some(Overlay::Approval) {
                        "reject execution"
                    } else {
                        "close panel"
                    },
                ),
                ("Ctrl+C twice", "exit"),
            ],
            p,
        );
    } else if state
        .editor
        .selection_range()
        .is_some_and(|(start, end)| start != end)
    {
        let mut items = vec![("Ctrl+C", "copy"), ("Ctrl+X", "cut")];
        if state.can_undo_attachment() {
            items.push(("Ctrl+Z", "undo attachment"));
        }
        items.push(("F1", "help"));
        shortcuts(frame, footer, &items, p);
    } else if state.entries.is_empty()
        && state.editor.lines() == [""]
        && (!state.busy() || state.preparing())
        && state.queued.is_empty()
        && state.attachments.is_empty()
        && !state.can_undo_attachment()
    {
        pair(
            frame,
            footer,
            if state.saved_drafts.is_empty() {
                "@ attach · Ctrl+V paste · / commands"
            } else {
                "/drafts recover saved input · /sessions history"
            },
            concat!("Aster ", env!("CARGO_PKG_VERSION")),
            p.muted(),
            p.muted(),
        );
    } else {
        let mut items = Vec::new();
        if state.can_undo_attachment() {
            items.push(("Ctrl+Z", "undo attachment"));
        }
        items.push(("Enter", if state.busy() { "queue" } else { "send" }));
        if state.steering_supported && state.phase == Phase::Busy && state.active_turn_id.is_some()
        {
            items.push(("Alt+S", "steer"));
        }
        if !state.attachments.is_empty() {
            items.extend([
                ("Click", "preview"),
                ("×", "remove"),
                ("Alt+Backspace", "last attachment"),
            ]);
        } else {
            items.extend([("Alt+Enter", "newline"), ("Drag", "select")]);
        }
        items.extend([("/", "commands"), ("F1", "help"), ("Ctrl+C twice", "exit")]);
        shortcuts(frame, footer, &items, p);
    }
    overlay(frame, state, p, &mut hits);
    if state.preparing()
        && !state.queued.is_empty()
        && activity_height == 0
        && state.overlay.is_none()
    {
        frame.render_widget(Block::default().style(p.text()), footer);
        let hint = if state.queue_paused {
            "Queue paused · Ctrl+E"
        } else {
            "Preparing… Esc recalls"
        };
        line_at(
            frame,
            footer,
            Line::styled(fit(hint, footer.width as usize), p.muted()),
        );
    }
    if confirming_exit && (activity_height == 0 || state.overlay.is_some()) {
        // Confirmation remains visible over panels and in terminals too short
        // for the activity row, without replacing the underlying task status.
        frame.render_widget(Block::default().style(p.text()), footer);
        line_at(
            frame,
            footer,
            Line::styled(fit(exit_hint, footer.width as usize), p.accent()),
        );
    }
    if state.escape_confirmation.is_some() && state.overlay.is_none() && !confirming_exit {
        let hint = if state.editor.lines().join("\n").is_empty() {
            "Press Esc again to rewind conversation"
        } else {
            "Press Esc again to clear draft · Ctrl+Z undoes"
        };
        frame.render_widget(Block::default().style(p.text()), footer);
        line_at(
            frame,
            footer,
            Line::styled(fit(hint, footer.width as usize), p.muted()),
        );
    }
    hits
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod render_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    use std::time::Instant;

    fn tool() -> Tool {
        Tool {
            id: "edit-1".into(),
            title: "Update welcome screen".into(),
            status: "completed".into(),
            content: "Applied patch successfully".into(),
            diff: "--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old_title\n+new_title".into(),
            expanded: false,
            started: Some(Instant::now()),
            elapsed: Some(1.2),
        }
    }

    fn snapshot(state: &mut State, columns: u16, rows: u16) -> (String, HitAreas) {
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        let mut hits = HitAreas::default();
        terminal.draw(|frame| hits = draw(frame, state)).unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .chunks(columns as usize)
            .map(|row| {
                let mut text = String::new();
                let mut skip = 0;
                for cell in row {
                    if skip > 0 {
                        skip -= 1;
                        continue;
                    }
                    text.push_str(cell.symbol());
                    skip = width(cell.symbol()).saturating_sub(1);
                }
                text
            })
            .collect::<Vec<_>>()
            .join("\n");
        (text, hits)
    }

    fn saved_drafts_state() -> State {
        use crate::drafts::{DraftSnapshot, LoadedDraft};

        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.saved_drafts = vec![
            LoadedDraft {
                session_id: Some("local-draft-offline".into()),
                revision: 2,
                snapshot: DraftSnapshot {
                    composer: "Review 中文 parser".into(),
                    queued: vec!["Run checks".into()],
                    ..Default::default()
                },
                warning: None,
            },
            LoadedDraft {
                session_id: Some("saved-session".into()),
                revision: 1,
                snapshot: DraftSnapshot {
                    composer: "Update DOCS".into(),
                    in_flight: Some("Inspect output".into()),
                    ..Default::default()
                },
                warning: None,
            },
        ];
        state
    }

    fn approval_state() -> State {
        use crate::state::{ApprovalOption, ApprovalPreview, ApprovalRequest};

        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Busy;
        state.approvals.push_back(ApprovalRequest {
            id: serde_json::json!(7),
            title: "Run shell command?".into(),
            code: "result = await self.shell.run('Get-Location')\nprint(result)".into(),
            preview: Some(ApprovalPreview {
                summary: "Run shell command".into(),
                details: vec!["Command: Get-Location".into()],
                complete: true,
            }),
            show_code: false,
            options: vec![
                ApprovalOption {
                    id: "allow_once".into(),
                    name: "Allow once".into(),
                    kind: "allow_once".into(),
                },
                ApprovalOption {
                    id: "allow_session".into(),
                    name: "Allow all code for this session".into(),
                    kind: "allow_always".into(),
                },
                ApprovalOption {
                    id: "reject".into(),
                    name: "Reject".into(),
                    kind: "reject_once".into(),
                },
            ],
        });
        state.open(Overlay::Approval);
        state.selected = state.approvals[0].default_selection();
        state
    }

    #[test]
    fn short_execution_approval_keeps_the_conversation_visible() {
        let mut state = approval_state();
        let (text, hits) = snapshot(&mut state, 110, 36);
        let lines: Vec<_> = text.lines().collect();
        let top = lines
            .iter()
            .position(|line| line.contains("Review execution"))
            .unwrap();
        let bottom = lines
            .iter()
            .position(|line| line.contains(" Esc reject "))
            .unwrap();
        let panel_rows = bottom - top + 1;
        assert!(
            panel_rows <= 22,
            "short approval fills the terminal:\n{text}"
        );
        assert_eq!(hits.choice_rows.len(), 3);
        assert!(text.contains("Command: Get-Location"));
        assert!(text.contains("Allow once approves the entire Python cell."));
        assert!(text.contains("Allow all code for this session"));
        assert!(text.contains("Tab full code"));
        assert!(!text.contains("self.shell.run"));
        assert!(text.find("Command: Get-Location").unwrap() < text.find("1. Allow once").unwrap());
    }

    #[test]
    fn approval_keeps_scope_choices_and_review_controls_on_narrow_terminals() {
        for (columns, rows) in [(48, 24), (80, 24), (110, 36)] {
            for show_code in [false, true] {
                let mut state = approval_state();
                state.approvals[0].show_code = show_code;
                let (text, hits) = snapshot(&mut state, columns, rows);
                assert_eq!(hits.choice_rows.len(), 3, "{columns}x{rows}:\n{text}");
                assert!(text.contains("entire Python"), "{text}");
                assert!(text.contains("Allow all code for this session"), "{text}");
                assert!(text.contains("Esc reject"), "{text}");
                assert!(text.contains("Enter confirm"), "{text}");
                assert!(text.contains("PgUp/PgDn"), "{text}");
                assert!(
                    text.contains(if show_code {
                        "Tab summary"
                    } else {
                        "Tab full code"
                    }),
                    "{text}"
                );
                assert!(
                    text.contains(if show_code {
                        "print(result)"
                    } else {
                        "Command: Get-Location"
                    }),
                    "{text}"
                );
            }
        }
    }

    #[test]
    fn long_approval_details_scroll_without_hiding_permission_choices() {
        for complete in [false, true] {
            let mut state = approval_state();
            let approval = &mut state.approvals[0];
            approval.code = (0..50)
                .map(|index| format!("print('code-line-{index}')\n"))
                .collect();
            let preview = approval.preview.as_mut().unwrap();
            preview.complete = complete;
            preview.details = vec![format!(
                "Command: {}final-command-marker",
                "Get-Location; ".repeat(80)
            )];
            let (_, hits) = snapshot(&mut state, 48, 24);
            assert_eq!(hits.choice_rows.len(), 3);
            state.detail_scroll = u16::MAX;
            let (text, hits) = snapshot(&mut state, 48, 24);
            assert!(state.detail_scroll > 0);
            assert_eq!(hits.choice_rows.len(), 3);
            assert!(text.contains("Allow all code for this session"), "{text}");
            assert!(
                text.contains(if complete {
                    "final-command-marker"
                } else {
                    "code-line-49"
                }),
                "{text}"
            );
        }
    }

    #[test]
    fn short_terminals_keep_an_approval_preview_that_can_be_scrolled() {
        for show_code in [false, true] {
            let mut state = approval_state();
            let approval = &mut state.approvals[0];
            approval.show_code = show_code;
            approval.code = (0..30)
                .map(|index| format!("print('code-line-{index}')\n"))
                .collect();
            approval.preview.as_mut().unwrap().details = vec![format!(
                "Command: {}final-command-marker",
                "Get-Location; ".repeat(30)
            )];
            let (text, hits) = snapshot(&mut state, 48, 16);
            assert!(
                text.contains(if show_code {
                    "Full Python code"
                } else {
                    "Command:"
                }),
                "{text}"
            );
            assert!(text.contains("entire Python"), "{text}");
            assert_eq!(hits.choice_rows.len(), 3);
            state.detail_scroll = u16::MAX;
            let (text, hits) = snapshot(&mut state, 48, 16);
            assert!(state.detail_scroll > 0);
            assert!(
                text.contains(if show_code {
                    "code-line-29"
                } else {
                    "final-command-marker"
                }),
                "{text}"
            );
            assert_eq!(hits.choice_rows.len(), 3);
            assert!(text.contains("Allow all code for this session"), "{text}");
        }
    }

    #[test]
    fn small_approvals_keep_the_operation_visible_and_full_code_reachable() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        for (columns, rows) in [(40, 12), (41, 12), (48, 11)] {
            let mut controller = Controller::new(approval_state(), None, false);
            let (text, hits) = snapshot(&mut controller.state, columns, rows);
            assert!(
                text.contains("Command: Get-Location"),
                "{columns}x{rows}:\n{text}"
            );
            assert_eq!(hits.choice_rows.len(), 3, "{text}");
            assert_eq!(controller.state.selected, 0);
            assert!(controller
                .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
                .is_empty());
            let mut saw_result = false;
            for _ in 0..20 {
                let (text, hits) = snapshot(&mut controller.state, columns, rows);
                saw_result |= text.contains("print(result)");
                assert_eq!(hits.choice_rows.len(), 3, "{text}");
                assert_eq!(controller.state.selected, 0);
                assert!(controller
                    .key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE))
                    .is_empty());
            }
            assert!(saw_result, "{columns}x{rows}: full code was never visible");
            assert!(controller
                .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
                .is_empty());
            let (text, _) = snapshot(&mut controller.state, columns, rows);
            assert!(text.contains("Command: Get-Location"), "{text}");
        }
    }

    #[test]
    fn approval_paging_visits_every_code_line_in_both_directions() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use std::collections::BTreeSet;

        let markers: Vec<_> = (0..37).map(|index| format!("line_{index:02}")).collect();
        for (columns, rows, complete) in [
            (24, 10, true),
            (32, 10, true),
            (40, 12, true),
            (41, 12, true),
            (48, 11, true),
            (40, 12, false),
            (41, 12, false),
            (48, 11, false),
            (48, 12, true),
            (48, 16, true),
            (80, 24, true),
        ] {
            let mut state = approval_state();
            state.approvals[0].show_code = true;
            state.approvals[0].preview.as_mut().unwrap().complete = complete;
            state.approvals[0].code = markers
                .iter()
                .map(|marker| format!("print('{marker}')\n"))
                .collect();
            let mut controller = Controller::new(state, None, false);
            for direction in [KeyCode::PageDown, KeyCode::PageUp] {
                let mut seen = BTreeSet::new();
                for _ in 0..80 {
                    let (text, hits) = snapshot(&mut controller.state, columns, rows);
                    assert_eq!(hits.choice_rows.len(), 3, "{columns}x{rows}:\n{text}");
                    for marker in &markers {
                        if text.contains(marker) {
                            seen.insert(marker.clone());
                        }
                    }
                    let before = controller.state.detail_scroll;
                    assert!(controller
                        .key(KeyEvent::new(direction, KeyModifiers::NONE))
                        .is_empty());
                    snapshot(&mut controller.state, columns, rows);
                    if controller.state.detail_scroll == before {
                        break;
                    }
                }
                let missing: Vec<_> = markers
                    .iter()
                    .filter(|marker| !seen.contains(*marker))
                    .collect();
                assert!(
                    missing.is_empty(),
                    "{columns}x{rows} {direction:?} skipped {missing:?}"
                );
            }
        }
    }

    #[test]
    fn approval_fallback_never_hides_unknown_python() {
        let mut state = approval_state();
        state.approvals[0].preview = None;
        state.approvals[0].show_code = false;
        let (text, hits) = snapshot(&mut state, 80, 24);
        assert!(text.contains("preview unavailable or incomplete"));
        assert!(text.contains("print(result)"));
        assert!(!text.contains("Tab summary"));
        assert!(!text.contains("Tab full code"));
        assert_eq!(hits.choice_rows.len(), 3);
        for (columns, rows) in [(4, 4), (8, 8), (24, 12)] {
            snapshot(&mut state, columns, rows);
        }
    }

    fn attachment_state() -> State {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.editor.insert_str("keep draft");
        for name in [
            "a.txt",
            "source.rs",
            "长文件名称.png",
            "context.txt",
            "👩🏽‍💻 中文 e\u{301} image.png",
        ] {
            state
                .attachments
                .push(crate::attachments::Attachment::Text {
                    name: name.into(),
                    uri: format!("file:///{name}"),
                    text: "snapshot".into(),
                });
        }
        state
    }

    fn image_attachment_state() -> State {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 48, 48);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[80, 140, 200].repeat(48 * 48))
                .unwrap();
            writer.finish().unwrap();
        }
        let mut state = attachment_state();
        state.monochrome = false;
        state.attachments =
            vec![crate::attachments::Attachment::from_image("photo.png", &bytes).unwrap()];
        state.open(Overlay::Attachments);
        state
    }

    #[test]
    fn native_image_preview_is_fitted_below_controls_and_preserves_the_draft() {
        for (columns, rows) in [(48, 18), (80, 24), (132, 40)] {
            let mut state = image_attachment_state();
            state.terminal_graphics = crate::terminal_input::GraphicsCapabilities {
                sixel: true,
                cell_pixels: crate::terminal_image::CellPixels::new(10, 20),
                sixel_display_mode: Some(false),
            };
            let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
            let mut hits = HitAreas::default();
            let pending = terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            assert!(!pending
                .buffer
                .content
                .iter()
                .any(|cell| cell.symbol().contains("\x1bP")));
            assert!(!pending
                .buffer
                .content
                .iter()
                .any(|cell| cell.symbol() == "▀"));
            let request = state
                .terminal_image
                .take_request()
                .expect("visible PNG requests a pixel preview");
            assert!(state.terminal_image.complete(request.render()));
            state.detail_scroll = u16::MAX;
            let completed = terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            let images: Vec<_> = completed
                .buffer
                .content
                .iter()
                .enumerate()
                .filter(|(_, cell)| cell.symbol().contains("\x1bP"))
                .collect();
            assert_eq!(images.len(), 1, "{columns}x{rows}");
            let origin = completed.buffer.pos_of(images[0].0);
            let image = state.terminal_image.ready().unwrap();
            let area = Rect::new(origin.0, origin.1, image.columns, image.rows);
            assert!(hits.attachment_original.contains(&area));
            assert!(area.right() <= columns && area.bottom() < rows);
            assert!(!hits.query.unwrap().intersects(area));
            assert!(hits
                .choice_rows
                .iter()
                .all(|(rect, _)| !rect.intersects(area)));
            assert_eq!(state.detail_scroll, 0);
            assert_eq!(state.editor.lines(), &["keep draft"]);
            assert!(state.terminal_image.take_request().is_none());

            state.overlay = None;
            let closed = terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            assert!(hits.attachment_original.is_empty());
            assert!(!closed
                .buffer
                .content
                .iter()
                .any(|cell| cell.skip || cell.symbol().contains("\x1bP")));
            assert!(state.terminal_image.ready().is_none());
            assert_eq!(state.editor.lines(), &["keep draft"]);
        }
    }

    #[test]
    fn native_preview_falls_back_for_small_viewports_and_recovers_when_pixels_fit() {
        let mut state = image_attachment_state();
        state.terminal_graphics = crate::terminal_input::GraphicsCapabilities {
            sixel: true,
            cell_pixels: crate::terminal_image::CellPixels::new(1, 1),
            sixel_display_mode: Some(false),
        };
        let (fallback, hits) = snapshot(&mut state, 48, 18);
        assert!(fallback.contains('▀'));
        assert!(!fallback.contains("Rendering image…"));
        assert!(!hits.attachment_original.is_empty());
        assert!(state.terminal_image.take_request().is_none());
        assert!(state.terminal_image.ready().is_none());
        assert_eq!(state.editor.lines(), &["keep draft"]);

        state.terminal_graphics.cell_pixels = crate::terminal_image::CellPixels::new(10, 20);
        let (pending, _) = snapshot(&mut state, 48, 18);
        assert!(pending.contains("Rendering image…"));
        assert!(!pending.contains('▀'));
        let request = state
            .terminal_image
            .take_request()
            .expect("a supported viewport queues the pixel preview");
        assert!(state.terminal_image.complete(request.render()));
        let (ready, hits) = snapshot(&mut state, 48, 18);
        assert!(!ready.contains("Rendering image…"));
        assert!(!ready.contains('▀'));
        assert!(!hits.attachment_original.is_empty());
        assert!(state.terminal_image.ready().is_some());
        assert!(state.terminal_image.take_request().is_none());
        assert_eq!(state.editor.lines(), &["keep draft"]);
    }

    #[test]
    fn native_preview_requires_complete_capabilities_and_cancels_when_hidden() {
        let mut state = image_attachment_state();
        for capabilities in [
            crate::terminal_input::GraphicsCapabilities::default(),
            crate::terminal_input::GraphicsCapabilities {
                sixel: true,
                cell_pixels: crate::terminal_image::CellPixels::new(10, 20),
                sixel_display_mode: None,
            },
            crate::terminal_input::GraphicsCapabilities {
                sixel: true,
                cell_pixels: None,
                sixel_display_mode: Some(false),
            },
        ] {
            state.terminal_graphics = capabilities;
            let (text, _) = snapshot(&mut state, 132, 40);
            assert!(text.contains('▀'));
            assert!(state.terminal_image.take_request().is_none());
        }
        state.terminal_graphics.cell_pixels = crate::terminal_image::CellPixels::new(10, 20);
        if let crate::attachments::Attachment::Image { name, .. } = &mut state.attachments[0] {
            *name = format!("{}.png", "long-name-".repeat(30));
        }
        snapshot(&mut state, 132, 40);
        let request = state
            .terminal_image
            .take_request()
            .expect("long names leave room for the image");
        state.open(Overlay::Help);
        snapshot(&mut state, 132, 40);
        assert!(
            !state.terminal_image.complete(request.render()),
            "closing rejects a late image"
        );
        assert!(state.terminal_image.ready().is_none());
        state.open(Overlay::Attachments);
        state.monochrome = true;
        snapshot(&mut state, 132, 40);
        assert!(state.terminal_image.take_request().is_none());
    }

    #[test]
    fn original_image_targets_cover_visible_actions_and_thumbnail_cells() {
        for (columns, rows) in [(24, 8), (48, 18), (80, 24), (132, 40)] {
            let mut state = image_attachment_state();
            let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
            let mut hits = HitAreas::default();
            terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            assert!(!hits.attachment_original.is_empty(), "{columns}x{rows}");
            for rect in &hits.attachment_original {
                assert!(rect.width > 0 && rect.height > 0);
                assert!(rect.right() <= columns && rect.bottom() <= rows);
                assert!(!hits.query.unwrap().intersects(*rect));
                assert!(hits
                    .choice_rows
                    .iter()
                    .all(|(choice, _)| !choice.intersects(*rect)));
                let text: String = (rect.x..rect.right())
                    .map(|x| terminal.backend().buffer()[(x, rect.y)].symbol())
                    .collect();
                assert!(!text.trim().is_empty(), "{rect:?}");
                assert!(
                    !text.contains("snapshot") && !text.contains("of "),
                    "{text}"
                );
            }
            let mut thumbnail_cells = 0;
            for y in 0..rows {
                for x in 0..columns {
                    if terminal.backend().buffer()[(x, y)].symbol() == "▀" {
                        thumbnail_cells += 1;
                        assert!(hits
                            .attachment_original
                            .iter()
                            .any(|rect| rect.contains(Position::new(x, y))));
                    }
                }
            }
            if columns >= 48 {
                assert!(thumbnail_cells > 0, "{columns}x{rows}");
            }
            assert_eq!(state.editor.lines(), &["keep draft"]);
        }
    }

    #[test]
    fn original_image_actions_follow_filtered_selection_and_popup_focus() {
        let mut state = image_attachment_state();
        state
            .attachments
            .insert(0, attachment_state().attachments.remove(0));
        let (text, hits) = snapshot(&mut state, 100, 30);
        assert!(hits.attachment_original.is_empty());
        assert!(!text.contains("Enter open original"));
        state.query = "photo".into();
        let (text, hits) = snapshot(&mut state, 100, 30);
        assert!(text.contains("Open original"));
        assert!(!hits.attachment_original.is_empty());
        state.query = "no-matching-attachment".into();
        assert!(snapshot(&mut state, 100, 30)
            .1
            .attachment_original
            .is_empty());
        state.query.clear();
        state.selected = 1;
        assert!(!snapshot(&mut state, 100, 30)
            .1
            .attachment_original
            .is_empty());
        for overlay in [
            None,
            Some(Overlay::Commands),
            Some(Overlay::Files),
            Some(Overlay::Help),
            Some(Overlay::Approval),
        ] {
            state.overlay = overlay;
            assert!(snapshot(&mut state, 100, 30)
                .1
                .attachment_original
                .is_empty());
        }
    }

    #[test]
    fn metadata_only_images_keep_original_controls_and_visible_errors() {
        for image in [
            image_attachment_state().attachments.remove(0),
            crate::attachments::Attachment::from_image("photo.jpg", &[0xff, 0xd8, 0xff]).unwrap(),
            crate::attachments::Attachment::from_image("photo.gif", b"GIF89a").unwrap(),
            crate::attachments::Attachment::from_image("photo.webp", b"RIFF0000WEBP").unwrap(),
        ] {
            let mut state = image_attachment_state();
            state.monochrome = true;
            state.attachments = vec![image];
            state.original_image_notice = Some((
                "Could not open photo: no image viewer is installed.".into(),
                true,
            ));
            let (text, hits) = snapshot(&mut state, 80, 24);
            assert!(text.contains("no image viewer is installed"), "{text}");
            assert!(text.contains("Open original"), "{text}");
            assert!(!text.contains('▀'));
            assert!(!hits.attachment_original.is_empty());
            state.query = "no-match".into();
            let (text, hits) = snapshot(&mut state, 80, 24);
            assert!(text.contains("no image viewer is installed"), "{text}");
            assert!(hits.attachment_original.is_empty());
        }
    }

    #[test]
    fn scrolled_original_actions_do_not_overlap_detail_scroll_indicator() {
        let mut state = image_attachment_state();
        if let crate::attachments::Attachment::Image { name, .. } = &mut state.attachments[0] {
            *name = format!("{}.png", "long-name-".repeat(30));
        }
        state.detail_scroll = u16::MAX;
        let mut terminal = Terminal::new(TestBackend::new(48, 18)).unwrap();
        let mut hits = HitAreas::default();
        terminal
            .draw(|frame| hits = draw(frame, &mut state))
            .unwrap();
        assert!(state.detail_scroll > 0);
        assert!(hits.attachment_original.len() >= 2);
        for rect in &hits.attachment_original {
            let line: String = (0..48)
                .map(|x| terminal.backend().buffer()[(x, rect.y)].symbol())
                .collect();
            assert!(!line.contains("of "), "{line}");
            assert!(
                line.contains("Open original") || line.contains("Enter open original"),
                "{line}"
            );
        }
    }

    #[test]
    fn attachment_close_targets_match_visible_buttons_and_keep_the_latest_on_narrow_screens() {
        for monochrome in [false, true] {
            for (columns, rows) in [(4, 8), (8, 8), (24, 12), (48, 18), (80, 24), (132, 40)] {
                let mut state = attachment_state();
                state.monochrome = monochrome;
                let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
                let mut hits = HitAreas::default();
                terminal
                    .draw(|frame| hits = draw(frame, &mut state))
                    .unwrap();
                assert!(!hits.attachment_remove.is_empty(), "{columns}x{rows}");
                assert_eq!(
                    hits.attachment_remove.last().unwrap().1,
                    state.attachments.len() - 1
                );
                for (button, index) in &hits.attachment_remove {
                    assert!(*index < state.attachments.len());
                    assert_eq!((button.width, button.height), (1, 1));
                    assert_eq!(
                        terminal.backend().buffer()[(button.x, button.y)].symbol(),
                        "×"
                    );
                    assert!(button.bottom() <= hits.composer.y);
                    assert!(button.right() <= columns);
                }
                assert_eq!(state.editor.lines(), &["keep draft"]);
            }
        }
    }

    #[test]
    fn attachment_close_targets_are_disabled_while_a_panel_is_open() {
        for overlay in [
            Overlay::Commands,
            Overlay::Attachments,
            Overlay::Help,
            Overlay::Files,
            Overlay::Queue,
            Overlay::Approval,
        ] {
            let mut controller =
                crate::controller::Controller::new(attachment_state(), None, false);
            assert!(controller.remove_attachment(0));
            controller.state.open(overlay);
            let (_, hits) = snapshot(&mut controller.state, 100, 30);
            assert!(hits.attachment_preview.is_empty());
            assert!(hits.attachment_remove.is_empty());
            assert!(hits.attachment_overflow.is_none());
            assert!(hits.attachment_undo.is_none());
        }
    }

    #[test]
    fn attachment_preview_remove_overflow_and_undo_have_separate_visible_targets() {
        for monochrome in [false, true] {
            for columns in [24, 48, 132] {
                let mut controller =
                    crate::controller::Controller::new(attachment_state(), None, false);
                controller.state.monochrome = monochrome;
                assert!(controller.remove_attachment(2));
                assert!(controller.state.can_undo_attachment());
                let mut terminal = Terminal::new(TestBackend::new(columns, 30)).unwrap();
                let mut hits = HitAreas::default();
                terminal
                    .draw(|frame| hits = draw(frame, &mut controller.state))
                    .unwrap();
                assert!(!hits.attachment_preview.is_empty());
                let undo = hits
                    .attachment_undo
                    .expect("last removal should be undoable");
                for (offset, character) in "Undo".chars().enumerate() {
                    assert_eq!(
                        terminal.backend().buffer()[(undo.x + offset as u16, undo.y)].symbol(),
                        character.to_string()
                    );
                }
                let mut controls = vec![undo];
                for (preview, index) in &hits.attachment_preview {
                    let remove = hits
                        .attachment_remove
                        .iter()
                        .find(|(_, removed_index)| removed_index == index)
                        .unwrap()
                        .0;
                    assert!(preview.right() < remove.x);
                    assert_eq!(
                        terminal.backend().buffer()[(remove.x, remove.y)].symbol(),
                        "×"
                    );
                    controls.extend([*preview, remove]);
                }
                if let Some(overflow) = hits.attachment_overflow {
                    assert_eq!(
                        terminal.backend().buffer()[(overflow.x, overflow.y)].symbol(),
                        "+"
                    );
                    controls.push(overflow);
                } else {
                    assert_eq!(
                        hits.attachment_preview.len(),
                        controller.state.attachments.len()
                    );
                }
                for (index, control) in controls.iter().enumerate() {
                    assert!(control.right() <= columns && control.bottom() <= hits.composer.y);
                    assert!(controls[index + 1..]
                        .iter()
                        .all(|other| !control.intersects(*other)));
                }
            }
        }
    }

    #[test]
    fn removing_the_last_attachment_keeps_a_visible_undo_control() {
        for columns in [24, 80, 132] {
            let mut state = attachment_state();
            state.attachments.truncate(1);
            let mut controller = crate::controller::Controller::new(state, None, false);
            assert!(controller.remove_attachment(0));
            let (text, hits) = snapshot(&mut controller.state, columns, 24);
            assert!(text.contains("Undo"), "{text}");
            assert!(hits.attachment_undo.is_some());
            assert!(hits.attachment_preview.is_empty());
            assert!(hits.attachment_remove.is_empty());
            assert!(hits.attachment_overflow.is_none());
            if columns >= 80 {
                assert!(text.contains("Attachment removed"), "{text}");
                assert!(text.contains("Ctrl+Z:undo attachment"), "{text}");
            }
        }
    }

    #[test]
    fn composer_text_hit_area_starts_at_rendered_text_and_excludes_the_frame() {
        for columns in [8, 24, 48, 80, 132] {
            let mut state = State::new("workspace".into(), "model".into(), false);
            state.phase = Phase::Ready;
            state.editor.insert_str("CURSOR");
            state.editor.move_cursor(tui_textarea::CursorMove::Head);
            let mut terminal = Terminal::new(TestBackend::new(columns, 24)).unwrap();
            let mut hits = HitAreas::default();
            terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            let text = hits.composer_text;
            assert_eq!(terminal.backend().buffer()[(text.x, text.y)].symbol(), "C");
            assert_eq!(hits.cursor, Some(Position::new(text.x, text.y)));
            assert!(text.x > hits.composer.x && text.y > hits.composer.y);
            assert!(text.right() < hits.composer.right() && text.bottom() < hits.composer.bottom());
            state.editor.select_all();
            state
                .editor
                .insert_str(format!("{}Y", "X".repeat(text.width as usize)));
            state.editor.move_cursor(tui_textarea::CursorMove::Head);
            terminal
                .draw(|frame| hits = draw(frame, &mut state))
                .unwrap();
            let text = hits.composer_text;
            assert_eq!(
                terminal.backend().buffer()[(text.right() - 1, text.y)].symbol(),
                "X"
            );
            assert_eq!(
                terminal.backend().buffer()[(text.x, text.y + 1)].symbol(),
                "Y"
            );
        }
    }

    #[test]
    fn composer_hints_follow_empty_input_selection_attachments_and_running_work() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("@ attach · Ctrl+V paste"), "{text}");
        state.editor.insert_str("select this draft");
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Drag:select"), "{text}");
        state.editor.select_all();
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Ctrl+C:copy"), "{text}");
        assert!(text.contains("Ctrl+X:cut"), "{text}");
        state.editor.cancel_selection();
        state.attachments = attachment_state().attachments;
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Click:preview"), "{text}");
        assert!(text.contains("×:remove"), "{text}");
        assert!(text.contains("Alt+Backspace:last attachment"), "{text}");
        state.phase = Phase::Busy;
        state.steering_supported = true;
        state.active_turn_id = Some("turn-1".into());
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Enter:queue"), "{text}");
        assert!(text.contains("Alt+S:steer"), "{text}");
    }

    #[test]
    fn saved_drafts_hint_keeps_new_conversations_empty() {
        for (columns, rows) in [(40, 12), (48, 12), (80, 24)] {
            for phase in [Phase::Connecting, Phase::Preparing, Phase::Ready] {
                let mut state = saved_drafts_state();
                state.phase = phase;
                let (text, hits) = snapshot(&mut state, columns, rows);
                assert!(
                    text.contains("/drafts recover saved input"),
                    "{columns}x{rows} {phase:?}\n{text}"
                );
                assert!(!text.contains("Review 中文 parser"), "{text}");
                assert!(!text.contains("Update DOCS"), "{text}");
                assert!(state.entries.is_empty());
                assert_eq!(state.editor.lines(), &[""]);
                assert!(state.queued.is_empty());
                assert!(state.recovery.is_none());
                assert!(hits.cursor.is_some());
            }
        }
    }

    #[test]
    fn saved_drafts_picker_keeps_previews_and_filtered_selection_visible() {
        for (columns, rows) in [(40, 12), (48, 12), (80, 24)] {
            let mut state = saved_drafts_state();
            state.editor.insert_str("keep this draft");
            state.open(Overlay::Drafts);
            let (text, hits) = snapshot(&mut state, columns, rows);
            for label in [
                "Saved drafts",
                "newest first",
                "Find saved input",
                "Enter restore",
                "Review 中文 parser",
                "New conversation",
                "Update DOCS",
                "Saved conversation",
            ] {
                assert!(text.contains(label), "{columns}x{rows} {label:?}\n{text}");
            }
            assert_eq!(hits.choice_rows.len(), 2, "{text}");
            assert!(hits.query.is_some());
            if columns >= 80 {
                assert!(text.contains("restore for review"), "{text}");
                assert!(text.contains("interrupted input"), "{text}");
            }

            state.selected = 1;
            state.query = "docs".into();
            let (filtered, hits) = snapshot(&mut state, columns, rows);
            assert!(filtered.contains("› Update DOCS"), "{filtered}");
            assert!(!filtered.contains("Review 中文 parser"), "{filtered}");
            assert!(filtered.contains("Enter restore"), "{filtered}");
            assert_eq!(state.selected, 0);
            assert_eq!(state.choices()[0].value, "1");
            assert_eq!(hits.choice_rows.len(), 1, "{filtered}");
            assert_eq!(hits.choice_rows[0].1, 0);
            assert_eq!(state.editor.lines(), &["keep this draft"]);
        }
    }

    #[test]
    fn saved_drafts_picker_distinguishes_empty_list_from_no_matches() {
        for (columns, rows) in [(40, 12), (48, 12), (80, 24)] {
            let mut state = saved_drafts_state();
            state.saved_drafts.clear();
            state.open(Overlay::Drafts);
            let (empty, hits) = snapshot(&mut state, columns, rows);
            assert!(empty.contains("Saved drafts"), "{empty}");
            assert!(empty.contains("No saved drafts to recover."), "{empty}");
            assert!(empty.contains("saved as you type."), "{empty}");
            assert!(empty.contains("Enter restore"), "{empty}");
            assert!(hits.choice_rows.is_empty());

            state.saved_drafts = saved_drafts_state().saved_drafts;
            state.query = "missing".into();
            let (filtered, hits) = snapshot(&mut state, columns, rows);
            assert!(
                filtered.contains("No matches. Try another search."),
                "{filtered}"
            );
            assert!(!filtered.contains("No saved drafts"), "{filtered}");
            assert!(!filtered.contains("Update DOCS"), "{filtered}");
            assert!(hits.choice_rows.is_empty());
            assert_eq!(state.saved_drafts.len(), 2);
        }
    }

    #[test]
    fn attachment_row_keeps_loading_recovery_and_keyboard_hints() {
        let mut state = attachment_state();
        state.recovery = Some("interrupted input".into());
        state.attachment_loading = true;
        let (text, hits) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Reading attachment…"), "{text}");
        assert!(text.contains("/retry"), "{text}");
        assert!(!hits.attachment_remove.is_empty());
        state.attachment_loading = false;
        let (text, _) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Alt+Backspace remove last"), "{text}");
        assert!(text.contains("/retry"), "{text}");
        state.attachments.clear();
        let (text, hits) = snapshot(&mut state, 132, 30);
        assert!(text.contains("Interrupted input saved"), "{text}");
        assert!(hits.attachment_remove.is_empty());
    }

    #[test]
    fn rewind_picker_shows_scope_latest_selection_and_handles_small_terminals() {
        for (columns, rows) in [(24, 6), (48, 12), (80, 24), (120, 36)] {
            let mut state = State::new("workspace".into(), "model".into(), false);
            state.phase = Phase::Ready;
            state.rewind_points = vec![
                crate::state::RewindPoint {
                    user_message_index: 0,
                    prompt: "First request".into(),
                },
                crate::state::RewindPoint {
                    user_message_index: 2,
                    prompt: "Latest 中文 request\nsearchable detail".into(),
                },
            ];
            state.open(Overlay::Rewind);
            let (text, hits) = snapshot(&mut state, columns, rows);
            assert!(text.contains("Rewind"), "{columns}x{rows}\n{text}");
            if columns >= 48 {
                assert!(text.contains("Files unchanged"), "{columns}x{rows}\n{text}");
                assert!(
                    text.contains("Original conversation kept"),
                    "{columns}x{rows}\n{text}"
                );
                assert!(
                    text.contains("Latest 中文 request"),
                    "{columns}x{rows}\n{text}"
                );
                assert!(!hits.choice_rows.is_empty());
            }
            assert_eq!(state.selected, 0);
            assert_eq!(state.choices()[0].value, "2");
            state.query = "searchable detail".into();
            assert_eq!(state.choices().len(), 1);
            assert_eq!(state.choices()[0].value, "2");
        }
    }

    #[test]
    fn rewind_picker_distinguishes_loading_empty_and_filtered_results() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.open(Overlay::Rewind);
        state.rewind_loading = true;
        assert!(snapshot(&mut state, 100, 30)
            .0
            .contains("Loading rewind points"));
        state.rewind_loading = false;
        let (text, _) = snapshot(&mut state, 100, 30);
        assert!(text.contains("No saved rewind points"));
        assert!(text.contains("Older messages without a checkpoint"));
        state.query = "missing".into();
        assert!(snapshot(&mut state, 100, 30).0.contains("No matches"));
    }

    #[test]
    fn double_escape_hint_tracks_composer_without_changing_the_underlying_status() {
        for draft in ["", "draft"] {
            let mut state = State::new("workspace".into(), "model".into(), false);
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.editor.insert_str(draft);
            state.escape_confirmation = Some(Instant::now());
            let (text, _) = snapshot(&mut state, 100, 30);
            let hint = if draft.is_empty() {
                "Press Esc again to rewind conversation"
            } else {
                "Press Esc again to clear draft"
            };
            assert!(text.contains(hint), "{text}");
            assert_eq!(state.status, "Ready");
            state.escape_confirmation = None;
            assert!(!snapshot(&mut state, 100, 30).0.contains(hint));
        }
    }

    #[test]
    fn exit_confirmation_is_visible_over_panels_without_replacing_task_or_draft() {
        for (columns, rows) in [(24, 6), (48, 12), (80, 24), (120, 36)] {
            for overlay in [None, Some(Overlay::Help), Some(Overlay::Queue)] {
                for phase in [
                    Phase::Connecting,
                    Phase::Preparing,
                    Phase::Ready,
                    Phase::Busy,
                    Phase::Cancelling,
                ] {
                    let mut state = State::new("workspace".into(), "model".into(), false);
                    state.phase = phase;
                    state.status = "Underlying task status".into();
                    state.editor.insert_str("keep draft");
                    let cursor = state.editor.cursor();
                    state.overlay = overlay;
                    state.exit_confirmation = Some(Instant::now());
                    let (text, _) = snapshot(&mut state, columns, rows);
                    let expected = if columns == 24 {
                        "Ctrl+C again to exit"
                    } else {
                        "Press Ctrl+C again within 2s to exit"
                    };
                    assert!(
                        text.contains(expected),
                        "{columns}x{rows}, {overlay:?}, {phase:?}\n{text}"
                    );
                    assert!(!text.to_lowercase().contains("ctrl+c to stop"));
                    assert_eq!(state.status, "Underlying task status");
                    assert_eq!(state.editor.lines(), &["keep draft"]);
                    assert_eq!(state.editor.cursor(), cursor);
                    state.exit_confirmation = None;
                    let (text, _) = snapshot(&mut state, columns, rows);
                    assert!(!text.contains(expected));
                }
            }
        }
    }

    #[test]
    fn task_and_help_hints_use_escape_to_stop_and_double_ctrl_c_to_exit() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Busy;
        state.status = "Working".into();
        let (text, _) = snapshot(&mut state, 100, 30);
        assert!(text.contains("Esc to stop"));
        assert!(!text.to_lowercase().contains("ctrl+c to stop"));
        state.open(Overlay::Help);
        let (text, _) = snapshot(&mut state, 120, 50);
        assert!(text.contains("Ctrl+C twice"));
        assert!(text.contains("Exit within 2s (panel or no draft selection)"));
        assert!(text.contains("Stop a task / close an open panel"));
        assert!(text.contains("Recall a waiting message"));
        assert!(!text.contains("clear draft"));
    }

    #[test]
    fn command_surfaces_do_not_offer_removed_monitor_controls() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.open(Overlay::Commands);
        for query in ["dev", "perf"] {
            state.query = query.into();
            let (text, hits) = snapshot(&mut state, 120, 36);
            assert!(hits.choice_rows.is_empty());
            assert!(!text.contains(&format!("/{query}")));
        }
        state.query = "tasks".into();
        assert!(snapshot(&mut state, 120, 36).0.contains("/tasks"));
        state.open(Overlay::Help);
        let (text, _) = snapshot(&mut state, 120, 100);
        assert!(text.contains("/tasks"));
        assert!(text.contains("/permissions"));
        assert!(!text.contains("/dev"));
        assert!(!text.contains("/perf"));
    }

    #[test]
    fn background_preparation_keeps_welcome_and_draft_quiet() {
        for phase in [Phase::Connecting, Phase::Preparing] {
            for (columns, rows) in [(24, 6), (48, 12), (80, 24), (120, 36)] {
                let mut state = State::new("workspace".into(), "model".into(), false);
                state.phase = phase;
                state.status = "INITIALIZATION_SENTINEL".into();
                let (empty, hits) = snapshot(&mut state, columns, rows);
                assert!(!empty.contains("INITIALIZATION_SENTINEL"));
                assert!(!empty.contains("Esc to stop"));
                assert!(hits.cursor.is_some());
                if columns >= 80 {
                    assert!(empty.contains("Resume session"));
                }
                state.editor.insert_str("draft 中文");
                state.editor.move_cursor(tui_textarea::CursorMove::Back);
                let cursor = state.editor.cursor();
                let (text, hits) = snapshot(&mut state, columns, rows);
                assert!(text.contains("draft 中文"));
                assert!(!text.contains("INITIALIZATION_SENTINEL"));
                assert!(!text.contains("Esc to stop"));
                assert_eq!(state.editor.lines(), &["draft 中文"]);
                assert_eq!(state.editor.cursor(), cursor);
                assert!(hits.cursor.is_some());
            }
        }
    }

    #[test]
    fn background_preparation_only_shows_waiting_messages_and_keeps_pauses_visible() {
        for phase in [Phase::Connecting, Phase::Preparing] {
            for (columns, rows) in [(24, 6), (48, 12), (80, 24), (120, 36)] {
                let mut state = State::new("workspace".into(), "model".into(), false);
                state.phase = phase;
                state.queued.push_back("waiting message".into());
                state.editor.insert_str("another draft");
                let (text, _) = snapshot(&mut state, columns, rows);
                assert!(text.contains("Preparing"), "{columns}x{rows}\n{text}");
                assert!(text.contains("Esc recalls"), "{columns}x{rows}\n{text}");
                assert!(!text.contains("Esc to stop"));
                state.queue_paused = true;
                let (paused, _) = snapshot(&mut state, columns, rows);
                assert!(
                    paused.to_lowercase().contains("paused"),
                    "{columns}x{rows}\n{paused}"
                );
                assert_eq!(state.editor.lines(), &["another draft"]);
            }
        }
    }

    #[test]
    fn responsive_workspace_preserves_draft_and_editor_cursor() {
        for (columns, rows) in [(48, 12), (80, 24), (120, 36)] {
            let mut state = State::new("E:/工作区/project".into(), "local-model".into(), false);
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.editor.insert_str("检查中文路径\nkeep this draft");
            let cursor = state.editor.cursor();
            let (text, hits) = snapshot(&mut state, columns, rows);
            assert!(text.contains("E:/工作区/project"));
            assert!(text.contains("local-model"));
            assert!(text.contains("检查中文路径"));
            assert!(text.contains("keep this draft"));
            assert_eq!(state.editor.cursor(), cursor);
            assert!(hits.composer.height >= 3);
            assert_eq!(hits.sidebar.width, 0);
            state.show_sidebar = true;
            assert_eq!(
                snapshot(&mut state, columns, rows).1.sidebar.width > 0,
                columns >= 104
            );
        }
    }

    #[test]
    fn tool_diff_is_collapsed_then_visible_and_scroll_reaches_all_output() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.entries.push(Entry::Tool(tool()));
        let (collapsed, hits) = snapshot(&mut state, 80, 24);
        assert!(!collapsed.contains("+new_title"));
        assert!(!hits.tool_rows.is_empty());
        if let Entry::Tool(tool) = &mut state.entries[0] {
            tool.expanded = true;
        }
        let (expanded, _) = snapshot(&mut state, 80, 24);
        assert!(expanded.contains("+new_title"));
        for i in 0..80 {
            state
                .entries
                .push(Entry::Assistant(format!("message-{i:03}")));
        }
        let (latest, _) = snapshot(&mut state, 48, 12);
        assert!(latest.contains("message-079"));
        state.scroll = usize::MAX;
        let (oldest, _) = snapshot(&mut state, 48, 12);
        assert!(oldest.contains("Update welcome screen"));
        assert!(state.scroll < usize::MAX);
    }

    #[test]
    fn inspectors_show_complete_diff_and_filtered_tool_details() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.entries.push(Entry::Tool(tool()));
        state.open(Overlay::Changes);
        state.conversation_changes = true;
        let (changes, _) = snapshot(&mut state, 80, 24);
        assert!(changes.contains("-old_title"));
        assert!(changes.contains("+new_title"));
        state.open(Overlay::Tools);
        state.query = "welcome".into();
        let (details, _) = snapshot(&mut state, 120, 36);
        assert!(details.contains("Applied patch successfully"));
        assert!(details.contains("+new_title"));
        state.query = "missing".into();
        assert!(snapshot(&mut state, 48, 12).0.contains("No matches"));
    }

    #[test]
    fn git_changes_show_separate_stages_search_and_safe_narrow_layout() {
        use crate::workspace_changes::{ChangeStage, ChangedFile, WorkspaceChanges};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.editor.insert_str("draft to keep");
        state.workspace_changes = Some(WorkspaceChanges {
            root: "workspace".into(),
            files: vec![
                ChangedFile {
                    path: "src/中文.rs".into(),
                    original_path: None,
                    status: "Modified".into(),
                    stage: ChangeStage::Staged,
                    patch: "--- a/src/中文.rs\n+++ b/src/中文.rs\n@@ -1 +1 @@\n-old\n+staged"
                        .into(),
                },
                ChangedFile {
                    path: "src/中文.rs".into(),
                    original_path: None,
                    status: "Modified".into(),
                    stage: ChangeStage::Unstaged,
                    patch: "--- a/src/中文.rs\n+++ b/src/中文.rs\n@@ -1 +1 @@\n-staged\n+working"
                        .into(),
                },
                ChangedFile {
                    path: "new file.md".into(),
                    original_path: None,
                    status: "Added".into(),
                    stage: ChangeStage::Untracked,
                    patch: "@@ -0,0 +1 @@\n+new".into(),
                },
            ],
            notice: Some("Preview is limited".into()),
        });
        state.open(Overlay::Changes);
        let (wide, hits) = snapshot(&mut state, 120, 36);
        assert!(wide.contains("[Git workspace]"));
        assert!(wide.contains("Staged"));
        assert!(wide.contains("Unstaged"));
        assert!(wide.contains("+staged"));
        assert!(wide.contains("Preview is limited"));
        assert!(hits.query.is_some());
        state.query = "untracked".into();
        let (filtered, _) = snapshot(&mut state, 80, 24);
        assert!(filtered.contains("new file.md"));
        assert!(filtered.contains("+new"));
        state.detail_scroll = 100;
        assert!(snapshot(&mut state, 80, 12).0.contains("+new"));
        for (columns, rows) in [(48, 18), (32, 10), (12, 6), (4, 3)] {
            let (_, hits) = snapshot(&mut state, columns, rows);
            assert!(hits
                .choice_rows
                .iter()
                .all(|(rect, _)| rect.right() <= columns && rect.bottom() <= rows));
            if let Some(rect) = hits.query {
                assert!(rect.right() <= columns && rect.bottom() <= rows);
            }
        }
        assert_eq!(state.editor.lines(), &["draft to keep"]);
        state.changes_error = Some("Not inside a Git repository".into());
        state.workspace_changes = None;
        state.query.clear();
        assert!(snapshot(&mut state, 80, 24)
            .0
            .contains("Not inside a Git repository"));
    }

    #[test]
    fn monochrome_renderer_uses_terminal_default_colors() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.monochrome = true;
        state.entries.push(Entry::Tool(tool()));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| {
                draw(frame, &mut state);
            })
            .unwrap();
        assert!(terminal
            .backend()
            .buffer()
            .content
            .iter()
            .all(|cell| cell.fg == Color::Reset && cell.bg == Color::Reset));
    }

    #[test]
    fn new_output_keeps_scrolled_conversation_at_the_same_rows() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        for i in 0..30 {
            state
                .entries
                .push(Entry::Assistant(format!("message-{i:03}")));
        }
        snapshot(&mut state, 80, 24);
        state.scroll = 25;
        let (before, _) = snapshot(&mut state, 80, 24);
        state
            .entries
            .push(Entry::Assistant("new output arrives".into()));
        let (after, _) = snapshot(&mut state, 80, 24);
        let shown = |text: &str| {
            text.lines()
                .filter(|line| line.contains("message-"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(shown(&before), shown(&after));
    }

    #[test]
    fn search_jump_reaches_the_match_deep_in_a_long_message_and_preserves_draft() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.entries.push(Entry::Assistant(format!(
            "{}\n检索目标\n{}",
            "before\n".repeat(150),
            "after\n".repeat(150)
        )));
        state.editor.insert_str("keep my draft 中文");
        let cursor = state.editor.cursor();
        let mut controller = Controller::new(state, None, false);
        controller.key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        controller.paste("  检索目标  ".into());
        let (results, hits) = snapshot(&mut controller.state, 48, 12);
        assert!(results.contains("检索目标"));
        assert!(!hits.choice_rows.is_empty());
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let (jumped, _) = snapshot(&mut controller.state, 48, 12);
        assert!(jumped.contains("检索目标"));
        assert!(controller.state.scroll > 0);
        assert_eq!(controller.state.editor.cursor(), cursor);
        assert_eq!(controller.state.editor.lines(), &["keep my draft 中文"]);
        assert!(controller.state.focus_entry.is_none());
    }

    #[test]
    fn search_visible_markdown_phrase_jumps_and_highlights_across_styles() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.monochrome = false;
        state.phase = Phase::Ready;
        state.entries.push(Entry::Assistant(format!(
            "{}\nPlease inspect **src/main.rs** before continuing.\n{}",
            "before\n".repeat(100),
            "after\n".repeat(100)
        )));
        state.editor.insert_str("keep this draft 中文");
        let cursor = state.editor.cursor();
        let mut controller = Controller::new(state, None, false);
        controller.key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        controller.paste("inspect src/main.rs".into());
        assert_eq!(controller.state.choices().len(), 1);
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| {
                draw(frame, &mut controller.state);
            })
            .unwrap();
        let highlighted: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .filter(|cell| cell.bg == Palette::new(false).amber)
            .map(|cell| cell.symbol())
            .collect();
        assert_eq!(highlighted, "inspect src/main.rs");
        assert!(controller.state.scroll > 0);
        assert_eq!(controller.state.editor.lines(), &["keep this draft 中文"]);
        assert_eq!(controller.state.editor.cursor(), cursor);
    }

    #[test]
    fn long_user_messages_fold_by_visual_rows_and_expand_on_search_or_click() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.monochrome = false;
        state.entries.push(Entry::User(format!(
            "{}\n隐藏目标\n{}",
            "中文长段落".repeat(40),
            "after\n".repeat(15)
        )));
        let p = Palette::new(false);
        let rows = transcript_rows(&mut state, 40, p);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2].toggle, Some(0));
        assert_eq!(rows[2].entry, None);
        let (screen, hits) = snapshot(&mut state, 48, 12);
        assert!(!screen.contains("隐藏目标"));
        assert!(!hits.detail_rows.is_empty());
        let mut controller = Controller::new(state, None, false);
        controller.toggle_entry(0);
        assert!(controller.state.expanded_entries.contains(&0));
        let rows = transcript_rows(&mut controller.state, 40, p);
        assert!(rows.len() > 3);
        assert!(rows.iter().any(|row| row
            .line
            .spans
            .iter()
            .any(|s| s.content.contains("隐藏目标"))));
        controller.toggle_entry(0);
        controller.input("/search 隐藏目标".into());
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(snapshot(&mut controller.state, 48, 12)
            .0
            .contains("隐藏目标"));
        controller.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(snapshot(&mut controller.state, 48, 12)
            .0
            .contains("隐藏目标"));
    }

    #[test]
    fn expanded_tools_preview_head_and_tail_but_search_reveals_the_middle() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        let mut item = tool();
        item.expanded = true;
        item.diff.clear();
        item.content = (0..30)
            .map(|i| format!("output-{i:02}"))
            .collect::<Vec<_>>()
            .join("\n");
        state.entries.push(Entry::Tool(item));
        let rows = transcript_rows(&mut state, 80, Palette::new(false));
        let text: String = rows
            .iter()
            .flat_map(|row| &row.line.spans)
            .map(|s| s.content.as_ref())
            .collect();
        assert!(text.contains("output-00") && text.contains("output-29"));
        assert!(!text.contains("output-15"));
        assert!(text.contains("25 more lines"));
        assert_eq!(rows.iter().filter(|row| row.toggle == Some(0)).count(), 1);
        let mut controller = Controller::new(state, None, false);
        controller.input("/search output-15".into());
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(snapshot(&mut controller.state, 80, 24)
            .0
            .contains("output-15"));
        controller.state.open(Overlay::Tools);
        assert_eq!(
            controller
                .state
                .selected_tool()
                .unwrap()
                .content
                .lines()
                .count(),
            30
        );
        controller.state.overlay = None;
        if let Entry::Tool(item) = &mut controller.state.entries[0] {
            item.diff = "+diff-only-needle".into();
        }
        controller.input("/search diff-only-needle".into());
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(snapshot(&mut controller.state, 48, 12)
            .0
            .contains("diff-only-needle"));
    }

    #[test]
    fn composer_wraps_long_drafts_without_changing_the_message() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        let text = "输入中文与代码路径 src/main.rs，长消息自动折行。".repeat(3);
        state.editor.insert_str(&text);
        let cursor = state.editor.cursor();
        let (_, hits) = snapshot(&mut state, 48, 20);
        assert!(hits.composer.height > 3);
        assert!(hits.composer.height <= 10);
        assert_eq!(state.editor.lines(), &[text]);
        assert_eq!(state.editor.cursor(), cursor);
        assert!(state.editor.undo());
        assert_eq!(state.editor.lines(), &[""]);
    }

    #[test]
    fn search_expands_and_jumps_into_hidden_tool_output() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        let mut tool = tool();
        tool.content = format!(
            "{}\nneedle123\n{}",
            "before\n".repeat(100),
            "after\n".repeat(100)
        );
        state.entries.push(Entry::Tool(tool));
        let mut controller = Controller::new(state, None, false);
        controller.input("/search needle123".into());
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(&controller.state.entries[0], Entry::Tool(t) if t.expanded));
        assert!(snapshot(&mut controller.state, 80, 24)
            .0
            .contains("needle123"));
    }

    #[test]
    fn queue_remains_visible_on_narrow_screens_and_panel_keeps_multiline_draft() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Busy;
        state.queued.push_back("检查中文路径".into());
        state.queue_paused = true;
        state.editor.insert_str("unsent draft\nsecond line");
        for (columns, rows) in [(48, 12), (80, 24), (120, 36)] {
            state.overlay = None;
            let (screen, hits) = snapshot(&mut state, columns, rows);
            assert!(screen.contains("1 queued"));
            assert!(screen.contains("检查中文路径"));
            assert!(screen.contains("unsent draft"));
            assert!(hits.transcript.bottom() < hits.composer.y);
            state.open(Overlay::Queue);
            let (panel, hits) = snapshot(&mut state, columns, rows);
            assert!(panel.contains("检查中文路径"));
            assert!(panel.contains("paused"));
            assert!(panel.contains("Clear the draft"));
            assert_eq!(hits.choice_rows.len(), 1);
            assert_eq!(state.editor.lines(), &["unsent draft", "second line"]);
        }
    }

    #[test]
    fn command_palette_shows_draft_warning_and_keeps_builtin_commands_available() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.editor.insert_str("keep this draft 中文");
        let cursor = state.editor.cursor();
        let mut controller = Controller::new(state, None, false);
        controller.key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));
        controller.paste("resume".into());
        controller.key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE));
        let (screen, _) = snapshot(&mut controller.state, 120, 36);
        assert!(screen.contains("Clear the draft to insert a command with arguments."));
        assert_eq!(controller.state.editor.lines(), &["keep this draft 中文"]);
        assert_eq!(controller.state.editor.cursor(), cursor);
        controller.state.query = "help".into();
        controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(controller.state.overlay, Some(Overlay::Help));
        assert_eq!(controller.state.editor.lines(), &["keep this draft 中文"]);
    }

    #[test]
    fn multiline_notices_are_separate_rows_and_scroll_hint_has_no_tool_hit() {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.notice("Workspace: example\nModel: local", false);
        let text = snapshot(&mut state, 80, 24).0;
        assert!(text
            .lines()
            .any(|s| s.contains("Workspace: example") && !s.contains("Model:")));
        let mut tool = tool();
        tool.expanded = true;
        tool.content = "line\n".repeat(100);
        state.entries.push(Entry::Tool(tool));
        // Show the complete output, so the test requires scrolling regardless
        // of whether the activity row is visible.
        state.expanded_entries.insert(state.entries.len() - 1);
        state.scroll = 10;
        let (text, hits) = snapshot(&mut state, 80, 24);
        assert!(state.scroll > 0);
        assert!(text.contains("rows above latest"));
        assert!(!hits.tool_rows.is_empty());
        assert!(hits
            .tool_rows
            .iter()
            .all(|(rect, _)| rect.y != hits.transcript.bottom() - 1));
    }

    #[test]
    fn search_highlights_all_matches_across_soft_wrapped_rows() {
        let p = Palette::new(false);
        let mut rows = vec![
            Row {
                line: Line::styled("before 中", p.text()),
                tool: None,
                entry: Some(0),
                gutter: None,
                toggle: None,
            },
            Row {
                line: Line::styled("文 after 中文", p.text()),
                tool: None,
                entry: Some(0),
                gutter: None,
                toggle: None,
            },
        ];
        highlight_rows(&mut rows, "中文", p);
        let highlighted: String = rows
            .iter()
            .flat_map(|row| &row.line.spans)
            .filter(|span| span.style.bg == Some(p.amber))
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(highlighted, "中文中文");
        let preview = search_preview(
            &format!("{}检索目标后文", "前文".repeat(30)),
            "检索目标",
            38,
        );
        assert!(preview.contains("检索目标"));
        assert!(width(&preview) <= 38);
    }

    #[test]
    fn search_highlights_match_fragments_at_viewport_edges() {
        let p = Palette::new(false);
        let rows = || {
            vec![
                Row {
                    line: Line::styled("abc", p.text()),
                    tool: None,
                    entry: Some(0),
                    gutter: None,
                    toggle: None,
                },
                Row {
                    line: Line::styled("def", p.text()),
                    tool: None,
                    entry: Some(0),
                    gutter: None,
                    toggle: None,
                },
            ]
        };
        for (visible_start, visible_end, expected) in [(0, 1, "abc"), (1, 2, "def")] {
            let mut visible = rows();
            highlight_visible_rows(&mut visible, visible_start, visible_end, "abcdef", p);
            let highlighted: String = visible[visible_start..visible_end]
                .iter()
                .flat_map(|row| &row.line.spans)
                .filter(|span| span.style.bg == Some(p.amber))
                .map(|span| span.content.as_ref())
                .collect();
            assert_eq!(highlighted, expected);
        }
    }

    #[test]
    fn inline_commands_keep_the_composer_and_hide_covered_hit_targets() {
        use crate::controller::Controller;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        for (columns, height) in [(32, 12), (48, 12), (80, 24), (120, 36)] {
            let mut state = State::new("workspace".into(), "model".into(), false);
            state.phase = Phase::Ready;
            state.entries.push(Entry::Tool(tool()));
            state.editor.insert_str("/");
            state.slash_selected = state.commands.len() - 1;
            let (_, hits) = snapshot(&mut state, columns, height);
            assert!(!hits.slash_rows.is_empty());
            assert!(hits
                .slash_rows
                .iter()
                .any(|(_, index)| *index == state.slash_selected));
            assert!(hits
                .slash_rows
                .iter()
                .all(|(rect, _)| rect.bottom() < hits.composer.y));
            assert!(hits.tool_rows.iter().all(|(rect, _)| !hits
                .slash_rows
                .iter()
                .any(|(menu, _)| rect.intersects(*menu))));
            assert!(state.overlay.is_none());
            assert_eq!(state.editor.lines(), &["/"]);
            let mut controller = Controller::new(state, None, false);
            controller.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
            assert!(snapshot(&mut controller.state, columns, height)
                .1
                .slash_rows
                .is_empty());
            assert_eq!(controller.state.editor.lines(), &["/"]);
        }
    }

    #[test]
    fn resizing_keeps_controls_inside_the_terminal_and_search_cursor_visible() {
        for (columns, height) in [(4, 3), (20, 6), (32, 8), (48, 12), (80, 24), (160, 40)] {
            for overlay in [
                None,
                Some(Overlay::Search),
                Some(Overlay::Queue),
                Some(Overlay::Tools),
            ] {
                let mut state = State::new("workspace".into(), "model".into(), false);
                state.phase = Phase::Ready;
                state.editor.insert_str("保留草稿");
                state.entries.push(Entry::Tool(tool()));
                state.overlay = overlay;
                state.query = "中文".repeat(100);
                let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
                let hits = render(&mut terminal, &mut state).unwrap();
                assert!(hits.composer.right() <= columns && hits.composer.bottom() <= height);
                assert!(hits
                    .choice_rows
                    .iter()
                    .all(|(rect, _)| rect.right() <= columns && rect.bottom() <= height));
                assert_eq!(state.editor.lines(), &["保留草稿"]);
                if columns >= 48 && overlay == Some(Overlay::Search) {
                    let cursor = terminal.get_cursor_position().unwrap();
                    assert!(cursor.x < columns - 1 && cursor.y < height - 1);
                    assert!(cursor.y < hits.composer.y);
                }
                if columns == 48 && overlay.is_none() {
                    assert!(hits.transcript.height >= 4);
                }
                if columns == 160 {
                    assert_eq!(hits.composer.width, columns - 4);
                }
            }
        }
    }
}

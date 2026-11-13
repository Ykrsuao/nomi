// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Readable terminal Markdown. Unsupported constructs remain visible as text.

use std::{collections::HashMap, sync::OnceLock};

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use syntect::{
    easy::HighlightLines,
    highlighting::{FontStyle, Theme as SyntaxTheme, ThemeSet},
    parsing::SyntaxSet,
};
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Copy)]
pub(crate) struct Theme {
    pub text: Style,
    pub muted: Style,
    pub accent: Style,
    pub code: Style,
    pub added: Style,
    pub removed: Style,
    pub syntax: bool,
}

fn syntax_resources() -> &'static (SyntaxSet, SyntaxTheme) {
    static RESOURCES: OnceLock<(SyntaxSet, SyntaxTheme)> = OnceLock::new();
    RESOURCES.get_or_init(|| {
        (
            SyntaxSet::load_defaults_newlines(),
            ThemeSet::load_defaults()
                .themes
                .remove("base16-ocean.dark")
                .unwrap_or_default(),
        )
    })
}

fn syntax_highlighter(info: &str) -> Option<HighlightLines<'static>> {
    let token = info.split_whitespace().next()?;
    let (syntaxes, theme) = syntax_resources();
    let syntax = syntaxes.find_syntax_by_token(token)?;
    Some(HighlightLines::new(syntax, theme))
}

fn code_spans(
    code: &str,
    highlighter: &mut Option<HighlightLines<'static>>,
    base: Style,
) -> Vec<Span<'static>> {
    if let Some(parser) = highlighter {
        // Newlines keep multiline strings and comments in the correct state.
        let input = format!("{code}\n");
        if let Ok(highlighted) = parser.highlight_line(&input, &syntax_resources().0) {
            let mut spans = Vec::new();
            for (token, text) in highlighted {
                let text = text.trim_end_matches('\n');
                let mut style = base.fg(Color::Rgb(
                    token.foreground.r,
                    token.foreground.g,
                    token.foreground.b,
                ));
                for (font, modifier) in [
                    (FontStyle::BOLD, Modifier::BOLD),
                    (FontStyle::ITALIC, Modifier::ITALIC),
                    (FontStyle::UNDERLINE, Modifier::UNDERLINED),
                ] {
                    if token.font_style.contains(font) {
                        style = style.add_modifier(modifier);
                    }
                }
                push(&mut spans, text, style);
            }
            return spans;
        }
        // A failed grammar must not hide content or corrupt subsequent lines.
        *highlighter = None;
    }
    vec![Span::styled(code.to_owned(), base)]
}

fn push(spans: &mut Vec<Span<'static>>, text: impl Into<String>, style: Style) {
    let text = text.into();
    if text.is_empty() {
        return;
    }
    if let Some(last) = spans.last_mut().filter(|last| last.style == style) {
        last.content.to_mut().push_str(&text);
    } else {
        spans.push(Span::styled(text, style));
    }
}

fn closing_marker(text: &str, marker: &str) -> Option<usize> {
    let mut escaped = false;
    for (offset, ch) in text.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if text[offset..].starts_with(marker) {
            return Some(offset);
        }
    }
    None
}

/// Index complete links once. Searching the remaining suffix at every `[` is
/// quadratic for ordinary array notation and unfinished link destinations.
fn links(text: &str) -> HashMap<usize, (usize, usize)> {
    let mut links = HashMap::new();
    if !text.contains('[') {
        return links;
    }
    let mut brackets = Vec::new();
    let mut parentheses = Vec::new();
    let mut label = None;
    let mut escaped = false;
    for (offset, ch) in text.char_indices() {
        let preceding_label = label.take();
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '[' => brackets.push(offset),
            ']' => {
                if let Some(start) = brackets.pop() {
                    if text.as_bytes().get(offset + 1) == Some(&b'(') {
                        label = Some((start, offset));
                    }
                }
            }
            '(' => parentheses.push(preceding_label),
            ')' => {
                if let Some(Some((start, label_end))) = parentheses.pop() {
                    links.insert(start, (label_end, offset));
                }
            }
            _ => {}
        }
    }
    links
}

fn inline(text: &str, theme: Theme, base: Style, depth: usize) -> Vec<Span<'static>> {
    if depth >= 8 {
        return vec![Span::styled(text.to_owned(), base)];
    }
    let links = links(text);
    let mut spans = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let rest = &text[offset..];
        let ch = rest.chars().next().unwrap();
        if ch == '\\' {
            if let Some(next) = rest[1..]
                .chars()
                .next()
                .filter(|c| c.is_ascii_punctuation())
            {
                push(&mut spans, next.to_string(), base);
                offset += 1 + next.len_utf8();
                continue;
            }
        }
        if ch == '`' {
            let count = rest.bytes().take_while(|b| *b == b'`').count();
            let mut end = count;
            while end < rest.len() {
                let Some(next) = rest[end..].find('`') else {
                    break;
                };
                end += next;
                let run = rest[end..].bytes().take_while(|b| *b == b'`').count();
                if run == count {
                    let mut code = &rest[count..end];
                    if code.starts_with(' ') && code.ends_with(' ') && !code.trim().is_empty() {
                        code = &code[1..code.len() - 1];
                    }
                    push(&mut spans, code, base.patch(theme.code));
                    offset += end + count;
                    break;
                }
                end += run;
            }
            if offset > text.len() - rest.len() {
                continue;
            }
            push(&mut spans, &rest[..count], base);
            offset += count;
            continue;
        }
        let image = rest.starts_with("![");
        if ch == '[' || image {
            let start = offset + usize::from(image);
            if let Some(&(label_end, end)) = links.get(&start) {
                let label = &text[start + 1..label_end];
                let destination = &text[label_end + 2..end];
                if image {
                    push(&mut spans, "image: ", theme.muted);
                }
                spans.extend(inline(
                    label,
                    theme,
                    base.patch(theme.accent).add_modifier(Modifier::UNDERLINED),
                    depth + 1,
                ));
                if destination != label && !destination.is_empty() {
                    push(&mut spans, format!(" ({destination})"), theme.muted);
                }
                offset = end + 1;
                continue;
            }
        }
        let mut formatted = false;
        for (marker, modifier) in [
            ("**", Modifier::BOLD),
            ("__", Modifier::BOLD),
            ("~~", Modifier::CROSSED_OUT),
            ("*", Modifier::ITALIC),
            ("_", Modifier::ITALIC),
        ] {
            if !rest.starts_with(marker)
                || rest[marker.len()..].starts_with(char::is_whitespace)
                || (marker.starts_with('_')
                    && text[..offset]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_alphanumeric))
            {
                continue;
            }
            if let Some(end) = closing_marker(&rest[marker.len()..], marker).filter(|end| *end > 0)
            {
                let body = &rest[marker.len()..marker.len() + end];
                if body.ends_with(char::is_whitespace) {
                    continue;
                }
                spans.extend(inline(body, theme, base.add_modifier(modifier), depth + 1));
                offset += marker.len() * 2 + end;
                formatted = true;
                break;
            }
        }
        if !formatted {
            push(&mut spans, ch.to_string(), base);
            offset += ch.len_utf8();
        }
    }
    spans
}

#[derive(Clone, Copy)]
struct Fence {
    marker: char,
    count: usize,
    indent: usize,
    diff: bool,
}

fn fence(line: &str) -> Option<(Fence, &str)> {
    let indent = line.bytes().take_while(|b| *b == b' ').count();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let marker = rest.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let count = rest.chars().take_while(|ch| *ch == marker).count();
    if count < 3 {
        return None;
    }
    let info = rest[count..].trim();
    if marker == '`' && info.contains('`') {
        return None;
    }
    Some((
        Fence {
            marker,
            count,
            indent,
            diff: matches!(info, "diff" | "patch"),
        },
        info,
    ))
}

fn expand_tabs(line: &str) -> String {
    let mut result = String::new();
    let mut column = 0;
    for ch in line.chars() {
        if ch == '\t' {
            let spaces = 4 - column % 4;
            result.push_str(&" ".repeat(spaces));
            column += spaces;
        } else {
            result.push(ch);
            column += unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        }
    }
    result
}

fn cells(line: &str) -> Vec<&str> {
    let line = line.trim();
    let line = line.strip_prefix('|').unwrap_or(line);
    let mut result = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    let mut code_run = 0;
    let mut offset = 0;
    while offset < line.len() {
        let ch = line[offset..].chars().next().unwrap();
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '`' {
            let count = line[offset..].bytes().take_while(|b| *b == b'`').count();
            if code_run == 0 {
                code_run = count;
            } else if code_run == count {
                code_run = 0;
            }
            offset += count;
            continue;
        } else if ch == '|' && code_run == 0 {
            result.push(line[start..offset].trim());
            start = offset + 1;
        }
        offset += ch.len_utf8();
    }
    if start < line.len() || !line.ends_with('|') {
        result.push(line[start..].trim());
    }
    result
}

fn table_alignment(line: &str) -> Option<Vec<(bool, bool)>> {
    if !line.contains('|') {
        return None;
    }
    cells(line)
        .into_iter()
        .map(|cell| {
            let dashes = cell.trim_matches(':');
            (dashes.len() >= 3 && dashes.bytes().all(|b| b == b'-'))
                .then_some((cell.starts_with(':'), cell.ends_with(':')))
        })
        .collect()
}

fn table(lines: &[&str], alignment: &[(bool, bool)], theme: Theme) -> Vec<Line<'static>> {
    let rows: Vec<Vec<Vec<Span<'static>>>> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            cells(line)
                .iter()
                .map(|cell| {
                    inline(
                        cell,
                        theme,
                        if index == 0 {
                            theme.text.add_modifier(Modifier::BOLD)
                        } else {
                            theme.text
                        },
                        0,
                    )
                })
                .collect()
        })
        .collect();
    let widths: Vec<usize> = (0..alignment.len())
        .map(|column| {
            rows.iter()
                .map(|row| row[column].iter().map(|s| s.content.width()).sum())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut result = Vec::new();
    for (index, row) in rows.into_iter().enumerate() {
        let mut spans = Vec::new();
        for (column, content) in row.into_iter().enumerate() {
            if column > 0 {
                push(&mut spans, " │ ", theme.muted);
            }
            let padding = widths[column] - content.iter().map(|s| s.content.width()).sum::<usize>();
            let (left, right) = match alignment[column] {
                (true, true) => (padding / 2, padding - padding / 2),
                (false, true) => (padding, 0),
                _ => (0, padding),
            };
            push(&mut spans, " ".repeat(left), theme.text);
            spans.extend(content);
            push(&mut spans, " ".repeat(right), theme.text);
        }
        result.push(Line::from(spans));
        if index == 0 {
            result.push(Line::styled(
                widths
                    .iter()
                    .map(|width| "─".repeat(*width))
                    .collect::<Vec<_>>()
                    .join("─┼─"),
                theme.muted,
            ));
        }
    }
    result
}

fn paragraph(line: &str, theme: Theme) -> Line<'static> {
    let expanded = expand_tabs(line);
    let mut rest = expanded.as_str();
    let mut spans = Vec::new();
    let mut base = theme.text;
    let indent = rest.len() - rest.trim_start_matches(' ').len();
    if indent > 0 {
        push(&mut spans, &rest[..indent], base);
        rest = &rest[indent..];
    }
    while let Some(quote) = rest.strip_prefix('>') {
        push(&mut spans, "│ ", theme.accent);
        rest = quote.strip_prefix(' ').unwrap_or(quote);
        base = theme.muted;
    }
    let hashes = rest.bytes().take_while(|b| *b == b'#').count();
    if (1..=6).contains(&hashes) && (rest.len() == hashes || rest[hashes..].starts_with(' ')) {
        rest = rest[hashes..].trim_start();
        let without_hashes = rest.trim_end_matches('#');
        if without_hashes.ends_with(' ') {
            rest = without_hashes.trim_end();
        }
        base = base.add_modifier(Modifier::BOLD);
    } else {
        let unordered = ["- ", "* ", "+ "]
            .iter()
            .find_map(|prefix| rest.strip_prefix(prefix));
        if let Some(item) = unordered {
            let (marker, item) = if let Some(item) = item.strip_prefix("[ ] ") {
                ("[ ] ", item)
            } else if let Some(item) = item
                .strip_prefix("[x] ")
                .or_else(|| item.strip_prefix("[X] "))
            {
                ("[x] ", item)
            } else {
                ("• ", item)
            };
            push(&mut spans, marker, theme.accent);
            rest = item;
        } else {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            if (1..=9).contains(&digits)
                && (rest[digits..].starts_with(". ") || rest[digits..].starts_with(") "))
            {
                push(&mut spans, &rest[..digits + 2], theme.accent);
                rest = &rest[digits + 2..];
            }
        }
    }
    spans.extend(inline(rest, theme, base, 0));
    Line::from(spans)
}

fn render_lines(text: &str, theme: Theme, transcript: bool) -> Vec<Line<'static>> {
    let lines: Vec<_> = text.lines().collect();
    let mut result = Vec::new();
    let mut active: Option<Fence> = None;
    let mut highlighter = None;
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if let Some(open) = active {
            if fence(line).is_some_and(|(close, info)| {
                close.marker == open.marker && close.count >= open.count && info.is_empty()
            }) {
                if !transcript {
                    result.push(Line::from(Span::styled("└─", theme.muted)).style(theme.code));
                }
                active = None;
                highlighter = None;
            } else {
                let trim = line
                    .bytes()
                    .take_while(|b| *b == b' ')
                    .count()
                    .min(open.indent);
                let code = expand_tabs(&line[trim..]);
                let style = if open.diff && code.starts_with('+') {
                    theme.added
                } else if open.diff && code.starts_with('-') {
                    theme.removed
                } else {
                    theme.code
                };
                let mut spans = if transcript {
                    Vec::new()
                } else {
                    vec![Span::styled("│ ", theme.muted)]
                };
                spans.extend(code_spans(&code, &mut highlighter, style));
                result.push(Line::from(spans).style(theme.code));
            }
        } else if let Some((open, info)) = fence(line) {
            if !transcript {
                result.push(
                    Line::from(Span::styled(
                        format!("┌─ {}", if info.is_empty() { "code" } else { info }),
                        theme.muted,
                    ))
                    .style(theme.code),
                );
            }
            if theme.syntax && !open.diff {
                highlighter = syntax_highlighter(info);
            }
            active = Some(open);
        } else if let Some(alignment) = lines
            .get(index + 1)
            .and_then(|line| table_alignment(line))
            .filter(|alignment| line.contains('|') && cells(line).len() == alignment.len())
        {
            let mut table_lines = vec![line];
            index += 2;
            while index < lines.len()
                && lines[index].contains('|')
                && cells(lines[index]).len() == alignment.len()
            {
                table_lines.push(lines[index]);
                index += 1;
            }
            result.extend(table(&table_lines, &alignment, theme));
            continue;
        } else {
            result.push(paragraph(line, theme));
        }
        index += 1;
    }
    result
}

#[cfg(test)]
fn render(text: &str, theme: Theme) -> Vec<Line<'static>> {
    render_lines(text, theme, false)
}

/// Code fences are hidden; actual code and blank lines retain the panel style.
/// Search and display share these lines so decoration cannot shift a match.
pub(crate) fn render_transcript(text: &str, theme: Theme) -> Vec<Line<'static>> {
    render_lines(text, theme, true)
}

pub(crate) fn plain_text(text: &str) -> String {
    let theme = Theme {
        text: Style::default(),
        muted: Style::default(),
        accent: Style::default(),
        // Distinguish fences from literal box drawing characters in paragraphs.
        code: Style::default().add_modifier(Modifier::REVERSED),
        added: Style::default(),
        removed: Style::default(),
        syntax: false,
    };
    render_transcript(text, theme)
        .into_iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    fn theme() -> Theme {
        Theme {
            text: Style::default(),
            muted: Style::default().fg(Color::DarkGray),
            accent: Style::default().fg(Color::Cyan),
            code: Style::default().bg(Color::Black),
            added: Style::default().fg(Color::Green),
            removed: Style::default().fg(Color::Red),
            syntax: false,
        }
    }

    fn plain(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn fences_require_matching_marker_and_length_and_preserve_code() {
        let lines = render("````python\n  print('中文')\n```\n~~~\n````\nDone", theme());
        assert_eq!(
            plain(&lines),
            "┌─ python\n│   print('中文')\n│ ```\n│ ~~~\n└─\nDone"
        );
        assert_eq!(
            plain(&render("~~~\n-2 + 1\n~~~", theme())),
            "┌─ code\n│ -2 + 1\n└─"
        );
        assert_eq!(
            render("```python\n-2\n```", theme())[1].spans[1].style.fg,
            None
        );
        assert_eq!(
            render("```diff\n-old\n+new\n```", theme())[1].spans[1]
                .style
                .fg,
            Some(Color::Red)
        );
    }

    #[test]
    fn nested_inline_formatting_keeps_code_and_link_destinations() {
        let lines = render(
            "**Use `中文_id`** and [**manual**](https://example.test/a_(b))",
            theme(),
        );
        assert_eq!(
            plain(&lines),
            "Use 中文_id and manual (https://example.test/a_(b))"
        );
        let code = lines[0]
            .spans
            .iter()
            .find(|s| s.content == "中文_id")
            .unwrap();
        assert!(code.style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(code.style.bg, Some(Color::Black));
        let label = lines[0]
            .spans
            .iter()
            .find(|s| s.content == "manual")
            .unwrap();
        assert!(label
            .style
            .add_modifier
            .contains(Modifier::UNDERLINED | Modifier::BOLD));
    }

    #[test]
    fn ordinary_brackets_do_not_capture_later_links() {
        assert_eq!(
            plain(&render(
                "Inspect array[42] before [**manual**](https://example.test/a_(b)).",
                theme(),
            )),
            "Inspect array[42] before manual (https://example.test/a_(b))."
        );
        assert_eq!(
            plain(&render(
                r"[unfinished](url [good](target) and \[literal] ![图](image.png)",
                theme(),
            )),
            "[unfinished](url good (target) and [literal] image: 图 (image.png)"
        );
    }

    #[test]
    fn search_text_matches_transcript_with_formatting_links_and_fences() {
        let input = "## Inspect **src/main.rs**\nUse `value[42]` and [manual](docs/readme.md).\n```rust\nlet value = 42;\n```\n┌─ literal\n│ literal\n└─";
        let expected = "Inspect src/main.rs\nUse value[42] and manual (docs/readme.md).\nlet value = 42;\n┌─ literal\n│ literal\n└─";
        assert_eq!(plain(&render_transcript(input, theme())), expected);
        assert_eq!(plain_text(input), expected);
    }

    #[test]
    fn code_panels_preserve_blank_lines_indentation_and_unterminated_tail() {
        let input =
            "Before\n  ```rust\n  fn main() {\n\n      call();\n  }\n  ```\nAfter\n```\ntail";
        let lines = render_transcript(input, theme());
        assert_eq!(
            plain(&lines),
            "Before\nfn main() {\n\n    call();\n}\nAfter\ntail"
        );
        for index in [1, 2, 3, 4, 6] {
            assert_eq!(lines[index].style.bg, Some(Color::Black));
        }
        assert_eq!(plain_text(input), plain(&lines));
        assert!(render_transcript("```rust\n```", theme()).is_empty());
    }

    #[test]
    fn syntax_colors_common_languages_without_changing_code_or_panel_background() {
        for (language, source) in [
            ("rust", "let value: i32 = 42; // number"),
            ("python", "value = 42 # number"),
            ("javascript", "const value = \"hello\"; // greeting"),
            ("json", r#"{"value": true, "count": 42}"#),
            ("bash", "echo \"hello\" # greeting"),
        ] {
            let mut palette = theme();
            palette.syntax = true;
            let input = format!("```{language}\n{source}\n```");
            let lines = render_transcript(&input, palette);
            assert_eq!(plain(&lines), source, "{language}");
            assert_eq!(plain_text(&input), source, "{language}");
            let colors: std::collections::HashSet<_> = lines[0]
                .spans
                .iter()
                .filter(|span| !span.content.trim().is_empty())
                .filter_map(|span| span.style.fg)
                .collect();
            assert!(colors.len() > 1, "{language} should have syntax colors");
            assert_eq!(lines[0].style.bg, palette.code.bg);
            assert!(lines[0]
                .spans
                .iter()
                .all(|span| span.style.bg == palette.code.bg));
        }
    }

    #[test]
    fn multiline_syntax_state_survives_blank_lines_and_resets_between_fences() {
        let mut palette = theme();
        palette.syntax = true;
        let input = "```rust\n/* start\n\nlet value = 42;\n*/\n```\n```rust\nlet value = 42;\n```";
        let lines = render_transcript(input, palette);
        let comment = lines[2].spans[0].style.fg;
        let keyword = lines[4].spans[0].style.fg;
        assert_ne!(comment, keyword);
        assert_eq!(lines[1].style.bg, palette.code.bg);
        assert_eq!(plain_text(input), plain(&lines));
    }

    #[test]
    fn monochrome_and_unknown_languages_remain_plain_and_literal_box_drawing_survives() {
        let reset = Style::default().fg(Color::Reset).bg(Color::Reset);
        let palette = Theme {
            text: reset,
            muted: reset,
            accent: reset,
            code: reset,
            added: reset,
            removed: reset,
            syntax: false,
        };
        let input = "┌─ literal\n│ literal\n└─\n```rust\nlet value = 42;\n\n```";
        let lines = render_transcript(input, palette);
        assert_eq!(
            plain(&lines),
            "┌─ literal\n│ literal\n└─\nlet value = 42;\n"
        );
        assert_eq!(plain_text(input), plain(&lines));
        assert!(lines.iter().flat_map(|line| &line.spans).all(|span| {
            matches!(span.style.fg, None | Some(Color::Reset))
                && matches!(span.style.bg, None | Some(Color::Reset))
        }));

        let mut palette = theme();
        palette.syntax = true;
        let lines = render_transcript("```unknown_language_xyz\nvalue[42]\n```", palette);
        assert_eq!(plain(&lines), "value[42]");
        assert_eq!(lines[0].spans[0].style, palette.code);
    }

    #[test]
    #[ignore = "Run with --release --ignored --nocapture to measure long inline input"]
    fn benchmark_long_brackets_and_unfinished_destinations() {
        for (label, input) in [
            ("ordinary brackets", "value[42] ".repeat(12_000)),
            ("unfinished destinations", "[x](".repeat(30_000)),
        ] {
            let started = std::time::Instant::now();
            let output = plain(&render(&input, theme()));
            let elapsed = started.elapsed();
            assert_eq!(output, input);
            println!("{label}: {} bytes in {elapsed:?}", input.len());
        }
    }

    #[test]
    #[ignore = "Run with --release --ignored --nocapture to measure code highlighting"]
    fn benchmark_code_highlighting_cold_and_warm() {
        let mut palette = theme();
        palette.syntax = true;
        let body = "let answer: usize = 42; // readable code\n".repeat(1_000);
        let input = format!("```rust\n{body}```");
        for label in ["first grammar load", "cached grammar"] {
            let started = std::time::Instant::now();
            let lines = render_transcript(&input, palette);
            let elapsed = started.elapsed();
            assert_eq!(lines.len(), 1_000);
            assert_eq!(plain(&lines), body.trim_end_matches('\n'));
            println!("{label}: {} bytes in {elapsed:?}", body.len());
        }
    }

    #[test]
    fn literal_paths_escapes_and_unfinished_markup_are_not_lost() {
        let input = r"some_field_name C:\Users\Qin \*literal\* **unfinished [bad](url";
        assert_eq!(
            plain(&render(input, theme())),
            r"some_field_name C:\Users\Qin *literal* **unfinished [bad](url"
        );
        assert_eq!(
            plain(&render("Use `` `literal` `` and `open", theme())),
            "Use `literal` and `open"
        );
        assert_eq!(
            plain(&render("#path/to/file\n####### literal", theme())),
            "#path/to/file\n####### literal"
        );
    }

    #[test]
    fn quotes_and_nested_task_lists_keep_hierarchy_and_inline_styles() {
        let lines = render(
            "## **标题**\n> > Use `code`\n- [x] done\n  - [ ] nested\n  2. next",
            theme(),
        );
        assert_eq!(
            plain(&lines),
            "标题\n│ │ Use code\n[x] done\n  [ ] nested\n  2. next"
        );
        assert!(lines[0].spans[0]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
    }

    #[test]
    fn tables_align_wide_characters_and_keep_escaped_and_code_pipes() {
        let lines = render(
            "| 名称 | 状态 |\n| :--- | ---: |\n| 中文 | `a|b` |\n| x | a\\|b |\nAfter",
            theme(),
        );
        assert_eq!(
            plain(&lines),
            "名称 │ 状态\n─────┼─────\n中文 │  a|b\nx    │  a|b\nAfter"
        );
        assert_eq!(lines[0].width(), lines[2].width());
        assert_eq!(lines[2].width(), lines[3].width());
    }

    #[test]
    fn unterminated_code_and_indented_tabs_stay_visible() {
        assert_eq!(
            plain(&render("  ```text\n  \t中文\n  ~~", theme())),
            "┌─ text\n│     中文\n│ ~~"
        );
    }
}

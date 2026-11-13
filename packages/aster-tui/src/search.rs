// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Search the whole conversation, including output hidden inside collapsed tools.
use std::{borrow::Cow, ops::Range};

use crate::state::{Choice, Entry};

/// Return original UTF-8 boundaries even when lowercase expands a character.
pub(crate) fn match_range(text: &str, query: &str) -> Option<Range<usize>> {
    if query.is_empty() {
        return None;
    }
    let folded = text.to_lowercase();
    let needle = query.to_lowercase();
    let start = folded.find(&needle)?;
    let end = start + needle.len();
    let mut offset = 0;
    let mut original_start = None;
    for (index, c) in text.char_indices() {
        let next = offset + c.to_lowercase().map(char::len_utf8).sum::<usize>();
        if original_start.is_none() && start < next {
            original_start = Some(index);
        }
        if end <= next {
            return Some(original_start.unwrap_or(index)..index + c.len_utf8());
        }
        offset = next;
    }
    None
}

fn snippet(text: &str, found: Range<usize>) -> String {
    let start = text[..found.start]
        .char_indices()
        .rev()
        .nth(28)
        .map_or(0, |(index, _)| index);
    let end = text[found.end..]
        .char_indices()
        .nth(100)
        .map_or(text.len(), |(index, _)| found.end + index);
    let excerpt = text[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        excerpt,
        if end < text.len() { "…" } else { "" }
    )
}

pub(crate) fn choices(entries: &[Entry], query: &str) -> Vec<Choice> {
    let query = query.trim();
    if query.is_empty() {
        return Vec::new();
    }
    entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let (label, fields): (String, Vec<Cow<'_, str>>) = match entry {
                Entry::User(text) => ("YOU".into(), vec![Cow::Borrowed(text)]),
                Entry::Assistant(text) => (
                    "ASTER".into(),
                    vec![Cow::Owned(crate::markdown::plain_text(text))],
                ),
                Entry::Notice { text, .. } => ("NOTICE".into(), vec![Cow::Borrowed(text)]),
                Entry::Tool(tool) => (
                    format!("TOOL · {}", tool.title),
                    vec![
                        Cow::Borrowed(&tool.title),
                        Cow::Borrowed(&tool.content),
                        Cow::Borrowed(&tool.diff),
                    ],
                ),
            };
            fields.into_iter().find_map(|text| {
                match_range(&text, query).map(|found| Choice {
                    label: format!("{} · #{}", label, index + 1),
                    detail: snippet(&text, found),
                    value: index.to_string(),
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Tool;
    use std::time::Instant;

    #[test]
    fn searches_chinese_hidden_tool_output_and_diff_in_conversation_order() {
        let entries = vec![
            Entry::User("检查中文路径".into()),
            Entry::Assistant("Other answer".into()),
            Entry::Tool(Tool {
                id: "edit".into(),
                title: "Apply patch".into(),
                status: "completed".into(),
                content: "Output has 中文".into(),
                diff: "+ needle in diff".into(),
                expanded: false,
                started: Some(Instant::now()),
                elapsed: Some(1.0),
            }),
        ];
        let hits = choices(&entries, "中文");
        assert_eq!(
            hits.iter().map(|c| c.value.as_str()).collect::<Vec<_>>(),
            ["0", "2"]
        );
        assert!(hits[1].detail.contains("中文"));
        assert_eq!(choices(&entries, "NEEDLE")[0].value, "2");
        assert!(choices(&entries, "  ").is_empty());
    }

    #[test]
    fn snippet_keeps_the_match_deep_in_a_long_unicode_message() {
        let text = format!("{}检索目标{}", "前文🌍".repeat(200), "后文".repeat(200));
        let hits = choices(&[Entry::Assistant(text)], "检索目标");
        assert!(hits[0].detail.contains("检索目标"));
        assert!(hits[0].detail.starts_with('…'));
        assert!(hits[0].detail.ends_with('…'));
        assert!(hits[0].detail.chars().count() < 140);
    }

    #[test]
    fn expanded_lowercase_matches_keep_original_character_boundaries() {
        let text = "前缀 İSTANBUL 后缀";
        let range = match_range(text, "i\u{307}stan").unwrap();
        assert_eq!(&text[range], "İSTAN");
        assert_eq!(match_range("界ABC", "abc"), Some(3..6));
    }

    #[test]
    fn finds_visible_phrases_across_markdown_styles_without_rewriting_literal_text() {
        let entries = vec![
            Entry::Assistant("Please inspect **src/main.rs** before continuing.".into()),
            Entry::Assistant("Read **中文** and `src/main.rs`.".into()),
            Entry::User("inspect **literal**".into()),
            Entry::Assistant("```text\ninspect **literal**\n```".into()),
        ];
        let hits = choices(&entries, "inspect src/main.rs");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].value, "0");
        assert!(hits[0].detail.contains("inspect src/main.rs"));
        assert_eq!(choices(&entries, "中文 and src/main.rs")[0].value, "1");
        assert!(choices(&entries, "inspect literal").is_empty());
        assert_eq!(choices(&entries, "inspect **literal**").len(), 2);
    }
}

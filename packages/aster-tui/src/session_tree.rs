// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Validated message ancestry and bounded tree rows for the conversation picker.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::state::{clean, Choice};

const MAX_NODES: usize = 10_000;
const MAX_INDENT: usize = 4;
pub const MAX_QUERY_CHARS: usize = 4096;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TreeRole {
    User,
    Assistant,
}

impl TreeRole {
    fn label(self) -> &'static str {
        match self {
            Self::User => "You",
            Self::Assistant => "Assistant",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub role: TreeRole,
    pub preview: String,
    pub navigable: bool,
}

#[derive(Clone, Debug)]
struct TreeRow {
    index: usize,
    prefix: String,
}

#[derive(Clone, Debug, Default)]
pub struct SessionTree {
    pub nodes: Vec<TreeNode>,
    pub current_node_id: Option<String>,
    indices: HashMap<String, usize>,
    rows: Vec<TreeRow>,
    matches: Option<(String, Vec<usize>)>,
}

impl SessionTree {
    pub fn from_response(value: &Value) -> Result<Self> {
        let items = value["nodes"]
            .as_array()
            .context("The conversation tree has no node list")?;
        if items.len() > MAX_NODES {
            bail!("The conversation tree is too large to display");
        }
        let nodes: Vec<TreeNode> = serde_json::from_value(value["nodes"].clone())
            .context("The conversation tree contains an invalid message")?;
        let current_node_id = match value.get("currentNodeId") {
            Some(Value::Null) => None,
            Some(Value::String(id)) => Some(id.clone()),
            _ => bail!("The conversation tree has no valid current position"),
        };
        let mut indices = HashMap::new();
        for (index, node) in nodes.iter().enumerate() {
            if node.id.is_empty()
                || node.id.len() > 256
                || node.id.chars().any(char::is_control)
                || node.preview.len() > 16 * 1024
                || indices.insert(node.id.clone(), index).is_some()
            {
                bail!("The conversation tree contains invalid or duplicate message IDs");
            }
        }
        if current_node_id
            .as_ref()
            .is_some_and(|id| !indices.contains_key(id))
        {
            bail!("The conversation tree's current message is unavailable");
        }
        let matches = match (value.get("query"), value.get("matchedNodeIds")) {
            (None, None) => None,
            (Some(query), Some(ids)) => {
                let query = query.as_str().context("The tree search query is invalid")?;
                if query.chars().count() > MAX_QUERY_CHARS {
                    bail!("The tree search query is too long");
                }
                let ids = ids
                    .as_array()
                    .context("The tree search matches are invalid")?;
                if ids.len() > nodes.len() {
                    bail!("The tree search contains too many matches");
                }
                let mut matched = Vec::with_capacity(ids.len());
                let mut seen = HashSet::new();
                for id in ids {
                    let id = id
                        .as_str()
                        .context("The tree search has an invalid message ID")?;
                    let index = *indices
                        .get(id)
                        .context("A tree search match is unavailable")?;
                    if !seen.insert(index) {
                        bail!("The tree search contains duplicate matches");
                    }
                    matched.push(index);
                }
                Some((query.to_owned(), matched))
            }
            _ => bail!("The tree search response is incomplete"),
        };
        let mut children = vec![Vec::new(); nodes.len()];
        let mut roots = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            if let Some(parent) = &node.parent_id {
                let parent = *indices
                    .get(parent)
                    .context("The conversation tree contains a missing parent")?;
                children[parent].push(index);
            } else {
                roots.push(index);
            }
        }
        let mut stack: Vec<_> = roots
            .into_iter()
            .rev()
            .map(|index| (index, Vec::<bool>::new(), 0_usize))
            .collect();
        let mut rows = Vec::with_capacity(nodes.len());
        let mut visited = vec![false; nodes.len()];
        while let Some((index, columns, depth)) = stack.pop() {
            if visited[index] {
                bail!("The conversation tree contains a cycle");
            }
            visited[index] = true;
            let mut prefix = String::new();
            for (column, continues) in columns.iter().enumerate() {
                prefix.push_str(if column + 1 == columns.len() {
                    if *continues {
                        "├─ "
                    } else {
                        "└─ "
                    }
                } else if *continues {
                    "│  "
                } else {
                    "   "
                });
            }
            if depth > MAX_INDENT {
                prefix.push_str("… ");
            }
            rows.push(TreeRow { index, prefix });
            for (position, child) in children[index].iter().enumerate().rev() {
                let mut next_columns = columns.clone();
                if next_columns.len() < MAX_INDENT {
                    next_columns.push(position + 1 < children[index].len());
                }
                stack.push((*child, next_columns, depth + 1));
            }
        }
        if rows.len() != nodes.len() {
            bail!("The conversation tree contains a cycle");
        }
        Ok(Self {
            nodes,
            current_node_id,
            indices,
            rows,
            matches,
        })
    }

    pub fn node(&self, id: &str) -> Option<&TreeNode> {
        self.indices.get(id).map(|index| &self.nodes[*index])
    }

    pub fn compact_label(&self, id: &str) -> Option<String> {
        let node = self.node(id)?;
        let current = self.current_node_id.as_deref() == Some(id);
        let role = match node.role {
            TreeRole::User => "You",
            TreeRole::Assistant => "AI",
        };
        Some(format!(
            "{}{role}: {}",
            if current { "● " } else { "" },
            preview(&node.preview)
        ))
    }

    pub fn choices(&self, query: &str) -> Vec<Choice> {
        let normalized = query.trim().to_lowercase();
        let mut included = vec![normalized.is_empty(); self.nodes.len()];
        if !normalized.is_empty() {
            let matches =
                if let Some((_, indices)) = self.matches.as_ref().filter(|(q, _)| q == query) {
                    indices.clone()
                } else {
                    self.nodes
                        .iter()
                        .enumerate()
                        .filter_map(|(index, node)| {
                            format!("{} {}", node.role.label(), clean(&node.preview))
                                .to_lowercase()
                                .contains(&normalized)
                                .then_some(index)
                        })
                        .collect()
                };
            for index in matches {
                // Ancestors preserve the branch context of a matching message.
                let mut next = Some(index);
                while let Some(index) = next {
                    if included[index] {
                        break;
                    }
                    included[index] = true;
                    next = self.nodes[index]
                        .parent_id
                        .as_ref()
                        .and_then(|id| self.indices.get(id).copied());
                }
            }
        }
        self.rows
            .iter()
            .filter(|row| included[row.index])
            .map(|row| {
                let node = &self.nodes[row.index];
                let current = self.current_node_id.as_deref() == Some(node.id.as_str());
                let preview = preview(&node.preview);
                let action = if !node.navigable {
                    "History only · restore point unavailable"
                } else if node.role == TreeRole::User {
                    "Edit this message and continue from its parent"
                } else {
                    "Continue after this reply"
                };
                Choice {
                    label: format!(
                        "{}{}{}: {preview}",
                        row.prefix,
                        if current { "● " } else { "" },
                        node.role.label()
                    ),
                    detail: if current {
                        format!("Current position · {action}")
                    } else {
                        action.into()
                    },
                    value: node.id.clone(),
                }
            })
            .collect()
    }

    pub fn has_search_results(&self, query: &str) -> bool {
        self.matches
            .as_ref()
            .is_some_and(|(matched_query, _)| matched_query == query)
    }
}

fn preview(text: &str) -> String {
    clean(text)
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("(empty message)")
        .chars()
        .take(240)
        .collect()
}

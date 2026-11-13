# Grok Build 界面参考

记录日期：2026-11-09。用于 Aster Rust TUI 的视觉调整，不代表以下行为均已实现。

## 证据边界

- 实测对象：本机安装的 Grok Build **1.0.77**。外层布局与有效颜色以此为参考。
- 源码对象：官方 `xai-org/grok-build`，固定提交 `2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8`，本地检出位于 `E:/aster/tui-debug/grok-reference`。稀疏检出以外的文件通过 `git show` 阅读。
- 两者不保证完全同版本、同配置。主题默认值不能直接当作实测值。

1.0.77 实测：统一背景 RGB `(20,20,20)`，主文字 `(225,225,225)`，弱化文字 `(128,128,128)`，观察到的边界 `(140,140,148)`。内容使用可用宽度，左右外边距各两列；输入框三行，模型名在下缘。

下文源码路径缩写：`P` = `crates/codegen/xai-grok-pager/src/`；`R` = `crates/codegen/xai-grok-pager-render/src/`；`M` = `crates/codegen/xai-grok-markdown/src/`。

## 消息与间距

```text
❯ Add a clear empty state.

I'll inspect the workspace and its tests.

◆ Read src/main.rs
◆ Run cargo test

The empty state now explains how to start.
```

示意省略颜色、内边距和临时选中状态。

| 元素 | 源码行为 |
| --- | --- |
| 用户消息 | 首行 `❯ `，续行缩进两列；灰色整行背景带覆盖内边距；普通 RGB 模式正文不强制加粗。 |
| 用户折叠 | 长消息折叠到最多三视觉行，截断处有省略号；默认有上下留白，compact 模式取消。 |
| 助手消息 | 直接输出 Markdown，没有星标、品牌名、角色标题、边框或竖向装饰线；不额外加块内上下留白。 |
| 工具间距 | 连续折叠工具紧邻；其他相邻块间隔一空行。 |
| 公共布局 | 外侧默认各两列，块内左右各两列，并保留 accent 列；保留列不等于画竖线。 |

换行按字符显示宽度计算。terminal-native/minimal 模式可以取消用户背景带；terminal-native 使用属性强调。旧 ConHost 将 `❯ ` 替换为 `> `。

依据：`P/scrollback/blocks/{user,agent,markdown_content}.rs`、`P/scrollback/state/layout.rs`（`gap_after_between`）、`P/scrollback/layout.rs`、`R/appearance/config.rs`、`R/glyphs.rs`。

## 工具记录

默认单个 `◆` 加空格，动作词加粗、括号详情弱化。折叠时整行偏灰，展开后路径可使用主题路径色。成功记录没有统一的绿色勾号。选中/展开指示复用标记位置，不同时堆叠多个图标。旧 ConHost 使用 `♦`。

| 工具 | 折叠行与展开行为 |
| --- | --- |
| Read | `◆ Read src/main.rs (10-30)`；默认及完成后均折叠。展开为标题、空行、带行号和语法高亮的内容。 |
| Agent command | `◆ Run cargo test` 或描述；默认折叠，完成保持折叠状态。截断视图保留前两行、后三行，中间 `… +N lines`。 |
| User shell | 起始截断，完成展开。 |
| Edit | `◆ Edit src/main.rs`，可选变更统计；有效默认展开状态由 `collapsed_edit_blocks` 和外观设置决定。 |

普通工具块无外围底色和上下内边距。Read/command 展开内容有代码底色。展开的 command 可带运行/成功/错误竖线，但折叠摘要隐藏竖线。Read/search/list 可汇总为 `Read 3 files`、`Searched 4 patterns`；commands/edits 的分组规则不同。

依据：`P/scrollback/block.rs`（`prepend_bullet`）、`P/scrollback/blocks/tool/{mod,read,execute,edit}.rs`、`P/scrollback/wrappers/entry_renderer.rs`。

## Markdown 与代码

Markdown 隐藏语法标记，保留粗体、斜体、标题、链接样式。行内代码为加粗彩色文字，不要求背景框。普通软换行合并为空格再按宽度折行，表格单元格换行。

围栏代码保留语法高亮和整行底色，隐藏围栏与语言名；不额外画代码框或语言标题行。空代码行以及末尾未换行的行也保持底色。

依据：`R/theme/md_style.rs`、`M/render.rs`、`P/scrollback/blocks/markdown_content.rs`。

## Diff

普通 RGB 视图：两列缩进、单列右对齐行号、两空格、代码。删除用旧行号，新增和未改动行用新行号；通过颜色区分变化，不在每行前添加 `+`/`-`。

```text
  10  let x = 1;
  11  let y = 2;
  11  let y = 3;
  12  let z = 4;
```

第一个 11 为删除，第二个为新增，纯文本不能呈现色差。代码区域语义底色延伸至右缘，默认不覆盖缩进和行号栏；语法前景色保留。省略区段用 `…`，不显示原始 `@@`；双行号栏可配置。部分主题用整行前景色替代底色。

注意：Edit 的无上下文默认方法返回 collapsed，但有效配置在 `collapsed_edit_blocks=false` 时通常展开，不能据前者断言所有 Edit 默认折叠。

依据：`P/scrollback/blocks/tool/edit.rs`（`render_gutter`、`rendered_output`）、其 `tool/snapshots/` 下的 `diff_basic.snap`，以及 `R/appearance/config.rs`（`EditBlockConfig::effective_expanded`）。

## GrokNight 源码色板

| 用途 | RGB 默认值 |
| --- | --- |
| 画布 / 主文字 | `(20,20,20)` / `(225,225,225)` |
| Markdown 正文 / 弱化灰 | `(200,200,200)` / `(108,108,108)` |
| 用户带 / 代码和输出底色 | `(36,36,36)` / `(28,28,28)` |
| 输入边框 / 激活边框 | `(50,50,55)` / `(80,80,88)` |
| 运行 / 成功 / 错误 | `(187,154,247)` / `(158,206,106)` / `(247,118,142)` |
| Diff 删除 / 新增底色 | `(66,14,20)` / `(6,56,6)` |
| 行内代码 | `(58,149,171)`，加粗 |

依据：`R/theme/groknight.rs`。源码弱化灰和边框与实测 shell 不同。终端能力检测还会量化 RGB；terminal-native/minimal 模式会改变表现。

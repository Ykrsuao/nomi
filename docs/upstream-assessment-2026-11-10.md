# 上游更新吸收评估（2026-11-10）

实施更新：第一批已吸收 #242、#511 和 #563，并补齐 Windows PowerShell 锁修复；第二批已实现 #560 有界等待队列及本地 microsandbox 适配。下文保留实施前的评估依据；实际修改、验证结果和限制见“实施记录”及末尾“第二批：有界等待队列 #560”。

结论：先吸收 `main` 的两项修复，其中 shell 修复需要补齐本地 Windows 后端；第二批优先考虑 #563 的调度策略错误传播。Atom 的会话恢复、输入消费事务和子代理活动展示值得分项借鉴，不建议整分支合并。队列上限、Cell 状态恢复和 reflection 需要额外运行时适配；Relay 则仅在使用相应集成时考虑。

## 核对范围

- 已从官方 `origin` fetch；主线为 `707e1ce3`，提交时间为 2026-11-09 17:09:13 +01:00（北京时间 2026-11-10 00:09:13）。本地 `HEAD` 为 `dadc5621`，共同基线为 `fe7a849e`。`HEAD..origin/main` 有 11 个提交，但净变化属于两个 PR，共 6 个文件、131 行新增、37 行删除。[主线提交](https://github.com/NVIDIA-NeMo/labs-OO-Agents/commit/707e1ce31f1a6a28f848aab7902ab0e8fb1a0628)
- 当前分支 `aster-terminal-chat` 还有未提交的 Rust TUI、ACP、流式回复和会话历史修改。首次评估只新增本文；后续按用户指示实施第一批修复，保留这些既有工作。
- `git merge-tree --write-tree --name-only HEAD origin/main` 的三方合并模拟成功；这只覆盖已提交内容，不能证明未提交工作区也可直接合并。
- 以下开放 PR 状态通过官方 GitHub API 核实；代码评估使用 fetch 后固定提交。开放、非 draft 不等于已发布或兼容本地分支。

## 主线：建议本轮吸收

| 项目 | 收益 | 本地处理 | 优先级 |
| --- | --- | --- | --- |
| [#242：跨事件循环 shell 锁](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/242)，核心 `b91a9f57`，合并 `6e08ce1e` | 修复重建 shell 期间重新分配锁，让另一条命令绕过原锁并发进入的问题 | 吸收 Bash 修改，同时给本地 `PowerShellSession` 做等价修复与平台化测试 | 高 |
| [#511：connect 测试输出隔离](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/511)，合并 `707e1ce3` | JSON 测试只解析 stdout，不受 stderr 提示和 tracing 初始化状态影响；明确 Click 最低版本 | 吸收最终三文件差异，保留本地 TUI extras 和锁文件其他修改 | 中，成本低 |

### Shell 锁：必须照顾 Windows 实际执行路径

上游将锁的事件循环归属与 shell 进程的事件循环归属分开，保证 reset/start 的 await 期间，同一循环内的调用仍使用同一把锁。本地 Bash 仍按 `_started_on_loop` 判断是否换锁，`close()` 也会重建锁；Windows 的 `BashSession.__new__()` 则实际返回 `PowerShellSession`，后者的 `_ensure_loop()` 也有相同的锁与进程状态混用问题。因此只合并上游 Bash 文件，Windows 问题仍然存在。[上游最终代码](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/707e1ce3/src/nooa/tools/_bash_session.py)，本地 [_bash_session.py](../src/aster/tools/_bash_session.py) 与 [_powershell_session.py](../src/aster/tools/_powershell_session.py)。

建议在两个后端都维护独立的锁归属，在第一次 await 前完成设置，把进程重启/清理纳入同一串行区间。测试要覆盖“第一次调用暂停在 reset/start，第二次调用此时到达”，而不只验证两个完整事件循环顺序执行；上游直接访问 Bash 私有方法的测试需要按 Windows 工厂行为适配。[上游测试一](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/707e1ce3/src/nooa/tools/tests/test_bash_lock_loop.py)、[上游测试二](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/707e1ce3/tests/tools/test_bash_cross_loop.py)。

本次只读、无 shell 启动的确定性内存探针结果：本地 Bash reset 未完成时第二命令能进入；相同探针加载上游代码后不能进入。本地 PowerShell 模拟 `_run` 的峰值并发为 2。它们验证了锁问题，尚不能替代 Linux/Windows 的真实进程与 microsandbox 集成测试。

### Connect：吸收最终补丁，保留当前依赖集合

最终变化是 `click>=8.2.1`、测试从 `result.stdout` 解析 JSON、固定测试 API key 并显式模拟 401、独立注入 stderr notice。实际 connect stage 已使用 `click.echo()` 输出 JSON，这主要是测试隔离与依赖契约修正，不是更换业务协议。当前锁定和已安装的 Click 都是 8.5.0，无需为此升级版本。[最终测试](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/707e1ce3/packages/nooa-cli/tests/test_connect_entrypoint.py)、[依赖变化](https://github.com/NVIDIA-NeMo/labs-OO-Agents/commit/d2dd4bde)，本地 [_connect_stages.py](../packages/aster-cli/src/aster_cli/commands/_connect_stages.py)。

保留 `packages/aster-cli/pyproject.toml` 中本地 TUI extras，只修改 Click 下限，并同步 `uv.lock` 的对应依赖元数据。不要覆盖整个文件，也不要只摘最早的 `dcdd6270` 而漏掉后续测试修正。本次当前测试文件 pytest 为 **5 passed**；上游最终测试模块从 Git 对象读入内存后直接调用为 **6/6 通过**，后者不是完整 pytest 生命周期运行。

## 尚未合并的功能：适合分项吸收

### #563：小范围修复，优先于 Atom 功能

[#563](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/563) 当前开放、非 draft，固定头为 `82bb8f7b3dad7be7b738a1b7324559f858ec0a06`。它不依赖 Atom 或 #560；共 4 文件、594 行新增、2 行删除，但生产代码 `codeact.py` 仅新增 4 行、删除 2 行，其余主要是测试。关键变化是在模型调用边界将 `ReasoningReplayError` 的特殊传播扩大到 `(ReasoningReplayError, AdmissionError)`，使调度策略拒绝保留原异常，而不是被误当作生成错误重试或包装成 `GenerationError`。[固定代码](https://github.com/cpakkamisaac-sae/labs-OO-Agents/blob/82bb8f7b3dad7be7b738a1b7324559f858ec0a06/src/nooa/strategies/codeact.py)

本地已有 `AdmissionError` 基类，CodeAct 的相应 catch 尚未包含它；CodeActV2 继承该策略行为，当前未提交修改也没有触及 `codeact.py`。建议作为第二批优先吸收项，连同其测试合入并验证“拒绝只调用一次、异常类型保留”。本轮为静态核查，未复跑该 PR 测试。[本地 codeact.py](../src/aster/strategies/codeact.py)、[本地 admission.py](../src/aster/unifiedllm/admission.py)

### #560：多代理并发有价值，但要适配 microsandbox

[#560](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/560) 当前开放、非 draft，固定头为 `5c4d067ee1cf6191858a9fb02c253f7215b8bf21`，14 文件、1,821 行新增、42 行删除。它给共享模型等待队列增加 `max_pending`，在 HTTP 调用及调用预算消耗之前拒绝超额排队；`None` 保留无上限行为，`0` 表示不等待。对多代理高并发时限制等待任务堆积有实际价值。[固定补丁](https://github.com/cpakkamisaac-sae/labs-OO-Agents/commit/5c4d067ee1cf6191858a9fb02c253f7215b8bf21)

但补丁将 broker 协议从 v3 改为 v4，需要 owner 与所有 worker 同步升级；其异常和 token 运输仍修改旧 fork sandbox 的 executor/worker。本地已替换为 microsandbox，需把对应契约移到 `microsandbox_executor` 和 wire 协议，不能直接 cherry-pick 全 PR。与 #563 相交的 catch/import 应保留 `AdmissionError` 基类捕获，同时保留 #560 对 cell/prefill 队列满错误的终止传播。建议作为独立并发稳定性项目后置；本轮未运行其测试。[本地 microsandbox 执行器](../src/aster/runtime/sandbox/microsandbox_executor.py)

### Atom #558：借鉴会话机制，暂缓整体迁移

[#558](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/558) 当前开放，头提交 `297ff379`；README 明确标记 pre-release。相对共同基线涉及 **412 文件、93,266 行新增、2,124 行删除**，新增 Atom 包、会话树、统一作业、ACP 适配与路由、MCP/技能管理，并修改 core runtime、CodeAct、存储和配置。[包说明](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/297ff379/packages/nooa-atom/README.md)

针对本地已提交 HEAD 的三方合并模拟发现 9 个冲突文件：ACP 的 `cli.py`、`dispatcher.py`、`server.py` 和 `test_cli.py`，以及 `interactive.py`、runtime `actor.py`、sandbox `worker.py`、`_bash_session.py`、`test_executor.py`。未提交的 TUI/ACP 修改尚未计入，因此整体迁移会与已完成的 Windows/microsandbox 适配重叠。值得拆开的能力是：

1. **崩溃会话的复制恢复**：`SessionStore.in_use()` 列出被占用会话，`fork()` 复制成新会话并记录来源；不改原库，明确丢弃未消费输入、不自动复制子会话。对终端意外退出后继续工作有直接价值。可将 `/recover` 和复制恢复迁入当前 SessionStore，按本地锁/SQLite 语义实现。[store.py:858、904](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/297ff379/packages/nooa-atom/src/nooa_atom/session/store.py#L858)
2. **输入消费与 turn 开始的原子记录**：本地 `record_user_message()` 已在派发前保存用户输入；Atom 进一步把 `ItemConsumed` 与 `TurnStarted` 放入一次 `record_batch()`，成功后才移除内存归属。适合未来消息排队、steer 和崩溃重放；需要连同 turn 生命周期和事务接口适配，不能只复制一处函数。[session.py:847](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/297ff379/packages/nooa-atom/src/nooa_atom/session/session.py#L847)、[本地 SessionHandle](../packages/aster-cli/src/aster_cli/sessions/store.py)
3. **子代理活动的独立展示**：ACP 的工具卡按 `(session_id, tool_call_id)` 区分，子代理结束只清理自己的卡；支持 verbose/concise/silent。适合以后给 TUI 加多代理状态与可调活动密度，应迁移显示契约并适配本地 Rust 端。[event_bridge.py:399、440、478](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/297ff379/packages/nooa-atom/src/nooa_atom/acp/event_bridge.py#L399)

### Cell 状态恢复 #572：价值较高，但有运行时前置条件

[#572](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/572) 当前开放，头提交 `d96f9347`，目标分支是 Atom 集成分支，**不是 main**。功能增量为 8 文件、1,394 行新增、4 行删除。可保存普通变量、符合现有 snapshot 白名单的对象、受限的 cell 函数/类及实例；会报告跳过项，并设置每变量 1 MiB、总计 8 MiB 上限。[cell_state.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/d96f9347/src/nooa/runtime/cell_state.py)

实现依赖 `ExecutionNamespace` 的源码元数据、Atom `before_checkpoint()`/`after_restore()`、Session checkpoint。当前 microsandbox 的运行环境快照主要用于准备环境复用，与保存 Python 会话 namespace 不是同一层。若要实现 VM 重启后的变量恢复，还需在 guest 与 host 之间提供状态导出/恢复通路。建议先确定期望保留的数据范围，再单独适配；不建议直接 cherry-pick 两个提交即视为功能完成。[Atom 接入](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/d96f9347/packages/nooa-atom/src/nooa_atom/agent/agent.py#L787)、[Session 接入](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/d96f9347/packages/nooa-atom/src/nooa_atom/session/session.py#L1061)、[本地 microsandbox 执行器](../src/aster/runtime/sandbox/microsandbox_executor.py)。

### Reflection #573：等基础会话层稳定后再试

[#573](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/573) 当前开放，头提交 `0c0281a2`，也以 Atom 集成分支为基线；增量为 18 文件、1,075 行新增、10 行删除。它按回合数、时间或上下文占用排入一个静默整理回合，处理命名、历史压缩、todo、变量和失联作业恢复提示。交互 host 默认启用、headless 默认关闭，默认触发值为 20 回合、60 分钟或 50% 上下文，并有最少 5 回合间隔。[reflection.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/0c0281a2/packages/nooa-atom/src/nooa_atom/session/reflection.py)

这是会产生额外模型调用、改变上下文和会话调度的功能，依赖 Atom Session/作业/ACP 生命周期。建议在恢复和输入消费规则稳定后以开关做对照测试，衡量长期任务收益、token 成本和交互延迟。本次未运行该分支测试或性能评估。

### Relay #568：需要观测或轨迹导出时再吸收

[#568](https://github.com/NVIDIA-NeMo/labs-OO-Agents/pull/568) 当前为开放 draft，头提交 `64a51b54`，12 文件、1,275 行新增、176 行删除。将可选 Relay 依赖从 `>=0.6,<0.7` 改为 `>=0.11.1,<0.12`，使用公开 scope 传播接口、避免重叠安装导致重复中间件、保留 sanitizer 契约，并修正工具结果与 ATIF 投影。[补丁文档](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/64a51b54/docs/nemo-relay.md)、[中间件](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/64a51b54/src/nooa/nemo_relay_middleware.py)

该变更不依赖 Atom，适合单独集成；但当前 CLI/ACP 源码没有发现启用 Relay 的调用，不能假定用户正在使用。只有需要 Relay 策略/观测/轨迹导出时，才值得承担升级和验证成本；应合入两个修正提交及配套测试，保留本地根 `pyproject.toml`/`uv.lock` 改动。分支文档仍注明 prefill 轨迹归属限制，本次没有运行联网 Relay 集成测试。

## 建议实施顺序

1. 保存当前 TUI/ACP 工作基线后，吸收 `main` 两项最终差异，并补 Windows shell 锁修复。
2. 针对 connect stdout/stderr、Bash/PowerShell 并发与 loop reset 跑精确回归；再跑受影响的 ACP/终端集成检查。
3. 第二批吸收 #563，并验证调度策略拒绝的异常类型和不重试行为。
4. 将 #560 的并发上限、Atom 的会话复制恢复和输入消费事务作为独立任务；多代理展示按 TUI 需求引入。
5. Cell 状态恢复、reflection 等会话/runtime 前置能力明确后再做；Relay 按实际使用需求决定。

本报告中的源码结论来自固定 Git 对象及当前本地实现。合并模拟、内存探针、测试模块直接调用与真实进程/VM 集成测试是不同层次的证据，尚未验证的部分不作为已经通过的兼容性保证。

## 实施记录（2026-11-10）

本批以补丁方式落入当前工作区，未创建提交或推送。来源为主线 `707e1ce3` 的 #242/#511 最终差异，以及 #563 固定提交 `82bb8f7b3dad7be7b738a1b7324559f858ec0a06`。

- **#242**：Bash 引入独立 `_lock_loop`，close/reset 不再更换命令仍持有的锁，保留本地 Windows 工厂分派。PowerShell 同样独立管理锁归属，并将 `start/reset/close/run/run_stream` 放入同一个串行区间；内部启动/停止使用不重复加锁的 helper，避免初始化命令和失败清理发生死锁。跨 loop 清理不 await 旧 loop 的 Future，Job Object 管理保留。
- **#511**：Click 下限改为 `>=8.2.1` 并同步 lock metadata，保留 TUI extras；完整吸收 stdout JSON、固定测试凭据和独立 stderr 通知测试。实际 Click 仍是 8.5.0。
- **#563**：模型生成边界将 `AdmissionError` 标为终止并原样传播；普通 provider 错误仍按原策略重试。引入上游错误矩阵和 loopback HTTP 回归，将旧 Linux-only sandbox 测试适配到本地 microsandbox。
- 第一批结束时，#560、Atom、cell-state、reflection 和 Relay 尚未实施；#560 的后续进展见末尾第二批记录。

先失败再修复的验证：shell 新回归在原实现上出现 7 项确定性失败，覆盖重启期间越过锁和生命周期操作干扰命令；#563 原实现把 `AdmissionCallCapError` 包装成 `GenerationError`，修复后相同 4 个参数组合通过。

| 验证范围 | 结果 |
| --- | --- |
| Windows shell 锁、普通/流式生命周期、原生 Windows 功能 | 32 passed；新增流式生命周期参数另 3 passed |
| WSL Linux：Bash 生命周期、超时、跨 loop 回归 | 18 passed |
| WSL Linux：上游锁回归及新 PowerShell 测试中的平台无关部分 | 14 passed，8 个 native Windows 参数按平台跳过 |
| #563 非 sandbox 错误矩阵、admission、重试、loopback HTTP | 285 passed，14 个 sandbox 参数由选择条件排除 |
| connect 入口、Windows、UTF-8 输出及平台导入 | 14 passed |
| 现有 ACP 事件桥、prompt progress、流式回复、reply strategy、会话和 usage | 138 passed |
| Windows WHP 真实 microsandbox：两种 CodeAct、执行后限额耗尽、cell/prefill 嵌套错误处理 | 14 passed，181.43 秒 |
| 本批 Python 文件 Ruff lint/format，全部 Python SPDX，`uv lock --check --offline`，`git diff --check` | 全部通过 |

真实 VM 验证使用完整独立环境、正式版 microsandbox 0.7.13 和 cloudpickle 3.1.2，没有改变日常 `.venv`。最初尝试依赖叠加环境时，2 项测试在 namespace 序列化阶段失败于 `_pydantic_core._schema_gather`；这是此前验证记录中已有的叠加环境限制。换完整环境后同组 14 项全部通过。测试前后 `msb ls` 均无活动沙箱。

独立测试环境位于 `.codex-debug/upstream-absorb-venv-20261110`。收尾时已核实其绝对路径和非链接目录属性；自动审批仍按策略拒绝递归删除，因此保留该环境及测试证据，未尝试绕过清理限制。

### 扩展检查的初次结果

初次额外运行 CLI connect 组（Python UTF-8 模式）得到 **1584 passed、34 skipped、49 failed**，当时尚未逐项完成旧基线对照。失败分布为 command 5、diagnostic 1、save_policy 7、edit 1、docs_contract 35。用户要求继续排查后，已完成前后对照及修复，详见下方“Connect 失败排查与修复”。HOME 用例在真实用户目录产生的纯测试配置已按完整内容和创建时间核实并清理，原有用户配置保留；后续所有测试均隔离 HOME、USERPROFILE 及配置目录。

跨 loop 测试仍有 subprocess transport 在旧 loop 关闭后析构的警告。隔离进程中对原 HEAD 和本次 PowerShell 实现运行相同清理探针，警告相同；没有为清除此既有警告扩大补丁。交叉审查另确认 `run_stream` 取消后的 `Queue.get` 残留在 HEAD 与当前实现均存在，本批未修改。

工作区保护：实施前记录 63 个已修改/未跟踪文件的 SHA-256，核对保留原有 TUI、ACP、streaming 内容；本批在已有文件中只调整 CLI manifest/lock 的 Click 下限，并更新本评估记录。其余实现补丁落在原先干净的文件或新建回归测试中。

### 复现与证据

在仓库根目录运行（已有开发依赖）：

```powershell
uv run --no-sync --offline python -X utf8 -m pytest -q packages/aster-cli/tests/test_connect_entrypoint.py packages/aster-cli/tests/test_connect_windows.py packages/aster-cli/tests/test_cli_utf8_output.py tests/test_native_windows.py::test_import_and_cli_need_no_platform_override
uv run --no-sync --offline python -X utf8 -m pytest -q tests/tools/test_powershell_session_lifecycle.py src/aster/tools/tests/test_bash_lock_loop.py tests/tools/test_bash_cross_loop.py tests/test_native_windows.py
uv run --no-sync --offline python -X utf8 -m pytest -q tests/strategies/test_codeact_admission_errors.py tests/unifiedllm/test_admission.py tests/unifiedllm/test_admission_attempt.py tests/unifiedllm/test_pacing_end_to_end.py
# 启用完整 sandbox 环境后，单独选择真实 microVM 用例：
uv run --no-sync --offline python -X utf8 -m pytest -q tests/strategies/test_codeact_admission_errors.py -m sandbox
```

本机 JUnit 证据保存在仓库外 `.codex-debug/`：`upstream-absorb-acp-20261110.xml`、`upstream-absorb-linux-lifecycle-20261110.xml`、`upstream-absorb-linux-lock-20261110.xml`、`upstream-absorb-microvm-standalone-20261110.xml`。初次环境失败记录为 `upstream-absorb-microvm-20261110.xml`，不与最终通过结果混同。

## Connect 失败排查与修复（2026-11-10）

### 前后基线对照

在 detached worktree `.codex-debug/connect-before-20261110` 恢复吸收前的 `dadc5621` 及原有 62 个工作区文件，排除评估报告，并撤回 Click 下限变更。因此对照保留了原有 TUI/ACP 工作，并非仅拿干净 HEAD 比较。使用相同 Python、已安装依赖、隔离环境和五个失败测试文件，吸收前后均为 **49 failed、187 passed**，失败节点集合完全一致，**没有由本批吸收新增的失败**。两侧实际导入源码路径也已记录并检查。

| 原有失败 | 数量 | 原因与处理 |
| --- | ---: | --- |
| 文档契约的 socket 保护 | 35 | 全面禁用 `socket.connect` 误拦 Windows asyncio 创建内部 socketpair。只允许标准库 socketpair 实现的精确调用者，普通回环/外部连接、`connect_ex`、连接辅助函数和 DNS 仍拒绝；新增 6 项正负回归。 |
| 用户目录与路径断言 | 6 | 补齐 `USERPROFILE` 隔离；以原生绝对路径、解码后的 JSON 和 `OSError` 的文件名表示验证结果。 |
| 文件锁断言 | 7 | 不能用锁文件释放后是否留存判断互斥，也不能在 Windows 无条件导入 `fcntl`。改为真实独立锁所有者竞争，覆盖 CAS、事务及回滚期间互斥和结束后释放；保留 POSIX 的 sidecar 断言。 |
| 文件权限断言 | 1 | POSIX 保留严格 `0600`；Windows 检查文件可读写并准确记录既有继承 ACL 行为。未修改权限实现，这不等于证明 Windows 文件仅属主可访问。 |

关闭 Python UTF-8 模式后，还复现了测试文件隐式编码问题，以及生产代码 `TextSkill.read_file()` 在 GBK 默认编码下读取 UTF-8 技能文件失败。测试文本读写及该生产读取点改为显式 UTF-8；新增 cp1252/GBK 两参数回归，均先确认失败再修复。除此单行技能读取修复外，本轮未改变 connect 业务、锁或权限的生产行为，`secrets.py` 仅澄清现有平台语义。

验证使用仓库外 `.codex-debug/run_connect_investigation.py` 隔离真实用户配置和凭据，并核对源码导入位置。Linux 的 HOME/临时文件使用原生 `/tmp`，以真实验证 POSIX 权限；最初把临时文件放在 WSL 的 `/mnt/e` 时有 1 项 `0600` 失败，移到原生文件系统后原样 165 项全过，没有放松断言。

证据目录为 `.codex-debug/connect-investigation-20261110`：

- `baseline-comparison.json`、`before-initial.xml`、`current-initial.xml`：49 项失败的完整前后对照；相应 `*-source.json` 记录导入来源。
- `shell-path-green-01.xml`：路径回归 90 passed；`lock-green-01.xml`：锁/权限回归 75 passed。
- `encoding-native-red-01.xml` → `encoding-native-green-01.xml`：3 个文档/交互测试文件在默认编码下从 8 failed、42 passed 到 50 passed。
- `skill-regression-red.xml` → `docs-and-skill-final.xml`：技能编码回归先红；文档 77 项与技能 21 项最终全过，7 个无关脚本执行用例未选中。
- `linux-path-lock-native-01.xml`：Linux 路径/锁/权限 165 passed。

### 扩大到 connect 库测试

原先 1,667 项仅是 CLI connect 组。补充底层 `tests/unifiedllm/connect` 的 1,000 项后，Windows 默认编码的首次全组为 **2637 passed、34 skipped、2 failed**；其中 CLI 已为 **1639 passed、34 skipped、0 failed**（新增 6 项离线保护回归）。额外两项在吸收前工作树也分别复现：

- 源码依赖扫描测试默认按 GBK 读取 Python 文件；改为 UTF-8 后该文件 66 项通过。
- 符号链接测试在创建链接时遇到 Windows `WinError 1314`。拆成普通文件与符号链接两参数：普通保存、注释、CRLF 和原生权限始终验证；仅在 Windows 实测缺少创建权限时跳过链接参数。其他错误仍失败。Windows 为 16 passed、1 skipped；Linux 为 17 passed，真实符号链接及严格 `0640/0600` 均执行通过。未修改系统权限或生产实现。

Linux 完整组为 **2673 passed、0 skipped、0 failed**，其中 Windows 跳过的 34 项 Linux PTY 测试均执行。该次全组在拆分符号链接用例前启动，拆分后同文件另跑 **17 passed**；技能读取的 21 项定向回归在 Linux 也全部通过（7 个脚本执行用例未选中）。证据分别为 `linux-connect-full-native.xml`、`symlink-linux-green-01.xml`、`linux-skill-encoding-final.xml`。额外两项基线失败见 `library-source-baseline-01.xml` 和 `symlink-before-red-01.xml`。

**Windows 最终全组：2639 passed、35 skipped、0 failed，110.17 秒。** 明确使用 `python -X utf8=0`，结果不依赖强制 UTF-8 模式。35 个跳过为原有 34 项 Linux PTY 测试及实测无权限的 1 个符号链接参数；原来的 49 个失败节点均执行通过。证据为 `windows-connect-full-final.xml` 及对应 `*-source.json`。

复现最终 Windows 全组：先将 `$investigationDir` 设为保存调查脚本的实际目录，再从当前仓库根目录运行。下文 `.codex-debug/` 是历史证据目录名，不是产品安装路径。

```powershell
$connectTests = @(Get-ChildItem -LiteralPath packages\aster-cli\tests -Filter 'test_connect*.py' | Sort-Object Name | ForEach-Object { $_.FullName })
uv run --no-sync --offline python -X utf8=0 "$investigationDir/run_connect_investigation.py" --repo . --label windows-connect-rerun @connectTests tests\unifiedllm\connect -q --tb=short
```

每次复跑使用新 `--label`，避免复用已有测试临时目录；Linux 同样使用该 runner，并让临时目录位于原生 `/tmp`。

全仓库 Ruff lint/format、SPDX、`uv lock --check --offline` 及 `git diff --check` 通过；最终两个测试文件修改后重新做了对应 lint/format 检查。排查开始时的 73 个已修改/未跟踪文件经过 SHA-256 复核，除本报告外均保持原样，本轮修复落在此前干净的文件中。所有工作仍未提交或推送。

## Zed 沙箱试用环境修复（2026-11-10）

用户在 Zed 的“Aster 沙箱版 (microsandbox)”发送消息后，出现 `microsandbox initialization failed: Install aster[sandbox] to enable microsandbox execution`。Zed 指向当前仓库 `.venv/Scripts/aster-acp.exe`；此前 14 项真实 VM 验证使用的是独立环境，而日常 `.venv` 缺少 `cloudpickle` 与 `microsandbox`。直接用该环境执行 `SandboxedExecutor.run_cell("20 + 22")` 复现了截图中的同一异常，Zed 日志也记录了缺少 cloudpickle 的原因。

通过以下命令补齐日常环境，仅新增锁定的 `cloudpickle==3.1.2` 和 `microsandbox==0.7.13`，没有卸载或升级其他包，也未修改 Zed 配置、生产源码、依赖声明或锁文件：

```powershell
uv sync --project . --locked --inexact --extra sandbox --extra acp --extra tui --group dev --no-install-workspace --offline
```

同一日常环境下 `msb doctor` 确认 Windows Hypervisor Platform 可用；真实 VM 的跨 cell 状态/工具代理测试与 ACP 子进程 Linux guest 执行测试 **2 passed，17.15 秒**。ACP 测试使用确定性 FakeLLM，覆盖初始化、会话、Python 执行及关闭，没有调用用户模型。测试隔离用户配置和会话，仅复用 Zed 的 MSB/uv 缓存路径；测试后 `msb ls` 无活动沙箱。证据为 `.codex-debug/connect-investigation-20261110\zed-sandbox-fixed.xml` 和 `.codex-debug/zed-sandbox-fixed-20261110.log`。

## 第二批：有界等待队列 #560（2026-11-10）

按原实施顺序，本批吸收共享 admission 等待队列上限，并适配本地 microsandbox；Atom 会话复制恢复、输入消费事务及其他 Atom 功能不在本批范围。重新 fetch `refs/pull/560/head` 后固定头仍为 `5c4d067ee1cf6191858a9fb02c253f7215b8bf21`，包含 `8c2a93d6`、`4179a36b`、`5c4d067e` 三个提交，基线为 `fe7a849e`。GitHub API 本次遇到未认证限流，因此只确认实际 Git 补丁头，不据此更新 PR 开放/合并状态。

### 实现与兼容性

- `AdmissionControlConfig`、进程内共享控制器和 `AdmissionBroker` 增加 `max_pending`：默认 `None` 保持无上限，`0` 仅允许立即执行，正整数限定共享等待名额。并发、pacing、cooldown 和未确认预算预留导致的等待都纳入上限；已授予名额不计入 pending。
- 队列满抛出公开的 `AdmissionQueueFullError`，拒绝发生在模型请求、provider task 和调用预算消耗之前。取消或超时归还等待名额；已耗尽的 committed call-cap 仍优先报告 call-cap。最终握手超时的既有保守预算语义未变，不把“不消耗预算”泛化到所有超时。
- broker 协议升为 v4，owner/worker 必须一起重启升级并重新取得连接配置；版本不符在派发前拒绝。snapshot 与 tracing 公开队列上限，harness 新增 `llm_queue.full_rejections` 指标。
- 保留第一批 #563 的 `AdmissionError` 基类传播。在 CodeAct/CodeActV2 的嵌套 cell、pre-ellipsis 和 prefill plugin 中，未捕获的 queue-full 在保留失败输出/trace 后终止；普通执行异常保持原有恢复行为。cell 主动捕获 queue-full 时可自行恢复。
- 沿用 microsandbox 的 guest → host msgpack 通道。每个 cell 由宿主生成独立 HMAC 密钥，只有真实宿主工具拒绝才签名；guest 可捕获或重抛该错误，宿主验证后才恢复终止类型。伪造类型、改写消息/token、旧 cell token 均不提升为可信 admission 错误，密钥不放入 guest namespace，也不维护无界异常对象表。
- 补充本地路径适配：错误消息恰为绝对路径时，双向路径转换曾会改写签名字节。认证错误元数据现在保持原文，普通工具参数和返回值继续使用路径映射；使用实际收发帧验证 Windows 正斜杠路径、guest-looking 路径及 Linux 路径。

这是一项可选 API 能力，本批没有自动给现有 Zed 模型或会话设置排队上限，也没有修改模型配置或凭据。使用示例见 [有界等待队列](concepts/llm-admission-control.md#bounded-pending-queues)。

### 先失败再修复与平台测试调整

- 未实现时 `max_pending=0` 因未知参数失败；实现后最小拒绝/恢复测试通过。
- 在只有 admission 核心、尚未修改策略时，真实嵌套调用的 3 个确定性回归未抛 `AdmissionQueueFullError`；增加 cell/prefill 传播后通过。队列满统计也先验证缺失，再验证计数与 span 属性。
- 沙箱错误认证先验证接口缺失；路径转换边界另有 3 项签名失效回归，修复后通过。
- 一个已有 broker 停止后连接测试的 0.5 秒 deadline 在 Windows 会先触发 timeout；吸收前基线也复现。实测本机 TCP refusal 约 2.026 秒，测试 deadline 改为 5 秒，仍严格要求 `AdmissionUnavailableError`，生产行为未改。
- 上游 FIFO 测试并发启动连接，却按任务创建顺序等待并释放 permit；Linux 上连接到达乱序会导致测试自己扣住 permit。定向 2 失败和延迟首连接的因果探针确认后，测试改为逐一确认真实入队顺序，超量连接仍并发，10 秒期限不变；Windows/Linux 各 4 项通过。首次 Linux 运行期间测试源文件正在更新，导致 inspect 源码与已加载函数错位；冻结源码后 48 项嵌套用例完整通过。

### 验证与工作区保护

所有测试使用现有环境、`uv run --no-sync --offline` 和隔离用户目录。真实 VM 使用已经补齐 sandbox extra 的日常 `.venv`，不会发真实模型 API 请求。测试证据存于 `.codex-debug/connect-investigation-20261110`，日志存于 `.codex-debug/upstream-batch2-20261110`。

已完成的补充验证包括 CodeAct/prefill/cancel/反馈/指标 **308 passed**，sandbox 全部非 VM 用例及 ACP 事件/进度 **247 passed**；均与最终核心测试部分重叠，不累计为独立用例总数。

最终 admission/broker/pacing/CodeAct 错误传播/指标核心组：**Windows 559 passed，62 deselected，56.37 秒；Linux 559 passed，62 deselected，88.14 秒**。62 个未选用例为 sandbox 标记，真实 VM 另组验证。两侧源码冻结后运行，覆盖上面的 FIFO 测试修正；证据为 `batch2-windows-final.xml`、`batch2-linux-final.xml` 及对应 `*-source.json`。

**Windows WHP 真实 microVM 最终组：71 passed，239 deselected，687.46 秒**，包括 9 项新增错误认证/伪造/捕获重抛用例、48 项嵌套调用矩阵和第一批 #563 的 14 项错误传播回归。VM 内通过 Linux 平台值与宿主专属随机环境标记缺失确认隔离执行，不依赖跨 VM PID 必须不同的错误假设。证据为 `batch2-microvm.xml` 与 `batch2-microvm-source.json`；测试结束后 `uv run --no-sync --offline msb ls` 返回 `No sandboxes found.`。

全仓库 Ruff lint/format、SPDX、`uv lock --check --offline` 及 `git diff --check` 均通过；最后的 FIFO 测试修改另行通过 lint/format 复核。上面的 308 项与 247 项补充回归证据分别为 `batch2-codeact-regression.xml`、`batch2-sandbox-unit.xml`。第二批验证中发现的失败均已定位并解决，最终各组没有失败或错误。

开始前保存 86 个既有已修改/未跟踪文件的 SHA-256。收尾复核确认其中 82 个完全一致、无文件丢失，仅在 `codeact.py`、admission 文档和 pacing 端到端测试上叠加本批差异，并更新本报告；其余 TUI、ACP、streaming 及第一批实现保持原样。依赖声明和锁文件与本批基线完全一致，工作区仍未提交或推送。

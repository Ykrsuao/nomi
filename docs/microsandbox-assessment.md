# NOOA 沙箱替换与原生 Windows 支持

更新日期：2026-11-07。实现保留 `CodeActConfig.execution_backend` 的 `inprocess` / `sandbox` 两种使用方式。

## 实现与运行

- `sandbox` 现在只创建 microsandbox microVM，运行一个长期 Python interpreter；旧的 host fork/Pipe 执行器不再启动，也没有回退路径。
- `inprocess` 原生支持 Windows，无需 WSL、WHP 或平台跳过环境变量。Windows shell 使用 PowerShell，Linux/macOS 继续使用 bash。
- 框架导入和 CLI 启动的 Windows 拦截已移除；SQLite/模型密钥文件锁和 `monitor()` 的 Windows 进程清理已适配。
- 沙箱依赖为 `microsandbox>=0.7.13,<0.8` 与 cloudpickle，使用 `uv sync --extra sandbox` 安装；关闭沙箱时不初始化 SDK。
- ACP 支持 `--sandbox` / `--no-sandbox` 和 `NOOA_SANDBOX`；每个 ACP session 使用独立策略，其沙箱 workspace 设置为该 session 的仓库根目录。显式 `--no-sandbox` 覆盖环境设置，默认使用原生执行。

在仓库根目录启用 ACP 沙箱：

```powershell
uv run --project . --extra acp --extra sandbox nooa-acp --model my-alias --sandbox
```

ACP 的 stdio 通信、模型请求、workspace skill 导入与 `self.*` 工具仍在宿主执行；生成的 Python 直接操作进入 microVM，运行中取消和关闭负责回收 VM。

默认 guest image 使用宿主的精确 Python 版本。首次启用会在不运行用户代码的准备 VM 中用 uv 安装匹配的依赖，缓存磁盘快照；执行 VM 再以配置的网络权限启动。已预装依赖的镜像可以设置 `SandboxConfig(image=..., prepare_environment=False)`。

microsandbox 的 guest kernel 未提供 Landlock，因此文件白名单由 guest 的私有 mount namespace、只读 bind mounts、chroot 和权限降级实现。网络由 VM 策略及 guest 的 socket filter 约束。CPU/内存由不可提高的 guest rlimits 和资源观察约束，超时或异常清理会销毁整个 VM。

Windows 授权 workspace 在 guest 中使用 `/workspace`，额外 allow 路径使用 `/allowed/N`；参数、结果和工具调用中的授权路径在宿主与 guest 间转换。Python 状态、函数、`self.*` 同步/异步代理、结构化结果、异常、输出/媒体限额、恢复与清理均由新适配层承担。

现有 `self.*` 工具仍按框架语义在宿主执行；这些工具动作的授权由工具接口负责。microVM 限制约束 guest Python 的直接操作。

## 已执行验证

- 原生 Windows：23 项导入、CLI、inprocess、shell/原始 stdin、SQLite 和配置测试通过；默认环境的 `windows_smoke.py` 通过。
- 真实 WHP microVM：状态、CodeAct、异步延迟调用、路径、精确文件授权、网络、输出截断、超时/恢复/关闭、CPU/内存测试已通过；原有六项 CodeAct sandbox 端到端测试通过。
- Linux 回归：72 项 shell/存储/进程测试和 260 项 CodeAct 测试通过。
- ACP：94 项普通回归测试通过，3 项既有 xfail；另有 2 项真实 stdio + WHP microVM 测试通过，覆盖 Linux guest 内执行 Python、运行中取消和 session 关闭。模型响应使用 FakeLLM；测试后 `msb ls` 确认无残留沙箱。
- 新核心模块的类型检查及 Ruff 检查已通过。
- 本机未实际运行 macOS；macOS 保留原 host 路径，并通过同一 microsandbox adapter 使用 Linux guest。

## macOS CI 与实际 VM 验证

当前正式支持范围为 Linux 和原生 Windows，macOS 暂不支持。保留现有兼容代码，以及手动 `workflow_dispatch` 的 `macos_native` 选项：在 `macos-15` 的 Apple Silicon 运行机上分别验证 Python 3.12 和 3.13 的原生 CodeAct、shell、存储与 ACP。该检查不在普通 push 或 pull request 中自动运行。

[GitHub 托管 macOS 运行机不支持嵌套虚拟化](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。初次实测中 SDK 安装和 doctor 检查通过，但创建 VM 失败于 `VmSetup(VmCreate)`，尚未进入 NOOA guest。真实 VM 和 ACP 沙箱验证保留为手动 `workflow_dispatch` 的 `macos_sandbox` 选项，要求仓库已有能创建 VM 的 `[self-hosted, macOS, ARM64]` 运行机；没有运行机时保持未验证状态。

下面保留迁移验收要求和初始源码依据。

## 已明确的目标

- **完整替换 NOOA 内置沙箱。** 启用 sandbox 后，由 Microsandbox 承担隔离和执行，新适配层承接原沙箱的全部框架职责。旧 `SandboxedExecutor`、fork/Pipe 和 guards 隔离路径不再执行，也不能作为失败回退。
- **关闭沙箱也正式支持原生 Windows。** 删除框架导入和 CLI 启动时拒绝 Windows 的平台检查，取消对 `NOOA_SKIP_PLATFORM_CHECK` 的依赖；`inprocess` 不依赖 WSL、WHP 或 Microsandbox 初始化。操作系统识别用于选择匹配平台的实现。
- Microsandbox 的 Linux、macOS、Windows SDK/runtime 作为现成依赖直接使用；本次不改造其跨平台实现。把整套 NOOA 外包进 VM，或只隔离 shell 命令，都不满足上述完整替换目标。

当前 NOOA 默认 `inprocess`，可选 `sandbox`；CodeAct 硬编码创建旧执行器，再由 actor 分派 `run_cell()`。[N1][N5] 此开关、生命周期与分派需要进入新后端，同时保留框架的 cell 语义。

## 新后端必须承接的职责

| 原沙箱职责 | 迁移要求 |
|---|---|
| 每 session 的持久 REPL、模块/参数初始化、async cell、显式和末尾表达式返回。[N2][N3][N7] | 在 guest 中运行长期 Python worker，初始化不能再依赖宿主 fork 继承 live Agent、CurrentCall 和闭包。 |
| `self` 同步/异步工具、属性读写、nested proxy、模块可变状态的只读语义。[N4][N8] | 保留 Agent 接口与状态语义，以明确协议提供对象描述、数据传输和受限 broker。 |
| `return_result` 控制信号、结构化结果/异常、stdout/stderr、诊断截断与媒体。[N4][N7][N9] | 完整还原 `ExecutionResult`；guest 上行数据仍须安全解码，不能让宿主直接 unpickle 不可信内容。 |
| cell 硬超时、broker 独立超时、CPU/内存限制、取消。[N2][N10] | broker 等待不能消耗 cell deadline；内存 headroom 和 CPU-seconds 不能简单换成 VM 总内存或 vCPU 数，配置须有准确映射。 |
| workspace、额外 allow/read-only 授权、system paths、network 开关。[N3] | 使用 runtime 的挂载、网络和资源接口；明确宿主/guest 路径及编码，不能照搬 Linux 宿主路径或静默扩大权限。 |
| 崩溃/超限硬终止、`restart_empty`/`disabled`、非阻塞清理。[N2][N10] | 恢复清空 namespace 或禁用会话的行为必须保留，清理不得阻塞宿主事件循环或遗漏进程。 |
| 配置合并、能力失败、真实约束 context block、验证/middleware/hooks/tracing。[N3][N5][N8] | 旧配置的保留、改义或移除需明确；runtime 无法启动或兑现限制时报告错误，禁止回到旧沙箱或自动改为 inprocess。 |

Microsandbox `exec()` 每次启动新的 guest 执行进程；反复 `exec("python", ["-c", code])` **不保留 Python 变量**。[M4] 现成 `exec_stream()`、可写 stdin 和输出事件可承载 NOOA 的长期 worker 协议；这属于 NOOA 后端适配，不是 Microsandbox 的跨平台改造。[M5]

现有 `self.*` broker 在宿主执行工具。[N4] 新后端必须明确这些动作的权限边界：让 guest 请求宿主任意执行 shell 或读写文件，会绕过 VM 的相应约束。需要隔离的动作应在 guest 执行，宿主 broker 只提供明确授权的能力。[M6]

## 原生 Windows 不启用沙箱的独立验收

当前 NOOA 和 CLI 都有 Windows 拒绝策略；shell、SQLite 会话锁和 `monitor()` 还依赖 POSIX。[W1–W4] 既有 Windows smoke 设置 override，只能证明部分核心代码可运行，不能证明完整原生支持。[W5] 下列范围必须独立验收：

- **导入与 CLI：** 移除主动拒绝 Windows 的启动检查；不设置 override 正常 import、运行 CLI、模型配置和 Predict；关闭 sandbox 时不加载或启动 Microsandbox，也不要求 WHP。
- **inprocess CodeAct：** 多 cell 持久状态、工具调用、结构化返回、异常和取消在原生 Windows 正常工作。
- **shell：** 可用的本机 shell、进程启动、控制 pipe/句柄、stdout/stderr/退出状态、取消及整棵子进程树清理；不能继续依赖 `/bin/bash`、`pass_fds`、`killpg`。[W2]
- **存储与 monitor：** 文件型 SQLite 锁的竞争、释放、重开和删除；`monitor()` 输出与取消清理；不能只验内存 SQLite。[W3][W4]
- **路径与测试：** Windows 路径、空格/Unicode、编码和句柄资源；将拒绝策略/override smoke 改为原生 Windows 正向测试，并保持 Linux/macOS 回归覆盖。[W5]

启用 sandbox 的三端 runtime 集成验证与上述 Windows inprocess 验证是两个独立验收组。任何一组未通过，均不能报告完整替换和完整 Windows 支持已完成。

## 实现接口核对

以下为当前 Python 0.7.13 SDK 的可调用接口核对，未运行 VM。新后端可通过 guest 持久 Python worker 复用 `cell_core` 和 `wire.Codec`，重建 `self`/框架能力的 broker；不调用旧 fork worker 或 `install_guards()`。

| 工作 | 可用接口 |
|---|---|
| 创建和挂载 | `await Sandbox.create(name, image=..., memory=512, cpus=1, volumes={"/workspace": Volume.bind(host_absolute_path, readonly=False)}, network=Network.none())`；memory 单位为 MiB，挂载字典的 key 属于 guest。[I1] |
| 文件和启动脚本 | `await sandbox.fs.mkdir(guest_dir)`、`fs.write(guest_file, bytes)`、`fs.copy_from_host(host_file, guest_file)`；目录用 `Patch.copy_dir()` 或只读 bind。`Patch.file()`/`Patch.text()` 可在 create 时通过 `patches=` 写入 rootfs。[I2] |
| 长期进程和下行 | `handle = await sandbox.exec_stream("python", ["-u", worker_path], stdin=Stdin.pipe(), tty=False, timeout=None)`；`sink = handle.take_stdin()` 只能获取一次，随后 `await sink.write(bytes)`/`close()`。[I3] |
| 上行和终止 | `await handle.recv()` 或 `async for event in handle`；检查 `event.event_type`、`data`、`code`、`pid`。`await handle.signal(guest_linux_signal_number)`/`kill()` 与接收独立；不能并发 `recv()` 和 `wait()`/`collect()` 消费同一流。[I3] |
| 进程资源 | `rlimits=[Rlimit.cpu(seconds), Rlimit.as_(soft=absolute_bytes, hard=absolute_bytes)]`，必须传 SDK `Rlimit` 实例；限制在 guest exec 前应用。[I4] |

通信必须有长度边界和独立接收泵，Exec stdout chunk 不等于消息。可信宿主到 guest 可以用 cloudpickle；guest 到宿主只用受限 msgpack/`Codec`，包括结果中的嵌套载荷。启动最早可复制 stdout fd 留给协议，再把普通 stdout fd 重定向到 stderr，避免 `os.write(1)` 和导入输出污染协议；stderr 也须限制累计大小。

bootstrap 只传经过可见性筛选的 namespace 数据、类型和能力描述；不要直接序列化整个 Agent、CurrentCall 或捕获客户端/锁的闭包。`self`、`return_result`、`_call`、`show` 等能力须在 guest 重建。cloudpickle 默认对可导入模块按引用保存；动态定义须明确按值传输或部署模块，并保持两端 Python 精确版本、cloudpickle 和数据类型依赖一致。不能静默丢弃无法迁移的 namespace 值。初始审计版本的 NOOA 未声明 cloudpickle 依赖；仅复制源码或 rootfs patch 不会安装 NOOA 的依赖，Windows `.pyd`/宿主 venv 也不能直接充当 Linux guest 依赖。[I5][I8]

`Rlimit.cpu()` 是长期进程的累计 CPU 时间，SDK `Rlimit.as_()` 是绝对地址空间字节数。要保留原来的“初始化后 VmSize + headroom”，可先让可信 bootstrap 报告 guest PID/VmSize，再通过 runtime exec 的 guest helper 使用 `resource.prlimit()` 设置该进程的最终限制后才接收 cell；不能把 VM memory 指标当作 worker VmSize。受限制的 worker 应使用非 root guest user，避免生成代码提高硬限制；root 仅做可信准备。`RESTRICTED` 本身只移除 mount-admin 能力，不等于移除所有资源管理权限。[N10][I4][I6]

runtime 的进程组 kill 不等于保证回收主动创建独立会话的所有进程；硬失败/会话关闭要终止并销毁对应 VM。SDK context manager 的退出会尝试 kill/remove，但会忽略清理错误，可靠 `aclose()` 需要显式确认和记录清理结果。[I3][I7]

## 源码依据

- [N1] [strategy_config.py](../src/nooa/config/strategy_config.py)，L109–117：执行后端和默认值。
- [N2] [executor.py](../src/nooa/runtime/sandbox/executor.py)，L3–9、L74–84、L253–354：持久 worker、能力要求、结果和 deadline。
- [N3] [config.py](../src/nooa/runtime/sandbox/config.py)，L36–123、L225–266；[executor.py](../src/nooa/runtime/sandbox/executor.py)，L172–187：配置、路径和 fork 初始化。
- [N4] [worker.py](../src/nooa/runtime/sandbox/worker.py)，L52–127、L179–336、L357、L431；[executor.py](../src/nooa/runtime/sandbox/executor.py)，L441；[wire.py](../src/nooa/runtime/sandbox/wire.py)，L3–11：代理、namespace 和安全上行协议。
- [N5] [actor.py](../src/nooa/runtime/actor.py)，L1630–1645；[codeact.py](../src/nooa/strategies/codeact.py)，L839–868、L939–940：分派、hooks 和生命周期。
- [N7] [cell_core.py](../src/nooa/runtime/sandbox/cell_core.py)，L3–9、L75–120、L135–152：cell 和返回语义。
- [N8] [readonly.py](../src/nooa/runtime/sandbox/readonly.py)，L3–16、L159；[context_block.py](../src/nooa/runtime/sandbox/context_block.py)，L15：只读模块状态和约束说明。
- [N9] [serialization.py](../src/nooa/runtime/sandbox/serialization.py)，L99–179、L313：结果/异常/信号 DTO。
- [N10] [executor.py](../src/nooa/runtime/sandbox/executor.py)，L217–246、L348–398；[guards.py](../src/nooa/runtime/sandbox/guards.py)，L207–229：非阻塞终止、broker deadline 和资源语义。
- [W1] [_platform_policy.py 基线快照](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fe7a849e791019c2db8b5b69683eea78f0c85f68/src/nooa/_platform_policy.py#L57)；[_platform.py](../src/nooa/_platform.py)，L3–14；[CLI __init__.py](../packages/nooa-cli/src/nooa_cli/__init__.py)，L80–99、L128：迁移前拒绝策略。
- [W2] [_bash_session.py](../src/nooa/tools/_bash_session.py)，L224–244、L799–817、L863–867：POSIX shell、控制 pipe、进程组。
- [W3] [sqlite.py](../src/nooa/storage/sqlite.py)，L672–727：文件型会话锁。
- [W4] [producers.py](../src/nooa/runtime/producers.py)，L74–117：monitor 子进程与清理。
- [W5] [test_windows_platform.py](../tests/test_windows_platform.py)，L33–48；[windows_smoke.py](../scripts/windows_smoke.py)，L5、L23；[ci.yml](../.github/workflows/ci.yml)，L244–258：当前拒绝测试和 override smoke。
- [M1] [Cargo.toml](../../microsandbox/Cargo.toml)，L55：初始核对版本。
- [M4] [execution.mdx](../../microsandbox/docs/sdk/python/execution.mdx)，L45；[agent.rs](../../microsandbox/crates/agentd/lib/agent.rs)，L2897；[session.rs](../../microsandbox/crates/agentd/lib/session.rs)，L1268、L1402：exec 进程模型。
- [M5] [execution.mdx](../../microsandbox/docs/sdk/python/execution.mdx)，L151–182；[_microsandbox.pyi](../../microsandbox/sdk/python/microsandbox/_microsandbox.pyi)，L675–690：长期 worker 所需 I/O。
- [M6] [filesystem.mdx](../../microsandbox/docs/security/filesystem.mdx)，L7–16、L40–44；[isolation.mdx](../../microsandbox/docs/security/isolation.mdx)，L44–54：VM 文件权限和宿主控制通道。
- [I1] [_microsandbox.pyi](../../microsandbox/sdk/python/microsandbox/_microsandbox.pyi)，L174–211、L856–871；[types.py](../../microsandbox/sdk/python/microsandbox/types.py)，L1882–1893：创建、bind 和网络工厂。
- [I2] [fs.rs](../../microsandbox/sdk/python/src/fs.rs)，L84–97、L207–222；[types.py](../../microsandbox/sdk/python/microsandbox/types.py)，L1088–1150：文件 bytes、单文件复制和 patches。
- [I3] [exec.rs](../../microsandbox/sdk/python/src/exec.rs)，L130–180、L221–237；[session.rs](../../microsandbox/crates/agentd/lib/session.rs)，L977–989：stdin、事件消费锁、独立 control 和 guest 进程组信号。
- [I4] [types.py](../../microsandbox/sdk/python/microsandbox/types.py)，L565–582；[sandbox.rs](../../microsandbox/sdk/python/src/sandbox.rs)，L2179–2190；[session.rs](../../microsandbox/crates/agentd/lib/session.rs)，L1290–1317：CPU/AS 资源工厂、精确类型检查与 exec 前限制。
- [I5] [pyproject.toml](../pyproject.toml)，L14–53；[__init__.py](../src/nooa/__init__.py)，L30–44：Python/依赖要求和导入链。
- [I6] [isolation.mdx](../../microsandbox/docs/security/isolation.mdx)，L61–64；[session.rs](../../microsandbox/crates/agentd/lib/session.rs)，L1495–1545：guest 默认身份与 restricted capabilities。
- [I7] [sandbox.rs](../../microsandbox/sdk/python/src/sandbox.rs)，L1488–1508：context manager 的退出和清理错误处理。
- [I8] [cloudpickle 官方 README](https://github.com/cloudpipe/cloudpickle/blob/master/README.md)：可信来源、Python 精确版本、按值/按引用和 import 依赖限制。

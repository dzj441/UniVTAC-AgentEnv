# Agent Run Viewer

`Agent Run Viewer` 是一个零前端依赖、只读的 interaction trace 回放器。它直接读取
`agent_runs` 中现有的 JSON/JSONL、PNG 和 H.264 文件，不复制数据，也不引入第二套
日志格式。

## 启动

在仓库根目录运行：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/run_agent_viewer.py --port 8765
```

默认行为：

- 监听 `0.0.0.0:8765`，从而能被 code-server 的端口代理访问；
- 读取 `<repo>/agent_runs`；可用 `--runs-root PATH` 覆盖；
- 不扫描同级的 `<repo>/agent_runs_legacy`。历史实验可移动到该 ignored archive，既保留
  原始结果，又不会继续出现在默认前端列表；
- 如果端口设为 `0`，由系统选择一个空闲端口；
- 将环境变量 `VSCODE_PROXY_URI` 中的 `{{port}}` 替换为实际端口，并打印完整可点击
  URL；
- 同时打印 `http://127.0.0.1:<port>/` 作为本机诊断地址。

例如，本机变量为：

```text
https://host/.../proxy/{{port}}/
```

启动在 8765 后会输出：

```text
Code-server URL: https://host/.../proxy/8765/
Local URL: http://127.0.0.1:8765/
```

浏览器端的脚本、样式、API、图片和视频全部使用相对 URL，因此 code-server 前面的
任意长路径前缀不会被错误地丢掉。服务端也接受带代理前缀的路由。视频端点实现了
HTTP Range，可以在 VS Code/code-server 浏览器中拖动 H.264 视频进度。

## 页面如何组织一次交互

主时间线采用聊天结构：环境在左，Agent 在右。每一个 round 严格按发生顺序显示：

1. Agent 基于上一条 ENV 气泡返回的 `observation_id`；
2. 按 App Server 时间戳展示这次机器人调用前发生的全部可观察 Agent 活动，包括公开
   reasoning summary、agent message、shell 命令与输出、view image、文件修改、MCP、
   子 Agent、网络/搜索和 context compaction；
3. 展示实际发送给仿真或宿主感知服务的工具名、参数和 raw request；
4. 左侧 ENV 气泡展示仿真反馈或 SAM/UniDepth 派生图；物理动作还展示实际
   gripper/EE 状态变化和新 observation；
5. 下一轮 Agent 气泡明确标注它基于这个新 observation 行动。

宿主是否成功接收/转发工具调用与仿真器是否真正执行动作是两个独立状态。Viewer 会
分别显示 `host accepted` 和 `execution_succeeded`，在侧栏、指标区和 step rail 中单独
标出 simulator execution failure，避免把成功返回的工具结果误读成成功执行的物理动作。

页面提供两种查看方式：

- `完整对话`：连续展示完整 episode，不重复相邻 round 共享的 observation；
- `单步聚焦`：补充显示当前动作的输入 observation，再依次显示 Agent 决策与动作后
  的 ENV 结果，适合逐动作检查。

Head、wrist、左右 tactile 图按 Level/Profile 实际开放的模态显示。v1 记录还会显示
米制 depth 预览、匿名 BBox/Mask，并可展开相机内外参与冻结后的机器人状态。点击任意图
可进入大图模式；键盘左右键可切换动作或大图。所有 raw JSON 默认折叠，但随时可展开。

## 使用哪些落盘文件

对正式 Codex run，后端按 `call_id` 和时间戳关联：

- `codex_app_server_events.jsonl`：Viewer 的主要 Agent activity 数据源；从中读取公开
  reasoning summary、agent message、命令/输出、文件、MCP、子 Agent、网络与图像事件；
- `codex_messages.jsonl`：旧 run 缺少原始 App Server stream 时的兼容 fallback；
- `codex_decisions.jsonl`：仅旧 schema run 可能存在的历史审计产物；Viewer 不读取其
  rationale 来构造 Agent 思考过程；
- `codex_tool_calls.jsonl`：实际工具参数、仿真命令、反馈和 observation 转移；
- `semantic_perception/`：SAM mask/overlay/contact sheet 与 UniDepth depth/confidence
  产物；它们是当前公开 RGB 的派生结果，不是新的 simulator observation；
- `observations/obs_*/`：Agent 在各轮收到的公开传感器图；
- `codex_run_manifest.json`、`codex_run_outcome.json` 与
  `evaluator_outcome.json`：能力、模型、审计和终局结果；
- `agent_observations_h264.mp4`：完整公开 observation 回放。
- `sim_step_composite_h264.mp4`：evaluator-private 的连续仿真动作窗口回放；页面可在两条
  视频间切换，详细时间语义见
  [`SimulatorStepWindowRecorder.md`](SimulatorStepWindowRecorder.md)。

对只有 `agent_transcript.jsonl` 的标准化环境采集，viewer 仍会显示命令、环境反馈和
传感器变化，并清楚标记“没有 Codex App Server activity stream”，不会伪造 Agent 活动。
在首次工具调用前失败、但已写出 `codex_run_outcome.json` 的 Codex run 也会保留在列表中，
并明确标记为不可计分的 invalid run，而不会伪装成 sensor capture。

## “Agent 想了什么”的边界

Viewer 展示 App Server 实际发布和执行的完整可观察事件序列，包括 reasoning summary、
agent message、shell、代码/文件修改、MCP、子 Agent、网络/搜索、图像查看、context
compaction、机器人 tool call 及其结果。它不依赖 Agent 在工具参数中另填 rationale 或
`decision_record`；generic v1 已移除这些强制字段，Viewer 无需兼容性改造。

这里的“完整”严格限定为完整的**可观察事件流**。模型没有通过 App Server 发布的隐藏
chain-of-thought 只有 token 计数，没有可供 Viewer 恢复的文本；页面不会推断或伪造它。
为防止超大 shell 输出拖垮页面，UI payload 会对单个文本字段提供带明确标记的有界预览，
append-only 原始 App Server JSONL 保持不变。

## 只读与路径安全

- API 仅发现配置的 `runs-root` 下含 Codex tool stream 或 AgentEnv transcript 的目录；
- artifact 路径在解析后必须仍位于所选 run 内，`..` 和指向外部的符号链接会被拒绝；
- 即使文件位于 run 内，也只有标准化 UI payload 明确引用的 observation、语义派生图、
  视频和公开报告能通过 artifact API 读取；`.host_sensor_metadata` 等宿主私有侧车、
  simulator audit 与未引用日志均不可由前端按路径探测；
- 页面没有修改、删除或触发仿真的接口；刷新只会重新读取 append-only 记录；
- 服务本身不实现账户登录，远程使用时依赖 code-server 代理的访问控制。不要把端口直接
  暴露到不可信网络。

## 验收

```bash
./scripts/accept_agent_env.sh
node --check agent_env/viewer_static/app.js
```

静态 suite 覆盖 run 发现、Codex 三流关联、capture-only 兼容、路径逃逸拒绝、代理路径
前缀以及 MP4 Range 请求。启动后还可以检查：

```bash
curl http://127.0.0.1:8765/api/health
curl http://127.0.0.1:8765/api/runs
```

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
2. 展示这次调用前发布的 Codex reasoning summary；
3. 展示宿主校验后的结构化 `decision_record`：观察证据、备选方案、不确定度、预期
   效果，以及具体动作幅度的理由；
4. 展示实际发送给仿真或宿主感知服务的工具名、参数和 raw request；
5. 左侧 ENV 气泡展示仿真反馈或 SAM/UniDepth 派生图；物理动作还展示实际
   gripper/EE 状态变化和新 observation；
6. 下一轮 Agent 气泡明确标注它基于这个新 observation 决策。

页面提供两种查看方式：

- `完整对话`：连续展示完整 episode，不重复相邻 round 共享的 observation；
- `单步聚焦`：补充显示当前动作的输入 observation，再依次显示 Agent 决策与动作后
  的 ENV 结果，适合逐动作检查。

Head、wrist、左右 tactile 图按 Level 实际开放的模态显示。点击任意图可进入大图模式；
键盘左右键可切换动作或大图。基础机器人状态和所有 raw JSON 默认折叠，但随时可展开。

## 使用哪些落盘文件

对正式 Codex run，后端按 `call_id` 和时间戳关联：

- `codex_messages.jsonl`：Codex 协议公开的 reasoning summary 与 agent message；
- `codex_decisions.jsonl`：通过宿主能力网关校验的显式 decision record；
- `codex_tool_calls.jsonl`：实际工具参数、仿真命令、反馈和 observation 转移；
- `semantic_perception/`：SAM mask/overlay/contact sheet 与 UniDepth depth/confidence
  产物；它们是当前公开 RGB 的派生结果，不是新的 simulator observation；
- `observations/obs_*/`：Agent 在各轮收到的公开传感器图；
- `codex_run_manifest.json`、`codex_run_outcome.json` 与
  `evaluator_outcome.json`：能力、模型、审计和终局结果；
- `agent_observations_h264.mp4`：完整公开 observation 回放。

对只有 `agent_transcript.jsonl` 的标准化环境采集，viewer 仍会显示命令、环境反馈和
传感器变化，并清楚标记“没有 Codex decision stream”，不会伪造 Agent reasoning。

## “Agent 想了什么”的边界

Viewer 能完整显示 Codex 协议发布的 reasoning summary，以及 benchmark 工具 schema 强制
Agent 填写的显式 decision record。这包括用于复盘的证据、替代方案、不确定度、参数
理由和预期效果。它不包含也不声称包含模型未通过协议公开的隐藏 chain-of-thought。

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

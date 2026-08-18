# Agentic Embodied Benchmark v1

v1 是当前通用 benchmark 接口。它把“Agent 能看到什么”与“Agent 如何控制机器人”分开：

- 六个 Observation Profile 是主榜能力轴；
- 匿名 BBox 与 Mask 是两个可独立开关的诊断轴；
- 所有条件共享同一个世界坐标系 EEF 动作接口；
- 控制过程中不返回 reward 或 task success；
- `finish_episode` 后才执行终局评测并公开 success。

旧的 `grasp_classify` 三 level 接口保留为 v0 兼容路径，但不是 v1 主榜定义。

## 六个 Observation Profile

所有 Profile 都包含冻结字段的基础机器人状态：

- `joint_position_9d`
- `joint_velocity_9d`
- `gripper_width_m`
- `end_effector_pose_robot_base_wxyz_7d`

| Profile | 公开观测 |
|---|---|
| P1 `head_state` | Head RGB + state |
| P2 `head_wrist_state` | P1 + Wrist RGB |
| P3 `head_tactile_state` | P1 + 左右 tactile RGB |
| P4 `head_wrist_tactile_state` | Head/Wrist RGB + 左右 tactile RGB + state |
| P5 `head_wrist_tactile_depth_intrinsics_state` | P4 + Head/Wrist 米制 depth + 两相机内参 |
| P6 `head_wrist_tactile_depth_intrinsics_extrinsics_state` | P5 + 动态更新的两相机外参 |

Depth 同时落盘为无损 `float32` 米制 `.npy`、valid mask 和带明确 near/far 米制范围的
PNG 预览。相机外参是 `T_robot_base_camera_ros` 4×4 矩阵，ROS optical 约定为
`+Z forward, -Y up`。Codex 适配器把可视 PNG 作为图像输入，并在结构化响应中保留米制
artifact、统计、内参与外参；其他模型适配器可直接消费无损数组。

## 匿名标注诊断轴

命令行的两个开关完全独立，可组成四种条件：

```text
none
--provide-bbox
--provide-mask
--provide-bbox --provide-mask
```

每个公开相机按需返回两个匿名角色：

- `manipulated_object`
- `goal_fixture`

BBox 使用 `[x1, y1, x2, y2]`、右下角 exclusive 的像素坐标；Mask 是同分辨率二值
PNG。物体不在当前视野时会返回 `visible=false`、空 BBox/Mask，而不会伪造位置。
Isaac 的 raw instance ID、label、actor 名和 USD prim path 只在 host 内存中用于映射，
终局前不会写入公开记录。SAM 3/UniDepth V2 不参与该 oracle 轴。

## 任务

| Task | 指令 | 初始状态 | 终局策略 |
|---|---|---|---|
| `pull_out_key` | Grasp the key and pull it completely out of the slot. | 默认机器人 home pose，key 未抓取 | 使用原任务 checker；拔出阈值仍需后续视频校准 |
| `put_bottle_in_shelf` | Pick up the bottle from the table, place it upright inside the shelf, and release it. | 默认机器人 home pose，bottle 未抓取 | 原位置/姿态 checker + 已松爪 + 60 physics steps 后稳定 |

v1 默认关闭 task-specific `pre_move()`，因此 Agent 必须自行定位、接近并抓取物体。为复现
旧的抓取后评测，可显式添加 `--pre-move`；该模式会原样调用上游任务的 privileged
`pre_move()`，并在 manifest 中记为 `start_condition=pregrasped`。默认模式记为
`start_condition=ungrasped`。跳过 `pre_move()` 时只建立旧 checker 必需的 host-private
参考状态，不移动机器人或物体。

Bottle 的 release 阈值为 gripper qpos `>= 0.0175 m`；稳定要求 60 steps 前后物体平移
不超过 `0.01 m`、旋转不超过 `10°`。三项必须同时成立。

## Agent 工具与动作边界

模型侧只注册三个 dynamic tools：

1. `start_episode`
2. `step_eef`
3. `finish_episode`

`step_eef` 使用世界坐标系 XYZ/RPY 增量和夹爪增量：

| 参数 | 边界 |
|---|---|
| 每轴平移 | `abs <= 0.04 m` |
| 平移向量范数 | `<= 0.06 m` |
| 每轴 RPY | `abs <= 0.35 rad` |
| Gripper delta | `abs <= 0.005 m` |
| Accepted `step_eef` 数量 | 最多 50 |

全零动作合法，可用于等待物理稳定。每个 step 必须引用最新 `observation_id`；陈旧 ID、
未知字段、非有限数和越界动作在改变世界之前被拒绝。接口不返回目标方向、目标半空间、
语义动作提示、actor pose、IK/joint target 或过程 task success。

## 启动环境

从仓库根目录启动一条 P6 + BBox + Mask episode：

```bash
./scripts/launch_agent_env.sh \
  --task pull_out_key \
  --profile 6 \
  --provide-bbox \
  --provide-mask \
  --device cuda:0 \
  --run-dir agent_runs/example_pull_key_p6
```

上例使用默认的未抓取初态。旧的 pregrasped 条件需显式增加：

```bash
./scripts/launch_agent_env.sh \
  --task pull_out_key --profile 6 --pre-move --device cuda:0
```

标准输入/输出是 newline-delimited JSON，响应以 `AGENT_ENV_RESULT ` 开头。模型可见命令
顺序是 `start -> step* -> finish`；`close` 仅由 host 在终局后清理进程，不在 agent tool
schema 中。

## 启动隔离 Codex Agent

先用 dry-run 审查完整 prompt-independent capability manifest：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/run_codex_benchmark.py \
  --task pull_out_key \
  --profile 6 \
  --provide-bbox \
  --provide-mask \
  --dry-run
```

Codex runner 同样默认使用 ungrasped；只有显式 `--pre-move` 才运行旧初始化。

正式执行时移除 `--dry-run` 并指定新的 `--run-dir`。Runner 使用临时空 workspace 与
临时 `CODEX_HOME`，只复制认证文件；Codex 是单一连续 thread/turn，在一次 turn 中完成
多轮 observation + action。它没有 shell、文件、网络、MCP/app、skill、subagent、IK、
planner 或 simulator code 能力。所有已发布 reasoning summary、显式 decision record、
tool call、环境响应和图像都落盘，隐藏 chain-of-thought 不在 Codex 协议中公开。

## 推理 Token 预算

Codex runner 可选配置单 episode 的累计输出 token 上限：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/run_codex_benchmark.py \
  --task pull_out_key \
  --profile 6 \
  --max-output-tokens 12000 \
  --run-dir agent_runs/example_pull_key_budgeted
```

当前实现是 `posthoc_terminal_checker`：完整 turn 结束后读取 App Server 最后一次
`thread/tokenUsage/updated` 的累计 `total.outputTokens` 并与上限比较。该字段已经包含
`reasoningOutputTokens`，二者不会相加重复计数；等于上限仍算通过。配置预算后，准确数值
会写入 operator prompt，使 Agent 在开始前知道限制。

超限 run 仍保留完整记录且 `valid_for_scoring=true`，但 `benchmark_success=false`，
`score.failure_reasons` 包含 `token_budget_exceeded`。如果配置了预算但 App Server 没有
提供可靠 usage，run 会 fail closed 为 `completed_not_scorable`。当前版本不会在生成途中
强制中断；在线中断将在明确 interrupted episode 的清理和计分语义后单独升级。

`medium` / `high` 暂不绑定拍脑袋的固定数值。应在最终 wrist camera 合并后，用无预算
P6 rollout 的成功轨迹分布校准，再将两个名字映射到冻结的数值上限。

## 运行产物

环境记录包含：

- `manifest.json`：task、Profile、annotation 组合、动作边界和 seed commitment；
- `agent_transcript.jsonl`：完整公开命令/响应流；
- `observations/obs_*/`：RGB、tactile、depth、calibration、annotation 和 composite；
- `agent_observations_h264.mp4`：H.264/yuv420p 完整 observation 回放；
- `evaluator_outcome.json`：仅终局公开的 success 和细分 checker；
- `evaluator_private_audit.json`：终局才生成、权限 0600 的 host 审计。

Codex run 还包含 `codex_app_server_events.jsonl`、`codex_messages.jsonl`、
`codex_decisions.jsonl`、`codex_tool_calls.jsonl`、`CODEX_TRACE.md` 和 agent timeline MP4。
Base、developer 与 task/operator 三层指令分别落盘并在 manifest 中记录 SHA-256；完整
dynamic-tool schema 单独保存在 `codex_capabilities.json`。
这些内容可用只读 Viewer 复盘：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/run_agent_viewer.py --port 8765
```

## 验收

静态回归：

```bash
../miniconda3/envs/UniVTAC/bin/python -m pytest -q tests/agent_env
node --check agent_env/viewer_static/app.js
```

两任务完整 feature-smoke（真实 Isaac，依次执行）：

```bash
../miniconda3/envs/UniVTAC/bin/python tests/agent_env/real_benchmark_smoke.py \
  --task pull_out_key --profile 6 --provide-bbox --provide-mask \
  --run-dir agent_runs/accept_pull_key_p6

../miniconda3/envs/UniVTAC/bin/python tests/agent_env/real_benchmark_smoke.py \
  --task put_bottle_in_shelf --profile 6 --provide-bbox --provide-mask \
  --run-dir agent_runs/accept_bottle_p6
```

Feature-smoke 验证 `start -> step_eef -> finish -> host close`、全部 P6 字段、独立匿名标注、
终局前无 success、终局 checker、私有审计权限和 H.264 编码；它不把零动作 episode 的
任务成功率当作功能验收条件。

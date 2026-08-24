# Simulator Step-Window Recorder

## 目的

`agent_observations_h264.mp4` 只包含 `start_episode`、各次 `step_eef` 返回值和
`finish_episode` 终局 observation。它适合复盘 Agent 看到了什么，但无法回答机器人在一条
cuRobo 轨迹中何时碰倒、挤压或释放了物体。

generic AgentEnv 因此增加 evaluator-private 的仿真动作窗口录像：在同一次 episode 内，直接
从仿真物理步采集 head RGB、wrist RGB、left tactile RGB 和 right tactile RGB，并把四路图像
拼成一个 H.264 MP4。它是诊断产物，不是新的 Agent observation 模态。

## 时间语义

默认配置为：

- 仿真物理频率：120 Hz；
- 录像采样率：10 FPS，即每 12 个 physics step 采一帧；
- 每次通过协议校验的 `step_eef` 后固定追加 60 个 physics step、即 0.5 秒的 settling；
- recorder 默认记录全部 60 个 settling steps，也可只记录其前缀；
- `finish_episode` 已有的 60-step terminal settling 同样被记录；
- Agent 的推理、shell、图片分析或网络等待期间不推进仿真，也不产生重复录像帧。

每个 `step_eef` segment 的顺序固定为：

```text
pre_action frame
  -> control trajectory physics steps
  -> exact control/post-action boundary
  -> configured post_action_settle physics steps
  -> next Agent-visible observation
```

post-action settling 是 `step_eef` 的显式 rollout dynamics，不是 recorder 设置。它可能让
自由运动物体继续下落或稳定，因此只能在 settling 配置相同的 runs 之间比较 benchmark 结果。
即使 recorder 被关闭或 ffmpeg 中途失败，这 60 个 physics step 仍会执行，避免录像状态改变
episode 物理语义。cuRobo 规划失败的 `step_eef` 也保留相同 settling 窗口，用来观察规划拒绝
时场景中已有运动是否继续。

环境等待和录像窗口是两个独立参数：

```text
0 <= recorder_post_action_steps <= post_action_settle_steps
```

recorder 较短时，环境会继续推进剩余 settling steps，但不再写视频帧；下一公开 observation
仍在完整 settling 结束后采集。recorder 不能长于 settling，否则要么必须额外推进权威仿真、
改变 Agent 收到 observation 的时刻，要么只能生成没有新物理信息的重复帧。默认情况下未显式
设置 recorder steps 时，它自动继承完整 settling steps。

## 仿真接入

`AgentEnvTask._step()` 在调用原始 task `_step()` 后检查 `step_count` 是否真实增加。只有增加时
才通知 recorder，因此 `plan_success=false` 导致的空返回不会伪造 physics frame。原始 eval
路径本来就在每个 physics step 后执行 `_update_render()`；recorder 回调读取的正是该次更新后
的 sensor output，不新增第二次物理 stepping。

recorder 只在以下 segment 激活：

1. `step_eef` 从动作执行前到所选 post-action recorder 子窗口结束；
2. `finish_episode` 的 terminal settling。

`start_episode` reset、Agent 思考间隔和 simulator idle 时间不进入视频。

## 产物

每个 generic run 目录新增：

- `sim_step_composite_h264.mp4`：960×648、H.264/yuv420p/fast-start 四宫格视频；
- `sim_step_frames.jsonl`：每个视频帧对应的 segment、phase、sim step 和 sim time；
- `sim_step_recorder_manifest.json`：冻结配置、segment 边界、动作、执行结果、视频 hash 和错误；
- `evaluator_outcome.json.sim_step_recording`：manifest/video 的摘要与完整性信息。

每帧顶部显示对应 `step_eef` 序号、来源 observation id、XYZ/RPY/gripper delta、当前 phase、
physics step 和 simulated time。manifest 中的 segment 还记录：

- `start_sim_step` 与 `control_end_sim_step`；
- `post_action_record_end_sim_step` 与整个 segment 的 `end_sim_step`；
- 实际执行的 settling steps 与配置的 recorder steps；
- control route 与 `execution_succeeded`；
- 控制调用的 wall-clock duration。

录像器采用流式 ffmpeg pipe，不在内存中积累整轮原始帧。编码或 frame-supply 错误 fail open：
错误写入 manifest，机器人 episode 继续；最终 run 可据此判断诊断视频是否完整。

## 可见性边界

录像固定包含四路 RGB，即使当前 Agent profile 没有开放 wrist 或 tactile。这是为了让 evaluator
跨 P1–P6 使用一致的物理诊断视角。以下边界必须保持：

- MP4、帧索引和 recorder manifest 只写入 host/evaluator run directory；
- 不复制进隔离的 Agent workspace；
- 不在 `start_episode`、`step_eef` 或 `finish_episode` 的 tool result 中引用；
- 不改变 `PublicObservationFrame` 或 Profile allowlist；
- 不包含 depth、segmentation、actor pose、planner target、IK/joint internals 或 checker details。

这延续 UniVTAC 的责任边界：benchmark 保证主动提供给 Agent 的 observation 符合 Profile；
评测部署方负责是否进一步阻止 Agent 主动搜索 evaluator 文件系统。

## 命令行

正式 Codex runner 和直接 simulator bridge 使用相同参数：

```bash
--post-action-settle-steps 60
--sim-step-recorder / --no-sim-step-recorder
--sim-step-recorder-fps 10
--sim-step-recorder-post-action-steps 60
```

省略最后一项时，它自动等于 `--post-action-settle-steps`。大于 settling steps 的配置会在启动
前被拒绝。host 会分别校验 simulator `ready` 返回的 `step_eef` timing contract 与 recorder
contract，并把二者写入 `codex_run_manifest.json`。

## Viewer

Viewer 优先显示“连续仿真动作窗口回放”，并允许在页面内切换回原来的“完整 Observation
回放”。前者用于看动作内部物理过程，后者仍是 Agent 实际 observation 序列的权威回放。

## 当前限制

- 录像是 10 FPS 的 physics-step sampling，不是无损 120 FPS dump；
- 只保存 RGB composite，不保存每个内部 step 的米制 depth 或相机标定；
- cuRobo 的规划 wall time 不推进 simulator，所以不会产生视频帧；
- recorder 只接入 generic eight-task AgentEnv，legacy `grasp_classify` v0 runner 不变；
- 修改 `post_action_settle_steps` 会形成不同评测条件，必须随结果一同报告；只缩短 recorder
  子窗口不会改变 Agent observation 时刻或权威仿真状态。

# Formal16 复盘：结果、环境缺陷与 Agent 能力证据

日期：2026-08-20

## 结论

Formal16 的官方成功率是 `0/16`，但它不适合用来比较 P1--P6、annotation 或 ICL 的优劣。
这批 rollout 使用了后来确认存在缺陷的 legacy EEF 旋转语义，而且 Bottle 的纯夹爪和全零
动作也被错误地送入 cuRobo 手臂规划器。Bottle 共 263 个已接受的 `step_eef` 中有 65 个
没有执行，失败率为 `24.7%`。这些缺陷直接改变了动作物理含义和 Agent 的恢复能力。

这批数据仍然很有价值，适合用于：

- 验证 P1--P6、首帧 bbox/mask、fixed-demo ICL 和 App Server 记录链路确实端到端工作；
- 观察 Agent 是否会主动读取原始 depth、标定、mask 和专家 bundle，并自行写程序分析；
- 展示 Agent 对未知控制缺陷的在线系统辨识和补偿能力；
- 发现 benchmark 控制层、传感器和 checker/停止策略中的问题。

它不能用于宣称 observation profile 或 ICL 带来统计显著的成功率提升。

## 实验范围

正式数据位于：

```text
agent_runs/formal16_b21184e_seed_1830315042/
```

共同条件如下：

| 项目 | 条件 |
|---|---|
| source | clean commit `b21184e9314165d4bf5656f6454c7d6bccbc272a` |
| evaluation seed | `1830315042`，仅一个 seed |
| Agent | `gpt-5.6-sol`, reasoning effort `high` |
| interaction | `action_per_turn`，单一 Codex thread |
| runtime | 通用能力开启，`workspace-write`，网络允许 |
| task | `pull_out_key`, `put_bottle_in_shelf` |
| configs/task | P1--P6、P6+bbox+mask、P6+bbox+mask+fixed_demo |
| annotations | bbox/mask 仅初始 observation |
| ICL representation | 成功专家 observation/EEF waypoint bundle，`actions_present=false` |

目录 `01_key_p6_bbox_mask_fixed_demo` 是一次未产生动作的启动失败，不属于 16 个正式结果；
正式的第 1 项是 `01_key_p6_bbox_mask_fixed_demo_attempt2`。16 项均以
`completed_valid` 结束。

## 官方结果

### Pull Out Key

| 配置 | 目录 | accepted steps | execution failures | official success |
|---|---|---:|---:|---:|
| P1 | `02_key_p1` | 50 | 0 | false |
| P2 | `03_key_p2` | 49 | 0 | false |
| P3 | `04_key_p3` | 50 | 0 | false |
| P4 | `05_key_p4` | 48 | 0 | false |
| P5 | `06_key_p5` | 9 | 0 | false |
| P6 | `07_key_p6` | 11 | 0 | false |
| P6 + bbox/mask | `08_key_p6_bbox_mask` | 19 | 0 | false |
| P6 + bbox/mask + ICL | `01_key_p6_bbox_mask_fixed_demo_attempt2` | 50 | 0 | false |

Key 合计 286 个动作，全部送入仿真执行。官方成功率为 `0/8`。

### Put Bottle in Shelf

| 配置 | 目录 | accepted steps | execution failures | official success |
|---|---|---:|---:|---:|
| P1 | `09_bottle_p1` | 21 | 12 | false |
| P2 | `10_bottle_p2` | 21 | 7 | false |
| P3 | `11_bottle_p3` | 20 | 11 | false |
| P4 | `12_bottle_p4` | 50 | 2 | false |
| P5 | `13_bottle_p5` | 21 | 7 | false |
| P6 | `14_bottle_p6` | 46 | 6 | false |
| P6 + bbox/mask | `15_bottle_p6_bbox_mask` | 34 | 17 | false |
| P6 + bbox/mask + ICL | `16_bottle_p6_bbox_mask_fixed_demo` | 50 | 3 | false |

Bottle 合计 263 个已接受动作，其中 65 个动作完全没有执行。官方成功率为 `0/8`。

## 行为层结果

### Key

多数 Key rollout 将主要精力放在夹取和直接上拔，没有稳定完成“保持抓取、转动钥匙、再拔出”
的完整序列。最终只有 P6 的 `07_key_p6` 清楚地把钥匙提离了插槽：60-step settle 后钥匙
位置为 `[0.4912, 0.0075, 0.1409] m`，高度超过 `0.09 m`，保持直立，且 EEF 与钥匙的
高度差约 `0.1326 m`，小于 checker 的 `0.14 m` 上限。

该 run 在 `obs_010` 后的 evaluator-private step checker 曾为 true，但 Agent 不会收到过程
success，随后继续操作，终局 checker 变回 false。终局时上述三个条件仍满足，因此剩余失败
条件是 slot 相对初始姿态的 `slot_x_rotate > 0.99`。这不是“明显拔出却被无理由拒绝”，而是：
Agent 曾到达成功状态，但未能从公开 observation 判断应当停止，并在后续动作中扰动了 slot。

其他终局行为包括：P5 没有完成抽出；P6+bbox/mask 将钥匙碰落到桌面；formal ICL run
终局也没有保持抽出状态。因而 Key 的可靠结论是“出现过一次完整成功状态和一次清楚的终局
抽出行为”，不是官方成功。

### Bottle

没有一条轨迹把瓶子直立释放在 shelf 内。P1、P2 和 P5 的瓶子终局仍大致直立并停留在起始
区域；P3、P4、P6、P6+bbox/mask 和 ICL 的瓶子终局倒在桌面。P6 与 ICL 确实把瓶子移动到
更靠近 shelf 的位置，但没有放入目标区域：

- P6 终局位置约为 `[0.635, -0.086, 0.026] m`，瓶子横倒；
- P6+bbox/mask+ICL 终局位置约为 `[0.613, 0.099, 0.026] m`，瓶子横倒且未释放。

Agent 多次表现出失败后的重新定位、重新抓取和缩小步长，但 legacy 控制路由使部分本应可用
的松爪、等待和退出动作也被拒绝，不能把这些 rollout 当成未受干扰的 policy 评测。

## 后来确认的主要混杂因素

### 1. EEF 旋转会绕世界原点公转

Formal16 所用版本把平移后的绝对 EEF 位置也乘以增量旋转矩阵：

```text
p_target = R_delta @ (p_current + delta_position)
q_target = q_delta * q_current
```

因此 Agent 请求“零平移、原地旋转”时，EEF 仍会发生大幅位移。代表性证据包括：

- Key P1 的纯 yaw 动作可造成约 `18 cm` 横向位移；
- Bottle P6+bbox/mask 的一次 `pitch=+0.15 rad`、零平移动作实际令 EEF x 增加约
  `4.2 cm`、z 降低约 `7.6 cm`。

Key ICL Agent 在 rollout 中自行识别了这一规律，Bottle 的 pitch 则常被错误公转送入低位
碰撞/不可达区域。所有包含非零 `delta_rpy` 的 Formal16 rollout 都受此影响。

### 2. 每个动作都被包装成 `Action("all")`

Legacy `step_eef` 即使只改变 gripper，甚至四项 delta 全为零，也会重新调用 cuRobo 手臂
规划器。Bottle 与大体积 shelf/object 的接触和低位状态更容易被规划器拒绝，形成连续锁死：

- Bottle 8 runs 的 planner failure 数依次为 `12, 7, 11, 2, 7, 6, 17, 3`；
- 旧 Bottle P1 最长连续失败为 11 次；
- Key 286 个动作没有出现同类 planning failure。

因此这 65 次不是仿真崩溃，而是动作未执行；其中一部分是合理的不可达 move，另一部分是
不应该调用 arm planner 的纯夹爪/no-op/all 路由污染。

### 3. Wrist metric depth 看穿 deformable GelSight pad

P5/P6 的 wrist RGB 与机器人 state 显示 pad 已闭合时，`depth_m.npy` 的 pad 区域可能返回
其后方几何体的深度，使深度图看起来像夹爪仍然张开。最清楚的回归锚点是
`07_key_p6/obs_012`。该问题影响 Agent 对抓取状态和局部几何的判断，尤其污染 P5/P6 的
跨 profile 比较。

### 4. 强制 decision record/rationale

Formal16 要求每个机器人动作重复填写结构化 rationale、evidence、alternatives 和参数依据。
这些记录能辅助早期调试，但会显著占用输出与注意力，并主动塑造 policy。后续通用 v1 已改为
最小动作 schema，真实公开 reasoning summary、shell、图像、文件、MCP、子 Agent 与消息
直接从 App Server 事件流记录。

### 5. 单 seed 与运行时噪声

所有配置只使用同一个 seed，且模型 rollout 本身具有随机性。若干 run 还出现了自动恢复的
App Server stream reconnect。它们仍是有效完整 episode，但时延、token 量和单次行为不能
被解释为 profile 的因果效果。

## Formal16 实际揭示的 Agent 能力

### 主动使用通用计算工具

低 profile 的 P1--P4 没有调用 shell；当公开数据包含可计算的 depth、标定和 annotation
后，Agent 会自主编写分析程序：

| run | shell | view_image | 代表性行为 |
|---|---:|---:|---|
| Key P5 | 2 | 0 | 读取 raw NPY depth；NumPy 不可用后手工解析 NPY |
| Key P6 | 3 | 0 | 读取 head/wrist depth，并用 extrinsics 投影采样点 |
| Key P6+bbox/mask | 4 | 0 | 手工解析 NPY/PNG，用 mask+depth+calibration 估计 3D centroid |
| Key ICL | 3 | 1 | 枚举专家 bundle、读取 trajectory/state、计算 quaternion 差分 |
| Bottle P5 | 1 | 0 | 编写 depth 分析脚本 |
| Bottle P6 | 6 | 0 | 多轮 depth/标定分析与在线修正 |
| Bottle P6+bbox/mask | 3 | 0 | mask/depth 数值分析 |
| Bottle ICL | 13 | 2 | 解析 demo、mask 和实时帧，计算 EEF waypoint/orientation 差分 |

这说明 richer observation 没有被动停留在 prompt 中；Agent 会主动把公开 artifact 转成程序
可计算的几何信息。它也会在缺少 Python package 时寻找替代路径，而不是立即放弃。

### ICL bundle 被真实消费

两条 ICL run 都主动列出 `benchmark_inputs/expert_demo/`，读取 manifest、trajectory 和
state，并计算相邻 waypoint 或 quaternion 的差异。Key ICL 的 decision evidence 中有 21 次
引用 expert demo，Bottle ICL 有 11 次。因而可以确认 ICL 接入链路有效，不能确认 ICL 提升
成功率：两条 ICL run 的终局官方结果仍为 false。

### 在线识别并补偿未知控制缺陷

Key ICL 的公开 reasoning summary 依次出现：

- `Diagnosing end-effector yaw-induced position drift`；
- `Analyzing step delta pose rotation effects`；
- `Recomputing position compensation increments`；
- `Adjusting orientation and yaw compensation`。

首次较大 yaw 后，Agent 观察到远大于命令平移的横向漂移；后续在约 `-0.07 rad` yaw 中加入
约 `+3.4--3.6 cm` 的 y 补偿，在约 `+0.07 rad` yaw 中反向加入约 `-3.5 cm` 补偿，使实际
EEF 位置维持在毫米量级。这是很强的在线系统辨识、定量建模和闭环适应证据，也是保留
legacy demo 分支和对应 run 的主要理由。

Bottle ICL 同样公开记录了 `Analyzing yaw rotation causing orbit translation`、位置补偿、瓶子
倒下后的 replan/regrasp，以及对更大旋转步长的权衡。它没有完成任务，但不是盲目重复固定
轨迹。

### 闭环多模态操控与恢复

Key P6 使用 RGB、tactile、raw depth、内外参和机器人 state 完成抓取与上提；终局图像、
左右 tactile health 和 evaluator-private key pose 相互一致。Key P6+bbox/mask 在首步利用
object mask、head depth 和 state 推出世界坐标，随后利用 tactile 判断双侧接触。Bottle 多条
run 在动作拒绝或物体倒下后缩小步长、重新定位、重新抓取或后退，说明 Agent 会利用
`execution_succeeded` 和新 observation 做闭环修正。

这些行为证明了探索和适应能力，但还没有证明稳定、可复现的任务完成能力。

## 可以和不可以从这批数据得出的结论

可以得出：

- AgentEnv、P1--P6、首帧 annotation、ICL 投影和记录/Viewer 链路能够完整运行；
- Agent 会主动使用 shell、图像、代码和公开数值 artifact；
- ICL 资产确实被读取和用于规划；
- Agent 能发现未告知的 EEF 语义缺陷，估计其后果并在线补偿；
- 当前 benchmark 的控制、depth 和停止判定仍会显著影响最终结果。

不可以得出：

- P6 必然优于 P1，或 bbox/mask 必然有效；
- fixed-demo ICL 提升或降低成功率；
- `0/16` 等价于 Agent 没有具身探索能力；
- Bottle 的 65 次 planning failure 全部属于 Agent policy 错误；
- Formal16 可直接作为修复后 benchmark 的 leaderboard 基线。

## 修复后的最小重测顺序

1. 用同 seed Bottle P1 验收 move/gripper/all/no-op 路由和失败后恢复；该 smoke 已完成。
2. 用 Key yaw 与 Bottle pitch 的物理 consequence test 验收 EEF 原地旋转。
3. 决定或修复 GelSight pad 的 wrist metric-depth 表示，并保留固定回归锚点。
4. 在不强制 rationale/decision record 的最小 prompt/schema 下，重跑一小组端到端 task。
5. 上述条件稳定后，再用多个、与 demonstration 不同的 evaluation seed 重跑完整配置矩阵。

后续同 seed Bottle P1 smoke 的旧/新对照为：旧版 `21` steps、`12` failures、最长连续失败
`11`；修复版 `49` steps、`7` failures、最长连续失败 `3`。修复版的 18 个纯 gripper、1 个
no-op 和 1 个 all 全部成功，7 个失败全部来自真实 move，private cuRobo status 为 5 个
`IK_FAIL` 和 2 个 `FINETUNE_TRAJOPT_FAIL`。这证明路由/恢复修复生效，但该 P1 policy 仍未
抓取并放置瓶子。

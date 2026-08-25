# Agentic Embodied Benchmark v1

v1 是当前通用 benchmark 接口。它把“Agent 能看到什么”与“Agent 如何控制机器人”分开：

- 六个 Observation Profile 是主榜能力轴；
- 匿名 BBox 与 Mask 是两个可独立开关的诊断轴；
- `icl=none|fixed_demo` 是独立的固定示范诊断轴；
- 所有条件共享同一个世界坐标系 EEF 动作接口；
- 控制过程中不返回 reward 或 task success；
- `finish_episode` 后才执行终局评测并公开 success。

旧的 `grasp_classify` 三 level 协议保留为 v0 兼容路径；同名任务也已作为标准
`start_episode/step_eef/finish_episode` 任务接入 v1。

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

Wrist metric depth 包含刚性夹爪和 GelSight housing，但不包含可变形 optical gel
surface。UniVTAC 在 runtime USD session layer 中只恢复左右 `case/plate` 的
secondary-ray visibility，并明确保持 gelpad 隐藏；这既修复刚性夹爪在 depth 中的空洞，
也避免改变 GelSight 内部触觉渲染。该策略由真实 closed-key A/B 回归验证。

## 匿名标注诊断轴

命令行的两个开关完全独立，可组成四种条件：

```text
none
--provide-bbox
--provide-mask
--provide-bbox --provide-mask
```

`start_episode` 返回的首帧 `obs_000` 中，每个 Profile 公开的相机按需返回两个匿名角色：

- `manipulated_object`
- `goal_fixture`

BBox 使用 `[x1, y1, x2, y2]`、右下角 exclusive 的像素坐标；Mask 是同分辨率二值
PNG。物体不在首帧视野时会返回 `visible=false`、空 BBox/Mask，而不会伪造位置。
`obs_001` 及之后不再返回 annotation 字段、坐标、Mask 或 overlay；固定示范同样只在
`frame_000000` 中提供。这一 `initial_observation_only` 规则让后续跟踪仍由 Agent 完成。
Isaac 的 raw instance ID、label、actor 名和 USD prim path 只在 host 内存中用于映射，
终局前不会写入公开记录。SAM 3/UniDepth V2 不参与该 oracle 轴。

`grasp_classify` 的正确目标垫取决于隐藏类别，因此它的 `goal_fixture` 是绿、橙两个候选垫
的联合 Mask/BBox，不会只标出正确目标而泄露分类答案。其他任务的 `goal_fixture` 对应单个
目标结构。

## 固定专家 ICL 诊断轴

`--icl none` 不创建、不提示任何专家资产。`--icl fixed_demo` 则在 Agent 启动前，把当前
任务已验证的 P6 master 按本次 `Profile × BBox × Mask` 条件投影到临时 workspace 的
`benchmark_inputs/expert_demo/`。原始 HDF5、P6 host manifest、seed 与 checker 证明不会
进入 Agent workspace。

专家轨迹公开的是采样到的状态/EEF waypoint，不是不存在的 `step_eef` 调用记录；因此
`trajectory.jsonl` 明确使用 `observed_expert_waypoint`，不提供或伪造动作 delta。整条示范
只公开一次 episode 级的 `successful expert demonstration`，不公开逐步 success、首次成功
时刻或 checker 细节。BBox/Mask 依然只出现在专家首帧。

固定示范是 ungrasped 条件，因此 runner 拒绝 `--icl fixed_demo --pre-move`。完整静态数据
格式、投影规则和验证命令见
[`PublicObservationAndICLDataContract.md`](PublicObservationAndICLDataContract.md)。

## 任务

| Task | 指令 | 初始状态 | 终局策略 |
|---|---|---|---|
| `grasp_classify` | Grasp the center prism and place it upright on the green pad. | home pose，prism 未抓取 | 固定绿色目标 checker |
| `insert_HDMI` | Grasp the HDMI connector and insert it fully into the port. | home pose，connector 未抓取 | 原任务 checker |
| `insert_hole` | Grasp the peg and insert it fully into the angled hole. | home pose，peg 未抓取 | 原任务 checker；Agent close-gripper 后冻结 evaluator-private in-hand reference |
| `insert_tube` | Grasp the tube and insert it fully into the fixture. | home pose，tube 未抓取 | 原任务 checker；Agent close-gripper 后冻结 evaluator-private in-hand reference |
| `lift_bottle` | Grasp the bottle, rotate it upright beside the wall, and release it stably. | home pose，bottle 未抓取 | 原任务 checker |
| `lift_can` | Grasp the horizontal can, rotate it upright on the table, and release it. | home pose，can 未抓取 | 原任务 checker |
| `pull_out_key` | Grasp the key and pull it completely out of the slot. | 默认机器人 home pose，key 未抓取；key 相对 slot 的初始 yaw 恢复原始 `Uniform(-π/2, -π/4)` 分布 | 使用原任务 checker；拔出阈值仍需后续视频校准 |
| `put_bottle_in_shelf` | Pick up the bottle from the table, place it upright inside the shelf, and release it. | 默认机器人 home pose，bottle 未抓取 | 原位置/姿态 checker + 已松爪 + 60 physics steps 后稳定 |

generic `grasp_classify` benchmark 固定以绿色 pad 为目标，不再要求根据触觉区分 rough/plain；
上游数采任务的默认 `material_classification` 模式保持不变。注册的 seed-0 ICL 轨迹本来就是
plain prism 放到绿色 pad 的成功演示，因此可直接复用，无需重新采集或修改 frozen master。

v1 默认关闭 task-specific `pre_move()`，因此 Agent 必须自行定位、接近并抓取物体。为复现
旧的抓取后评测，可显式添加 `--pre-move`；该模式会原样调用上游任务的 privileged
`pre_move()`，并在 manifest 中记为 `start_condition=pregrasped`。默认模式记为
`start_condition=ungrasped`。跳过 `pre_move()` 时只建立旧 checker 必需的 host-private
参考状态，不移动机器人或物体。

`insert_hole` 与 `insert_tube` 的上游 checker 把 pre-move 抓取后的物体—夹爪相对位姿用作
防滑参考。v1 ungrasped 路径在 Agent 实际 close-gripper 周期结束、开始搬运时建立并冻结同一
类 evaluator-private 参考；否则 reset 时的未抓取相对位姿会令正常完成也无法通过 checker。
这项兼容层并非原六任务接入需求的一部分，当前仍属于尚待物理校准的 milestone 行为；其原始
假设、精确状态机、已知风险、验证证据与回滚边界完整记录在
[`CheckerLifecycleCompatibility.md`](CheckerLifecycleCompatibility.md)。

Key 的相对初始 yaw 可通过 `--key-initial-relative-yaw-rad RADIANS` 固定；例如诊断用的
垂直条件传入 `-1.5707963267948966`。省略该参数时始终使用上述原始随机分布，所选模式与
固定值或随机范围会写入 evaluator manifest，但不会加入 Agent task prompt。

Bottle 的 release 阈值为 gripper qpos `>= 0.0175 m`；稳定要求 60 steps 前后物体平移
不超过 `0.01 m`、旋转不超过 `10°`。三项必须同时成立。

## Agent 工具与动作边界

模型侧只注册三个 dynamic tools：

1. `start_episode`
2. `step_eef`
3. `finish_episode`

工具参数保持为最小控制协议：`start_episode` 不带参数，`step_eef` 只带最新
`observation_id` 和四项动作数值，`finish_episode` 只带最新 `observation_id`。v1 不再要求
`rationale`、`decision_record`、`agent_note` 或 `final_note`。Agent 的公开 reasoning summary、
消息、shell/文件/MCP/子 Agent/网络/图像等活动直接从 App Server 事件流记录，不要求模型
把推理压缩、改写或重复填入机器人 tool call。

`step_eef` 使用世界坐标系 XYZ/RPY 增量和夹爪增量：

| 参数 | 边界 |
|---|---|
| XYZ delta | Benchmark 不设幅度上限；必须是三个有限数值，下层规划器仍可因不可达或碰撞而拒绝目标 |
| RPY delta | Benchmark 不设幅度上限；必须是三个有限数值，下层规划器仍可拒绝无效目标 |
| Gripper delta | 每指 qpos 的目标值必须保持在物理范围 `[0, 0.039] m`；不做静默钳制 |
| Accepted `step_eef` 数量 | 最多 50 |

`step_eef` 按实际非零分量路由：仅 EEF 改变时使用 `move`，仅 gripper 改变时使用
`gripper`，两者都改变时才使用 `all`。全零动作合法，并以不调用 arm/gripper planner 的
固定 20 physics steps wait 作为其控制阶段。这样纯夹爪和 wait 不会因为无关的 cuRobo arm
planning failure 而被拒绝。每个 step 必须引用最新 `observation_id`；陈旧 ID、
未知字段、非有限数和超出物理开合范围的 gripper 目标在改变世界之前被拒绝。接口不返回目标方向、目标半空间、
语义动作提示、actor pose、IK/joint target 或过程 task success。

每个通过协议校验的 `step_eef` 在控制阶段结束后，默认统一推进 60 个 post-action settling
physics steps，再采集下一条 Agent-visible observation；该时序在 recorder 关闭或编码失败时
仍保持不变。全零动作因此默认是 20-step no-op 控制加 60-step 公共 settling。连续 recorder
可以只保存 settling 的前缀，但不能超过这 60 steps；省略 recorder 子窗口配置时默认完整
记录。精确定义见 [`SimulatorStepWindowRecorder.md`](SimulatorStepWindowRecorder.md)。

每个 arm planning 结果的原始 cuRobo status、query validity、attempt 数、timing 和终端误差
仅写入终局生成且权限为 `0600` 的 evaluator-private audit。Agent-visible response 仍只有
`execution_succeeded` 布尔值，不公开 collision、IK、trajectory optimization 或 planner
内部状态。

真实路由验收 `bottle_curobo_routing_smoke_seed_1830315042` 复用了 Formal16 Bottle P1 的
前两次下降：第一次 arm-only `z=-0.04 m` 成功，第二次到 `z≈0.219 m` 的目标被精确记录为
`MotionGenStatus.IK_FAIL`、`valid_query=true`、`attempts=10`。失败后紧接的纯 gripper 动作
未调用 cuRobo，并把 gripper width 从约 `0.040 m` 改为 `0.030 m`；随后的全零动作同样未
调用 planner，并恰好推进 20 physics steps。private audit 权限为 `0600`，planner status 与
逐步 control route 在 manifest、public transcript 和 evaluator outcome 中均未出现。

完整 Codex A/B smoke `bottle_p1_curobo_fix_smoke_seed_1830315042` 又以 Formal16 Bottle P1
相同 seed、P1、无 annotation、无 ICL、同一 `gpt-5.6-sol/high` 与 action-per-turn 条件跑满
一轮。旧 run 的 21 个 step 中 12 次执行失败、最长连续失败 11 次；修复后 49 个 step 中
7 次失败、最长连续失败 3 次。修复后 18 个 gripper-only、1 个 all 和 1 个 no-op 全部成功，
7 次失败全部来自 move，private status 为 5 次 `IK_FAIL` 和 2 次
`FINETUNE_TRAJOPT_FAIL`。其中三次 move 连败后，全零 20-step settle 成功，紧接的同方向
退升也成功；另有多组 move 拒绝后纯夹爪立即成功。该 run 最终仍未完成 Bottle，因此证明的
是动作路由和恢复语义已经生效，不是 P1 policy 或 Bottle 任务已经解决。Codex 生成具有
随机性，A/B 的总 step 数不能单独作因果证据；逐 route 的 planner/bypass 记录才是验收依据。

### StepEEF 世界旋转与 legacy Pose 旋转

`8ccafe0` 及以前的 `delta_ee` 实现尚未满足上面的独立 XYZ/RPY 增量契约。
`BaseTask.take_action(..., action_type="delta_ee")` 先对当前位置加平移，再调用
`Pose.add_rotation(..., coord="world")`。后者不仅左乘姿态，还对绝对位置应用同一个
旋转矩阵。因此 legacy 实际目标为：

```text
p_target = R_delta @ (p_current + delta_p)
q_target = q_delta * q_current
```

当前 `step_eef` 改为先调用 `add_bias(..., coord="world")`，再调用 orientation-only 的
`add_orientation_delta(..., frame="world")`，从而实现契约要求的语义：

```text
p_target = p_current + delta_p
q_target = q_delta * q_current
```

在 legacy 语义下，即使 `delta_p == 0`，非零 `delta_rpy` 也会使 EEF 目标绕世界原点发生平移，表现为
“公转”，而不是保持工具中心点不动的原地旋转。该行为来自 UniVTAC 初始提交
`b371b18` 中的 `envs/_base_task.py` 与 `envs/utils/transforms.py`；AgentEnv 只是把已有的
`delta_ee` 路径暴露为 `step_eef`，cuRobo 则规划到已经发生偏移的目标，因此两者都不是
这次位姿耦合的源头。

对 UniVTAC 全部公开 issue 的审计尚未发现对这一精确缺陷的报告。公开的
[#6：EE action 训练标签与部署重规划语义不匹配](https://github.com/univtac/UniVTAC/issues/6)
讨论的是保存的绝对 EEF label 被再次规划而产生绕行，是相邻但不同的问题；它没有指出
`delta_rpy` 会通过 `R_delta @ p` 改变位置。

修复没有改变通用 `Pose.add_rotation()` 的 legacy 行为：`coord="world"` 仍表示绕世界
原点旋转整个 Pose；传入具体 `Pose` 仍可用于绕指定支点运动。当前 tracked codebase 中，
曾将 `add_rotation(..., coord="world")` 用于 orientation-only 控制的调用只有 `delta_ee`。
默认 `local` 分支的历史四元数组合语义也暂不改变，避免影响 scripted expert 与任务初始化；
新的 `add_orientation_delta()` 则明确规定 world frame 左乘、local frame 右乘，且两者都不
改变 position。

该缺陷不要求重新采集当前 fixed-expert 数据。正式专家 HDF5 保存的是 scripted
expert/cuRobo 产生的状态轨迹，派生 P6 master 也通过 `qpos` replay 采集 observation；二者
都不包含或执行 `step_eef` action。修复后应对原 frozen HDF5、P6 master 和终局 checker
做一次回归验证，但无需重新录专家或重新补录 P6。相反，所有执行过非零 `delta_rpy` 的
旧 AgentEnv rollout 都属于 legacy action semantics，不能与修复后的正式结果直接比较，
应重新评测。任何未来若保存了 `delta_ee` action 的数据集也必须重新生成，或显式标为
legacy semantics。

静态验收覆盖：纯旋转不改变位置、平移与旋转解耦、world 左乘与 local 右乘不交换，以及
legacy `add_rotation(coord="world")` 仍保留绕原点旋转位置的行为。真实仿真验收使用
`delta_position=[0,0,0]`、`delta_rpy=[0,0,0.1]`。同一任务与 seed 下，修复前该命令产生
`49.28 mm` 平移；修复后的 `eef_yaw_semantics_fixed_seed_1938475621` 在 action complete 时
仅平移 `0.0813 mm`，实际姿态增量为 `0.10015 rad`，60-step settle 后相对初始位置仍只偏移
`0.1661 mm`。这表明剩余位移属于规划与控制跟踪误差，而非目标构造中的世界原点公转。

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
  --icl fixed_demo \
  --dry-run
```

Dry-run 不读取大型专家资产，只展示将采用的 Profile、annotation、ICL 和机器人工具契约。
Codex runner 默认使用 ungrasped；只有 `icl=none` 时才允许显式 `--pre-move`。

正式执行时移除 `--dry-run`、指定新的 `--run-dir`，并通过参数或 evaluator-private 环境
变量定位固定示范 master：

```bash
UNIVTAC_FIXED_EXPERT_MASTER_ROOT=/path/to/expert_observation_master \
../miniconda3/envs/UniVTAC/bin/python scripts/run_codex_benchmark.py \
  --task pull_out_key --profile 6 \
  --provide-bbox --provide-mask \
  --icl fixed_demo \
  --interaction-mode action_per_turn \
  --codex-sandbox danger-full-access \
  --run-dir agent_runs/pull_key_p6_fixed_demo
```

Generic v1 runner 为每次 rollout 创建干净的临时 workspace，但继承评测方的正常
`CODEX_HOME` 配置，也不再通过启动参数禁用 shell、文件、`view_image`、MCP/app、skill、
subagent 等通用能力。由于当前 reference host 的 bubblewrap 不可用，runner 当前默认使用
`danger-full-access` 并允许网络；评测方仍可改用 `--codex-sandbox`，并在
`read-only`/`workspace-write` 模式下用 `--no-codex-network-access` 或显式
`--codex-network-access` 控制网络。`danger-full-access` 始终允许网络。请求值与实际生效值
都会写入 manifest。临时 workspace 此时只负责运行数据组织与跨 episode 分离，不构成安全边界。
Benchmark core 不把 shell、网络、插件或其他通用工具使用判为 capability violation；这些
行为由 App Server 原始事件流审计，是否属于 hacking 由评测方判断。机器人控制仍只能经过
`start_episode`、`step_eef`、`finish_episode` 三个 host 校验的 dynamic tools，通用工具
不会获得第二条仿真控制通路。

TODO：安装并验证可工作的 bubblewrap 后，再恢复 `workspace-write` 作为可选的 reference
sandbox 模式；这不会改变 Benchmark core 的 observation-contract 责任边界。另一个独立
TODO 是把当前笼统的 tool rejection 文本升级为安全、非泄漏的错误类别，帮助 Agent 自我
修复参数错误。

Operator prompt 默认只包含当前 task instruction；`icl=fixed_demo` 时额外声明已验证
示范位于 `benchmark_inputs/expert_demo/`，它来自同一任务的另一个 episode，
当前 episode 的场景配置及物体或目标位姿可能不同。该声明只提供资产关系，
不告诉 Agent 如何适配；配置显式 token budget 时才再增加预算声明。非策略性的三工具
生命周期和
task-success 终局可见性放在 base instruction/tool description 中。当前 UniVTAC base
instruction 仅说明三工具生命周期、每次机器人调用后等待其结果 observation，再进行下一次
机器人调用，以及 success 只由 `finish_episode` 返回；不再发送额外的 UniVTAC developer
instruction。
同时为 ICL manifest 设计显式但非策略性的关系元数据，例如：

```json
{
  "target_task": "pull_out_key",
  "target_variant": "perpendicular",
  "demo_source_task": "pull_out_key",
  "demo_source_variant": "standard",
  "demo_relation": "same_task_different_variant"
}
```

该 manifest 关系 schema 目前只作为 TODO，不在本轮实现；后续实现时只声明资产关系，
不向 Agent 提供适配方法。
prompt 不再要求 Agent
“只能使用本 run 信息”，也不要求结构化 rationale。Reference runner 默认使用
`action_per_turn`；`single_turn` 仅保留为显式兼容/诊断选项。

在线 RGB、触觉、depth preview 与首帧 annotation overlay 由 Agent transport 主动发送；
每张图前的标签使用完整公开 JSON 字段路径（例如
`observation.modalities.wrist_depth.visualization`），避免重复的 `valid_mask` 或
`bbox_overlay` 名称失去视角和语义角色。
无法作为 image content 传输的米制 `.npy` depth 与 raw mask 只发布到
`benchmark_inputs/current_observation/`，每个新 observation 原子替换前一帧，不提供自动
历史。完整在线轨迹仍仅保存在 evaluator-private run directory；Agent 如需历史应自行保存。

Codex 在整个 episode 中始终使用一个连续 thread。Runner 支持两种显式记录在 manifest
中的传输模式：

- `single_turn`：保持旧行为，observation 作为 dynamic-tool result 返回，模型可在同一
  turn 中继续调用下一个机器人动作；
- `action_per_turn`：每个非终局机器人调用后结束当前 turn，并把该调用产生的完整公开
  JSON + images 作为同一 thread 的下一个顶层 multimodal turn 输入。Tool result 只返回
  非策略性的 deferred acknowledgement；`finish_episode` 的终局结果仍在最后一个 turn
  中直接返回。

`action_per_turn` 不创建新 session、thread 或 simulator episode，也不改变 50-step 动作
预算；它只把 observation/action 的环境边界映射为自然的对话 turn 边界。每个 turn 在
机器人动作前仍可使用 evaluator 允许的 shell、图片、MCP 等通用能力。
如果模型在 interrupt 生效前已批量生成额外机器人 tool call，host 只执行该 turn 的第一项，
其余调用以普通 `tool_rejected` 记录并返回，随后仍用第一项动作产生的真实 observation 开启
下一 turn；这种 transport race 不再被误判为 capability violation。
所有已发布 reasoning summary、agent message、shell/文件/MCP/子 Agent/网络/图像活动、tool call、
环境响应和 observation 都落盘。Viewer 直接从 App Server 事件流重建完整可观察活动；新
v1 run 不生成结构化 decision stream，旧 run 的 decision record 仅作为历史兼容产物。隐藏 chain-of-thought 不在 Codex
协议中公开。旧 v0 `grasp_classify` runner 仍保持自身的严格隔离策略，见
[`CodexAgentRunner.md`](CodexAgentRunner.md)。

## 推理 Token 预算

Codex runner 可选配置单 episode 的累计输出 token 上限：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/run_codex_benchmark.py \
  --task pull_out_key \
  --profile 6 \
  --max-output-tokens 12000 \
  --run-dir agent_runs/example_pull_key_budgeted
```

当前实现是 `posthoc_terminal_checker`：完整 episode/thread 结束后读取 App Server 最后一次
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
- `sim_step_composite_h264.mp4`：仅覆盖 simulator-active 动作窗口的 head/wrist/双触觉
  连续诊断录像；对应 frame index 与配置见
  [`SimulatorStepWindowRecorder.md`](SimulatorStepWindowRecorder.md)；
- `evaluator_outcome.json`：仅终局公开的 success 和细分 checker；
- `evaluator_private_audit.json`：终局才生成、权限 0600 的 host 审计。

Codex run 还包含 `codex_app_server_events.jsonl`、`codex_messages.jsonl`、
`codex_tool_calls.jsonl`、`CODEX_TRACE.md` 和 agent timeline MP4。旧 schema run 还可能包含
`codex_decisions.jsonl`；rationale-free v1 run 不要求也不创建该文件。
`icl=fixed_demo` 时还会在 evaluator run directory 中保存
`icl_projection_receipt.json`，用于认证所用 master 与 public bundle 哈希；该 receipt 不会
放入 Agent workspace。运行 manifest 同时记录 ICL 条件、通用 capability 策略、sandbox、
网络条件，以及严格固定的三项机器人控制工具。
Base 与 task/operator 两层 UniVTAC 指令分别落盘并在 manifest 中记录 SHA-256；新 run 的
developer-instruction manifest 字段为 `null`，且 `thread/start` 不发送
`developerInstructions`。完整 dynamic-tool schema 单独保存在
`codex_capabilities.json`。
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

两条正式专家资产的完整投影矩阵：

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/validate_fixed_demo_assets.py \
  --fixed-demo-root /path/to/expert_observation_master --summary-only
```

该命令覆盖 `2 tasks × 6 Profiles × 4 annotation conditions = 48` 组，并逐帧检查源数据与
投影后 artifact 哈希、字段 allowlist、相对路径、initial-only annotation 和无额外文件。

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

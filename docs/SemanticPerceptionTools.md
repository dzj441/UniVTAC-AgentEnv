# SAM 3 与 UniDepth V2 感知工具

这组工具是 AgentEnv 的一个正交实验轴，不改变 Level 1/2/3 的定义。Level 仍然控制
原始视觉、触觉与 success guidance；`perception_profile` 只决定 agent 能否在当前公开
RGB 上调用只读的模型感知工具。

| Perception profile | 额外动态工具 | 每个 observation 的调用上限 | 返回内容 |
|---|---|---:|---|
| `none`（默认） | 无 | 0 | 保持原始三 Level benchmark |
| `sam3` | `sam3_segment` | SAM3: 4 | text/point prompt 的 mask、bbox、score、overlay |
| `unidepth_v2` | `estimate_metric_depth` | UniDepth: 2 | 预测 metric depth、相对 confidence、可视化与可选像素采样 |
| `sam3_unidepth_v2` | 上述两者 | SAM3: 4；UniDepth: 2 | 两类能力同时开放 |

SAM 3 用于开放词汇目标分割：text mode 接收简短视觉短语，point mode 接收原始
480×270 RGB 上的前景/背景点。UniDepth V2 从单目 RGB 和相机标定预测米制深度。
后者是模型 prior，不是 Isaac 的 Z-buffer、物体 pose 或碰撞安全证明。

## 公平性边界

- agent 只能提交 `observation_id`、`head_rgb`/`wrist_rgb` 和受限 prompt/像素参数；
- 宿主按最新 observation 解析真实图片，agent 不能传文件路径、base64 或任意图片；
- 旧 observation、越界像素、未知字段和未开放工具均在宿主侧拒绝；
- 每个 observation 最多调用 4 次 SAM3、2 次 UniDepth（失败请求也消耗预算），防止
  无限 text-prompt fishing 或后端失败重试；新 observation 会获得新预算；
- 精确相机内参来自 Isaac Lab `TiledCamera`，保存到 mode-0600 的宿主侧车；内参、外参
  与原始 simulator depth 从不进入 prompt、公开 observation 或工具结果；
- 语义结果成功返回后，才允许同一 observation 的后续 `decision_record` 引用
  `sam3_result` 或 `unidepth_v2_result`；
- 感知调用只读，不刷新 observation、不消耗仿真动作预算，也不开放 IK、joint target、
  planner、shell、文件或网络能力；
- 每次请求、结果、模型 revision、耗时和派生图片都进入 Codex audit stream 与 viewer。

服务采用 OpenETA 在 commit
`fbf102dfd30e44b981d50f1c29dede4274eaaf8f` 中的“独立模型环境、lazy load、localhost
服务、固定工具 contract”设计，并适配成 UniVTAC 宿主拥有的动态工具。模型服务本身
不会直接暴露给 Codex。

## 安装布局

所有下载命令会显式清除大小写两套 HTTP/HTTPS/ALL proxy 环境变量，并继续使用机器
现有 Conda/pip 默认源。

```bash
./scripts/setup_semantic_tools.sh
```

实际 Python 环境直接位于父目录 Miniconda，不使用数据盘环境或环境软链接：

```text
../miniconda3/envs/univtac-sam3
../miniconda3/envs/univtac-unidepth-v2
```

若现有 `../miniconda3/envs/UniVTAC` 缺少 `libGLU.so.1`，安装脚本会从默认 Conda 源
在该原有 simulator 环境中补 `libglu`（当前解析只新增 `libglu` 和两个 Xorg 小依赖），
不调用 apt，也不改宿主系统图形栈。

模型和 Hugging Face cache 位于：

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/univtac_semantic_tools/
```

仓库的 `.semantic_models` 只链接到上述数据盘的 `models/`。脚本从仓库 `hf_token`
读取 gated SAM 3 权限，但不会打印或复制 token。源码、Torch、模型 revision 均固定在
`semantic_tools/versions.json`。下载完成后，数据盘的 `installation_manifest.json` 还会记录
模型 revision、字节数与 SHA-256。也可以用 `--envs-only` 或 `--models-only` 断点续跑。

模型净权重约 4.9 GB（SAM 3 约 3.45 GB，UniDepth V2 ViT-L 约 1.42 GB）；两个隔离
Torch/CUDA 环境还会占用更多父目录磁盘。UniDepth 官方代码/权重是 CC BY-NC 4.0，
用于商业或对外服务前需要重新检查许可证条件。

## 启动与停止

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/manage_semantic_services.py start all --json
../miniconda3/envs/UniVTAC/bin/python scripts/manage_semantic_services.py health all --json
../miniconda3/envs/UniVTAC/bin/python scripts/manage_semantic_services.py stop all --json
```

默认监听仅本机的 `127.0.0.1:8783`（SAM 3）和 `127.0.0.1:8784`（UniDepth V2）。
启动 health check 不加载权重；第一次真实请求才 lazy-load。PID、日志保存在被 git
忽略的 `.semantic_runtime/`，停止时会核对 PID、进程启动时刻、module 和端口，避免
误杀复用 PID 的无关进程。

## 运行 Codex episode

先启动服务，再选择任意 Level 与 perception profile：

```bash
./scripts/run_codex_agent_env.py \
  --level 2 \
  --perception-profile sam3_unidepth_v2 \
  --device cuda:0 \
  --run-dir agent_runs/example_l2_semantic
```

不传 `--perception-profile` 时严格保持 `none`，已有实验可直接复现。SAM 与 UniDepth
可分别通过 `--sam3-url`、`--unidepth-v2-url` 覆盖，但 URL 必须是 localhost HTTP。

每次调用的派生产物位于：

```text
semantic_perception/<observation_id>/<tool>/call_<index>/
```

`codex_tool_calls.jsonl` 会区分 `execution_target=simulator|perception|rejected`，并记录去掉
图像载荷的 `backend_request`；`CODEX_TRACE.md` 和 Agent Run Viewer 会显示 SAM contact
sheet、depth/confidence 图、agent 的显式 decision record 与下一步动作。

部署后可运行一次“不移动机器人、不计分”的真实链路验收。它会经过 Isaac RGB、host
侧标定、CapabilityGateway、两个 GPU 服务、派生产物、终局视频与清理协议：

```bash
../miniconda3/envs/UniVTAC/bin/python tests/agent_env/real_semantic_smoke.py \
  --run-dir agent_runs/semantic_gateway_validation
```

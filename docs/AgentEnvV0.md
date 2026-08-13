# Grasp-Classify AgentEnv v0

This is the first agent-native UniVTAC vertical slice. It keeps one simulator,
action space, initialization distribution, checker, and budget while changing
only what information the agent receives.

All three profiles also share a 240-second reset guard. This only tolerates
shared-machine initialization jitter; it does not extend the post-reset action
budget or change task dynamics, initialization, control, or success checking.

## Three levels

| Level | Public observation | Public feedback after an action |
|---|---|---|
| 1 `vision_only_control` | head RGB, wrist RGB, 8-D joint state, gripper qpos, 7-D end-effector pose | low-level `execution_succeeded` |
| 2 `visuotactile_control` | Level 1 plus left/right tactile marker RGB | low-level `execution_succeeded` |
| 3 `success_guided_visuotactile_control` | Same observation as Level 2 | `execution_succeeded` plus `task_success`, but only after `submit_prediction` |

Depth, bounding boxes, camera calibration, actor poses, contact forces, reward,
and privileged simulator state are absent in v0. “Absent” is literal: a field
is not serialized as `null`, `withheld`, or a placeholder.

`execution_succeeded` only says that the controller/planner executed a command.
It is available at every level and must not be confused with the task checker.

## Irreversible prediction boundary

Translation is locked until the agent calls:

```json
{
  "command": "submit_prediction",
  "observation_id": "obs_000",
  "predicted_class": "rough",
  "target_pad": "orange",
  "rationale": "The tactile deformation is spatially irregular."
}
```

The task mapping is `rough -> orange` and `plain -> green`. The prediction may
be submitted exactly once. Afterwards, a cumulative world-Y guard prevents the
agent from crossing to the other pad. Level 3 success guidance is unlocked only
after this boundary, so it can refine placement geometry but cannot be used to
try both labels. This transition is also the intended insertion point for a
future post-prediction `early_failure` guidance channel.

Before prediction, an agent may make up to two optional gripper-only `probe`
calls. They change the physical contact but never expose task success. No probe
is required, which keeps Level 1 usable even though it cannot see tactile data.

## Start a session

From the repository root:

```bash
./scripts/launch_agent_env.sh --level 1 --device cuda:0
./scripts/launch_agent_env.sh --level 2 --device cuda:0
./scripts/launch_agent_env.sh --level 3 --device cuda:0
```

The launcher defaults to the configured parent environment:

```text
../miniconda3/envs/UniVTAC/bin/python
```

Set `UNIVTAC_PYTHON` to override it. On the current cluster the launcher uses
the independently prepared NVIDIA 570.124.06 **userspace-only** rendering stack
at:

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/570.124.06
```

The machine also needs three small, generic runtime packages (no NVIDIA or CUDA
package is installed by this command):

```bash
apt-get install -y --no-install-recommends libxt6 libglu1-mesa libxrandr2
```

It removes the host-injected NVIDIA/compat library directories from the child
process, loads EGL/Vulkan/GLX libraries from that data-disk bundle, sets its
Vulkan/EGL manifests, and refuses to start unless every visible GPU reports the
matching 570.124.06 kernel driver. The kernel-facing system `libcuda` remains the
single CUDA driver image, as intended by the bundle README. The launcher never
installs or changes a kernel module, CUDA toolkit, or NVIDIA package. To use
another machine, point `UNIVTAC_NVIDIA_RENDER_ROOT` at an exact-match bundle; if
its directory basename is not the driver version, also set
`UNIVTAC_NVIDIA_RENDER_VERSION`.

The transport is newline-delimited JSON on stdin/stdout. Machine-readable
responses start with `AGENT_ENV_RESULT `. The command sequence is:

```text
start
  -> zero to two probe calls
  -> submit_prediction (irreversible)
  -> act and/or wait (at most ten)
  -> finish
  -> close
```

Every physical/prediction command cites the latest `observation_id` and includes
a non-empty `rationale`. An `act` command uses bounded world-frame delta control:

```json
{
  "command": "act",
  "observation_id": "obs_001",
  "delta_position": [0.02, -0.04, -0.02],
  "delta_rpy": [0.0, 0.0, 0.0],
  "delta_gripper": 0.0,
  "rationale": "Move toward the committed orange pad in two bounded stages."
}
```

## Audit artifacts

Each run under `agent_runs/` contains:

- `manifest.json`: selected Level, exact capability set, bounds, budget, and seed commitment;
- `agent_transcript.jsonl`: complete public command/observation/response history;
- `observations/`: only images public at the selected Level;
- `agent_observations_h264.mp4`: H.264/yuv420p video made only from public panels;
- `evaluator_outcome.json`: terminal classification, committed-pad, and official-success metrics;
- `evaluator_private_audit.json`: private checker events, created only when the episode terminates.

The API prevents accidental leakage. A truly adversarial agent must additionally
run under a different UID/container because a same-UID coding agent could inspect
simulator memory or private files outside the API.

## Acceptance

Fast protocol, state-machine, serialization, and no-leak checks:

```bash
./scripts/accept_agent_env.sh
```

Real Isaac/TacEx launch and a complete black-box protocol smoke episode:

```bash
./scripts/accept_agent_env.sh --real --level 3 --device cuda:0
```

The real check verifies launch, public modalities, irreversible prediction,
post-prediction feedback gating, terminal artifacts, private-file permissions,
and H.264 encoding. Run it once per Level for a release acceptance matrix.

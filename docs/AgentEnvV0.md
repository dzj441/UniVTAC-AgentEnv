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
process and loads the full NVIDIA userspace stack—including `libcuda`, NVML,
EGL/GLX vendor libraries, OptiX, and video libraries—from
`runtime-libs-full/`. Generic GLVND/Vulkan loaders may still come from the Conda
environment, but no NVIDIA driver library comes from `/usr`. The launcher sets
the bundle's Vulkan/EGL manifests, checks `/proc/driver/nvidia/version`, performs
a CUDA Driver API smoke test, and audits actual mapped library paths both before
Kit startup and inside the live Isaac process. The audit is stored in
`manifest.json`.

The host must still provide a matching 570.124.06 kernel module and GPU device
nodes; a userspace bundle cannot replace those. The launcher never installs or
changes a kernel module, CUDA toolkit, or NVIDIA package. To use another machine,
point `UNIVTAC_NVIDIA_RENDER_ROOT` at an exact-match bundle; if its directory
basename is not the driver version, also set `UNIVTAC_NVIDIA_RENDER_VERSION`.

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

The initial `ready` response and `manifest.json` include the complete JSON
Schema for every request. Unknown or missing fields are rejected before they
can consume a physical-action slot. In particular, the accepted motion request
keys are `delta_position`, `delta_rpy`, and `delta_gripper`; response records
use more explicit unit-bearing names such as `delta_position_world_m`.
`finish` requires a non-empty `final_note`.

For a controlled evaluator matrix, the launcher process may receive one fixed
seed through the reserved `UNIVTAC_EVALUATOR_SEED` environment variable. The
runner consumes and removes that variable immediately. The agent-facing API
still exposes only the seed commitment before termination; the seed and salt
are revealed only in the terminal outcome. This mechanism is evaluator-only
and must never be placed in an agent prompt or public live-episode artifact.

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

The manifest also records the selected NVIDIA bundle/version/library directory
and every NVIDIA driver library mapped by the live Isaac process. Acceptance
fails if any such library resolves outside the declared bundle.

The API prevents accidental leakage. A truly adversarial agent must additionally
run under a different UID/container because a same-UID coding agent could inspect
simulator memory or private files outside the API.

## Isolated Codex operator

The host-side operator now launches Codex through `codex app-server --stdio`
and gives it a fresh empty workspace and a temporary `CODEX_HOME`. Only
`auth.json` is copied into that home; config, prior sessions, memories, MCP
servers, plugins, skills, goals, and repository files are absent. The temporary
home, including its authentication copy, is removed when the run ends.

Codex does not receive the simulator's generic JSON transport. The host builds
a Level-specific immutable dynamic-tool registry and relays only registered
calls. The registry exposes bounded gripper probes, the irreversible
classification commitment, bounded delta-EE control, physics wait, finish, and
public status. It contains no IK, joint-target, object-pose, trajectory-planner,
shell, file, web, app, plugin, skill, or subagent capability. Every call is
validated by the host gateway and then validated independently by the simulator
protocol. A shell/file/MCP/web/subagent or other non-embodied Codex event
invalidates the episode instead of being approved.

The environment's internal controller may implement an accepted bounded
delta-EE command using IK. That is an implementation detail behind the
simulator boundary: the agent cannot invoke the IK solver, choose its solution,
inspect its state, or send joint targets.

Run one episode as follows:

```bash
./scripts/run_codex_agent_env.py \
  --level 1 \
  --device cuda:0 \
  --run-dir agent_runs/example_level1
```

Use `--model MODEL` to pin a Codex model and `--effort high` (or another
supported effort) to pin reasoning effort. `--dry-run` prints the exact public
capability manifest without launching Codex or Isaac. Run Levels serially on
this Isaac configuration; each simulator process uses `cuda:0` internally.

### Codex audit artifacts

In addition to the simulator artifacts listed above, the operator writes:

- `codex_run_manifest.json`: model, effort, prompt/tool hashes, source revision,
  isolation settings, and threat model;
- `codex_capabilities.json`: exact Level-specific dynamic-tool contracts and
  their SHA-256 commitment;
- `codex_operator_prompt.txt`: exact model task prompt;
- `codex_app_server_events.jsonl`: every published app-server request,
  notification, and response, with image payload bytes omitted but hashed;
- `codex_tool_calls.jsonl`: exact dynamic-tool arguments, translated simulator
  command, scrubbed response, latency, and observation-id transition;
- `codex_decisions.jsonl`: structured evidence, alternatives, uncertainty,
  expected effect, exact parameter justification, and chosen command;
- `codex_messages.jsonl`: published agent messages and reasoning summaries;
- `capability_violations.jsonl`: fail-closed attempts to leave the embodied
  capability surface;
- `CODEX_TRACE.md`: a Chinese-readable, image-linked join of decisions, actions,
  observations, feedback, messages, and terminal metrics;
- `codex_run_outcome.json`: whether the episode is valid for scoring, terminal
  capability audit, resolved model/reasoning settings, token usage,
  accepted/rejected tool-call counts, and total wall time;
- `codex_app_server_stderr.log` and `simulator_stdout.log`: process diagnostics.

Every JSONL stream is append-only during the run and contains a per-event hash
chain. The existing `agent_observations_h264.mp4` remains the browser-compatible
H.264/yuv420p visual replay. These artifacts record every event Codex publishes
and every explicit decision required by the tool schema. They cannot expose
model-internal hidden chain-of-thought that the Codex protocol does not publish.
Host-rejected attempts retain a decision record with `chosen_action=null`, so
the rationale remains reviewable while the audit proves no simulator command
was sent. The video has one frame per public observation; it is not yet a
continuous physics-step recording.

Tactile marker health is a reset-time rendering requirement and a per-frame
diagnostic thereafter. A heavily loaded contact may legitimately merge marker
components; the runner still publishes that observation after the action. It
does not report `command_error` after the physical world has already changed.

### Threat model

The implemented boundary is suitable for a non-adversarial benchmark agent:
the model sees only registered dynamic tools, has an empty read-only workspace,
and never receives a simulator handle. This machine does not permit user/mount
namespace creation. Therefore a deliberately malicious same-UID native process
is not contained by this runner; a security-grade deployment must additionally
place the Codex process under a separate UID or external container. The run
manifest records this limitation rather than claiming container isolation.

The design follows OpenETA's host-owned immutable `ToolSpec`/registry pattern
and execution-time validation, while replacing OpenETA's unwired legacy Codex
adapter with Codex app-server dynamic tools and a complete event recorder. See
[Codex runner design](CodexAgentRunner.md) for the detailed mapping.

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

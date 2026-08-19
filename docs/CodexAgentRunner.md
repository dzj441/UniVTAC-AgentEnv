# Codex embodied-agent runner

This document describes the boundary between Codex, the benchmark host, and the
Isaac/TacEx simulator for `grasp_classify`.

This is the legacy v0 runner contract. Its embodied-only capability boundary
is intentionally preserved for compatibility. The generic v1 task runner uses
a fresh temporary workspace while inheriting evaluator-configured Codex
capabilities and keeps only robot control behind the three dynamic tools; see
[`EmbodiedBenchmarkV1.md`](EmbodiedBenchmarkV1.md). Statements below about
disabling shell, files, network, plugins, skills, or subagents apply only to
`scripts/run_codex_agent_env.py`, not `scripts/run_codex_benchmark.py`.

## Why this boundary exists

A prompt saying “do not inspect the simulator” is not an evaluation boundary.
The benchmark must define which observations and actions exist, project only
those capabilities to the agent, reject everything else at execution time, and
retain enough evidence to reproduce every externally visible decision.

The runner therefore separates three authorities:

```text
Codex app-server
  │  Level-scoped dynamic tool call
  ▼
CapabilityGateway
  │  validated AgentEnv JSON command
  ▼
Isaac/TacEx AgentEnv process
  │  rich host response (audit only)
  └──────────────────────────────────────────────► CapabilityGateway ► Codex
                                      minimal sensor/state projection
```

Codex never receives the simulator subprocess, stdin/stdout file descriptors,
run-directory path, repository workspace, evaluator-private audit, seed before
terminal evaluation, or a generic shell from this pipeline.

## What was adopted from OpenETA

The adjacent OpenETA repository establishes useful architectural principles:

- `agent/tools/registry.py` makes `ToolSpec` host-owned and immutable and binds
  executable handlers separately;
- each tool declares its effect, allowing world-mutating calls to force a fresh
  observation before another action;
- the planner is built from executable registered tools rather than allowing a
  model-authored skill to create tools;
- execution-time gates revalidate a tool call even after the model selected it;
- the episode logger separates observation, action, result, latency, and
  terminal metadata.

OpenETA's `adapter/codex_cli.py` is explicitly a legacy placeholder and raises
`NotImplementedError`; it does not solve Codex conversation capture. UniVTAC
uses the registry/gate ideas but connects Codex through app-server dynamic
tools. The app-server event stream provides authoritative `item/started`,
`item/completed`, tool request, agent-message, reasoning-summary, and terminal
turn events.

## Level capability matrix

All three Levels deliberately share the same robot action space and budget, so
only the information intervention changes.

| Capability | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| Head RGB | yes | yes | yes |
| Wrist RGB | yes | yes | yes |
| Basic robot state | yes | yes | yes |
| Left/right tactile marker RGB | no | yes | yes |
| `task_success` before commitment | no | no | no |
| `task_success` after commitment | no | no | yes |
| Direct IK/joint target/planner | no | no | no |
| Shell/files/web/MCP/subagent | no | no | no |

The tool names are stable across Levels, but their observation payloads and
`decision_record.evidence.source` enums are generated from the selected
profile. A Level 1 call that claims tactile evidence is rejected before it
reaches the simulator. A Level 2 response containing `task_success` is treated
as an environment capability leak. Level 3 permits that field only after the
irreversible target commitment.

## Registered tools

- `start_episode`: reset exactly once and return the first public observation;
- `probe_gripper`: at most two nonzero gripper-only probes before commitment;
- `commit_classification`: irreversibly commit `rough→orange` or `plain→green`;
- `act_delta_ee`: bounded world-frame translation, rotation, and gripper delta;
- `wait_physics`: advance 1–60 physics steps without a control delta;
- `finish_episode`: request terminal evaluation at the current state;

`inspect_episode_status` is deliberately not registered for the agent. Host
status remains available for orchestration and audit without entering the
Codex context.

There is intentionally no `solve_ik`, `move_joints`, `get_object_pose`,
`get_depth`, `read_file`, `python`, or `shell` tool. Unknown fields are rejected
instead of silently becoming no-ops.

## Required decision record

Every probe, commitment, physical action, wait, and finish call contains:

- evidence entries tied to sources actually available at that Level;
- alternatives considered;
- uncertainty in `[0, 1]`;
- expected observable effect;
- rationale for the exact chosen magnitude rather than a vague direction;
- concise overall rationale.

The host stores the record before returning the resulting observation. Failed
and rejected calls remain visible in the raw, tool-call, and decision streams;
a rejected call has `chosen_action=null`, proving it did not reach the
simulator. Exact rejection causes stay host-side; Codex receives only a generic
contract rejection so it cannot probe hidden state through error text. This is
an explicit, reviewable rationale—not hidden model chain-of-thought.

The task turn prompt contains only the task name and its semantic goal. It does
not identify the Level, available modalities, strategy, action budget, target
direction, success availability, or recommended sequence. Units and bounds
remain in the dynamic-tool schemas because they define valid actions rather
than solve the task.

## Process isolation

For each rollout, `IsolatedCodexEnvironment` creates:

- a fresh empty working directory;
- a fresh home and `CODEX_HOME`;
- only a temporary copy of `auth.json`;
- no inherited Codex config, sessions, memories, MCP servers, plugins, skills,
  goals, or repository instruction files;
- a reduced child environment without evaluator seed or project `PYTHONPATH`.

Codex runs read-only with network access disabled for the turn. Shell,
unified-exec, multi-agent, browser, computer-use, image-generation, plugin,
skill, hook, and related features are disabled on app-server startup. The host
also uses an explicit item-type allowlist: corresponding forbidden items and
future unknown item types fail closed. A configuration or protocol regression
is therefore observable and makes the episode invalid.

The current container denies user/mount namespace creation. Consequently this
is a benchmark capability boundary, not a hostile same-UID process security
boundary. Production deployment should wrap the Codex process in a separate
UID/container while keeping the same dynamic-tool gateway.

## Recording and validity

The recorder keeps raw app-server traffic, normalized tool calls, structured
decisions, published messages/reasoning summaries, and violations in separate
hash-chained JSONL files. Image bytes sent to Codex are omitted from the raw
event log to avoid duplicating large PNGs; their byte count and SHA-256 remain,
and the canonical PNG already exists under `observations/`.

`codex_run_outcome.json` also records the app-server-resolved model, reasoning
settings, cumulative/last-turn token usage, accepted/rejected tool-call counts,
and end-to-end wall time. `CODEX_TRACE.md` is a Chinese, image-linked review of
the complete published interaction. The evaluator-owned
`agent_observations_h264.mp4` contains one sensor composite per public
observation. After the Codex turn is complete, the host additionally builds
`agent_timeline_h264.mp4`: the same composite on the left, with every tool call
based on that observation on the right. It includes exact compact arguments,
execution target, accepted/rejected status, returned observation id, and a
bounded preview of the public decision rationale. Neither file is a continuous
every-physics-step recording.
The timeline uses 1 FPS so the compact call panel is readable during normal
playback; the sensor-only replay retains the existing 2 FPS encoding.

The mapping uses the gateway-recorded `prior_observation_id`, not the id typed
by the model. Consequently a malformed/stale-id attempt remains visible beside
the observation the agent actually had, and is marked `HOST REJECTED`. Calls
that do not create a new observation, such as classification commit followed
by control, share the same frame.
The timeline is a supplemental post-visualization artifact; the browser Viewer
continues to use `agent_observations_h264.mp4`. It can also be regenerated from
any completed Codex run without launching the simulator:

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/render_agent_toolcall_video.py \
  agent_runs/<run-directory>
```

An episode is `valid_for_scoring=true` only when:

1. the simulator reached its terminal state through an agent-visible tool;
2. `evaluator_outcome.json` exists;
3. no live capability violation was observed;
4. the post-run event audit finds no forbidden Codex item or approval request.

The host alone sends the final simulator `close` command. It never adds a robot
action, prediction, recovery, or finish on behalf of an agent that stopped
early.

## Usage and acceptance

Inspect an exact Level without launching expensive processes:

```bash
./scripts/run_codex_agent_env.py --level 2 --dry-run
```

Run one real episode:

```bash
./scripts/run_codex_agent_env.py \
  --level 2 \
  --device cuda:0 \
  --model MODEL \
  --effort high \
  --run-dir agent_runs/grasp_classify_codex_level2
```

Run the static suite, including a fake app-server end-to-end rollout:

```bash
./scripts/accept_agent_env.sh
```

The static suite does not consume model quota or start Isaac. A release matrix
should additionally perform one serial real rollout per Level and inspect
`codex_run_outcome.json`, `CODEX_TRACE.md`, and the H.264 replay.

## Current experiment boundary and next extensions

The three-Level matrix is now executable and auditable, but it is still one
task vertical slice. The next benchmark work should prioritize task design over
adding more generic robot APIs:

1. define several task families and difficulty axes using the same capability
   registry (perception ambiguity, contact need, recovery need, and horizon);
2. build evaluator-owned matched-reset triples so Levels 1/2/3 see exactly the
   same initial state without exposing the seed;
3. add richer Level-3 guidance categories only after irreversible commitment,
   such as `not_grasped`, `wrong_region`, or `unstable`, while keeping labels
   and geometry private;
4. add continuous physics-step video only if dynamics between decisions matter;
5. move the Codex process to a separate UID/container for adversarial-agent
   claims, and reduce repeated image/context cost before scaling the matrix.

## Protocol references

- [Codex non-interactive JSONL events](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex app-server threads, turns, dynamic tools, and streamed events](https://learn.chatgpt.com/docs/app-server)

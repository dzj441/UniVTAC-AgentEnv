# Public Observation and ICL Data Contract v1

Status: v1 contract with the static ICL projector and reference-runner
integration implemented. Direct maximal P6 recording in the normal collection
and replay paths remains follow-up production work.

This document separates the benchmark's internal assets, the static expert
demonstration made available to an Agent, and observations produced during a
live episode. They share one public observation definition but have different
storage and delivery rules.

## Responsibility boundary

UniVTAC enforces observation-contract integrity, not adversarial Agent
containment.

The benchmark core is responsible for:

- projecting exactly the modalities allowed by P1--P6;
- applying the independent bbox and mask switches at the documented time;
- omitting private simulator, planner, evaluator, and source-asset data;
- exposing robot control only through `start_episode`, `step_eef`, and
  `finish_episode`;
- recording the public interaction and the declared runtime conditions.

The benchmark core does not decide whether shell, Python, `view_image`, web,
plugins, skills, MCP, or multi-Agent use is cheating. An evaluator may choose
its own network, sandbox, filesystem, and policy restrictions. The reference
runner should not disable general Agent capabilities on behalf of the
benchmark.

A clean temporary workspace remains useful for run-to-run data separation and
reproducibility. Workspace isolation and Agent capability restriction are
separate concerns: the former may be enabled without replacing or disabling
the evaluator's normal Agent configuration.

The current reference host cannot use Codex's bubblewrap-based
`workspace-write` sandbox reliably, so the reference runner temporarily
defaults to `danger-full-access`. In this mode the temporary workspace is a
data-layout boundary, not containment, and Codex permits network access.
UniVTAC records both the requested and effective network setting together with
the declared sandbox and native App Server activity, but does not reject general
Agent activity as hacking; containment and adjudication belong to the
evaluator. Restoring a validated bubblewrap/`workspace-write` deployment is a
runtime TODO, not a benchmark data-contract prerequisite.

## Three data planes

### Internal expert observation master

The host-side master is the maximal, authenticated source used for later
projection. New masters must contain:

- every P6 modality at every recorded expert waypoint;
- public robot state at every waypoint;
- head and wrist camera calibration, including dynamic wrist extrinsics;
- anonymous head and wrist bbox and mask annotations at the initial waypoint;
- the original motion provenance and an independently verified terminal
  success result.

The master is not Agent-ready. It may contain absolute host paths and private
verification evidence. Those fields never pass through to the Agent bundle.
The expert trajectory is a sequence of observed states or waypoints, not a
record of `step_eef` calls. It must not claim or synthesize actions that were
not recorded.

For new data, collection and independent replay should both record the maximal
observation payload directly. The successful replay master is the authority
for Agent-visible demonstration observations. A later sensor-enhanced replay
is retained only as a migration path for historical successful trajectories.

### Static ICL expert bundle

When `icl=fixed_demo`, the host projects the expert master once into an
Agent-visible directory before the Agent starts. The original HDF5 and the
host-side master are not placed in the workspace.

```text
benchmark_inputs/
└── expert_demo/
    ├── manifest.json
    ├── trajectory.jsonl
    ├── state.jsonl
    ├── camera_intrinsics.json
    ├── camera_extrinsics.jsonl
    ├── overview/
    │   └── contact_sheets/
    └── frames/
        ├── frame_000000/
        │   ├── observation.json
        │   ├── head/
        │   ├── wrist/
        │   ├── tactile/
        │   └── annotations/
        ├── frame_000001/
        └── ...
```

All paths inside the bundle are relative to `expert_demo/`. The manifest
contains one episode-level label, `successful expert demonstration`. It never
contains per-step success, the first-success time, checker details, evaluation
seed lists, actor poses, raw instance metadata, planner targets, contact
points, IK state, or private joint targets.

The Agent is told only that a verified demonstration exists at
`benchmark_inputs/expert_demo/`. Frames are not automatically inserted into
the initial prompt. The Agent may inspect, script against, transform, or ignore
the bundle using any capabilities available in its runtime.

The bundle also contains deterministic per-modality contact sheets. The
`uniform_endpoint_preserving_v1` rule chooses at most 12 frames and always
keeps both endpoints. This is a browsing aid, not an additional observation:
every tile comes from a frame already present in the projected bundle.

### Live episode observations

A live observation is actively returned to the Agent and may also be recorded
in an evaluator-private audit area:

```text
simulator observation
        |
        v
profile and annotation projection
        |
        v
PublicObservationFrame
        +--> AgentTransport
        +--> PrivateAuditRecorder
```

`start_episode` returns `obs_000`. Every accepted `step_eef` returns a fresh
observation. `finish_episode` may return a final post-settle observation and is
the only response allowed to reveal terminal success. Its Agent-visible result
contains the official success bit but not evaluator/checker diagnostics or the
seed/salt commitment opening; those terminal details remain evaluator-private.

Current state, calibration, annotations, and relative artifact references are
included in the JSON tool result. Current visual modalities are also sent as
image content so that the Agent does not need an extra file-reading call to see
the observation.

The benchmark does not provide an Agent-visible online history. File-only
modalities such as metric `.npy` depth may be exposed through a current-frame
artifact directory or an equivalent blob handle. A filesystem implementation
must atomically replace that directory for every new observation and must not
leave prior frames browsable. If the Agent wants a history, it must preserve
one itself using its own context or tools. The evaluator may retain a complete
private interaction record outside the Agent workspace for replay and audit.

## Common public frame definition

Static expert frames and live observations use the same field names, units,
camera conventions, and artifact formats. Their identifiers differ only by
namespace: `frame_000000` for a demonstration and `obs_000` for a live run.

### State

In a static expert bundle, state is stored once in an ordered
`state.jsonl`:

```json
{"observation_id":"frame_000003","record_index":3,"relative_sim_step":60,"joint_position_9d":[],"joint_velocity_9d":[],"gripper_width_m":0.06,"end_effector_pose_robot_base_wxyz_7d":[]}
```

Every expert frame's `observation.json` contains a relative `state_ref` with
the file and record index instead of duplicating the state vector. A live tool
response instead inlines the current state for immediate use; it does not
publish a growing `state.jsonl` to the Agent. The evaluator-private audit may
use the same JSONL representation without making it Agent-visible.

The 9D joint position and velocity, gripper width, and robot-base EEF pose are
public state. Planner output, IK diagnostics, requested joint targets, and
private controller internals are not public state.

### Camera calibration

Static P5 and P6 expert bundles store fixed intrinsics once in
`camera_intrinsics.json`. A static P6 bundle stores one head and wrist
extrinsic record per observation in `camera_extrinsics.jsonl`. Per-frame
records are required because the wrist camera moves with the robot. Each
expert frame references the corresponding calibration record. A live tool
result instead inlines the current enabled calibration; any history remains in
the evaluator-private audit.

Extrinsics use a single named convention:

```text
matrix_T_robot_base_camera_ros_4x4
ROS optical camera axes: +Z forward, -Y up
translation unit: metre
quaternion order where present: wxyz
```

### Frame artifacts

The physical files are:

| Public modality | File representation |
| --- | --- |
| head RGB | `head/rgb.png` |
| wrist RGB | `wrist/rgb.png` |
| left tactile marker RGB | `tactile/left_rgb_marker.png` |
| right tactile marker RGB | `tactile/right_rgb_marker.png` |
| metric depth | `depth_m.npy`, float32 metres |
| depth validity | `depth_valid_mask.png` |
| depth display | `depth_visualization.png` plus explicit display range |
| bbox | exclusive `[x1, y1, x2, y2]` in annotation JSON plus overlay PNG |
| mask | single-channel PNG, background 0 and foreground 255, plus overlay PNG |

Large numeric depth arrays are never embedded in JSON. Artifact metadata
records relative path, SHA-256, media type, dtype, shape, unit, validity rule,
and visualization range as applicable.

### GelSight wrist-depth surface policy

The public P5/P6 contract is:

> Wrist metric depth includes the rigid gripper and GelSight housing, but
> excludes the deformable optical gel surface.

The TacEx GelSight Mini asset authors
`primvars:invisibleToSecondaryRays=true` on both the rigid `case`/`plate`
meshes and the deformable gelpad meshes. Leaving all six meshes hidden makes
the wrist depth annotator see through mechanically rigid parts of the gripper.
UniVTAC therefore applies a runtime USD session-layer override after scene
cloning and before sensor initialization or the first reset/render:

- the four left/right rigid `case/mesh` and `plate/mesh` prims are set to
  `false`, so they participate in metric depth;
- the two deformable `gelpad_*/mesh` prims are explicitly kept `true`.

The override resolves roots from the configured GelSight sensor and gelpad
paths, validates exact child paths and expected per-environment counts, and
fails closed for inactive, unloaded, non-Mesh, instance-proxy, prototype, or
wrong-typed targets. It is authored only in the stage session layer; the
vendored TacEx USD remains unchanged.

This split is deliberate. A same-state closed-key A/B probe recovered 9,979
nearer wrist-depth pixels when only the rigid housing became visible, with a
median recovered surface-distance difference of about `0.03459 m`. Both
GelSight sensors' `camera_depth`, `height_map`, `tactile_rgb`, and
`marker_motion` remained elementwise identical. In contrast, making the
deformable pad visible changed every pixel in the tactile height map by about
`6.6 mm`, because the same deformable optical surface participates in the
internal GelSight rendering pipeline. UniVTAC consequently does not claim
pixelwise RGB/depth agreement on that optical gel surface.

The pre-fix Formal16 rollout remains a useful historical regression anchor at
`formal16_b21184e_seed_1830315042/07_key_p6/obs_012`: it shows the larger hole
created when the rigid housing was also hidden. Initial frames may conceal the
problem because nearby background geometry has a similar range and each depth
preview is normalized independently. The metric `.npy` data, rather than only
the colorized preview, is authoritative.

The public serializer itself does not alter metric values: it copies
`camera.data.output["depth"]`, removes singleton camera dimensions, and casts
to `float32`. P5/P6 observations or expert masters captured before the runtime
override retain the legacy rendering and must be replayed/exported before use
in depth-sensitive comparisons. Their authenticated expert motion and success
proof do not need to be recollected.

## Profile projection

Projection is physical: a disabled modality has neither a JSON field nor a
file in the Agent workspace.

| Profile | Agent-visible data |
| --- | --- |
| P1 | head RGB and public robot state |
| P2 | P1 plus wrist RGB |
| P3 | P1 plus left and right tactile marker RGB |
| P4 | head RGB, wrist RGB, tactile RGB, and state |
| P5 | P4 plus head/wrist metric depth and intrinsics |
| P6 | P5 plus head/wrist extrinsics |

BBox and mask are independent switches, not additional profile levels. The
projector validates the output allowlist after materialization. It must fail
closed if a requested field cannot be produced.

## Initial-only annotation contract

Both switches use the fixed schedule `initial_observation_only`:

- in a live episode, annotations may appear only in the `obs_000` returned by
  `start_episode`;
- in an expert bundle, annotations may appear only in `frame_000000`;
- later frame JSON, files, composites, videos, and tool responses contain no
  annotation coordinates, masks, or overlays.

The two public roles are exactly `manipulated_object` and `goal_fixture`.
Instance IDs, raw labels, private prim names, and USD paths are host-only.

If bbox is enabled, initial annotation JSON contains exclusive pixel bounds
and the initial frame contains deterministic bbox overlays. If mask is
enabled, it contains the binary mask files and deterministic overlays. If both
are enabled, both representations are present. A maximal internal master
captures both once so all four switch combinations can be projected without
another simulator run.

Initial-only annotation provides object/goal localization while leaving all
subsequent tracking to the Agent.

## Static trajectory index

`trajectory.jsonl` is an index, not an action log. One record identifies the
public frame, state record, and physics-step offset from the first demonstrated
waypoint:

```json
{"frame_index":12,"relative_sim_step":240,"observation":"frames/frame_000012/observation.json","state_record_index":12,"representation":"observed_expert_waypoint"}
```

The EEF waypoint itself lives in the referenced state record. There is no
`step_eef`, action delta, planner target, or per-frame outcome field. The
authenticated masters do not define a public wall-clock rate, so the exporter
does not invent seconds from a presumed simulation frequency.

## Agent transport

For a live observation, transport sends:

- one JSON object with the current state, enabled calibration, initial
  annotations when applicable, and relative artifact identifiers;
- enabled RGB and tactile images;
- enabled metric-depth visualizations;
- initial bbox/mask overlays when their switches are enabled.

This public payload is independent of the Codex turn policy. In `single_turn`
mode it is returned as dynamic-tool content. In `action_per_turn` mode the host
returns only a non-strategic acknowledgement to the dynamic tool, ends that
turn, and sends the exact same public payload as top-level multimodal input in
the next turn of the same Codex thread. The latter creates no new simulator
episode and does not add observation history.

Metric `.npy` depth is exposed through a current-frame relative artifact path
or an equivalent binary handle because it cannot be attached as image content.
When paths are used, the referenced files are replaced on the next observation
instead of accumulating into an Agent-visible history. The response must
describe every current artifact. Its bytes must be identical to the public
frame bytes retained in the evaluator-private audit.

## Shared contract, two delivery paths

The static projector and live gateway both use the frozen Profile and
annotation registries, the same public field names, and exact artifact
validation. Their outputs have two different consumers:

- `AgentTransport` inlines current numeric data, sends current display images,
  and publishes only current-frame file artifacts when necessary;
- `PrivateAuditRecorder` writes the complete interaction outside the Agent
  workspace.

The current implementation keeps the offline projector and online transport as
separate code paths because their source schemas and retention rules differ.
Their cross-path contract is enforced by tests. A later refactor may introduce
one `PublicObservationSerializer`, but the class name is not part of the public
format.

## Implemented fixed-demo workflow

The projector authenticates the registered P6 master manifest before reading
any frame, validates every source JSON/artifact hash, copies only allowlisted
files (never symlinks), rewrites every path relative to the bundle, validates
the completed bundle, and then publishes it atomically. The evaluator-private
projection receipt records the source master path/hash and projected bundle
hash; this receipt is not copied into the Agent workspace.

Export one bundle for inspection:

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/export_fixed_demo.py \
  --fixed-demo-root /path/to/expert_observation_master \
  --task pull_out_key --profile 6 \
  --provide-bbox --provide-mask \
  --output /tmp/pull_key_fixed_demo
```

Validate the complete registered matrix (two tasks, P1--P6, four annotation
conditions):

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/validate_fixed_demo_assets.py \
  --fixed-demo-root /path/to/expert_observation_master \
  --summary-only
```

The reference runner selects the diagnostic condition with
`--icl none|fixed_demo`. `none` creates no expert directory and does not mention
one to the Agent. `fixed_demo` projects the matching task/profile/annotation
bundle into the temporary workspace before the Agent thread and adds only a minimal
discoverability notice. It is incompatible with `--pre-move` because the
registered demonstrations begin ungrasped. The operator prompt otherwise contains
only the task instruction; the ICL condition adds one sentence naming
`benchmark_inputs/expert_demo/`. Tool lifecycle semantics live in the static base
instruction and tool descriptions, not in task-specific strategy text. The
UniVTAC base instruction contains only the three-tool lifecycle and terminal
success visibility. It also states the transport invariant that the Agent must
wait for a robot call's resulting observation before issuing another robot
call. The runner sends no additional UniVTAC developer instruction.

The fixed demonstration's private collection seed is also excluded from the
same-task evaluation condition. Default evaluator seeds already come from a
disjoint high random range; if an evaluator explicitly supplies the registered
demo seed, a non-dry `fixed_demo` run fails before simulator startup. The seed
itself remains absent from the public bundle and prompt.

## Production pipeline

The final production path is:

1. collect a successful full pre-move expert trajectory while recording the
   maximal P6 payload and initial anonymous annotation source;
2. independently replay the same motion from a fresh ungrasped reset while
   recording the same maximal payload;
3. require complete motion execution, terminal checker success, at least 60
   settling steps, and task-specific release/stability checks;
4. freeze the replay observation master and content hashes;
5. project an Agent bundle for the requested
   `profile x bbox x mask x icl` configuration.

HDF5 may remain an internal collection container. It is never the public Agent
format. New collection and replay commands should produce the observation
master directly so a normal asset no longer needs a separate sensor-enhanced
replay pass.

## Acceptance requirements

A release is valid only if automated checks establish all of the following:

- every JSON object has an exact allowlisted schema;
- every referenced artifact exists, stays inside the bundle, and matches its
  recorded SHA-256;
- no absolute host path occurs in an Agent-visible file;
- disabled Profile modalities have no fields and no files;
- annotations are absent after the initial frame;
- bbox and mask independently obey their switches;
- masks are single-channel binary PNGs and bbox coordinates are exclusive;
- public annotations contain only anonymous public roles;
- live tool results/current artifacts and private recorded public frames refer
  to identical bytes;
- prior online frames are absent from the Agent-visible current-artifact area;
- the expert master covers every recorded source waypoint and retains its
  independent terminal-success proof;
- no expert frame claims an unrecorded action.

## Implementation gap audit for the P6 master milestone

| Area | Current status | Required change |
| --- | --- | --- |
| P1--P6 live sensing | Implemented | Preserve behavior |
| live JSON and image delivery | Current JSON and display images are sent directly; every image is announced with its complete public JSON field path | Preserve Profile filtering and semantic labels |
| live filesystem recording | Private per-frame JSON/PNG/NPY plus current-only Agent artifacts implemented | Preserve replacement and hash checks |
| live state | Inlined in every tool result | Preserve; do not publish an Agent-visible online state history |
| live calibration | Inlined in every applicable tool result | Preserve; keep any calibration history private |
| live annotations | Initial-only schedule and current-file replacement tested | Add real-run regression when simulator testing is available |
| annotation masks | Single-channel binary PNG implemented | Preserve exact mode/value checks |
| bbox display | Initial overlay is marked as transportable image content | Preserve initial-only transport |
| expert P6 master | Complete P6 plus initial head/wrist bbox/mask and authenticated wrist-depth policy | Treat suffixed v3 `expert_observation_master/` assets as the projection source |
| expert action semantics | Correctly marked as observation waypoints | Preserve; do not add `step_eef` conversion |
| static ICL bundle | Implemented with exact validation and atomic publication | Freeze schema only after rollout review |
| new expert collection | Historical source is RGB-focused | Record maximal P6 and initial annotation directly |
| independent replay | P6 migration mode exists | Make maximal observation recording a normal replay product |
| Codex general capabilities | Generic v1 runner inherits evaluator configuration and does not disable built-ins; general activity is audited rather than prohibited by the benchmark | Keep runtime policy evaluator-controlled and recorded |
| Codex sandbox | Reference runner defaults to `danger-full-access` because the current host's bubblewrap path is unavailable | TODO: install and validate bubblewrap before treating `workspace-write` as a supported containment mode |
| tool rejection feedback | Public rejection text is intentionally generic | TODO: expose safe, non-leaking error categories without checker, planner, or private-state details |
| Codex action narration | `rationale`, `decision_record`, `agent_note`, and `final_note` removed from generic v1 tools | Record native App Server activity instead of forcing duplicate prose |

The two replay-proven demonstrations now have v3 maximal masters with the
initial annotation source and the rigid-housing/deformable-gel depth contract.
Their 48 public projections have been validated. Unsuffixed v2 assets retain
the pre-fix wrist-depth rendering only as regression evidence. The remaining
production data work is to make maximal P6 plus initial annotations a normal
output of collection and replay for future assets.

## Recommended work split

With projection and runtime integration completed, development can proceed
independently:

1. ICL path: run real Codex rollouts with the minimal rationale-free prompt and
   review demonstration usability;
2. collection/replay path: make maximal P6 plus initial annotation a normal
   product of both stages and remove the extra migration pass for new assets;
3. runtime path: retain the implemented clean temporary workspace for data
   separation and current-only file publication, validate bubblewrap as a
   future optional containment mode, and extend real-run acceptance coverage.

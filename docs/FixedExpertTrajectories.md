# Fixed Expert Trajectories

This pipeline creates the single positive expert demonstration used by the
initial `icl=none|fixed_demo` diagnostic axis.  It is a scripted/cuRobo expert
trajectory, not a human teleoperation trajectory.

Here, "successful expert demonstration" is one terminal label attached to the
whole frozen episode.  It means that a fresh simulator reset replayed the full
trajectory and passed the terminal checker.  It is not per-step success
feedback, and the agent is never shown checker internals or the moment at which
a legacy checker first became true.

## Why a new collection mode is required

Historical UniVTAC collection executes `pre_move()` before saving observations.
Consequently the published HDF5 episodes begin after privileged setup or
pre-grasping and cannot demonstrate the default ungrasped AgentEnv task.

`record_pre_move` is opt-in and leaves historical collection unchanged.  When
enabled, the collector saves:

1. one exact ungrasped observation after actor reset and scene settling;
2. timestamped planner states from `pre_move()` including approach and grasp;
3. the normal `_play_once()` task trajectory.

Every saved frame contains `collection/phase = pre_move|task`.  A valid fixed
expert source has exactly one `pre_move -> task` transition.

## Collect one successful source

Large HDF5/video artifacts should live on the shared data volume:

```bash
UNIVTAC_EXPERT_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/univtac_fixed_experts

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  ../miniconda3/envs/UniVTAC/bin/python scripts/collect_data.py \
  pull_out_key fixed_expert \
  --record-pre-move \
  --episode_num 1 --start_seed 0 --max_seed 100 --gpu 0 \
  --save-dir "${UNIVTAC_EXPERT_ROOT}/source"
```

Replace `pull_out_key` with `put_bottle_in_shelf` for the second fixed source.
The collector searches seeds serially and retains the first checker-successful
episode.  Source artifacts appear below
`source/<task>/fixed_expert/{hdf5,video}/`.

## Replay from the ungrasped state

The upstream replay implementation already performs real simulation replay by
forcing recorded robot qpos into a same-seed reset and rerunning the checker.
For a full expert trajectory, `--trajectory-includes-pre-move` additionally:

- skips privileged `pre_move()` during replay reset;
- initializes only task checker references;
- applies the complete trajectory even if a legacy checker becomes transiently
  true before release;
- reconstructs every recorded simulator step by interpolating between saved
  qpos frames according to their original `step` timestamps;
- settles for 60 physics steps and evaluates task success at the end;
- requires bottle release and pose stability in addition to the legacy checker.

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/replay.py \
  pull_out_key fixed_expert \
  --data-path "${UNIVTAC_EXPERT_ROOT}/source/pull_out_key/fixed_expert/hdf5/SEED.hdf5" \
  --seed SEED --trajectory-includes-pre-move --stride 1 \
  --output-dir "${UNIVTAC_EXPERT_ROOT}/replay/pull_out_key"
```

The fresh replay writes `replay_report/SEED.json`, qpos tracking diagnostics,
and an independently rendered H.264 replay video.

## Freeze a positive ICL asset

Only a source that passes independent ungrasped replay can be frozen:

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/freeze_fixed_expert.py \
  --task pull_out_key --seed SEED \
  --source-hdf5 SOURCE.hdf5 \
  --source-video SOURCE_success.mp4 \
  --replay-report REPLAY_REPORT.json \
  --output FIXED_EXPERT_MANIFEST.json
```

The manifest hashes the HDF5, collection video, replay report, and replay
video.  It exposes only the semantic label "successful expert demonstration";
stepwise checker state and checker details remain private.  Later ICL prompt
assembly must project the P6 master to the active observation Profile and must
strip actor poses, planner targets, contact points, IK/joint internals, and
checker details.

The current frozen assets were collected with wrist-camera commit `c0e6a64`
(optics and visual housing moved together), using the original task-gallery
runtime settings: `decimation=1`, `save_frequency=20`, `video_frequency=2`,
`render_frequency=0`, RGB camera observations, and scene-query support enabled.

## Current replay-proven validation assets

The following assets pass the full collection/replay/freeze pipeline. Both
begin in the ungrasped state and were replayed without calling privileged
`pre_move()`:

| Task | Seed | Source frames | Replayed physics actions | Frozen manifest |
| --- | ---: | ---: | ---: | --- |
| `pull_out_key` | 0 | 28 | 551 | `manifests/pull_out_key_seed_0.json` |
| `put_bottle_in_shelf` | 1 | 44 | 871 | `manifests/put_bottle_in_shelf_seed_1.json` |

The paths in this table are relative to
`/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/univtac_fixed_experts_camera_c0e6a64_20260819_staging/`.
Both replay reports passed the official task checker after 60 settling steps.
The bottle replay additionally verified gripper release and post-release pose
stability. Source HDF5 and both source/replay H.264 videos are content-hashed
by their frozen manifests.

## Derive the P6 observation master

The frozen source remains the authority for expert motion and its independent
success proof. It does not contain complete P5/P6 camera observations, so a
sensor-enhanced replay records one full P6 observation after reaching each
original source waypoint. The final public-data contract is defined in
[`PublicObservationAndICLDataContract.md`](PublicObservationAndICLDataContract.md).

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/replay.py \
  pull_out_key fixed_expert \
  --data-path SOURCE.hdf5 --seed SEED \
  --trajectory-includes-pre-move --stride 1 \
  --capture-p6-master --fixed-expert-manifest FROZEN_MANIFEST.json \
  --output-dir P6_MASTER_OUTPUT
```

This is an observation-state trajectory, not an Agent action trace. Replay
continues to follow the authenticated qpos source at its recorded timestamps;
it neither derives nor claims `step_eef` actions. Each waypoint contains:

- head and wrist RGB plus metric depth, validity masks, and depth previews;
- camera intrinsics and dynamic robot-base camera extrinsics;
- left and right public tactile marker RGB;
- the public joint, gripper-width, and EEF-pose state;
- an inspection composite.

The initial waypoint additionally contains anonymous head/wrist bbox and
single-channel binary mask assets for `manipulated_object` and `goal_fixture`.
No later waypoint contains annotation fields, files, or annotated composites.

The writer authenticates the frozen manifest and source HDF5 before reset. It
then validates the exact P6 schema and hashes every observation artifact before
writing `p6_master_manifest.json`. A master is rejected unless replay reaches
all source waypoints, passes the terminal task checker after 60 settling steps,
and, for the bottle task, also passes release and pose-stability checks.

The v2 P6 manifest is host-side provenance and deliberately has
`agent_ready=false`, `actions_present=false`, and
`step_eef_conversion_performed=false`. It may contain evaluator evidence and
must not be handed directly to an Agent. The implemented fixed-demo projector
authenticates this master and creates a separate, Profile-safe Agent bundle.

## Current maximal expert observation masters

The original frozen HDF5, videos, manifests, and replay reports remain
unchanged. The following derived masters have independently re-run the same
motion and success checks:

| Task | Seed | P6 observations | Replayed physics actions | P6 master manifest |
| --- | ---: | ---: | ---: | --- |
| `pull_out_key` | 0 | 28 | 551 | `expert_observation_master/pull_out_key_seed_0/p6_master_manifest.json` |
| `put_bottle_in_shelf` | 1 | 44 | 871 | `expert_observation_master/put_bottle_in_shelf_seed_1/p6_master_manifest.json` |

These paths are relative to the same staging root shown above. Both v2 masters
contain complete P6 observations at every original saved waypoint and both
independent annotation sources at the initial waypoint only. The earlier
`p6_master/` v1 directories are retained as immutable migration evidence but
are superseded because they do not contain annotation sources.

## Export Agent-visible fixed demonstrations

The host registry pins each task to the expected v2 master manifest SHA-256.
Projection fails closed if the registered manifest, a source observation, or
any referenced artifact has changed. It then physically materializes only the
requested Profile and annotation condition; disabled data has neither a field
nor a file in the output. Absolute host paths, private seed, source manifest,
checker evidence, raw labels, and planner internals are not copied.

```bash
FIXED_DEMO_ROOT=/path/to/expert_observation_master

../miniconda3/envs/UniVTAC/bin/python scripts/export_fixed_demo.py \
  --fixed-demo-root "${FIXED_DEMO_ROOT}" \
  --task pull_out_key --profile 6 \
  --provide-bbox --provide-mask \
  --output /tmp/univtac-pull-key-demo
```

The public output uses `manifest.json`, `trajectory.jsonl`, `state.jsonl`,
optional calibration streams, per-frame artifacts, and deterministic contact
sheets. Its trajectory records `observed_expert_waypoint`; it never claims a
recorded `step_eef` action. BBox/mask are independent and occur only in
`frame_000000`.

Validate every registered public projection:

```bash
../miniconda3/envs/UniVTAC/bin/python scripts/validate_fixed_demo_assets.py \
  --fixed-demo-root "${FIXED_DEMO_ROOT}" --summary-only
```

The formal matrix contains 48 combinations: two tasks, six observation
Profiles, and four bbox/mask conditions. The current two registered masters
pass all 48 projections.

## Select the ICL condition at runtime

`scripts/run_codex_benchmark.py` exposes `--icl none|fixed_demo`. With
`fixed_demo`, it projects the matching public bundle into
`benchmark_inputs/expert_demo/` inside the fresh Agent workspace before the
Codex turn. With `none`, it creates no expert bundle and gives no demo
discoverability notice. Fixed demos are ungrasped and therefore cannot be
combined with `--pre-move`. A non-dry fixed-demo run also rejects an explicitly
selected same-task evaluation seed that equals the registered demonstration
seed; normal randomly generated evaluation seeds are already disjoint.

The runner saves `icl_projection_receipt.json` only in the evaluator run
directory. It contains host-side source provenance and is not visible in the
temporary Agent workspace. Specific Agent prompt wording is intentionally not
frozen by this data-pipeline milestone.

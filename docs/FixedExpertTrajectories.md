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

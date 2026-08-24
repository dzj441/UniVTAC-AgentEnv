# Data Collection

UniVTAC's data synthesizer enables fully automated data collection by executing scripted manipulation policies (defined in the `envs/` directory) in combination with the cuRobo motion planner. Data collection is configured through task-shared configuration files in `task_config/`, which define parameters such as the target tactile sensor type, observation modalities, texture randomization, and the number of episodes to collect.

The pipeline iterates over random seeds, executing the scripted policy for each seed and saving observation data on success. Failed seeds are skipped automatically, and progress is tracked in `suc_map.txt` to support resuming from interruptions. The entire process is fully automated — just run a single command to get started.

Running the following command will start data collection for the specified task:

```bash
bash collect_data.sh ${task_name} ${config_name} ${gpu_id}
# Example: bash collect_data.sh lift_bottle demo 0
```

For faster collection with multiple parallel simulation workers: (Note: the parallel collection is implemented with Python's multiprocessing, so multiple Isaac Sim Apps will be launched on the same time)

```bash
bash parallel_collect.sh ${task_name} ${config_name} ${gpu_id} [num_processes]
# Example: bash parallel_collect.sh lift_bottle demo 0 3
```

All available `task_name` options correspond to Python modules in the `envs/` directory (e.g., `lift_bottle`, `insert_HDMI`, `pull_out_key`, `grasp_classify`, etc.). The `config_name` parameter specifies a YAML configuration file in `task_config/` (without the `.yml` extension). The `gpu_id` parameter specifies which GPU to use (multiple GPUs are supported).

## Task Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `save_dir` | `str` | `./data` | Root directory for saving collected data. |
| `decimation` | `int` | `1` | Physics sub-stepping factor. |
| `save_frequency` | `int` | `2` | Save observations every N simulation steps. |
| `video_frequency` | `int` | `2` | Record video frames every N steps. |
| `render_frequency` | `int` | `0` | Render GUI every N steps (`0` = headless). |
| `random_texture` | `bool` | `false` | Enable random texture domain randomization. |
| `use_seed` | `bool` | `true` | Use deterministic seeding for reproducibility. |
| `episode_num` | `int` | `100` | Number of successful episodes to collect. |
| `sensor_type` | `str` | `gsmini` | Tactile sensor type: `gsmini`, `gf225`, or `xensews`. |
| `observations` | `dict` | — | Which observation modalities to record (see below). |
| `record_pre_move` | `bool` | `false` | Include the initial ungrasped frame and scripted setup/grasp phase. |
| `capture_p6_observations` | `bool` | `false` | Save a sidecar P6 stream with complete depth/calibration and initial-only bbox/mask. |

## Data Structure

After data collection is completed, the collected data will be stored under `data/${config_name}/${task_name}/`:

- Each episode's observation and action data are saved as an individual HDF5 file in the `hdf5/` directory.
- Visualization videos of each episode (combining camera and tactile views) can be found in the `video/` directory.
- Per-episode metadata (step counts, timing, success/failure results) is stored in `metadata.json`.
- The `suc_map.txt` and `scene/` directory are auxiliary outputs generated during the data collection process.

When `capture_p6_observations` is enabled, each successful episode also has:

```text
p6_collection_candidates/SEED/
├── p6_collection_candidate_manifest.json
└── p6_observations/
    ├── demo_obs_000/
    └── ...
```

This sidecar uses the same maximal observation schema as replay: head/wrist
RGB and metric depth, tactile marker RGB, robot state, camera intrinsics and
dynamic extrinsics, plus bbox/mask on `demo_obs_000` only. It is deliberately
marked `agent_ready=false` and `replay_verified=false`. A collector-side
checker pass is not sufficient to publish an ICL asset; the independent replay
and freeze procedure in
[FixedExpertTrajectories.md](FixedExpertTrajectories.md) remains mandatory.

Below is the structure of the saved observation data for each episode (stored in HDF5 format). `HDF5Handler` in `envs/utils/data.py` can be used to read and write this data format:

```json
{
    "actor": {
        "prism": "np.ndarray(7,)",
        "prism_base": "np.ndarray(7,)",
        "slot": "np.ndarray(7,)"
    },
    "atom": {
        "id": "type: <class \"numpy.int64\">",
        "tag": "type: <class \"numpy.bytes_\">"
    },
    "embodiment": {
        "ee": "np.ndarray(7,)",
        "joint": "np.ndarray(9,)"
    },
    "observation": {
        "head": {
            "rgb": "np.ndarray(270, 480, 3)"
        },
        "wrist": {
            "rgb": "np.ndarray(270, 480, 3)"
        }
    },
    "step": "type: <class \"numpy.int64\">",
    "tactile": {
        "left_tactile": {
            "depth": "np.ndarray(240, 320)",
            "marker": "np.ndarray(2, 63, 2)",
            "pose": "np.ndarray(7,)",
            "rgb": "np.ndarray(240, 320, 3)",
            "rgb_marker": "np.ndarray(240, 320, 3)"
        },
        "right_tactile": {
            "depth": "np.ndarray(240, 320)",
            "marker": "np.ndarray(2, 63, 2)",
            "pose": "np.ndarray(7,)",
            "rgb": "np.ndarray(240, 320, 3)",
            "rgb_marker": "np.ndarray(240, 320, 3)"
        }
    }
}
```

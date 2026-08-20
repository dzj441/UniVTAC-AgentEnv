#!/usr/bin/env python3
"""Real-Isaac A/B acceptance for the GelSight wrist-depth policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1102891212)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    args.livestream = 2
    return args


ARGS = parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


# agentic team comment: Isaac/pxr/task imports require a live SimulationApp.
import torch
from pxr import Usd

from envs.pull_out_key import Task, TaskCfg
from envs.robot.gelsight_depth_visibility import (
    SECONDARY_RAY_ATTRIBUTE,
)


TACTILE_CORE_FIELDS = (
    "camera_depth",
    "height_map",
    "tactile_rgb",
    "marker_motion",
)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array.copy()


def make_task() -> Task:
    cfg = TaskCfg()
    cfg.save_dir = ARGS.output_dir / "simulator_internal"
    cfg.scene.num_envs = 1
    cfg.reset_time_limit = 300.0
    cfg.execute_pre_move = True
    cfg.record_pre_move = False
    cfg.decimation = 1
    cfg.obs_data_type = {
        "camera": ["rgb", "depth"],
        "embodiment": ["joint", "ee"],
    }
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False
    if ARGS.device is not None:
        cfg.sim.device = ARGS.device
    return Task(cfg, mode="eval")


def capture(task: Task) -> dict[str, Any]:
    for _ in range(8):
        task._update_render()
    camera = task._camera_manager.cameras["wrist"]
    tactile: dict[str, dict[str, np.ndarray]] = {}
    for sensor_name, tactile_sensor in task._tactile_manager.tactiles.items():
        output = tactile_sensor.sensor.data.output
        tactile[sensor_name] = {
            field: as_numpy(output[field]) for field in TACTILE_CORE_FIELDS
        }
    return {
        "depth": as_numpy(camera.data.output["depth"]).astype(np.float32),
        "rgb": as_numpy(camera.data.output["rgb"])[..., :3].astype(np.uint8),
        "tactile": tactile,
    }


def save_capture(root: Path, value: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "depth_m.npy", value["depth"])
    Image.fromarray(value["rgb"], mode="RGB").save(root / "rgb.png")
    depth = value["depth"]
    finite = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.percentile(depth[finite], [1.0, 99.0])
        if high <= low:
            high = low + 1e-6
        preview[finite] = (
            np.clip((depth[finite] - low) / (high - low), 0, 1) * 255
        ).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(root / "depth.png")


def set_visibility(task: Task, paths: list[str], invisible: bool) -> None:
    stage = task.scene.stage
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in paths:
            attribute = stage.GetPrimAtPath(path).GetAttribute(
                SECONDARY_RAY_ATTRIBUTE
            )
            if attribute.Set(invisible) is False:
                raise RuntimeError(f"Failed to set secondary-ray visibility: {path}")


def assert_runtime_policy(task: Task) -> tuple[list[str], list[str]]:
    report = task._gelsight_depth_visibility
    case_paths = report["rigid_case_plate_visible"]
    gelpad_paths = report["deformable_gelpads_hidden"]
    if len(case_paths) != 4 or len(gelpad_paths) != 2:
        raise AssertionError(
            f"Unexpected GelSight visibility targets: {report['overrides']}"
        )
    stage = task.scene.stage
    for path in case_paths:
        assert stage.GetPrimAtPath(path).GetAttribute(
            SECONDARY_RAY_ATTRIBUTE
        ).Get() is False
    for path in gelpad_paths:
        assert stage.GetPrimAtPath(path).GetAttribute(
            SECONDARY_RAY_ATTRIBUTE
        ).Get() is True
    return case_paths, gelpad_paths


def compare_tactile(
    hidden: dict[str, dict[str, np.ndarray]],
    visible: dict[str, dict[str, np.ndarray]],
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for sensor_name in sorted(hidden):
        for field in TACTILE_CORE_FIELDS:
            key = f"{sensor_name}/{field}"
            results[key] = bool(
                np.array_equal(
                    hidden[sensor_name][field],
                    visible[sensor_name][field],
                    equal_nan=True,
                )
            )
    return results


def main() -> None:
    ARGS.output_dir.mkdir(parents=True, exist_ok=False)
    task = make_task()
    case_paths: list[str] = []
    try:
        task.reset(seed=ARGS.seed)
        case_paths, gelpad_paths = assert_runtime_policy(task)

        set_visibility(task, case_paths, True)
        hidden = capture(task)
        set_visibility(task, case_paths, False)
        visible = capture(task)

        save_capture(ARGS.output_dir / "rigid_housing_hidden", hidden)
        save_capture(ARGS.output_dir / "rigid_housing_visible", visible)

        delta = hidden["depth"] - visible["depth"]
        changed = np.isfinite(delta) & (np.abs(delta) > 1e-5)
        changed_count = int(changed.sum())
        median_recovered_depth = (
            float(np.median(delta[changed])) if changed.any() else 0.0
        )
        tactile_equal = compare_tactile(hidden["tactile"], visible["tactile"])
        report = {
            "schema_version": "univtac.real_wrist_depth_visibility_smoke.v1",
            "seed": ARGS.seed,
            "rigid_case_plate_paths": case_paths,
            "deformable_gelpad_paths": gelpad_paths,
            "changed_depth_pixels": changed_count,
            "median_recovered_depth_m": median_recovered_depth,
            "tactile_core_equal": tactile_equal,
        }
        (ARGS.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        if changed_count < 5_000:
            raise AssertionError(
                f"Rigid housing recovered only {changed_count} depth pixels"
            )
        if median_recovered_depth <= 0.005:
            raise AssertionError(
                "Rigid housing did not replace background with nearer depth: "
                f"median={median_recovered_depth:.6f} m"
            )
        if not all(tactile_equal.values()):
            raise AssertionError(
                f"Rigid housing visibility changed tactile core data: {tactile_equal}"
            )
        print(json.dumps(report, indent=2), flush=True)
        print("REAL_WRIST_DEPTH_VISIBILITY_SMOKE_OK", flush=True)
    finally:
        if case_paths:
            set_visibility(task, case_paths, False)
        task.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()

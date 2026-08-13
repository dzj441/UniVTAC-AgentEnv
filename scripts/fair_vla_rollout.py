"""Interactive, observation-only rollout bridge for the grasp-classify task.

The process owns the simulator and exposes only the modalities used by the
official UniVTAC ACT policy: head RGB, left/right marker RGB, and 8-D joint
state.  Privileged task state and the success predicate stay inside this
process until the caller explicitly finishes a rollout.

Commands are newline-delimited JSON on stdin.  Machine-readable responses are
printed with the ``FAIR_VLA_RESULT `` prefix and every command/response is
appended to the run transcript.
"""

from __future__ import annotations

import argparse
import cv2
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1000000, 1000001, 1000002)
RESULT_PREFIX = "FAIR_VLA_RESULT "
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair interactive VLA rollout bridge")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    # TacEx/UIPC currently initializes empty attachment points under Isaac
    # Lab's headless-rendering experience, which corrupts the GelSight marker
    # projection. Match the official evaluator's rendering experience even
    # when the process itself has no local display.
    args.livestream = 2
    return args


ARGS = parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

# Imports that depend on a running Isaac application must stay below AppLauncher.
from envs.grasp_classify import Task, TaskCfg  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_safe) + "\n")


def emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_safe), flush=True)


def tensor_to_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB tensor, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def marker_grid_health(array: np.ndarray) -> dict[str, Any]:
    """Count plausible marker blobs and reject broken camera projections."""
    dark_mask = (array.astype(np.float32).mean(axis=2) < 45.0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    plausible = areas[(areas >= 25) & (areas <= 400)]
    return {
        "expected_markers": 63,
        "plausible_marker_components": int(len(plausible)),
        "all_dark_components": int(component_count - 1),
        "dark_pixel_count": int(dark_mask.sum()),
        "healthy": bool(len(plausible) >= 40),
    }


def h264_encoder_args() -> list[str]:
    """Return encoder arguments for a VS Code/browser-compatible H.264 MP4."""
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-profile:v", "main", "-crf", "23"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-profile:v", "main", "-b:v", "2M"]
    raise RuntimeError("No H.264 encoder is available in the active ffmpeg build")


def encode_observation_video(seed_dir: Path) -> dict[str, Any]:
    """Encode the lossless composite frames and verify the resulting codec."""
    frame_pattern = str(seed_dir / "observations" / "seed_*_obs_*" / "composite.png")
    video_path = seed_dir / "agent_observations_h264.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "2",
            "-pattern_type",
            "glob",
            "-i",
            frame_pattern,
            *h264_encoder_args(),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"Unexpected replay video format: {stream}")
    return {
        "path": str(video_path.resolve()),
        "sha256": file_sha256(video_path),
        **stream,
    }


def make_composite(head: np.ndarray, left: np.ndarray, right: np.ndarray, path: Path) -> None:
    head_img = Image.fromarray(head, mode="RGB")
    left_img = Image.fromarray(left, mode="RGB")
    right_img = Image.fromarray(right, mode="RGB")

    target_width = max(head_img.width, left_img.width + right_img.width)
    head_height = round(head_img.height * target_width / head_img.width)
    tactile_width = target_width // 2
    tactile_height = max(
        round(left_img.height * tactile_width / left_img.width),
        round(right_img.height * tactile_width / right_img.width),
    )
    label_height = 22
    canvas = Image.new("RGB", (target_width, head_height + tactile_height + 2 * label_height), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), "head_rgb", fill="white")
    canvas.paste(head_img.resize((target_width, head_height)), (0, label_height))
    tactile_y = label_height + head_height + label_height
    draw.text((6, label_height + head_height + 4), "left_tactile_marker", fill="white")
    draw.text((tactile_width + 6, label_height + head_height + 4), "right_tactile_marker", fill="white")
    canvas.paste(left_img.resize((tactile_width, tactile_height)), (0, tactile_y))
    canvas.paste(right_img.resize((target_width - tactile_width, tactile_height)), (tactile_width, tactile_y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class FairRolloutBridge:
    """Keep privileged simulator state behind a strict observation/action API."""

    MAX_TRANSLATION_COMPONENT = 0.04
    MAX_TRANSLATION_NORM = 0.06
    MAX_ROTATION_COMPONENT = 0.35
    MAX_GRIPPER_DELTA = 0.02
    MAX_WAIT_STEPS = 60

    def __init__(self, task: Task, run_dir: Path, seeds: list[int]):
        self.task = task
        self.run_dir = run_dir.resolve()
        self.seeds = seeds
        self.transcript_path = self.run_dir / "agent_transcript.jsonl"
        self.outcomes_path = self.run_dir / "evaluator_outcomes.jsonl"
        self.seed_index = 0
        self.active = False
        self.finished = False
        self.current_seed: int | None = None
        self.round_index = 0
        self.observation_index = 0
        self.current_observation_id: str | None = None
        self.rollout_start_time = 0.0

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.transcript_path, {"timestamp_utc": utc_now(), "kind": kind, **payload})

    def _capture_observation(self) -> dict[str, Any]:
        raw = self.task._get_observations()
        if raw.get("actor"):
            raise RuntimeError("Fairness violation: privileged actor observations are enabled")

        camera_keys = set(raw["observation"])
        if camera_keys != {"head"}:
            raise RuntimeError(f"Fairness violation: expected only head camera, got {sorted(camera_keys)}")
        tactile_keys = set(raw["tactile"])
        if tactile_keys != {"left_tactile", "right_tactile"}:
            raise RuntimeError(f"Unexpected tactile sensor keys: {sorted(tactile_keys)}")

        head = tensor_to_rgb(raw["observation"]["head"]["rgb"])
        left = tensor_to_rgb(raw["tactile"]["left_tactile"]["rgb_marker"])
        right = tensor_to_rgb(raw["tactile"]["right_tactile"]["rgb_marker"])
        joint = raw["embodiment"]["joint"][:8].detach().cpu().to(torch.float64).tolist()

        obs_id = f"seed_{self.current_seed}_obs_{self.observation_index:03d}"
        obs_dir = self.run_dir / f"seed_{self.current_seed}" / "observations" / obs_id
        paths = {
            "head_rgb": obs_dir / "head_rgb.png",
            "left_tactile_marker": obs_dir / "left_tactile_marker.png",
            "right_tactile_marker": obs_dir / "right_tactile_marker.png",
            "composite": obs_dir / "composite.png",
        }
        save_rgb(paths["head_rgb"], head)
        save_rgb(paths["left_tactile_marker"], left)
        save_rgb(paths["right_tactile_marker"], right)
        make_composite(head, left, right, paths["composite"])

        tactile_health = {
            "left": marker_grid_health(left),
            "right": marker_grid_health(right),
        }
        if not all(sensor["healthy"] for sensor in tactile_health.values()):
            failure = {
                "seed": self.current_seed,
                "round": self.round_index,
                "tactile_health": tactile_health,
                "saved_modalities": {name: str(path.resolve()) for name, path in paths.items()},
            }
            self._record("tactile_health_failure", failure)
            raise RuntimeError(
                "Tactile marker-grid health check failed; refusing to expose a corrupted observation: "
                f"{tactile_health}"
            )

        observation = {
            "observation_id": obs_id,
            "seed": self.current_seed,
            "round": self.round_index,
            "modalities": {
                name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for name, path in paths.items()
            },
            "joint_state_8d": joint,
            "tactile_health": tactile_health,
        }
        self.current_observation_id = obs_id
        self.observation_index += 1
        self._record("observation", observation)
        return observation

    def start(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("A rollout is already active")
        if self.seed_index >= len(self.seeds):
            raise RuntimeError("All configured rollouts are already complete")
        requested_seed = int(command.get("seed", self.seeds[self.seed_index]))
        expected_seed = self.seeds[self.seed_index]
        if requested_seed != expected_seed:
            raise ValueError(f"The next one-shot rollout must use seed {expected_seed}")

        self.current_seed = expected_seed
        self.round_index = 0
        self.observation_index = 0
        self.current_observation_id = None
        self.rollout_start_time = time.perf_counter()
        self.task.mode = "eval"
        self.task.reset(seed=expected_seed, instructions=["Empty"])
        self.task.mean_steps = self.task.cfg.step_lim
        self.active = True
        observation = self._capture_observation()
        payload = {
            "status": "rollout_started",
            "seed": expected_seed,
            "one_shot_index": self.seed_index + 1,
            "one_shot_total": len(self.seeds),
            "observation": observation,
        }
        self._record(
            "rollout_start",
            {
                "seed": expected_seed,
                "agent_note": command.get("agent_note", ""),
                "observation_id": observation["observation_id"],
            },
        )
        return payload

    def _validate_observation_id(self, command: dict[str, Any]) -> None:
        supplied = command.get("observation_id")
        if supplied != self.current_observation_id:
            raise ValueError(
                f"Action must cite latest observation_id {self.current_observation_id!r}; got {supplied!r}"
            )

    def act(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("No rollout is active")
        self._validate_observation_id(command)
        rationale = str(command.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Every action requires a non-empty rationale for the audit trail")

        dp = np.asarray(command.get("delta_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        dr = np.asarray(command.get("delta_rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
        dg = float(command.get("delta_gripper", 0.0))
        if dp.shape != (3,) or dr.shape != (3,):
            raise ValueError("delta_position and delta_rpy must each contain exactly 3 values")
        if not np.all(np.isfinite(dp)) or not np.all(np.isfinite(dr)) or not np.isfinite(dg):
            raise ValueError("Action values must be finite")
        if np.max(np.abs(dp)) > self.MAX_TRANSLATION_COMPONENT or np.linalg.norm(dp) > self.MAX_TRANSLATION_NORM:
            raise ValueError("Translation exceeds the per-round bound")
        if np.max(np.abs(dr)) > self.MAX_ROTATION_COMPONENT:
            raise ValueError("Rotation exceeds the per-round bound")
        if abs(dg) > self.MAX_GRIPPER_DELTA:
            raise ValueError("Gripper delta exceeds the per-round bound")

        prior_observation_id = self.current_observation_id
        # BaseTask's delta-EE path is NumPy-based (Pose.add_bias/add_rotation),
        # while its qpos path is torch-based.  Keep this bounded relative action
        # as a NumPy vector so no privileged state is needed for conversion.
        action_vector = np.concatenate((dp, dr, [dg])).astype(np.float32)
        started = time.perf_counter()
        executed, _hidden_success = self.task.take_action(
            action_vector, action_type="delta_ee"
        )
        duration = time.perf_counter() - started
        self.round_index += 1
        observation = self._capture_observation()
        action_record = {
            "seed": self.current_seed,
            "round": self.round_index,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {
                "primitive": "bounded_delta_ee",
                "delta_position_world_m": dp.tolist(),
                "delta_rpy_world_rad": dr.tolist(),
                "delta_gripper_qpos_m": dg,
            },
            "low_level_execution_succeeded": bool(executed),
            "execution_duration_seconds": duration,
            "next_observation_id": observation["observation_id"],
        }
        self._record("agent_action", action_record)
        return {
            "status": "action_complete",
            **action_record,
            "observation": observation,
            # Deliberately no reward/done/success here.
        }

    def wait(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("No rollout is active")
        self._validate_observation_id(command)
        rationale = str(command.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Every wait requires a non-empty rationale for the audit trail")
        steps = int(command.get("steps", 20))
        if not 1 <= steps <= self.MAX_WAIT_STEPS:
            raise ValueError(f"steps must be in [1, {self.MAX_WAIT_STEPS}]")

        prior_observation_id = self.current_observation_id
        started = time.perf_counter()
        executed = self.task.delay(steps=steps, is_save=False, force=False)
        duration = time.perf_counter() - started
        self.round_index += 1
        observation = self._capture_observation()
        action_record = {
            "seed": self.current_seed,
            "round": self.round_index,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {"primitive": "wait", "physics_steps": steps},
            "low_level_execution_succeeded": bool(executed),
            "execution_duration_seconds": duration,
            "next_observation_id": observation["observation_id"],
        }
        self._record("agent_action", action_record)
        return {"status": "action_complete", **action_record, "observation": observation}

    def finish(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("No rollout is active")
        self._validate_observation_id(command)
        final_note = str(command.get("final_note", "")).strip()
        if not final_note:
            raise ValueError("finish requires a non-empty final_note")

        # This is the first point at which the evaluator predicate crosses the
        # observation boundary.  No further action is permitted for this seed.
        success = bool(self.task.eval_success or self.task.check_success())
        try:
            replay_video: dict[str, Any] = encode_observation_video(
                self.run_dir / f"seed_{self.current_seed}"
            )
        except Exception as exc:
            replay_video = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        outcome = {
            "timestamp_utc": utc_now(),
            "seed": self.current_seed,
            "success": success,
            "agent_rounds": self.round_index,
            "sim_action_count": int(self.task.take_action_cnt),
            "wall_seconds": time.perf_counter() - self.rollout_start_time,
            "final_observation_id": self.current_observation_id,
            "final_note_before_outcome": final_note,
            "replay_video": replay_video,
        }
        append_jsonl(self.outcomes_path, outcome)
        self._record("rollout_finish_and_outcome_reveal", outcome)
        self.active = False
        self.seed_index += 1
        self.current_seed = None
        self.current_observation_id = None
        self.finished = self.seed_index == len(self.seeds)
        return {
            "status": "rollout_finished",
            **outcome,
            "remaining_rollouts": len(self.seeds) - self.seed_index,
        }

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        name = command.get("command")
        if name == "start":
            return self.start(command)
        if name == "act":
            return self.act(command)
        if name == "wait":
            return self.wait(command)
        if name == "finish":
            return self.finish(command)
        if name == "status":
            return {
                "status": "bridge_status",
                "active": self.active,
                "completed_rollouts": self.seed_index,
                "total_rollouts": len(self.seeds),
                "expected_next_seed": None if self.seed_index >= len(self.seeds) else self.seeds[self.seed_index],
                "latest_observation_id": self.current_observation_id,
                "run_dir": str(self.run_dir),
            }
        if name == "close":
            if self.active:
                raise RuntimeError("Cannot close while a rollout is active; finish it first")
            if not self.finished:
                raise RuntimeError("Cannot close before all configured one-shot rollouts finish")
            return {"status": "closing", "run_dir": str(self.run_dir)}
        raise ValueError(f"Unknown command: {name!r}")


def write_manifest(run_dir: Path, seeds: list[int]) -> None:
    manifest = {
        "created_utc": utc_now(),
        "task": "grasp_classify",
        "seeds": seeds,
        "rollout_rule": "exactly one attempt per seed",
        "task_semantics_given_to_agent": "rough prism -> orange pad; plain prism -> green pad",
        "allowed_observations": [
            "head RGB",
            "left tactile marker RGB",
            "right tactile marker RGB",
            "first 8 robot joint positions",
        ],
        "forbidden_observations": [
            "actor/object/pad poses",
            "selected prism class",
            "target identity or target pose",
            "wrist camera",
            "tactile depth/pose/raw marker coordinates",
            "contacts, reward, done, and success before finish",
            "official ACT actions or checkpoint inference",
        ],
        "allowed_actions": {
            "bounded_delta_ee": {
                "translation_frame": "world",
                "max_abs_component_m": FairRolloutBridge.MAX_TRANSLATION_COMPONENT,
                "max_norm_m": FairRolloutBridge.MAX_TRANSLATION_NORM,
                "max_abs_rpy_component_rad": FairRolloutBridge.MAX_ROTATION_COMPONENT,
                "max_abs_gripper_delta_m": FairRolloutBridge.MAX_GRIPPER_DELTA,
            },
            "wait": {"max_physics_steps": FairRolloutBridge.MAX_WAIT_STEPS},
        },
        "outcome_isolation": "success is withheld until finish; finish permanently closes that seed",
        "rendering_experience": "official evaluator-compatible rendering.kit via livestream=2",
        "tactile_health_gate": "each rgb_marker image must retain at least 40 of 63 plausible marker components",
        "replay_video": "H.264 Main profile, yuv420p, fast-start MP4 generated from composite observations",
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "fairness_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not ARGS.seeds or len(set(ARGS.seeds)) != len(ARGS.seeds):
        raise ValueError("At least one seed is required and all rollout seeds must be distinct")
    run_dir = ARGS.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "agent_runs" / f"grasp_classify_fair_{stamp}"
    elif not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    write_manifest(run_dir, list(ARGS.seeds))

    cfg = TaskCfg()
    cfg.save_dir = run_dir / "simulator_internal"
    cfg.scene.num_envs = 1
    if ARGS.device is not None:
        cfg.sim.device = ARGS.device
    cfg.decimation = 1
    cfg.obs_data_type = {
        "camera": ["rgb"],
        "tactile": ["rgb_marker"],
        "embodiment": ["joint"],
    }
    cfg.cameras = [camera for camera in cfg.cameras if camera.name == "head"]
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False

    task = Task(cfg, mode="eval")
    bridge = FairRolloutBridge(task, run_dir, list(ARGS.seeds))
    append_jsonl(
        bridge.transcript_path,
        {
            "timestamp_utc": utc_now(),
            "kind": "bridge_ready",
            "run_dir": str(run_dir.resolve()),
            "seeds": list(ARGS.seeds),
        },
    )
    emit(
        {
            "status": "ready",
            "run_dir": str(run_dir.resolve()),
            "expected_next_seed": ARGS.seeds[0],
            "protocol": "newline-delimited JSON; commands: start, act, wait, finish, status, close",
        }
    )

    close_requested = False
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise TypeError("Command must be a JSON object")
                append_jsonl(
                    bridge.transcript_path,
                    {"timestamp_utc": utc_now(), "kind": "raw_command", "command": command},
                )
                response = bridge.handle(command)
                emit(response)
                if command.get("command") == "close":
                    close_requested = True
                    break
            except Exception as exc:
                error = {
                    "status": "command_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                append_jsonl(
                    bridge.transcript_path,
                    {
                        "timestamp_utc": utc_now(),
                        "kind": "command_error",
                        **error,
                        "traceback": traceback.format_exc(),
                    },
                )
                emit(error)
    except Exception:
        append_jsonl(
            bridge.transcript_path,
            {"timestamp_utc": utc_now(), "kind": "fatal_error", "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if not close_requested:
            append_jsonl(
                bridge.transcript_path,
                {"timestamp_utc": utc_now(), "kind": "bridge_stopped_without_close"},
            )
        task.close()
        SIMULATION_APP.close()


if __name__ == "__main__":
    main()

"""Auditable wrist-camera agentic rollout for ``grasp_classify``.

This bridge separates semantic classification from geometric correction:

* A fresh seed is generated inside the process and hidden until termination.
  Before reset, the bridge publishes a salted SHA-256 commitment to that seed.
* The agent receives head RGB, wrist RGB, left/right marker RGB, and the first
  eight robot joint positions. Privileged task state is never serialized.
* Before seeing any task-success bit, the agent must irreversibly commit to the
  green or orange pad. It may first perform at most two gripper-only probes.
* After commitment, task success is returned after every bounded action, but a
  cumulative-y guard prevents crossing from the committed pad's half-space to
  the other pad. Thus failure can refine geometry but cannot classify texture.
* Success terminates the episode immediately. Failure terminates at the fixed
  action budget. Only then are seed, salt, and ground-truth class revealed.

Commands and responses are newline-delimited JSON. Responses begin with the
``WRIST_REFEREE_RESULT `` prefix, and all public interaction is also recorded
in ``agent_transcript.jsonl`` under the run directory.
"""

from __future__ import annotations

import argparse
import contextlib
import cv2
import hashlib
import io
import json
import secrets
import shutil
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
RESULT_PREFIX = "WRIST_REFEREE_RESULT "
COMMITMENT_DOMAIN = "univtac-wrist-referee-v1"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrist-camera success-feedback referee bridge")
    parser.add_argument("--run-dir", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    # TacEx/UIPC marker rendering is healthy under the same experience used by
    # the official evaluator, even when there is no local display.
    args.livestream = 2
    return args


ARGS = parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

# Imports requiring an initialized Isaac application stay below AppLauncher.
from envs.grasp_classify import Task, TaskCfg  # noqa: E402


class RefereeTask(Task):
    """Use opaque save paths so BaseTask never places a secret seed in a path."""

    def _setup_save(self) -> None:
        self.save_root = Path(self.cfg.save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tmp_save_dir = self.save_root / ".cache" / "private_episode"
        self.save_path = self.save_root / "hdf5" / "private_episode.hdf5"
        self.save_video_path = self.save_root / "video" / "private_episode.mp4"
        self.metadata_path = self.save_root / "metadata.json"
        self.cfg.uipc_sim.workspace = str(self.save_root / "scene")


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
    """Reject the previously observed broken, non-grid marker projection."""
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


def make_composite(
    head: np.ndarray,
    wrist: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    path: Path,
) -> None:
    panel_width, panel_height, label_height = 480, 270, 24
    panels = [
        ("head_rgb", head),
        ("wrist_rgb", wrist),
        ("left_tactile_marker", left),
        ("right_tactile_marker", right),
    ]
    canvas = Image.new(
        "RGB",
        (panel_width * 2, (panel_height + label_height) * 2),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        column, row = index % 2, index // 2
        x = column * panel_width
        y = row * (panel_height + label_height)
        draw.text((x + 6, y + 5), label, fill="white")
        image = Image.fromarray(array, mode="RGB").resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def media_executable(name: str) -> str:
    """Find media tools even when Isaac launches Python without conda's bin on PATH."""
    on_path = shutil.which(name)
    if on_path is not None:
        return on_path
    beside_python = Path(sys.executable).resolve().parent / name
    if beside_python.is_file():
        return str(beside_python)
    raise FileNotFoundError(f"Could not find {name!r} on PATH or beside {sys.executable}")


def h264_encoder_args(ffmpeg: str) -> list[str]:
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-profile:v", "main", "-crf", "23"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-profile:v", "main", "-b:v", "2M"]
    raise RuntimeError("No H.264 encoder is available in the active ffmpeg build")


def encode_observation_video(episode_dir: Path) -> dict[str, Any]:
    frame_pattern = str(episode_dir / "observations" / "obs_*" / "composite.png")
    video_path = episode_dir / "agent_observations_h264.mp4"
    ffmpeg = media_executable("ffmpeg")
    ffprobe = media_executable("ffprobe")
    subprocess.run(
        [
            ffmpeg,
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
            *h264_encoder_args(ffmpeg),
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
            ffprobe,
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


class WristFeedbackReferee:
    """Expose a strict public API while retaining seed and task state privately."""

    MAX_TRANSLATION_COMPONENT = 0.04
    MAX_TRANSLATION_NORM = 0.06
    MAX_ROTATION_COMPONENT = 0.35
    MAX_PROBE_GRIPPER_DELTA = 0.002
    MAX_ACTION_GRIPPER_DELTA = 0.005
    MAX_PROBES = 2
    MAX_POST_COMMIT_ACTIONS = 10
    MAX_WAIT_STEPS = 60
    MIN_LOCKED_TARGETWARD_Y = 0.02
    TARGET_SIGN = {"green": 1.0, "orange": -1.0}

    def __init__(self, task: RefereeTask, run_dir: Path):
        self.task = task
        self.run_dir = run_dir.resolve()
        self.episode_dir = self.run_dir / "episode"
        self.transcript_path = self.run_dir / "agent_transcript.jsonl"
        self.outcomes_path = self.run_dir / "evaluator_outcome.json"

        # Keep well away from every seed used in the earlier visible trials.
        self._secret_seed = 10_000_000 + secrets.randbelow((2**31 - 1) - 10_000_000)
        self._secret_salt = secrets.token_hex(32)
        self.seed_commitment = hashlib.sha256(
            f"{COMMITMENT_DOMAIN}|{self._secret_seed}|{self._secret_salt}".encode("utf-8")
        ).hexdigest()

        self.started = False
        self.active = False
        self.terminal = False
        self.observation_index = 0
        self.current_observation_id: str | None = None
        self.probe_count = 0
        self.action_count = 0
        self.committed_target: str | None = None
        self.cumulative_y = 0.0
        self.target_halfspace_locked = False
        self.rollout_start_time = 0.0

    def write_manifest(self) -> None:
        manifest = {
            "created_utc": utc_now(),
            "protocol_version": COMMITMENT_DOMAIN,
            "task": "grasp_classify",
            "episodes": 1,
            "seed": "hidden until terminal outcome",
            "seed_commitment_sha256": self.seed_commitment,
            "commitment_preimage_format": f"{COMMITMENT_DOMAIN}|<decimal_seed>|<hex_salt>",
            "task_semantics_given_to_agent": "rough prism -> orange pad; plain prism -> green pad",
            "allowed_observations": [
                "head RGB",
                "wrist RGB",
                "left tactile marker RGB",
                "right tactile marker RGB",
                "first 8 robot joint positions",
            ],
            "forbidden_observations": [
                "actor/object/pad poses",
                "secret seed and selected prism class before termination",
                "target identity or target pose",
                "tactile depth/pose/raw marker coordinates",
                "contacts and reward",
                "task success before irreversible target commitment",
                "official checkpoint actions or inference",
            ],
            "classification_gate": {
                "required_before_commit": "at least one gripper-only tactile probe",
                "max_gripper_only_probes": self.MAX_PROBES,
                "commitment": "exactly one irreversible choice of green or orange",
                "success_before_commit": "withheld",
            },
            "geometric_feedback": {
                "success_after_commit": "returned after each action",
                "success_true": "immediate automatic termination",
                "max_actions": self.MAX_POST_COMMIT_ACTIONS,
                "target_switching": "forbidden",
                "cumulative_y_rule": (
                    "green requires cumulative y >= 0; orange requires cumulative y <= 0; "
                    f"after reaching {self.MIN_LOCKED_TARGETWARD_Y:.3f} m targetward, "
                    "the targetward displacement may never fall below that threshold"
                ),
            },
            "action_bounds": {
                "translation_frame": "world",
                "max_abs_translation_component_m": self.MAX_TRANSLATION_COMPONENT,
                "max_translation_norm_m": self.MAX_TRANSLATION_NORM,
                "max_abs_rpy_component_rad": self.MAX_ROTATION_COMPONENT,
                "max_abs_action_gripper_delta_m": self.MAX_ACTION_GRIPPER_DELTA,
                "max_wait_physics_steps": self.MAX_WAIT_STEPS,
            },
            "rendering_experience": "official evaluator-compatible rendering.kit via livestream=2",
            "tactile_health_gate": "each rgb_marker image must retain at least 40 plausible marker components",
            "replay_video": "H.264 Main profile, yuv420p, fast-start MP4 from 2x2 composites",
            "threat_model_note": (
                "The bridge makes cheating visible and structurally blocks success-as-classification. "
                "Because the operator owns the OS process, the audit also requires that the operator "
                "not inspect process memory or private simulator attributes during the episode."
            ),
        }
        (self.run_dir / "fairness_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.transcript_path, {"timestamp_utc": utc_now(), "kind": kind, **payload})

    def _stage(self) -> str:
        if self.terminal:
            return "terminal"
        if self.committed_target is None:
            return "classification"
        return "geometric_correction"

    def _capture_observation(self) -> dict[str, Any]:
        raw = self.task._get_observations()
        if raw.get("actor"):
            raise RuntimeError("Fairness violation: privileged actor observations are enabled")

        camera_keys = set(raw["observation"])
        if camera_keys != {"head", "wrist"}:
            raise RuntimeError(f"Expected head+wrist cameras, got {sorted(camera_keys)}")
        tactile_keys = set(raw["tactile"])
        if tactile_keys != {"left_tactile", "right_tactile"}:
            raise RuntimeError(f"Unexpected tactile sensor keys: {sorted(tactile_keys)}")

        head = tensor_to_rgb(raw["observation"]["head"]["rgb"])
        wrist = tensor_to_rgb(raw["observation"]["wrist"]["rgb"])
        left = tensor_to_rgb(raw["tactile"]["left_tactile"]["rgb_marker"])
        right = tensor_to_rgb(raw["tactile"]["right_tactile"]["rgb_marker"])
        joint = raw["embodiment"]["joint"][:8].detach().cpu().to(torch.float64).tolist()

        obs_id = f"obs_{self.observation_index:03d}"
        obs_dir = self.episode_dir / "observations" / obs_id
        paths = {
            "head_rgb": obs_dir / "head_rgb.png",
            "wrist_rgb": obs_dir / "wrist_rgb.png",
            "left_tactile_marker": obs_dir / "left_tactile_marker.png",
            "right_tactile_marker": obs_dir / "right_tactile_marker.png",
            "composite": obs_dir / "composite.png",
        }
        save_rgb(paths["head_rgb"], head)
        save_rgb(paths["wrist_rgb"], wrist)
        save_rgb(paths["left_tactile_marker"], left)
        save_rgb(paths["right_tactile_marker"], right)
        make_composite(head, wrist, left, right, paths["composite"])

        tactile_health = {
            "left": marker_grid_health(left),
            "right": marker_grid_health(right),
        }
        if not all(sensor["healthy"] for sensor in tactile_health.values()):
            failure = {
                "observation_id": obs_id,
                "tactile_health": tactile_health,
                "saved_modalities": {name: str(path.resolve()) for name, path in paths.items()},
            }
            self._record("tactile_health_failure", failure)
            raise RuntimeError(f"Tactile marker-grid health check failed: {tactile_health}")

        observation = {
            "observation_id": obs_id,
            "stage": self._stage(),
            "probe_count": self.probe_count,
            "post_commit_action_count": self.action_count,
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

    def _validate_latest_observation(self, command: dict[str, Any]) -> None:
        supplied = command.get("observation_id")
        if supplied != self.current_observation_id:
            raise ValueError(
                f"Command must cite latest observation_id {self.current_observation_id!r}; got {supplied!r}"
            )

    @staticmethod
    def _require_rationale(command: dict[str, Any]) -> str:
        rationale = str(command.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Every decision requires a non-empty rationale for the audit trail")
        return rationale

    @staticmethod
    def _parse_delta(command: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        dp = np.asarray(command.get("delta_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        dr = np.asarray(command.get("delta_rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
        dg = float(command.get("delta_gripper", 0.0))
        if dp.shape != (3,) or dr.shape != (3,):
            raise ValueError("delta_position and delta_rpy must each contain exactly 3 values")
        if not np.all(np.isfinite(dp)) or not np.all(np.isfinite(dr)) or not np.isfinite(dg):
            raise ValueError("Action values must be finite")
        return dp, dr, dg

    def start(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("The one-shot episode has already been started")
        if "seed" in command:
            raise ValueError("The referee does not accept a caller-selected seed")

        self.rollout_start_time = time.perf_counter()
        self.task.mode = "eval"
        # Python-level reset output includes implementation details on some
        # Isaac versions. Suppress it rather than risk leaking the private seed.
        reset_stdout, reset_stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(reset_stdout), contextlib.redirect_stderr(reset_stderr):
            self.task.reset(seed=self._secret_seed, instructions=["Empty"])
        self.task.mean_steps = self.task.cfg.step_lim
        self.started = True
        self.active = True
        observation = self._capture_observation()
        self._record(
            "rollout_start",
            {
                "seed_commitment_sha256": self.seed_commitment,
                "agent_note": str(command.get("agent_note", "")),
                "observation_id": observation["observation_id"],
                "suppressed_reset_stdout_chars": len(reset_stdout.getvalue()),
                "suppressed_reset_stderr_chars": len(reset_stderr.getvalue()),
            },
        )
        return {
            "status": "rollout_started",
            "seed_commitment_sha256": self.seed_commitment,
            "stage": self._stage(),
            "required_next_decision": "one or two gripper-only probes, then commit_target",
            "success_feedback_available": False,
            "observation": observation,
        }

    def probe(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active or self.committed_target is not None:
            raise RuntimeError("probe is allowed only during the active classification stage")
        self._validate_latest_observation(command)
        rationale = self._require_rationale(command)
        if self.probe_count >= self.MAX_PROBES:
            raise RuntimeError("The gripper-only probe budget is exhausted; commit_target is required")

        dp, dr, dg = self._parse_delta(command)
        if not np.allclose(dp, 0.0) or not np.allclose(dr, 0.0):
            raise ValueError("A pre-commit probe must have zero translation and zero rotation")
        if dg == 0.0 or abs(dg) > self.MAX_PROBE_GRIPPER_DELTA:
            raise ValueError(
                f"Probe delta_gripper must be nonzero and at most {self.MAX_PROBE_GRIPPER_DELTA} m"
            )

        prior_observation_id = self.current_observation_id
        action_vector = np.concatenate((dp, dr, [dg])).astype(np.float32)
        started = time.perf_counter()
        executed, _withheld_success = self.task.take_action(action_vector, action_type="delta_ee")
        duration = time.perf_counter() - started
        self.probe_count += 1
        observation = self._capture_observation()
        record = {
            "probe_index": self.probe_count,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {"primitive": "gripper_only_probe", "delta_gripper_qpos_m": dg},
            "low_level_execution_succeeded": bool(executed),
            "execution_duration_seconds": duration,
            "next_observation_id": observation["observation_id"],
            "task_success": "withheld before target commitment",
        }
        self._record("classification_probe", record)
        return {
            "status": "probe_complete",
            **record,
            "remaining_probes": self.MAX_PROBES - self.probe_count,
            "success_feedback_available": False,
            "observation": observation,
        }

    def commit_target(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active or self.committed_target is not None:
            raise RuntimeError("A target can be committed exactly once during classification")
        self._validate_latest_observation(command)
        rationale = self._require_rationale(command)
        if self.probe_count < 1:
            raise RuntimeError("At least one tactile probe is required before target commitment")
        target = str(command.get("target", "")).strip().lower()
        if target not in self.TARGET_SIGN:
            raise ValueError("target must be exactly 'green' or 'orange'")

        self.committed_target = target
        record = {
            "observation_id": self.current_observation_id,
            "committed_target": target,
            "rationale": rationale,
            "irreversible": True,
            "success_seen_before_commit": False,
        }
        self._record("irreversible_target_commitment", record)
        return {
            "status": "target_committed",
            **record,
            "stage": self._stage(),
            "success_feedback_available": True,
            "remaining_post_commit_actions": self.MAX_POST_COMMIT_ACTIONS,
            "targetward_y_sign": int(self.TARGET_SIGN[target]),
        }

    def _validate_post_commit_delta(
        self, dp: np.ndarray, dr: np.ndarray, dg: float
    ) -> tuple[float, bool]:
        if np.max(np.abs(dp)) > self.MAX_TRANSLATION_COMPONENT or np.linalg.norm(dp) > self.MAX_TRANSLATION_NORM:
            raise ValueError("Translation exceeds the per-action bound")
        if np.max(np.abs(dr)) > self.MAX_ROTATION_COMPONENT:
            raise ValueError("Rotation exceeds the per-action bound")
        if abs(dg) > self.MAX_ACTION_GRIPPER_DELTA:
            raise ValueError("Gripper delta exceeds the per-action bound")

        sign = self.TARGET_SIGN[self.committed_target]
        proposed_y = self.cumulative_y + float(dp[1])
        targetward = sign * proposed_y
        if targetward < -1e-9:
            raise ValueError(
                "Anti-cheat guard: cumulative y would enter the opposite target's half-space"
            )
        if self.target_halfspace_locked and targetward < self.MIN_LOCKED_TARGETWARD_Y - 1e-9:
            raise ValueError(
                "Anti-cheat guard: cumulative y would leave the irreversibly locked target region"
            )
        becomes_locked = self.target_halfspace_locked or targetward >= self.MIN_LOCKED_TARGETWARD_Y
        return proposed_y, becomes_locked

    def act(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active or self.committed_target is None:
            raise RuntimeError("act requires an active episode and irreversible target commitment")
        self._validate_latest_observation(command)
        rationale = self._require_rationale(command)
        if self.action_count >= self.MAX_POST_COMMIT_ACTIONS:
            raise RuntimeError("The post-commit action budget is exhausted")

        dp, dr, dg = self._parse_delta(command)
        proposed_y, becomes_locked = self._validate_post_commit_delta(dp, dr, dg)
        prior_observation_id = self.current_observation_id
        action_vector = np.concatenate((dp, dr, [dg])).astype(np.float32)
        started = time.perf_counter()
        executed, task_success = self.task.take_action(action_vector, action_type="delta_ee")
        duration = time.perf_counter() - started
        self.action_count += 1
        self.cumulative_y = proposed_y
        self.target_halfspace_locked = becomes_locked
        observation = self._capture_observation()
        success = bool(task_success)
        record = {
            "post_commit_action_index": self.action_count,
            "prior_observation_id": prior_observation_id,
            "committed_target": self.committed_target,
            "rationale": rationale,
            "action": {
                "primitive": "bounded_delta_ee",
                "delta_position_world_m": dp.tolist(),
                "delta_rpy_world_rad": dr.tolist(),
                "delta_gripper_qpos_m": dg,
            },
            "cumulative_y_m": self.cumulative_y,
            "target_halfspace_locked": self.target_halfspace_locked,
            "low_level_execution_succeeded": bool(executed),
            "execution_duration_seconds": duration,
            "next_observation_id": observation["observation_id"],
            "task_success": success,
        }
        self._record("post_commit_action_and_feedback", record)

        if success:
            return self._finalize(
                reason="success_auto_stop",
                final_note="Task success became true; referee stopped the episode immediately.",
                success=True,
                final_observation=observation,
                last_action=record,
            )
        if self.action_count >= self.MAX_POST_COMMIT_ACTIONS:
            return self._finalize(
                reason="action_budget_exhausted",
                final_note="Fixed post-commit action budget exhausted.",
                success=False,
                final_observation=observation,
                last_action=record,
            )
        return {
            "status": "action_complete",
            **record,
            "remaining_post_commit_actions": self.MAX_POST_COMMIT_ACTIONS - self.action_count,
            "observation": observation,
        }

    def wait(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active or self.committed_target is None:
            raise RuntimeError("wait requires an active episode and irreversible target commitment")
        self._validate_latest_observation(command)
        rationale = self._require_rationale(command)
        if self.action_count >= self.MAX_POST_COMMIT_ACTIONS:
            raise RuntimeError("The post-commit action budget is exhausted")
        steps = int(command.get("steps", 20))
        if not 1 <= steps <= self.MAX_WAIT_STEPS:
            raise ValueError(f"steps must be in [1, {self.MAX_WAIT_STEPS}]")

        prior_observation_id = self.current_observation_id
        started = time.perf_counter()
        executed = self.task.delay(steps=steps, is_save=False, force=False)
        duration = time.perf_counter() - started
        self.action_count += 1
        success = bool(self.task.eval_success or self.task.check_success())
        if success:
            self.task.eval_success = True
        observation = self._capture_observation()
        record = {
            "post_commit_action_index": self.action_count,
            "prior_observation_id": prior_observation_id,
            "committed_target": self.committed_target,
            "rationale": rationale,
            "action": {"primitive": "wait", "physics_steps": steps},
            "cumulative_y_m": self.cumulative_y,
            "target_halfspace_locked": self.target_halfspace_locked,
            "low_level_execution_succeeded": bool(executed),
            "execution_duration_seconds": duration,
            "next_observation_id": observation["observation_id"],
            "task_success": success,
        }
        self._record("post_commit_action_and_feedback", record)

        if success:
            return self._finalize(
                reason="success_auto_stop",
                final_note="Task success became true after settling; referee stopped immediately.",
                success=True,
                final_observation=observation,
                last_action=record,
            )
        if self.action_count >= self.MAX_POST_COMMIT_ACTIONS:
            return self._finalize(
                reason="action_budget_exhausted",
                final_note="Fixed post-commit action budget exhausted.",
                success=False,
                final_observation=observation,
                last_action=record,
            )
        return {
            "status": "action_complete",
            **record,
            "remaining_post_commit_actions": self.MAX_POST_COMMIT_ACTIONS - self.action_count,
            "observation": observation,
        }

    def finish(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self.active or self.committed_target is None:
            raise RuntimeError("finish requires an active post-commit episode")
        self._validate_latest_observation(command)
        final_note = str(command.get("final_note", "")).strip()
        if not final_note:
            raise ValueError("finish requires a non-empty final_note")
        success = bool(self.task.eval_success or self.task.check_success())
        return self._finalize(
            reason="agent_finish",
            final_note=final_note,
            success=success,
            final_observation=None,
            last_action=None,
        )

    def _finalize(
        self,
        *,
        reason: str,
        final_note: str,
        success: bool,
        final_observation: dict[str, Any] | None,
        last_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Close the action boundary before accessing/revealing privileged state.
        self.active = False
        self.terminal = True
        true_class = str(self.task.choice)
        expected_target = "orange" if true_class == "rough" else "green"
        classification_correct = self.committed_target == expected_target
        verifier = hashlib.sha256(
            f"{COMMITMENT_DOMAIN}|{self._secret_seed}|{self._secret_salt}".encode("utf-8")
        ).hexdigest()
        if verifier != self.seed_commitment:
            raise RuntimeError("Internal seed commitment verification failed")

        try:
            replay_video: dict[str, Any] = encode_observation_video(self.episode_dir)
        except Exception as exc:
            replay_video = {"error_type": type(exc).__name__, "message": str(exc)}
        outcome = {
            "timestamp_utc": utc_now(),
            "terminal_reason": reason,
            "success": bool(success),
            "committed_target": self.committed_target,
            "true_class": true_class,
            "expected_target": expected_target,
            "classification_correct": classification_correct,
            "probe_count": self.probe_count,
            "post_commit_action_count": self.action_count,
            "sim_action_count": int(self.task.take_action_cnt),
            "wall_seconds": time.perf_counter() - self.rollout_start_time,
            "final_observation_id": self.current_observation_id,
            "final_note": final_note,
            "seed_reveal": self._secret_seed,
            "salt_reveal": self._secret_salt,
            "seed_commitment_sha256": self.seed_commitment,
            "commitment_verified": True,
            "replay_video": replay_video,
        }
        self.outcomes_path.write_text(
            json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._record("terminal_outcome_and_private_reveal", outcome)
        response = {
            "status": "rollout_finished",
            **outcome,
            "observation": final_observation,
            "last_action": last_action,
        }
        return response

    def status(self) -> dict[str, Any]:
        response = {
            "status": "bridge_status",
            "started": self.started,
            "active": self.active,
            "terminal": self.terminal,
            "stage": self._stage(),
            "seed_commitment_sha256": self.seed_commitment,
            "probe_count": self.probe_count,
            "post_commit_action_count": self.action_count,
            "committed_target": self.committed_target,
            "latest_observation_id": self.current_observation_id,
            "run_dir": str(self.run_dir),
        }
        if self.terminal:
            response["seed_reveal"] = self._secret_seed
            response["salt_reveal"] = self._secret_salt
        return response

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        name = command.get("command")
        if name == "start":
            return self.start(command)
        if name == "probe":
            return self.probe(command)
        if name == "commit_target":
            return self.commit_target(command)
        if name == "act":
            return self.act(command)
        if name == "wait":
            return self.wait(command)
        if name == "finish":
            return self.finish(command)
        if name == "status":
            return self.status()
        if name == "close":
            if not self.terminal:
                raise RuntimeError("close is allowed only after the one-shot episode terminates")
            return {"status": "closing", "run_dir": str(self.run_dir)}
        raise ValueError(f"Unknown command: {name!r}")


def main() -> None:
    run_dir = ARGS.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "agent_runs" / f"grasp_classify_wrist_referee_{stamp}"
    elif not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)

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
    if {camera.name for camera in cfg.cameras} != {"head", "wrist"}:
        raise RuntimeError("TaskCfg must provide exactly head and wrist cameras")
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False

    task = RefereeTask(cfg, mode="eval")
    bridge = WristFeedbackReferee(task, run_dir)
    bridge.write_manifest()
    append_jsonl(
        bridge.transcript_path,
        {
            "timestamp_utc": utc_now(),
            "kind": "bridge_ready",
            "run_dir": str(run_dir.resolve()),
            "seed_commitment_sha256": bridge.seed_commitment,
            "seed": "hidden",
        },
    )
    emit(
        {
            "status": "ready",
            "run_dir": str(run_dir.resolve()),
            "seed_commitment_sha256": bridge.seed_commitment,
            "protocol": (
                "newline JSON commands: start; probe; commit_target; act/wait; "
                "finish; status; close"
            ),
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

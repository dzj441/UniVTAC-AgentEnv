#!/usr/bin/env python3
"""Generic, auditable AgentEnv v1 for UniVTAC manipulation tasks.

The simulator process owns all privileged task state.  Its public lifecycle is
start -> bounded EEF steps -> finish; task success is materialized only by the
terminal finish response.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import secrets
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from agent_env.artifacts import (
    encode_observation_video,
    marker_grid_health,
    require_initial_tactile_health,
    save_composite,
    tensor_to_rgb,
)
from agent_env.benchmark_checker import PostGraspReferenceTracker
from agent_env.benchmark_contract import (
    public_benchmark_command_schema,
    validate_benchmark_command_fields,
)
from agent_env.benchmark_observations import (
    camera_plane,
    image_artifact,
    instance_role_mask,
    normalize_instance_mapping,
    robot_base_camera_extrinsic,
    save_annotation_artifacts,
    save_depth_artifacts,
    to_numpy,
)
from agent_env.benchmark_profiles import (
    AnnotationCapabilities,
    ObservationProfile,
    get_observation_profile,
)
from agent_env.benchmark_protocol import BenchmarkEpisodeProtocol
from agent_env.benchmark_tasks import (
    BenchmarkTaskSpec,
    benchmark_task_parameters,
    get_benchmark_task,
    list_benchmark_tasks,
)
from agent_env.contract import consume_private_evaluator_seed
from agent_env.expert_tasks import BASE_TASK_SUCCESS
from agent_env.nvidia_runtime import audit_current_process
from isaaclab.app import AppLauncher


RESULT_PREFIX = "AGENT_ENV_RESULT "
COMMITMENT_DOMAIN = "univtac.embodied_agent_env.v1"
RESET_TIME_LIMIT_SECONDS = 240.0
FINISH_SETTLE_STEPS = 60
BOTTLE_RELEASE_GRIPPER_QPOS_M = 0.0175
BOTTLE_STABLE_TRANSLATION_M = 0.01
BOTTLE_STABLE_ROTATION_RAD = np.deg2rad(10.0)
INSTANCE_DATA_TYPE = "instance_id_segmentation_fast"


PRIVATE_EVALUATOR_SEED = consume_private_evaluator_seed(os.environ)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVTAC generic embodied AgentEnv v1")
    parser.add_argument(
        "--task",
        required=True,
        choices=tuple(task.name for task in list_benchmark_tasks()),
    )
    parser.add_argument("--profile", required=True, choices=tuple(str(i) for i in range(1, 7)))
    parser.add_argument("--provide-bbox", action="store_true")
    parser.add_argument("--provide-mask", action="store_true")
    parser.add_argument(
        "--key-initial-relative-yaw-rad",
        type=float,
        default=None,
        help=(
            "Optional fixed pull_out_key yaw relative to its slot in radians; "
            "the default restores the legacy random range [-pi/2, -pi/4]."
        ),
    )
    parser.add_argument(
        "--pre-move",
        action="store_true",
        help=(
            "Run the task's legacy privileged pre_move before the first observation. "
            "The v1 default is an ungrasped object and the robot's fixed home state."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    # TacEx marker rendering uses the official rendering experience selected by
    # AppLauncher when livestream is enabled, even though no WebRTC UI is needed.
    args.livestream = 2
    return args


ARGS = parse_args()
TASK_SPEC = get_benchmark_task(ARGS.task)
TASK_PARAMETERS = benchmark_task_parameters(
    TASK_SPEC.name,
    key_initial_relative_yaw_rad=ARGS.key_initial_relative_yaw_rad,
)
PROFILE = get_observation_profile(ARGS.profile)
ANNOTATIONS = AnnotationCapabilities(
    provide_bbox=bool(ARGS.provide_bbox),
    provide_mask=bool(ARGS.provide_mask),
)
PRE_MOVE_ENABLED = bool(ARGS.pre_move)
START_CONDITION = TASK_SPEC.start_condition(PRE_MOVE_ENABLED)
TASK_INSTRUCTION = TASK_SPEC.instruction_for(pre_move=PRE_MOVE_ENABLED)
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


# Task modules import Isaac APIs and must be loaded only after SimulationApp.
TASK_MODULE = importlib.import_module(TASK_SPEC.module)
TASK_CLASS = TASK_MODULE.Task
TASK_CFG_CLASS = TASK_MODULE.TaskCfg


class AgentEnvTask(TASK_CLASS):
    """Hide the evaluator seed from BaseTask-generated filenames."""

    def _setup_save(self) -> None:
        self.save_root = Path(self.cfg.save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tmp_save_dir = self.save_root / ".cache" / "private_episode"
        self.save_path = self.save_root / "hdf5" / "private_episode.hdf5"
        self.save_video_path = self.save_root / "video" / "private_episode.mp4"
        self.metadata_path = self.save_root / "metadata.json"
        self.cfg.uipc_sim.workspace = str(self.save_root / "scene")

    def pre_move(self) -> None:
        """Select legacy pre-grasping or the default fixed ungrasped reset.

        The upstream tasks historically initialized checker reference poses as
        a side effect of their privileged pre_move.  The ungrasped path creates
        only those private references; it performs no robot or actor motion.
        """

        if PRE_MOVE_ENABLED:
            super().pre_move()
            return
        self.initialize_task_references()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_safe) + "\n")


def write_json(path: Path, payload: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe) + "\n",
        encoding="utf-8",
    )
    if private:
        os.chmod(path, 0o600)


def pose_snapshot(actor: Any) -> dict[str, list[float]]:
    pose = actor.get_pose()
    return {
        "position_m": np.asarray(pose.p, dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).tolist(),
    }


def pose_drift(before: dict[str, list[float]], after: dict[str, list[float]]) -> tuple[float, float]:
    before_p = np.asarray(before["position_m"], dtype=np.float64)
    after_p = np.asarray(after["position_m"], dtype=np.float64)
    translation = float(np.linalg.norm(after_p - before_p))

    from agent_env.benchmark_observations import quaternion_wxyz_matrix

    before_r = quaternion_wxyz_matrix(before["quaternion_wxyz"])
    after_r = quaternion_wxyz_matrix(after["quaternion_wxyz"])
    relative = before_r.T @ after_r
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(np.arccos(cosine))


class EmbodiedAgentEnv:
    def __init__(
        self,
        task: AgentEnvTask,
        task_spec: BenchmarkTaskSpec,
        profile: ObservationProfile,
        annotations: AnnotationCapabilities,
        run_dir: Path,
    ) -> None:
        self.task = task
        self.task_spec = task_spec
        self.profile = profile
        self.annotations = annotations
        self.run_dir = run_dir.resolve()
        self.observation_root = self.run_dir / "observations"
        self.public_transcript_path = self.run_dir / "agent_transcript.jsonl"
        self.outcome_path = self.run_dir / "evaluator_outcome.json"
        self.private_audit_path = self.run_dir / "evaluator_private_audit.json"
        self.protocol = BenchmarkEpisodeProtocol(profile)
        self._secret_seed = PRIVATE_EVALUATOR_SEED
        self._secret_salt = secrets.token_hex(32)
        self.seed_commitment = hashlib.sha256(
            (
                f"{COMMITMENT_DOMAIN}|{self.task_spec.name}|"
                f"{START_CONDITION}|"
                f"{self._secret_seed}|{self._secret_salt}"
            ).encode()
        ).hexdigest()
        self._observation_index = 0
        self._rollout_start_time = 0.0
        self._private_events: list[dict[str, Any]] = []
        self._inhand_reference_tracker = PostGraspReferenceTracker(
            required=(
                task_spec.requires_post_grasp_reference and not PRE_MOVE_ENABLED
            )
        )

    def write_manifest(self) -> None:
        manifest = self.protocol.contract_manifest()
        bundle_root = Path(os.environ["UNIVTAC_NVIDIA_RENDER_ROOT"])
        manifest.update(
            {
                "created_utc": utc_now(),
                "task": self.task_spec.to_manifest(pre_move=PRE_MOVE_ENABLED),
                "task_parameters": TASK_PARAMETERS,
                "start_condition": START_CONDITION,
                "pre_move_enabled": PRE_MOVE_ENABLED,
                "observation_profile": self.profile.to_manifest(),
                "annotations": self.annotations.to_manifest(),
                "episodes": 1,
                "seed": "hidden until terminal outcome",
                "seed_commitment_sha256": self.seed_commitment,
                "commitment_preimage_format": (
                    f"{COMMITMENT_DOMAIN}|{self.task_spec.name}|"
                    "<start_condition>|<decimal_seed>|<hex_salt>"
                ),
                "command_schema": public_benchmark_command_schema(),
                "public_tools": ["start_episode", "step_eef", "finish_episode"],
                "forbidden_agent_observations": [
                    "raw instance IDs, labels, and USD prim paths",
                    "actor or goal ground-truth poses",
                    "reward or task success before finish_episode",
                    "planner, IK, joint-target, or trajectory interfaces",
                ],
                "terminal_evaluation": {
                    "physics_settle_steps": FINISH_SETTLE_STEPS,
                    "policy": self.task_spec.terminal_policy,
                    "requires_release": self.task_spec.terminal_policy
                    == "released_stable_bottle_v1",
                    "requires_stability": self.task_spec.terminal_policy
                    == "released_stable_bottle_v1",
                },
                "rendering": "official evaluator-compatible livestream=2 experience",
                "nvidia_userspace": {
                    "version": os.environ["UNIVTAC_NVIDIA_RENDER_VERSION"],
                    "library_dir": os.environ["UNIVTAC_NVIDIA_USERSPACE_LIB_DIR"],
                    **audit_current_process(bundle_root),
                },
                "replay_video": "H.264 yuv420p fast-start MP4 from public panels only",
                "audit_files": {
                    "public": ["manifest.json", "agent_transcript.jsonl", "evaluator_outcome.json"],
                    "private_terminal_only": "evaluator_private_audit.json (mode 0600)",
                },
            }
        )
        write_json(self.run_dir / "manifest.json", manifest)

    def record_public(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.public_transcript_path,
            {"timestamp_utc": utc_now(), "kind": kind, **payload},
        )

    def _record_private(self, kind: str, payload: dict[str, Any]) -> None:
        self._private_events.append({"timestamp_utc": utc_now(), "kind": kind, **payload})

    def _next_observation_id(self) -> str:
        value = f"obs_{self._observation_index:03d}"
        self._observation_index += 1
        return value

    def _update_post_grasp_reference(
        self,
        *,
        delta_position: np.ndarray,
        delta_rpy: np.ndarray,
        delta_gripper: float,
        execution_succeeded: bool,
    ) -> None:
        arm_changed = bool(np.any(delta_position != 0) or np.any(delta_rpy != 0))
        phase = self._inhand_reference_tracker.after_action(
            arm_changed=arm_changed,
            delta_gripper=delta_gripper,
            execution_succeeded=execution_succeeded,
        )
        if phase == "captured_after_gripper_close":
            self.task.initialize_replay_task_phase()
        if phase is not None:
            self._record_private(
                "post_grasp_checker_reference",
                {
                    "phase": phase,
                    "prior_observation_id": self.protocol.current_observation_id,
                },
            )

    def _camera_info_for_instance(self, camera: Any) -> Any:
        info: Any = camera.data.info
        if isinstance(info, list):
            if len(info) != 1:
                raise RuntimeError("Expected exactly one environment in camera metadata")
            info = info[0]
        if isinstance(info, dict):
            return info.get(INSTANCE_DATA_TYPE, {})
        return {}

    def _capture_observation(
        self,
        observation_id: str,
        *,
        initial_observation: bool = False,
    ) -> dict[str, Any]:
        raw = self.task._get_observations()
        if raw.get("actor"):
            raise RuntimeError("Privileged actor observations must remain disabled")
        obs_dir = self.observation_root / observation_id
        modalities: dict[str, Any] = {}
        calibration: dict[str, Any] = {}
        public_annotations: dict[str, Any] = {}
        panels: list[tuple[str, np.ndarray]] = []

        robot = self.task._robot_manager.robot
        robot_position = to_numpy(robot.data.root_link_pos_w[0])
        robot_quaternion = to_numpy(robot.data.root_link_quat_w[0])
        annotation_prim_names = (
            self.task_spec.annotation_prim_names(self.task)
            if initial_observation and self.annotations.enabled_features
            else {}
        )

        for camera_name in self.profile.camera_names:
            camera_key = f"{camera_name}_rgb"
            rgb = tensor_to_rgb(raw["observation"][camera_name]["rgb"])
            rgb_artifact = image_artifact(obs_dir / camera_name / "rgb.png", rgb)
            modalities[camera_key] = rgb_artifact
            panels.append((camera_key, rgb))
            camera = self.task._camera_manager.cameras[camera_name]

            if self.profile.expose_metric_depth:
                depth_artifact, _, depth_preview = save_depth_artifacts(
                    obs_dir / camera_name,
                    camera.data.output["depth"],
                )
                modalities[f"{camera_name}_depth"] = depth_artifact
                panels.append((f"{camera_name}_depth_m", depth_preview))

            if self.profile.expose_camera_intrinsics:
                matrix = to_numpy(camera.data.intrinsic_matrices[0]).astype(np.float64)
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                    raise RuntimeError(f"Invalid {camera_name} camera intrinsics")
                calibration.setdefault(camera_name, {})["intrinsic_matrix_3x3"] = matrix.tolist()

            if self.profile.expose_camera_extrinsics:
                camera_position = to_numpy(camera.data.pos_w[0])
                camera_quaternion_ros = to_numpy(camera.data.quat_w_ros[0])
                extrinsic = robot_base_camera_extrinsic(
                    robot_position_world=robot_position,
                    robot_quaternion_world_wxyz=robot_quaternion,
                    camera_position_world=camera_position,
                    camera_quaternion_world_ros_wxyz=camera_quaternion_ros,
                )
                calibration.setdefault(camera_name, {})["extrinsics"] = {
                    "matrix_T_robot_base_camera_ros_4x4": extrinsic,
                    "camera_convention": "ROS optical: +Z forward, -Y up",
                    "translation_unit": "metre",
                }
                if camera_name == "wrist":
                    ee_position = np.asarray(
                        self.task._robot_manager.get_ee_pose().p, dtype=np.float64
                    )
                    distance = float(np.linalg.norm(camera_position - ee_position))
                    if distance >= 0.20:
                        raise RuntimeError(
                            "Wrist extrinsics appear stale: camera-to-EEF distance "
                            f"is {distance:.4f} m"
                        )

            if initial_observation and self.annotations.enabled_features:
                instance = camera_plane(camera.data.output[INSTANCE_DATA_TYPE])
                mapping = normalize_instance_mapping(self._camera_info_for_instance(camera))
                if not mapping:
                    raise RuntimeError(f"{camera_name} instance metadata is unavailable")
                camera_annotations: dict[str, Any] = {}
                private_camera: dict[str, Any] = {
                    "raw_id_to_labels": {str(key): value for key, value in mapping.items()},
                    "roles": {},
                }
                for role, private_names in annotation_prim_names.items():
                    mask, selected_ids = instance_role_mask(
                        instance, mapping, private_names
                    )
                    annotation, annotation_panels = save_annotation_artifacts(
                        obs_dir / "annotations" / camera_name,
                        role=role,
                        mask=mask,
                        rgb=rgb,
                        provide_bbox=self.annotations.provide_bbox,
                        provide_mask=self.annotations.provide_mask,
                    )
                    camera_annotations[role] = annotation
                    panels.extend(
                        (f"{camera_name}_{label}", panel)
                        for label, panel in annotation_panels
                    )
                    private_camera["roles"][role] = {
                        "private_prim_names": list(private_names),
                        "selected_instance_ids": selected_ids,
                        "area_px": int(mask.sum()),
                    }
                public_annotations[camera_name] = camera_annotations
                self._record_private(
                    "instance_annotation_mapping",
                    {"observation_id": observation_id, "camera": camera_name, **private_camera},
                )

        tactile_health: dict[str, Any] | None = None
        if self.profile.expose_tactile:
            tactile_health = {}
            for internal_name, public_name in (
                ("left_tactile", "left_tactile_rgb"),
                ("right_tactile", "right_tactile_rgb"),
            ):
                tactile = tensor_to_rgb(raw["tactile"][internal_name]["rgb_marker"])
                modalities[public_name] = image_artifact(
                    obs_dir / internal_name / "rgb_marker.png", tactile
                )
                panels.append((public_name, tactile))
                tactile_health[internal_name] = marker_grid_health(tactile)
            require_initial_tactile_health(
                tactile_health,
                initial_observation=initial_observation,
            )
            self._record_private(
                "tactile_renderer_health",
                {"observation_id": observation_id, "health": tactile_health},
            )

        joint_position = to_numpy(robot.data.joint_pos[0]).astype(np.float64)
        joint_velocity = to_numpy(robot.data.joint_vel[0]).astype(np.float64)
        if joint_position.shape != (9,) or joint_velocity.shape != (9,):
            raise RuntimeError("Expected a 9-DoF Franka robot state")
        ee_pose = np.asarray(
            self.task._robot_manager.get_ee_pose().totensor(), dtype=np.float64
        )
        robot_state = {
            "joint_position_9d": joint_position.tolist(),
            "joint_velocity_9d": joint_velocity.tolist(),
            "gripper_width_m": float(joint_position[-2] + joint_position[-1]),
            "end_effector_pose_robot_base_wxyz_7d": ee_pose.tolist(),
        }
        composite = save_composite(panels, obs_dir / "composite.png")
        composite["media_type"] = "image/png"
        composite["content_image"] = False
        observation: dict[str, Any] = {
            "schema_version": "univtac.embodied_observation.v1",
            "observation_id": observation_id,
            "task": self.task_spec.name,
            "observation_profile": self.profile.name,
            "modalities": modalities,
            "robot_state": robot_state,
            "artifacts": {"composite": composite},
        }
        if calibration:
            observation["camera_calibration"] = calibration
        if public_annotations:
            observation["annotations"] = public_annotations
        write_json(obs_dir / "observation.json", observation)
        self.record_public("observation", {"observation": observation})
        return observation

    def start(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.protocol.started:
            raise RuntimeError("The one-shot episode has already started")
        if "seed" in command:
            raise ValueError("The public API does not accept a caller-selected seed")
        self._rollout_start_time = time.perf_counter()
        self.task.mode = "eval"
        reset_stdout, reset_stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(reset_stdout), contextlib.redirect_stderr(reset_stderr):
            self.task.reset(seed=self._secret_seed, instructions=[TASK_INSTRUCTION])
        self.task.mean_steps = self.protocol.MAX_STEPS
        self.task.eval_success = False
        self.task.plan_success = True
        observation_id = self._next_observation_id()
        self.protocol.start(observation_id)
        observation = self._capture_observation(observation_id, initial_observation=True)
        self._record_private(
            "rollout_start",
            {
                "secret_seed": self._secret_seed,
                "start_condition": START_CONDITION,
                "pre_move_enabled": PRE_MOVE_ENABLED,
                "suppressed_reset_stdout_chars": len(reset_stdout.getvalue()),
                "suppressed_reset_stderr_chars": len(reset_stderr.getvalue()),
            },
        )
        return {
            "status": "rollout_started",
            "task": self.task_spec.name,
            "instruction": TASK_INSTRUCTION,
            "start_condition": START_CONDITION,
            "seed_commitment_sha256": self.seed_commitment,
            "observation": observation,
        }

    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        dp, dr, dg = self.protocol.prepare_step(
            observation_id=command.get("observation_id"),
            delta_position=command.get("delta_position"),
            delta_rpy=command.get("delta_rpy"),
            delta_gripper=command.get("delta_gripper"),
            current_gripper_qpos=self.task._robot_manager.get_gripper_qpos(),
        )
        prior_observation_id = self.protocol.current_observation_id
        action = np.concatenate((dp, dr, [dg])).astype(np.float32)
        self.task.eval_success = False
        self.task.plan_success = True
        started = time.perf_counter()
        execution_succeeded, private_checker_value = self.task.take_action(
            action, action_type="delta_ee"
        )
        duration = time.perf_counter() - started
        self._update_post_grasp_reference(
            delta_position=dp,
            delta_rpy=dr,
            delta_gripper=dg,
            execution_succeeded=bool(execution_succeeded),
        )
        # Generic benchmark success is terminal-only.  Clear BaseTask's early
        # latching so a pose that is momentarily valid cannot freeze later steps.
        self.task.eval_success = False
        if not execution_succeeded:
            self.task.plan_success = True
        observation_id = self._next_observation_id()
        self.protocol.complete_step(observation_id)
        observation = self._capture_observation(observation_id)
        self._record_private(
            "step_checker",
            {
                "after_observation_id": observation_id,
                "base_task_checker_value": bool(private_checker_value),
                "execution_succeeded": bool(execution_succeeded),
                "control_route": getattr(self.task, "last_delta_ee_route", None),
                "curobo_motion_gen": getattr(
                    self.task._robot_manager,
                    "last_arm_plan_diagnostics",
                    None,
                ),
            },
        )
        return {
            "status": "action_complete",
            "prior_observation_id": prior_observation_id,
            "action": {
                "primitive": "step_eef_delta",
                "delta_position_world_m": dp.tolist(),
                "delta_rpy_world_rad": dr.tolist(),
                "delta_gripper_m": dg,
            },
            "execution_succeeded": bool(execution_succeeded),
            "execution_duration_seconds": duration,
            "observation": observation,
        }

    def finish(self, command: dict[str, Any]) -> dict[str, Any]:
        current_observation_id = command.get("observation_id")
        self.protocol.finish(current_observation_id)
        manipulated = self.task_spec.manipulated_actor(self.task)
        pose_before = pose_snapshot(manipulated)
        self.task.eval_success = False
        self.task.plan_success = True
        self.task.delay(steps=FINISH_SETTLE_STEPS, is_save=False, force=True)
        pose_after = pose_snapshot(manipulated)
        drift_translation, drift_rotation = pose_drift(pose_before, pose_after)
        base_success = bool(self.task.check_success())
        evaluator_checks: dict[str, Any] = {
            "base_task_success": base_success,
            "settle_steps": FINISH_SETTLE_STEPS,
        }
        official_success = base_success
        if self.task_spec.terminal_policy == "released_stable_bottle_v1":
            gripper_qpos = float(self.task._robot_manager.get_gripper_qpos())
            released = gripper_qpos >= BOTTLE_RELEASE_GRIPPER_QPOS_M
            stable = (
                drift_translation <= BOTTLE_STABLE_TRANSLATION_M
                and drift_rotation <= BOTTLE_STABLE_ROTATION_RAD
            )
            evaluator_checks.update(
                {
                    "released": released,
                    "gripper_qpos_m": gripper_qpos,
                    "minimum_release_gripper_qpos_m": BOTTLE_RELEASE_GRIPPER_QPOS_M,
                    "stable_after_release": stable,
                    "settle_translation_drift_m": drift_translation,
                    "settle_rotation_drift_rad": drift_rotation,
                    "max_translation_drift_m": BOTTLE_STABLE_TRANSLATION_M,
                    "max_rotation_drift_rad": BOTTLE_STABLE_ROTATION_RAD,
                }
            )
            official_success = bool(base_success and released and stable)
        elif self.task_spec.terminal_policy not in {
            BASE_TASK_SUCCESS,
            "pull_out_key_v1",
        }:
            raise RuntimeError(f"Unknown terminal policy {self.task_spec.terminal_policy!r}")

        final_observation_id = self._next_observation_id()
        final_observation = self._capture_observation(final_observation_id)
        try:
            replay_video = encode_observation_video(
                self.observation_root,
                self.run_dir / "agent_observations_h264.mp4",
            )
        except Exception as exc:
            replay_video = {"error_type": type(exc).__name__, "message": str(exc)}
        outcome = {
            "timestamp_utc": utc_now(),
            "terminal_reason": "agent_finish",
            "task": self.task_spec.name,
            "start_condition": START_CONDITION,
            "pre_move_enabled": PRE_MOVE_ENABLED,
            "observation_profile": self.profile.name,
            "profile_index": self.profile.index,
            "annotations": self.annotations.to_manifest(),
            "official_task_success": official_success,
            "evaluator_checks": evaluator_checks,
            "step_eef_count": self.protocol.step_count,
            "sim_action_count": int(self.task.take_action_cnt),
            "wall_seconds": time.perf_counter() - self._rollout_start_time,
            "final_observation_id": final_observation_id,
            "seed_reveal": self._secret_seed,
            "salt_reveal": self._secret_salt,
            "seed_commitment_sha256": self.seed_commitment,
            "commitment_verified": True,
            "replay_video": replay_video,
        }
        write_json(self.outcome_path, outcome)
        write_json(
            self.private_audit_path,
            {
                "materialized_utc": utc_now(),
                "terminal": True,
                "task": self.task_spec.name,
                "pose_before_settle": pose_before,
                "pose_after_settle": pose_after,
                "events": self._private_events,
                "outcome": outcome,
            },
            private=True,
        )
        self.record_public("terminal_outcome", outcome)
        return {
            "status": "rollout_finished",
            **outcome,
            "observation": final_observation,
        }

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        validate_benchmark_command_fields(command)
        name = command["command"]
        if name == "start":
            return self.start(command)
        if name == "step":
            return self.step(command)
        if name == "finish":
            return self.finish(command)
        if name == "close":
            if not self.protocol.terminal:
                raise RuntimeError("close is allowed only after terminal evaluation")
            return {"status": "closing", "run_dir": str(self.run_dir)}
        raise ValueError(f"Unknown command: {name!r}")


def make_run_dir() -> Path:
    run_dir = ARGS.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = (
            REPO_ROOT
            / "agent_runs"
            / (
                f"{TASK_SPEC.name}_agentenv_p{PROFILE.index}_"
                f"{START_CONDITION}_{stamp}"
            )
        )
    elif not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_task(run_dir: Path) -> AgentEnvTask:
    cfg = TASK_CFG_CLASS()
    if TASK_SPEC.name == "pull_out_key":
        cfg.key_initial_relative_yaw_rad = ARGS.key_initial_relative_yaw_rad
    cfg.save_dir = run_dir / "simulator_internal"
    cfg.scene.num_envs = 1
    cfg.reset_time_limit = RESET_TIME_LIMIT_SECONDS
    if ARGS.device is not None:
        cfg.sim.device = ARGS.device
    cfg.decimation = 1
    camera_observations = ["rgb"]
    camera_data_types = ["rgb"]
    if PROFILE.expose_metric_depth:
        camera_observations.append("depth")
        camera_data_types.append("depth")
    if ANNOTATIONS.enabled_features:
        camera_data_types.append(INSTANCE_DATA_TYPE)
    cfg.obs_data_type = {
        "camera": camera_observations,
        "embodiment": ["joint", "ee"],
    }
    if PROFILE.expose_tactile:
        cfg.obs_data_type["tactile"] = ["rgb_marker"]
    if {camera.name for camera in cfg.cameras} != {"head", "wrist"}:
        raise RuntimeError("TaskCfg must provide exactly head and wrist cameras")
    for camera in cfg.cameras:
        camera.data_types = list(camera_data_types)
        camera.update_latest_camera_pose = PROFILE.expose_camera_extrinsics
        if ANNOTATIONS.enabled_features:
            camera.colorize_instance_id_segmentation = False
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False
    cfg.step_lim = max(int(cfg.step_lim), BenchmarkEpisodeProtocol.MAX_STEPS)
    return AgentEnvTask(cfg, mode="eval")


def emit(env: EmbodiedAgentEnv, payload: dict[str, Any]) -> None:
    env.record_public("response", {"response": payload})
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_safe), flush=True)


def main() -> None:
    run_dir = make_run_dir()
    task = make_task(run_dir)
    env = EmbodiedAgentEnv(task, TASK_SPEC, PROFILE, ANNOTATIONS, run_dir)
    env.write_manifest()
    env.record_public(
        "bridge_ready",
        {
            "run_dir": str(run_dir.resolve()),
            "task": TASK_SPEC.name,
            "task_parameters": TASK_PARAMETERS,
            "start_condition": START_CONDITION,
            "pre_move_enabled": PRE_MOVE_ENABLED,
            "observation_profile": PROFILE.name,
            "annotations": ANNOTATIONS.to_manifest(),
            "seed_commitment_sha256": env.seed_commitment,
        },
    )
    emit(
        env,
        {
            "status": "ready",
            "task": TASK_SPEC.name,
            "task_parameters": TASK_PARAMETERS,
            "start_condition": START_CONDITION,
            "pre_move_enabled": PRE_MOVE_ENABLED,
            "observation_profile": PROFILE.to_manifest(),
            "annotations": ANNOTATIONS.to_manifest(),
            "run_dir": str(run_dir.resolve()),
            "seed_commitment_sha256": env.seed_commitment,
            "command_schema": public_benchmark_command_schema(),
            "agent_tools": ["start_episode", "step_eef", "finish_episode"],
        },
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
                env.record_public("command", {"command": command})
                response = env.handle(command)
                emit(env, response)
                if command.get("command") == "close":
                    close_requested = True
                    break
            except Exception as exc:
                error = {
                    "status": "command_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                env.record_public("command_error", error)
                env._record_private(
                    "command_error_traceback",
                    {"error": error, "traceback": traceback.format_exc()},
                )
                emit(env, error)
    except Exception:
        env._record_private("fatal_error", {"traceback": traceback.format_exc()})
        raise
    finally:
        if not close_requested:
            env.record_public("bridge_stopped_without_close", {})
        task.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()

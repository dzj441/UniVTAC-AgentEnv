#!/usr/bin/env python3
"""Configurable and auditable AgentEnv v0 runner for ``grasp_classify``.

The process owns all simulator-private state and exposes newline-delimited JSON
commands on stdin.  Every response starts with ``AGENT_ENV_RESULT ``.  Level 1
shows vision and robot state, Level 2 additionally shows tactile marker RGB,
and Level 3 additionally returns task success after ``submit_prediction``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
    sys.path.append(str(REPO_ROOT))

import numpy as np

from agent_env.artifacts import (
    encode_observation_video,
    marker_grid_health,
    require_initial_tactile_health,
    save_composite,
    save_rgb,
    tensor_to_rgb,
)
from agent_env.contract import (
    consume_private_evaluator_seed,
    public_command_schema,
    validate_command_fields,
)
from agent_env.profiles import AgentEnvProfile, get_profile
from agent_env.protocol import EpisodeProtocol
from agent_env.nvidia_runtime import audit_current_process
from agent_env.serialization import serialize_observation

from isaaclab.app import AppLauncher


RESULT_PREFIX = "AGENT_ENV_RESULT "
COMMITMENT_DOMAIN = "univtac.agent_env.grasp_classify.v0"
RESET_TIME_LIMIT_SECONDS = 240.0

# A controlled evaluator may select one seed for every Level in a matrix. Read
# and remove it before AppLauncher starts any child tool. Only its commitment is
# public before termination; the value itself is revealed in the outcome.
PRIVATE_EVALUATOR_SEED = consume_private_evaluator_seed(os.environ)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVTAC grasp_classify AgentEnv v0")
    parser.add_argument("--level", required=True, choices=("1", "2", "3"))
    parser.add_argument("--run-dir", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    # This experience is required for the TacEx marker grid to render correctly.
    args.livestream = 2
    return args


ARGS = parse_args()
PROFILE = get_profile(ARGS.level)
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

# Imports that require a live Isaac application stay below AppLauncher.
from envs.grasp_classify import Task, TaskCfg  # noqa: E402


class AgentEnvTask(Task):
    """Prevent a private seed from appearing in BaseTask-generated paths."""

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
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_safe) + "\n")


def write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


class GraspClassifyAgentEnv:
    def __init__(self, task: AgentEnvTask, profile: AgentEnvProfile, run_dir: Path):
        self.task = task
        self.profile = profile
        self.run_dir = run_dir.resolve()
        self.observation_root = self.run_dir / "observations"
        self.host_sensor_metadata_root = self.run_dir / ".host_sensor_metadata"
        self.host_sensor_metadata_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.host_sensor_metadata_root, 0o700)
        self.public_transcript_path = self.run_dir / "agent_transcript.jsonl"
        self.outcome_path = self.run_dir / "evaluator_outcome.json"
        self.private_audit_path = self.run_dir / "evaluator_private_audit.json"
        self.protocol = EpisodeProtocol(profile)

        self._secret_seed = PRIVATE_EVALUATOR_SEED
        self._secret_salt = secrets.token_hex(32)
        self.seed_commitment = hashlib.sha256(
            f"{COMMITMENT_DOMAIN}|{self._secret_seed}|{self._secret_salt}".encode()
        ).hexdigest()
        self._observation_index = 0
        self._rollout_start_time = 0.0
        self._private_events: list[dict[str, Any]] = []
        self._ever_task_success = False

    def write_manifest(self) -> None:
        manifest = self.protocol.contract_manifest()
        nvidia_bundle_root = Path(os.environ["UNIVTAC_NVIDIA_RENDER_ROOT"])
        nvidia_runtime_audit = audit_current_process(nvidia_bundle_root)
        manifest.update(
            {
                "created_utc": utc_now(),
                "task": "grasp_classify",
                "episodes": 1,
                "seed": "hidden until terminal outcome",
                "seed_commitment_sha256": self.seed_commitment,
                "commitment_preimage_format": (
                    f"{COMMITMENT_DOMAIN}|<decimal_seed>|<hex_salt>"
                ),
                "command_schema": public_command_schema(),
                "forbidden_agent_observations": [
                    "actor/object/pad pose",
                    "selected prism class before terminal outcome",
                    "raw simulator camera depth",
                    "camera intrinsics/extrinsics",
                    "contact force or reward",
                    "task success at Level 1 or Level 2",
                    "task success before submit_prediction at Level 3",
                ],
                "execution_feedback_note": (
                    "execution_succeeded is retained for host audit but removed from the "
                    "agent-visible projection; the next robot state is the action feedback"
                ),
                "rendering": "official evaluator-compatible livestream=2 experience",
                "nvidia_userspace": {
                    "version": os.environ["UNIVTAC_NVIDIA_RENDER_VERSION"],
                    "library_dir": os.environ["UNIVTAC_NVIDIA_USERSPACE_LIB_DIR"],
                    "kernel_module_version_checked": os.environ[
                        "UNIVTAC_NVIDIA_RENDER_VERSION"
                    ],
                    **nvidia_runtime_audit,
                },
                "initialization_reset_time_limit_seconds": self.task.cfg.reset_time_limit,
                "replay_video": "H.264 yuv420p fast-start MP4 built only from public panels",
                "audit_files": {
                    "public": "agent_transcript.jsonl and evaluator_outcome.json",
                    "private_until_terminal": "evaluator_private_audit.json is materialized at terminal",
                },
                "semantic_perception_host_contract": (
                    "Exact camera intrinsics are stored in a mode-0600 host sidecar for "
                    "calibrated derived models. They are never serialized into the public "
                    "observation, transcript, Codex prompt, or tool response."
                ),
                "threat_model": (
                    "The API and public transcript prevent accidental information leakage. "
                    "Adversarial OS-level isolation still requires running the agent under a "
                    "different UID/container from this simulator process."
                ),
            }
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def record_public(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.public_transcript_path,
            {"timestamp_utc": utc_now(), "kind": kind, **payload},
        )

    def _record_private(self, kind: str, payload: dict[str, Any]) -> None:
        # Kept in memory until termination so no live episode secret is placed
        # in an agent-readable artifact path.
        self._private_events.append({"timestamp_utc": utc_now(), "kind": kind, **payload})

    @staticmethod
    def _require_rationale(command: dict[str, Any]) -> str:
        rationale = str(command.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Every physical or prediction decision requires a non-empty rationale")
        return rationale

    def _next_observation_id(self) -> str:
        observation_id = f"obs_{self._observation_index:03d}"
        self._observation_index += 1
        return observation_id

    def _capture_observation(
        self,
        observation_id: str,
        *,
        initial_observation: bool = False,
    ) -> dict[str, Any]:
        raw = self.task._get_observations()
        if raw.get("actor"):
            raise RuntimeError("Fairness violation: privileged actor observations are enabled")

        camera_keys = set(raw["observation"])
        if camera_keys != {"head", "wrist"}:
            raise RuntimeError(f"Expected head+wrist RGB, got camera keys {sorted(camera_keys)}")

        obs_dir = self.observation_root / observation_id
        head = tensor_to_rgb(raw["observation"]["head"]["rgb"])
        wrist = tensor_to_rgb(raw["observation"]["wrist"]["rgb"])
        modality_artifacts: dict[str, dict[str, str]] = {
            "head_rgb": save_rgb(obs_dir / "head_rgb.png", head),
            "wrist_rgb": save_rgb(obs_dir / "wrist_rgb.png", wrist),
        }
        self._write_host_camera_metadata(observation_id, head, wrist)
        panels = [("head_rgb", head), ("wrist_rgb", wrist)]
        tactile_health: dict[str, Any] | None = None

        if self.profile.expose_tactile:
            tactile_keys = set(raw["tactile"])
            if tactile_keys != {"left_tactile", "right_tactile"}:
                raise RuntimeError(f"Unexpected tactile keys: {sorted(tactile_keys)}")
            left = tensor_to_rgb(raw["tactile"]["left_tactile"]["rgb_marker"])
            right = tensor_to_rgb(raw["tactile"]["right_tactile"]["rgb_marker"])
            tactile_health = {
                "left": marker_grid_health(left),
                "right": marker_grid_health(right),
            }
            require_initial_tactile_health(
                tactile_health,
                initial_observation=initial_observation,
            )
            modality_artifacts.update(
                {
                    "left_tactile_marker": save_rgb(
                        obs_dir / "left_tactile_marker.png", left
                    ),
                    "right_tactile_marker": save_rgb(
                        obs_dir / "right_tactile_marker.png", right
                    ),
                }
            )
            panels.extend(
                [("left_tactile_marker", left), ("right_tactile_marker", right)]
            )

        composite = save_composite(panels, obs_dir / "composite.png")
        joint = [
            float(value)
            for value in raw["embodiment"]["joint"][:8].detach().cpu().tolist()
        ]
        ee_pose = [
            float(value)
            for value in raw["embodiment"]["ee"][:7].detach().cpu().tolist()
        ]
        robot_state = {
            "joint_position_8d": joint,
            "gripper_qpos": float(joint[7]),
            "end_effector_pose_robot_base_7d": ee_pose,
        }
        observation = serialize_observation(
            profile=self.profile,
            observation_id=observation_id,
            stage=self.protocol.stage,
            available_modalities=modality_artifacts,
            available_robot_state=robot_state,
            probe_count=self.protocol.probe_count,
            post_prediction_action_count=self.protocol.action_count,
            tactile_health=tactile_health,
            artifacts={"composite": composite},
        )
        self.record_public("observation", {"observation": observation})
        return observation

    def _write_host_camera_metadata(
        self,
        observation_id: str,
        head: np.ndarray,
        wrist: np.ndarray,
    ) -> None:
        """Persist calibrated K for host tools without widening the agent surface."""

        cameras: dict[str, Any] = {}
        for public_name, internal_name, image in (
            ("head_rgb", "head", head),
            ("wrist_rgb", "wrist", wrist),
        ):
            tensor = self.task._camera_manager.cameras[
                internal_name
            ].data.intrinsic_matrices
            matrix = np.asarray(tensor[0].detach().cpu(), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise RuntimeError(f"Invalid calibrated intrinsics for {internal_name}")
            height, width = int(image.shape[0]), int(image.shape[1])
            fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
            cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
            if fx <= 0 or fy <= 0 or not (0 <= cx < width and 0 <= cy < height):
                raise RuntimeError(f"Out-of-range calibrated intrinsics for {internal_name}")
            cameras[public_name] = {
                "width": width,
                "height": height,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            }
        write_private_json(
            self.host_sensor_metadata_root / f"{observation_id}.json",
            {
                "schema_version": "univtac.host_sensor_metadata.v1",
                "observation_id": observation_id,
                "cameras": cameras,
            },
        )

    def _internal_task_success(self) -> bool:
        return bool(self.task.eval_success or self.task.check_success())

    def _observe_internal_success(self, success: bool) -> None:
        self._ever_task_success = self._ever_task_success or bool(success)
        # L1/L2 must not acquire a success side channel through BaseTask's
        # early no-op behavior.  Keep executing until agent finish/budget.
        if not self.profile.expose_task_success_after_prediction:
            self.task.eval_success = False

    def start(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.protocol.started:
            raise RuntimeError("The one-shot episode has already started")
        if "seed" in command:
            raise ValueError("The public API does not accept a caller-selected seed")
        self._rollout_start_time = time.perf_counter()
        self.task.mode = "eval"
        reset_stdout, reset_stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(reset_stdout), contextlib.redirect_stderr(reset_stderr):
            self.task.reset(seed=self._secret_seed, instructions=["Empty"])
        self.task.mean_steps = self.task.cfg.step_lim

        observation_id = self._next_observation_id()
        self.protocol.start(observation_id)
        observation = self._capture_observation(
            observation_id,
            initial_observation=True,
        )
        self._record_private(
            "rollout_start",
            {
                "secret_seed": self._secret_seed,
                "true_class": str(self.task.choice),
                "suppressed_reset_stdout_chars": len(reset_stdout.getvalue()),
                "suppressed_reset_stderr_chars": len(reset_stderr.getvalue()),
            },
        )
        return {
            "status": "rollout_started",
            "level": self.profile.level,
            "profile": self.profile.name,
            "seed_commitment_sha256": self.seed_commitment,
            "stage": self.protocol.stage,
            "required_before_translation": "submit_prediction",
            "optional_before_prediction": f"up to {self.protocol.MAX_PROBES} gripper-only probes",
            "task_success_feedback": (
                "locked until submit_prediction"
                if self.profile.expose_task_success_after_prediction
                else "not available at this level"
            ),
            "observation": observation,
        }

    def probe(self, command: dict[str, Any]) -> dict[str, Any]:
        rationale = self._require_rationale(command)
        delta_gripper = float(command.get("delta_gripper", 0.0))
        self.protocol.prepare_probe(command.get("observation_id"), delta_gripper)
        prior_observation_id = self.protocol.current_observation_id
        action_vector = np.asarray([0, 0, 0, 0, 0, 0, delta_gripper], dtype=np.float32)
        started = time.perf_counter()
        executed, returned_success = self.task.take_action(action_vector, action_type="delta_ee")
        duration = time.perf_counter() - started
        internal_success = bool(returned_success or self._internal_task_success())
        # A pre-prediction checker result is never eligible for success and is
        # never public, including at Level 3.
        self.task.eval_success = False

        next_observation_id = self._next_observation_id()
        self.protocol.complete_probe(next_observation_id)
        observation = self._capture_observation(next_observation_id)
        self._record_private(
            "pre_prediction_checker",
            {"after_observation_id": next_observation_id, "task_success": internal_success},
        )
        return {
            "status": "probe_complete",
            "probe_index": self.protocol.probe_count,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {"primitive": "gripper_only_probe", "delta_gripper": delta_gripper},
            "feedback": {"execution_succeeded": bool(executed)},
            "execution_duration_seconds": duration,
            "remaining_probes": self.protocol.MAX_PROBES - self.protocol.probe_count,
            "observation": observation,
        }

    def submit_prediction(self, command: dict[str, Any]) -> dict[str, Any]:
        rationale = self._require_rationale(command)
        result = self.protocol.submit_prediction(
            observation_id=command.get("observation_id"),
            predicted_class=command.get("predicted_class"),
            target_pad=command.get("target_pad"),
        )
        record = {
            **result,
            "observation_id": self.protocol.current_observation_id,
            "rationale": rationale,
            "success_seen_before_prediction": False,
        }
        self.record_public("irreversible_prediction", record)
        return {
            "status": "prediction_submitted",
            **record,
            "stage": self.protocol.stage,
            "task_success_feedback": (
                "available after each subsequent act/wait"
                if self.profile.expose_task_success_after_prediction
                else "not available at this level"
            ),
            "remaining_post_prediction_actions": (
                self.protocol.MAX_POST_PREDICTION_ACTIONS - self.protocol.action_count
            ),
        }

    def act(self, command: dict[str, Any]) -> dict[str, Any]:
        rationale = self._require_rationale(command)
        dp, dr, dg = self.protocol.prepare_delta(
            observation_id=command.get("observation_id"),
            delta_position=command.get("delta_position", [0, 0, 0]),
            delta_rpy=command.get("delta_rpy", [0, 0, 0]),
            delta_gripper=command.get("delta_gripper", 0),
        )
        return self._execute_post_prediction_delta(command, rationale, dp, dr, dg)

    def _execute_post_prediction_delta(
        self,
        command: dict[str, Any],
        rationale: str,
        dp: np.ndarray,
        dr: np.ndarray,
        dg: float,
    ) -> dict[str, Any]:
        prior_observation_id = self.protocol.current_observation_id
        action_vector = np.concatenate((dp, dr, [dg])).astype(np.float32)
        started = time.perf_counter()
        executed, returned_success = self.task.take_action(action_vector, action_type="delta_ee")
        duration = time.perf_counter() - started
        internal_success = bool(returned_success or self._internal_task_success())
        self._observe_internal_success(internal_success)

        next_observation_id = self._next_observation_id()
        self.protocol.complete_delta(next_observation_id)
        observation = self._capture_observation(next_observation_id)
        feedback = self.protocol.public_action_feedback(
            execution_succeeded=bool(executed), internal_task_success=internal_success
        )
        self._record_private(
            "post_prediction_checker",
            {
                "after_observation_id": next_observation_id,
                "task_success": internal_success,
                "execution_succeeded": bool(executed),
            },
        )
        record = {
            "post_prediction_action_index": self.protocol.action_count,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {
                "primitive": "bounded_delta_ee",
                "delta_position_world_m": dp.tolist(),
                "delta_rpy_world_rad": dr.tolist(),
                "delta_gripper": dg,
            },
            "committed_target": self.protocol.committed_target,
            "feedback": feedback,
            "execution_duration_seconds": duration,
            "observation": observation,
        }
        return self._post_action_response(record, internal_success)

    def wait(self, command: dict[str, Any]) -> dict[str, Any]:
        rationale = self._require_rationale(command)
        steps = self.protocol.prepare_wait(
            command.get("observation_id"), command.get("steps", 20)
        )
        prior_observation_id = self.protocol.current_observation_id
        started = time.perf_counter()
        executed = self.task.delay(steps=steps, is_save=False, force=False)
        duration = time.perf_counter() - started
        internal_success = self._internal_task_success()
        self._observe_internal_success(internal_success)

        next_observation_id = self._next_observation_id()
        self.protocol.complete_wait(next_observation_id)
        observation = self._capture_observation(next_observation_id)
        feedback = self.protocol.public_action_feedback(
            execution_succeeded=bool(executed), internal_task_success=internal_success
        )
        self._record_private(
            "post_prediction_checker",
            {
                "after_observation_id": next_observation_id,
                "task_success": internal_success,
                "execution_succeeded": bool(executed),
            },
        )
        record = {
            "post_prediction_action_index": self.protocol.action_count,
            "prior_observation_id": prior_observation_id,
            "rationale": rationale,
            "action": {"primitive": "wait", "physics_steps": steps},
            "committed_target": self.protocol.committed_target,
            "feedback": feedback,
            "execution_duration_seconds": duration,
            "observation": observation,
        }
        return self._post_action_response(record, internal_success)

    def _post_action_response(
        self, record: dict[str, Any], internal_success: bool
    ) -> dict[str, Any]:
        if self.protocol.should_auto_stop(internal_success):
            self.protocol.force_terminal()
            return self._finalize(
                reason="success_guidance_auto_stop",
                final_note="Level 3 task_success became true; episode stopped immediately.",
                final_observation=record["observation"],
                last_action=record,
            )
        if self.protocol.action_count >= self.protocol.MAX_POST_PREDICTION_ACTIONS:
            self.protocol.force_terminal()
            return self._finalize(
                reason="action_budget_exhausted",
                final_note="Fixed post-prediction action budget exhausted.",
                final_observation=record["observation"],
                last_action=record,
            )
        return {
            "status": "action_complete",
            **record,
            "remaining_post_prediction_actions": (
                self.protocol.MAX_POST_PREDICTION_ACTIONS - self.protocol.action_count
            ),
        }

    def finish(self, command: dict[str, Any]) -> dict[str, Any]:
        final_note = str(command.get("final_note", "")).strip()
        if not final_note:
            raise ValueError("finish requires a non-empty final_note")
        self.protocol.finish(command.get("observation_id"))
        self._observe_internal_success(self._internal_task_success())
        return self._finalize(
            reason="agent_finish",
            final_note=final_note,
            final_observation=None,
            last_action=None,
        )

    def _finalize(
        self,
        *,
        reason: str,
        final_note: str,
        final_observation: dict[str, Any] | None,
        last_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.protocol.terminal:
            self.protocol.force_terminal()
        true_class = str(self.task.choice)
        expected_target = self.protocol.CLASS_TO_TARGET[true_class]
        classification_correct = self.protocol.predicted_class == true_class
        committed_pad_correct = self.protocol.committed_target == expected_target
        verifier = hashlib.sha256(
            f"{COMMITMENT_DOMAIN}|{self._secret_seed}|{self._secret_salt}".encode()
        ).hexdigest()
        if verifier != self.seed_commitment:
            raise RuntimeError("Internal seed commitment verification failed")

        try:
            replay_video: dict[str, Any] = encode_observation_video(
                self.observation_root,
                self.run_dir / "agent_observations_h264.mp4",
            )
        except Exception as exc:
            replay_video = {"error_type": type(exc).__name__, "message": str(exc)}

        outcome = {
            "timestamp_utc": utc_now(),
            "terminal_reason": reason,
            "level": self.profile.level,
            "profile": self.profile.name,
            "predicted_class": self.protocol.predicted_class,
            "true_class": true_class,
            "classification_correct": classification_correct,
            "committed_target": self.protocol.committed_target,
            "expected_target": expected_target,
            "committed_pad_correct": committed_pad_correct,
            "official_task_success": bool(self._ever_task_success),
            "probe_count": self.protocol.probe_count,
            "post_prediction_action_count": self.protocol.action_count,
            "sim_action_count": int(self.task.take_action_cnt),
            "wall_seconds": time.perf_counter() - self._rollout_start_time,
            "final_observation_id": self.protocol.current_observation_id,
            "final_note": final_note,
            "seed_reveal": self._secret_seed,
            "salt_reveal": self._secret_salt,
            "seed_commitment_sha256": self.seed_commitment,
            "commitment_verified": True,
            "replay_video": replay_video,
        }
        self.outcome_path.write_text(
            json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_private_json(
            self.private_audit_path,
            {
                "materialized_utc": utc_now(),
                "terminal": True,
                "events": self._private_events,
                "outcome": outcome,
            },
        )
        self.record_public("terminal_outcome", outcome)
        return {
            "status": "rollout_finished",
            **outcome,
            "observation": final_observation,
            "last_action": last_action,
        }

    def status(self) -> dict[str, Any]:
        return {
            "status": "bridge_status",
            **self.protocol.public_status(),
            "seed_commitment_sha256": self.seed_commitment,
            "run_dir": str(self.run_dir),
        }

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        validate_command_fields(command)
        name = command.get("command")
        if name == "start":
            return self.start(command)
        if name == "probe":
            return self.probe(command)
        if name == "submit_prediction":
            return self.submit_prediction(command)
        if name == "act":
            return self.act(command)
        if name == "wait":
            return self.wait(command)
        if name == "finish":
            return self.finish(command)
        if name == "status":
            return self.status()
        if name == "close":
            if not self.protocol.terminal:
                raise RuntimeError("close is allowed only after the one-shot episode terminates")
            return {"status": "closing", "run_dir": str(self.run_dir)}
        raise ValueError(f"Unknown command: {name!r}")


def make_run_dir(profile: AgentEnvProfile) -> Path:
    run_dir = ARGS.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "agent_runs" / f"grasp_classify_agentenv_l{profile.level}_{stamp}"
    elif not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_task(run_dir: Path, profile: AgentEnvProfile) -> AgentEnvTask:
    cfg = TaskCfg()
    cfg.save_dir = run_dir / "simulator_internal"
    cfg.scene.num_envs = 1
    # UI/RTX initialization jitter can push the official 120 s guard slightly
    # over its limit on shared machines. This changes only setup patience, not
    # the task state, controller, action budget, or terminal checker.
    cfg.reset_time_limit = RESET_TIME_LIMIT_SECONDS
    if ARGS.device is not None:
        cfg.sim.device = ARGS.device
    cfg.decimation = 1
    cfg.obs_data_type = {
        "camera": ["rgb"],
        "embodiment": ["joint", "ee"],
    }
    if profile.expose_tactile:
        cfg.obs_data_type["tactile"] = ["rgb_marker"]
    if {camera.name for camera in cfg.cameras} != {"head", "wrist"}:
        raise RuntimeError("TaskCfg must provide exactly head and wrist cameras")
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False
    return AgentEnvTask(cfg, mode="eval")


def emit(env: GraspClassifyAgentEnv, payload: dict[str, Any]) -> None:
    env.record_public("response", {"response": payload})
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_safe), flush=True)


def main() -> None:
    run_dir = make_run_dir(PROFILE)
    task = make_task(run_dir, PROFILE)
    env = GraspClassifyAgentEnv(task, PROFILE, run_dir)
    env.write_manifest()
    env.record_public(
        "bridge_ready",
        {
            "run_dir": str(run_dir.resolve()),
            "level": PROFILE.level,
            "profile": PROFILE.name,
            "seed_commitment_sha256": env.seed_commitment,
        },
    )
    emit(
        env,
        {
            "status": "ready",
            "task": "grasp_classify",
            "level": PROFILE.level,
            "profile": PROFILE.name,
            "capabilities": PROFILE.to_manifest(),
            "run_dir": str(run_dir.resolve()),
            "seed_commitment_sha256": env.seed_commitment,
            "command_schema": public_command_schema(),
            "commands": [
                "start",
                "probe",
                "submit_prediction",
                "act",
                "wait",
                "finish",
                "status",
                "close",
            ],
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
        SIMULATION_APP.close()


if __name__ == "__main__":
    main()

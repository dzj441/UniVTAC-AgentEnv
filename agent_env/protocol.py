"""Pure state machine for the ``grasp_classify`` AgentEnv v0 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .profiles import AgentEnvProfile


class ProtocolError(RuntimeError):
    """Raised when an agent command violates the public episode contract."""


@dataclass(frozen=True)
class PreparedDelta:
    cumulative_y_m: float
    target_halfspace_locked: bool


class EpisodeProtocol:
    """Track public episode state without access to simulator-private truth.

    ``submit_prediction`` is the deliberate extension seam for future early
    failure guidance.  Guidance channels are locked before this transition and
    target switching is structurally impossible afterwards.
    """

    CLASS_TO_TARGET = {"rough": "orange", "plain": "green"}
    TARGET_Y_SIGN = {"green": 1.0, "orange": -1.0}

    MAX_TRANSLATION_COMPONENT_M = 0.04
    MAX_TRANSLATION_NORM_M = 0.06
    MAX_ROTATION_COMPONENT_RAD = 0.35
    MAX_PROBE_GRIPPER_DELTA = 0.002
    MAX_ACTION_GRIPPER_DELTA = 0.005
    MAX_PROBES = 2
    MAX_POST_PREDICTION_ACTIONS = 10
    MAX_WAIT_PHYSICS_STEPS = 60
    MIN_LOCKED_TARGETWARD_Y_M = 0.02

    def __init__(self, profile: AgentEnvProfile):
        self.profile = profile
        self.started = False
        self.active = False
        self.terminal = False
        self.current_observation_id: str | None = None
        self.predicted_class: str | None = None
        self.committed_target: str | None = None
        self.probe_count = 0
        self.action_count = 0
        self.cumulative_y_m = 0.0
        self.target_halfspace_locked = False

    @property
    def stage(self) -> str:
        if self.terminal:
            return "terminal"
        if not self.started:
            return "ready"
        if self.predicted_class is None:
            return "classification"
        return "post_prediction_control"

    @property
    def guidance_unlocked(self) -> bool:
        return self.predicted_class is not None and self.active

    def start(self, observation_id: str) -> None:
        if self.started:
            raise ProtocolError("The one-shot episode has already started")
        self._require_observation_id_value(observation_id)
        self.started = True
        self.active = True
        self.current_observation_id = observation_id

    def validate_latest_observation(self, observation_id: object) -> None:
        if not self.active:
            raise ProtocolError("The episode is not active")
        if observation_id != self.current_observation_id:
            raise ProtocolError(
                f"Command must cite latest observation_id {self.current_observation_id!r}; "
                f"got {observation_id!r}"
            )

    def prepare_probe(self, observation_id: object, delta_gripper: float) -> None:
        self.validate_latest_observation(observation_id)
        if self.predicted_class is not None:
            raise ProtocolError("probe is available only before submit_prediction")
        if self.probe_count >= self.MAX_PROBES:
            raise ProtocolError("The pre-prediction probe budget is exhausted")
        if not np.isfinite(delta_gripper):
            raise ProtocolError("delta_gripper must be finite")
        if delta_gripper == 0.0 or abs(delta_gripper) > self.MAX_PROBE_GRIPPER_DELTA:
            raise ProtocolError(
                "Probe delta_gripper must be nonzero and no greater than "
                f"{self.MAX_PROBE_GRIPPER_DELTA}"
            )

    def complete_probe(self, next_observation_id: str) -> None:
        self._require_new_observation(next_observation_id)
        self.probe_count += 1
        self.current_observation_id = next_observation_id

    def submit_prediction(
        self,
        *,
        observation_id: object,
        predicted_class: object,
        target_pad: object,
    ) -> dict[str, Any]:
        self.validate_latest_observation(observation_id)
        if self.predicted_class is not None:
            raise ProtocolError("submit_prediction is irreversible and may be called exactly once")

        normalized_class = str(predicted_class).strip().lower()
        normalized_target = str(target_pad).strip().lower()
        if normalized_class not in self.CLASS_TO_TARGET:
            raise ProtocolError("predicted_class must be exactly 'rough' or 'plain'")
        expected_from_prediction = self.CLASS_TO_TARGET[normalized_class]
        if normalized_target != expected_from_prediction:
            raise ProtocolError(
                f"Task semantics require predicted_class={normalized_class!r} to use "
                f"target_pad={expected_from_prediction!r}"
            )

        self.predicted_class = normalized_class
        self.committed_target = normalized_target
        return {
            "predicted_class": normalized_class,
            "committed_target": normalized_target,
            "irreversible": True,
            "guidance_unlocked": self.guidance_unlocked,
        }

    def prepare_delta(
        self,
        *,
        observation_id: object,
        delta_position: object,
        delta_rpy: object,
        delta_gripper: object,
    ) -> tuple[np.ndarray, np.ndarray, float, PreparedDelta]:
        self._validate_post_prediction_action(observation_id)
        dp = self._finite_vector(delta_position, "delta_position", 3)
        dr = self._finite_vector(delta_rpy, "delta_rpy", 3)
        try:
            dg = float(delta_gripper)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("delta_gripper must be a finite number") from exc
        if not np.isfinite(dg):
            raise ProtocolError("delta_gripper must be a finite number")

        if np.max(np.abs(dp)) > self.MAX_TRANSLATION_COMPONENT_M:
            raise ProtocolError("Translation exceeds the per-component action bound")
        if np.linalg.norm(dp) > self.MAX_TRANSLATION_NORM_M:
            raise ProtocolError("Translation exceeds the per-action norm bound")
        if np.max(np.abs(dr)) > self.MAX_ROTATION_COMPONENT_RAD:
            raise ProtocolError("Rotation exceeds the per-component action bound")
        if abs(dg) > self.MAX_ACTION_GRIPPER_DELTA:
            raise ProtocolError("Gripper delta exceeds the per-action bound")

        sign = self.TARGET_Y_SIGN[self.committed_target]
        proposed_y = self.cumulative_y_m + float(dp[1])
        targetward_y = sign * proposed_y
        if targetward_y < -1e-9:
            raise ProtocolError("Committed target cannot change: action enters the opposite pad half-space")
        if (
            self.target_halfspace_locked
            and targetward_y < self.MIN_LOCKED_TARGETWARD_Y_M - 1e-9
        ):
            raise ProtocolError("Committed target cannot change: action leaves the locked target region")
        prepared = PreparedDelta(
            cumulative_y_m=proposed_y,
            target_halfspace_locked=(
                self.target_halfspace_locked
                or targetward_y >= self.MIN_LOCKED_TARGETWARD_Y_M
            ),
        )
        return dp, dr, dg, prepared

    def complete_delta(self, prepared: PreparedDelta, next_observation_id: str) -> None:
        self._require_new_observation(next_observation_id)
        self.action_count += 1
        self.cumulative_y_m = prepared.cumulative_y_m
        self.target_halfspace_locked = prepared.target_halfspace_locked
        self.current_observation_id = next_observation_id

    def prepare_wait(self, observation_id: object, physics_steps: object) -> int:
        self._validate_post_prediction_action(observation_id)
        try:
            steps = int(physics_steps)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("steps must be an integer") from exc
        if not 1 <= steps <= self.MAX_WAIT_PHYSICS_STEPS:
            raise ProtocolError(f"steps must be in [1, {self.MAX_WAIT_PHYSICS_STEPS}]")
        return steps

    def complete_wait(self, next_observation_id: str) -> None:
        self._require_new_observation(next_observation_id)
        self.action_count += 1
        self.current_observation_id = next_observation_id

    def public_action_feedback(
        self, *, execution_succeeded: bool, internal_task_success: bool
    ) -> dict[str, bool]:
        """Filter private checker output through the selected capability profile."""

        if not self.guidance_unlocked:
            raise ProtocolError("Feedback remains locked until submit_prediction")
        feedback = {"execution_succeeded": bool(execution_succeeded)}
        if self.profile.expose_task_success_after_prediction:
            feedback["task_success"] = bool(internal_task_success)
        return feedback

    def should_auto_stop(self, internal_task_success: bool) -> bool:
        return bool(
            self.guidance_unlocked
            and self.profile.expose_task_success_after_prediction
            and internal_task_success
        )

    def finish(self, observation_id: object) -> None:
        self._validate_post_prediction(observation_id)
        self.active = False
        self.terminal = True

    def force_terminal(self) -> None:
        if not self.started:
            raise ProtocolError("Cannot terminate an episode before start")
        self.active = False
        self.terminal = True

    def public_status(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "level": self.profile.level,
            "started": self.started,
            "active": self.active,
            "terminal": self.terminal,
            "stage": self.stage,
            "guidance_unlocked": self.guidance_unlocked,
            "probe_count": self.probe_count,
            "post_prediction_action_count": self.action_count,
            "predicted_class": self.predicted_class,
            "committed_target": self.committed_target,
            "latest_observation_id": self.current_observation_id,
        }

    def contract_manifest(self) -> dict[str, Any]:
        return {
            "protocol_version": "univtac.agent_env.grasp_classify.v0",
            "profile": self.profile.to_manifest(),
            "task_semantics": {"rough": "orange", "plain": "green"},
            "stages": ["classification", "post_prediction_control", "terminal"],
            "submit_prediction": {
                "required_before_translation": True,
                "irreversible": True,
                "target_switching_afterwards": False,
                "unlocks_guidance_channels": True,
                "future_extension_point": "post-prediction early_failure guidance",
            },
            "action_budget": {
                "optional_gripper_only_probes": self.MAX_PROBES,
                "post_prediction_actions": self.MAX_POST_PREDICTION_ACTIONS,
            },
            "action_bounds": {
                "translation_frame": "world",
                "max_abs_translation_component_m": self.MAX_TRANSLATION_COMPONENT_M,
                "max_translation_norm_m": self.MAX_TRANSLATION_NORM_M,
                "max_abs_rpy_component_rad": self.MAX_ROTATION_COMPONENT_RAD,
                "max_abs_gripper_delta": self.MAX_ACTION_GRIPPER_DELTA,
                "max_wait_physics_steps": self.MAX_WAIT_PHYSICS_STEPS,
            },
            "field_absence_is_part_of_contract": {
                "level1_tactile": "absent",
                "level1_task_success": "absent",
                "level2_task_success": "absent",
                "level3_task_success_before_submit_prediction": "absent",
            },
        }

    def _validate_post_prediction(self, observation_id: object) -> None:
        self.validate_latest_observation(observation_id)
        if self.predicted_class is None or self.committed_target is None:
            raise ProtocolError("submit_prediction is required before post-prediction control")

    def _validate_post_prediction_action(self, observation_id: object) -> None:
        self._validate_post_prediction(observation_id)
        if self.action_count >= self.MAX_POST_PREDICTION_ACTIONS:
            raise ProtocolError("The post-prediction action budget is exhausted")

    def _require_new_observation(self, observation_id: str) -> None:
        self._require_observation_id_value(observation_id)
        if observation_id == self.current_observation_id:
            raise ProtocolError("A world-changing operation must produce a fresh observation_id")

    @staticmethod
    def _require_observation_id_value(observation_id: object) -> None:
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ProtocolError("observation_id must be a non-empty string")

    @staticmethod
    def _finite_vector(value: object, name: str, size: int) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{name} must contain exactly {size} finite values") from exc
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ProtocolError(f"{name} must contain exactly {size} finite values")
        return vector

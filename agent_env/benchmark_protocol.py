"""Task-agnostic lifecycle and bounded EEF action protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .benchmark_profiles import ObservationProfile


class BenchmarkProtocolError(ValueError):
    pass


@dataclass
class BenchmarkEpisodeProtocol:
    profile: ObservationProfile

    MAX_STEPS = 50
    MAX_TRANSLATION_COMPONENT_M = 0.04
    MAX_TRANSLATION_NORM_M = 0.06
    MAX_ROTATION_COMPONENT_RAD = 0.35
    MAX_GRIPPER_DELTA_M = 0.005

    def __post_init__(self) -> None:
        self.started = False
        self.active = False
        self.terminal = False
        self.step_count = 0
        self.current_observation_id: str | None = None

    @property
    def stage(self) -> str:
        if self.terminal:
            return "terminal"
        if self.active:
            return "active"
        return "ready"

    def start(self, observation_id: str) -> None:
        if self.started:
            raise BenchmarkProtocolError("start_episode may be called exactly once")
        self._require_observation_id(observation_id)
        self.started = True
        self.active = True
        self.current_observation_id = observation_id

    def prepare_step(
        self,
        *,
        observation_id: object,
        delta_position: object,
        delta_rpy: object,
        delta_gripper: object,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self._validate_active_observation(observation_id)
        if self.step_count >= self.MAX_STEPS:
            raise BenchmarkProtocolError("The step_eef action budget is exhausted")
        dp = self._finite_vector(delta_position, "delta_position", 3)
        dr = self._finite_vector(delta_rpy, "delta_rpy", 3)
        try:
            dg = float(delta_gripper)
        except (TypeError, ValueError) as exc:
            raise BenchmarkProtocolError("delta_gripper must be a finite number") from exc
        if not np.isfinite(dg):
            raise BenchmarkProtocolError("delta_gripper must be a finite number")
        if np.max(np.abs(dp)) > self.MAX_TRANSLATION_COMPONENT_M:
            raise BenchmarkProtocolError("Translation exceeds the per-component bound")
        if np.linalg.norm(dp) > self.MAX_TRANSLATION_NORM_M:
            raise BenchmarkProtocolError("Translation exceeds the per-step norm bound")
        if np.max(np.abs(dr)) > self.MAX_ROTATION_COMPONENT_RAD:
            raise BenchmarkProtocolError("Rotation exceeds the per-component bound")
        if abs(dg) > self.MAX_GRIPPER_DELTA_M:
            raise BenchmarkProtocolError("Gripper delta exceeds the per-step bound")
        return dp, dr, dg

    def complete_step(self, observation_id: str) -> None:
        self._require_observation_id(observation_id)
        if observation_id == self.current_observation_id:
            raise BenchmarkProtocolError("step_eef must produce a fresh observation_id")
        self.step_count += 1
        self.current_observation_id = observation_id

    def finish(self, observation_id: object) -> None:
        self._validate_active_observation(observation_id)
        self.active = False
        self.terminal = True

    def public_status(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "profile_index": self.profile.index,
            "started": self.started,
            "active": self.active,
            "terminal": self.terminal,
            "stage": self.stage,
            "step_count": self.step_count,
            "latest_observation_id": self.current_observation_id,
        }

    def contract_manifest(self) -> dict[str, Any]:
        return {
            "protocol_version": "univtac.embodied_agent_env.v1",
            "profile": self.profile.to_manifest(),
            "tools": ["start_episode", "step_eef", "finish_episode"],
            "action_budget": {"step_eef": self.MAX_STEPS},
            "action_bounds": {
                "translation_frame": "world",
                "rotation_frame": "world",
                "rotation_parameterization": "roll_pitch_yaw_delta_radians",
                "max_abs_translation_component_m": self.MAX_TRANSLATION_COMPONENT_M,
                "max_translation_norm_m": self.MAX_TRANSLATION_NORM_M,
                "max_abs_rotation_component_rad": self.MAX_ROTATION_COMPONENT_RAD,
                "max_abs_gripper_delta_m": self.MAX_GRIPPER_DELTA_M,
                "zero_delta_allowed": True,
                "zero_delta_behavior": "wait_20_physics_steps_without_planning",
                "control_routing": {
                    "arm_only": "move",
                    "gripper_only": "gripper",
                    "arm_and_gripper": "all",
                    "all_zero": "no_op_wait",
                },
            },
            "task_success_visibility": "terminal finish_episode response only",
        }

    def _validate_active_observation(self, observation_id: object) -> None:
        if not self.active or self.terminal:
            raise BenchmarkProtocolError("Episode is not active")
        if observation_id != self.current_observation_id:
            raise BenchmarkProtocolError(
                f"observation_id must be the latest id {self.current_observation_id!r}"
            )

    @staticmethod
    def _require_observation_id(value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkProtocolError("observation_id must be a non-empty string")

    @staticmethod
    def _finite_vector(value: object, name: str, size: int) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise BenchmarkProtocolError(
                f"{name} must contain exactly {size} finite values"
            ) from exc
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise BenchmarkProtocolError(
                f"{name} must contain exactly {size} finite values"
            )
        return vector

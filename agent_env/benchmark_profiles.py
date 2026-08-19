"""Observation profiles and orthogonal oracle-annotation capabilities.

The six profiles are the benchmark's primary observation axis.  Anonymous
bounding boxes and masks are deliberately independent diagnostic switches and
never widen the simulator's raw instance-metadata surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ROBOT_STATE_FIELDS = (
    "joint_position_9d",
    "joint_velocity_9d",
    "gripper_width_m",
    "end_effector_pose_robot_base_wxyz_7d",
)


@dataclass(frozen=True)
class ObservationProfile:
    """One immutable, main-leaderboard observation capability profile."""

    index: int
    name: str
    expose_wrist_rgb: bool
    expose_tactile: bool
    expose_metric_depth: bool
    expose_camera_intrinsics: bool
    expose_camera_extrinsics: bool

    @property
    def level(self) -> int:
        """Compatibility alias for existing Codex recording infrastructure."""

        return self.index

    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("head", "wrist") if self.expose_wrist_rgb else ("head",)

    @property
    def public_modalities(self) -> tuple[str, ...]:
        modalities: list[str] = ["head_rgb"]
        if self.expose_wrist_rgb:
            modalities.append("wrist_rgb")
        if self.expose_tactile:
            modalities.extend(("left_tactile_rgb", "right_tactile_rgb"))
        if self.expose_metric_depth:
            modalities.extend(("head_depth", "wrist_depth"))
        return tuple(modalities)

    @property
    def public_robot_state(self) -> tuple[str, ...]:
        return ROBOT_STATE_FIELDS

    def to_manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "camera_names": list(self.camera_names),
            "public_modalities": list(self.public_modalities),
            "public_robot_state": list(self.public_robot_state),
            "metric_depth": self.expose_metric_depth,
            "camera_intrinsics": self.expose_camera_intrinsics,
            "camera_extrinsics": self.expose_camera_extrinsics,
            "task_success_during_episode": False,
        }


@dataclass(frozen=True)
class AnnotationCapabilities:
    """Independent, composable, anonymous oracle-annotation switches."""

    provide_bbox: bool = False
    provide_mask: bool = False

    @property
    def enabled_features(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.provide_bbox:
            result.append("bbox")
        if self.provide_mask:
            result.append("mask")
        return tuple(result)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "bbox": self.provide_bbox,
            "mask": self.provide_mask,
            "schedule": "initial_observation_only",
            "enabled_features": list(self.enabled_features),
            "public_roles": ["manipulated_object", "goal_fixture"],
            "raw_instance_ids": False,
            "raw_labels": False,
            "usd_prim_paths": False,
        }


_PROFILES = {
    1: ObservationProfile(
        index=1,
        name="head_state",
        expose_wrist_rgb=False,
        expose_tactile=False,
        expose_metric_depth=False,
        expose_camera_intrinsics=False,
        expose_camera_extrinsics=False,
    ),
    2: ObservationProfile(
        index=2,
        name="head_wrist_state",
        expose_wrist_rgb=True,
        expose_tactile=False,
        expose_metric_depth=False,
        expose_camera_intrinsics=False,
        expose_camera_extrinsics=False,
    ),
    3: ObservationProfile(
        index=3,
        name="head_tactile_state",
        expose_wrist_rgb=False,
        expose_tactile=True,
        expose_metric_depth=False,
        expose_camera_intrinsics=False,
        expose_camera_extrinsics=False,
    ),
    4: ObservationProfile(
        index=4,
        name="head_wrist_tactile_state",
        expose_wrist_rgb=True,
        expose_tactile=True,
        expose_metric_depth=False,
        expose_camera_intrinsics=False,
        expose_camera_extrinsics=False,
    ),
    5: ObservationProfile(
        index=5,
        name="head_wrist_tactile_depth_intrinsics_state",
        expose_wrist_rgb=True,
        expose_tactile=True,
        expose_metric_depth=True,
        expose_camera_intrinsics=True,
        expose_camera_extrinsics=False,
    ),
    6: ObservationProfile(
        index=6,
        name="head_wrist_tactile_depth_intrinsics_extrinsics_state",
        expose_wrist_rgb=True,
        expose_tactile=True,
        expose_metric_depth=True,
        expose_camera_intrinsics=True,
        expose_camera_extrinsics=True,
    ),
}


def get_observation_profile(value: int | str) -> ObservationProfile:
    """Resolve an index or canonical name to a profile."""

    if isinstance(value, str) and not value.isdigit():
        for profile in _PROFILES.values():
            if profile.name == value:
                return profile
        raise ValueError(f"Unknown observation profile: {value!r}")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown observation profile: {value!r}") from exc
    try:
        return _PROFILES[index]
    except KeyError as exc:
        raise ValueError("Observation profile must be one of 1, 2, 3, 4, 5, 6") from exc


def list_observation_profiles() -> tuple[ObservationProfile, ...]:
    return tuple(_PROFILES[index] for index in sorted(_PROFILES))

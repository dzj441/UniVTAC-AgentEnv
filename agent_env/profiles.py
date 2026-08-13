"""Capability profiles for the first ``grasp_classify`` AgentEnv experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentEnvProfile:
    """An immutable agent-visible capability set.

    Every level shares the same action space, budget, task initialization, and
    execution-status feedback.  Only the two experimental variables below are
    changed: tactile visibility and post-prediction task-success guidance.
    """

    level: int
    name: str
    description: str
    expose_tactile: bool
    expose_task_success_after_prediction: bool

    @property
    def public_modalities(self) -> tuple[str, ...]:
        modalities = ["head_rgb", "wrist_rgb"]
        if self.expose_tactile:
            modalities.extend(("left_tactile_marker", "right_tactile_marker"))
        return tuple(modalities)

    @property
    def public_robot_state(self) -> tuple[str, ...]:
        return (
            "joint_position_8d",
            "gripper_qpos",
            "end_effector_pose_robot_base_7d",
        )

    @property
    def public_feedback(self) -> tuple[str, ...]:
        feedback = ["execution_succeeded"]
        if self.expose_task_success_after_prediction:
            feedback.append("task_success_after_prediction")
        return tuple(feedback)

    def to_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "public_modalities": list(self.public_modalities),
                "public_robot_state": list(self.public_robot_state),
                "public_feedback": list(self.public_feedback),
            }
        )
        return payload


_PROFILES = {
    1: AgentEnvProfile(
        level=1,
        name="vision_only_control",
        description="Head/wrist RGB and basic robot state; no tactile or task-success feedback.",
        expose_tactile=False,
        expose_task_success_after_prediction=False,
    ),
    2: AgentEnvProfile(
        level=2,
        name="visuotactile_control",
        description="Level 1 plus left/right tactile marker RGB.",
        expose_tactile=True,
        expose_task_success_after_prediction=False,
    ),
    3: AgentEnvProfile(
        level=3,
        name="success_guided_visuotactile_control",
        description=(
            "Level 2 plus a task-success boolean after the irreversible prediction; "
            "the selected target cannot then change."
        ),
        expose_tactile=True,
        expose_task_success_after_prediction=True,
    ),
}


def get_profile(level: int | str) -> AgentEnvProfile:
    """Resolve a level number or stable profile name."""

    aliases: dict[str, int] = {
        "1": 1,
        "level1": 1,
        "l1": 1,
        "vision_only_control": 1,
        "2": 2,
        "level2": 2,
        "l2": 2,
        "visuotactile_control": 2,
        "3": 3,
        "level3": 3,
        "l3": 3,
        "success_guided_visuotactile_control": 3,
    }
    key = str(level).strip().lower()
    try:
        return _PROFILES[aliases[key]]
    except KeyError as exc:
        choices = ", ".join(profile.name for profile in _PROFILES.values())
        raise ValueError(f"Unknown AgentEnv level {level!r}; choose 1, 2, 3, or {choices}") from exc


def list_profiles() -> tuple[AgentEnvProfile, ...]:
    return tuple(_PROFILES[level] for level in sorted(_PROFILES))

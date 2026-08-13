"""Whitelist-only serialization for agent-visible observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .profiles import AgentEnvProfile


def serialize_observation(
    *,
    profile: AgentEnvProfile,
    observation_id: str,
    stage: str,
    available_modalities: Mapping[str, Mapping[str, Any]],
    available_robot_state: Mapping[str, Any],
    probe_count: int,
    post_prediction_action_count: int,
    tactile_health: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct an observation exclusively from profile-declared fields.

    The caller may hold privileged or higher-level data in ``available_*``;
    iteration over those mappings is intentionally forbidden.  New simulator
    fields therefore remain private until explicitly added to a profile.
    """

    modalities: dict[str, Mapping[str, Any]] = {}
    for name in profile.public_modalities:
        if name not in available_modalities:
            raise KeyError(f"Profile requires missing modality {name!r}")
        modalities[name] = dict(available_modalities[name])

    robot_state: dict[str, Any] = {}
    for name in profile.public_robot_state:
        if name not in available_robot_state:
            raise KeyError(f"Profile requires missing robot-state field {name!r}")
        robot_state[name] = available_robot_state[name]

    payload: dict[str, Any] = {
        "observation_id": observation_id,
        "level": profile.level,
        "profile": profile.name,
        "stage": stage,
        "probe_count": int(probe_count),
        "post_prediction_action_count": int(post_prediction_action_count),
        "modalities": modalities,
        "robot_state": robot_state,
    }
    if profile.expose_tactile and tactile_health is not None:
        payload["tactile_health"] = dict(tactile_health)
    if artifacts:
        payload["artifacts"] = dict(artifacts)
    return payload

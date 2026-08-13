from agent_env.profiles import get_profile
from agent_env.serialization import serialize_observation


ALL_MODALITIES = {
    "head_rgb": {"path": "/public/head.png"},
    "wrist_rgb": {"path": "/public/wrist.png"},
    "left_tactile_marker": {"path": "/hidden/left.png"},
    "right_tactile_marker": {"path": "/hidden/right.png"},
    "depth": {"path": "/hidden/depth.exr"},
    "actor_pose": {"value": [1, 2, 3]},
}
ALL_ROBOT_STATE = {
    "joint_position_8d": [0] * 8,
    "gripper_qpos": 0.01,
    "end_effector_pose_robot_base_7d": [0] * 7,
    "contact_force": 100,
}


def observation(level: int) -> dict:
    return serialize_observation(
        profile=get_profile(level),
        observation_id="obs_000",
        stage="classification",
        available_modalities=ALL_MODALITIES,
        available_robot_state=ALL_ROBOT_STATE,
        probe_count=0,
        post_prediction_action_count=0,
        tactile_health={"left": {"healthy": True}, "right": {"healthy": True}},
        artifacts={"composite": {"path": "/public/composite.png"}},
    )


def test_level1_whitelist_drops_tactile_and_every_privileged_field() -> None:
    payload = observation(1)
    assert set(payload["modalities"]) == {"head_rgb", "wrist_rgb"}
    assert set(payload["robot_state"]) == {
        "joint_position_8d",
        "gripper_qpos",
        "end_effector_pose_robot_base_7d",
    }
    assert "tactile_health" not in payload
    serialized = repr(payload)
    assert "depth" not in serialized
    assert "actor_pose" not in serialized
    assert "contact_force" not in serialized
    assert "task_success" not in serialized


def test_level2_and_level3_expose_the_same_observation_fields() -> None:
    level2 = observation(2)
    level3 = observation(3)
    assert level2["modalities"] == level3["modalities"]
    assert level2["robot_state"] == level3["robot_state"]
    assert level2["tactile_health"] == level3["tactile_health"]
    assert "task_success" not in repr(level2)
    assert "task_success" not in repr(level3)

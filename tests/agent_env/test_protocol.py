import pytest

from agent_env.profiles import get_profile
from agent_env.protocol import EpisodeProtocol, ProtocolError


def started_protocol(level: int = 3) -> EpisodeProtocol:
    protocol = EpisodeProtocol(get_profile(level))
    protocol.start("obs_000")
    return protocol


def submit_rough(protocol: EpisodeProtocol) -> None:
    protocol.submit_prediction(
        observation_id=protocol.current_observation_id,
        predicted_class="rough",
        target_pad="orange",
    )


def test_submit_prediction_is_required_irreversible_and_semantically_consistent() -> None:
    protocol = started_protocol()

    with pytest.raises(ProtocolError, match="submit_prediction is required"):
        protocol.prepare_delta(
            observation_id="obs_000",
            delta_position=[0, 0, 0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )
    with pytest.raises(ProtocolError, match="Task semantics require"):
        protocol.submit_prediction(
            observation_id="obs_000", predicted_class="rough", target_pad="green"
        )

    result = protocol.submit_prediction(
        observation_id="obs_000", predicted_class="rough", target_pad="orange"
    )
    assert result == {
        "predicted_class": "rough",
        "committed_target": "orange",
        "irreversible": True,
        "guidance_unlocked": True,
    }
    with pytest.raises(ProtocolError, match="irreversible"):
        protocol.submit_prediction(
            observation_id="obs_000", predicted_class="plain", target_pad="green"
        )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (1, {"execution_succeeded": True}),
        (2, {"execution_succeeded": True}),
        (3, {"execution_succeeded": True, "task_success": False}),
    ],
)
def test_success_feedback_is_filtered_by_level(level: int, expected: dict[str, bool]) -> None:
    protocol = started_protocol(level)
    with pytest.raises(ProtocolError, match="locked until submit_prediction"):
        protocol.public_action_feedback(
            execution_succeeded=True, internal_task_success=True
        )

    submit_rough(protocol)
    feedback = protocol.public_action_feedback(
        execution_succeeded=True, internal_task_success=False
    )
    assert feedback == expected


def test_level1_and_level2_never_serialize_success_even_when_true() -> None:
    for level in (1, 2):
        protocol = started_protocol(level)
        submit_rough(protocol)
        feedback = protocol.public_action_feedback(
            execution_succeeded=False, internal_task_success=True
        )
        assert feedback == {"execution_succeeded": False}
        assert "task_success" not in feedback
        assert not protocol.should_auto_stop(True)


def test_level3_success_auto_stop_is_enabled_only_after_prediction() -> None:
    protocol = started_protocol(3)
    assert not protocol.should_auto_stop(True)
    submit_rough(protocol)
    assert protocol.should_auto_stop(True)
    assert not protocol.should_auto_stop(False)


def test_committed_target_halfspace_cannot_change() -> None:
    protocol = started_protocol()
    submit_rough(protocol)  # orange is negative world y

    _, _, _, prepared = protocol.prepare_delta(
        observation_id="obs_000",
        delta_position=[0.01, -0.03, 0.0],
        delta_rpy=[0, 0, 0],
        delta_gripper=0,
    )
    protocol.complete_delta(prepared, "obs_001")
    assert protocol.target_halfspace_locked

    with pytest.raises(ProtocolError, match="leaves the locked target region"):
        protocol.prepare_delta(
            observation_id="obs_001",
            delta_position=[0.0, 0.02, 0.0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )
    with pytest.raises(ProtocolError, match="opposite pad half-space"):
        protocol.prepare_delta(
            observation_id="obs_001",
            delta_position=[0.0, 0.04, 0.0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )


def test_world_change_requires_fresh_observation_and_rejects_stale_commands() -> None:
    protocol = started_protocol()
    protocol.prepare_probe("obs_000", -0.001)
    with pytest.raises(ProtocolError, match="fresh observation_id"):
        protocol.complete_probe("obs_000")
    protocol.complete_probe("obs_001")
    with pytest.raises(ProtocolError, match="latest observation_id"):
        protocol.prepare_probe("obs_000", 0.001)


def test_action_bounds_and_budget_are_enforced() -> None:
    protocol = started_protocol()
    submit_rough(protocol)
    with pytest.raises(ProtocolError, match="per-component"):
        protocol.prepare_delta(
            observation_id="obs_000",
            delta_position=[0.041, 0, 0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )

    for index in range(protocol.MAX_POST_PREDICTION_ACTIONS):
        observation_id = f"obs_{index:03d}"
        next_observation_id = f"obs_{index + 1:03d}"
        _, _, _, prepared = protocol.prepare_delta(
            observation_id=observation_id,
            delta_position=[0, 0, 0],
            delta_rpy=[0, 0, 0],
            delta_gripper=0,
        )
        protocol.complete_delta(prepared, next_observation_id)

    with pytest.raises(ProtocolError, match="budget is exhausted"):
        protocol.prepare_wait(protocol.current_observation_id, 1)


def test_manifest_names_future_early_failure_extension_boundary() -> None:
    manifest = started_protocol(3).contract_manifest()
    assert manifest["submit_prediction"]["future_extension_point"] == (
        "post-prediction early_failure guidance"
    )
    assert manifest["field_absence_is_part_of_contract"] == {
        "level1_tactile": "absent",
        "level1_task_success": "absent",
        "level2_task_success": "absent",
        "level3_task_success_before_submit_prediction": "absent",
    }

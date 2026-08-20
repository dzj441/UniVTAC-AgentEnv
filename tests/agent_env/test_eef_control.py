from __future__ import annotations

import json
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from agent_env.benchmark_profiles import get_observation_profile
from agent_env.benchmark_protocol import BenchmarkEpisodeProtocol
from agent_env.eef_control import (
    classify_delta_eef_action,
    motion_gen_result_diagnostics,
)


@pytest.mark.parametrize(
    ("delta_position", "delta_rpy", "delta_gripper", "expected"),
    [
        ([0, 0, 0], [0, 0, 0], 0, ("no_op", False, False)),
        ([0.01, 0, 0], [0, 0, 0], 0, ("move", True, False)),
        ([0, 0, 0], [0, -0.1, 0], 0, ("move", True, False)),
        ([0, 0, 0], [0, 0, 0], 0.001, ("gripper", False, True)),
        ([0, 0.01, 0], [0, 0, 0], -0.001, ("all", True, True)),
        ([1e-12, 0, 0], [0, 0, 0], 0, ("move", True, False)),
    ],
)
def test_delta_eef_route_uses_exact_nonzero_components(
    delta_position: list[float],
    delta_rpy: list[float],
    delta_gripper: float,
    expected: tuple[str, bool, bool],
) -> None:
    route = classify_delta_eef_action(
        delta_position,
        delta_rpy,
        delta_gripper,
    )

    assert (route.name, route.arm_changed, route.gripper_changed) == expected


@pytest.mark.parametrize(
    ("delta_position", "delta_rpy", "delta_gripper"),
    [
        ([0, 0], [0, 0, 0], 0),
        ([0, 0, 0], [0, np.nan, 0], 0),
        ([0, 0, 0], [0, 0, 0], np.inf),
    ],
)
def test_delta_eef_route_rejects_malformed_or_nonfinite_values(
    delta_position: list[float],
    delta_rpy: list[float],
    delta_gripper: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        classify_delta_eef_action(delta_position, delta_rpy, delta_gripper)


class FakeMotionGenStatus(Enum):
    INVALID_START_STATE_WORLD_COLLISION = "Start state is colliding with world"


def test_motion_gen_failure_diagnostics_are_complete_and_json_safe() -> None:
    result = SimpleNamespace(
        success=np.asarray([False]),
        status=FakeMotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
        valid_query=False,
        attempts=10,
        trajopt_attempts=4,
        used_graph=True,
        goalset_index=np.asarray([0]),
        path_buffer_last_tstep=[12],
        solve_time=np.float32(0.42),
        ik_time=0.1,
        graph_time=0.2,
        trajopt_time=0.11,
        finetune_time=0.01,
        total_time=0.43,
        position_error=np.asarray([0.003]),
        rotation_error=np.asarray([0.02]),
        cspace_error=np.asarray([0.04]),
        debug_info={"intentionally_not_serialized": object()},
    )

    diagnostics = motion_gen_result_diagnostics(result)

    assert diagnostics["success"] is False
    assert diagnostics["status"] == {
        "type": "FakeMotionGenStatus",
        "name": "INVALID_START_STATE_WORLD_COLLISION",
        "value": "Start state is colliding with world",
        "text": "FakeMotionGenStatus.INVALID_START_STATE_WORLD_COLLISION",
    }
    assert diagnostics["valid_query"] is False
    assert diagnostics["attempts"] == 10
    assert diagnostics["trajopt_attempts"] == 4
    assert diagnostics["used_graph"] is True
    assert diagnostics["goalset_index"] == 0
    assert diagnostics["path_buffer_last_tstep"] == [12]
    assert diagnostics["debug_info_present"] is True
    assert diagnostics["timing_seconds"]["solve"] == pytest.approx(0.42)
    assert diagnostics["terminal_error"] == pytest.approx(
        {"position": 0.003, "rotation": 0.02, "cspace": 0.04}
    )
    json.dumps(diagnostics)


def test_contract_declares_planner_free_routing_and_zero_wait() -> None:
    manifest = BenchmarkEpisodeProtocol(
        get_observation_profile(6)
    ).contract_manifest()
    action_bounds = manifest["action_bounds"]

    assert action_bounds["zero_delta_behavior"] == (
        "wait_20_physics_steps_without_planning"
    )
    assert action_bounds["control_routing"] == {
        "arm_only": "move",
        "gripper_only": "gripper",
        "arm_and_gripper": "all",
        "all_zero": "no_op_wait",
    }

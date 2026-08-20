from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import transforms3d as t3d


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_transforms_module() -> ModuleType:
    module_path = REPO_ROOT / "envs" / "utils" / "transforms.py"
    spec = importlib.util.spec_from_file_location(
        "univtac_pose_rotation_semantics",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFORMS = _load_transforms_module()
Pose = TRANSFORMS.Pose


def _assert_same_rotation(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        t3d.quaternions.quat2mat(actual),
        t3d.quaternions.quat2mat(expected),
        atol=1e-12,
    )


def test_world_orientation_delta_preserves_position_and_pre_multiplies() -> None:
    pose = Pose(
        p=[0.48, 0.02, 0.30],
        q=t3d.euler.euler2quat(0.4, -0.3, 0.2),
    )
    delta_euler = [0.1, 0.2, -0.35]
    delta_q = t3d.euler.euler2quat(*delta_euler)

    result = pose.add_orientation_delta(delta_euler, frame="world")

    np.testing.assert_array_equal(result.p, pose.p)
    _assert_same_rotation(
        result.q,
        t3d.quaternions.qmult(delta_q, pose.q),
    )


def test_local_orientation_delta_preserves_position_and_post_multiplies() -> None:
    pose = Pose(
        p=[0.48, 0.02, 0.30],
        q=t3d.euler.euler2quat(0.4, -0.3, 0.2),
    )
    delta_euler = [0.1, 0.2, -0.35]
    delta_q = t3d.euler.euler2quat(*delta_euler)

    world_result = pose.add_orientation_delta(delta_euler, frame="world")
    local_result = pose.add_orientation_delta(delta_euler, frame="local")

    np.testing.assert_array_equal(local_result.p, pose.p)
    _assert_same_rotation(
        local_result.q,
        t3d.quaternions.qmult(pose.q, delta_q),
    )
    assert not np.allclose(world_result.R, local_result.R)


def test_step_eef_composition_keeps_world_translation_independent() -> None:
    pose = Pose(
        p=[0.480707049369812, 0.0, 0.2993789315223694],
        q=t3d.euler.euler2quat(0.25, -0.15, 0.05),
    )
    delta_position = np.array([0.01, -0.02, 0.03])
    delta_rpy = [0.0, 0.0, 0.1]

    result = (
        pose.add_bias(delta_position, coord="world")
        .add_orientation_delta(delta_rpy, frame="world")
    )

    np.testing.assert_allclose(result.p, pose.p + delta_position, atol=1e-12)
    _assert_same_rotation(
        result.q,
        t3d.quaternions.qmult(
            t3d.euler.euler2quat(*delta_rpy),
            pose.q,
        ),
    )


def test_pure_world_yaw_does_not_orbit_position() -> None:
    pose = Pose(
        p=[0.480707049369812, 0.0, 0.2993789315223694],
        q=t3d.euler.euler2quat(0.0, 0.0, 0.05),
    )

    result = pose.add_orientation_delta([0.0, 0.0, 0.1], frame="world")

    np.testing.assert_array_equal(result.p, pose.p)


def test_legacy_world_rotation_still_orbits_position() -> None:
    pose = Pose(
        p=[0.480707049369812, 0.0, 0.2993789315223694],
        q=t3d.euler.euler2quat(0.0, 0.0, 0.05),
    )
    delta_euler = [0.0, 0.0, 0.1]

    result = pose.add_rotation(delta_euler, coord="world")

    expected_position = t3d.euler.euler2mat(*delta_euler) @ pose.p
    np.testing.assert_allclose(result.p, expected_position, atol=1e-12)
    assert not np.allclose(result.p, pose.p)


def test_orientation_delta_rejects_unknown_frame() -> None:
    with pytest.raises(ValueError, match="orientation-delta frame"):
        Pose().add_orientation_delta([0.0, 0.0, 0.1], frame="camera")

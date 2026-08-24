from __future__ import annotations

from agent_env.benchmark_checker import PostGraspReferenceTracker


def test_post_grasp_reference_tracks_one_open_close_transport_cycle() -> None:
    tracker = PostGraspReferenceTracker(required=True)
    assert tracker.state == "armed"
    assert tracker.after_action(
        arm_changed=True,
        delta_gripper=0.0,
        execution_succeeded=True,
    ) is None
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=-0.005,
        execution_succeeded=True,
    ) == "captured_after_gripper_close"
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=-0.002,
        execution_succeeded=True,
    ) == "captured_after_gripper_close"
    assert tracker.after_action(
        arm_changed=True,
        delta_gripper=0.0,
        execution_succeeded=True,
    ) == "frozen_before_transport"
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=-0.001,
        execution_succeeded=True,
    ) is None
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=0.01,
        execution_succeeded=True,
    ) is None
    assert tracker.state == "armed"
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=-0.005,
        execution_succeeded=True,
    ) == "captured_after_gripper_close"


def test_post_grasp_reference_ignores_failed_and_unneeded_actions() -> None:
    tracker = PostGraspReferenceTracker(required=True)
    assert tracker.after_action(
        arm_changed=False,
        delta_gripper=-0.005,
        execution_succeeded=False,
    ) is None
    assert tracker.state == "armed"
    disabled = PostGraspReferenceTracker(required=False)
    assert disabled.after_action(
        arm_changed=False,
        delta_gripper=-0.005,
        execution_succeeded=True,
    ) is None
    assert disabled.state == "not_required"

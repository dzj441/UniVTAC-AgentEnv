"""Evaluator-private checker lifecycle helpers for ungrasped Agent episodes."""

from __future__ import annotations


class PostGraspReferenceTracker:
    """Capture a legacy in-hand reference once per open/close grasp cycle."""

    def __init__(self, *, required: bool) -> None:
        self.state = "armed" if required else "not_required"

    def after_action(
        self,
        *,
        arm_changed: bool,
        delta_gripper: float,
        execution_succeeded: bool,
    ) -> str | None:
        if self.state == "not_required" or not execution_succeeded:
            return None
        if delta_gripper > 0:
            self.state = "armed"
            return None
        if delta_gripper < 0 and self.state in {"armed", "closing"}:
            self.state = "closing"
            return "captured_after_gripper_close"
        if arm_changed and self.state == "closing":
            self.state = "frozen"
            return "frozen_before_transport"
        return None

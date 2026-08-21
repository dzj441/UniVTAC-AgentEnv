"""Task registry for the generic embodied AgentEnv benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


LEGACY_KEY_INITIAL_RELATIVE_YAW_RANGE_RAD = (-math.pi / 2, -math.pi / 4)


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    name: str
    module: str
    ungrasped_instruction: str
    pregrasped_instruction: str
    manipulated_prim_name: str
    goal_prim_name: str
    terminal_policy: str

    @property
    def instruction(self) -> str:
        """Default v1 instruction; episodes start without task-specific pre-move."""

        return self.ungrasped_instruction

    @staticmethod
    def start_condition(pre_move: bool) -> str:
        return "pregrasped" if pre_move else "ungrasped"

    def instruction_for(self, *, pre_move: bool) -> str:
        return self.pregrasped_instruction if pre_move else self.ungrasped_instruction

    def to_manifest(self, *, pre_move: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "instruction": self.instruction_for(pre_move=pre_move),
            "start_condition": self.start_condition(pre_move),
            "pre_move_enabled": pre_move,
            "initial_state": (
                "task pre_move completed; manipulated object already grasped"
                if pre_move
                else "task actors reset; robot at fixed default home state; object ungrasped"
            ),
            "annotation_roles": {
                "manipulated_object": "anonymous",
                "goal_fixture": "anonymous",
            },
            "terminal_policy": self.terminal_policy,
        }


_TASKS = {
    "pull_out_key": BenchmarkTaskSpec(
        name="pull_out_key",
        module="envs.pull_out_key",
        ungrasped_instruction="Grasp the key and pull it completely out of the slot.",
        pregrasped_instruction="Pull the already-grasped key completely out of the slot.",
        manipulated_prim_name="key",
        goal_prim_name="slot",
        terminal_policy="pull_out_key_v1",
    ),
    "put_bottle_in_shelf": BenchmarkTaskSpec(
        name="put_bottle_in_shelf",
        module="envs.put_bottle_in_shelf",
        ungrasped_instruction=(
            "Pick up the bottle from the table, place it upright inside the shelf, "
            "and release it."
        ),
        pregrasped_instruction=(
            "Place the already-grasped bottle upright inside the shelf and release it."
        ),
        # The task currently registers BottleLift.usd under the actor name
        # ``prism``.  That private implementation name is never serialized.
        manipulated_prim_name="prism",
        goal_prim_name="shelf",
        terminal_policy="released_stable_bottle_v1",
    ),
}


def get_benchmark_task(value: str) -> BenchmarkTaskSpec:
    try:
        return _TASKS[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_TASKS))
        raise ValueError(f"Unknown benchmark task {value!r}; choose one of: {choices}") from exc


def list_benchmark_tasks() -> tuple[BenchmarkTaskSpec, ...]:
    return tuple(_TASKS[name] for name in sorted(_TASKS))


def benchmark_task_parameters(
    task_name: str,
    *,
    key_initial_relative_yaw_rad: float | None = None,
) -> dict[str, Any]:
    """Validate and describe evaluator-selected task reset parameters."""

    get_benchmark_task(task_name)
    if key_initial_relative_yaw_rad is not None and task_name != "pull_out_key":
        raise ValueError(
            "--key-initial-relative-yaw-rad is valid only for pull_out_key"
        )
    if task_name != "pull_out_key":
        return {}
    if key_initial_relative_yaw_rad is None:
        return {
            "key_initial_relative_yaw": {
                "mode": "legacy_random",
                "range_rad": list(LEGACY_KEY_INITIAL_RELATIVE_YAW_RANGE_RAD),
            }
        }
    value = float(key_initial_relative_yaw_rad)
    if not math.isfinite(value):
        raise ValueError("key initial relative yaw must be finite")
    return {
        "key_initial_relative_yaw": {
            "mode": "fixed",
            "value_rad": value,
        }
    }

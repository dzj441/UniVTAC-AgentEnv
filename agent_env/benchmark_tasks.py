"""Task registry for the generic embodied AgentEnv benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .expert_tasks import BASE_TASK_SUCCESS, get_expert_task


LEGACY_KEY_INITIAL_RELATIVE_YAW_RANGE_RAD = (-math.pi / 2, -math.pi / 4)


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    name: str
    module: str
    ungrasped_instruction: str
    pregrasped_instruction: str
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

    def manipulated_actor(self, task: Any) -> Any:
        return get_expert_task(self.name).manipulated_actor(task)

    def annotation_prim_names(self, task: Any) -> dict[str, tuple[str, ...]]:
        return get_expert_task(self.name).annotation_prim_names(task)

    @property
    def requires_post_grasp_reference(self) -> bool:
        return get_expert_task(self.name).requires_post_grasp_reference

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
    "grasp_classify": BenchmarkTaskSpec(
        name="grasp_classify",
        module="envs.grasp_classify",
        ungrasped_instruction=(
            "Grasp the center prism and place it upright on the green pad."
        ),
        pregrasped_instruction=(
            "Place the already-grasped center prism upright on the green pad."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "insert_HDMI": BenchmarkTaskSpec(
        name="insert_HDMI",
        module="envs.insert_HDMI",
        ungrasped_instruction=(
            "Grasp the HDMI connector and insert it fully into the port."
        ),
        pregrasped_instruction=(
            "Insert the already-grasped HDMI connector fully into the port."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "insert_hole": BenchmarkTaskSpec(
        name="insert_hole",
        module="envs.insert_hole",
        ungrasped_instruction="Grasp the peg and insert it fully into the angled hole.",
        pregrasped_instruction=(
            "Insert the already-grasped peg fully into the angled hole."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "insert_tube": BenchmarkTaskSpec(
        name="insert_tube",
        module="envs.insert_tube",
        ungrasped_instruction="Grasp the tube and insert it fully into the fixture.",
        pregrasped_instruction=(
            "Insert the already-grasped tube fully into the fixture."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "lift_bottle": BenchmarkTaskSpec(
        name="lift_bottle",
        module="envs.lift_bottle",
        ungrasped_instruction=(
            "Grasp the bottle, rotate it upright beside the wall, and release it stably."
        ),
        pregrasped_instruction=(
            "Rotate the already-grasped bottle upright beside the wall and release it "
            "stably."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "lift_can": BenchmarkTaskSpec(
        name="lift_can",
        module="envs.lift_can",
        ungrasped_instruction=(
            "Grasp the horizontal can, rotate it upright on the table, and release it."
        ),
        pregrasped_instruction=(
            "Rotate the already-grasped can upright on the table and release it."
        ),
        terminal_policy=BASE_TASK_SUCCESS,
    ),
    "pull_out_key": BenchmarkTaskSpec(
        name="pull_out_key",
        module="envs.pull_out_key",
        ungrasped_instruction="Grasp the key and pull it completely out of the slot.",
        pregrasped_instruction="Pull the already-grasped key completely out of the slot.",
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
    if task_name == "grasp_classify":
        return {
            "target_pad": {
                "mode": "fixed_color",
                "color": "green",
                "material_classification_required": False,
            }
        }
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

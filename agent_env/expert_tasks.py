"""Private actor metadata shared by expert assets and online evaluation.

The online benchmark registry separately owns public instructions and terminal
policies.  This registry resolves dynamic task actors without serializing their
private names to the Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BASE_TASK_SUCCESS = "base_task_success_v1"
RELEASED_STABLE_MANIPULATED_OBJECT = "released_stable_manipulated_object_v1"


@dataclass(frozen=True)
class ExpertTaskSpec:
    name: str
    module: str
    manipulated_actor_attribute: str
    goal_actor_attribute: str | None
    goal_prim_name: str | None
    goal_candidate_actor_attributes: tuple[str, ...] = ()
    terminal_policy: str = BASE_TASK_SUCCESS
    requires_post_grasp_reference: bool = False

    @staticmethod
    def _actor(task: Any, attribute: str, role: str) -> Any:
        actor = getattr(task, attribute, None)
        if actor is None or getattr(actor, "cfg", None) is None:
            raise ValueError(
                f"Task {task.__class__.__module__} has no resolved {role} actor "
                f"attribute {attribute!r}"
            )
        private_name = getattr(actor.cfg, "name", None)
        if not isinstance(private_name, str) or not private_name:
            raise ValueError(f"Resolved {role} actor has no private prim name")
        return actor

    def manipulated_actor(self, task: Any) -> Any:
        return self._actor(task, self.manipulated_actor_attribute, "manipulated")

    def annotation_prim_names(self, task: Any) -> dict[str, tuple[str, ...]]:
        manipulated = self.manipulated_actor(task)
        if self.goal_candidate_actor_attributes:
            goal_names = tuple(
                self._actor(task, attribute, "goal candidate").cfg.name
                for attribute in self.goal_candidate_actor_attributes
            )
        elif self.goal_actor_attribute is not None:
            goal_names = (
                self._actor(task, self.goal_actor_attribute, "goal").cfg.name,
            )
        else:
            goal_names = (self.goal_prim_name,)
        if any(not isinstance(name, str) or not name for name in goal_names):
            raise ValueError(f"Task {self.name!r} has no goal annotation prim")
        return {
            "manipulated_object": (manipulated.cfg.name,),
            "goal_fixture": goal_names,
        }

    @property
    def requires_release_stability(self) -> bool:
        return self.terminal_policy == RELEASED_STABLE_MANIPULATED_OBJECT


_TASKS = {
    "grasp_classify": ExpertTaskSpec(
        name="grasp_classify",
        module="envs.grasp_classify",
        manipulated_actor_attribute="prism",
        goal_actor_attribute=None,
        goal_prim_name=None,
        goal_candidate_actor_attributes=("green_pad", "orange_pad"),
    ),
    "insert_HDMI": ExpertTaskSpec(
        name="insert_HDMI",
        module="envs.insert_HDMI",
        manipulated_actor_attribute="prism",
        goal_actor_attribute="slot",
        goal_prim_name=None,
    ),
    "insert_hole": ExpertTaskSpec(
        name="insert_hole",
        module="envs.insert_hole",
        manipulated_actor_attribute="prism",
        goal_actor_attribute="slot",
        goal_prim_name=None,
        requires_post_grasp_reference=True,
    ),
    "insert_tube": ExpertTaskSpec(
        name="insert_tube",
        module="envs.insert_tube",
        manipulated_actor_attribute="prism",
        goal_actor_attribute="slot",
        goal_prim_name=None,
        requires_post_grasp_reference=True,
    ),
    "lift_bottle": ExpertTaskSpec(
        name="lift_bottle",
        module="envs.lift_bottle",
        manipulated_actor_attribute="bottle",
        goal_actor_attribute="wall",
        goal_prim_name=None,
    ),
    "lift_can": ExpertTaskSpec(
        name="lift_can",
        module="envs.lift_can",
        manipulated_actor_attribute="can",
        goal_actor_attribute=None,
        goal_prim_name="ground_plate",
    ),
    "pull_out_key": ExpertTaskSpec(
        name="pull_out_key",
        module="envs.pull_out_key",
        manipulated_actor_attribute="key",
        goal_actor_attribute="slot",
        goal_prim_name=None,
    ),
    "put_bottle_in_shelf": ExpertTaskSpec(
        name="put_bottle_in_shelf",
        module="envs.put_bottle_in_shelf",
        manipulated_actor_attribute="bottle",
        goal_actor_attribute="shelf",
        goal_prim_name=None,
        terminal_policy=RELEASED_STABLE_MANIPULATED_OBJECT,
    ),
}


def get_expert_task(value: str) -> ExpertTaskSpec:
    try:
        return _TASKS[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_TASKS))
        raise ValueError(
            f"Unknown fixed-expert task {value!r}; choose one of: {choices}"
        ) from exc


def list_expert_tasks() -> tuple[ExpertTaskSpec, ...]:
    return tuple(_TASKS[name] for name in sorted(_TASKS))

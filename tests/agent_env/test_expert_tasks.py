from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.expert_tasks import (
    BASE_TASK_SUCCESS,
    RELEASED_STABLE_MANIPULATED_OBJECT,
    get_expert_task,
    list_expert_tasks,
)


EXPECTED_TASKS = [
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
]


def actor(name: str) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(name=name))


def test_all_eight_manipulation_tasks_support_fixed_expert_assets() -> None:
    assert [spec.name for spec in list_expert_tasks()] == EXPECTED_TASKS
    assert get_expert_task("pull_out_key").terminal_policy == BASE_TASK_SUCCESS
    bottle = get_expert_task("put_bottle_in_shelf")
    assert bottle.terminal_policy == RELEASED_STABLE_MANIPULATED_OBJECT
    assert bottle.requires_release_stability is True
    assert get_expert_task("insert_hole").requires_post_grasp_reference is True
    assert get_expert_task("insert_tube").requires_post_grasp_reference is True
    assert get_expert_task("insert_HDMI").requires_post_grasp_reference is False


def test_dynamic_grasp_classify_annotation_roles_follow_seed_choice() -> None:
    task = SimpleNamespace(
        prism=actor("rough_prism"),
        green_pad=actor("green_pad"),
        orange_pad=actor("orange_pad"),
    )
    names = get_expert_task("grasp_classify").annotation_prim_names(task)
    assert names == {
        "manipulated_object": ("rough_prism",),
        "goal_fixture": ("green_pad", "orange_pad"),
    }

    task.prism = actor("plain_prism")
    assert get_expert_task("grasp_classify").annotation_prim_names(task) == {
        "manipulated_object": ("plain_prism",),
        "goal_fixture": ("green_pad", "orange_pad"),
    }


def test_lift_can_uses_dynamic_can_and_static_ground_fixture() -> None:
    task = SimpleNamespace(can=actor("can_d6"))
    assert get_expert_task("lift_can").annotation_prim_names(task) == {
        "manipulated_object": ("can_d6",),
        "goal_fixture": ("ground_plate",),
    }


def test_unresolved_dynamic_actor_fails_closed() -> None:
    with pytest.raises(ValueError, match="no resolved manipulated actor"):
        get_expert_task("grasp_classify").annotation_prim_names(
            SimpleNamespace(
                green_pad=actor("green_pad"),
                orange_pad=actor("orange_pad"),
            )
        )

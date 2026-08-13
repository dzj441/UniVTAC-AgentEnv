from agent_env.profiles import get_profile, list_profiles


def test_three_levels_change_only_tactile_and_success_guidance() -> None:
    level1, level2, level3 = list_profiles()

    assert [profile.level for profile in (level1, level2, level3)] == [1, 2, 3]
    assert level1.public_modalities == ("head_rgb", "wrist_rgb")
    assert level2.public_modalities == (
        "head_rgb",
        "wrist_rgb",
        "left_tactile_marker",
        "right_tactile_marker",
    )
    assert level3.public_modalities == level2.public_modalities

    assert level1.public_robot_state == level2.public_robot_state == level3.public_robot_state
    assert level1.public_feedback == level2.public_feedback == ("execution_succeeded",)
    assert level3.public_feedback == (
        "execution_succeeded",
        "task_success_after_prediction",
    )


def test_profiles_have_stable_aliases() -> None:
    assert get_profile("l1") is get_profile(1)
    assert get_profile("level2") is get_profile(2)
    assert get_profile("success_guided_visuotactile_control") is get_profile(3)

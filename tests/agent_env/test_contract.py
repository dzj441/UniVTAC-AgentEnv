import pytest

from agent_env.contract import (
    EVALUATOR_SEED_ENV,
    MAX_SEED,
    consume_private_evaluator_seed,
    public_command_schema,
    validate_command_fields,
)


def test_command_schema_names_exact_action_inputs_and_finish_note() -> None:
    variants = {
        variant["title"]: variant for variant in public_command_schema()["oneOf"]
    }
    act = variants["act"]
    assert act["additionalProperties"] is False
    assert set(act["required"]) == {
        "command",
        "observation_id",
        "delta_position",
        "delta_rpy",
        "delta_gripper",
        "rationale",
    }
    assert "delta_position_world_m" not in act["properties"]
    assert "delta_translation" not in act["properties"]
    assert "final_note" in variants["finish"]["required"]


def test_unknown_or_missing_action_fields_cannot_become_silent_noops() -> None:
    base = {
        "command": "act",
        "observation_id": "obs_000",
        "delta_rpy": [0, 0, 0],
        "delta_gripper": 0,
        "rationale": "bounded transport",
    }
    with pytest.raises(ValueError, match="Missing required.*delta_position"):
        validate_command_fields({**base, "delta_translation": [0, 0.04, 0]})
    with pytest.raises(ValueError, match="Unknown field.*delta_position_world_m"):
        validate_command_fields(
            {
                **base,
                "delta_position": [0, 0.04, 0],
                "delta_position_world_m": [0, 0.04, 0],
            }
        )
    validate_command_fields({**base, "delta_position": [0, 0.04, 0]})


def test_finish_requires_final_note_and_allows_explicit_rationale() -> None:
    with pytest.raises(ValueError, match="Missing required.*final_note"):
        validate_command_fields({"command": "finish", "observation_id": "obs_003"})
    validate_command_fields(
        {
            "command": "finish",
            "observation_id": "obs_003",
            "final_note": "Placed and checked.",
            "rationale": "No additional motion is useful.",
        }
    )


def test_evaluator_seed_is_consumed_and_not_left_in_environment() -> None:
    environ = {EVALUATOR_SEED_ENV: "1049121881", "KEEP": "yes"}
    assert consume_private_evaluator_seed(environ) == 1049121881
    assert environ == {"KEEP": "yes"}


@pytest.mark.parametrize("raw", ["-1", "+1", "1.5", "seed", str(MAX_SEED + 1)])
def test_invalid_evaluator_seed_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match=EVALUATOR_SEED_ENV):
        consume_private_evaluator_seed({EVALUATOR_SEED_ENV: raw})


def test_absent_evaluator_seed_preserves_private_random_behavior() -> None:
    environ: dict[str, str] = {}
    seed = consume_private_evaluator_seed(environ)
    assert 10_000_000 <= seed <= MAX_SEED
    assert environ == {}

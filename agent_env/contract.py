"""Public command contract and evaluator-private seed handling for AgentEnv."""

from __future__ import annotations

import re
import secrets
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any


EVALUATOR_SEED_ENV = "UNIVTAC_EVALUATOR_SEED"
MIN_RANDOM_SEED = 10_000_000
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class CommandFields:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional


COMMAND_FIELDS: dict[str, CommandFields] = {
    "start": CommandFields(
        required=frozenset({"command"}),
        optional=frozenset({"agent_note", "rationale"}),
    ),
    "probe": CommandFields(
        required=frozenset(
            {"command", "observation_id", "delta_gripper", "rationale"}
        ),
    ),
    "submit_prediction": CommandFields(
        required=frozenset(
            {
                "command",
                "observation_id",
                "predicted_class",
                "target_pad",
                "rationale",
            }
        ),
    ),
    "act": CommandFields(
        required=frozenset(
            {
                "command",
                "observation_id",
                "delta_position",
                "delta_rpy",
                "delta_gripper",
                "rationale",
            }
        ),
    ),
    "wait": CommandFields(
        required=frozenset({"command", "observation_id", "steps", "rationale"}),
    ),
    "finish": CommandFields(
        required=frozenset({"command", "observation_id", "final_note"}),
        optional=frozenset({"rationale"}),
    ),
    "status": CommandFields(
        required=frozenset({"command"}),
        optional=frozenset({"rationale"}),
    ),
    "close": CommandFields(
        required=frozenset({"command"}),
        optional=frozenset({"observation_id", "rationale"}),
    ),
}


def validate_command_fields(command: dict[str, Any]) -> None:
    """Reject missing and unknown fields before a command reaches the simulator.

    A misspelled action field must not become a successful zero-motion action
    that consumes budget. Numeric bounds and state semantics remain the
    protocol state machine's responsibility.
    """

    name = command.get("command")
    if not isinstance(name, str) or name not in COMMAND_FIELDS:
        choices = ", ".join(COMMAND_FIELDS)
        raise ValueError(f"command must be one of: {choices}; got {name!r}")
    fields = COMMAND_FIELDS[name]
    supplied = frozenset(command)
    missing = sorted(fields.required - supplied)
    if missing:
        raise ValueError(
            f"Missing required field(s) for {name}: {', '.join(missing)}"
        )
    unknown = sorted(supplied - fields.allowed)
    if unknown:
        allowed = ", ".join(sorted(fields.allowed))
        raise ValueError(
            f"Unknown field(s) for {name}: {', '.join(unknown)}; "
            f"allowed fields: {allowed}"
        )


def public_command_schema() -> dict[str, Any]:
    """Return the JSON Schema advertised identically to every AgentEnv level."""

    text = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    vector3 = {
        "type": "array",
        "items": number,
        "minItems": 3,
        "maxItems": 3,
    }
    shared = {
        "observation_id": {
            **text,
            "description": "Latest public observation_id; stale IDs are rejected.",
        },
        "rationale": {
            **text,
            "description": "Explicit evidence and parameter justification.",
        },
    }
    properties: dict[str, dict[str, Any]] = {
        "start": {
            "command": {"const": "start"},
            "agent_note": {"type": "string"},
            "rationale": shared["rationale"],
        },
        "probe": {
            "command": {"const": "probe"},
            "observation_id": shared["observation_id"],
            "delta_gripper": {
                **number,
                "description": "Nonzero gripper-only delta; absolute value <= 0.002.",
            },
            "rationale": shared["rationale"],
        },
        "submit_prediction": {
            "command": {"const": "submit_prediction"},
            "observation_id": shared["observation_id"],
            "predicted_class": {"enum": ["rough", "plain"]},
            "target_pad": {"enum": ["orange", "green"]},
            "rationale": shared["rationale"],
        },
        "act": {
            "command": {"const": "act"},
            "observation_id": shared["observation_id"],
            "delta_position": {
                **vector3,
                "description": "World-frame XYZ translation in metres.",
            },
            "delta_rpy": {
                **vector3,
                "description": "World-frame roll/pitch/yaw delta in radians.",
            },
            "delta_gripper": {
                **number,
                "description": "Gripper delta; absolute value <= 0.005.",
            },
            "rationale": shared["rationale"],
        },
        "wait": {
            "command": {"const": "wait"},
            "observation_id": shared["observation_id"],
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 60,
                "description": "Physics steps to advance without a control delta.",
            },
            "rationale": shared["rationale"],
        },
        "finish": {
            "command": {"const": "finish"},
            "observation_id": shared["observation_id"],
            "final_note": {
                **text,
                "description": "Required terminal summary; must be non-empty.",
            },
            "rationale": shared["rationale"],
        },
        "status": {
            "command": {"const": "status"},
            "rationale": shared["rationale"],
        },
        "close": {
            "command": {"const": "close"},
            "observation_id": shared["observation_id"],
            "rationale": shared["rationale"],
        },
    }
    variants: list[dict[str, Any]] = []
    for name, fields in COMMAND_FIELDS.items():
        variants.append(
            {
                "title": name,
                "type": "object",
                "additionalProperties": False,
                "required": sorted(fields.required),
                "properties": properties[name],
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "UniVTAC grasp_classify AgentEnv command",
        "oneOf": variants,
    }


def consume_private_evaluator_seed(environ: MutableMapping[str, str]) -> int:
    """Consume an evaluator-selected seed without publishing it before terminal.

    If the reserved variable is absent, preserve the original random-seed
    behavior. The variable is popped immediately so later child tools cannot
    inherit it accidentally.
    """

    raw = environ.pop(EVALUATOR_SEED_ENV, None)
    if raw is None:
        return MIN_RANDOM_SEED + secrets.randbelow(MAX_SEED - MIN_RANDOM_SEED + 1)
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError(f"{EVALUATOR_SEED_ENV} must be an unsigned decimal integer")
    seed = int(raw)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"{EVALUATOR_SEED_ENV} must be in [0, {MAX_SEED}]")
    return seed

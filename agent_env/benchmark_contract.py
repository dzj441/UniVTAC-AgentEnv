"""Strict wire-command contract for the generic embodied AgentEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .benchmark_protocol import BenchmarkEpisodeProtocol


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
    ),
    "step": CommandFields(
        required=frozenset(
            {
                "command",
                "observation_id",
                "delta_position",
                "delta_rpy",
                "delta_gripper",
            }
        )
    ),
    "finish": CommandFields(
        required=frozenset({"command", "observation_id"}),
    ),
    # Host-owned cleanup command.  It is accepted by the stdio bridge only
    # after terminal evaluation and is never registered as an agent tool.
    "close": CommandFields(
        required=frozenset({"command"}),
        optional=frozenset({"observation_id", "rationale"}),
    ),
}


def validate_benchmark_command_fields(command: dict[str, Any]) -> None:
    name = command.get("command")
    if not isinstance(name, str) or name not in COMMAND_FIELDS:
        choices = ", ".join(COMMAND_FIELDS)
        raise ValueError(f"command must be one of: {choices}; got {name!r}")
    fields = COMMAND_FIELDS[name]
    supplied = frozenset(command)
    missing = sorted(fields.required - supplied)
    if missing:
        raise ValueError(f"Missing required field(s) for {name}: {', '.join(missing)}")
    unknown = sorted(supplied - fields.allowed)
    if unknown:
        raise ValueError(
            f"Unknown field(s) for {name}: {', '.join(unknown)}; "
            f"allowed fields: {', '.join(sorted(fields.allowed))}"
        )


def public_benchmark_command_schema() -> dict[str, Any]:
    """Return the three-command public protocol schema (host close excluded)."""

    text = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    vector3 = {
        "type": "array",
        "items": number,
        "minItems": 3,
        "maxItems": 3,
    }
    variants = [
        {
            "title": "start",
            "type": "object",
            "additionalProperties": False,
            "required": ["command"],
            "properties": {
                "command": {"const": "start"},
            },
        },
        {
            "title": "step",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(COMMAND_FIELDS["step"].required),
            "properties": {
                "command": {"const": "step"},
                "observation_id": text,
                "delta_position": {
                    **vector3,
                    "description": "World-frame XYZ delta in metres.",
                },
                "delta_rpy": {
                    **vector3,
                    "description": "World-frame roll/pitch/yaw delta in radians.",
                },
                "delta_gripper": {
                    **number,
                    "minimum": -BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M,
                    "maximum": BenchmarkEpisodeProtocol.GRIPPER_MAX_QPOS_M,
                    "description": (
                        "Per-finger gripper-qpos delta in metres; the resulting target "
                        "must remain within [0, 0.039]."
                    ),
                },
            },
        },
        {
            "title": "finish",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(COMMAND_FIELDS["finish"].required),
            "properties": {
                "command": {"const": "finish"},
                "observation_id": text,
            },
        },
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "UniVTAC generic embodied AgentEnv command",
        "oneOf": variants,
    }

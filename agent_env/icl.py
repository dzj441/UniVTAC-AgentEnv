"""Diagnostic in-context-learning conditions for embodied benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ICLCondition:
    name: str
    fixed_demo_available: bool

    def to_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "name": self.name,
            "fixed_demo_available": self.fixed_demo_available,
        }
        if self.fixed_demo_available:
            manifest.update(
                {
                    "workspace_path": "benchmark_inputs/expert_demo",
                    "representation": "observed_expert_waypoints",
                    "actions_present": False,
                    "episode_outcome": "successful expert demonstration",
                }
            )
        return manifest


_CONDITIONS = {
    "none": ICLCondition(name="none", fixed_demo_available=False),
    "fixed_demo": ICLCondition(name="fixed_demo", fixed_demo_available=True),
}


def get_icl_condition(value: str) -> ICLCondition:
    try:
        return _CONDITIONS[value]
    except KeyError as exc:
        choices = ", ".join(_CONDITIONS)
        raise ValueError(f"Unknown ICL condition {value!r}; choose one of: {choices}") from exc


def list_icl_conditions() -> tuple[ICLCondition, ...]:
    return tuple(_CONDITIONS[name] for name in _CONDITIONS)

"""Orthogonal semantic-perception profiles for AgentEnv rollouts.

The Level 1/2/3 profiles continue to describe embodied observations and
feedback.  This module controls a separate experimental axis: which derived,
read-only perception services the agent may call on the latest public RGB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PerceptionProfile:
    """An immutable allow-list of model-backed perception capabilities."""

    name: str
    description: str
    expose_sam3: bool
    expose_unidepth_v2: bool
    max_sam3_calls_per_observation: int
    max_unidepth_v2_calls_per_observation: int

    @property
    def evidence_sources(self) -> tuple[str, ...]:
        sources: list[str] = []
        if self.expose_sam3:
            sources.append("sam3_result")
        if self.expose_unidepth_v2:
            sources.append("unidepth_v2_result")
        return tuple(sources)

    def to_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence_sources"] = list(self.evidence_sources)
        return payload


_PROFILES = {
    "none": PerceptionProfile(
        name="none",
        description="No model-backed semantic perception tools.",
        expose_sam3=False,
        expose_unidepth_v2=False,
        max_sam3_calls_per_observation=0,
        max_unidepth_v2_calls_per_observation=0,
    ),
    "sam3": PerceptionProfile(
        name="sam3",
        description="SAM3 text/point segmentation on the latest public head or wrist RGB.",
        expose_sam3=True,
        expose_unidepth_v2=False,
        max_sam3_calls_per_observation=4,
        max_unidepth_v2_calls_per_observation=0,
    ),
    "unidepth_v2": PerceptionProfile(
        name="unidepth_v2",
        description="UniDepth V2 predicted metric-depth prior on the latest public RGB.",
        expose_sam3=False,
        expose_unidepth_v2=True,
        max_sam3_calls_per_observation=0,
        max_unidepth_v2_calls_per_observation=2,
    ),
    "sam3_unidepth_v2": PerceptionProfile(
        name="sam3_unidepth_v2",
        description="SAM3 segmentation plus UniDepth V2 predicted metric-depth prior.",
        expose_sam3=True,
        expose_unidepth_v2=True,
        max_sam3_calls_per_observation=4,
        max_unidepth_v2_calls_per_observation=2,
    ),
}

_ALIASES = {
    "": "none",
    "off": "none",
    "raw": "none",
    "p0": "none",
    "p1": "sam3",
    "sam": "sam3",
    "depth": "unidepth_v2",
    "unidepth": "unidepth_v2",
    "full": "sam3_unidepth_v2",
    "p3": "sam3_unidepth_v2",
}


def get_perception_profile(value: str | PerceptionProfile | None) -> PerceptionProfile:
    if isinstance(value, PerceptionProfile):
        return value
    key = str(value or "none").strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    try:
        return _PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown perception profile {value!r}; choose "
            + ", ".join(sorted(_PROFILES))
        ) from exc


def list_perception_profiles() -> tuple[PerceptionProfile, ...]:
    return tuple(_PROFILES[name] for name in _PROFILES)

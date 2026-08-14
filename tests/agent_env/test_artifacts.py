import pytest

from agent_env.artifacts import require_initial_tactile_health


def test_tactile_health_fails_fast_only_for_initial_observation() -> None:
    unhealthy = {
        "left": {"healthy": False, "plausible_marker_components": 28},
        "right": {"healthy": False, "plausible_marker_components": 22},
    }
    with pytest.raises(RuntimeError, match="Initial tactile"):
        require_initial_tactile_health(unhealthy, initial_observation=True)

    # A strong-contact frame remains public after the physical action. This
    # prevents the world from changing behind a command_error response.
    require_initial_tactile_health(unhealthy, initial_observation=False)


def test_initial_tactile_health_accepts_healthy_marker_grids() -> None:
    healthy = {
        "left": {"healthy": True, "plausible_marker_components": 63},
        "right": {"healthy": True, "plausible_marker_components": 60},
    }
    require_initial_tactile_health(healthy, initial_observation=True)

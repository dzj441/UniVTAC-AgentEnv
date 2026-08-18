import json
from pathlib import Path

import pytest
from PIL import Image

import agent_env.artifacts as artifacts
from agent_env.artifacts import (
    encode_observation_toolcall_video,
    render_observation_toolcall_frame,
    require_initial_tactile_health,
    tool_calls_by_observation,
)


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tool_calls() -> list[dict]:
    return [
        {
            "sequence": 0,
            "tool": "start_episode",
            "success": True,
            "prior_observation_id": None,
            "next_observation_id": "obs_000",
            "arguments": {"agent_note": "start"},
        },
        {
            "sequence": 1,
            "tool": "commit_classification",
            "success": True,
            "execution_target": "simulator",
            "prior_observation_id": "obs_000",
            "next_observation_id": "obs_000",
            "arguments": {
                "observation_id": "obs_000",
                "predicted_class": "plain",
                "target_pad": "green",
                "decision_record": {"rationale": "The surface is visually uniform."},
            },
        },
        {
            "sequence": 2,
            "tool": "act_delta_ee",
            "success": True,
            "execution_target": "simulator",
            "prior_observation_id": "obs_000",
            "next_observation_id": "obs_001",
            "arguments": {
                "observation_id": "obs_000",
                "delta_position": [0.04, 0.0, 0.0],
                "delta_rpy": [0.0, 0.0, 0.0],
                "delta_gripper": 0.0,
            },
        },
        {
            "sequence": 3,
            "tool": "act_delta_ee",
            "success": False,
            "execution_target": "rejected",
            "prior_observation_id": "obs_001",
            "next_observation_id": "obs_001",
            "arguments": {
                "observation_id": "obs_000001",
                "delta_position": [0.0, 0.0, 0.0],
                "delta_rpy": [0.0, 0.0, 0.0],
                "delta_gripper": 0.005,
            },
        },
    ]


def test_tool_calls_are_mapped_by_host_recorded_prior_observation(tmp_path: Path) -> None:
    tool_path = tmp_path / "codex_tool_calls.jsonl"
    _write_jsonl(tool_path, _tool_calls())

    grouped = tool_calls_by_observation(tool_path)

    assert [call["tool"] for call in grouped["obs_000"]] == [
        "commit_classification",
        "act_delta_ee",
    ]
    assert grouped["obs_001"][0]["execution_target"] == "rejected"
    assert grouped["obs_001"][0]["arguments"]["observation_id"] == "obs_000001"
    assert all(call["tool"] != "start_episode" for calls in grouped.values() for call in calls)


def test_toolcall_frame_preserves_observation_and_adds_even_width_panel(
    tmp_path: Path,
) -> None:
    composite = tmp_path / "composite.png"
    Image.new("RGB", (960, 294), (20, 40, 60)).save(composite)
    calls = _tool_calls()[1:3]

    frame = render_observation_toolcall_frame(composite, "obs_000", calls)

    assert frame.size == (1920, 480)
    assert frame.getpixel((10, 10)) == (20, 40, 60)
    assert frame.getpixel((1000, 250)) != (20, 40, 60)
    assert frame.getpixel((1000, 80)) == (15, 118, 110)


def test_generic_step_eef_uses_the_numeric_action_renderer(tmp_path: Path) -> None:
    composite = tmp_path / "composite.png"
    Image.new("RGB", (960, 294), (20, 40, 60)).save(composite)
    call = {
        "tool": "step_eef",
        "success": True,
        "prior_observation_id": "obs_000",
        "next_observation_id": "obs_001",
        "arguments": {
            "observation_id": "obs_000",
            "delta_position": [0.01, -0.02, 0.03],
            "delta_rpy": [0.1, 0.0, -0.1],
            "delta_gripper": 0.005,
        },
    }
    frame = render_observation_toolcall_frame(composite, "obs_000", [call])
    assert frame.size == (1920, 480)


def test_video_font_loader_honors_requested_size() -> None:
    small = artifacts._font(14, bold=True)
    large = artifacts._font(28, bold=True)
    small_box = small.getbbox("TOOL CALL")
    large_box = large.getbbox("TOOL CALL")

    assert large_box[2] - large_box[0] > (small_box[2] - small_box[0]) * 1.8
    assert large_box[3] - large_box[1] > small_box[3] - small_box[1]


def test_toolcall_video_has_one_frame_per_observation_and_keeps_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_root = tmp_path / "observations"
    for observation_id in ("obs_000", "obs_001"):
        directory = observation_root / observation_id
        directory.mkdir(parents=True)
        Image.new("RGB", (960, 294), (30, 50, 70)).save(directory / "composite.png")
    tool_path = tmp_path / "codex_tool_calls.jsonl"
    _write_jsonl(tool_path, _tool_calls())
    inspected_sizes: list[tuple[int, int]] = []

    def fake_encode(
        frame_pattern: str,
        video_path: Path,
        *,
        frames_per_second: float = 2.0,
    ) -> dict:
        assert frames_per_second == 1.0
        frame_paths = sorted(Path(frame_pattern).parent.glob("frame_*.png"))
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                inspected_sizes.append(frame.size)
        video_path.write_bytes(b"fake-h264")
        return {
            "path": str(video_path),
            "sha256": "temporary",
            "codec_name": "h264",
            "profile": "Main",
            "pix_fmt": "yuv420p",
            "width": 1920,
            "height": 480,
            "nb_frames": "2",
            "frames_per_second": frames_per_second,
        }

    monkeypatch.setattr(artifacts, "_encode_h264", fake_encode)
    output = tmp_path / "agent_timeline_h264.mp4"

    metadata = encode_observation_toolcall_video(observation_root, tool_path, output)

    assert output.read_bytes() == b"fake-h264"
    assert inspected_sizes == [(1920, 480), (1920, 480)]
    assert metadata["observation_frame_count"] == 2
    assert metadata["mapped_tool_call_count"] == 3
    assert metadata["frames_per_second"] == 1.0
    assert metadata["layout"] == "public_observation_plus_agent_tool_calls"

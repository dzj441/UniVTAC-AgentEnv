from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from agent_env.run_viewer import (
    RunRepository,
    ViewerDataError,
    create_server,
    public_viewer_url,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[object], *, partial_tail: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(value) for value in values) + "\n"
    if partial_tail:
        body += '{"incomplete":'
    path.write_text(body, encoding="utf-8")


def _observation(run: Path, observation_id: str, gripper: float, x: float) -> dict:
    image = run / "observations" / observation_id / "head_rgb.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fake-png")
    return {
        "observation_id": observation_id,
        "level": 1,
        "profile": "vision_only_control",
        "stage": "classification",
        "probe_count": int(observation_id.split("_")[1]),
        "post_prediction_action_count": 0,
        "modalities": {
            "head_rgb": {
                "artifact_id": f"observations/{observation_id}/head_rgb.png",
                "sha256": "a" * 64,
            }
        },
        "robot_state": {
            "gripper_qpos": gripper,
            "joint_position_8d": [0.0] * 8,
            "end_effector_pose_robot_base_7d": [x, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0],
        },
    }


def _make_codex_run(root: Path) -> Path:
    run = root / "round" / "level1"
    first = _observation(run, "obs_000", 0.007, 0.35)
    second = _observation(run, "obs_001", 0.006, 0.36)
    _write_json(
        run / "codex_run_manifest.json",
        {
            "created_utc": "2026-08-14T00:00:00+00:00",
            "task": "grasp_classify",
            "level": 1,
            "model": "test-model",
            "effort": "high",
            "profile": {
                "level": 1,
                "name": "vision_only_control",
                "description": "vision",
                "public_modalities": ["head_rgb"],
                "public_robot_state": ["gripper_qpos"],
                "expose_tactile": False,
                "expose_task_success_after_prediction": False,
            },
        },
    )
    _write_json(
        run / "codex_run_outcome.json",
        {
            "level": 1,
            "status": "completed_valid",
            "valid_for_scoring": True,
            "total_wall_seconds": 30.0,
            "codex_runtime": {
                "actual_model": "test-model",
                "reasoning_effort": "high",
                "token_usage": {"total": {"totalTokens": 123}},
            },
        },
    )
    _write_json(
        run / "evaluator_outcome.json",
        {
            "level": 1,
            "predicted_class": "rough",
            "true_class": "rough",
            "classification_correct": True,
            "committed_target": "orange",
            "expected_target": "orange",
            "committed_pad_correct": True,
            "official_task_success": True,
        },
    )
    decision = {
        "rationale": "probe once",
        "evidence": [
            {"source": "head_rgb", "finding": "dark face", "implication": "probe"}
        ],
        "alternatives_considered": ["commit now"],
        "uncertainty": 0.4,
        "expected_effect": "fresh observation",
        "parameter_rationale": "1 mm is conservative",
    }
    _write_jsonl(
        run / "codex_tool_calls.jsonl",
        [
            {
                "sequence": 0,
                "call_id": "start",
                "tool": "start_episode",
                "success": True,
                "elapsed_seconds": 10,
                "duration_seconds": 2,
                "arguments": {"agent_note": "test"},
                "simulator_command": {"command": "start"},
                "simulator_response": {"status": "rollout_started", "observation": first},
                "prior_observation_id": None,
                "next_observation_id": "obs_000",
            },
            {
                "sequence": 1,
                "call_id": "probe",
                "tool": "probe_gripper",
                "success": True,
                "elapsed_seconds": 20,
                "duration_seconds": 3,
                "arguments": {
                    "observation_id": "obs_000",
                    "delta_gripper": -0.001,
                    "decision_record": decision,
                },
                "simulator_command": {
                    "command": "probe",
                    "observation_id": "obs_000",
                    "delta_gripper": -0.001,
                },
                "simulator_response": {
                    "status": "probe_complete",
                    "feedback": {"execution_succeeded": True},
                    "observation": second,
                },
                "prior_observation_id": "obs_000",
                "next_observation_id": "obs_001",
            },
        ],
        partial_tail=True,
    )
    _write_jsonl(
        run / "codex_decisions.jsonl",
        [{"call_id": "probe", "decision_record": decision}],
    )
    _write_jsonl(
        run / "codex_messages.jsonl",
        [
            {"kind": "published_reasoning_summary", "summary": ["start"], "elapsed_seconds": 5},
            {"kind": "published_reasoning_summary", "summary": ["probe plan"], "elapsed_seconds": 15},
            {"kind": "agent_message", "text": "done", "elapsed_seconds": 25},
        ],
    )
    (run / "codex_operator_prompt.txt").write_text("test prompt", encoding="utf-8")
    (run / "agent_observations_h264.mp4").write_bytes(b"0123456789")
    return run


def _make_capture_run(root: Path) -> Path:
    run = root / "comparison" / "rough"
    observation = _observation(run, "obs_000", 0.007, 0.35)
    _write_json(
        run / "manifest.json",
        {
            "created_utc": "2026-08-14T01:00:00+00:00",
            "task": "grasp_classify",
            "profile": {"level": 2, "name": "visuotactile_control"},
        },
    )
    _write_jsonl(
        run / "agent_transcript.jsonl",
        [
            {"timestamp_utc": "2026-08-14T01:00:00+00:00", "kind": "bridge_ready"},
            {
                "timestamp_utc": "2026-08-14T01:00:01+00:00",
                "kind": "command",
                "command": {"command": "start", "agent_note": "capture"},
            },
            {
                "timestamp_utc": "2026-08-14T01:00:03+00:00",
                "kind": "response",
                "response": {"status": "rollout_started", "observation": observation},
            },
        ],
    )
    return run


def test_public_viewer_url_resolves_code_server_template() -> None:
    template = "https://example.test/code/proxy/{{port}}/"
    assert public_viewer_url(8765, template) == "https://example.test/code/proxy/8765/"
    assert public_viewer_url(42, "https://x/proxy/{port}") == "https://x/proxy/42/"
    assert public_viewer_url(42, "https://x/no-placeholder") is None


def test_repository_discovers_codex_and_capture_runs(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    _make_codex_run(root)
    _make_capture_run(root)
    summaries = RunRepository(root).list_runs()
    assert {(item["id"], item["kind"]) for item in summaries} == {
        ("round/level1", "codex"),
        ("comparison/rough", "capture"),
    }
    codex = next(item for item in summaries if item["kind"] == "codex")
    assert codex["tool_count"] == 2
    assert codex["observation_count"] == 2
    assert codex["total_tokens"] == 123


def test_codex_detail_joins_reasoning_decision_action_and_observation(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    _make_codex_run(root)
    detail = RunRepository(root).detail("round/level1")
    assert len(detail["steps"]) == 2
    assert detail["steps"][0]["messages"][0]["parts"] == ["start"]
    probe = detail["steps"][1]
    assert probe["messages"][0]["parts"] == ["probe plan"]
    assert probe["decision_source"] == "host_validated"
    assert probe["decision"]["parameter_rationale"] == "1 mm is conservative"
    assert probe["arguments"] == {"observation_id": "obs_000", "delta_gripper": -0.001}
    assert probe["output_observation"]["modalities"]["head_rgb"]["artifact"].endswith(
        "obs_001/head_rgb.png"
    )
    assert probe["state_delta"]["gripper_delta_mm"] == pytest.approx(-1.0)
    assert probe["state_delta"]["end_effector_translation_m"] == pytest.approx(0.01)
    assert detail["tail_messages"][0]["parts"] == ["done"]


def test_protocol_status_shell_does_not_overwrite_complete_observation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_runs"
    run = _make_codex_run(root)
    tool_path = run / "codex_tool_calls.jsonl"
    calls = []
    for line in tool_path.read_text(encoding="utf-8").splitlines():
        try:
            calls.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    calls.append(
        {
            "sequence": 2,
            "call_id": "commit",
            "tool": "commit_classification",
            "success": True,
            "elapsed_seconds": 22,
            "duration_seconds": 0.01,
            "arguments": {
                "observation_id": "obs_001",
                "predicted_class": "rough",
                "target_pad": "orange",
            },
            "simulator_command": {
                "command": "submit_prediction",
                "observation_id": "obs_001",
                "predicted_class": "rough",
                "target_pad": "orange",
            },
            # This is an acknowledgement of a stage transition, not a fresh
            # sensor observation, despite carrying the current observation_id.
            "simulator_response": {
                "status": "prediction_submitted",
                "observation_id": "obs_001",
                "stage": "post_prediction_control",
            },
            "prior_observation_id": "obs_001",
            "next_observation_id": "obs_001",
        }
    )
    _write_jsonl(tool_path, calls)

    detail = RunRepository(root).detail("round/level1")
    probe_observation = detail["steps"][1]["output_observation"]
    commit = detail["steps"][2]
    assert probe_observation["observation_id"] == "obs_001"
    assert probe_observation["stage"] == "classification"
    assert set(probe_observation["modalities"]) == {"head_rgb"}
    assert probe_observation["robot_state"]["gripper_qpos"] == 0.006
    assert commit["input_observation"] == probe_observation
    assert commit["fresh_observation"] is False
    assert commit["environment_response"]["stage"] == "post_prediction_control"


def test_capture_detail_keeps_commands_but_does_not_invent_reasoning(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    _make_capture_run(root)
    detail = RunRepository(root).detail("comparison/rough")
    assert detail["summary"]["kind"] == "capture"
    assert len(detail["steps"]) == 1
    assert detail["steps"][0]["tool"] == "start_episode"
    assert detail["steps"][0]["decision"] is None
    assert detail["steps"][0]["output_observation"]["observation_id"] == "obs_000"
    assert "没有 Codex" in detail["recording_note"]


def test_repository_rejects_path_escape_and_outside_symlink(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    run = _make_codex_run(root)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (run / "outside-link").symlink_to(outside)
    _write_json(run / ".host_sensor_metadata" / "obs_000.json", {"fx": 123.0})
    (run / "unreferenced.txt").write_text("not public", encoding="utf-8")
    repository = RunRepository(root)
    assert repository.resolve_artifact(
        "round/level1", "observations/obs_000/head_rgb.png"
    ).is_file()
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact("round/level1", "../../secret.txt")
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact("round/level1", "outside-link")
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact(
            "round/level1", ".host_sensor_metadata/obs_000.json"
        )
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact("round/level1", "unreferenced.txt")


def test_http_api_supports_proxy_prefix_and_video_ranges(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    _make_codex_run(root)
    server = create_server("127.0.0.1", 0, root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/code/proxy/999/api/runs", timeout=5) as response:
            payload = json.load(response)
        assert payload["runs"][0]["id"] == "round/level1"

        video_url = (
            f"{base}/code/proxy/999/api/artifact?"
            "run=round%2Flevel1&path=agent_observations_h264.mp4"
        )
        request = Request(video_url, headers={"Range": "bytes=2-5"})
        with urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"

        with pytest.raises(HTTPError) as error:
            urlopen(
                f"{base}/api/artifact?run=round%2Flevel1&path=..%2F..%2Fsecret",
                timeout=5,
            )
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

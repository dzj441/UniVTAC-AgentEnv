from __future__ import annotations

import json
from pathlib import Path

from agent_env.codex_events import (
    EventRecorder,
    audit_codex_events,
    build_benchmark_score,
    build_human_trace,
    check_token_budget,
    summarize_codex_runtime,
    summarize_tool_calls,
    token_budget_manifest,
)


def test_event_streams_are_append_only_hash_chains_and_redact_data_urls(
    tmp_path: Path,
) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw(
        "host_to_server",
        {
            "method": "item/tool/call",
            "params": {
                "imageUrl": "data:image/png;base64,aGk=",
                "access_token": "secret",
            },
        },
    )
    recorder.record_raw("server_to_host", {"method": "turn/completed"})

    rows = [
        json.loads(line)
        for line in recorder.raw_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["previous_event_sha256"] == "0" * 64
    assert rows[1]["previous_event_sha256"] == rows[0]["event_sha256"]
    serialized = json.dumps(rows)
    assert "secret" not in serialized
    assert "aGk=" not in serialized
    assert "payload-omitted bytes=2" in serialized
    summary = recorder.summary()["app_server_events"]
    assert summary["event_count"] == 2
    assert len(summary["file_sha256"]) == 64
    assert audit_codex_events(recorder.raw_path)["passed"] is True


def test_event_audit_detects_hash_chain_tampering(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw("server_to_host", {"method": "turn/started"})
    recorder.record_raw("server_to_host", {"method": "turn/completed"})
    rows = recorder.raw_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[0])
    tampered["message"]["method"] = "item/started"
    rows[0] = json.dumps(tampered)
    recorder.raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    audit = audit_codex_events(recorder.raw_path)
    assert audit["passed"] is False
    assert "event_hash_mismatch" in {
        violation["reason"] for violation in audit["violations"]
    }


def test_event_audit_fails_on_shell_file_mcp_and_subagent_items(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path)
    for item_type in (
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "imageGeneration",
        "hookPrompt",
    ):
        recorder.record_raw(
            "server_to_host",
            {
                "method": "item/started",
                "params": {"item": {"id": item_type, "type": item_type}},
            },
        )
    audit = audit_codex_events(recorder.raw_path)
    assert audit["passed"] is False
    assert {item["item_type"] for item in audit["violations"]} == {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "imageGeneration",
        "hookPrompt",
    }


def test_event_audit_fails_closed_on_future_unknown_item(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw(
        "server_to_host",
        {
            "method": "item/started",
            "params": {"item": {"id": "future-1", "type": "futurePowerTool"}},
        },
    )
    audit = audit_codex_events(recorder.raw_path)
    assert audit["passed"] is False
    assert audit["violations"][0]["reason"] == "unknown_item_type_fail_closed"


def test_event_audit_fails_closed_on_future_server_request(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw(
        "server_to_host",
        {"id": 99, "method": "future/tool/request", "params": {}},
    )
    audit = audit_codex_events(recorder.raw_path)
    assert audit["passed"] is False
    assert audit["violations"][0]["reason"] == "unknown_server_request_fail_closed"


def test_runtime_and_tool_summaries_use_terminal_app_server_state(
    tmp_path: Path,
) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw(
        "server_to_host",
        {
            "id": 2,
            "result": {
                "model": "initial-model",
                "modelProvider": "openai",
                "serviceTier": None,
            },
        },
    )
    recorder.record_raw(
        "server_to_host",
        {
            "method": "thread/settings/updated",
            "params": {
                "threadSettings": {
                    "model": "resolved-model",
                    "modelProvider": "openai",
                    "serviceTier": "priority",
                    "effort": "high",
                    "summary": "detailed",
                }
            },
        },
    )
    recorder.record_raw(
        "server_to_host",
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "tokenUsage": {
                    "total": {"totalTokens": 123, "outputTokens": 7},
                    "modelContextWindow": 258400,
                }
            },
        },
    )
    recorder.record_tool_call({"tool": "start_episode", "success": True})
    recorder.record_tool_call({"tool": "act_delta_ee", "success": False})

    runtime = summarize_codex_runtime(recorder.raw_path)
    assert runtime == {
        "actual_model": "resolved-model",
        "model_provider": "openai",
        "service_tier": "priority",
        "reasoning_effort": "high",
        "reasoning_summary": "detailed",
        "token_usage": {
            "total": {"totalTokens": 123, "outputTokens": 7},
            "modelContextWindow": 258400,
        },
    }
    assert summarize_tool_calls(recorder.tool_path) == {
        "total": 2,
        "relayed_to_simulator": 0,
        "relayed_to_perception": 0,
        "host_rejected": 2,
        "successful_results": 1,
        "failed_results": 1,
    }


def test_output_token_budget_uses_cumulative_output_without_double_counting_reasoning(
    tmp_path: Path,
) -> None:
    recorder = EventRecorder(tmp_path)
    recorder.record_raw(
        "server_to_host",
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "tokenUsage": {
                    "total": {
                        "totalTokens": 10800,
                        "inputTokens": 8000,
                        "outputTokens": 2800,
                        "reasoningOutputTokens": 1200,
                    }
                }
            },
        },
    )
    runtime = summarize_codex_runtime(recorder.raw_path)

    exact = check_token_budget(runtime, 2800)
    assert exact["observed_output_tokens"] == 2800
    assert exact["observed_reasoning_output_tokens"] == 1200
    assert exact["within_budget"] is True
    assert exact["budget_passed"] is True

    exceeded = check_token_budget(runtime, 2799)
    assert exceeded["within_budget"] is False
    assert exceeded["budget_passed"] is False
    assert exceeded["failure_reason"] == "token_budget_exceeded"
    score = build_benchmark_score(
        valid_for_scoring=True,
        official_task_success=True,
        token_budget_check=exceeded,
    )
    assert score["benchmark_success"] is False
    assert score["failure_reasons"] == ["token_budget_exceeded"]


def test_configured_token_budget_fails_closed_when_usage_is_unavailable() -> None:
    missing = check_token_budget({}, 1000)
    assert missing["measurement_available"] is False
    assert missing["within_budget"] is None
    assert missing["budget_passed"] is False
    assert missing["failure_reason"] == "token_usage_unavailable"

    unlimited = check_token_budget({}, None)
    assert unlimited["configured"] is False
    assert unlimited["budget_passed"] is True
    assert unlimited["failure_reason"] is None


def test_token_budget_rejects_non_positive_or_boolean_limits() -> None:
    for invalid in (0, -1, True):
        try:
            token_budget_manifest(invalid)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"Expected invalid token budget {invalid!r} to fail")


def test_human_trace_joins_decision_action_observation_and_outcome(tmp_path: Path) -> None:
    (tmp_path / "codex_run_manifest.json").write_text(
        json.dumps(
            {
                "task": "grasp_classify",
                "level": 2,
                "model": "test-model",
                "effort": "high",
                "capability_manifest_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "evaluator_outcome.json").write_text(
        json.dumps(
            {
                "predicted_class": "rough",
                "true_class": "rough",
                "committed_target": "orange",
                "expected_target": "orange",
                "classification_correct": True,
                "official_task_success": False,
                "wall_seconds": 12.0,
            }
        ),
        encoding="utf-8",
    )
    recorder = EventRecorder(tmp_path)
    recorder.record_message(
        {
            "kind": "published_reasoning_summary",
            "summary": ["bounded move"],
            "elapsed_seconds": 1.0,
        }
    )
    recorder.record_tool_call(
        {
            "call_id": "call-1",
            "tool": "act_delta_ee",
            "success": True,
            "prior_observation_id": "obs_002",
            "next_observation_id": "obs_003",
            "duration_seconds": 1.5,
            "elapsed_seconds": 2.0,
            "simulator_command": {"command": "act", "delta_position": [0, -0.02, 0]},
            "simulator_response": {
                "status": "action_complete",
                "observation": {
                    "observation_id": "obs_003",
                    "modalities": {
                        "head_rgb": {
                            "artifact_id": "observations/obs_003/head_rgb.png",
                            "sha256": "deadbeef",
                        }
                    },
                },
            },
        }
    )
    recorder.record_decision(
        {
            "call_id": "call-1",
            "decision_record": {
                "uncertainty": 0.2,
                "rationale": "Move toward orange.",
                "parameter_rationale": "Two centimetres avoids overshoot.",
                "expected_effect": "Orange becomes centered.",
                "evidence": [
                    {
                        "source": "head_rgb",
                        "finding": "Orange is left.",
                        "implication": "Move left.",
                    }
                ],
                "alternatives_considered": ["Move one centimetre, but it is too small."],
            },
        }
    )
    trace = build_human_trace(
        tmp_path,
        runtime_summary={"actual_model": "resolved-test-model"},
    )
    rendered = trace.read_text(encoding="utf-8")
    assert "Codex Agent Rollout 复盘" in rendered
    assert "obs_002" in rendered and "obs_003" in rendered
    assert "Two centimetres avoids overshoot" in rendered
    assert "observations/obs_003/head_rgb.png" in rendered
    assert "Official task success：`False`" in rendered
    assert "Actual model：`resolved-test-model`" in rendered
    assert rendered.index("Codex 公开 reasoning summary") < rendered.index(
        "`act_delta_ee`"
    )

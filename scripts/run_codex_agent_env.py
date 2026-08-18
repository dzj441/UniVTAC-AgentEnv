#!/usr/bin/env python3
"""Launch a simulator-owned AgentEnv and an isolated Codex embodied operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.capabilities import (  # noqa: E402
    CapabilityGateway,
    CapabilityViolation,
    capability_manifest,
)
from agent_env.artifacts import (  # noqa: E402
    encode_observation_toolcall_video,
)
from agent_env.codex_app_server import CodexAppServerClient  # noqa: E402
from agent_env.codex_events import (  # noqa: E402
    EventRecorder,
    audit_codex_events,
    build_human_trace,
    summarize_codex_runtime,
    summarize_tool_calls,
    utc_now,
)
from agent_env.codex_isolation import (  # noqa: E402
    IsolatedCodexEnvironment,
    embodied_codex_command,
)
from agent_env.profiles import get_profile  # noqa: E402
from agent_env.perception_profiles import (  # noqa: E402
    get_perception_profile,
    list_perception_profiles,
)
from agent_env.perception_runtime import PerceptionRuntime  # noqa: E402
from agent_env.stdio_bridge import SimulatorProcessClient  # noqa: E402


BASE_INSTRUCTIONS = """\
You are the control policy in an agentic embodied benchmark. The host owns the
simulator and exposes an intentionally small set of typed embodied tools. Use
only those tools. Do not request shell commands, files, web access, MCP/apps,
plugins, skills, subagents, arbitrary code, inverse kinematics, joint targets,
trajectory planning, or simulator internals. A capability that is absent is
unavailable, not an invitation to reconstruct or bypass it.

All externally meaningful reasoning must be placed in each tool's structured
decision_record. Ground evidence only in the latest observation and only in
sources accepted by the tool schema. The host records published reasoning
summaries and agent messages, but hidden model reasoning is not an observable
benchmark artifact.
"""


def operator_prompt(level: int, perception_profile_name: str = "none") -> str:
    """Return task semantics and generic placement guidance."""

    # Keep a stable call signature for manifests and external launchers while
    # deliberately withholding Level, modality, coordinate, budget, and feedback
    # descriptions from the task prompt. The placement hint is identical across
    # Levels and does not expose simulator state or checker geometry.
    del level, perception_profile_name
    return """\
Task: grasp_classify

Determine whether the grasped prism is rough or plain. Place a rough prism on
the orange pad and a plain prism on the green pad.
For a stable placement, align the prism upright over the center of the target
pad before releasing it; avoid releasing near the pad edge.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated Codex grasp_classify AgentEnv episode"
    )
    parser.add_argument("--level", required=True, choices=("1", "2", "3"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--auth-home", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--perception-profile",
        default="none",
        choices=tuple(profile.name for profile in list_perception_profiles()),
        help="Orthogonal read-only perception surface; default keeps the original benchmark.",
    )
    parser.add_argument(
        "--sam3-url",
        default=os.environ.get("UNIVTAC_SAM3_URL", "http://127.0.0.1:8783"),
    )
    parser.add_argument(
        "--unidepth-v2-url",
        default=os.environ.get("UNIVTAC_UNIDEPTH_V2_URL", "http://127.0.0.1:8784"),
    )
    parser.add_argument("--perception-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact public capability manifest without launching Codex/Isaac.",
    )
    return parser.parse_args()


def resolve_run_dir(value: Path | None, level: int) -> Path:
    if value is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return REPO_ROOT / "agent_runs" / f"codex_agentenv_l{level}_{stamp}"
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_agent_timeline_video(run_dir: Path, tool_call_path: Path) -> dict[str, Any]:
    """Create a supplemental post-run replay without mutating Viewer media."""

    return encode_observation_toolcall_video(
        run_dir / "observations",
        tool_call_path,
        run_dir / "agent_timeline_h264.mp4",
    )


def command_version(command: str) -> str:
    completed = subprocess.run(
        [command, "--version"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip() or completed.stderr.strip()


def git_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": revision, "working_tree_dirty": dirty}


def main() -> int:
    run_started_monotonic = time.monotonic()
    args = parse_args()
    level = int(args.level)
    perception_profile = get_perception_profile(args.perception_profile)
    capabilities = capability_manifest(level, perception_profile)
    if args.dry_run:
        print(json.dumps(capabilities, ensure_ascii=False, indent=2))
        return 0

    run_dir = resolve_run_dir(args.run_dir, level)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_fd, log_name = tempfile.mkstemp(
        prefix=f".{run_dir.name}.simulator-", suffix=".log", dir=run_dir.parent
    )
    os.close(log_fd)
    temporary_simulator_log = Path(log_name)

    simulator: SimulatorProcessClient | None = None
    codex: CodexAppServerClient | None = None
    recorder: EventRecorder | None = None
    perception_runtime: PerceptionRuntime | None = None
    perception_health: dict[str, Any] | None = None
    host_close_response: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "schema_version": "univtac.codex_rollout_outcome.v1",
        "created_utc": utc_now(),
        "level": level,
        "run_dir": str(run_dir),
        "valid_for_scoring": False,
        "status": "starting",
    }

    try:
        simulator_command = [
            str(REPO_ROOT / "scripts" / "launch_agent_env.sh"),
            "--level",
            str(level),
            "--device",
            args.device,
            "--run-dir",
            str(run_dir),
        ]
        simulator = SimulatorProcessClient(
            simulator_command,
            cwd=REPO_ROOT,
            timeout_seconds=args.timeout_seconds,
            startup_log_path=temporary_simulator_log,
        )
        ready = simulator.ready
        if ready.get("level") != level:
            raise RuntimeError(f"Simulator level mismatch: {ready}")
        if ready.get("capabilities") != get_profile(level).to_manifest():
            raise RuntimeError("Simulator capability profile disagrees with host registry")

        recorder = EventRecorder(run_dir)
        if perception_profile.name != "none":
            perception_runtime = PerceptionRuntime(
                profile=perception_profile,
                run_dir=run_dir,
                sam3_url=args.sam3_url,
                unidepth_v2_url=args.unidepth_v2_url,
                timeout_seconds=args.perception_timeout_seconds,
            )
            perception_health = perception_runtime.health_manifest()
        gateway = CapabilityGateway(
            level=level,
            simulator_request=simulator.request,
            simulator_run_dir=run_dir,
            perception_profile=perception_profile,
            perception_runtime=perception_runtime,
        )
        prompt = operator_prompt(level, perception_profile.name)
        (run_dir / "codex_operator_prompt.txt").write_text(prompt, encoding="utf-8")
        write_json(run_dir / "codex_capabilities.json", capabilities)

        with IsolatedCodexEnvironment(args.auth_home) as isolated:
            isolation_manifest = isolated.manifest()
            app_server_command = embodied_codex_command(args.codex_bin)
            manifest = {
                "schema_version": "univtac.codex_rollout_manifest.v1",
                "created_utc": utc_now(),
                "task": "grasp_classify",
                "level": level,
                "profile": get_profile(level).to_manifest(),
                "perception_profile": perception_profile.to_manifest(),
                "perception_service_health_at_start": perception_health,
                "requested_model": args.model,
                "model": args.model or "Codex configured default",
                "effort": args.effort,
                "reasoning_summary": "detailed",
                "codex_version": command_version(args.codex_bin),
                "app_server_command": app_server_command,
                "operator_prompt_file": "codex_operator_prompt.txt",
                "operator_prompt_sha256": sha256_text(prompt),
                "capability_manifest_file": "codex_capabilities.json",
                "capability_manifest_sha256": capabilities["sha256"],
                "simulator_ready_commitment": ready.get("seed_commitment_sha256"),
                "source": git_revision(),
                "isolation": isolation_manifest,
                "enforcement": {
                    "dynamic_tool_allowlist": True,
                    "host_side_tool_validation": True,
                    "simulator_side_command_validation": True,
                    "host_resolved_latest_rgb_for_perception": True,
                    "semantic_evidence_requires_prior_successful_tool_result": True,
                    "semantic_call_budget_per_observation": {
                        "sam3_segment": perception_profile.max_sam3_calls_per_observation,
                        "estimate_metric_depth": (
                            perception_profile.max_unidepth_v2_calls_per_observation
                        ),
                    },
                    "host_only_calibrated_intrinsics": True,
                    "codex_read_only_sandbox": True,
                    "codex_network_access": False,
                    "forbidden_item_fail_closed": True,
                    "max_consecutive_rejected_tool_calls": (
                        gateway.MAX_CONSECUTIVE_REJECTED_CALLS
                    ),
                    "model_visible_close_command": False,
                },
                "threat_model": {
                    "benchmark_isolation": (
                        "The model receives only dynamic tool outputs in a fresh empty "
                        "workspace and cannot obtain a simulator handle through those tools."
                    ),
                    "security_boundary": (
                        "This host forbids user/mount namespaces; same-UID adversarial process "
                        "isolation still requires an external container or separate UID."
                    ),
                },
                "recording": {
                    "published_codex_events": "codex_app_server_events.jsonl",
                    "tool_calls": "codex_tool_calls.jsonl",
                    "explicit_decisions": "codex_decisions.jsonl",
                    "published_messages_and_reasoning_summaries": "codex_messages.jsonl",
                    "capability_violations": "capability_violations.jsonl",
                    "simulator_transcript": "agent_transcript.jsonl",
                    "video": (
                        "agent_observations_h264.mp4 remains the evaluator-owned Viewer "
                        "replay; agent_timeline_h264.mp4 is a supplemental post-run "
                        "observation + tool-call sharing artifact"
                    ),
                    "semantic_perception": "semantic_perception/<observation>/<tool>/<call>",
                    "hidden_chain_of_thought": "not exposed by the Codex protocol",
                },
            }
            write_json(run_dir / "codex_run_manifest.json", manifest)

            codex = CodexAppServerClient(
                app_server_command,
                cwd=isolated.workspace,
                env=isolated.child_environment(),
                recorder=recorder,
                stderr_path=run_dir / "codex_app_server_stderr.log",
                timeout_seconds=args.timeout_seconds,
            )
            codex.initialize()
            codex.start_thread(
                gateway=gateway,
                model=args.model,
                base_instructions=BASE_INSTRUCTIONS,
                developer_instructions=(
                    "This is a single-agent benchmark. Never delegate. Treat the dynamic "
                    "embodied tools as the complete capability boundary."
                ),
            )
            turn_result = codex.run_turn(
                gateway=gateway,
                prompt=prompt,
                model=args.model,
                effort=args.effort,
                reasoning_summary="detailed",
            )
            result.update(
                {
                    "status": "codex_turn_completed",
                    "thread_id": turn_result["thread_id"],
                    "turn_id": turn_result["turn_id"],
                    "turn": turn_result["turn"],
                    "episode_terminal": gateway.terminal,
                    "perception_profile": perception_profile.name,
                    "capability_violation": codex.capability_violation,
                }
            )
            if gateway.terminal:
                host_close_response = simulator.request(
                    {
                        "command": "close",
                        "observation_id": gateway.latest_observation_id,
                        "rationale": "Host-owned lifecycle cleanup after terminal evaluation.",
                    }
                )
                simulator.wait(timeout_seconds=120.0)
            event_audit = audit_codex_events(recorder.raw_path)
            result.update(
                {
                    "event_audit": event_audit,
                    "recording_artifacts": recorder.summary(),
                    "host_close_response": host_close_response,
                    "evaluator_outcome_present": (
                        run_dir / "evaluator_outcome.json"
                    ).is_file(),
                }
            )
            result["valid_for_scoring"] = bool(
                gateway.terminal
                and not codex.capability_violation
                and event_audit["passed"]
                and (run_dir / "evaluator_outcome.json").is_file()
            )
            result["status"] = (
                "completed_valid"
                if result["valid_for_scoring"]
                else "completed_not_scorable"
            )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, CapabilityViolation):
            status = "capability_violation"
        else:
            status = "failed"
        result.update(
            {
                "status": status,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "valid_for_scoring": False,
            }
        )
        if recorder is not None:
            result["recording_artifacts"] = recorder.summary()
    finally:
        if codex is not None:
            codex.close()
        if simulator is not None:
            simulator.close()
        if run_dir.exists():
            destination = run_dir / "simulator_stdout.log"
            if temporary_simulator_log.exists():
                temporary_simulator_log.replace(destination)
            if recorder is not None:
                final_event_audit = audit_codex_events(recorder.raw_path)
                runtime_summary = summarize_codex_runtime(recorder.raw_path)
                result["event_audit"] = final_event_audit
                result["codex_runtime"] = runtime_summary
                result["tool_call_summary"] = summarize_tool_calls(
                    recorder.tool_path
                )
                result["recording_artifacts"] = recorder.summary()
                result["capability_violation"] = bool(
                    result.get("capability_violation")
                    or (codex is not None and codex.capability_violation)
                )
                if result.get("status") in {
                    "completed_valid",
                    "completed_not_scorable",
                    "codex_turn_completed",
                }:
                    result["evaluator_outcome_present"] = (
                        run_dir / "evaluator_outcome.json"
                    ).is_file()
                    result["valid_for_scoring"] = bool(
                        result.get("episode_terminal")
                        and not result["capability_violation"]
                        and final_event_audit["passed"]
                        and result["evaluator_outcome_present"]
                    )
                    result["status"] = (
                        "completed_valid"
                        if result["valid_for_scoring"]
                        else "completed_not_scorable"
                    )
                try:
                    timeline_video = build_agent_timeline_video(
                        run_dir,
                        recorder.tool_path,
                    )
                except Exception as exc:
                    timeline_video = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                result["timeline_video"] = timeline_video
                if "error_type" not in timeline_video:
                    recording_artifacts = result.setdefault("recording_artifacts", {})
                    recording_artifacts["timeline_video"] = {
                        key: timeline_video[key]
                        for key in (
                            "path",
                            "sha256",
                            "codec_name",
                            "profile",
                            "pix_fmt",
                            "width",
                            "height",
                            "nb_frames",
                            "frames_per_second",
                            "layout",
                            "observation_frame_count",
                            "mapped_tool_call_count",
                        )
                        if key in timeline_video
                    }
            else:
                runtime_summary = None
            trace_path = build_human_trace(
                run_dir,
                runtime_summary=runtime_summary,
            )
            result["human_trace"] = trace_path.name
            result["finished_utc"] = utc_now()
            result["total_wall_seconds"] = round(
                time.monotonic() - run_started_monotonic, 6
            )
            write_json(run_dir / "codex_run_outcome.json", result)
        elif temporary_simulator_log.exists():
            failed_log = run_dir.parent / f"{run_dir.name}.startup_failed.log"
            temporary_simulator_log.replace(failed_log)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("valid_for_scoring") else 1


if __name__ == "__main__":
    raise SystemExit(main())

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


def operator_prompt(level: int) -> str:
    profile = get_profile(level)
    tactile = (
        "Head RGB, wrist RGB, tactile marker RGB, and basic robot state are available."
        if profile.expose_tactile
        else "Only head RGB, wrist RGB, and basic robot state are available."
    )
    guidance = (
        "After the irreversible classification commitment, each control result also "
        "contains task_success. It may refine motion but cannot change the target."
        if profile.expose_task_success_after_prediction
        else "Task-success feedback is never available during control at this Level."
    )
    return f"""\
Independently execute exactly one grasp_classify rollout at Level {level}.

Task semantics: determine whether the already grasped prism is rough or plain,
then place rough on orange or plain on green. {tactile} {guidance}

Begin with start_episode. You may use at most two pre-classification gripper
probes. Commit exactly once before any Cartesian motion. Then operate only
through bounded delta end-effector or wait tools. Continue until a tool returns
a terminal rollout result; if it has not auto-terminated, call finish_episode
when you judge the best attainable terminal state has been reached. Do not end
the turn while the episode is active. Never use knowledge from prior runs.

For every decision_record, explicitly state what you observed, at least one
alternative, uncertainty, expected effect, and why the exact numeric magnitude
was chosen instead of nearby smaller or larger values. Numeric translation and
gripper units are metres: a probe is at most 0.002 m (2 mm), a post-commit
gripper delta is at most 0.005 m (5 mm), and a Cartesian component is at most
0.04 m (4 cm). Values such as 0.01 or 0.02 are invalid gripper deltas.
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
    capabilities = capability_manifest(level)
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
        gateway = CapabilityGateway(
            level=level,
            simulator_request=simulator.request,
            simulator_run_dir=run_dir,
        )
        prompt = operator_prompt(level)
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
                    "codex_read_only_sandbox": True,
                    "codex_network_access": False,
                    "forbidden_item_fail_closed": True,
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
                    "video": "agent_observations_h264.mp4 after terminal evaluation",
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

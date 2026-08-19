#!/usr/bin/env python3
"""Run one isolated Codex rollout against generic embodied AgentEnv v1."""

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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.artifacts import encode_observation_toolcall_video  # noqa: E402
from agent_env.benchmark_capabilities import (  # noqa: E402
    BenchmarkCapabilityGateway,
    benchmark_capability_manifest,
)
from agent_env.benchmark_profiles import (  # noqa: E402
    AnnotationCapabilities,
    get_observation_profile,
)
from agent_env.benchmark_tasks import (  # noqa: E402
    get_benchmark_task,
    list_benchmark_tasks,
)
from agent_env.capabilities import CapabilityViolation  # noqa: E402
from agent_env.codex_app_server import CodexAppServerClient  # noqa: E402
from agent_env.codex_events import (  # noqa: E402
    EventRecorder,
    audit_codex_events,
    build_benchmark_score,
    build_human_trace,
    check_token_budget,
    summarize_codex_runtime,
    summarize_tool_calls,
    token_budget_manifest,
    utc_now,
)
from agent_env.codex_isolation import (  # noqa: E402
    IsolatedCodexEnvironment,
    embodied_codex_command,
)
from agent_env.contract import EVALUATOR_SEED_ENV  # noqa: E402
from agent_env.fixed_demo_bundle import (  # noqa: E402
    get_fixed_demo_asset_spec,
    project_fixed_demo_bundle,
    resolve_fixed_demo_root,
)
from agent_env.icl import (  # noqa: E402
    ICLCondition,
    get_icl_condition,
    list_icl_conditions,
)
from agent_env.stdio_bridge import SimulatorProcessClient  # noqa: E402


BASE_INSTRUCTIONS = """\
You are the control policy in an agentic embodied benchmark. The host owns the
simulator and exposes three typed embodied tools. Robot interaction must use
only start_episode, step_eef, and finish_episode. General runtime capabilities
may be used to inspect or transform the public benchmark inputs and to support
reasoning, but they do not provide a second simulator-control path.

All externally meaningful reasoning must be placed in each tool's structured
decision_record. Cite only public sources accepted by that tool's schema. Task
success is available only from finish_episode. The host records published
reasoning summaries and agent messages, but hidden model reasoning is not an
observable benchmark artifact.
"""

DEVELOPER_INSTRUCTIONS = """\
Treat the dynamic embodied tools as the complete robot-control boundary.
"""


def operator_prompt(
    task_name: str,
    *,
    pre_move: bool,
    icl_condition: ICLCondition | str = "none",
    max_output_tokens: int | None = None,
    interaction_mode: str = "single_turn",
) -> str:
    if isinstance(icl_condition, str):
        icl_condition = get_icl_condition(icl_condition)
    task = get_benchmark_task(task_name)
    if interaction_mode not in {"single_turn", "action_per_turn"}:
        raise ValueError(f"Unsupported interaction mode: {interaction_mode!r}")
    prompt = (
        f"Task: {task.name}\n\n"
        f"{task.instruction_for(pre_move=pre_move)}\n\n"
        "Execute exactly one episode. Begin with start_episode, operate only "
        "through step_eef, and call finish_episode when you judge the best "
        "attainable terminal state has been reached. Use only information made "
        "available in this run.\n"
    )
    if interaction_mode == "single_turn":
        prompt += "Do not end the turn while the episode is active.\n"
    if icl_condition.fixed_demo_available:
        prompt += (
            "\nA verified successful demonstration is available at "
            "benchmark_inputs/expert_demo/.\n"
        )
    if max_output_tokens is not None:
        prompt += (
            "\nYour episode-wide budget is "
            f"{max_output_tokens} cumulative output tokens. This count already includes "
            "reasoning output. Exceeding it makes the benchmark result a failure, so "
            "reason and communicate concisely while still completing the task.\n"
        )
    return prompt


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one generic UniVTAC Codex rollout")
    parser.add_argument(
        "--task",
        required=True,
        choices=tuple(task.name for task in list_benchmark_tasks()),
    )
    parser.add_argument("--profile", required=True, choices=tuple(str(i) for i in range(1, 7)))
    parser.add_argument("--provide-bbox", action="store_true")
    parser.add_argument("--provide-mask", action="store_true")
    parser.add_argument(
        "--icl",
        default="none",
        choices=tuple(condition.name for condition in list_icl_conditions()),
    )
    parser.add_argument(
        "--fixed-demo-root",
        type=Path,
        help=(
            "Host-side P6 master root for --icl fixed_demo; defaults to "
            "UNIVTAC_FIXED_EXPERT_MASTER_ROOT"
        ),
    )
    parser.add_argument(
        "--pre-move",
        action="store_true",
        help="Use the legacy pregrasped initial state; default is ungrasped.",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="high")
    parser.add_argument(
        "--interaction-mode",
        choices=("single_turn", "action_per_turn"),
        default="single_turn",
        help=(
            "single_turn returns observations inside dynamic-tool results; "
            "action_per_turn keeps one Codex thread but delivers each resulting "
            "observation as the next top-level multimodal turn"
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_integer,
        default=None,
        help=(
            "Optional episode-wide cumulative Codex output-token budget. "
            "The initial implementation checks it after the turn completes."
        ),
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--auth-home", type=Path, default=None)
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
        help="Evaluator-controlled Codex sandbox; benchmark workspace stays temporary.",
    )
    parser.add_argument(
        "--codex-network-access",
        action="store_true",
        help="Allow network access for general Codex tools and record that condition.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_run_dir(
    value: Path | None,
    task: str,
    profile: int,
    *,
    pre_move: bool,
    icl_condition: str = "none",
) -> Path:
    if value is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        condition = "pregrasped" if pre_move else "ungrasped"
        return (
            REPO_ROOT
            / "agent_runs"
            / f"codex_{task}_p{profile}_{condition}_{icl_condition}_{stamp}"
        )
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def codex_sandbox_policy(
    mode: str,
    *,
    workspace: Path,
    network_access: bool,
) -> dict[str, Any]:
    if mode == "read-only":
        return {"type": "readOnly", "networkAccess": network_access}
    if mode == "workspace-write":
        return {
            "type": "workspaceWrite",
            "writableRoots": [str(workspace.resolve())],
            "networkAccess": network_access,
        }
    if mode == "danger-full-access":
        return {"type": "dangerFullAccess"}
    raise ValueError(f"Unsupported Codex sandbox mode: {mode!r}")


def effective_codex_network_access(mode: str, requested: bool) -> bool:
    """Return what the selected Codex sandbox policy itself permits."""

    if mode == "danger-full-access":
        return True
    if mode in {"read-only", "workspace-write"}:
        return bool(requested)
    raise ValueError(f"Unsupported Codex sandbox mode: {mode!r}")


def validate_fixed_demo_evaluation_seed(
    *,
    task_name: str,
    icl_condition: ICLCondition,
    environ: Mapping[str, str],
) -> None:
    """Reject an explicitly selected evaluation seed reused by the demo."""

    if not icl_condition.fixed_demo_available:
        return
    raw_seed = environ.get(EVALUATOR_SEED_ENV)
    if raw_seed is None or not raw_seed.isdecimal():
        return
    if int(raw_seed) == get_fixed_demo_asset_spec(task_name).seed:
        raise ValueError(
            "fixed_demo evaluation seed must differ from the registered "
            "demonstration seed"
        )


def main() -> int:
    run_started = time.monotonic()
    args = parse_args()
    task = get_benchmark_task(args.task)
    profile = get_observation_profile(args.profile)
    annotations = AnnotationCapabilities(args.provide_bbox, args.provide_mask)
    icl_condition = get_icl_condition(args.icl)
    if icl_condition.fixed_demo_available and args.pre_move:
        raise ValueError("fixed_demo is an ungrasped demonstration and cannot use --pre-move")
    if not icl_condition.fixed_demo_available and args.fixed_demo_root is not None:
        raise ValueError("--fixed-demo-root is valid only with --icl fixed_demo")
    network_access = effective_codex_network_access(
        args.codex_sandbox,
        args.codex_network_access,
    )
    budget_contract = token_budget_manifest(args.max_output_tokens)
    capabilities = benchmark_capability_manifest(
        task,
        profile,
        annotations,
        pre_move=args.pre_move,
        icl_condition=icl_condition,
    )
    if args.dry_run:
        print(json.dumps(capabilities, ensure_ascii=False, indent=2))
        return 0
    validate_fixed_demo_evaluation_seed(
        task_name=task.name,
        icl_condition=icl_condition,
        environ=os.environ,
    )
    fixed_demo_root = (
        resolve_fixed_demo_root(args.fixed_demo_root)
        if icl_condition.fixed_demo_available
        else None
    )

    run_dir = resolve_run_dir(
        args.run_dir,
        task.name,
        profile.index,
        pre_move=args.pre_move,
        icl_condition=icl_condition.name,
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_fd, log_name = tempfile.mkstemp(
        prefix=f".{run_dir.name}.simulator-", suffix=".log", dir=run_dir.parent
    )
    os.close(log_fd)
    temporary_log = Path(log_name)

    simulator: SimulatorProcessClient | None = None
    codex: CodexAppServerClient | None = None
    recorder: EventRecorder | None = None
    host_close_response: dict[str, Any] | None = None
    result: dict[str, Any] = {
        "schema_version": "univtac.embodied_codex_rollout_outcome.v1",
        "created_utc": utc_now(),
        "task": task.name,
        "start_condition": task.start_condition(args.pre_move),
        "pre_move_enabled": bool(args.pre_move),
        "profile": profile.name,
        "profile_index": profile.index,
        "annotations": annotations.to_manifest(),
        "icl": icl_condition.to_manifest(),
        "interaction_mode": args.interaction_mode,
        "codex_runtime_policy": {
            "general_capabilities_enabled": True,
            "sandbox": args.codex_sandbox,
            "network_access_requested": bool(args.codex_network_access),
            "network_access_permitted_by_codex_sandbox": network_access,
            "configuration_source": "evaluator Codex configuration",
        },
        "inference_budget": budget_contract,
        "run_dir": str(run_dir),
        "valid_for_scoring": False,
        "status": "starting",
    }
    try:
        simulator_command = [
            str(REPO_ROOT / "scripts" / "launch_agent_env.sh"),
            "--task",
            task.name,
            "--profile",
            str(profile.index),
            "--device",
            args.device,
            "--run-dir",
            str(run_dir),
        ]
        if annotations.provide_bbox:
            simulator_command.append("--provide-bbox")
        if annotations.provide_mask:
            simulator_command.append("--provide-mask")
        if args.pre_move:
            simulator_command.append("--pre-move")
        simulator = SimulatorProcessClient(
            simulator_command,
            cwd=REPO_ROOT,
            timeout_seconds=args.timeout_seconds,
            startup_log_path=temporary_log,
        )
        ready = simulator.ready
        if ready.get("task") != task.name:
            raise RuntimeError(f"Simulator task mismatch: {ready}")
        if ready.get("observation_profile") != profile.to_manifest():
            raise RuntimeError("Simulator observation profile disagrees with host registry")
        if ready.get("annotations") != annotations.to_manifest():
            raise RuntimeError("Simulator annotation capabilities disagree with host registry")
        if ready.get("pre_move_enabled") is not bool(args.pre_move):
            raise RuntimeError("Simulator pre-move condition disagrees with the host request")
        if ready.get("start_condition") != task.start_condition(args.pre_move):
            raise RuntimeError("Simulator start condition disagrees with the host registry")

        recorder = EventRecorder(run_dir)
        prompt = operator_prompt(
            task.name,
            pre_move=args.pre_move,
            icl_condition=icl_condition,
            max_output_tokens=args.max_output_tokens,
            interaction_mode=args.interaction_mode,
        )
        (run_dir / "codex_operator_prompt.txt").write_text(prompt, encoding="utf-8")
        (run_dir / "codex_base_instructions.txt").write_text(
            BASE_INSTRUCTIONS, encoding="utf-8"
        )
        (run_dir / "codex_developer_instructions.txt").write_text(
            DEVELOPER_INSTRUCTIONS, encoding="utf-8"
        )
        write_json(run_dir / "codex_capabilities.json", capabilities)

        with IsolatedCodexEnvironment(
            args.auth_home,
            inherit_agent_configuration=True,
        ) as isolated:
            icl_projection_receipt: dict[str, Any] | None = None
            if icl_condition.fixed_demo_available:
                if fixed_demo_root is None:
                    raise RuntimeError("fixed_demo master root was not resolved")
                icl_projection_receipt = project_fixed_demo_bundle(
                    asset_root=fixed_demo_root,
                    destination=(
                        isolated.workspace / "benchmark_inputs" / "expert_demo"
                    ),
                    task=task.name,
                    profile=profile,
                    annotations=annotations,
                )
                write_json(
                    run_dir / "icl_projection_receipt.json",
                    icl_projection_receipt,
                )
                result["icl_bundle"] = {
                    "manifest_sha256": icl_projection_receipt["agent_bundle"][
                        "manifest_sha256"
                    ],
                    "content_integrity": icl_projection_receipt["agent_bundle"][
                        "content_integrity"
                    ],
                }
            gateway = BenchmarkCapabilityGateway(
                task=task,
                profile=profile,
                annotations=annotations,
                simulator_request=simulator.request,
                simulator_run_dir=run_dir,
                icl_condition=icl_condition,
                agent_workspace=isolated.workspace,
            )
            app_server_command = embodied_codex_command(
                args.codex_bin,
                enable_general_capabilities=True,
            )
            sandbox_policy = codex_sandbox_policy(
                args.codex_sandbox,
                workspace=isolated.workspace,
                network_access=network_access,
            )
            manifest = {
                "schema_version": "univtac.embodied_codex_rollout_manifest.v1",
                "created_utc": utc_now(),
                "task": task.to_manifest(pre_move=args.pre_move),
                "start_condition": task.start_condition(args.pre_move),
                "pre_move_enabled": bool(args.pre_move),
                "observation_profile": profile.to_manifest(),
                "annotations": annotations.to_manifest(),
                "icl": icl_condition.to_manifest(),
                "icl_projection_receipt_file": (
                    "icl_projection_receipt.json"
                    if icl_projection_receipt is not None
                    else None
                ),
                "icl_bundle_manifest_sha256": (
                    icl_projection_receipt["agent_bundle"]["manifest_sha256"]
                    if icl_projection_receipt is not None
                    else None
                ),
                "requested_model": args.model,
                "model": args.model or "Codex configured default",
                "effort": args.effort,
                "reasoning_summary": "detailed",
                "interaction": {
                    "mode": args.interaction_mode,
                    "codex_thread_count": 1,
                    "observation_delivery": (
                        "next_turn_top_level_multimodal"
                        if args.interaction_mode == "action_per_turn"
                        else "same_turn_dynamic_tool_result"
                    ),
                    "embodied_actions_per_turn": (
                        1 if args.interaction_mode == "action_per_turn" else None
                    ),
                },
                "inference_budget": budget_contract,
                "codex_version": command_version(args.codex_bin),
                "app_server_command": app_server_command,
                "operator_prompt_file": "codex_operator_prompt.txt",
                "operator_prompt_sha256": sha256_text(prompt),
                "base_instructions_file": "codex_base_instructions.txt",
                "base_instructions_sha256": sha256_text(BASE_INSTRUCTIONS),
                "developer_instructions_file": "codex_developer_instructions.txt",
                "developer_instructions_sha256": sha256_text(DEVELOPER_INSTRUCTIONS),
                "capability_manifest_file": "codex_capabilities.json",
                "capability_manifest_sha256": capabilities["sha256"],
                "simulator_ready_commitment": ready.get("seed_commitment_sha256"),
                "source": git_revision(),
                "isolation": isolated.manifest(),
                "runtime_capabilities": {
                    "general_codex_capabilities_enabled": True,
                    "configuration_inherited_from_evaluator": True,
                    "workspace": "fresh temporary workspace",
                    "sandbox": args.codex_sandbox,
                    "network_access_requested": bool(args.codex_network_access),
                    "network_access_permitted_by_codex_sandbox": network_access,
                    "policy_owner": "evaluator",
                },
                "enforcement": {
                    "dynamic_tool_allowlist": True,
                    "exact_robot_control_tools": [
                        "start_episode",
                        "step_eef",
                        "finish_episode",
                    ],
                    "host_side_tool_validation": True,
                    "simulator_side_command_validation": True,
                    "task_success_terminal_only": True,
                    "raw_instance_metadata_host_only": True,
                    "general_codex_items_are_capability_violations": False,
                    "max_consecutive_rejected_tool_calls": (
                        gateway.MAX_CONSECUTIVE_REJECTED_CALLS
                    ),
                    "model_visible_close_command": False,
                },
                "recording": {
                    "published_codex_events": "codex_app_server_events.jsonl",
                    "tool_calls": "codex_tool_calls.jsonl",
                    "explicit_decisions": "codex_decisions.jsonl",
                    "published_messages_and_reasoning_summaries": "codex_messages.jsonl",
                    "simulator_transcript": "agent_transcript.jsonl",
                    "sensor_video": "agent_observations_h264.mp4",
                    "agent_timeline_video": "agent_timeline_h264.mp4",
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
                enforce_embodied_only=False,
            )
            codex.initialize()
            codex.start_thread(
                gateway=gateway,
                model=args.model,
                base_instructions=BASE_INSTRUCTIONS,
                developer_instructions=DEVELOPER_INSTRUCTIONS,
                sandbox=args.codex_sandbox,
            )
            turn_summaries: list[dict[str, Any]] = []
            next_prompt: str | None = prompt
            next_input_items: list[dict[str, Any]] | None = None
            while True:
                turn = codex.run_turn(
                    gateway=gateway,
                    prompt=next_prompt,
                    input_items=next_input_items,
                    model=args.model,
                    effort=args.effort,
                    reasoning_summary="detailed",
                    task_name=task.name,
                    profile_name=profile.name,
                    sandbox_policy=sandbox_policy,
                    interrupt_after_dynamic_tool=(
                        args.interaction_mode == "action_per_turn"
                    ),
                    defer_dynamic_tool_content=(
                        args.interaction_mode == "action_per_turn"
                    ),
                )
                turn_summaries.append(
                    {
                        "index": len(turn_summaries),
                        "turn_id": turn["turn_id"],
                        "status": (
                            turn["turn"].get("status")
                            if isinstance(turn.get("turn"), dict)
                            else None
                        ),
                        "dynamic_tool_call_count": turn[
                            "dynamic_tool_call_count"
                        ],
                        "last_dynamic_tool": turn["last_dynamic_tool"],
                        "interrupted_after_dynamic_tool": turn[
                            "interrupted_after_dynamic_tool"
                        ],
                    }
                )
                if args.interaction_mode == "single_turn" or gateway.terminal:
                    break
                if turn["dynamic_tool_call_count"] != 1:
                    raise RuntimeError(
                        "action_per_turn requires exactly one embodied tool call "
                        "before each nonterminal turn boundary"
                    )
                deferred = turn.get("deferred_input_items")
                if not isinstance(deferred, list) or not deferred:
                    raise RuntimeError(
                        "action_per_turn did not produce a public follow-up observation"
                    )
                next_prompt = None
                next_input_items = deferred
            result.update(
                {
                    "status": "codex_interaction_completed",
                    "thread_id": turn["thread_id"],
                    "turn_id": turn["turn_id"],
                    "turn": turn["turn"],
                    "turn_count": len(turn_summaries),
                    "turns": turn_summaries,
                    "episode_terminal": gateway.terminal,
                    "capability_violation": codex.capability_violation,
                }
            )
            if gateway.terminal:
                host_close_response = simulator.request(
                    {
                        "command": "close",
                        "observation_id": gateway.latest_observation_id,
                        "rationale": "Host-owned cleanup after terminal evaluation.",
                    }
                )
                simulator.wait(timeout_seconds=120.0)
            event_audit = audit_codex_events(
                recorder.raw_path,
                enforce_embodied_only=False,
            )
            result.update(
                {
                    "event_audit": event_audit,
                    "recording_artifacts": recorder.summary(),
                    "host_close_response": host_close_response,
                    "evaluator_outcome_present": (run_dir / "evaluator_outcome.json").is_file(),
                }
            )
            result["valid_for_scoring"] = bool(
                gateway.terminal
                and not codex.capability_violation
                and event_audit["passed"]
                and result["evaluator_outcome_present"]
            )
            result["status"] = (
                "completed_valid" if result["valid_for_scoring"] else "completed_not_scorable"
            )
    except BaseException as exc:
        result.update(
            {
                "status": (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "capability_violation"
                    if isinstance(exc, CapabilityViolation)
                    else "failed"
                ),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "valid_for_scoring": False,
            }
        )
    finally:
        if codex is not None:
            codex.close()
        if simulator is not None:
            simulator.close()
        if run_dir.exists():
            if temporary_log.exists():
                temporary_log.replace(run_dir / "simulator_stdout.log")
            runtime_summary = None
            if recorder is not None:
                event_audit = audit_codex_events(
                    recorder.raw_path,
                    enforce_embodied_only=False,
                )
                runtime_summary = summarize_codex_runtime(recorder.raw_path)
                budget_check = check_token_budget(
                    runtime_summary,
                    args.max_output_tokens,
                )
                result["event_audit"] = event_audit
                result["codex_runtime"] = runtime_summary
                result["inference_budget"] = budget_check
                result["tool_call_summary"] = summarize_tool_calls(recorder.tool_path)
                result["recording_artifacts"] = recorder.summary()
                result["capability_violation"] = bool(
                    result.get("capability_violation")
                    or (codex is not None and codex.capability_violation)
                )
                result["evaluator_outcome_present"] = (
                    run_dir / "evaluator_outcome.json"
                ).is_file()
                result["valid_for_scoring"] = bool(
                    result.get("episode_terminal")
                    and not result["capability_violation"]
                    and event_audit["passed"]
                    and result["evaluator_outcome_present"]
                    and (
                        not budget_check["configured"]
                        or budget_check["measurement_available"]
                    )
                )
                evaluator_outcome = read_json(run_dir / "evaluator_outcome.json")
                score = build_benchmark_score(
                    valid_for_scoring=result["valid_for_scoring"],
                    official_task_success=evaluator_outcome.get("official_task_success"),
                    token_budget_check=budget_check,
                )
                result["score"] = score
                result["benchmark_success"] = score["benchmark_success"]
                if result.get("status") not in {"failed", "interrupted", "capability_violation"}:
                    result["status"] = (
                        "completed_valid"
                        if result["valid_for_scoring"]
                        else "completed_not_scorable"
                    )
                try:
                    timeline = encode_observation_toolcall_video(
                        run_dir / "observations",
                        recorder.tool_path,
                        run_dir / "agent_timeline_h264.mp4",
                    )
                except Exception as exc:
                    timeline = {"error_type": type(exc).__name__, "message": str(exc)}
                result["timeline_video"] = timeline
            trace_path = build_human_trace(run_dir, runtime_summary=runtime_summary)
            result["human_trace"] = trace_path.name
            result["finished_utc"] = utc_now()
            result["total_wall_seconds"] = round(time.monotonic() - run_started, 6)
            write_json(run_dir / "codex_run_outcome.json", result)
        elif temporary_log.exists():
            temporary_log.replace(run_dir.parent / f"{run_dir.name}.startup_failed.log")

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("valid_for_scoring") else 1


if __name__ == "__main__":
    raise SystemExit(main())

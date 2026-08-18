#!/usr/bin/env python3
"""Real Isaac -> capability gateway -> SAM3/UniDepth acceptance smoke.

This is deliberately not named ``test_*.py``: it launches Isaac Lab and two
GPU model services, so it is an explicit deployment check rather than a unit
test.  The rollout makes no Cartesian movement and is not a score sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.capabilities import CapabilityGateway, GatewayExecution  # noqa: E402
from agent_env.perception_runtime import PerceptionRuntime  # noqa: E402
from agent_env.stdio_bridge import SimulatorProcessClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8783")
    parser.add_argument("--unidepth-v2-url", default="http://127.0.0.1:8784")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def decision(*sources: str, rationale: str) -> dict[str, Any]:
    return {
        "evidence": [
            {
                "source": source,
                "finding": f"Acceptance smoke has a current {source} observation.",
                "implication": "Use only to exercise the declared data path.",
            }
            for source in (sources or ("head_rgb",))
        ],
        "alternatives_considered": [
            "Skip this operation; rejected because the deployment path would remain untested."
        ],
        "uncertainty": 1.0,
        "expected_effect": "Exercise one bounded read-only or terminal protocol operation.",
        "parameter_rationale": "Use the lowest-cost settings sufficient for an acceptance check.",
        "rationale": rationale,
    }


def require_success(execution: GatewayExecution, name: str) -> GatewayExecution:
    if not execution.success:
        raise RuntimeError(f"{name} failed: {execution.public_response}")
    return execution


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else REPO_ROOT / "agent_runs" / f"semantic_gateway_smoke_{stamp}"
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    simulator: SimulatorProcessClient | None = None
    summary: dict[str, Any] = {
        "schema_version": "univtac.semantic_gateway_smoke.v1",
        "run_dir": str(run_dir),
        "success": False,
    }
    try:
        runtime = PerceptionRuntime(
            profile="sam3_unidepth_v2",
            run_dir=run_dir,
            sam3_url=args.sam3_url,
            unidepth_v2_url=args.unidepth_v2_url,
            timeout_seconds=args.timeout,
        )
        summary["service_health_before"] = runtime.health_manifest()
        simulator = SimulatorProcessClient(
            [
                str(REPO_ROOT / "scripts" / "launch_agent_env.sh"),
                "--level",
                "1",
                "--device",
                args.device,
                "--run-dir",
                str(run_dir),
            ],
            cwd=REPO_ROOT,
            timeout_seconds=args.timeout,
            startup_log_path=run_dir.parent / f".{run_dir.name}.simulator.log",
        )
        gateway = CapabilityGateway(
            level=1,
            simulator_request=simulator.request,
            simulator_run_dir=run_dir,
            perception_profile="sam3_unidepth_v2",
            perception_runtime=runtime,
        )
        started = require_success(
            gateway.execute("start_episode", {"agent_note": "semantic deployment smoke"}),
            "start_episode",
        )
        observation_id = gateway.latest_observation_id
        if observation_id != "obs_000":
            raise RuntimeError(f"Unexpected initial observation: {observation_id!r}")
        sam = require_success(
            gateway.execute(
                "sam3_segment",
                {
                    "observation_id": observation_id,
                    "camera": "head_rgb",
                    "mode": "text",
                    "prompt": "dark grasped prism",
                    "confidence_threshold": 0.25,
                    "decision_record": decision(
                        "head_rgb",
                        rationale="Run one text segmentation on the current public head RGB.",
                    ),
                },
            ),
            "sam3_segment",
        )
        depth = require_success(
            gateway.execute(
                "estimate_metric_depth",
                {
                    "observation_id": observation_id,
                    "camera": "wrist_rgb",
                    "resolution_level": 2,
                    "sample_points": [{"x": 240, "y": 135}],
                    "decision_record": decision(
                        "wrist_rgb",
                        "sam3_result",
                        rationale="Run calibrated predicted depth on the current public wrist RGB.",
                    ),
                },
            ),
            "estimate_metric_depth",
        )
        committed = require_success(
            gateway.execute(
                "commit_classification",
                {
                    "observation_id": observation_id,
                    "predicted_class": "rough",
                    "target_pad": "orange",
                    "decision_record": decision(
                        "head_rgb",
                        "sam3_result",
                        "unidepth_v2_result",
                        rationale="Commit a fixed dummy class solely to cross the protocol boundary.",
                    ),
                },
            ),
            "commit_classification",
        )
        terminal = require_success(
            gateway.execute(
                "finish_episode",
                {
                    "observation_id": observation_id,
                    "final_note": "No-motion semantic deployment acceptance completed.",
                    "decision_record": decision(
                        "head_rgb",
                        "sam3_result",
                        "unidepth_v2_result",
                        rationale="Finish without moving; this rollout validates plumbing, not policy score.",
                    ),
                },
            ),
            "finish_episode",
        )
        if not gateway.terminal:
            raise RuntimeError("Gateway did not enter the terminal stage")
        close = simulator.request(
            {
                "command": "close",
                "observation_id": observation_id,
                "rationale": "Host cleanup after semantic deployment acceptance.",
            }
        )
        return_code = simulator.wait(timeout_seconds=120.0)
        if close.get("status") != "closing" or return_code != 0:
            raise RuntimeError(f"Simulator close failed: {close}, exit={return_code}")
        semantic_root = run_dir / "semantic_perception" / observation_id
        required_artifacts = [
            semantic_root,
            run_dir / ".host_sensor_metadata" / f"{observation_id}.json",
            run_dir / "evaluator_outcome.json",
            run_dir / "agent_observations_h264.mp4",
        ]
        missing = [str(path) for path in required_artifacts if not path.exists()]
        if missing:
            raise RuntimeError(f"Acceptance artifacts are missing: {missing}")
        summary.update(
            {
                "success": True,
                "observation_id": observation_id,
                "start_status": started.public_response.get("status"),
                "sam3": sam.public_response,
                "unidepth_v2": depth.public_response,
                "commit_status": committed.public_response.get("status"),
                "terminal_status": terminal.public_response.get("status"),
                "simulator_exit_code": return_code,
            }
        )
        write_json(run_dir / "semantic_smoke_summary.json", summary)
        print(f"REAL_SEMANTIC_GATEWAY_SMOKE_OK run_dir={run_dir}")
        return 0
    except BaseException as exc:
        summary.update({"error_type": type(exc).__name__, "error": str(exc)})
        if run_dir.exists():
            write_json(run_dir / "semantic_smoke_summary.json", summary)
        raise
    finally:
        if simulator is not None:
            simulator.close()
        temporary_log = run_dir.parent / f".{run_dir.name}.simulator.log"
        if temporary_log.exists() and run_dir.exists():
            temporary_log.replace(run_dir / "simulator_stdout.log")


if __name__ == "__main__":
    raise SystemExit(main())
